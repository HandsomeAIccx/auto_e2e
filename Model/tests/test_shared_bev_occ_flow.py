"""Metric geometry, supervision, joint gradients and inference contracts."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from model_components.occupancy import (
    BEVOccupancyFlowHead,
    OccupancyGrid,
    TemporalBEVFusion,
    warp_history_bev,
)
from model_components.losses import OccupancyFlowLoss
from inference.scene_prediction import (
    OccupancyFlowStream,
    predict_scene,
    save_scene_outputs,
)
from training.occupancy_metrics import OccupancyFlowMetrics
from training.scene_multitask import (
    SceneMultitaskObjective,
    run_scene_epoch,
    evaluate_scene,
    save_scene_checkpoint,
    load_scene_checkpoint,
    scene_batch_inputs,
)


PC_RANGE = (-2.0, -3.0, -1.0, 2.0, 3.0, 1.0)


@pytest.mark.parametrize(
    "translation,location",
    [
        ((0.0, 0.0), (1, 2)),
        ((1.0, 0.0), (0, 2)),
        ((0.0, 1.0), (1, 1)),
        ((-1.0, 0.0), (2, 2)),
        ((0.0, -1.0), (1, 3)),
    ],
)
def test_warp_metric_translation_and_reversed_xy(translation, location):
    history = torch.zeros(1, 1, 1, 4, 6, requires_grad=True)
    with torch.no_grad():
        history[0, 0, 0, 1, 2] = 1
    transform = torch.eye(4).reshape(1, 1, 4, 4)
    transform[..., 0, 3], transform[..., 1, 3] = translation
    warped, _ = warp_history_bev(history, transform, PC_RANGE)
    expected = torch.zeros_like(warped)
    expected[0, 0, 0, location[0], location[1]] = 1
    torch.testing.assert_close(warped, expected, atol=1e-6, rtol=0)
    warped.sum().backward()
    assert history.grad is not None and history.grad.sum() > 0


def test_warp_yaw_rotation_and_no_support():
    history = torch.zeros(1, 1, 1, 4, 6)
    history[0, 0, 0, 1, 2] = 1  # ego XY = (+0.5,+0.5)
    transform = torch.eye(4).reshape(1, 1, 4, 4)
    transform[0, 0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    warped, _ = warp_history_bev(history, transform, PC_RANGE)
    assert warped[0, 0, 0, 2, 2] == pytest.approx(1)
    transform[..., 0, 3] = 100
    current = torch.randn(1, 1, 4, 6)
    fusion = TemporalBEVFusion(1, PC_RANGE)
    torch.testing.assert_close(
        fusion(current, history, transform, torch.ones(1, 1)), current
    )


def test_temporal_masks_bad_history_and_rejects_future_or_nonrigid_pose():
    current = torch.randn(2, 8, 4, 6)
    past = torch.full((2, 1, 8, 4, 6), float("nan"))
    identity = torch.eye(4).expand(2, 1, 4, 4).clone()
    fusion = TemporalBEVFusion(8, PC_RANGE)
    out = fusion(
        current, past, identity, torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool)
    )
    torch.testing.assert_close(out, current)
    with pytest.raises(ValueError, match="positive"):
        fusion(current, past.nan_to_num(), identity, torch.zeros(2, 1))
    identity[..., 0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        fusion(current, past.nan_to_num(), identity, torch.ones(2, 1))
    with pytest.raises(ValueError, match="metadata"):
        fusion(current, history_age_s=torch.ones(2, 1))


def test_head_nonsquare_grid_metadata_and_height_gradients():
    head = BEVOccupancyFlowHead(
        16, source_pc_range=PC_RANGE, grid_shape=(3, 8, 5), hidden_channels=16
    )
    bev = torch.randn(2, 16, 4, 6, requires_grad=True)
    past = torch.randn(2, 2, 16, 4, 6, requires_grad=True)
    out = head(
        bev,
        history_bev_features=past,
        history_to_current=torch.eye(4).expand(2, 2, 4, 4),
        history_age_s=torch.tensor([[0.2, 0.8], [0.5, 1.0]]),
    )
    assert out["occupancy_logits"].shape == (2, 17, 3, 8, 5)
    assert out["occupancy_flow_mps"].shape == (2, 2, 3, 8, 5)
    (
        out["occupancy_logits"].square().mean()
        + out["occupancy_flow_mps"].square().mean()
    ).backward()
    assert bev.grad.abs().sum() > 0 and past.grad.abs().sum() > 0
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert head.metadata()["voxel_size_xyz_m"] == (0.5, 1.2, 2 / 3)
    assert head.metadata()["flow_unit"] == "m/s"
    with pytest.raises(ValueError, match="contained"):
        BEVOccupancyFlowHead(
            16, source_pc_range=PC_RANGE, pc_range=(-3.0, -3.0, -1.0, 2.0, 3.0, 1.0)
        )
    with pytest.raises(ValueError):
        OccupancyGrid((0, 2, 3))


def test_occupancy_crop_resamples_metric_centers_instead_of_resizing_extent():
    head = BEVOccupancyFlowHead(
        8,
        source_pc_range=PC_RANGE,
        hidden_channels=8,
        pc_range=(-1.0, -2.0, -1.0, 1.0, 2.0, 1.0),
        grid_shape=(2, 4, 4),
    )
    head.reduce = torch.nn.Identity()
    x = torch.tensor([1.5, 0.5, -0.5, -1.5])
    y = torch.tensor([2.5, 1.5, 0.5, -0.5, -1.5, -2.5])
    features = (2 * x[:, None] + 3 * y[None, :])[None, None].expand(1, 8, -1, -1)
    sampled = []
    handle = head.lift.register_forward_pre_hook(
        lambda _, values: sampled.append(values[0].detach())
    )
    head(features)
    handle.remove()
    target_x = torch.tensor([0.75, 0.25, -0.25, -0.75])
    target_y = torch.tensor([1.5, 0.5, -0.5, -1.5])
    expected = (2 * target_x[:, None] + 3 * target_y[None, :])[None, None].expand(
        1, 8, -1, -1
    )
    torch.testing.assert_close(sampled[0], expected)


def test_loss_mixed_precision_and_nonfinite_inference():
    head = BEVOccupancyFlowHead(8, grid_shape=(2, 3, 4), hidden_channels=8)
    loss = OccupancyFlowLoss()
    features = torch.randn(1, 8, 3, 4, requires_grad=True)
    labels = torch.zeros(1, 2, 3, 4, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = head(features)
        terms = loss(
            outputs["occupancy_logits"],
            outputs["occupancy_flow_mps"],
            labels,
            valid,
            torch.zeros(1, 2, 2, 3, 4),
            valid,
        )
    assert all(
        term.dtype == torch.float32 and torch.isfinite(term) for term in terms.values()
    )
    (terms["occupancy"] + terms["flow"]).backward()
    assert torch.isfinite(features.grad).all()
    with pytest.raises(FloatingPointError, match="finite"):
        head.decode(
            outputs["occupancy_logits"].detach().fill_(float("nan")),
            outputs["occupancy_flow_mps"],
        )


def test_decoder_empty_static_and_dynamic_motion():
    head = BEVOccupancyFlowHead(8, grid_shape=(2, 2, 3), hidden_channels=8)
    logits = torch.full((1, 17, 2, 2, 3), -20.0)
    logits[:, 16] = 20
    logits[0, 16, 1, 0, 1] = -20
    logits[0, 0, 1, 0, 1] = 20
    logits[0, 16, 0, 1, 2] = -20
    logits[0, 10, 0, 1, 2] = 20
    flow = torch.ones(1, 2, 2, 2, 3) * 3
    scene = head.decode(logits, flow, include_probabilities=True)
    assert scene["flow_valid"].sum() == 1
    assert scene["bev_occupancy_labels"][0, 0, 1] == 0
    assert scene["bev_occupancy_labels"][0, 1, 2] == 10
    assert scene["flow_mps"][0, :, 1, 0, 1].tolist() == [3.0, 3.0]
    assert scene["flow_mps"][0, :, 0, 1, 2].tolist() == [0.0, 0.0]
    assert scene["bev_flow_mps"][0, :, 0, 1].tolist() == [3.0, 3.0]


def test_masked_losses_ignore_unknown_and_nan_and_empty_flow():
    loss = OccupancyFlowLoss(3, dynamic_classes=(0,))
    logits = torch.randn(1, 3, 1, 1, 3, requires_grad=True)
    flow = torch.zeros(1, 2, 1, 1, 3, requires_grad=True)
    labels = torch.tensor([[[[0, 255, 2]]]])
    motion = torch.tensor(
        [[[[[3.0, float("nan"), float("nan")]]], [[[4.0, float("nan"), float("nan")]]]]]
    )
    valid = torch.ones_like(labels, dtype=torch.bool)
    terms = loss(logits, flow, labels, valid, motion, valid)
    assert terms["flow"].item() == pytest.approx(3.5)
    sum(terms.values()).backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(flow.grad).all()
    assert logits.grad[..., 1].abs().sum() == 0
    assert flow.grad[..., 1:].abs().sum() == 0
    empty = loss(logits, flow, labels.fill_(2), valid, motion, valid)
    assert empty["flow"].item() == 0
    invalid = loss(logits, flow, labels, torch.zeros_like(valid))
    assert all(value.item() == 0 for value in invalid.values())
    invalid["occupancy"].backward()
    with pytest.raises(ValueError, match="outside"):
        loss(logits, flow, labels.fill_(8), valid)


def test_metrics_known_iou_and_velocity_endpoint_error():
    metrics = OccupancyFlowMetrics(3, dynamic_classes=(0,))
    labels = torch.tensor([[[[0, 0, 1, 2]]]])
    prediction = torch.tensor([[[[0, 2, 1, 2]]]])
    logits = torch.nn.functional.one_hot(prediction, 3).movedim(-1, 1).float() * 10
    flow = torch.zeros(1, 2, 1, 1, 4)
    target = torch.zeros_like(flow)
    target[:, 0, ..., :2] = 3
    target[:, 1, ..., :2] = 4
    valid = torch.ones_like(labels, dtype=torch.bool)
    metrics.update(logits, flow, labels, valid, target, valid)
    result = metrics.compute()
    assert result["occupancy_miou"] == pytest.approx(0.75)
    assert result["occupied_recall"] == pytest.approx(2 / 3)
    assert result["flow_epe_mps"] == pytest.approx(5.0)
    assert result["flow_valid_voxels"] == 2
    metrics.reset()
    assert metrics.compute()["flow_epe_mps"] is None


def _model(build_mock_model, *, planner="gru"):
    return build_mock_model(
        num_views=2,
        map_context_channels=14,
        map_type="semantic_raster",
        planner_mode=planner,
        enable_bev_segmentation=True,
        enable_route_reconstruction=True,
        enable_occupancy=True,
        occupancy_kwargs={"grid_shape": (3, 8, 6), "hidden_channels": 16},
    )


def _batch():
    return {
        "visual_tiles": torch.randn(1, 2, 3, 64, 64),
        "map_context": torch.rand(1, 14, 8, 8),
        "route_mask": torch.rand(1, 2, 8, 8),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
        "visual_history": torch.randn(1, 896),
        "egomotion_history": torch.zeros(1, 256),
        "trajectory_xy_m": torch.zeros(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "initial_speed_mps": torch.ones(1),
        "bev_segmentation_available": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_target": torch.randint(2, (1, 8, 8, 8)).float(),
        "bev_segmentation_valid": torch.ones(1, 8, 8, 8, dtype=torch.bool),
        "occupancy_target": torch.randint(17, (1, 3, 8, 6)),
        "occupancy_valid": torch.ones(1, 3, 8, 6, dtype=torch.bool),
        "flow_target_mps": torch.randn(1, 2, 3, 8, 6),
        "flow_valid": torch.ones(1, 3, 8, 6, dtype=torch.bool),
    }


def test_occ_flow_reaches_shared_camera_but_not_navigation(build_mock_model):
    model = _model(build_mock_model)
    batch = _batch()
    inputs = scene_batch_inputs(model, batch, device=torch.device("cpu"))
    controls, aux = model(**inputs, mode="train")
    objective = SceneMultitaskObjective(
        trajectory_weight=0, bev_weight=0, route_weight=0
    )
    # Calling the joint objective verifies the enabled-task routing as well.
    objective(controls, aux, batch)["total"].backward()
    for module in (
        model.Reactive_E2E.Backbone,
        model.Reactive_E2E.FeatureFusion,
        model.Reactive_E2E.OccupancyFlowHead,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(
        p.grad is None for p in model.Reactive_E2E.NavigationEncoder.parameters()
    )
    assert all(
        p.grad is None for p in model.Reactive_E2E.TrajectoryPlanner.parameters()
    )
    model.eval()
    with torch.no_grad():
        _, first = model(**inputs, mode="infer", return_auxiliary=True)
        inputs["map_context"] = torch.randn_like(inputs["map_context"]) * 10
        _, second = model(**inputs, mode="infer", return_auxiliary=True)
    torch.testing.assert_close(first["occupancy_logits"], second["occupancy_logits"])
    torch.testing.assert_close(
        first["occupancy_flow_mps"], second["occupancy_flow_mps"]
    )


@pytest.mark.parametrize("planner", ["gru", "bezier"])
def test_joint_inference_preserves_trajectory_and_exports(
    build_mock_model, tmp_path, planner
):
    model = _model(build_mock_model, planner=planner).eval()
    batch = _batch()
    inputs = scene_batch_inputs(model, batch, device=torch.device("cpu"))
    with torch.no_grad():
        baseline = model(**inputs, mode="infer")
        controls, aux = model(**inputs, mode="infer", return_auxiliary=True)
    assert baseline.shape == (1, 128)
    torch.testing.assert_close(baseline, controls)
    assert "occupancy_logits" in aux and "bev_segmentation_logits" in aux
    scene = predict_scene(model, inputs, initial_speed_mps=torch.ones(1))
    assert scene["trajectory_xy_m"].shape == (1, 64, 2)
    assert scene["occupancy_labels"].shape == (1, 3, 8, 6)
    save_scene_outputs(scene, tmp_path / "scene")
    with np.load(tmp_path / "scene.npz") as payload:
        np.testing.assert_allclose(payload["flow_mps"], scene["flow_mps"].numpy())
    assert (
        json.loads((tmp_path / "scene.json").read_text())["grid"]["flow_unit"] == "m/s"
    )
    ET.parse(tmp_path / "scene.svg")


def test_bfloat16_scene_inference_exports_numeric_npz(build_mock_model, tmp_path):
    model = _model(build_mock_model).eval()
    values = scene_batch_inputs(model, _batch(), device=torch.device("cpu"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scene = predict_scene(model, values, initial_speed_mps=torch.ones(1))
    save_scene_outputs(scene, tmp_path / "frame.001")
    with np.load(tmp_path / "frame.001.npz", allow_pickle=False) as payload:
        assert payload["trajectory_controls"].dtype == np.float32
        assert payload["flow_mps"].dtype == np.float32
        assert "allow_pickle" not in payload.files


def test_raw_history_encoding_optimizer_epoch_and_checkpoint(
    build_mock_model, tmp_path
):
    model = _model(build_mock_model)
    batch = _batch()
    batch.update(
        history_camera_tiles=batch["visual_tiles"][:, None].clone(),
        history_to_current=torch.eye(4).reshape(1, 1, 4, 4),
        history_age_s=torch.tensor([[0.5]]),
    )
    objective = SceneMultitaskObjective()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.Reactive_E2E.OccupancyFlowHead.flow.weight.detach().clone()
    losses = run_scene_epoch(
        model, [batch], objective, optimizer, device=torch.device("cpu")
    )
    assert math.isfinite(losses["total"]) and losses["flow"] > 0
    assert not torch.equal(before, model.Reactive_E2E.OccupancyFlowHead.flow.weight)
    evaluation = evaluate_scene(model, [batch], objective, device=torch.device("cpu"))
    assert evaluation["metrics"]["flow_epe_mps"] is not None and model.training
    destination = tmp_path / "checkpoint.pt"
    save_scene_checkpoint(
        destination,
        model,
        model_kwargs={"enable_occupancy": True},
        optimizer=optimizer,
        objective=objective,
    )
    loaded = _model(build_mock_model)
    loaded_optimizer = torch.optim.Adam(loaded.parameters())
    load_scene_checkpoint(
        destination, loaded, optimizer=loaded_optimizer, objective=objective
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    with pytest.raises(ValueError, match="objective configuration"):
        load_scene_checkpoint(
            destination, loaded, objective=SceneMultitaskObjective(flow_weight=0.1)
        )
    loaded.Reactive_E2E.OccupancyFlowHead.class_names = tuple(
        reversed(loaded.Reactive_E2E.OccupancyFlowHead.class_names)
    )
    with pytest.raises(ValueError, match="geometry/classes"):
        load_scene_checkpoint(destination, loaded)


def test_stream_resets_sequences_uses_age_and_rejects_future_order(build_mock_model):
    model = _model(build_mock_model).eval()
    inputs = scene_batch_inputs(model, _batch(), device=torch.device("cpu"))
    stream = OccupancyFlowStream(model, history_length=2, max_history_age_s=1.0)
    pose = torch.eye(4)

    def frame(time, scene="a"):
        return stream.predict(
            inputs,
            initial_speed_mps=torch.ones(1),
            world_from_ego=pose,
            timestamp_s=time,
            sequence_id=scene,
        )

    assert frame(0.0)["history_frames_used"] == 0
    pose[0, 3] = 0.5
    assert frame(0.5)["history_frames_used"] == 1
    with pytest.raises(ValueError, match="increase"):
        frame(0.5)
    assert frame(3.0)["history_frames_used"] == 0
    assert frame(0.0, "b")["history_frames_used"] == 0
    stream.reset()
    assert len(stream.history) == 0


def test_no_labels_for_disabled_tasks_and_wrong_source_range(build_mock_model):
    model = _model(build_mock_model)
    batch = _batch()
    inputs = scene_batch_inputs(model, batch, device=torch.device("cpu"))
    controls, aux = model(**inputs, mode="train")
    objective = SceneMultitaskObjective(
        trajectory_weight=0, bev_weight=0, route_weight=0
    )
    minimum = {
        name: batch[name]
        for name in (
            "occupancy_target",
            "occupancy_valid",
            "flow_target_mps",
            "flow_valid",
        )
    }
    assert torch.isfinite(objective(controls, aux, minimum)["total"])
    with pytest.raises(ValueError, match="requires targets"):
        objective(controls, aux, {})
    with pytest.raises(ValueError, match="source range"):
        build_mock_model(
            num_views=2,
            enable_occupancy=True,
            occupancy_kwargs={"source_pc_range": PC_RANGE},
        )


def test_flow_only_optimizer_preserves_inactive_heads_and_navigation(build_mock_model):
    model = _model(build_mock_model)
    protected = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not name.startswith(
            (
                "Reactive_E2E.Backbone.",
                "Reactive_E2E.FeatureFusion.",
                "Reactive_E2E.OccupancyFlowHead.",
            )
        )
        or name.startswith("Reactive_E2E.OccupancyFlowHead.occupancy.")
    }
    objective = SceneMultitaskObjective(
        trajectory_weight=0,
        bev_weight=0,
        route_weight=0,
        occupancy_weight=0,
        flow_weight=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    batch = _batch()
    minimum = {
        name: value
        for name, value in batch.items()
        if not name.startswith(("trajectory_", "bev_segmentation_"))
    }
    result = run_scene_epoch(
        model, [minimum], objective, optimizer, device=torch.device("cpu")
    )
    assert result["total"] == pytest.approx(result["flow"])
    for name, parameter in model.named_parameters():
        if name in protected:
            assert parameter.grad is None, name
            torch.testing.assert_close(parameter, protected[name], rtol=0, atol=0)


def test_history_encoder_explicit_gradient_policy(build_mock_model):
    model = _model(build_mock_model)
    history = torch.randn(1, 1, 2, 3, 64, 64, requires_grad=True)
    stopped = model.encode_bev_history(history, geometry_type="pseudo")
    assert not stopped.requires_grad
    connected = model.encode_bev_history(history, geometry_type="pseudo", detach=False)
    connected.square().mean().backward()
    assert history.grad is not None and history.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="per history"):
        model.encode_bev_history(history, projections=[])


def test_calibrated_history_uses_actual_image_size_and_distinct_projections(
    build_mock_model,
):
    model = _model(build_mock_model)
    batch = _batch()
    matrix = torch.tensor(
        [[32.0, -20.0, 0.0, 160.0], [32.0, 0.0, -20.0, 160.0], [1.0, 0.0, 0.0, 5.0]]
    )
    batch["camera_projection_matrix"] = matrix.expand(1, 2, 3, 4).clone()
    batch["camera_geometry_type"] = "pinhole"
    batch["history_camera_tiles"] = (
        batch["visual_tiles"][:, None].expand(-1, 2, -1, -1, -1, -1).clone()
    )
    batch["history_camera_projection_matrix"] = matrix.expand(1, 2, 2, 3, 4).clone()
    batch["history_camera_projection_matrix"][:, 1, :, 0, 3] += 10
    batch["history_to_current"] = torch.eye(4).expand(1, 2, 4, 4).clone()
    batch["history_age_s"] = torch.tensor([[0.5, 1.0]])
    values = scene_batch_inputs(model, batch, device=torch.device("cpu"))
    assert values["image_transform"].model_input_size == (64, 64)
    assert values["history_bev_features"].shape == (1, 2, 256, 8, 8)
    assert not torch.equal(
        values["history_bev_features"][:, 0], values["history_bev_features"][:, 1]
    )
    _, outputs = model(**values, mode="train")
    assert torch.isfinite(outputs["occupancy_logits"]).all()
    del batch["history_camera_projection_matrix"]
    with pytest.raises(ValueError, match="per-frame"):
        scene_batch_inputs(model, batch, device=torch.device("cpu"))


def test_head_disabled_keeps_original_state_dict_and_interfaces(build_mock_model):
    model = build_mock_model(num_views=2, planner_mode="gru")
    assert model.Reactive_E2E.OccupancyFlowHead is None
    assert not any("OccupancyFlowHead" in name for name in model.state_dict())
    with pytest.raises(ValueError, match="requires enable_occupancy"):
        predict_scene(model.eval(), {}, initial_speed_mps=torch.ones(1))
