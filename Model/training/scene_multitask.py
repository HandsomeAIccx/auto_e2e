"""Joint trajectory, 2D BEV, route, 3D occupancy and motion training.

Data-source independent: loaders supply the documented tensor contract. Unlike
the locked nuPlan/L2D stage runners, this runner keeps the new perception heads
active and reports their losses explicitly.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from model_components.losses import OccupancyFlowLoss
from model_components.view_fusion.projection import PinholeProjection, ImageTransform
from training.occupancy_metrics import OccupancyFlowMetrics
from training.reactive_multitask import (
    ReactiveMultitaskObjective,
    ReactiveTrainingStage,
    reactive_model_kwargs,
)
from training.reactive_stage_runner import (
    _batch_to_device,
    _loader_item,
    resolve_reactive_batch_projection,
)


def scene_model_kwargs(*, num_views, occupancy_kwargs=None):
    """Training preset for the fix branch's GRU planner and semantic map."""
    config = reactive_model_kwargs(
        ReactiveTrainingStage.NUPLAN_FULL, num_views=num_views
    )
    config.update(enable_occupancy=True, occupancy_kwargs=dict(occupancy_kwargs or {}))
    return config


class SceneMultitaskObjective(nn.Module):
    def __init__(
        self,
        *,
        num_classes=17,
        dynamic_classes=tuple(range(10)),
        bev_pos_weight=(1.0,) * 8,
        trajectory_weight=1.0,
        bev_weight=1.0,
        route_weight=1.0,
        occupancy_weight=1.0,
        flow_weight=1.0,
        corridor_pos_weight=1.0,
        occupancy_loss_kwargs=None,
    ):
        super().__init__()
        for name, value in (
            ("occupancy_weight", occupancy_weight),
            ("flow_weight", flow_weight),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1000:
                raise ValueError(f"{name} must be finite and between zero and 1000")
        if not any(
            v > 0
            for v in (
                trajectory_weight,
                bev_weight,
                route_weight,
                occupancy_weight,
                flow_weight,
            )
        ):
            raise ValueError("at least one scene task must be enabled")
        self.reactive = ReactiveMultitaskObjective(
            ReactiveTrainingStage.NUPLAN_FULL,
            bev_pos_weight=bev_pos_weight,
            trajectory_weight=trajectory_weight,
            bev_weight=bev_weight,
            route_weight=route_weight,
            corridor_pos_weight=corridor_pos_weight,
        )
        self.occupancy_loss = OccupancyFlowLoss(
            num_classes,
            dynamic_classes=dynamic_classes,
            **dict(occupancy_loss_kwargs or {}),
        )
        self.occupancy_weight = occupancy_weight
        self.flow_weight = flow_weight
        self.configuration = {
            "num_classes": num_classes,
            "dynamic_classes": tuple(dynamic_classes),
            "bev_pos_weight": torch.as_tensor(bev_pos_weight).float().tolist(),
            "trajectory_weight": trajectory_weight,
            "bev_weight": bev_weight,
            "route_weight": route_weight,
            "occupancy_weight": occupancy_weight,
            "flow_weight": flow_weight,
            "corridor_pos_weight": corridor_pos_weight,
            "occupancy_loss_kwargs": dict(occupancy_loss_kwargs or {}),
        }

    @property
    def compute_occupancy(self):
        return self.occupancy_weight > 0 or self.flow_weight > 0

    def forward(self, controls, auxiliary, batch):
        # Inactive control losses must not create zero gradients on the planner:
        # AdamW would otherwise decay its weights during perception-only training.
        reactive_controls = (
            controls if self.reactive.trajectory_weight > 0 else controls.detach()
        )
        terms = self.reactive(reactive_controls, auxiliary, batch)
        zero = controls.detach().reshape(-1)[:0].sum()
        occ = {
            "occupancy": zero,
            "occupancy_focal": zero,
            "occupancy_dice": zero,
            "flow": zero,
        }
        if self.compute_occupancy:
            if self.flow_weight > 0 and (
                "flow_target_mps" not in batch or "flow_valid" not in batch
            ):
                raise ValueError(
                    "enabled flow training requires targets and valid masks"
                )
            logits = auxiliary.get("occupancy_logits")
            flow = auxiliary.get("occupancy_flow_mps")
            if not torch.is_tensor(logits) or not torch.is_tensor(flow):
                raise ValueError("enabled OCC/flow training requires model outputs")
            occ = self.occupancy_loss(
                logits,
                flow,
                batch["occupancy_target"],
                batch["occupancy_valid"],
                batch.get("flow_target_mps") if self.flow_weight > 0 else None,
                batch.get("flow_valid") if self.flow_weight > 0 else None,
            )
        total = terms["total"]
        if self.occupancy_weight > 0:
            total = total + self.occupancy_weight * occ["occupancy"]
        if self.flow_weight > 0:
            total = total + self.flow_weight * occ["flow"]
        return {
            **terms,
            **occ,
            "total": total,
        }


def scene_batch_inputs(
    model, batch, *, device, projection=None, geometry_type="pseudo"
):
    """Prepare current inputs and either raw-frame or precomputed BEV history."""
    projection, geometry_type = resolve_reactive_batch_projection(
        batch, projection, geometry_type, device=device
    )
    values = {
        "camera_tiles": batch["visual_tiles"],
        "map_context": batch["map_context"],
        "visual_history": batch["visual_history"],
        "egomotion_history": batch["egomotion_history"],
        "projection": projection,
        "geometry_type": geometry_type,
        "image_transform": batch.get("image_transform")
        or ImageTransform(
            model_input_size=(
                batch["visual_tiles"].shape[-1],
                batch["visual_tiles"].shape[-2],
            )
        ),
    }
    for key in (
        "route_mask",
        "map_valid",
        "route_valid",
        "history_bev_features",
        "history_to_current",
        "history_age_s",
        "history_valid",
    ):
        if key in batch:
            values[key] = batch[key]
    if "history_camera_tiles" in batch:
        if "history_bev_features" in batch:
            raise ValueError("supply raw history images or BEV features, not both")
        matrices = batch.get("history_camera_projection_matrix")
        frames = batch["history_camera_tiles"]
        if matrices is None and geometry_type != "pseudo":
            raise ValueError(
                "calibrated history requires per-frame projection matrices"
            )
        projections = None
        if matrices is not None:
            if geometry_type not in ("pinhole", "rectified_pinhole"):
                raise ValueError("matrix history requires a pinhole current geometry")
            if matrices.shape != (*frames.shape[:3], 3, 4):
                raise ValueError("history projection matrix must be [B,T,V,3,4]")
            projections = [
                PinholeProjection(matrices[:, index], geometry_type=geometry_type)
                for index in range(frames.shape[1])
            ]
        values["history_bev_features"] = model.encode_bev_history(
            frames, projections=projections, geometry_type=geometry_type
        )
    return values


def _check_objective(model, objective):
    head = model.Reactive_E2E.OccupancyFlowHead
    if objective.compute_occupancy and (
        head is None
        or objective.occupancy_loss.num_classes != len(head.class_names)
        or objective.occupancy_loss.dynamic_classes != head.dynamic_classes
    ):
        raise ValueError("scene objective and model occupancy configuration differ")


def run_scene_epoch(model, loader, objective, optimizer, *, device, grad_clip=1.0):
    """Run real optimizer steps, including new heads, with finite gradient checks."""
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be positive and finite")
    _check_objective(model, objective)
    objective.to(device)
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for item in loader:
        raw, projection, geometry_type = _loader_item(item)
        batch = _batch_to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        inputs = scene_batch_inputs(
            model,
            batch,
            device=device,
            projection=projection,
            geometry_type=geometry_type,
        )
        output = model(
            **inputs,
            mode="train",
            compute_bev_segmentation=objective.reactive.compute_bev_segmentation,
            compute_route_reconstruction=objective.reactive.compute_route_reconstruction,
            compute_occupancy=objective.compute_occupancy,
        )
        controls, auxiliary = output if isinstance(output, tuple) else (output, {})
        terms = objective(controls, auxiliary, batch)
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError("scene objective became non-finite")
        terms["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("scene gradients became non-finite")
        optimizer.step()
        size = int(controls.shape[0])
        count += size
        for name, term in terms.items():
            totals[name] = totals.get(name, 0.0) + float(term.detach()) * size
    if count == 0:
        raise ValueError("scene training loader yielded no samples")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def evaluate_scene(model, loader, objective, *, device):
    """Joint losses plus global voxel IoU and physical-velocity endpoint error."""
    _check_objective(model, objective)
    head = model.Reactive_E2E.OccupancyFlowHead
    if head is None:
        raise ValueError("scene evaluation requires an occupancy head")
    objective.to(device)
    metrics = OccupancyFlowMetrics(
        len(head.class_names), dynamic_classes=head.dynamic_classes
    )
    totals: dict[str, float] = {}
    count = 0
    was_training = model.training
    model.eval()
    try:
        for item in loader:
            raw, projection, geometry_type = _loader_item(item)
            batch = _batch_to_device(raw, device)
            inputs = scene_batch_inputs(
                model,
                batch,
                device=device,
                projection=projection,
                geometry_type=geometry_type,
            )
            controls, aux = model(
                **inputs,
                mode="infer",
                return_auxiliary=True,
                compute_bev_segmentation=objective.reactive.compute_bev_segmentation,
                compute_route_reconstruction=objective.reactive.compute_route_reconstruction,
            )
            terms = objective(controls, aux, batch)
            size = int(controls.shape[0])
            count += size
            for name, term in terms.items():
                if not torch.isfinite(term):
                    raise FloatingPointError("scene evaluation loss became non-finite")
                totals[name] = totals.get(name, 0.0) + float(term) * size
            metrics.update(
                aux["occupancy_logits"],
                aux["occupancy_flow_mps"],
                batch["occupancy_target"],
                batch["occupancy_valid"],
                batch.get("flow_target_mps"),
                batch.get("flow_valid"),
            )
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("scene evaluation loader yielded no samples")
    return {
        "losses": {name: value / count for name, value in totals.items()},
        "metrics": metrics.compute(),
        "samples": count,
    }


def save_scene_checkpoint(path, model, *, model_kwargs, optimizer=None, objective=None):
    """Persist architecture and grid semantics alongside strict model weights."""
    head = model.Reactive_E2E.OccupancyFlowHead
    if head is None:
        raise ValueError("scene checkpoint requires occupancy enabled")
    payload = {
        "version": "autoe2e_scene_checkpoint_v1",
        "model_kwargs": model_kwargs,
        "occupancy_metadata": head.metadata(),
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if objective is not None:
        payload["objective_state_dict"] = objective.state_dict()
        payload["objective_configuration"] = objective.configuration
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_scene_checkpoint(path, model, *, optimizer=None, objective=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = model.Reactive_E2E.OccupancyFlowHead
    if (
        payload.get("version") != "autoe2e_scene_checkpoint_v1"
        or head is None
        or payload.get("occupancy_metadata") != head.metadata()
    ):
        raise ValueError("checkpoint schema or occupancy geometry/classes differ")
    if (
        objective is not None
        and payload.get("objective_configuration") != objective.configuration
    ):
        raise ValueError("checkpoint objective configuration differs")
    model.load_state_dict(payload["state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if objective is not None:
        objective.load_state_dict(payload["objective_state_dict"], strict=True)
    return payload["model_kwargs"]
