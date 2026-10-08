"""Data-free real-backbone smoke: optimizer step, stream inference and export.

Synthetic scenes and untrained weights validate execution and gradients only.
This is not a perception-accuracy or driving-performance benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from model_components.auto_e2e import AutoE2E
from inference.scene_prediction import OccupancyFlowStream, save_scene_outputs
from training.scene_multitask import (
    SceneMultitaskObjective,
    run_scene_epoch,
    evaluate_scene,
    scene_batch_inputs,
    save_scene_checkpoint,
)


def synthetic_batch():
    batch = {
        "visual_tiles": torch.randn(1, 2, 3, 128, 128),
        "map_context": torch.zeros(1, 14, 16, 12),
        "route_mask": torch.zeros(1, 2, 16, 12),
        "map_valid": torch.ones(1, dtype=torch.bool),
        "route_valid": torch.ones(1, dtype=torch.bool),
        "route_channel_valid": torch.ones(1, 2, dtype=torch.bool),
        "visual_history": torch.zeros(1, 896),
        "egomotion_history": torch.zeros(1, 256),
        "initial_speed_mps": torch.ones(1),
        "trajectory_xy_m": torch.zeros(1, 64, 2),
        "trajectory_valid": torch.ones(1, 64, dtype=torch.bool),
        "bev_segmentation_available": torch.ones(1, dtype=torch.bool),
        "bev_segmentation_target": torch.zeros(1, 8, 16, 12),
        "bev_segmentation_valid": torch.ones(1, 8, 16, 12, dtype=torch.bool),
        "occupancy_target": torch.full((1, 4, 16, 12), 16, dtype=torch.long),
        "occupancy_valid": torch.ones(1, 4, 16, 12, dtype=torch.bool),
        "flow_target_mps": torch.zeros(1, 2, 4, 16, 12),
        "flow_valid": torch.zeros(1, 4, 16, 12, dtype=torch.bool),
        "history_to_current": torch.eye(4).reshape(1, 1, 4, 4),
        "history_age_s": torch.tensor([[0.5]]),
    }
    batch["map_context"][:, 0] = 1
    batch["bev_segmentation_target"][:, 0] = 1
    batch["bev_segmentation_target"][:, 5, 4:8, 4:6] = 1
    batch["occupancy_target"][:, 0] = 10
    batch["occupancy_target"][:, 1:3, 4:8, 4:6] = 0
    batch["flow_target_mps"][:, 0, 1:3, 4:8, 4:6] = 2
    batch["flow_valid"][:, 1:3, 4:8, 4:6] = True
    batch["history_camera_tiles"] = batch["visual_tiles"][:, None].clone()
    batch["history_to_current"][:, :, 0, 3] = -0.5
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/autoe2e-occ-flow-smoke")
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.manual_seed(7)
    device = torch.device(args.device)
    kwargs: dict[str, Any] = {
        "backbone": "res_net_50",
        "is_pretrained": False,
        "num_views": 2,
        "view_fusion_kwargs": {
            "bev_h": 16,
            "bev_w": 12,
            "pc_range": (-8.0, -6.0, -2.0, 8.0, 6.0, 2.0),
            "image_size": 128,
        },
        "map_type": "semantic_raster",
        "map_context_channels": 14,
        "planner_mode": "gru",
        "enable_bev_segmentation": True,
        "enable_route_reconstruction": True,
        "enable_occupancy": True,
        "occupancy_kwargs": {"grid_shape": (4, 16, 12), "hidden_channels": 16},
    }
    model = AutoE2E(**kwargs).to(device)
    objective = SceneMultitaskObjective().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = synthetic_batch()
    train = run_scene_epoch(
        model, [batch] * args.steps, objective, optimizer, device=device
    )
    evaluation = evaluate_scene(model, [batch], objective, device=device)
    model.eval()
    current = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if not key.startswith("history_")
    }
    inputs = scene_batch_inputs(model, current, device=device)
    stream = OccupancyFlowStream(model)
    pose = torch.eye(4, device=device)
    stream.predict(
        inputs,
        initial_speed_mps=current["initial_speed_mps"],
        world_from_ego=pose,
        timestamp_s=0.0,
        sequence_id="synthetic",
    )
    pose[0, 3] = 0.5
    result = stream.predict(
        inputs,
        initial_speed_mps=current["initial_speed_mps"],
        world_from_ego=pose,
        timestamp_s=0.5,
        sequence_id="synthetic",
    )
    save_scene_outputs(result, args.output / "scene")
    save_scene_checkpoint(
        args.output / "checkpoint.pt",
        model,
        model_kwargs=kwargs,
        optimizer=optimizer,
        objective=objective,
    )
    report = {
        "synthetic_only": True,
        "training": train,
        "evaluation": evaluation,
        "history_frames_used": result["history_frames_used"],
        "output_shapes": {
            key: list(value.shape)
            for key, value in result.items()
            if torch.is_tensor(value)
        },
    }
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
