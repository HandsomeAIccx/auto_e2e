"""Joint trajectory/BEV/OCC/flow inference and explicitly owned stream state."""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import deque
from typing import Any

import numpy as np
import torch

from model_components.occupancy import validate_planar_transform
from model_components.view_fusion.projection import ImageTransform
from data_processing.reactive_training_artifacts import BEV_SEGMENTATION_CLASSES
from training.losses.control_rollout import integrate_controls_torch


@torch.inference_mode()
def predict_scene(
    model,
    inputs,
    *,
    initial_speed_mps,
    include_probabilities=False,
    return_bev_features=False,
    trajectory_dt=0.1,
):
    """Return dense results; caller must set eval() and supply raw model inputs."""
    if model.training:
        raise ValueError("scene inference requires model.eval()")
    head = model.Reactive_E2E.OccupancyFlowHead
    if head is None:
        raise ValueError("scene inference requires enable_occupancy=True")
    reserved = {"mode", "return_auxiliary", "return_bev_features", "compute_occupancy"}
    if reserved.intersection(inputs):
        raise ValueError("scene inputs must not override inference execution flags")
    inputs = dict(inputs)
    inputs.setdefault(
        "image_transform",
        ImageTransform(
            model_input_size=(
                inputs["camera_tiles"].shape[-1],
                inputs["camera_tiles"].shape[-2],
            )
        ),
    )
    controls, aux = model(
        **inputs,
        mode="infer",
        return_auxiliary=True,
        compute_occupancy=True,
        return_bev_features=return_bev_features,
    )
    positions, _, _ = integrate_controls_torch(
        controls, initial_speed_mps, dt=trajectory_dt
    )
    result = {
        "trajectory_controls": controls.reshape(controls.shape[0], -1, 2),
        "trajectory_xy_m": positions,
        "trajectory_dt_s": trajectory_dt,
        **head.decode(
            aux["occupancy_logits"],
            aux["occupancy_flow_mps"],
            include_probabilities=include_probabilities,
        ),
    }
    if "bev_segmentation_logits" in aux:
        result["bev_semantic_probabilities"] = (
            aux["bev_segmentation_logits"].float().sigmoid()
        )
        result["bev_semantic_classes"] = BEV_SEGMENTATION_CLASSES
    if return_bev_features:
        result["camera_bev_features"] = aux["camera_bev_features"]
    return result


class OccupancyFlowStream:
    """One explicit B=1 stream; detached BEV history, planar poses, real ages.

    Poses map ego coordinates into a common world frame. Changing sequence_id
    resets history. One stream object per vehicle/camera stream prevents batch
    cross-contamination; model.forward itself is always stateless.
    """

    def __init__(self, model, *, history_length=3, max_history_age_s=2.0):
        if not isinstance(history_length, int) or history_length <= 0:
            raise ValueError("history_length must be a positive integer")
        if not math.isfinite(max_history_age_s) or max_history_age_s <= 0:
            raise ValueError("max_history_age_s must be positive and finite")
        self.model = model
        self.max_history_age_s = max_history_age_s
        self.history: deque[tuple[float, torch.Tensor, torch.Tensor]] = deque(
            maxlen=history_length
        )
        self.sequence_id = None

    def reset(self):
        self.history.clear()
        self.sequence_id = None

    @torch.inference_mode()
    def predict(
        self,
        inputs,
        *,
        initial_speed_mps,
        world_from_ego,
        timestamp_s,
        sequence_id,
        include_probabilities=False,
    ):
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if inputs["camera_tiles"].shape[0] != 1:
            raise ValueError("stream inference supports batch size one")
        if any(
            name in inputs
            for name in (
                "history_bev_features",
                "history_to_current",
                "history_age_s",
                "history_valid",
            )
        ):
            raise ValueError("stream owns its history inputs")
        if world_from_ego.shape != (4, 4):
            raise ValueError("world_from_ego must have shape [4,4]")
        validate_planar_transform(world_from_ego)
        if sequence_id != self.sequence_id:
            self.reset()
        if self.history and timestamp_s <= self.history[-1][0]:
            raise ValueError("timestamps must strictly increase within a stream")
        pose = world_from_ego.to(
            device=inputs["camera_tiles"].device, dtype=torch.float32
        )
        active = [
            entry
            for entry in self.history
            if timestamp_s - entry[0] <= self.max_history_age_s
        ]
        values = dict(inputs)
        if active:
            values["history_bev_features"] = torch.stack(
                [entry[2] for entry in active], 1
            )
            values["history_to_current"] = torch.stack(
                [torch.linalg.inv(pose) @ entry[1] for entry in active]
            )[None]
            values["history_age_s"] = pose.new_tensor(
                [[timestamp_s - entry[0] for entry in active]]
            )
        scene = predict_scene(
            self.model,
            values,
            initial_speed_mps=initial_speed_mps,
            include_probabilities=include_probabilities,
            return_bev_features=True,
        )
        bev = scene.pop("camera_bev_features")
        self.history.append((timestamp_s, pose.detach().clone(), bev.detach().clone()))
        self.sequence_id = sequence_id
        scene["timestamp_s"] = timestamp_s
        scene["sequence_id"] = sequence_id
        scene["history_frames_used"] = len(active)
        return scene


def save_scene_outputs(scene, output_prefix):
    """Export batch tensors without converting flow into arbitrary pixel units."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        name: value.detach()
        .to(
            device="cpu",
            dtype=torch.float32 if value.dtype == torch.bfloat16 else value.dtype,
        )
        .numpy()
        for name, value in scene.items()
        if torch.is_tensor(value) and name != "camera_bev_features"
    }
    metadata = {
        name: value for name, value in scene.items() if not torch.is_tensor(value)
    }
    np.savez_compressed(str(prefix) + ".npz", **arrays)
    Path(str(prefix) + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    Path(str(prefix) + ".svg").write_text(render_bev_svg(scene), encoding="utf-8")


def render_bev_svg(scene, *, batch_index=0, stride=4):
    """Dependency-free top view with OCC classes, metric flow arrows and XY path."""
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    labels = scene["bev_occupancy_labels"][batch_index].detach().cpu().numpy()
    velocity = scene["bev_flow_mps"][batch_index].detach().cpu().numpy()
    valid = scene["bev_flow_valid"][batch_index].detach().cpu().numpy()
    trajectory = scene["trajectory_xy_m"][batch_index].detach().cpu().numpy()
    grid = scene["grid"]
    r = grid["pc_range"]
    h, w = labels.shape
    scale = 4
    colors = (
        "#ed6a5a",
        "#f4a261",
        "#e76f51",
        "#ea526f",
        "#ffb703",
        "#be95c4",
        "#9d4edd",
        "#ffd166",
        "#fb8500",
        "#bc6c25",
        "#3e6c55",
        "#4a7c59",
        "#627c85",
        "#5b7541",
        "#7b8c9b",
        "#52796f",
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * scale}" height="{h * scale + 32}" viewBox="0 0 {w} {h + 8}">',
        '<rect width="100%" height="100%" fill="#17212b"/>',
        '<g stroke="#98dfef" stroke-width="0.25">',
    ]
    for row in range(0, h, stride):
        for col in range(0, w, stride):
            label = int(labels[row, col])
            if label != grid["empty_class"]:
                parts.append(
                    f'<rect x="{col}" y="{row}" width="{min(stride, w - col)}" height="{min(stride, h - row)}" fill="{colors[label % len(colors)]}" stroke="none"/>'
                )
            if valid[row, col] and np.linalg.norm(velocity[:, row, col]) > 0.1:
                # Arrows represent one second of metric velocity, not future OCC.
                dx = -float(velocity[1, row, col]) * w / (r[4] - r[1])
                dy = -float(velocity[0, row, col]) * h / (r[3] - r[0])
                x2, y2 = col + 0.5 + dx, row + 0.5 + dy
                parts.append(
                    f'<path d="M{col + 0.5},{row + 0.5} L{x2:.3f},{y2:.3f}" fill="none"/>'
                )
                parts.append(
                    f'<circle cx="{x2:.3f}" cy="{y2:.3f}" r="0.4" fill="#98dfef" stroke="none"/>'
                )
    parts.append("</g>")
    points = [(r[4] - float(y)) * w / (r[4] - r[1]) for _, y in trajectory]
    rows = [(r[3] - float(x)) * h / (r[3] - r[0]) for x, _ in trajectory]
    path = " ".join(f"{x:.3f},{y:.3f}" for x, y in zip(points, rows))
    parts.append(
        f'<polyline points="{path}" fill="none" stroke="#ffffff" stroke-width="0.6"/>'
    )
    parts.append(
        f'<text x="1" y="{h + 5}" font-size="3" fill="white">OCC / velocity (1 s arrows) / ego trajectory</text></svg>'
    )
    return "".join(parts)
