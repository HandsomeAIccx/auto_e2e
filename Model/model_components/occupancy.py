"""Shared-BEV 3D semantic occupancy and planar velocity decoding.

Inspired by Apollo-Vision-Net's BEV-to-height lifting and parallel OCC/flow
heads. This implementation uses native PyTorch and AutoE2E's ego-FLU raster
convention; it does not load Apollo checkpoints or depend on OpenMMLab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_PC_RANGE = (-60.0, -60.0, -5.0, 120.0, 60.0, 3.0)
OCCUPANCY_CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "empty",
)


@dataclass(frozen=True)
class OccupancyGrid:
    """Tensor axes Z,X,Y: Z increases, raster X/Y decrease from max to min."""

    shape: tuple[int, int, int] = (16, 180, 120)
    pc_range: tuple[float, float, float, float, float, float] = DEFAULT_PC_RANGE

    def __post_init__(self):
        if len(self.shape) != 3 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.shape
        ):
            raise ValueError("occupancy shape must contain positive Z,X,Y integers")
        if len(self.pc_range) != 6 or not all(map(math.isfinite, self.pc_range)):
            raise ValueError("pc_range must contain six finite values")
        if any(self.pc_range[i + 3] <= self.pc_range[i] for i in range(3)):
            raise ValueError("pc_range maxima must exceed minima")

    def metadata(self) -> dict:
        z, x, y = self.shape
        r = self.pc_range
        return {
            "version": "autoe2e_occ_flow_v1",
            "frame": "ego_flu",
            "axes": "Z,X,Y",
            "shape": self.shape,
            "pc_range": self.pc_range,
            "voxel_size_xyz_m": (
                (r[3] - r[0]) / x,
                (r[4] - r[1]) / y,
                (r[5] - r[2]) / z,
            ),
            "raster_order": "x_descending,y_descending,z_ascending",
            "flow_components": ("vx", "vy"),
            "flow_unit": "m/s",
            "flow_reference": "physical_velocity_in_current_ego_frame",
        }


def _xy_points(shape, pc_range, *, device):
    """Homogeneous metric centers, matching BEVViewFusion's reversed XY axes."""
    h, w = shape
    x = pc_range[3] - (torch.arange(h, device=device).float() + 0.5) * (
        (pc_range[3] - pc_range[0]) / h
    )
    y = pc_range[4] - (torch.arange(w, device=device).float() + 0.5) * (
        (pc_range[4] - pc_range[1]) / w
    )
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy, torch.zeros_like(xx), torch.ones_like(xx)), -1)


def _sampling_grid(points, pc_range):
    # grid_sample's first coordinate addresses columns (Y), second rows (X).
    col = 1 - 2 * (points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    row = 1 - 2 * (points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    return torch.stack((col, row), -1)


def validate_planar_transform(transform: torch.Tensor) -> None:
    """Require rigid yaw/translation transforms; a 2D BEV cannot warp pitch/roll."""
    if transform.shape[-2:] != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError("BEV transforms must be finite [...,4,4] tensors")
    value = transform.detach().float()
    rotation = value[..., :3, :3]
    eye = torch.eye(3, device=value.device).expand_as(rotation)
    bottom = value.new_tensor([0.0, 0.0, 0.0, 1.0]).expand_as(value[..., 3, :])
    if (
        not torch.allclose(rotation.transpose(-1, -2) @ rotation, eye, atol=1e-4)
        or not torch.allclose(
            torch.linalg.det(rotation), torch.ones_like(value[..., 0, 0]), atol=1e-4
        )
        or not torch.allclose(value[..., 3, :], bottom, atol=1e-4)
        or bool((value[..., :2, 2].abs() > 1e-4).any())
        or bool((value[..., 2, :2].abs() > 1e-4).any())
        or not torch.allclose(
            value[..., 2, 2], torch.ones_like(value[..., 2, 2]), atol=1e-4
        )
    ):
        raise ValueError("BEV transforms must be rigid planar yaw/translation")


def warp_history_bev(history, history_to_current, pc_range):
    """Resample [B,T,C,H,W] into current ego; return warped features and support."""
    if history.ndim != 5:
        raise ValueError("history BEV must have shape [B,T,C,H,W]")
    b, t, c, h, w = history.shape
    if history_to_current.shape != (b, t, 4, 4):
        raise ValueError("history_to_current must have shape [B,T,4,4]")
    validate_planar_transform(history_to_current)
    if t == 0:
        return history, history.new_zeros(b, 0, 1, h, w)
    # Inversion and geometry stay FP32 even under mixed precision.
    with torch.autocast(device_type=history.device.type, enabled=False):
        transforms = history_to_current.to(device=history.device, dtype=torch.float32)
        points = _xy_points((h, w), pc_range, device=history.device)
        source_points = torch.einsum(
            "btij,hwj->bthwi", torch.linalg.inv(transforms), points
        )
        grid = _sampling_grid(source_points, pc_range).reshape(b * t, h, w, 2)
        warped = F.grid_sample(
            history.float().reshape(b * t, c, h, w),
            grid,
            align_corners=False,
            padding_mode="zeros",
        )
        support = F.grid_sample(
            torch.ones(b * t, 1, h, w, device=history.device),
            grid,
            align_corners=False,
            padding_mode="zeros",
        )
    return warped.reshape(b, t, c, h, w).to(history.dtype), support.reshape(
        b, t, 1, h, w
    )


class TemporalBEVFusion(nn.Module):
    """Stateless causal fusion with pose alignment, age decay and per-cell gates."""

    def __init__(self, channels, pc_range, *, decay_seconds=1.0):
        super().__init__()
        if not math.isfinite(decay_seconds) or decay_seconds <= 0:
            raise ValueError("decay_seconds must be finite and positive")
        self.pc_range = pc_range
        self.decay_seconds = decay_seconds
        self.gate = nn.Conv2d(2 * channels, 1, 1)

    def forward(
        self,
        current,
        history=None,
        history_to_current=None,
        history_age_s=None,
        history_valid=None,
    ):
        if history is None:
            if any(
                v is not None
                for v in (history_to_current, history_age_s, history_valid)
            ):
                raise ValueError("history metadata requires history BEV features")
            return current
        if (
            history.ndim != 5
            or history.shape[0] != current.shape[0]
            or history.shape[2:] != current.shape[1:]
        ):
            raise ValueError("history BEV must match current [B,T,C,H,W]")
        b, t, c, h, w = history.shape
        if history_to_current is None or history_age_s is None:
            raise ValueError("history requires transforms and positive ages in seconds")
        ages = history_age_s.to(device=current.device, dtype=torch.float32)
        if (
            ages.shape != (b, t)
            or not torch.isfinite(ages).all()
            or bool((ages <= 0).any())
        ):
            raise ValueError("history_age_s must be finite positive [B,T]")
        valid = torch.ones(b, t, dtype=torch.bool, device=current.device)
        if history_valid is not None:
            if history_valid.shape != (b, t):
                raise ValueError("history_valid must have shape [B,T]")
            valid = history_valid.to(device=current.device, dtype=torch.bool)
        if t == 0:
            return current
        # Mask before sampling: padded/invalid frames may contain NaNs.
        history = torch.where(valid[..., None, None, None], history.to(current), 0.0)
        if not torch.isfinite(history).all():
            raise ValueError("valid history BEV features must be finite")
        aligned, support = warp_history_bev(history, history_to_current, self.pc_range)
        pairs = torch.cat((current[:, None].expand(-1, t, -1, -1, -1), aligned), 2)
        gates = (
            self.gate(pairs.reshape(b * t, 2 * c, h, w))
            .sigmoid()
            .reshape(b, t, 1, h, w)
        )
        weights = (
            gates
            * support.to(gates)
            * torch.exp(-ages / self.decay_seconds)[..., None, None, None]
        )
        weights = weights * valid[..., None, None, None]
        # support already attenuated the sampled feature at grid edges.
        normalized = aligned / support.to(aligned).clamp_min(1e-6)
        return (current + (weights * normalized).sum(1)) / (1 + weights.sum(1))


class BEVOccupancyFlowHead(nn.Module):
    """Lift 2D BEV into height features, decode categorical OCC and metric flow."""

    def __init__(
        self,
        embed_dim=256,
        *,
        source_pc_range=DEFAULT_PC_RANGE,
        grid_shape=(16, 180, 120),
        pc_range=None,
        hidden_channels=32,
        class_names: Sequence[str] = OCCUPANCY_CLASSES,
        dynamic_classes: Sequence[int] = tuple(range(10)),
        decay_seconds=1.0,
    ):
        super().__init__()
        self.grid = OccupancyGrid(tuple(grid_shape), tuple(pc_range or source_pc_range))
        self.source_grid = OccupancyGrid((1, 1, 1), tuple(source_pc_range))
        self.class_names = tuple(class_names)
        if (
            len(self.class_names) < 2
            or len(set(self.class_names)) != len(self.class_names)
            or self.class_names[-1] != "empty"
        ):
            raise ValueError("class_names must be unique and end with 'empty'")
        self.empty_class = len(self.class_names) - 1
        self.dynamic_classes = tuple(dynamic_classes)
        if len(set(self.dynamic_classes)) != len(self.dynamic_classes) or any(
            not isinstance(i, int) or i < 0 or i >= self.empty_class
            for i in self.dynamic_classes
        ):
            raise ValueError("dynamic_classes must index non-empty classes")
        if embed_dim <= 0 or hidden_channels <= 0 or hidden_channels % 8:
            raise ValueError("hidden_channels must be positive and divisible by eight")
        source = self.source_grid.pc_range
        target = self.grid.pc_range
        if any(target[i] < source[i] or target[i + 3] > source[i + 3] for i in (0, 1)):
            raise ValueError("occupancy XY range must be contained in source BEV")
        self.temporal = TemporalBEVFusion(
            embed_dim, source, decay_seconds=decay_seconds
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_channels, 1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.lift = nn.Conv2d(hidden_channels, hidden_channels * self.grid.shape[0], 1)
        self.refine = nn.Sequential(
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.occupancy = nn.Conv3d(hidden_channels, len(self.class_names), 1)
        self.flow = nn.Conv3d(hidden_channels, 2, 1)
        self.hidden_channels = hidden_channels

    def forward(
        self,
        image_bev,
        *,
        history_bev_features=None,
        history_to_current=None,
        history_age_s=None,
        history_valid=None,
    ):
        if image_bev.ndim != 4:
            raise ValueError("image_bev must have shape [B,C,H,W]")
        bev = self.temporal(
            image_bev,
            history_bev_features,
            history_to_current,
            history_age_s,
            history_valid,
        )
        reduced = self.reduce(bev)
        points = _xy_points(self.grid.shape[1:], self.grid.pc_range, device=bev.device)
        sample_grid = _sampling_grid(points, self.source_grid.pc_range)
        with torch.autocast(device_type=bev.device.type, enabled=False):
            reduced = F.grid_sample(
                reduced.float(),
                sample_grid[None].expand(bev.shape[0], -1, -1, -1),
                align_corners=False,
                padding_mode="border",
            ).to(reduced.dtype)
        z, h, w = self.grid.shape
        volume = self.lift(reduced).reshape(bev.shape[0], self.hidden_channels, z, h, w)
        volume = self.refine(volume)
        return {
            "occupancy_logits": self.occupancy(volume),
            "occupancy_flow_mps": self.flow(volume),
        }

    def metadata(self):
        return {
            **self.grid.metadata(),
            "class_names": self.class_names,
            "empty_class": self.empty_class,
            "dynamic_classes": self.dynamic_classes,
            "source_bev_pc_range": self.source_grid.pc_range,
        }

    def decode(self, logits, flow, *, include_probabilities=False):
        """Dense semantic voxels, flow, and occupied probabilities projected to BEV."""
        if logits.ndim != 5 or logits.shape[1:] != (
            len(self.class_names),
            *self.grid.shape,
        ):
            raise ValueError("occupancy logits differ from configured grid/classes")
        if flow.shape != (logits.shape[0], 2, *self.grid.shape):
            raise ValueError("flow must have shape [B,2,Z,X,Y]")
        if not torch.isfinite(logits).all() or not torch.isfinite(flow).all():
            raise FloatingPointError("occupancy and flow predictions must be finite")
        probabilities = logits.float().softmax(1)
        labels = probabilities.argmax(1)
        moving_class = torch.zeros_like(labels, dtype=torch.bool)
        for label in self.dynamic_classes:
            moving_class |= labels == label
        result = {
            "occupancy_labels": labels,
            "occupancy_probability": 1 - probabilities[:, self.empty_class],
            "flow_mps": torch.where(moving_class[:, None], flow.float(), 0.0),
            "flow_valid": moving_class,
            "bev_occupancy_probability": (1 - probabilities[:, self.empty_class]).amax(
                1
            ),
            "grid": self.metadata(),
        }
        bev_classes = probabilities[:, :-1].amax(2).argmax(1)
        result["bev_occupancy_labels"] = torch.where(
            (labels != self.empty_class).any(1), bev_classes, self.empty_class
        )
        valid = moving_class.sum(1).clamp_min(1)
        result["bev_flow_mps"] = result["flow_mps"].sum(2) / valid[:, None]
        result["bev_flow_valid"] = moving_class.any(1)
        if include_probabilities:
            result["occupancy_class_probabilities"] = probabilities
        return result
