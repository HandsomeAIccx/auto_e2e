"""Streaming OCC confusion matrix and dynamic-voxel velocity endpoint error."""

from __future__ import annotations

import torch

from model_components.losses.occupancy_flow_loss import (
    occupancy_validity,
    flow_validity,
)


class OccupancyFlowMetrics:
    def __init__(self, num_classes=17, *, dynamic_classes=tuple(range(10))):
        if num_classes < 2 or any(
            i < 0 or i >= num_classes - 1 for i in dynamic_classes
        ):
            raise ValueError("dynamic class indices must belong to non-empty classes")
        self.num_classes = num_classes
        self.dynamic_classes = tuple(dynamic_classes)
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )
        self.flow_error_sum = 0.0
        self.flow_count = 0

    @torch.no_grad()
    def update(
        self,
        logits,
        flow,
        occupancy_target,
        occupancy_valid,
        flow_target_mps=None,
        flow_valid=None,
    ):
        labels, mask = occupancy_validity(logits, occupancy_target, occupancy_valid)
        if logits.shape[1] != self.num_classes:
            raise ValueError("metric classes differ from logits")
        predicted = logits.argmax(1)
        indices = labels[mask] * self.num_classes + predicted[mask]
        self.confusion += torch.bincount(
            indices.cpu(), minlength=self.num_classes**2
        ).reshape(self.confusion.shape)
        if (flow_target_mps is None) != (flow_valid is None):
            raise ValueError("flow target and mask must be supplied together")
        if flow_target_mps is not None:
            values, motion_mask = flow_validity(
                flow, flow_target_mps, flow_valid, labels, mask, self.dynamic_classes
            )
            errors = torch.linalg.vector_norm(
                (flow.float() - values).movedim(1, -1)[motion_mask], dim=-1
            )
            self.flow_error_sum += float(errors.sum().cpu())
            self.flow_count += errors.numel()

    def compute(self):
        counts = self.confusion.double()
        intersection = counts.diag()
        union = counts.sum(0) + counts.sum(1) - intersection
        iou = intersection / union.clamp_min(1)
        present = union[:-1] > 0
        tp = counts[:-1, :-1].sum()
        return {
            "class_iou": [
                float(iou[i]) if union[i] else None for i in range(self.num_classes)
            ],
            "occupancy_miou": float(iou[:-1][present].mean())
            if bool(present.any())
            else None,
            "occupied_precision": float(tp / (tp + counts[-1, :-1].sum()).clamp_min(1)),
            "occupied_recall": float(tp / (tp + counts[:-1, -1].sum()).clamp_min(1)),
            "flow_epe_mps": self.flow_error_sum / self.flow_count
            if self.flow_count
            else None,
            "flow_valid_voxels": self.flow_count,
            "occupancy_valid_voxels": int(counts.sum()),
        }
