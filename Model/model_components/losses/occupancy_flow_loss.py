"""Masked categorical occupancy and metric physical-velocity supervision."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def occupancy_validity(logits, target, valid):
    if logits.ndim != 5 or target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("occupancy target must match [B,Z,X,Y] logits grid")
    if (
        valid.shape != target.shape
        or target.dtype.is_floating_point
        or target.dtype == torch.bool
    ):
        raise ValueError("occupancy requires integer classes and a matching valid mask")
    labels = target.to(device=logits.device, dtype=torch.long)
    mask = valid.to(device=logits.device, dtype=torch.bool) & (labels != 255)
    if bool(((labels[mask] < 0) | (labels[mask] >= logits.shape[1])).any()):
        raise ValueError("valid occupancy labels are outside configured classes")
    return labels, mask


def flow_validity(flow, target, valid, labels, occ_valid, dynamic_classes):
    if (
        flow.shape != (labels.shape[0], 2, *labels.shape[1:])
        or target.shape != flow.shape
    ):
        raise ValueError("flow and target must have shape [B,2,Z,X,Y]")
    if valid.shape != labels.shape:
        raise ValueError("flow valid mask must have shape [B,Z,X,Y]")
    mask = valid.to(device=flow.device, dtype=torch.bool) & occ_valid
    foreground = torch.zeros_like(mask)
    for index in dynamic_classes:
        foreground |= labels == index
    mask &= foreground
    values = target.to(device=flow.device, dtype=torch.float32)
    if not torch.isfinite(values.movedim(1, -1)[mask]).all():
        raise ValueError("valid flow targets must be finite physical velocities")
    return values, mask


class OccupancyFlowLoss(nn.Module):
    """FP32 focal CE, non-empty class Dice, and dynamic-voxel L1 velocity.

    Class K-1 means empty; 255 means unknown and is always ignored. Masks are
    applied before computing losses, so unobserved NaN motion cannot poison a
    batch. Every absent task returns a differentiable zero, including batches
    with no moving/foreground objects.
    """

    class_weights: torch.Tensor

    def __init__(
        self,
        num_classes=17,
        *,
        dynamic_classes=tuple(range(10)),
        class_weights: Sequence[float] | None = None,
        focal_gamma=2.0,
        dice_weight=0.5,
    ):
        super().__init__()
        if num_classes < 2 or any(
            i < 0 or i >= num_classes - 1 for i in dynamic_classes
        ):
            raise ValueError(
                "dynamic classes must be non-empty configured class indices"
            )
        if not math.isfinite(focal_gamma) or focal_gamma < 0:
            raise ValueError("focal_gamma must be finite and nonnegative")
        if not math.isfinite(dice_weight) or dice_weight < 0:
            raise ValueError("dice_weight must be finite and nonnegative")
        weights = (
            torch.ones(num_classes)
            if class_weights is None
            else torch.as_tensor(class_weights).float()
        )
        if (
            weights.shape != (num_classes,)
            or not torch.isfinite(weights).all()
            or bool((weights <= 0).any())
        ):
            raise ValueError(
                "class_weights must contain positive finite values per class"
            )
        self.register_buffer("class_weights", weights)
        self.dynamic_classes = tuple(dynamic_classes)
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight

    def forward(
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
            raise ValueError("loss classes differ from occupancy logits")
        if flow.shape != (logits.shape[0], 2, *logits.shape[2:]):
            raise ValueError("flow must match occupancy grid")
        # Empty slices create finite zeros even if ignored predictions contain NaN.
        zero = logits.float().reshape(-1)[:0].sum()
        flow_zero = flow.float().reshape(-1)[:0].sum()
        focal, dice, motion = zero, zero, flow_zero
        with torch.autocast(device_type=logits.device.type, enabled=False):
            if bool(mask.any()):
                selected = logits.float().movedim(1, -1)[mask]
                truth = labels[mask]
                log_probability = selected.log_softmax(-1)
                correct_log_probability = log_probability.gather(
                    1, truth[:, None]
                ).squeeze(1)
                focal = (
                    -correct_log_probability
                    * (1 - correct_log_probability.exp()).pow(self.focal_gamma)
                    * self.class_weights.to(logits.device)[truth]
                ).mean()
                if self.dice_weight > 0:
                    probability = log_probability.exp()
                    truth_onehot = F.one_hot(truth, self.num_classes).float()
                    intersection = (probability * truth_onehot).sum(0)
                    denominator = probability.sum(0) + truth_onehot.sum(0)
                    present = truth_onehot.sum(0)[:-1] > 0
                    if bool(present.any()):
                        dice = (
                            1 - (2 * intersection[:-1] + 1) / (denominator[:-1] + 1)
                        )[present].mean()
            if (flow_target_mps is None) != (flow_valid is None):
                raise ValueError(
                    "flow target and flow validity must be supplied together"
                )
            if flow_target_mps is not None:
                values, motion_mask = flow_validity(
                    flow,
                    flow_target_mps,
                    flow_valid,
                    labels,
                    mask,
                    self.dynamic_classes,
                )
                if bool(motion_mask.any()):
                    motion = F.l1_loss(
                        flow.float().movedim(1, -1)[motion_mask],
                        values.movedim(1, -1)[motion_mask],
                    )
        return {
            "occupancy": focal + self.dice_weight * dice,
            "occupancy_focal": focal,
            "occupancy_dice": dice,
            "flow": motion,
        }
