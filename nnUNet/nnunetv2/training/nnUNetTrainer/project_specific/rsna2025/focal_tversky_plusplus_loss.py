"""
Trainer for RSNA2025 using Focal Tversky++ Loss

Reference:
- Base: nnUNet/nnunetv2/training/nnUNetTrainer/project_specific/rsna2025/tversky_loss.py
  - Implementation pattern compliant with MemoryEfficientTverskyLoss (non-linear, batch aggregation, DDP, background control, loss_mask)
"""

from typing import Callable

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv3 import (
    RSNA2025Trainer_moreDAv3,
)
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv5 import (
    RSNA2025Trainer_moreDAv5,
)
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientFocalTverskyPlusPlusLoss(nn.Module):
    """
    Focal Tversky++ Loss (Memory Efficient Version)

    Based on the definition of Tversky++, calculates the following:
        tp = sum(p * g)
        fp = alpha * sum((p * (1 - g)) ** gamma_pp)
        fn = beta  * sum(((1 - p) * g) ** gamma_pp)
        loss = (1 - (tp + smooth_nr) / (tp + fp + fn + smooth_dr)) ** gamma_focal

    Parameters:
        alpha: Weight for false positives
        beta: Weight for false negatives
        gamma_pp: Power exponent for Tversky++
        gamma_focal: Power exponent for Focal
        smooth_nr/dr: Smoothing term to prevent division by zero
        apply_nonlin: Assumes softmax or sigmoid
        batch_dice: Whether to aggregate across the entire batch
        do_bg: Whether to include background channel
        ddp: Whether to perform DDP aggregation
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma_pp: float = 2.0,
        gamma_focal: float = 4.0 / 3.0,
        apply_nonlin: Callable | None = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        ddp: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_pp = gamma_pp
        self.gamma_focal = gamma_focal
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.ddp = ddp

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y.to(dtype=x.dtype)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        p0 = x
        p1 = 1.0 - p0
        g0 = y_onehot
        g1 = 1.0 - g0

        # Define basic terms (targets for exponentiation) first
        base_tp = p0 * g0
        base_fp = p0 * g1
        base_fn = p1 * g0

        if loss_mask is None:
            tp = base_tp.sum(axes)
            fp = self.alpha * (base_fp**self.gamma_pp).sum(axes)
            fn = self.beta * (base_fn**self.gamma_pp).sum(axes)
        else:
            # Apply mask "outside" of exponentiation to keep weighting linear
            tp = (base_tp * loss_mask).sum(axes)
            fp = self.alpha * ((base_fp**self.gamma_pp) * loss_mask).sum(axes)
            fn = self.beta * ((base_fn**self.gamma_pp) * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

            nom = tp + self.smooth_nr
            denom = torch.clip(tp + fp + fn + self.smooth_dr, min=1e-8)
            per_class = (1.0 - nom / denom) ** self.gamma_focal
            loss = per_class.mean()
        else:
            nom = tp + self.smooth_nr
            denom = torch.clip(tp + fp + fn + self.smooth_dr, min=1e-8)
            per_sample_per_class = (1.0 - nom / denom) ** self.gamma_focal
            loss = per_sample_per_class.mean()

        return loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlus(RSNA2025Trainer_moreDAv3):
    """
    RSNA2025 trainer using FocalTverskyPlusPlusLoss.

    - Uses `sigmoid` for regions (multi-label).
    - Uses `softmax` and `to_onehot_y=True` for multi-class, excluding background.
    - `batch_dice` setting is reflected in `batch` of FocalTversky++.
    - Coefficients are similar to Tversky, weighting FN slightly more: alpha=0.3, beta=0.7.
    """

    def _build_loss(self):
        alpha = 0.3
        beta = 0.7
        gamma_pp = 2.0
        gamma_focal = 4.0 / 3.0  # 1.33...

        if self.label_manager.has_regions:
            # Multi-label (regions): sigmoid activation + include background
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=torch.sigmoid,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )
        else:
            # Multi-class: softmax activation + exclude background
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=softmax_helper_dim1,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=False,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )

        # Support Deep Supervision
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # Smaller weights for lower resolutions (exponential decay)
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                # Special case for DDP: tiny weight for final output
                weights[-1] = 1e-6
            else:
                # Usually ignore final output
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusWithBackground(RSNA2025Trainer_moreDAv3):
    """
    RSNA2025 trainer using FocalTverskyPlusPlusLoss.

    - Uses `sigmoid` for regions (multi-label).
    - Includes background even for multi-class.
    - `batch_dice` setting is reflected in `batch` of FocalTversky++.
    - Coefficients are similar to Tversky, weighting FN slightly more: alpha=0.3, beta=0.7.
    """

    def _build_loss(self):
        alpha = 0.3
        beta = 0.7
        gamma_pp = 2.0
        gamma_focal = 4.0 / 3.0  # 1.33...

        if self.label_manager.has_regions:
            # Multi-label (regions): sigmoid activation + include background
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=torch.sigmoid,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )
        else:
            # Multi-class: softmax activation + include background
            loss = MemoryEfficientFocalTverskyPlusPlusLoss(
                alpha=alpha,
                beta=beta,
                gamma_pp=gamma_pp,
                gamma_focal=gamma_focal,
                apply_nonlin=softmax_helper_dim1,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth_nr=1e-5,
                smooth_dr=1e-5,
                ddp=self.is_ddp,
            )

        # Support Deep Supervision
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            # Smaller weights for lower resolutions (exponential decay)
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                # Special case for DDP: tiny weight for final output
                weights[-1] = 1e-6
            else:
                # Usually ignore final output
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class FocalTverskyPlusPlusCELoss(nn.Module):
    """
    Loss combining Focal Tversky++ and Cross Entropy with weights.

    Similar to TverskyPlusCELoss in tversky_loss.py, masks the FocalTversky++ side
    using loss_mask according to ignore_label, while delegating the CE side to RobustCrossEntropyLoss.
    """

    def __init__(
        self,
        tverskypp_kwargs: dict,
        ce_kwargs: dict,
        weight_tverskypp: float = 1.0,
        weight_ce: float = 1.0,
        ignore_label: int | None = None,
    ) -> None:
        super().__init__()
        self.weight_tverskypp = weight_tverskypp
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.tverskypp = MemoryEfficientFocalTverskyPlusPlusLoss(**tverskypp_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            mask_bool = target != self.ignore_label
            mask = mask_bool.float()
            target_tversky = torch.clone(target)
            target_tversky[target == self.ignore_label] = 0
            num_fg = mask_bool.sum()
        else:
            mask = None
            target_tversky = target
            num_fg = None

        tverskypp_loss = self.tverskypp(net_output, target_tversky, loss_mask=mask)

        if self.ignore_label is None or (num_fg is not None and num_fg > 0):
            ce_loss = self.ce(net_output, target)
        else:
            ce_loss = torch.as_tensor(0.0, device=net_output.device, dtype=net_output.dtype)

        return self.weight_tverskypp * tverskypp_loss + self.weight_ce * ce_loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusPlusCE(RSNA2025Trainer_moreDAv3):
    """
    Trainer combining Focal Tversky++ and Cross Entropy (moreDAv3).

    Dedicated to multi-class (use BCE for regions=True).
    """

    def _build_loss(self):
        # Dedicated to multi-class
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusPlusCE is for multi-class only (please use BCE for regions)"

        tverskypp_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "gamma_pp": 2.0,
            "gamma_focal": 4.0 / 3.0,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth_nr": 1e-5,
            "smooth_dr": 1e-5,
            "do_bg": False,  # Exclude background for multi-class
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = FocalTverskyPlusPlusCELoss(
            tverskypp_kwargs=tverskypp_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tverskypp=1.0,
            weight_ce=1.0,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusWithBackgroundPlusCE(RSNA2025Trainer_moreDAv3):
    """
    Trainer combining Focal Tversky++ and Cross Entropy (moreDAv3).

    Dedicated to multi-class (use BCE for regions=True).
    """

    def _build_loss(self):
        # Dedicated to multi-class
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusPlusCE is for multi-class only (please use BCE for regions)"

        tverskypp_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "gamma_pp": 2.0,
            "gamma_focal": 4.0 / 3.0,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth_nr": 1e-5,
            "smooth_dr": 1e-5,
            "do_bg": True,  # Exclude background for multi-class
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = FocalTverskyPlusPlusCELoss(
            tverskypp_kwargs=tverskypp_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tverskypp=1.0,
            weight_ce=1.0,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
