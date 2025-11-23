"""
Trainer for RSNA2025 including Tversky Loss implementation

Tversky Loss is a generalized version of Dice Loss that allows different weighting for false positives and false negatives.
You can adjust the balance between FP and FN with alpha and beta parameters.
If alpha=beta=0.5, it becomes equivalent to Dice Loss.
"""

from typing import Callable
import warnings

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv3 import RSNA2025Trainer_moreDAv3
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv5 import RSNA2025Trainer_moreDAv5
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientTverskyLoss(nn.Module):
    """
    Tversky Loss implementation

    Tversky Index = TP / (TP + alpha*FP + beta*FN)

    Parameters:
        alpha: Weight for false positives (default: 0.3)
        beta: Weight for false negatives (default: 0.7)
        smooth: Smoothing term to avoid division by zero
        apply_nonlin: Activation function (softmax or sigmoid)
        batch_dice: Flag for batch-wise calculation
        do_bg: Whether to include background class
        ddp: Flag for distributed data parallel support
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
    ):
        super(MemoryEfficientTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

        # It is recommended (but not mandatory) that alpha and beta sum to 1
        if abs(self.alpha + self.beta - 1.0) > 0.01:
            warnings.warn(
                f"alpha ({alpha}) + beta ({beta}) = {alpha + beta} != 1.0. Usually, it is recommended to set alpha + beta = 1.0",
                UserWarning,
            )

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Convert dimensions to (b, c) shape
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # If GT is already one-hot encoded
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            # True Positive + False Negative (GT)
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # If excluding background
        if not self.do_bg:
            x = x[:, 1:]

        # True Positive
        if loss_mask is None:
            tp = (x * y_onehot).sum(axes)
            # False Positive + True Positive (Prediction)
            sum_pred = x.sum(axes)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        # False Positive = Prediction - TP
        fp = sum_pred - tp
        # False Negative = GT - TP
        fn = sum_gt - tp

        if self.ddp and self.batch_dice:
            # Aggregation during distributed training
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        # Calculate Tversky Index
        if self.batch_dice:
            # Sum over batch then class average (same aggregation as nnUNet Dice)
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            # Sum again as batch dimension still remains
            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

            tversky_per_class = (tp + self.smooth) / torch.clip(
                tp + self.alpha * fp + self.beta * fn + self.smooth, min=1e-8
            )
            tversky = tversky_per_class.mean()
        else:
            # Average Tversky per sample and per class
            nominator = tp + self.smooth
            denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky = (nominator / torch.clip(denominator, min=1e-8)).mean()

        # Return as negative value for loss
        return -tversky


class TverskyPlusCELoss(nn.Module):
    """
    Combination of Tversky Loss and Cross Entropy Loss
    """

    def __init__(self, tversky_kwargs, ce_kwargs, weight_tversky=1, weight_ce=1, ignore_label=None):
        super().__init__()
        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.tversky = MemoryEfficientTverskyLoss(**tversky_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output, target):
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

        tversky_loss = self.tversky(net_output, target_tversky, loss_mask=mask)

        if self.ignore_label is None or (num_fg is not None and num_fg > 0):
            ce_loss = self.ce(net_output, target)
        else:
            ce_loss = torch.as_tensor(0.0, device=net_output.device, dtype=net_output.dtype)

        result = self.weight_tversky * tversky_loss + self.weight_ce * ce_loss
        return result


class RSNA2025Trainer_moreDAv3_TverskyLoss(RSNA2025Trainer_moreDAv3):
    """
    RSNA2025 trainer using Tversky Loss

    You can adjust weights for false positives and false negatives with alpha and beta parameters.
    By default, it imposes a larger penalty on false negatives (misses).
    """

    def _build_loss(self):
        """Build Tversky Loss"""
        tversky_alpha = 0.3
        tversky_beta = 0.7

        if self.label_manager.has_regions:
            # Sigmoid activation for multi-label (regions)
            loss = MemoryEfficientTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                apply_nonlin=torch.sigmoid,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=True,
                smooth=1e-5,
                ddp=self.is_ddp,
            )
        else:
            # Softmax activation for multi-class
            loss = MemoryEfficientTverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                apply_nonlin=softmax_helper_dim1,
                batch_dice=self.configuration_manager.batch_dice,
                do_bg=False,
                smooth=1e-5,
                ddp=self.is_ddp,
            )

        # Deep Supervision settings
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # Exponentially decrease weights as resolution decreases
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if self.is_ddp and not self._do_i_compile():
                # Special case for DDP
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # Normalize weights
            weights = weights / weights.sum()

            # Wrap with Deep Supervision wrapper
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class RSNA2025Trainer_moreDAv3_TverskyPlusCE(RSNA2025Trainer_moreDAv3):
    """
    Trainer combining Tversky Loss and Cross Entropy Loss
    """

    def _build_loss(self):
        """Build composite Tversky + CE loss"""
        # This trainer is for multi-class only (use BCE for regions)
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv3_TverskyPlusCE is for multi-class only (please use BCE for regions)"

        tversky_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,  # Exclude background in multi-class
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = TverskyPlusCELoss(
            tversky_kwargs=tversky_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tversky=1,
            weight_ce=1,
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


class RSNA2025Trainer_moreDAv5_TverskyPlusCE(RSNA2025Trainer_moreDAv5):
    """
    Trainer combining Tversky Loss and Cross Entropy Loss
    """

    def _build_loss(self):
        """Build composite Tversky + CE loss"""
        # This trainer is for multi-class only (use BCE for regions)
        assert (
            not self.label_manager.has_regions
        ), "RSNA2025Trainer_moreDAv5_TverskyPlusCE is for multi-class only (please use BCE for regions)"

        tversky_kwargs = {
            "alpha": 0.3,
            "beta": 0.7,
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,  # Exclude background in multi-class
            "ddp": self.is_ddp,
            "apply_nonlin": softmax_helper_dim1,
        }

        ce_kwargs = {}
        if self.label_manager.ignore_label is not None:
            ce_kwargs["ignore_index"] = self.label_manager.ignore_label

        loss = TverskyPlusCELoss(
            tversky_kwargs=tversky_kwargs,
            ce_kwargs=ce_kwargs,
            weight_tversky=1,
            weight_ce=1,
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
