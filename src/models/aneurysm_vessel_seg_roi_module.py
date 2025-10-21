from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import numpy as np
from sklearn.metrics import roc_auc_score
from monai.transforms import Compose
from monai.losses import DiceCELoss, DiceLoss
from timm.utils import ModelEmaV3

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.region_mask_pooling import RegionMaskedPooling3D
from src.data.components.aneurysm_vessel_seg_dataset import ANEURYSM_CLASSES
from src.models.losses.balanced_bce import BalancedBCEWithLogitsLoss
from src.models.losses.focal_tversky_plusplus import FocalTverskyPlusPlusLoss
from src.data.components.custom_transforms import (
    RandMedianSmoothdVaried,
    InvertImageTransform,
    BatchedRandSimulateLowResolutiond,
    BatchedRandAffined,
    BatchedRandFlipd,
    BatchedRandAxisSwapd,
    BatchedRandGridDistortiond,
    BatchedRandShrinkPadToOriginald,
)
from src.models.components.metadata_embedding import MetadataEmbedding


class VesselROIRuntimeModule(nn.Module):
    """Helper module bundling backbone and heads, clarifying EMA target"""

    def __init__(
        self,
        net: nn.Module,
        cls_hidden: int,
        cls_dropout: float,
        mask_pool_modes_loc: Sequence[str] | str = "mean",
        global_pool_modes_loc: Sequence[str] | str = "mean",
        mask_pool_modes_ap: Sequence[str] | str = "mean",
        global_pool_modes_ap: Sequence[str] | str = "mean",
        gem_p: float = 3.0,
        gem_eps: float = 1e-6,
        use_encoder_global_pooling_loc: bool = False,
        use_encoder_global_pooling_ap: bool = False,
        # Optional: pooling with dilated masks (configurable for loc/AP)
        add_dilated_mask_loc: bool = False,
        add_dilated_mask_ap: bool = False,
        # Optional: dilation kernel (D,H,W) and iterations
        dilate_kernel: int | Sequence[int] = (1, 3, 3),
        dilate_iters: int = 1,
        # Optional: per-branch normalization and linear projection (Plan A/B). None keeps backward-compat.
        # Plan A: set proj_global_dim_* (e.g., 32) while masked remains None
        # Plan B: add proj_mask_dim_* (e.g., 64) in addition to Plan A
        branch_norm: bool = False,
        proj_global_dim_loc: Optional[int] = None,
        proj_mask_dim_loc: Optional[int] = None,
        proj_global_dim_ap: Optional[int] = None,
        proj_mask_dim_ap: Optional[int] = None,
        metadata_numeric_dim: int = 0,
        metadata_categorical_cardinalities: Optional[Sequence[int]] = None,
        metadata_embedding_cfg: Optional[Dict[str, Any]] = None,
        ema_enabled: bool = False,
        ema_decay: float = 0.999,
        ema_update_after_step: int = 0,
        # Number of extra segmentation mask branches (0=disabled, 1=one extra)
        num_extra_mask_branches: int = 0,
    ) -> None:
        super().__init__()
        self.net = net
        self.num_extra_mask_branches = int(max(0, num_extra_mask_branches))

        use_encoder_loc = bool(use_encoder_global_pooling_loc)
        use_encoder_ap = bool(use_encoder_global_pooling_ap)

        self._mask_feat_channels = int(self.net.feature_channels())
        self._encoder_global_feat_channels: Optional[int] = None

        def _resolve_global_channels(use_encoder: bool) -> int:
            if use_encoder:
                if self._encoder_global_feat_channels is None:
                    self._encoder_global_feat_channels = int(self._infer_encoder_feature_channels())
                return int(self._encoder_global_feat_channels)
            return int(self._mask_feat_channels)

        self._global_feat_channels_loc = _resolve_global_channels(use_encoder_loc)
        self._global_feat_channels_ap = _resolve_global_channels(use_encoder_ap)

        self.rmp_loc = RegionMaskedPooling3D(
            mask_pool_modes=mask_pool_modes_loc,
            global_pool_modes=global_pool_modes_loc,
            gem_p=gem_p,
            gem_eps=gem_eps,
            use_encoder_global_feat=use_encoder_loc,
            add_dilated_mask=bool(add_dilated_mask_loc),
            dilate_kernel=dilate_kernel,
            dilate_iters=int(dilate_iters),
            mask_feat_channels=self._mask_feat_channels,
            global_feat_channels=self._global_feat_channels_loc,
            branch_norm=branch_norm,
            proj_mask_dim=proj_mask_dim_loc,
            proj_global_dim=proj_global_dim_loc,
        )
        self.rmp_ap = RegionMaskedPooling3D(
            mask_pool_modes=mask_pool_modes_ap,
            global_pool_modes=global_pool_modes_ap,
            gem_p=gem_p,
            gem_eps=gem_eps,
            use_encoder_global_feat=use_encoder_ap,
            add_dilated_mask=bool(add_dilated_mask_ap),
            dilate_kernel=dilate_kernel,
            dilate_iters=int(dilate_iters),
            mask_feat_channels=self._mask_feat_channels,
            global_feat_channels=self._global_feat_channels_ap,
            branch_norm=branch_norm,
            proj_mask_dim=proj_mask_dim_ap,
            proj_global_dim=proj_global_dim_ap,
        )

        self.use_encoder_global_pooling_loc = bool(self.rmp_loc.use_encoder_global_feat)
        self.use_encoder_global_pooling_ap = bool(self.rmp_ap.use_encoder_global_feat)
        self.use_encoder_global_pooling = (
            self.use_encoder_global_pooling_loc or self.use_encoder_global_pooling_ap
        )

        # Extend local-branch output dim considering extra mask branches
        try:
            _mask_dim = int(self.rmp_loc.mask_output_dim())
        except Exception:
            _mask_dim = 0
        try:
            _glob_dim = int(self.rmp_loc.global_output_dim())
        except Exception:
            _glob_dim = 0
        self._pooled_feat_channels_loc = int(_glob_dim + _mask_dim * (1 + self.num_extra_mask_branches))
        self._pooled_feat_channels_ap = int(self.rmp_ap.output_dim())

        if self._pooled_feat_channels_loc <= 0:
            raise ValueError("RegionMaskedPooling3D (loc) did not produce features")
        if self._pooled_feat_channels_ap <= 0:
            raise ValueError("RegionMaskedPooling3D (AP) did not produce features")

        numeric_dim = int(metadata_numeric_dim)
        categorical_cards = tuple(metadata_categorical_cardinalities or [])
        embed_cfg = dict(metadata_embedding_cfg or {})
        if numeric_dim > 0 or categorical_cards:
            self.metadata_embedder = MetadataEmbedding(
                numeric_dim=numeric_dim,
                categorical_cardinalities=categorical_cards,
                **embed_cfg,
            )
            self._metadata_embed_dim = int(self.metadata_embedder.output_dim)
        else:
            self.metadata_embedder = None
            self._metadata_embed_dim = 0

        cls_in_dim = self._pooled_feat_channels_loc + self._metadata_embed_dim
        ap_in_dim = self._pooled_feat_channels_ap + self._metadata_embed_dim

        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, 1),
        )
        self.ap_head = nn.Sequential(
            nn.Linear(ap_in_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, 1),
        )

        self.ema_enabled = bool(ema_enabled)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_model: Optional[ModelEmaV3] = None
        if self.ema_enabled:
            # Map entire module for EMA to stabilize inference
            self.ema_model = ModelEmaV3(self, decay=float(ema_decay), device=None)

    def forward(
        self,
        x: torch.Tensor,
        vessel_seg: torch.Tensor | None = None,
        vessel_union: torch.Tensor | None = None,
        use_ema: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        module = self.get_runtime_module(use_ema=use_ema)
        if module is self:
            return self._forward_impl(x, vessel_seg=vessel_seg, vessel_union=vessel_union)
        return module._forward_impl(x, vessel_seg=vessel_seg, vessel_union=vessel_union)

    def _forward_impl(
        self,
        x: torch.Tensor,
        vessel_seg: torch.Tensor | None = None,
        vessel_union: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if vessel_seg is not None:
            try:
                out = self.net(x, vessel_seg=vessel_seg, vessel_union=vessel_union)
            except TypeError:
                out = self.net(x)
        else:
            out = self.net(x)

        if not isinstance(out, dict):
            raise TypeError("Backbone forward must return a dict")

        result = dict(out)
        if "feat" not in result:
            if "dec_feat" in result:
                result["feat"] = result["dec_feat"]
            else:
                raise KeyError("feat/dec_feat missing in backbone output")

        if self.use_encoder_global_pooling:
            if "enc_feat" not in result:
                raise KeyError("enc_feat is required when use_encoder_global_pooling=True")
            result["feat_global"] = result["enc_feat"]
        else:
            result["feat_global"] = result["feat"]

        return result

    def has_ema(self) -> bool:
        return self.ema_model is not None

    def get_runtime_module(self, use_ema: Optional[bool] = None) -> "VesselROIRuntimeModule":
        if not self.has_ema():
            return self
        if use_ema is None:
            use_ema = not self.training
        if use_ema:
            ema_module = self.ema_model.module
            ema_module.eval()
            return ema_module
        return self

    @torch.no_grad()
    def update_ema(self, step: Optional[int] = None) -> None:
        if not self.has_ema():
            return
        if step is not None and step < self.ema_update_after_step:
            return
        # ema_model does not require grad tracking
        self.ema_model.update(self)

    def feature_channels(self) -> int:
        """Return the feature channels exposed by the backbone"""
        return int(self.net.feature_channels())

    def global_feature_channels(self) -> int:
        """Feature channels used for global pooling (loc branch)"""
        return int(self._global_feat_channels_loc)

    def global_feature_channels_ap(self) -> int:
        """Feature channels used for global pooling (AP branch)"""
        return int(self._global_feat_channels_ap)

    def _infer_encoder_feature_channels(self) -> int:
        """Infer channel count of the last encoder feature"""
        if hasattr(self.net, "encoder_feature_channels"):
            ch = getattr(self.net, "encoder_feature_channels")
            return int(ch() if callable(ch) else ch)
        encoder = getattr(self.net, "encoder", None)
        if encoder is not None and hasattr(encoder, "encoder_feature_channels"):
            return int(encoder.encoder_feature_channels)
        if encoder is not None and hasattr(encoder, "feature_info"):
            info = encoder.feature_info()
            if isinstance(info, (list, tuple)) and len(info) > 0:
                return int(info[-1])
        if hasattr(self.net, "global_pool_feature_channels"):
            ch2 = getattr(self.net, "global_pool_feature_channels")
            return int(ch2() if callable(ch2) else ch2)
        raise AttributeError("Failed to infer encoder feature channels")

    def pooled_feature_channels(self) -> int:
        """Return channels after RegionMaskedPooling3D (post-branch projection)"""
        return int(self._pooled_feat_channels_loc + self._metadata_embed_dim)

    def pooled_feature_channels_ap(self) -> int:
        """Return channels after RegionMaskedPooling3D for AP (post-branch projection)"""
        return int(self._pooled_feat_channels_ap + self._metadata_embed_dim)

    @property
    def metadata_embed_dim(self) -> int:
        return int(self._metadata_embed_dim)

    def embed_metadata(
        self,
        metadata_numeric: Optional[torch.Tensor] = None,
        metadata_numeric_missing: Optional[torch.Tensor] = None,
        metadata_categorical: Optional[torch.Tensor] = None,
        *,
        device_hint: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Convert metadata tensors into an embedding"""

        if self.metadata_embedder is None:
            return None

        embedder = self.metadata_embedder
        kwargs: Dict[str, torch.Tensor] = {}

        param = next(embedder.parameters(), None)
        device: Optional[torch.device] = device_hint
        if device is None:
            if param is not None:
                device = param.device
            elif metadata_numeric is not None:
                device = metadata_numeric.device
            elif metadata_categorical is not None:
                device = metadata_categorical.device

        def _to(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if device is not None:
                return tensor.to(device=device, dtype=dtype)
            return tensor.to(dtype=dtype)

        if embedder.numeric_branch is not None:
            if metadata_numeric is None:
                raise ValueError("metadata_numeric must be provided")
            kwargs["numeric_values"] = _to(metadata_numeric, torch.float32)
            if embedder.numeric_use_missing_indicator or embedder.numeric_missing_embedding_enabled:
                if metadata_numeric_missing is None:
                    raise ValueError("metadata_numeric_missing must be provided")
                kwargs["numeric_missing"] = _to(metadata_numeric_missing, torch.float32)
            elif metadata_numeric_missing is not None:
                kwargs["numeric_missing"] = _to(metadata_numeric_missing, torch.float32)
        elif metadata_numeric is not None:
            kwargs["numeric_values"] = _to(metadata_numeric, torch.float32)
            if metadata_numeric_missing is not None:
                kwargs["numeric_missing"] = _to(metadata_numeric_missing, torch.float32)

        if embedder.categorical_embeddings is not None:
            if metadata_categorical is None:
                raise ValueError("metadata_categorical must be provided")
            kwargs["categorical_indices"] = _to(metadata_categorical, torch.long)
        elif metadata_categorical is not None:
            kwargs["categorical_indices"] = _to(metadata_categorical, torch.long)

        if not kwargs:
            return None

        return embedder(**kwargs)


class AneurysmVesselSegROILitModule(LightningModule):
    """
    Pool decoder features using vessel segmentation region masks and perform
    multi-label classification per location. Also train a 1ch sphere-mask as
    an auxiliary task.
    """

    def __init__(
        self,
        net: nn.Module,
        # Optimization
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        optimize_config: Optional[Dict[str, Any]] = None,
        compile: bool = False,
        # Pooling/Head
        cls_hidden: int = 256,
        cls_dropout: float = 0.1,
        mask_pool_modes_loc: Sequence[str] | str = "mean",
        global_pool_modes_loc: Sequence[str] | str = "mean",
        mask_pool_modes_ap: Sequence[str] | str = "mean",
        global_pool_modes_ap: Sequence[str] | str = "mean",
        gem_p: float = 3.0,
        gem_eps: float = 1e-6,
        use_encoder_global_pooling_loc: bool = False,
        use_encoder_global_pooling_ap: bool = False,
        # Dilated-mask pooling (optional)
        add_dilated_mask_loc: bool = False,
        add_dilated_mask_ap: bool = False,
        dilate_kernel: int | Sequence[int] = (1, 3, 3),
        dilate_iters: int = 1,
        # Per-branch normalization + projection (Plan A/B)
        branch_norm: bool = False,
        proj_global_dim_loc: Optional[int] = None,
        proj_mask_dim_loc: Optional[int] = None,
        proj_global_dim_ap: Optional[int] = None,
        proj_mask_dim_ap: Optional[int] = None,
        # Loss types
        use_dice_loss: bool = False,
        # Loss weights
        w_loc: float = 1.0,
        w_ap: float = 1.0,
        w_sphere: float = 1.0,
        # Number of classes
        num_location_classes: int = 13,
        # Number of vessel classes (K) used during training; None/<=0/>=13 -> use all (handled by dataset).
        train_select_k_vessels: Optional[int] = None,
        # Sphere re-generation radius (used after GPU augmentation)
        sphere_radius: int = 10,
        # Metadata
        metadata_numeric_dim: int = 0,
        metadata_categorical_cardinalities: Optional[Sequence[int]] = None,
        metadata_embedding_cfg: Optional[Dict[str, Any]] = None,
        # EMA-related hyperparameters
        ema: bool = False,
        ema_decay: float = 0.999,
        ema_update_after_step: int = 0,
        # How many extra segmentation branches to concatenate (0/1)
        num_extra_mask_branches: int = 0,
        # GPU augmentations
        rand_swap_prob: float = 0.0,
        distort_limit: float = 0.1,
        rand_shrink_pad_prob: float = 0.0,
        rand_shrink_pad_max_ratio: float = 0.1,
        # Load pretrained net weights (optional)
        pretrained_net_ckpt: Optional[str] = None,
        pretrained_state_key: Optional[str] = None,
        pretrained_strict: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Manage runtime-related modules
        self.model = VesselROIRuntimeModule(
            net=net,
            cls_hidden=cls_hidden,
            cls_dropout=cls_dropout,
            mask_pool_modes_loc=mask_pool_modes_loc,
            global_pool_modes_loc=global_pool_modes_loc,
            mask_pool_modes_ap=mask_pool_modes_ap,
            global_pool_modes_ap=global_pool_modes_ap,
            gem_p=gem_p,
            gem_eps=gem_eps,
            use_encoder_global_pooling_loc=use_encoder_global_pooling_loc,
            use_encoder_global_pooling_ap=use_encoder_global_pooling_ap,
            add_dilated_mask_loc=add_dilated_mask_loc,
            add_dilated_mask_ap=add_dilated_mask_ap,
            dilate_kernel=dilate_kernel,
            dilate_iters=dilate_iters,
            branch_norm=branch_norm,
            proj_global_dim_loc=proj_global_dim_loc,
            proj_mask_dim_loc=proj_mask_dim_loc,
            proj_global_dim_ap=proj_global_dim_ap,
            proj_mask_dim_ap=proj_mask_dim_ap,
            metadata_numeric_dim=metadata_numeric_dim,
            metadata_categorical_cardinalities=metadata_categorical_cardinalities,
            metadata_embedding_cfg=metadata_embedding_cfg,
            ema_enabled=ema,
            ema_decay=ema_decay,
            ema_update_after_step=ema_update_after_step,
            num_extra_mask_branches=int(max(0, num_extra_mask_branches)),
        )

        # Load pretrained net weights (if provided)
        if pretrained_net_ckpt is not None and str(pretrained_net_ckpt).strip():
            try:
                self._load_pretrained_net_weights(
                    ckpt_path=str(pretrained_net_ckpt),
                    state_key=pretrained_state_key,
                    strict=bool(pretrained_strict),
                )
            except Exception as e:
                # Warn but continue training if load fails
                print(f"[WARN] Failed to load pretrained net weights: {e}")

        # Losses
        self.crit_loc = nn.BCEWithLogitsLoss()
        self.crit_ap = nn.BCEWithLogitsLoss()
        self.crit_sphere = BalancedBCEWithLogitsLoss()
        if use_dice_loss:
            self.dice_sphere = DiceLoss(include_background=True, sigmoid=True, reduction="mean")
        else:
            self.dice_sphere = FocalTverskyPlusPlusLoss(
                alpha=0.3,
                beta=0.7,
                gamma_pp=2.0,
                gamma_focal=1.33,
                include_background=True,
                sigmoid=True,
                reduction="mean",
            )

        # Buffers (for computing AUC)
        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_logits: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

        # Visualization flags (only first step per epoch)
        self._did_train_viz_this_epoch: bool = False
        self._did_val_viz_this_epoch: bool = False

        # Batch-friendly GPU augmentations (custom torch implementation)
        # Build keys dynamically to support multiple extra segmentations
        num_extra = int(max(0, getattr(self.hparams, "num_extra_mask_branches", 0)))
        extra_label_keys: list[str] = []
        if num_extra > 0:
            extra_label_keys.append("extra_vessel_label")
            for i in range(1, num_extra):
                extra_label_keys.append(f"extra_vessel_label_{i}")
        self._extra_label_keys = extra_label_keys
        self._label_aug_keys = ["image", "vessel_label", *self._extra_label_keys]
        if distort_limit > 0:
            prob_distort = 0.3
        else:
            prob_distort = 0.0
        # Low-resolution simulation (image only)
        lr_image = BatchedRandSimulateLowResolutiond(
            keys=["image"], zoom_range=(0.5, 1.0), prob=0.3, allow_missing_keys=True
        )
        # Low-resolution simulation (image + labels)
        lr_keys = list(self._label_aug_keys)
        # Apply stronger shrink to first (image), milder to labels (nearest interpolation)
        zoom_seq = [(0.4, 0.7)] + [(0.8, 1.0)] * (len(lr_keys) - 1)
        lr_all = BatchedRandSimulateLowResolutiond(
            keys=lr_keys, zoom_range=zoom_seq, prob=0.3, allow_missing_keys=True, mode="nearest"
        )
        # Interpolation for distortion/affine (first trilinear, others nearest)
        modes = tuple(["trilinear"] + ["nearest"] * (len(self._label_aug_keys) - 1))
        self._gpu_train_tf_label = Compose(
            [
                lr_image,
                lr_all,
                # Shape-preserving: random shrink + center padding (shrink Z and XY independently; XY same ratio)
                BatchedRandShrinkPadToOriginald(
                    keys=self._label_aug_keys,
                    prob=rand_shrink_pad_prob,
                    max_shrink_ratio=rand_shrink_pad_max_ratio,
                    mode=modes,
                    pad_value=0.0,
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                ),
                BatchedRandGridDistortiond(
                    keys=self._label_aug_keys,
                    prob=prob_distort,
                    num_cells=(4, 4, 4),
                    distort_limit=distort_limit,
                    mode=modes,
                    padding_mode="zeros",
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                ),
                BatchedRandAffined(
                    keys=self._label_aug_keys,
                    prob=0.6,
                    rotate_range=(float(torch.pi) / 18, float(torch.pi) / 18, float(torch.pi) / 18),
                    scale_range=0.10,
                    shear_range=0.10,
                    mode=modes,
                    padding_mode="zeros",
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                ),
                # For torch.compile stability, swap XY while preserving shape (assume H==W)
                BatchedRandAxisSwapd(
                    keys=self._label_aug_keys,
                    prob=rand_swap_prob,
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                    permutations=[(0, 1, 2), (0, 2, 1)],
                    ensure_same_shape=True,
                ),
                BatchedRandFlipd(
                    keys=self._label_aug_keys,
                    spatial_axis=0,
                    prob=0.5,
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                ),
                BatchedRandFlipd(
                    keys=self._label_aug_keys,
                    spatial_axis=1,
                    prob=0.5,
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                ),
                BatchedRandFlipd(
                    keys=self._label_aug_keys,
                    spatial_axis=2,
                    prob=0.5,
                    allow_missing_keys=True,
                    point_keys=("ann_points",),
                ),
            ]
        )

        # Track optimizer.step() count for EMA update guard
        self._ema_last_optimizer_step: int = 0

    # ===== GPU augmentation utilities =====

    def _rasterize_spheres_from_points(
        self, pts: torch.Tensor, valid: torch.Tensor, out_shape: torch.Size, radius: int
    ) -> torch.Tensor:
        """Create a (sum of) sphere masks from (B,M,3)[z,y,x] coordinates entirely on GPU.
        Args:
            pts: (B,M,3) float coordinates (voxel units)
            valid: (B,M) 1/0 flags
            out_shape: output size matching (B,1,D,H,W)
            radius: radius (in voxels)
        Returns:
            (B,1,D,H,W) uint8
        """
        B, M, _ = pts.shape
        _, _, D, H, W = out_shape
        device = pts.device
        z = torch.arange(D, device=device, dtype=pts.dtype).view(1, 1, D, 1, 1)
        y = torch.arange(H, device=device, dtype=pts.dtype).view(1, 1, 1, H, 1)
        x = torch.arange(W, device=device, dtype=pts.dtype).view(1, 1, 1, 1, W)

        v = (valid > 0).to(pts.dtype).unsqueeze(-1)
        pz = pts[..., 0:1] * v + (1.0 - v) * (-1e6)
        py = pts[..., 1:2] * v + (1.0 - v) * (-1e6)
        px = pts[..., 2:3] * v + (1.0 - v) * (-1e6)

        pzv = pz.view(B, M, 1, 1, 1)
        pyv = py.view(B, M, 1, 1, 1)
        pxv = px.view(B, M, 1, 1, 1)
        dz2 = (z - pzv).pow(2)
        dy2 = (y - pyv).pow(2)
        dx2 = (x - pxv).pow(2)
        mask_multi = (dz2 + dy2 + dx2) <= float(radius * radius)
        return mask_multi.any(dim=1, keepdim=True).to(torch.uint8)

    def _label_to_masks(
        self,
        vessel_label: torch.Tensor,
        ref_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vessel_label.dim() not in (4, 5):
            raise ValueError(f"Invalid vessel_label dimensions: {vessel_label.shape}")
        lbl = vessel_label
        if lbl.dim() == 4:
            lbl = lbl.unsqueeze(1)
        lbl_long = lbl.long()
        num_loc = int(self.hparams.num_location_classes)
        one_hot = F.one_hot(lbl_long.squeeze(1), num_classes=num_loc + 1)
        vessel_masks = one_hot[..., 1:].permute(0, 4, 1, 2, 3).contiguous()
        vessel_masks = vessel_masks.to(dtype=ref_tensor.dtype)
        vessel_union = (lbl_long > 0).to(dtype=ref_tensor.dtype)
        return vessel_masks, vessel_union

    @torch.no_grad()
    def _gpu_augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        if "vessel_label" not in batch:
            raise ValueError("vessel_label is required for the seg module")

        batch_local = dict(batch)
        label = batch_local["vessel_label"]
        if label.dim() == 4:
            label = label.unsqueeze(1)
        label = label.to(dtype=torch.float32)
        batch_local["vessel_label"] = label

        # Collect extra labels (multiple)
        extra_labels: list[torch.Tensor] = []
        for i, key in enumerate(self._extra_label_keys):
            lab = batch_local.get(key)
            if lab is None:
                continue
            if lab.dim() == 4:
                lab = lab.unsqueeze(1)
            lab = lab.to(dtype=torch.float32)
            batch_local[key] = lab
            extra_labels.append(lab)

        data: Dict[str, torch.Tensor] = {
            "image": batch_local["image"],
            "vessel_label": batch_local["vessel_label"],
        }
        if "ann_points" in batch_local:
            data["ann_points"] = batch_local["ann_points"]
        # Prepare extra labels for transforms (missing allowed due to allow_missing_keys=True)
        for key in self._extra_label_keys:
            if key in batch_local:
                data[key] = batch_local[key]

        if training:
            data = self._gpu_train_tf_label(data)

        label_aug = data["vessel_label"].round().clamp_(0, self.hparams.num_location_classes).long()
        vessel_seg, vessel_union = self._label_to_masks(label_aug, data["image"])

        # Convert all extra segmentations to masks and concatenate channels (13ch each)
        extra_label_aug_first: Optional[torch.Tensor] = None
        extra_masks: list[torch.Tensor] = []
        for idx, key in enumerate(self._extra_label_keys):
            if key not in data:
                continue
            lab_aug = data[key].round().clamp_(0, self.hparams.num_location_classes).long()
            if extra_label_aug_first is None:
                extra_label_aug_first = lab_aug
            masks, union_e = self._label_to_masks(lab_aug, data["image"])
            extra_masks.append(masks)
            vessel_union = torch.maximum(vessel_union, union_e)
        extra_vessel_seg_cat: Optional[torch.Tensor] = None
        if len(extra_masks) > 0:
            extra_vessel_seg_cat = torch.cat(extra_masks, dim=1)

        if "ann_points" in data and "ann_points_valid" in batch_local:
            radius = int(self.hparams.sphere_radius)
            sphere_mask = self._rasterize_spheres_from_points(
                data["ann_points"],
                batch_local["ann_points_valid"].to(data["ann_points"].device),
                data["image"].shape,
                radius,
            )
        else:
            sphere_mask = torch.zeros_like(label_aug, dtype=torch.uint8)

        out_batch = dict(batch_local)
        out_batch["image"] = data["image"]
        out_batch["vessel_label"] = label_aug
        out_batch["vessel_seg"] = vessel_seg
        out_batch["vessel_union"] = vessel_union
        out_batch["sphere_mask"] = sphere_mask
        if "ann_points" in data:
            out_batch["ann_points"] = data["ann_points"]
        if extra_label_aug_first is not None and extra_vessel_seg_cat is not None:
            out_batch["extra_vessel_label"] = extra_label_aug_first
            out_batch["extra_vessel_seg"] = extra_vessel_seg_cat

        return out_batch

    def _use_ema_for_eval(self) -> bool:
        """Return True if EMA is used during evaluation"""
        return bool(getattr(self.hparams, "ema", False)) and self.model.has_ema()

    def _get_completed_optimizer_steps(self) -> Optional[int]:
        """Get number of optimizer.step() calls from Lightning progress"""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        fit_loop = getattr(trainer, "fit_loop", None)
        if fit_loop is None or not hasattr(fit_loop, "epoch_loop"):
            return None
        auto_loop = getattr(fit_loop.epoch_loop, "automatic_optimization", None)
        if auto_loop is None:
            return None
        try:
            return int(auto_loop.optim_progress.optimizer_steps)
        except Exception:
            try:
                step_total = auto_loop.optim_progress.optimizer.step.total.completed
            except Exception:
                return None
            return int(step_total)

    def forward(
        self,
        x: torch.Tensor,
        vessel_seg: torch.Tensor,
        vessel_union: torch.Tensor,
        extra_vessel_seg: Optional[torch.Tensor] = None,
        metadata_numeric: Optional[torch.Tensor] = None,
        metadata_numeric_missing: Optional[torch.Tensor] = None,
        metadata_categorical: Optional[torch.Tensor] = None,
        use_ema: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        vessel_seg = vessel_seg.to(dtype=x.dtype, device=x.device)
        vessel_union = vessel_union.to(dtype=x.dtype, device=x.device)
        if extra_vessel_seg is not None:
            extra_vessel_seg = extra_vessel_seg.to(dtype=x.dtype, device=x.device)

        runtime_module = self.model.get_runtime_module(use_ema=use_ema)
        out = runtime_module._forward_impl(x, vessel_seg=vessel_seg, vessel_union=vessel_union)

        feat = out["feat"]
        global_feat = self._get_global_pool_feature(out)
        metadata_embed = runtime_module.embed_metadata(
            metadata_numeric=metadata_numeric,
            metadata_numeric_missing=metadata_numeric_missing,
            metadata_categorical=metadata_categorical,
            device_hint=feat.device,
        )

        logits_loc = self._classify_with_masks_subset(
            runtime_module,
            feat,
            vessel_seg,
            extra_vessel=extra_vessel_seg,
            global_feat=global_feat,
            metadata_embed=metadata_embed,
        )
        logit_ap = self._classify_ap_from_union(
            runtime_module,
            feat,
            vessel_union,
            global_feat=global_feat,
            metadata_embed=metadata_embed,
        )

        out = dict(out)
        out["logits_loc"] = logits_loc
        out["logit_ap"] = logit_ap
        return out

    # ===== Visualization utilities =====
    def _ensure_viz_dir(self) -> str:
        """Create and return visualization directory (with per-fold subdir)"""
        import os

        # Fallback for environments without trainer.log_dir
        log_dir = getattr(self.trainer, "log_dir", None)
        if log_dir is None:
            # Prefer logger if available
            logger = getattr(self.trainer, "logger", None)
            log_dir = getattr(logger, "log_dir", None) if logger is not None else None
        if log_dir is None:
            log_dir = os.path.join("outputs", "logs")

        # Get fold index from DataModule (default 0)
        fold = 0
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "hparams"):
            try:
                fold = int(getattr(self.trainer.datamodule.hparams, "fold", 0))
            except Exception:
                fold = 0

        viz_dir = os.path.join(log_dir, "visualization", f"fold{fold}")
        os.makedirs(viz_dir, exist_ok=True)
        return viz_dir

    @staticmethod
    def _to_numpy_display(x: torch.Tensor) -> np.ndarray:
        """Normalize to [0,1] for imshow and convert to numpy. Input (D,H,W)."""
        x = x.detach().float().cpu().numpy()
        vmin = np.percentile(x, 1.0)
        vmax = np.percentile(x, 99.0)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = x - x.min()
            den = x.max() if x.max() > 0 else 1.0
            x = x / den
        return x

    @staticmethod
    def _center_from_annotations_or_mask(
        ann_points: Optional[torch.Tensor], ann_valid: Optional[torch.Tensor], mask_3d: Optional[torch.Tensor]
    ) -> tuple[int, int, int]:
        """
        Choose a center for slicing: use one of annotation points (z,y,x),
        otherwise mask centroid, otherwise volume center.
        Returns: (z,y,x)
        """
        # 1) First valid annotation point
        if ann_points is not None and ann_valid is not None:
            # (M,3), (M,)
            msk = (ann_valid > 0).nonzero(as_tuple=False)
            if msk.numel() > 0:
                i = int(msk[0].item())
                p = ann_points[i]
                zyx = (int(round(float(p[0]))), int(round(float(p[1]))), int(round(float(p[2]))))
                return zyx
        # 2) Mask centroid vicinity (median)
        if mask_3d is not None:
            nz = (mask_3d > 0).nonzero(as_tuple=False)
            if nz.numel() > 0:
                z = int(nz[:, 0].float().median().item())
                y = int(nz[:, 1].float().median().item())
                x = int(nz[:, 2].float().median().item())
                return (z, y, x)
        # 3) Fallback: center
        D, H, W = mask_3d.shape if mask_3d is not None else (128, 224, 224)
        return (D // 2, H // 2, W // 2)

    @torch.no_grad()
    def _visualize_batch_first(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        split: str,
        batch_idx: int,
    ) -> None:
        """
        Save image, sphere GT, and sphere prediction in 3 views at the first step of each epoch (batch_idx==0).
        Slice centered on the first available annotation point.
        """
        if batch_idx != 0:
            return
        # Only rank 0 to avoid duplicate saves under DDP
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        import os
        import matplotlib

        matplotlib.use("Agg")  # headless environments
        import matplotlib.pyplot as plt

        image = batch.get("image")  # (B,1,D,H,W)
        sphere_gt = batch.get("sphere_mask")  # (B,1,D,H,W) uint8
        ann_pts = batch.get("ann_points", None)  # optional (B,M,3)
        ann_val = batch.get("ann_points_valid", None)  # optional (B,M)
        series_uid = batch.get("series_uid", None)

        if image is None or sphere_gt is None:
            return

        # Visualize only the first sample in the batch
        img = image[0, 0]  # (D,H,W)
        gt = sphere_gt[0, 0].to(dtype=torch.float32)  # (D,H,W)

        # Resize predicted logits to image resolution and convert to probabilities
        logits_sphere = out.get("logits_sphere", None)
        pred_prob = None
        if logits_sphere is not None:
            pr = F.interpolate(logits_sphere[0:1], size=img.shape, mode="trilinear", align_corners=False)[
                0, 0
            ]
            pred_prob = torch.sigmoid(pr)  # (D,H,W)

        # Determine center coordinates
        pts0 = ann_pts[0] if ann_pts is not None else None
        val0 = ann_val[0] if ann_val is not None else None
        zc, yc, xc = self._center_from_annotations_or_mask(pts0, val0, gt)
        D, H, W = img.shape
        zc = int(max(0, min(zc, D - 1)))
        yc = int(max(0, min(yc, H - 1)))
        xc = int(max(0, min(xc, W - 1)))

        # Normalize for display
        img_np = self._to_numpy_display(img)
        gt_np = (gt > 0.5).cpu().numpy()
        pr_np = pred_prob.detach().cpu().numpy() if pred_prob is not None else None

        # Render three orthogonal views
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # Axial (Z)
        axes[0].imshow(img_np[zc], cmap="gray")
        if pr_np is not None:
            axes[0].imshow(pr_np[zc], cmap="hot", alpha=0.6, vmin=0, vmax=1)
        axes[0].contour(gt_np[zc].astype(float), levels=[0.5], colors="cyan", linewidths=1)
        axes[0].set_title(f"Axial z={zc}")
        axes[0].axis("off")
        # Coronal (Y)
        axes[1].imshow(img_np[:, yc, :], cmap="gray", origin="lower")
        if pr_np is not None:
            axes[1].imshow(pr_np[:, yc, :], cmap="hot", alpha=0.6, origin="lower", vmin=0, vmax=1)
        axes[1].contour(
            gt_np[:, yc, :].astype(float), levels=[0.5], colors="cyan", linewidths=1, origin="lower"
        )
        axes[1].set_title(f"Coronal y={yc}")
        axes[1].axis("off")
        # Sagittal (X)
        axes[2].imshow(img_np[:, :, xc], cmap="gray", origin="lower")
        if pr_np is not None:
            axes[2].imshow(pr_np[:, :, xc], cmap="hot", alpha=0.6, origin="lower", vmin=0, vmax=1)
        axes[2].contour(
            gt_np[:, :, xc].astype(float), levels=[0.5], colors="cyan", linewidths=1, origin="lower"
        )
        axes[2].set_title(f"Sagittal x={xc}")
        axes[2].axis("off")
        plt.tight_layout()

        # Save figure
        viz_dir = self._ensure_viz_dir()
        uid = None
        if isinstance(series_uid, (list, tuple)):
            uid = series_uid[0]
        elif isinstance(series_uid, torch.Tensor) and series_uid.dim() == 0:
            uid = str(series_uid.item())
        elif isinstance(series_uid, torch.Tensor):
            # Should not be Tensor[str], but guard anyway
            uid = None
        else:
            uid = series_uid if series_uid is not None else "sample0"
        fname = f"epoch_{self.current_epoch:04d}_{split}_b0_{uid}.png"
        save_path = os.path.join(viz_dir, fname)
        plt.savefig(save_path)
        plt.close(fig)

    @staticmethod
    def _ensure_mask_channels(vessel: torch.Tensor) -> torch.Tensor:
        """Ensure vessel-mask channels are in the 13-class detection order.
        - If input is (14,...): drop leading 0 (background) and reorder seg-order -> detection-order
        - If input is (13,...): assume it's already in detection order

        Note: dataset typically provides detection-order; reorder here for robustness.
        """
        if vessel.dim() != 5:
            raise ValueError("vessel_seg must be (B,C,D,H,W)")
        if vessel.shape[1] == 14:
            # Remove background -> seg-order (13ch)
            v13 = vessel[:, 1:, ...]
            # seg-order -> detection-order
            # det_idx(0..12) -> seg_offset(0..12): [5,4,7,6,9,8,12,11,10,3,2,1,0]
            det_to_seg = [5, 4, 7, 6, 9, 8, 12, 11, 10, 3, 2, 1, 0]
            idx = torch.tensor(det_to_seg, device=v13.device, dtype=torch.long)
            v13 = v13.index_select(dim=1, index=idx)
            return v13
        elif vessel.shape[1] == 13:
            return vessel
        else:
            raise ValueError(f"Invalid vessel_seg channel count: {vessel.shape}")

    @staticmethod
    def _resize_to_feat(x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Resize spatial dims to match feature map (trilinear).
        AMP-friendly: cast to feat dtype before interpolation.
        """
        if x.shape[-3:] != feat.shape[-3:]:
            x = F.interpolate(
                x.to(dtype=feat.dtype), size=feat.shape[-3:], mode="trilinear", align_corners=False
            )
        return x

    def _get_global_pool_feature(self, out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Get feature tensor used for global pooling"""
        feat_global = out.get("feat_global")
        if feat_global is not None:
            return feat_global
        if getattr(self.model, "use_encoder_global_pooling", False):
            if "enc_feat" not in out:
                raise KeyError("enc_feat not found in output")
            return out["enc_feat"]
        return None

    # ===== Pretrained weights loading =====

    def _load_pretrained_net_weights(
        self, ckpt_path: str, state_key: Optional[str] = None, strict: bool = False
    ) -> None:
        """Load net weights (pretrained on sphere-mask only) into self.model.net.

        - Supports Lightning .ckpt or plain state_dict
        - Auto-resolves key prefixes in state_dict ('net.', 'model.net.', 'model.module.net.', ...)
        """
        import torch

        # Load checkpoint
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint format: dict-like state_dict not found")

        target_state = self.model.net.state_dict()

        # Prefer explicit key if provided
        if state_key is not None and state_key in state:
            cand = state[state_key]
            if isinstance(cand, dict):
                state = cand

        # Find the prefix with the most matching keys
        prefixes = ["", "net.", "model.net.", "model.module.net.", "net._orig_mod."]
        best_prefix = ""
        best_match = -1
        for p in prefixes:
            cnt = 0
            for k in state.keys():
                sk = k[len(p) :] if p and k.startswith(p) else (k if p == "" else None)
                if sk is not None and sk in target_state:
                    cnt += 1
            if cnt > best_match:
                best_match = cnt
                best_prefix = p

        if best_match <= 0:
            # Try loading as-is (depends on 'strict')
            filtered = state
        else:
            filtered = {
                (k[len(best_prefix) :] if best_prefix and k.startswith(best_prefix) else k): v
                for k, v in state.items()
            }

        missing, unexpected = self.model.net.load_state_dict(filtered, strict=bool(strict))
        print(
            "[INFO] Pretrained net weights loaded:",
            f"missing={len(missing)}",
            f"unexpected={len(unexpected)}",
            f"strict={bool(strict)}",
        )

    @staticmethod
    def _ap_logit_from_loc_logits(logits_loc: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Build AP logit via soft-OR from location logits (with stabilization)"""
        prob = torch.sigmoid(logits_loc).clamp(min=eps, max=1 - eps)  # (B,13)
        ap_prob = 1.0 - torch.prod(1.0 - prob, dim=1)  # (B,)
        ap_prob = ap_prob.clamp(min=eps, max=1 - eps)
        ap_logit = torch.log(ap_prob) - torch.log1p(-ap_prob)
        return ap_logit

    def embed_metadata(
        self,
        metadata_numeric: Optional[torch.Tensor] = None,
        metadata_numeric_missing: Optional[torch.Tensor] = None,
        metadata_categorical: Optional[torch.Tensor] = None,
        *,
        device_hint: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Obtain metadata embedding via runtime module"""

        return self.model.embed_metadata(
            metadata_numeric=metadata_numeric,
            metadata_numeric_missing=metadata_numeric_missing,
            metadata_categorical=metadata_categorical,
            device_hint=device_hint,
        )

    def _classify_with_masks_subset(
        self,
        runtime_module: VesselROIRuntimeModule,
        feat: torch.Tensor,
        vessel: torch.Tensor,
        extra_vessel: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
        metadata_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for all 13 classes from region masks"""
        vessel13 = self._ensure_mask_channels(vessel)
        vessel13 = self._resize_to_feat(vessel13, feat)
        mask_primary, global_part = runtime_module.rmp_loc.forward_split(
            feat, vessel13, global_feat=global_feat
        )
        pooled_list: List[torch.Tensor] = [mask_primary.to(feat.dtype)]

        need_extra_n = int(getattr(runtime_module, "num_extra_mask_branches", 0))
        if need_extra_n > 0:
            if extra_vessel is None:
                raise ValueError("extra_vessel is required when using extra segmentations")
            # Split into 13-channel blocks according to channel count
            if extra_vessel.dim() != 5 or extra_vessel.shape[1] % 13 != 0:
                raise ValueError(
                    f"extra_vessel channels are not a multiple of 13: {tuple(extra_vessel.shape)}"
                )
            groups = extra_vessel.shape[1] // 13
            use_groups = min(need_extra_n, groups)
            for g in range(use_groups):
                sl = slice(g * 13, (g + 1) * 13)
                extra13 = extra_vessel[:, sl, ...]
                extra13 = self._ensure_mask_channels(extra13)
                extra13 = self._resize_to_feat(extra13, feat)
                mask_extra, _ = runtime_module.rmp_loc.forward_split(feat, extra13, global_feat=global_feat)
                pooled_list.append(mask_extra.to(feat.dtype))

        if global_part is not None:
            pooled_list.append(global_part.to(feat.dtype))

        pooled = pooled_list[0] if len(pooled_list) == 1 else torch.cat(pooled_list, dim=-1)

        if metadata_embed is not None:
            meta = metadata_embed.to(device=pooled.device, dtype=pooled.dtype)
            meta = meta.unsqueeze(1).expand(-1, pooled.shape[1], -1)
            pooled = torch.cat([pooled, meta], dim=-1)
        batch_size, num_classes, channels = pooled.shape
        logits_all = runtime_module.cls_head(pooled.view(batch_size * num_classes, channels)).view(
            batch_size, num_classes
        )
        return logits_all

    def _classify_ap_from_union(
        self,
        runtime_module: VesselROIRuntimeModule,
        feat: torch.Tensor,
        union_mask: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
        metadata_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Directly infer AP logit from union vessel mask (1,d,h,w).
        Returns: raw logit of shape (B,)
        """
        # Reshape to (B,1,d,h,w)
        if union_mask.dim() != 5:
            raise ValueError("vessel_union must be (B,1,D,H,W)")
        um = self._resize_to_feat(union_mask, feat)
        pooled = runtime_module.rmp_ap(feat, um, global_feat=global_feat).to(feat.dtype)  # (B,1,C_pool)
        if metadata_embed is not None:
            meta = metadata_embed.to(device=pooled.device, dtype=pooled.dtype)
            meta = meta.unsqueeze(1)
            pooled = torch.cat([pooled, meta], dim=-1)
        B, K, C = pooled.shape
        logit = runtime_module.ap_head(pooled.view(B * K, C)).view(B)  # (B,)
        return logit

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        labels = batch["labels"]  # (B,14)
        vessel_union = batch["vessel_union"]

        loss: Dict[str, torch.Tensor] = {}

        # Classification (locations + AP)
        logits_loc = out["logits_loc"]  # (B,13)
        num_loc = int(self.hparams.num_location_classes)
        labels_loc = labels[:, :num_loc]
        target_loc = labels_loc.to(logits_loc.dtype)
        loss["loss_loc"] = self.crit_loc(logits_loc, target_loc)

        # AP is directly inferred from union vessel mask (always required)
        logit_ap = out["logit_ap"]
        target_ap = labels[:, 13].to(logit_ap.dtype)
        loss["loss_ap"] = self.crit_ap(logit_ap, target_ap)

        # Sphere (optional)
        if "sphere_mask" in batch and out.get("logits_sphere") is not None:
            logits_sphere = out["logits_sphere"]  # (B,1,d,h,w)
            tgt = batch["sphere_mask"]  # (B,1,D,H,W) uint8
            # Match spatial size (nearest)
            if tgt.shape[-3:] != logits_sphere.shape[-3:]:
                tgt_rs = F.interpolate(tgt.float(), size=logits_sphere.shape[-3:], mode="nearest")
            else:
                tgt_rs = tgt.float()
            bce = self.crit_sphere(logits_sphere, tgt_rs)
            # Compute Dice only for samples with GT
            has_gt = (tgt_rs.sum(dim=(2, 3, 4)) > 0).squeeze(1)
            if has_gt.any():
                dice = self.dice_sphere(logits_sphere[has_gt], tgt_rs[has_gt])
            else:
                dice = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            loss["loss_sphere_bce"] = bce
            loss["loss_sphere_dice"] = dice

        # Aggregate losses
        total = self.hparams.w_loc * loss["loss_loc"] + self.hparams.w_ap * loss["loss_ap"]
        if "loss_sphere_bce" in loss:
            total = total + self.hparams.w_sphere * (loss["loss_sphere_bce"] + loss["loss_sphere_dice"])
        loss["loss_total"] = total
        # Return additional logits
        loss["logits_loc"] = logits_loc.detach()
        loss["logit_ap"] = logit_ap.detach()
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # GPU augmentations + sphere re-generation
        batch = self._gpu_augment_batch(
            batch,
            training=True,
        )

        out = self.forward(
            batch["image"],
            vessel_seg=batch["vessel_seg"],
            vessel_union=batch["vessel_union"],
            extra_vessel_seg=batch.get("extra_vessel_seg"),
            metadata_numeric=batch.get("metadata_numeric"),
            metadata_numeric_missing=batch.get("metadata_numeric_missing"),
            metadata_categorical=batch.get("metadata_categorical"),
            use_ema=False,
        )  # dict(feat, logits_sphere)
        loss = self._compute_losses(batch, out)
        # Visualization (first step of each epoch only)
        if not getattr(self, "_did_train_viz_this_epoch", False) and batch_idx == 0:
            try:
                self._visualize_batch_first(batch, out, split="train", batch_idx=batch_idx)
            finally:
                self._did_train_viz_this_epoch = True
        for k, v in loss.items():
            if k.startswith("loss_"):
                self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "loss_total"))
        return loss["loss_total"]

    def on_train_batch_end(self, outputs, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        if bool(getattr(self.hparams, "ema", False)) and self.model.has_ema():
            # Update EMA only on batches where optimizer.step() runs
            step_count = self._get_completed_optimizer_steps()
            if step_count is None:
                step_count = int(self.global_step)
            if step_count <= self._ema_last_optimizer_step:
                return
            self._ema_last_optimizer_step = step_count
            self.model.update_ema(step=self.global_step)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # Validation: no augmentations, sphere only re-generated
        batch = self._gpu_augment_batch(
            batch,
            training=False,
        )

        out = self.forward(
            batch["image"],
            vessel_seg=batch["vessel_seg"],
            vessel_union=batch["vessel_union"],
            extra_vessel_seg=batch.get("extra_vessel_seg"),
            metadata_numeric=batch.get("metadata_numeric"),
            metadata_numeric_missing=batch.get("metadata_numeric_missing"),
            metadata_categorical=batch.get("metadata_categorical"),
            use_ema=self._use_ema_for_eval(),
        )  # dict
        loss = self._compute_losses(batch, out)
        # Visualization (first step of each epoch only)
        if not getattr(self, "_did_val_viz_this_epoch", False) and batch_idx == 0:
            try:
                self._visualize_batch_first(batch, out, split="val", batch_idx=batch_idx)
            finally:
                self._did_val_viz_this_epoch = True
        for k, v in loss.items():
            if k.startswith("loss_"):
                self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "loss_total"))

        # Accumulate logits for competition metric (13 locations + 1 AP = 14)
        logits_loc = loss["logits_loc"].cpu()
        logit_ap = loss["logit_ap"].unsqueeze(1).cpu()
        logits_all = torch.cat([logits_loc, logit_ap], dim=1)
        self._val_logits.append(logits_all)
        self._val_labels.append(batch["labels"].cpu())

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # Test: no augmentations, sphere only re-generated
        batch = self._gpu_augment_batch(
            batch,
            training=False,
        )

        out = self.forward(
            batch["image"],
            vessel_seg=batch["vessel_seg"],
            vessel_union=batch["vessel_union"],
            extra_vessel_seg=batch.get("extra_vessel_seg"),
            metadata_numeric=batch.get("metadata_numeric"),
            metadata_numeric_missing=batch.get("metadata_numeric_missing"),
            metadata_categorical=batch.get("metadata_categorical"),
            use_ema=self._use_ema_for_eval(),
        )  # dict
        loss = self._compute_losses(batch, out)
        for k, v in loss.items():
            if k.startswith("loss_"):
                self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "loss_total"))

        logits_loc = loss["logits_loc"].cpu()
        logit_ap = loss["logit_ap"].unsqueeze(1).cpu()
        logits_all = torch.cat([logits_loc, logit_ap], dim=1)
        self._test_logits.append(logits_all)
        self._test_labels.append(batch["labels"].cpu())

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Return 0.5 when class has only positives or only negatives (avoid exceptions)"""
        try:
            if np.unique(y_true).size < 2:
                return 0.5
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return 0.5

    def _compute_final_competition_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Final = 0.5 * ( AUC_AP + mean(13 location AUCs) )"""
        assert y_true.shape[1] >= 14, "Labels expected to have 14 dims (13 locations + AP)"
        loc_idx = list(range(13))
        ap_idx = 13

        auc_loc = [self._safe_roc_auc(y_true[:, i], y_prob[:, i]) for i in loc_idx]
        auc_ap = self._safe_roc_auc(y_true[:, ap_idx], y_prob[:, ap_idx])
        mean_loc = float(np.mean(auc_loc)) if len(auc_loc) > 0 else 0.5
        final_score = 0.5 * (auc_ap + mean_loc)

        out: Dict[str, float] = {
            "final_score": final_score,
            "auc_ap": auc_ap,
            "auc_loc_mean": mean_loc,
        }
        for i, a in enumerate(auc_loc):
            out[f"auc_loc_{i}"] = a
        return out

    def on_validation_epoch_end(self) -> None:
        # Reset visualization flags for next epoch
        self._did_val_viz_this_epoch = False
        if len(self._val_logits) == 0:
            return
        logits = torch.cat(self._val_logits, dim=0)
        labels = torch.cat(self._val_labels, dim=0)
        probs = torch.sigmoid(logits).numpy()
        y_true = labels.numpy()
        scores = self._compute_final_competition_score(y_true, probs)
        self.log("val/final_score", scores["final_score"], prog_bar=True)
        self.log("val/auc_ap", scores["auc_ap"])
        self.log("val/auc_loc_mean", scores["auc_loc_mean"])
        for i in range(13):
            self.log(f"val/auc_loc_{i}", scores[f"auc_loc_{i}"])
        self._val_logits.clear()
        self._val_labels.clear()

    def on_test_epoch_end(self) -> None:
        if len(self._test_logits) == 0:
            return
        logits = torch.cat(self._test_logits, dim=0)
        labels = torch.cat(self._test_labels, dim=0)
        probs = torch.sigmoid(logits).numpy()
        y_true = labels.numpy()
        scores = self._compute_final_competition_score(y_true, probs)
        self.log("test/final_score", scores["final_score"], prog_bar=True)
        self.log("test/auc_ap", scores["auc_ap"])
        self.log("test/auc_loc_mean", scores["auc_loc_mean"])
        for i in range(13):
            self.log(f"test/auc_loc_{i}", scores[f"auc_loc_{i}"])
        self._test_logits.clear()
        self._test_labels.clear()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)
        # Initialize visualization flags at epoch start
        self._did_train_viz_this_epoch = False
        self._did_val_viz_this_epoch = False

    def on_train_epoch_start(self) -> None:
        """Reset visualization flags at start of train epoch"""
        self._did_train_viz_this_epoch = False

    def on_train_start(self) -> None:
        super().on_train_start()
        if bool(getattr(self.hparams, "ema", False)) and self.model.has_ema():
            # On resume, store optimizer.step() count as baseline
            step_count = self._get_completed_optimizer_steps()
            if step_count is None:
                step_count = int(self.global_step)
            self._ema_last_optimizer_step = step_count

    def on_validation_epoch_start(self) -> None:
        """Reset visualization flags at start of validation epoch"""
        self._did_val_viz_this_epoch = False

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See Lightning docs for details on configuring optimizers.

        :return: A dict containing the configured optimizers and LR schedulers used for training.
        """
        if self.hparams.optimize_config.mode == "normal":
            optimizer = self.hparams.optimizer(params=self.parameters())

        elif self.hparams.optimize_config.mode == "target_decay":
            target_layer_list = []
            for target_name in self.hparams.optimize_config.target_names:
                target_layer_list += self.get_target_param(target_name)

            target_params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] in target_layer_list, self.named_parameters())),
                )
            )
            base_params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] not in target_layer_list, self.named_parameters())),
                )
            )

            optimizer = self.hparams.optimizer(
                params=[
                    {
                        "params": target_params,
                        "lr": self.hparams.optimize_config.lr_base
                        * self.hparams.optimize_config.lr_decay_coef,
                    },
                    {"params": base_params},
                ],
            )

        else:
            assert False, "optimize_mode"

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_target_param(self, target_name):
        layer_list = []
        for name, param in self.named_parameters():
            if target_name in name:
                layer_list.append(name)

        assert len(layer_list) > 0

        return layer_list


class AneurysmVesselSegROILitModuleTransformer(AneurysmVesselSegROILitModule):
    """LightningModule that tokenizes region features and models relations via Transformer"""

    def __init__(
        self,
        *args: Any,
        transformer_embed_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ffn_dim: Optional[int] = None,
        transformer_dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        embed_dim = int(transformer_embed_dim)
        heads = int(transformer_heads)
        layers = int(transformer_layers)
        ffn_dim = int(transformer_ffn_dim) if transformer_ffn_dim is not None else embed_dim * 2
        pool_dim = int(self.model.pooled_feature_channels())
        num_loc = int(self.hparams.num_location_classes)

        self.loc_proj = nn.Linear(pool_dim, embed_dim)
        self.loc_dropout = nn.Dropout(transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=transformer_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.loc_transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.loc_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        self.loc_token_embed = nn.Parameter(torch.zeros(num_loc, embed_dim))
        nn.init.trunc_normal_(self.loc_token_embed, std=0.02)

        # Log Transformer-related hyperparameters as well
        self.transformer_embed_dim = embed_dim
        self.transformer_heads = heads
        self.transformer_layers = layers
        self.transformer_ffn_dim = ffn_dim
        self.transformer_dropout = float(transformer_dropout)

    def _encode_location_tokens(
        self,
        runtime_module: VesselROIRuntimeModule,
        feat: torch.Tensor,
        vessel: torch.Tensor,
        extra_vessel: Optional[torch.Tensor],
        global_feat: Optional[torch.Tensor],
        metadata_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert features from region masks into a token sequence"""
        vessel13 = self._ensure_mask_channels(vessel)
        vessel13 = self._resize_to_feat(vessel13, feat)
        mask_primary, global_part = runtime_module.rmp_loc.forward_split(
            feat, vessel13, global_feat=global_feat
        )
        pooled_list: List[torch.Tensor] = [mask_primary.to(feat.dtype)]

        need_extra_n = int(getattr(runtime_module, "num_extra_mask_branches", 0))
        if need_extra_n > 0:
            if extra_vessel is None:
                raise ValueError("extra_vessel is required when using extra segmentations")
            if extra_vessel.dim() != 5 or extra_vessel.shape[1] % 13 != 0:
                raise ValueError(
                    f"extra_vessel channels are not a multiple of 13: {tuple(extra_vessel.shape)}"
                )
            groups = extra_vessel.shape[1] // 13
            use_groups = min(need_extra_n, groups)
            for g in range(use_groups):
                sl = slice(g * 13, (g + 1) * 13)
                extra13 = extra_vessel[:, sl, ...]
                extra13 = self._ensure_mask_channels(extra13)
                extra13 = self._resize_to_feat(extra13, feat)
                mask_extra, _ = runtime_module.rmp_loc.forward_split(feat, extra13, global_feat=global_feat)
                pooled_list.append(mask_extra.to(feat.dtype))

        if global_part is not None:
            pooled_list.append(global_part.to(feat.dtype))

        pooled = pooled_list[0] if len(pooled_list) == 1 else torch.cat(pooled_list, dim=-1)
        if metadata_embed is not None:
            meta = metadata_embed.to(device=pooled.device, dtype=pooled.dtype)
            meta = meta.unsqueeze(1).expand(-1, pooled.shape[1], -1)
            pooled = torch.cat([pooled, meta], dim=-1)
        tokens = self.loc_proj(pooled)
        tokens = tokens + self.loc_token_embed.unsqueeze(0)
        tokens = self.loc_dropout(tokens)
        return self.loc_transformer(tokens)

    def _classify_with_masks_subset(
        self,
        runtime_module: VesselROIRuntimeModule,
        feat: torch.Tensor,
        vessel: torch.Tensor,
        extra_vessel: Optional[torch.Tensor] = None,
        global_feat: Optional[torch.Tensor] = None,
        metadata_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get logits for all 13 classes via Transformer"""
        tokens = self._encode_location_tokens(
            runtime_module,
            feat,
            vessel,
            extra_vessel,
            global_feat,
            metadata_embed=metadata_embed,
        )
        logits_all = self.loc_head(tokens).squeeze(-1)
        return logits_all


class LocationWiseMoETransformerLayer(nn.Module):
    """Transformer encoder layer with per-location expert MLPs"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_loc: int,
        norm_first: bool = True,
        expert_assignments: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_first = bool(norm_first)
        self.num_loc = int(num_loc)
        if expert_assignments is None:
            assignments = tuple(range(self.num_loc))
        else:
            assignments = tuple(int(x) for x in expert_assignments)
            if len(assignments) != self.num_loc:
                raise ValueError("Length of expert_assignments does not match number of locations")
        self.expert_assignments = assignments
        num_experts = max(self.expert_assignments) + 1 if self.expert_assignments else self.num_loc
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            src = src + self._self_attention_block(self.norm1(src))
            src = src + self._ffn_block(self.norm2(src))
            return src
        src = self.norm1(src + self._self_attention_block(src))
        src = self.norm2(src + self._ffn_block(src))
        return src

    def _self_attention_block(self, src: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(src, src, src, need_weights=False)
        return self.dropout_attn(attn_out)

    def _ffn_block(self, src: torch.Tensor) -> torch.Tensor:
        if src.shape[1] != self.num_loc:
            raise ValueError("Number of tokens does not match number of locations")
        chunks = []
        for idx in range(self.num_loc):
            expert_idx = self.expert_assignments[idx]
            expert = self.experts[expert_idx]
            token = src[:, idx, :]
            token = expert(token)
            chunks.append(token.unsqueeze(1))
        out = torch.cat(chunks, dim=1)
        return self.dropout_ffn(out)


class LocationWiseMoETransformer(nn.Module):
    """Encoder that transforms location tokens via MoE structure"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_layers: int,
        num_loc: int,
        norm_first: bool = True,
        final_norm: bool = False,
        expert_assignments: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if expert_assignments is not None and len(expert_assignments) != int(num_loc):
            raise ValueError("Length of expert_assignments does not match number of locations")
        self.expert_assignments = (
            tuple(int(x) for x in expert_assignments) if expert_assignments is not None else None
        )
        self.layers = nn.ModuleList(
            [
                LocationWiseMoETransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    num_loc=num_loc,
                    norm_first=norm_first,
                    expert_assignments=self.expert_assignments,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim) if final_norm else None

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out)
        if self.norm is not None:
            out = self.norm(out)
        return out


class AneurysmVesselSegROILitModuleTransformerMoE(AneurysmVesselSegROILitModuleTransformer):
    """LightningModule replacing Transformer FFN with location-wise experts"""

    def __init__(self, *args: Any, share_left_right_expert: bool = False, **kwargs: Any) -> None:
        self.share_left_right_expert = bool(share_left_right_expert)
        super().__init__(*args, **kwargs)
        num_loc = int(self.hparams.num_location_classes)
        expert_assignments = self._build_expert_assignments(num_loc=num_loc)
        self.loc_transformer = LocationWiseMoETransformer(
            embed_dim=self.transformer_embed_dim,
            num_heads=self.transformer_heads,
            ffn_dim=self.transformer_ffn_dim,
            dropout=self.transformer_dropout,
            num_layers=self.transformer_layers,
            num_loc=num_loc,
            norm_first=True,
            final_norm=False,
            expert_assignments=expert_assignments,
        )

    def _build_expert_assignments(self, num_loc: int) -> Optional[Sequence[int]]:
        if not self.share_left_right_expert:
            return None
        class_names = ANEURYSM_CLASSES[:num_loc]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        assignments: List[int] = list(range(num_loc))
        # Assign paired left/right classes to share the right-side expert
        for idx, name in enumerate(class_names):
            if not name.startswith("Left "):
                continue
            right_name = "Right " + name[len("Left ") :]
            if right_name not in class_to_idx:
                continue
            right_idx = class_to_idx[right_name]
            shared_idx = min(idx, right_idx)
            assignments[idx] = shared_idx
            assignments[right_idx] = shared_idx
        # Remap sparse indices to 0-based contiguous ids
        remap: Dict[int, int] = {}
        normalized: List[int] = []
        for val in assignments:
            if val not in remap:
                remap[val] = len(remap)
            normalized.append(remap[val])
        return tuple(normalized)


class AneurysmVesselSegROILitModuleTransformerVesselSeg(AneurysmVesselSegROILitModuleTransformer):
    """Transformer variant that adds vessel segmentation loss"""

    def __init__(
        self,
        *args: Any,
        w_vessel_seg: float = 1.0,
        dice_ce_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._dice_ce_kwargs = dict(dice_ce_kwargs) if dice_ce_kwargs is not None else {}
        self.w_vessel_seg = float(w_vessel_seg)
        super().__init__(*args, **kwargs)
        default_kwargs = {
            "include_background": True,
            "to_onehot_y": True,
            "softmax": True,
            "reduction": "mean",
        }
        default_kwargs.update(self._dice_ce_kwargs)
        self.crit_vessel_seg = DiceCELoss(**default_kwargs)

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss = super()._compute_losses(batch, out)
        logits_vessel_seg = out.get("logits_vessel_seg")
        vessel_label = batch.get("vessel_label")
        if logits_vessel_seg is None or vessel_label is None:
            return loss

        target = vessel_label.to(device=logits_vessel_seg.device)
        if target.dim() == 4:
            target = target.unsqueeze(1)
        if target.shape[1] != 1:
            target = target[:, :1]
        if target.shape[-3:] != logits_vessel_seg.shape[-3:]:
            target = F.interpolate(
                target.float(),
                size=logits_vessel_seg.shape[-3:],
                mode="nearest",
            ).long()
        target = target.squeeze(1)

        loss_vessel_seg = self.crit_vessel_seg(logits_vessel_seg, target.unsqueeze(1))
        loss["loss_vessel_seg"] = loss_vessel_seg
        loss["loss_total"] = loss["loss_total"] + self.w_vessel_seg * loss_vessel_seg
        return loss
