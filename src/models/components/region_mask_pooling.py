from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionMaskedPooling3D(nn.Module):
    """
    Pool features with region masks and concatenate with global pooled features.

    Inputs:
      - feat:  (B, C, D, H, W)
      - masks: (B, K, D, H, W)  K channels of region masks (probability or binary)

    Output:
      - (B, K, C_total)  = concat(mask_branch, global_branch)
        Note: C_total varies with LayerNorm/linear projection settings.

    Behavior:
      - Clamp mask to non-negative and use as weights (binary masks expected)
      - Mask branch supports mean/gem/max; for max, multiply features by mask as-is
      - Global branch also supports mean/gem/max
      - GEM clips inputs to a lower bound, averages powered features, then takes the p-th root
    """

    def __init__(
        self,
        eps: float = 1e-6,
        mask_pool_modes: Sequence[str] | str | None = "mean",
        global_pool_modes: Sequence[str] | str | None = "mean",
        gem_p: float = 3.0,
        gem_eps: float = 1e-6,
        use_encoder_global_feat: bool = False,
        # Optional: pooling also over dilated masks
        add_dilated_mask: bool = False,
        # Optional: dilation kernel size (D,H,W); default expands only XY for anisotropy
        dilate_kernel: int | Sequence[int] = (1, 3, 3),
        # Optional: number of dilation iterations (>1 expands stepwise)
        dilate_iters: int = 1,
        # Optional: per-branch LayerNorm and linear projection settings
        mask_feat_channels: Optional[int] = None,
        global_feat_channels: Optional[int] = None,
        branch_norm: bool = True,
        proj_mask_dim: Optional[int] = None,
        proj_global_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        valid_modes = {"mean", "gem", "max"}
        self.mask_pool_modes = self._normalize_modes(mask_pool_modes, valid_modes, "mask_pool_modes")
        self.global_pool_modes = self._normalize_modes(global_pool_modes, valid_modes, "global_pool_modes")
        self.gem_p = float(gem_p)
        self.gem_eps = float(gem_eps)
        self.use_encoder_global_feat = bool(use_encoder_global_feat)
        # Dilation-related
        self.add_dilated_mask = bool(add_dilated_mask)
        self.dilate_kernel = self._to_ks_tuple(dilate_kernel)
        self.dilate_iters = int(dilate_iters)
        # Multiplier for output-dim estimation (base mask + dilated variants)
        self.mask_pool_variant_multiplier: int = 1 + (int(self.dilate_iters) if self.add_dilated_mask else 0)
        # Settings for per-branch normalization/projection (precompute output dims)
        self._branch_norm_enabled: bool = bool(branch_norm)
        self._mask_feat_channels: Optional[int] = int(mask_feat_channels) if mask_feat_channels is not None else None
        self._global_feat_channels: Optional[int] = (
            int(global_feat_channels) if global_feat_channels is not None else None
        )
        self._mask_in_dim: Optional[int] = self._compute_mask_in_dim()
        self._global_in_dim: Optional[int] = self._compute_global_in_dim()
        self._validate_branch_configs(proj_mask_dim, proj_global_dim)
        self.mask_norm = self._build_norm(self._mask_in_dim)
        self.global_norm = self._build_norm(self._global_in_dim)
        self.mask_proj = self._build_proj(self._mask_in_dim, proj_mask_dim)
        self.global_proj = self._build_proj(self._global_in_dim, proj_global_dim)
        self._mask_out_dim: Optional[int] = self._resolve_out_dim(self._mask_in_dim, proj_mask_dim, bool(self.mask_pool_modes))
        self._global_out_dim: Optional[int] = self._resolve_out_dim(
            self._global_in_dim, proj_global_dim, bool(self.global_pool_modes)
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def forward_split(
        self,
        feat: torch.Tensor,
        masks: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass returning mask and global features separately

        Args:
            feat: (B, C, D, H, W)
            masks: (B, K, D, H, W)
            global_feat: (B, Cg, Dg, Hg, Wg) or None (encoder features for global pooling)

        Returns:
            mask_cat:   (B, K, C_mask)
            global_cat: (B, K, C_global) or None (when global_pool_modes is disabled)
        """
        assert feat.dim() == 5, "feat must be (B,C,D,H,W)"
        assert masks.dim() == 5, "masks must be (B,K,D,H,W)"
        B, C, D, H, W = feat.shape
        Bm, K, Dm, Hm, Wm = masks.shape
        assert B == Bm, "Batch dimension mismatch"

        feat = feat.float()
        masks = masks.float()
        if global_feat is not None:
            global_feat = global_feat.float()
        mask_cat: Optional[torch.Tensor] = None
        global_cat: Optional[torch.Tensor] = None

        if self.mask_pool_modes:
            mask_src = masks
            if (D, H, W) != (Dm, Hm, Wm):
                # If shapes differ, resize mask to feature size (probability map -> trilinear)
                mask_src = F.interpolate(masks.float(), size=(D, H, W), mode="trilinear", align_corners=False)

            # Clamp to non-negative (cast to match feat dtype)
            w = mask_src.clamp_min(0).to(dtype=feat.dtype)
            # Pooling with original mask
            mask_feature_chunks = [self._masked_pool_single(mode, feat, w) for mode in self.mask_pool_modes]
            mask_features: list[torch.Tensor] = [self._concat_features(mask_feature_chunks)]

            # Optional pooling with dilated masks
            if self.add_dilated_mask and self.dilate_iters > 0:
                for wd in self._dilate_mask_levels(w, iters=self.dilate_iters):
                    mask_chunks_d = [self._masked_pool_single(mode, feat, wd) for mode in self.mask_pool_modes]
                    mask_features.append(self._concat_features(mask_chunks_d))

            mask_cat = self._concat_features(mask_features)
            mask_cat = self.mask_proj(self.mask_norm(mask_cat))
            if self._mask_out_dim is not None and mask_cat.shape[-1] != int(self._mask_out_dim):
                raise RuntimeError(
                    f"Unexpected mask-branch output dim: {mask_cat.shape[-1]} != {self._mask_out_dim}"
                )

        if self.use_encoder_global_feat:
            if global_feat is None:
                raise ValueError("global_feat must be provided when use_encoder_global_feat=True")
            global_src = global_feat
        else:
            global_src = feat

        if global_src.dim() != 5:
            raise ValueError("global_feat must be a 5D tensor (B,C,D,H,W)")

        if self.global_pool_modes:
            global_feats = [self._global_pool_single(mode, global_src) for mode in self.global_pool_modes]
            global_rep_list = [g.unsqueeze(1).expand(-1, K, -1) for g in global_feats]
            global_cat = self._concat_features(global_rep_list)
            global_cat = self.global_proj(self.global_norm(global_cat))
            if self._global_out_dim is not None and global_cat.shape[-1] != int(self._global_out_dim):
                raise RuntimeError(
                    f"Unexpected global-branch output dim: {global_cat.shape[-1]} != {self._global_out_dim}"
                )
        return mask_cat if mask_cat is not None else torch.empty(B, K, 0, device=feat.device), global_cat

    def forward(
        self,
        feat: torch.Tensor,
        masks: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility API: internally calls forward_split and concatenates (mask, global)

        Returns:
            pooled: (B, K, C_total)
        """
        mask_cat, global_cat = self.forward_split(feat, masks, global_feat)
        if global_cat is None:
            if mask_cat.numel() == 0:
                raise ValueError("Both mask_pool_modes and global_pool_modes are disabled")
            return mask_cat
        if mask_cat.numel() == 0:
            return global_cat
        return torch.cat([mask_cat, global_cat], dim=-1)

    def mask_output_dim(self) -> int:
        """Return final output dim of mask branch"""
        if self._mask_out_dim is None:
            raise ValueError("mask_feat_channels must be specified to determine mask output dim")
        return int(self._mask_out_dim)

    def global_output_dim(self) -> int:
        """Return final output dim of global branch"""
        if self._global_out_dim is None:
            raise ValueError("global_feat_channels must be specified to determine global output dim")
        return int(self._global_out_dim)

    def output_dim(self) -> int:
        """Return concatenated dim after both branches"""
        return int(self.mask_output_dim() + self.global_output_dim())

    @staticmethod
    def _concat_features(features: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate one/more tensors along the last dim"""
        if len(features) == 0:
            raise ValueError("No valid pooled results found")
        if len(features) == 1:
            return features[0]
        return torch.cat(features, dim=-1)

    def _compute_mask_in_dim(self) -> Optional[int]:
        """Compute input dim for mask branch"""
        if len(self.mask_pool_modes) == 0:
            return 0
        if self._mask_feat_channels is None:
            return None
        count = len(self.mask_pool_modes) * int(self.mask_pool_variant_multiplier)
        return int(self._mask_feat_channels * count)

    def _compute_global_in_dim(self) -> Optional[int]:
        """Compute input dim for global branch"""
        if len(self.global_pool_modes) == 0:
            return 0
        if self._global_feat_channels is None:
            return None
        count = len(self.global_pool_modes)
        return int(self._global_feat_channels * count)

    def _validate_branch_configs(
        self,
        proj_mask_dim: Optional[int],
        proj_global_dim: Optional[int],
    ) -> None:
        """Validate that normalization/projection settings are consistent with dims"""
        if self._branch_norm_enabled:
            if self._mask_in_dim is None and len(self.mask_pool_modes) > 0:
                raise ValueError("mask_feat_channels must be set when branch_norm=True")
            if self._global_in_dim is None and len(self.global_pool_modes) > 0:
                raise ValueError("global_feat_channels must be set when branch_norm=True")
        if proj_mask_dim is not None and len(self.mask_pool_modes) > 0 and self._mask_in_dim is None:
            raise ValueError("mask_feat_channels must be set when using proj_mask_dim")
        if proj_global_dim is not None and len(self.global_pool_modes) > 0 and self._global_in_dim is None:
            raise ValueError("global_feat_channels must be set when using proj_global_dim")

    def _build_norm(self, in_dim: Optional[int]) -> nn.Module:
        """Create LayerNorm if needed"""
        if not self._branch_norm_enabled or in_dim in (None, 0):
            return nn.Identity()
        return nn.LayerNorm(int(in_dim))

    @staticmethod
    def _build_proj(in_dim: Optional[int], out_dim: Optional[int]) -> nn.Module:
        """Create linear projection if needed"""
        if out_dim is None or in_dim in (None, 0):
            return nn.Identity()
        return nn.Linear(int(in_dim), int(out_dim))

    @staticmethod
    def _resolve_out_dim(
        in_dim: Optional[int],
        proj_dim: Optional[int],
        has_branch: bool,
    ) -> Optional[int]:
        """Resolve output dim for a branch"""
        if not has_branch:
            return 0
        if in_dim is None:
            return None
        if proj_dim is not None:
            return int(proj_dim)
        return int(in_dim)

    def _masked_pool_single(self, mode: str, feat: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute masked pooling result in the specified mode"""
        feat_expanded = feat.unsqueeze(1)  # (B,1,C,D,H,W)
        if mode == "mean":
            denom = weights.flatten(2).sum(-1).unsqueeze(-1).clamp_min(self.eps)
            num = (weights.unsqueeze(2) * feat_expanded).flatten(3).sum(-1)
            return num / denom

        if mode == "gem":
            denom_raw = weights.flatten(2).sum(-1).unsqueeze(-1)
            # Skip GEM when mask sum is 0 and return zeros
            valid_mask = denom_raw > self.eps
            denom = denom_raw.clamp_min(self.eps)
            feat_clamped = feat_expanded.clamp_min(self.gem_eps)
            powered = torch.pow(feat_clamped, self.gem_p)
            num = (weights.unsqueeze(2) * powered).flatten(3).sum(-1)
            ratio = num / denom
            ratio = torch.where(valid_mask, ratio, torch.ones_like(ratio))
            gem = torch.pow(ratio, 1.0 / self.gem_p)
            return torch.where(valid_mask, gem, torch.zeros_like(gem))

        # mode == "max"
        masked_feat = feat_expanded * weights.unsqueeze(2)
        return masked_feat.flatten(3).max(dim=-1).values

    def _global_pool_single(self, mode: str, feat: torch.Tensor) -> torch.Tensor:
        """Compute global pooling result in the specified mode"""
        if mode == "mean":
            return feat.flatten(2).mean(-1)

        if mode == "gem":
            feat_clamped = feat.clamp_min(self.gem_eps)
            powered = torch.pow(feat_clamped, self.gem_p)
            gem = powered.flatten(2).mean(-1)
            return torch.pow(gem, 1.0 / self.gem_p)

        # mode == "max"
        return feat.flatten(2).max(dim=-1).values

    @staticmethod
    def _normalize_modes(
        modes: Sequence[str] | str | None,
        valid: set[str],
        name: str,
    ) -> tuple[str, ...]:
        """Normalize mode specification into a tuple and validate"""
        if modes is None:
            return ()
        if isinstance(modes, str):
            modes_tuple = (modes,)
        else:
            modes_tuple = tuple(modes)
        if len(modes_tuple) == 0:
            return ()
        invalid = [m for m in modes_tuple if m not in valid]
        if invalid:
            raise ValueError(f"Invalid modes in {name}: {invalid}")
        return modes_tuple

    @staticmethod
    def _to_ks_tuple(ks: int | Sequence[int]) -> tuple[int, int, int]:
        """Normalize kernel size to a (D,H,W) tuple. Beware 0/1 may disable dilation.
        Default is (Z=1, Y=3, X=3) to expand in-plane only.
        """
        if isinstance(ks, int):
            k = (ks, ks, ks)
        else:
            klist = list(ks)
            if len(klist) != 3:
                raise ValueError("dilate_kernel must be int or a length-3 sequence")
            k = (int(klist[0]), int(klist[1]), int(klist[2]))
        # At least 1
        k = tuple(max(1, int(v)) for v in k)
        return (k[0], k[1], k[2])

    @torch.no_grad()
    def _dilate_mask_levels(self, w: torch.Tensor, iters: int) -> list[torch.Tensor]:
        """Return a list of dilated masks via 3D max-pooling.
        Example: iters=3 -> [dilate1, dilate2, dilate3]
        Input/Output: (B,K,D,H,W) float
        """
        kD, kH, kW = self.dilate_kernel
        pad = (kD // 2, kH // 2, kW // 2)
        x = w
        outs: list[torch.Tensor] = []
        for _ in range(int(iters)):
            x = F.max_pool3d(x, kernel_size=(kD, kH, kW), stride=1, padding=pad)
            outs.append(x)
        return outs
