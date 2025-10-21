from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.aneurysm_vessel_seg_roi_module import (
    AneurysmVesselSegROILitModuleTransformer,
)


class AneurysmVesselSegROILitModuleTransformerHeatmap(AneurysmVesselSegROILitModuleTransformer):
    """Variant that uses Gaussian heatmaps as auxiliary targets instead of sphere masks"""

    def __init__(
        self,
        *args: Any,
        heatmap_sigma: float = 2.5,
        heatmap_vis_threshold: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {
                "heatmap_sigma": float(heatmap_sigma),
                "heatmap_vis_threshold": float(heatmap_vis_threshold),
            },
            logger=False,
        )

    def _rasterize_gaussians_from_points(
        self,
        pts: torch.Tensor,
        valid: torch.Tensor,
        out_shape: torch.Size,
        sigma: float,
    ) -> torch.Tensor:
        """Generate Gaussian heatmaps from annotation points"""
        B = int(pts.shape[0]) if pts.dim() >= 2 else 0
        _, _, D, H, W = out_shape
        device = pts.device
        dtype = pts.dtype

        if pts.numel() == 0 or sigma <= 0:
            return torch.zeros((B, 1, D, H, W), device=device, dtype=torch.float32)

        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
            B = int(pts.shape[0])
        if valid.dim() == 1:
            valid = valid.unsqueeze(0).expand(B, -1)
        if valid.dtype != dtype:
            valid = valid.to(device=device, dtype=dtype)
        v = (valid > 0).to(dtype=dtype).unsqueeze(-1)

        z = torch.arange(D, device=device, dtype=dtype).view(1, 1, D, 1, 1)
        y = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
        x = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, W)

        pz = pts[..., 0:1] * v + (1.0 - v) * (-1e6)
        py = pts[..., 1:2] * v + (1.0 - v) * (-1e6)
        px = pts[..., 2:3] * v + (1.0 - v) * (-1e6)

        dz2 = (z - pz.view(B, -1, 1, 1, 1)).pow(2)
        dy2 = (y - py.view(B, -1, 1, 1, 1)).pow(2)
        dx2 = (x - px.view(B, -1, 1, 1, 1)).pow(2)

        denom = 2.0 * float(sigma) * float(sigma)
        gauss = torch.exp(-(dz2 + dy2 + dx2) / denom)
        heatmap = gauss.max(dim=1, keepdim=True).values
        return heatmap.to(dtype=torch.float32)

    @torch.no_grad()
    def _gpu_augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        """Extend parent GPU augmentations by adding heatmap generation"""
        batch_out = super()._gpu_augment_batch(batch, training=training)

        ann = batch_out.get("ann_points")
        valid = batch_out.get("ann_points_valid")
        if ann is not None and valid is not None:
            heatmap = self._rasterize_gaussians_from_points(
                ann,
                valid,
                batch_out["image"].shape,
                float(self.hparams.heatmap_sigma),
            )
        else:
            heatmap = torch.zeros_like(batch_out["image"], dtype=torch.float32)

        batch_out["heatmap"] = heatmap
        return batch_out

    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Replace auxiliary loss with heatmap-based loss"""
        labels = batch["labels"]

        loss: Dict[str, torch.Tensor] = {}

        logits_loc = out["logits_loc"]
        num_loc = int(self.hparams.num_location_classes)
        target_loc = labels[:, :num_loc].to(logits_loc.dtype)
        loss["loss_loc"] = self.crit_loc(logits_loc, target_loc)

        logit_ap = out["logit_ap"]
        target_ap = labels[:, 13].to(logit_ap.dtype)
        loss["loss_ap"] = self.crit_ap(logit_ap, target_ap)

        logits_heatmap = out.get("logits_sphere")
        if logits_heatmap is not None and "heatmap" in batch:
            tgt = batch["heatmap"].to(logits_heatmap.device)
            if tgt.shape[-3:] != logits_heatmap.shape[-3:]:
                tgt = F.interpolate(
                    tgt,
                    size=logits_heatmap.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            loss["loss_sphere_bce"] = self.crit_sphere(logits_heatmap, tgt)

        total = self.hparams.w_loc * loss["loss_loc"] + self.hparams.w_ap * loss["loss_ap"]
        if "loss_sphere_bce" in loss:
            total = total + self.hparams.w_sphere * loss["loss_sphere_bce"]
        loss["loss_total"] = total

        loss["logits_loc"] = logits_loc.detach()
        loss["logit_ap"] = logit_ap.detach()
        return loss

    @torch.no_grad()
    def _visualize_batch_first(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        split: str,
        batch_idx: int,
    ) -> None:
        """Overlay GT heatmap and predicted probability and save for visualization"""
        if batch_idx != 0:
            return
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        import os
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        image = batch.get("image")
        heatmap_gt = batch.get("heatmap")
        ann_pts = batch.get("ann_points")
        ann_val = batch.get("ann_points_valid")
        series_uid = batch.get("series_uid")

        if image is None or heatmap_gt is None:
            return

        img = image[0, 0]
        hm_gt = heatmap_gt[0, 0].to(dtype=torch.float32)

        logits = out.get("logits_sphere")
        pred_prob: Optional[torch.Tensor] = None
        if logits is not None:
            pred_resized = F.interpolate(
                logits[0:1],
                size=img.shape,
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            pred_prob = torch.sigmoid(pred_resized)

        pts0 = ann_pts[0] if ann_pts is not None else None
        val0 = ann_val[0] if ann_val is not None else None
        zc, yc, xc = self._center_from_annotations_or_mask(
            pts0,
            val0,
            (hm_gt > float(self.hparams.heatmap_vis_threshold)).to(dtype=torch.bool),
        )
        D, H, W = img.shape
        zc = int(max(0, min(zc, D - 1)))
        yc = int(max(0, min(yc, H - 1)))
        xc = int(max(0, min(xc, W - 1)))

        img_np = self._to_numpy_display(img)
        gt_bin_np = (hm_gt > float(self.hparams.heatmap_vis_threshold)).cpu().numpy()
        pred_np = pred_prob.detach().cpu().numpy() if pred_prob is not None else None

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np[zc], cmap="gray")
        if pred_np is not None:
            axes[0].imshow(pred_np[zc], cmap="hot", alpha=0.6, vmin=0, vmax=1)
        axes[0].contour(gt_bin_np[zc].astype(float), levels=[0.5], colors="cyan", linewidths=1)
        axes[0].set_title(f"Axial z={zc}")
        axes[0].axis("off")

        axes[1].imshow(img_np[:, yc, :], cmap="gray", origin="lower")
        if pred_np is not None:
            axes[1].imshow(pred_np[:, yc, :], cmap="hot", alpha=0.6, origin="lower", vmin=0, vmax=1)
        axes[1].contour(
            gt_bin_np[:, yc, :].astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=1,
            origin="lower",
        )
        axes[1].set_title(f"Coronal y={yc}")
        axes[1].axis("off")

        axes[2].imshow(img_np[:, :, xc], cmap="gray", origin="lower")
        if pred_np is not None:
            axes[2].imshow(pred_np[:, :, xc], cmap="hot", alpha=0.6, origin="lower", vmin=0, vmax=1)
        axes[2].contour(
            gt_bin_np[:, :, xc].astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=1,
            origin="lower",
        )
        axes[2].set_title(f"Sagittal x={xc}")
        axes[2].axis("off")
        plt.tight_layout()

        viz_dir = self._ensure_viz_dir()
        uid = None
        if isinstance(series_uid, (list, tuple)):
            uid = series_uid[0]
        elif isinstance(series_uid, torch.Tensor) and series_uid.dim() == 0:
            uid = str(series_uid.item())
        else:
            uid = series_uid if series_uid is not None else "sample0"
        fname = f"epoch_{self.current_epoch:04d}_{split}_b0_{uid}.png"
        save_path = os.path.join(viz_dir, fname)
        plt.savefig(save_path)
        plt.close(fig)
