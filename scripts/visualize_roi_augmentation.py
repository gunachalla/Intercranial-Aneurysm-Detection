"""
Visualize training data processing (after GPU augmentation) for AneurysmVesselROILitModule.

Displays:
- Augmented ROI images
- Augmented vessel segmentations (union or selected channel)
- Augmented sphere masks (including regeneration from points)

Example:
  python scripts/visualize_roi_augmentation.py \
    --vessel_pred_dir /workspace/outputs/nnUNet_inference/predictions \
    --train_csv /workspace/data/train.csv \
    --save_dir /workspace/outputs/roi_aug_vis \
    --n_batches 2 --batch_size 2 --input_size 128 224 224

Notes:
- Takes a few batches from DataModule's training DataLoader and visualizes
  outputs after passing through LightningModule._gpu_augment_batch(...).
- Uses CUDA automatically if available, else CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

import rootutils

# Root setup (discover .project-root and add to sys.path)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.rsna_aneurysm_vessel_seg_datamodule import RSNAAneurysmVesselSegDataModule
from src.models.aneurysm_vessel_seg_roi_module import AneurysmVesselSegROILitModule


class _LightweightROIBackbone(nn.Module):
    """Lightweight backbone with minimal IO for visualization."""

    def __init__(self, in_channels: int = 1, feat_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feat_channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feat_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.sphere_head = nn.Conv3d(feat_channels, 1, kernel_size=1)
        self._feat_channels = int(feat_channels)

    def feature_channels(self) -> int:
        return self._feat_channels

    def forward(
        self,
        x: torch.Tensor,
        vessel_seg: torch.Tensor | None = None,
        vessel_union: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        sphere = self.sphere_head(feat)
        return {"feat": feat, "logits_sphere": sphere}


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert torch.Tensor to numpy (detach->cpu)."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_img_for_display(img: np.ndarray) -> Tuple[float, float]:
    """Determine intensity range for display (2-98 percentile)."""
    p2, p98 = np.percentile(img, (2, 98))
    if p2 == p98:
        p2, p98 = float(np.min(img)), float(np.max(img))
    return float(p2), float(p98)


def _plot_3views(
    base: np.ndarray,
    overlay: np.ndarray | None,
    title: str,
    save_path: Path | None,
    overlay_cmap: str = "Reds",
    overlay_alpha: float = 0.5,
) -> None:
    """
    Show three orthogonal views (Axial/Sagittal/Coronal) and optionally save.

    Args:
        base: (D,H,W) base image
        overlay: (D,H,W) overlay (optional)
        title: figure title
        save_path: output path (no save if None)
    """
    D, H, W = base.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

    axial_img = base[mid_d]
    sagittal_img = base[:, :, mid_w]
    coronal_img = base[:, mid_h, :]

    axial_m = overlay[mid_d] if overlay is not None else None
    sagittal_m = overlay[:, :, mid_w] if overlay is not None else None
    coronal_m = overlay[:, mid_h, :] if overlay is not None else None

    vmin, vmax = _normalize_img_for_display(base)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    views = [
        (axial_img, axial_m, "Axial (Z)"),
        (sagittal_img, sagittal_m, "Sagittal (X)"),
        (coronal_img, coronal_m, "Coronal (Y)"),
    ]
    for ax, (img, msk, name) in zip(axes, views):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if msk is not None and np.any(msk > 0):
            ax.imshow(msk, cmap=overlay_cmap, alpha=overlay_alpha, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def visualize_after_gpu_aug(
    batch: Dict[str, torch.Tensor],
    aug_batch: Dict[str, torch.Tensor],
    save_dir: Path,
    batch_idx: int,
) -> None:
    """
    Save 3-view images of augmented image, vessel (union), and sphere mask.
    """
    B = aug_batch["image"].shape[0]
    imgs = _to_numpy(aug_batch["image"])  # (B,1,D,H,W)
    # vessel channel dim is 13 or K, union is (B,1,D,H,W)
    vessel = _to_numpy(aug_batch.get("vessel_seg", None)) if "vessel_seg" in aug_batch else None
    vessel_union = _to_numpy(aug_batch.get("vessel_union", None)) if "vessel_union" in aug_batch else None
    sphere = _to_numpy(aug_batch.get("sphere_mask", None)) if "sphere_mask" in aug_batch else None

    # series_uid is a list (kept by collate)
    uids = batch.get("series_uid", None)
    if isinstance(uids, list):
        uids = [str(u) for u in uids]
    else:
        uids = [str(uids)] * B

    for b in range(B):
        uid = uids[b] if b < len(uids) else f"idx{b}"
        base = imgs[b, 0].astype(np.float32)

        # Prefer union; otherwise use channel-wise max
        if vessel_union is not None:
            v_ov = vessel_union[b, 0]
        elif vessel is not None:
            v_ov = np.max(vessel[b], axis=0)
        else:
            v_ov = None

        s_ov = sphere[b, 0] if sphere is not None else None

        # Image only
        _plot_3views(
            base,
            None,
            title=f"{uid} | after GPU aug: image",
            save_path=save_dir / f"b{batch_idx:02d}_s{b:02d}_{uid}_img.png",
        )
        # Vessel overlay
        _plot_3views(
            base,
            v_ov,
            title=f"{uid} | after GPU aug: vessel",
            save_path=save_dir / f"b{batch_idx:02d}_s{b:02d}_{uid}_vessel.png",
            overlay_cmap="turbo",
            overlay_alpha=0.5,
        )
        # Sphere mask overlay
        _plot_3views(
            base,
            s_ov,
            title=f"{uid} | after GPU aug: sphere",
            save_path=save_dir / f"b{batch_idx:02d}_s{b:02d}_{uid}_sphere.png",
            overlay_cmap="Reds",
            overlay_alpha=0.55,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualization after AneurysmVesselROI GPU augmentation")
    parser.add_argument("--vessel_pred_dir", type=str, required=True, help="nnUNet inference results directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory to save images")
    parser.add_argument(
        "--train_csv", type=str, default="/workspace/data/train.csv", help="Training CSV (train.csv)"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_batches", type=int, default=10, help="Number of batches to visualize")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, nargs=3, default=[128, 224, 224])
    parser.add_argument(
        "--spatial_transform",
        type=str,
        default="resize",
        choices=["resize", "pad", "pad_to_size", "resize_xy_long_z_downonly"],
        help="ROI space normalization method (resize/pad/pad_to_size/resize_xy_long_z_downonly)",
    )
    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=32,
        help="Multiple for padding when spatial_transform=pad",
    )
    parser.add_argument("--sphere_radius", type=int, default=5)
    parser.add_argument("--train_select_k_vessels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare DataModule
    dm = RSNAAneurysmVesselSegDataModule(
        vessel_pred_dir=args.vessel_pred_dir,
        train_csv=args.train_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        n_folds=5,
        fold=0,
        cache_data=True,
        input_size=tuple(args.input_size),
        spatial_transform=args.spatial_transform,
        pad_multiple=args.pad_multiple,
    )
    dm.prepare_data()
    dm.setup("fit")

    # Prepare Module (optimizer unused, pass Adam for compatibility)
    backbone = _LightweightROIBackbone(in_channels=1, feat_channels=64)

    module = AneurysmVesselSegROILitModule(
        net=backbone,
        optimizer=torch.optim.Adam,
        scheduler=None,
        optimize_config=SimpleNamespace(mode="normal"),
        compile=False,
        cls_hidden=256,
        cls_dropout=0.1,
        w_loc=1.0,
        w_ap=1.0,
        w_sphere=1.0,
        num_location_classes=13,
        train_select_k_vessels=args.train_select_k_vessels,
        sphere_radius=args.sphere_radius,
        rand_swap_prob=0.0,
    )
    module.to(device)
    module.eval()

    # Fetch batches, run GPU augmentation, then visualize
    loader = dm.train_dataloader()
    for bidx, batch in enumerate(loader):
        # Move to device (only keys required for visualization)
        batch_dev: Dict[str, torch.Tensor] = {}
        for k in [
            "image",
            "vessel_label",
            "vessel_seg",
            "vessel_union",
            "sphere_mask",
            "labels",
            "selected_loc_idx",
            "ann_points",
            "ann_points_valid",
            "extra_vessel_label",
        ]:
            if k in batch:
                batch_dev[k] = batch[k].to(device)
        # Keep series_uid as-is
        if "series_uid" in batch:
            batch_dev["series_uid"] = batch["series_uid"]

        with torch.no_grad():
            aug = module._gpu_augment_batch(batch_dev, training=True)

        visualize_after_gpu_aug(batch, aug, save_dir, bidx)

        print(f"Saved batch {bidx} visualizations to {save_dir}")
        if bidx + 1 >= int(args.n_batches):
            break

    print("\nVisualization after GPU augmentation completed.")


if __name__ == "__main__":
    main()
