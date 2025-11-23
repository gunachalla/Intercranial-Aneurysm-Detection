#!/usr/bin/env python3
"""
Debug tool to inspect aneurysm ROI dataset augmentations with napari.

Features:
- Load Hydra config (`train.yaml` + experiment) and construct dataset via DataModule
- Compare training CPU augmentations vs. validation transforms
- Visualize module-side GPU augmentations (BatchedRand*) as additional layers
 - Key bindings: n(next), p(prev), r(resample), g(toggle GPU), v(switch view)

Example:
  python scripts/napari_roi_augmentation_viewer.py \
    --experiment 250927-seg_tf_moe_shareLR-nnunet_pixshuV2-mask_mean-dilate1_xy-s96_192-z_xy-lr1e-4_1e-5-bs1_8-e20-ema995 \
    --override data.fold=0 paths.data_dir=/workspace/data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import napari
import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from monai.transforms import Compose
from omegaconf import DictConfig

import rootutils

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from napari.layers import Layer

LayerCollection = List[Layer]

from src.data.components.aneurysm_vessel_seg_dataset import (
    ANEURYSM_CLASSES,
    AneurysmVesselSegDataset,
    get_val_transforms,
)
from src.data.components.custom_transforms import (
    BatchedRandAffined,
    BatchedRandFlipd,
    BatchedRandSimulateLowResolutiond,
)
from src.data.rsna_aneurysm_vessel_seg_datamodule import RSNAAneurysmVesselSegDataModule


NUM_LOCATION_CLASSES = len(ANEURYSM_CLASSES) - 1

VIEW_LABELS = {
    "baseline": "Baseline",
    "train_cpu": "CPU",
    "train_gpu": "GPU",
}


def build_gpu_label_transforms() -> Compose:
    """Recreate the label-aware GPU augmentation Compose used in training."""
    label_keys = ["image", "vessel_label"]
    return Compose(
        [
            # BatchedRandSimulateLowResolutiond(
            #     keys=["image"],
            #     zoom_range=(0.1, 0.5),
            #     prob=1.0,
            #     allow_missing_keys=True,
            #     mode="trilinear",
            # ),
            BatchedRandSimulateLowResolutiond(
                keys=["image", "vessel_label"],
                zoom_range=[(0.5, 0.5), (0.9, 1.0), (0.9, 1.0)],
                prob=1.0,
                allow_missing_keys=True,
                mode="nearest",
            ),
            BatchedRandAffined(
                keys=label_keys,
                prob=0.6,
                rotate_range=(float(torch.pi) / 12, float(torch.pi) / 12, float(torch.pi) / 12),
                scale_range=0.10,
                mode=("trilinear", "nearest"),
                padding_mode="zeros",
                allow_missing_keys=True,
                point_keys=("ann_points",),
            ),
            BatchedRandFlipd(
                keys=label_keys,
                spatial_axis=0,
                prob=0.5,
                allow_missing_keys=True,
                point_keys=("ann_points",),
            ),
            BatchedRandFlipd(
                keys=label_keys,
                spatial_axis=1,
                prob=0.5,
                allow_missing_keys=True,
                point_keys=("ann_points",),
            ),
            BatchedRandFlipd(
                keys=label_keys,
                spatial_axis=2,
                prob=0.5,
                allow_missing_keys=True,
                point_keys=("ann_points",),
            ),
        ]
    )


def rasterize_spheres_from_points(
    pts: torch.Tensor,
    valid: torch.Tensor,
    out_shape: torch.Size,
    radius: int,
) -> torch.Tensor:
    """Rasterize spherical masks from points/valid flags under GPU-aug conditions."""
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


@dataclass
class SampleBundle:
    """Tensor bundle kept for napari display."""

    image_t: torch.Tensor
    vessel_label_t: torch.Tensor
    ann_points_t: Optional[torch.Tensor]
    ann_points_valid_t: Optional[torch.Tensor]
    sphere_mask_t: Optional[torch.Tensor]
    series_uid: str
    variant: str


class AugmentationNapariApp:
    """Wrap napari viewer and data sampling controls."""

    def __init__(
        self,
        datamodule: RSNAAneurysmVesselSegDataModule,
        baseline_dataset: AneurysmVesselSegDataset,
        gpu_transforms: Compose,
        stage: str = "train",
        enable_gpu_aug: bool = True,
        device: torch.device | str = "cpu",
        initial_index: int = 0,
        sphere_radius: int = 5,
    ) -> None:
        self.datamodule = datamodule
        self.dataset_aug = datamodule.data_train if stage == "train" else datamodule.data_val
        self.dataset_baseline = baseline_dataset
        self.gpu_transforms = gpu_transforms
        self.stage = stage
        self.enable_gpu_aug = enable_gpu_aug
        self.device = torch.device(device)
        self.sphere_radius = int(sphere_radius)
        self.view_order: list[str] = []
        self.current_view_idx: int = 0
        self.layers: Dict[str, Dict[str, LayerCollection | Layer]] = {}
        self.layer_optional_flags: Dict[str, Dict[str, bool]] = {}
        self._latest_bundle_map: Dict[str, SampleBundle] = {}
        self._last_text_info: Dict[str, object] = {}

        if self.dataset_aug is None:
            raise RuntimeError("Target dataset not found in DataModule")

        self.length = len(self.dataset_aug)
        self.index = int(initial_index) % self.length

        self.viewer = napari.Viewer(title="ROI Augmentation Preview")
        self._bind_keys()
        self.refresh()

    # ----------------------------------------------------------------------------------
    # napari events
    # ----------------------------------------------------------------------------------
    def _bind_keys(self) -> None:
        """Set key bindings."""

        @self.viewer.bind_key("n")
        def _next_viewer(event=None) -> None:  # noqa: ANN001
            self.step(1)

        @self.viewer.bind_key("p")
        def _prev_viewer(event=None) -> None:  # noqa: ANN001
            self.step(-1)

        @self.viewer.bind_key("r")
        def _resample(event=None) -> None:  # noqa: ANN001
            self.refresh(resample_only=True)

        @self.viewer.bind_key("g")
        def _toggle_gpu(event=None) -> None:  # noqa: ANN001
            self.enable_gpu_aug = not self.enable_gpu_aug
            self.refresh(resample_only=True)

        @self.viewer.bind_key("v")
        def _cycle_view(event=None) -> None:  # noqa: ANN001
            self._cycle_view_mode()

    # ----------------------------------------------------------------------------------
    # View updates
    # ----------------------------------------------------------------------------------
    def step(self, delta: int) -> None:
        """Update current index and redraw."""
        self.index = (self.index + delta) % self.length
        self.refresh()

    def _cycle_view_mode(self) -> None:
        """Cycle through available view modes."""
        if len(self.view_order) <= 1:
            return
        self.current_view_idx = (self.current_view_idx + 1) % len(self.view_order)
        self._apply_visibility()
        self._update_overlay_text(log=False)

    def _update_view_modes(self, available: Sequence[str]) -> None:
        """Update view mode order from available variants."""
        priority = ["baseline", "train_cpu", "train_gpu"]
        new_order = [key for key in priority if key in available]
        if not new_order:
            new_order = list(available)

        prev_variant = self.view_order[self.current_view_idx] if self.view_order else None
        self.view_order = new_order

        if prev_variant in self.view_order:
            self.current_view_idx = self.view_order.index(prev_variant)
        else:
            default_variant = "train_gpu" if "train_gpu" in self.view_order else self.view_order[0]
            self.current_view_idx = self.view_order.index(default_variant)

    def refresh(self, resample_only: bool = False) -> None:
        """Fetch current sample and update napari display."""
        baseline = self._fetch_baseline_sample(self.index)
        augmented = self._fetch_augmented_sample(self.index)
        gpu_aug = self._apply_gpu_aug(augmented) if self._should_apply_gpu() else None

        bundle_map = {
            "baseline": baseline,
            "train_cpu": augmented,
        }
        if gpu_aug is not None:
            bundle_map["train_gpu"] = gpu_aug

        self._update_view_modes(tuple(bundle_map.keys()))
        self._latest_bundle_map = bundle_map

        for variant, bundle in bundle_map.items():
            self._setup_or_update_layers(variant, bundle)

        self._apply_visibility()

        self._last_text_info = {
            "index": self.index,
            "length": self.length,
            "series_uid": augmented.series_uid,
            "stage": self.stage,
            "gpu": self._should_apply_gpu(),
        }
        self._update_overlay_text(log=not resample_only)

    def _should_apply_gpu(self) -> bool:
        """Whether GPU augmentation should be applied."""
        return self.stage == "train" and self.enable_gpu_aug

    # ----------------------------------------------------------------------------------
    # Data shaping utilities
    # ----------------------------------------------------------------------------------
    def _fetch_baseline_sample(self, index: int) -> SampleBundle:
        """Return numpy-converted sample with validation transforms."""
        sample = self.dataset_baseline[index]
        return self._to_bundle(sample, variant="baseline")

    def _fetch_augmented_sample(self, index: int) -> SampleBundle:
        """Return numpy-converted sample with training CPU augmentations."""
        sample = self.dataset_aug[index]
        return self._to_bundle(sample, variant="train_cpu")

    def _apply_gpu_aug(self, bundle: SampleBundle) -> Optional[SampleBundle]:
        """Apply GPU augmentations to CPU-augmented batch and format for display."""
        if bundle is None:
            return None

        image = bundle.image_t.to(self.device)
        vessel_label = bundle.vessel_label_t.to(self.device)

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if vessel_label.dim() == 3:
            vessel_label = vessel_label.unsqueeze(0)

        batch: Dict[str, torch.Tensor] = {
            "image": image.unsqueeze(0).to(dtype=torch.float32),
            "vessel_label": vessel_label.unsqueeze(0).to(dtype=torch.float32),
        }

        if bundle.ann_points_t is not None and bundle.ann_points_valid_t is not None:
            pts = bundle.ann_points_t.to(self.device).unsqueeze(0)
            valid = bundle.ann_points_valid_t.to(self.device).unsqueeze(0)
            batch["ann_points"] = pts.to(dtype=torch.float32)
            batch["ann_points_valid"] = valid.to(dtype=torch.uint8)

        data = self.gpu_transforms({k: v for k, v in batch.items() if k != "ann_points_valid"})

        label_aug = data["vessel_label"].round().clamp_(0, NUM_LOCATION_CLASSES).to(dtype=torch.int64)
        image_aug = data["image"].to(dtype=torch.float32)

        if "ann_points" in data and "ann_points_valid" in batch:
            pts_aug = data["ann_points"]
            valid_aug = batch["ann_points_valid"].to(device=pts_aug.device)
        else:
            pts_aug = None
            valid_aug = None

        if pts_aug is not None and valid_aug is not None:
            sphere = rasterize_spheres_from_points(
                pts_aug,
                valid_aug.to(dtype=torch.float32),
                image_aug.shape,
                radius=self.sphere_radius,
            )
        else:
            sphere = torch.zeros_like(label_aug, dtype=torch.uint8)

        bundle_gpu = SampleBundle(
            image_t=image_aug.squeeze(0),
            vessel_label_t=label_aug.squeeze(0),
            ann_points_t=pts_aug.squeeze(0) if pts_aug is not None else None,
            ann_points_valid_t=valid_aug.squeeze(0) if valid_aug is not None else None,
            sphere_mask_t=sphere.squeeze(0) if sphere is not None else None,
            series_uid=bundle.series_uid,
            variant="train_gpu",
        )
        return bundle_gpu

    def _to_bundle(self, sample: Dict[str, object], variant: str) -> SampleBundle:
        """Convert Dataset output dictionary for visualization."""
        image_tensor = torch.as_tensor(sample["image"]).detach()
        label_tensor = torch.as_tensor(sample["vessel_label"]).detach()

        ann_points = sample.get("ann_points")
        ann_valid = sample.get("ann_points_valid")

        bundle = SampleBundle(
            image_t=image_tensor,
            vessel_label_t=label_tensor,
            ann_points_t=torch.as_tensor(ann_points).detach() if ann_points is not None else None,
            ann_points_valid_t=torch.as_tensor(ann_valid).detach() if ann_valid is not None else None,
            sphere_mask_t=None,
            series_uid=str(sample["series_uid"]),
            variant=variant,
        )
        return bundle

    @staticmethod
    def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
        """Convert image tensor to numpy without normalization."""
        arr = tensor.detach().cpu().numpy()
        return arr

    @staticmethod
    def _to_numpy_label(tensor: torch.Tensor) -> np.ndarray:
        """Convert integer label tensor to numpy."""
        arr = tensor.detach().to(torch.int64).cpu().numpy()
        return arr

    @staticmethod
    def _maybe_points_numpy(data: Optional[torch.Tensor | np.ndarray | object]) -> Optional[np.ndarray]:
        """Convert annotation points to numpy."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        tensor = torch.as_tensor(data).detach().cpu()
        if tensor.numel() == 0:
            return None
        return tensor.numpy()

    @staticmethod
    def _maybe_valid_numpy(data: Optional[torch.Tensor | np.ndarray | object]) -> Optional[np.ndarray]:
        """Convert annotation point validity flags to numpy."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        tensor = torch.as_tensor(data).detach().cpu()
        if tensor.numel() == 0:
            return None
        return tensor.numpy()

    def _setup_or_update_layers(self, variant: str, bundle: SampleBundle) -> None:
        """Create or update napari layers and keep them for later display toggling."""
        variant_layers = self.layers.setdefault(variant, {})
        optional_flags = self.layer_optional_flags.setdefault(variant, {})
        name_prefix = variant

        # Image layer
        image = self._to_numpy_image(bundle.image_t)
        channels: List[np.ndarray]
        if image.ndim == 4:
            channels = [image[i] for i in range(image.shape[0])]
        else:
            channels = [image]

        existing_images = variant_layers.get("image")
        layers_list: LayerCollection
        if existing_images is None:
            layers_list = []
            for idx, channel_data in enumerate(channels):
                layer = self.viewer.add_image(
                    channel_data,
                    name=(f"{name_prefix}_image" if len(channels) == 1 else f"{name_prefix}_image_ch{idx}"),
                    # blending="additive",
                    # opacity=0.8,
                )
                layer.visible = False
                layers_list.append(layer)
            variant_layers["image"] = layers_list
        else:
            layers_list = list(existing_images) if isinstance(existing_images, list) else [existing_images]
            # Recreate layers if channel count changes
            if len(layers_list) != len(channels):
                for layer in layers_list:
                    self.viewer.layers.remove(layer)
                layers_list = []
                for idx, channel_data in enumerate(channels):
                    layer = self.viewer.add_image(
                        channel_data,
                        name=(
                            f"{name_prefix}_image" if len(channels) == 1 else f"{name_prefix}_image_ch{idx}"
                        ),
                        # blending="additive",
                        # opacity=0.8,
                    )
                    layer.visible = False
                    layers_list.append(layer)
                variant_layers["image"] = layers_list
            else:
                for layer, channel_data in zip(layers_list, channels):
                    layer.data = channel_data
        optional_flags["image"] = True

        # Label layer
        label = self._to_numpy_label(bundle.vessel_label_t)
        if label.ndim == 4:
            label = label.squeeze(0)
        label_layer_entry = variant_layers.get("label")
        label_layer: Layer
        if label_layer_entry is None or isinstance(label_layer_entry, list):
            label_layer = self.viewer.add_labels(
                label.astype(np.int16),
                name=f"{name_prefix}_label",
                opacity=0.4,
            )
            label_layer.visible = False
            variant_layers["label"] = label_layer
        else:
            label_layer = label_layer_entry
            label_layer.data = label.astype(np.int16)
        optional_flags["label"] = True

        # Sphere mask layer (optional)
        sphere_layer_entry = variant_layers.get("sphere")
        sphere = None
        if bundle.sphere_mask_t is not None:
            sphere = self._to_numpy_label(bundle.sphere_mask_t)
            if sphere.ndim == 4:
                sphere = sphere.squeeze(0)

        sphere_layer: Layer
        if sphere_layer_entry is None or isinstance(sphere_layer_entry, list):
            base_shape = label.shape
            empty = np.zeros(base_shape, dtype=np.uint8)
            sphere_layer = self.viewer.add_labels(
                empty,
                name=f"{name_prefix}_sphere",
                opacity=0.3,
            )
            try:
                sphere_layer.color = {1: "magenta"}
            except Exception:
                pass
            sphere_layer.visible = False
            variant_layers["sphere"] = sphere_layer
        else:
            sphere_layer = sphere_layer_entry

        if sphere is not None:
            sphere_layer.data = sphere.astype(np.uint8)
            optional_flags["sphere"] = True
        else:
            sphere_layer.data = np.zeros_like(sphere_layer.data)
            optional_flags["sphere"] = False

        # Annotation point layer (optional)
        points_layer_entry = variant_layers.get("points")
        pts_np = self._maybe_points_numpy(bundle.ann_points_t)
        valid_np = self._maybe_valid_numpy(bundle.ann_points_valid_t)
        if pts_np is None or valid_np is None:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            mask = valid_np > 0
            points = pts_np[mask] if mask.any() else np.empty((0, 3), dtype=np.float32)

        points_layer: Layer
        if points_layer_entry is None or isinstance(points_layer_entry, list):
            points_layer = self.viewer.add_points(
                points,
                name=f"{name_prefix}_ann_points",
                size=4,
                face_color="yellow",
            )
            points_layer.visible = False
            variant_layers["points"] = points_layer
        else:
            points_layer = points_layer_entry
            points_layer.data = points

        optional_flags["points"] = points.size > 0

    def _apply_visibility(self) -> None:
        """Update layer visibility according to current view mode."""
        if not self.view_order:
            return
        current_variant = self.view_order[self.current_view_idx]
        for variant, layer_dict in self.layers.items():
            is_active = variant == current_variant
            flags = self.layer_optional_flags.get(variant, {})
            for kind, layer in layer_dict.items():
                if layer is None:
                    continue
                layer_objs = layer if isinstance(layer, list) else [layer]
                visible = bool(is_active)
                if kind in ("sphere", "points"):
                    visible = bool(is_active and flags.get(kind, False))
                for obj in layer_objs:
                    obj.visible = visible

    def _update_overlay_text(self, log: bool = False) -> None:
        """Update text overlay to latest info."""
        if not self._last_text_info or not self.view_order:
            return
        current_variant = self.view_order[self.current_view_idx]
        info = self._last_text_info
        text_lines = [
            f"index: {int(info['index']) + 1}/{int(info['length'])}",
            f"series_uid: {info['series_uid']}",
            f"stage: {info['stage']}",
            f"view: {VIEW_LABELS.get(current_variant, current_variant)}",
            f"gpu_aug: {'on' if info['gpu'] else 'off'}",
            "press v: switch view",
        ]
        self.viewer.text_overlay.visible = True
        try:
            self.viewer.text_overlay.position = "top_left"
        except Exception:
            pass
        self.viewer.text_overlay.text = "\n".join(text_lines)

        if log:
            print("\n".join(text_lines))


def _normalize_percentile(volume: np.ndarray) -> np.ndarray:
    """Apply 1-99 percentile normalization."""
    vmin = np.percentile(volume, 1.0)
    vmax = np.percentile(volume, 99.0)
    if vmax > vmin:
        norm = (volume - vmin) / (vmax - vmin)
    else:
        norm = volume - volume.min()
        max_val = norm.max() if norm.max() > 0 else 1.0
        norm = norm / max_val
    return norm.astype(np.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ROI augmentation preview script")
    parser.add_argument(
        "--config-name",
        default="train",
        help="Hydra base config name (e.g., train)",
    )
    parser.add_argument(
        "--experiment",
        default="250927-seg_tf_moe_shareLR-nnunet_pixshuV2-mask_mean-dilate1_xy-s96_192-z_xy-lr1e-4_1e-5-bs1_8-e20-ema995",
        help="Experiment name under `configs/experiment`",
    )
    parser.add_argument(
        "--config-dir",
        default="../configs",
        help="Path to Hydra config folder",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Additional Hydra overrides (e.g., data.fold=0)",
    )
    parser.add_argument(
        "--stage",
        choices=["train", "val"],
        default="train",
        help="Target dataset to display",
    )
    parser.add_argument(
        "--spatial-transform",
        choices=["resize", "pad", "pad_to_size"],
        default=None,
        help="Force ROI preprocessing mode (use config if unspecified)",
    )
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=None,
        help="Padding multiple when spatial_transform=pad",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for GPU augmentation (cuda / cpu)",
    )
    parser.add_argument(
        "--initial-index",
        type=int,
        default=0,
        help="Initial index",
    )
    parser.add_argument(
        "--series-uid",
        default=None,
        help="SeriesInstanceUID to start from",
    )
    parser.add_argument(
        "--disable-gpu-aug",
        action="store_true",
        help="Disable adding GPU augmentation layers",
    )
    parser.add_argument(
        "--sphere-radius",
        type=int,
        default=5,
        help="Radius for sphere mask regeneration",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    """Load Hydra configuration."""
    overrides = list(args.override)
    overrides.append(f"experiment={args.experiment}")
    if args.spatial_transform is not None:
        overrides.append(f"data.spatial_transform={args.spatial_transform}")
    if args.pad_multiple is not None:
        overrides.append(f"data.pad_multiple={int(args.pad_multiple)}")
    with initialize(version_base="1.3", config_path=args.config_dir):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    return cfg


def instantiate_datamodule(cfg: DictConfig) -> RSNAAneurysmVesselSegDataModule:
    """Instantiate DataModule from Hydra config."""
    datamodule: RSNAAneurysmVesselSegDataModule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    return datamodule


def build_baseline_dataset(
    datamodule: RSNAAneurysmVesselSegDataModule,
    stage: str,
    input_size: Sequence[int],
    keep_ratio: str,
    spatial_transform: str,
    pad_multiple: Optional[int],
) -> AneurysmVesselSegDataset:
    """Build dataset with validation transforms on the same series list."""
    if stage == "train":
        series_list = getattr(datamodule, "train_series_list")
    else:
        series_list = getattr(datamodule, "val_series_list")

    transforms = get_val_transforms(
        input_size=tuple(map(int, input_size)),
        keep_ratio=keep_ratio,
        spatial_transform=spatial_transform,
        pad_multiple=pad_multiple if pad_multiple is not None else 32,
    )

    baseline = AneurysmVesselSegDataset(
        vessel_pred_dir=datamodule.hparams.vessel_pred_dir,
        train_csv=datamodule.hparams.train_csv,
        series_list=series_list,
        transform=transforms,
        cache_data=False,
    )
    return baseline


def main() -> None:
    """Script entry point."""
    args = parse_args()
    cfg = load_config(args)
    datamodule = instantiate_datamodule(cfg)

    input_size = cfg.data.get("input_size", datamodule.hparams.input_size)
    keep_ratio = cfg.data.get("keep_ratio", datamodule.hparams.keep_ratio)
    spatial_transform = cfg.data.get(
        "spatial_transform",
        getattr(datamodule.hparams, "spatial_transform", "resize"),
    )
    pad_multiple_cfg = cfg.data.get("pad_multiple", getattr(datamodule.hparams, "pad_multiple", None))
    pad_multiple = int(pad_multiple_cfg) if pad_multiple_cfg is not None else None

    baseline_dataset = build_baseline_dataset(
        datamodule,
        stage=args.stage,
        input_size=input_size,
        keep_ratio=keep_ratio,
        spatial_transform=spatial_transform,
        pad_multiple=pad_multiple,
    )

    initial_index = int(args.initial_index)
    if args.series_uid is not None:
        try:
            initial_index = baseline_dataset.cases.index(args.series_uid)
        except ValueError as exc:
            raise ValueError(f"series_uid {args.series_uid} not found in dataset") from exc

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"

    gpu_transforms = build_gpu_label_transforms()
    app = AugmentationNapariApp(
        datamodule=datamodule,
        baseline_dataset=baseline_dataset,
        gpu_transforms=gpu_transforms,
        stage=args.stage,
        enable_gpu_aug=not args.disable_gpu_aug,
        device=device,
        initial_index=initial_index,
        sphere_radius=args.sphere_radius,
    )
    napari.run()


if __name__ == "__main__":
    main()
