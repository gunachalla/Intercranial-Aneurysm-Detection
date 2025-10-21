#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSNA 2025 Brain Aneurysm Detection - Submission (ROI pipeline)

This module implements Kaggle-compatible `predict(series_path)` returning
probabilities for 14 labels (13 locations + AP) via the following steps:

1) DICOM → NIfTI conversion (dcm2niix; majority-size selection + fallback)
2) nnU-Net vessel segmentation (adaptive sparse search) → ROI image + label map
3) ROI classification (trained LightningModule) predicts 13 locations + AP
   - 13 locations pooled via region-masked averaging
   - AP predicted by a separate head on the vessel union mask
4) Fold ensemble (mean)

Configurable via environment variables:
- VESSEL_NNUNET_MODEL_DIR: nnU-Net vessel segmentation model path
- VESSEL_NNUNET_SPARSE_MODEL_DIR: nnU-Net model path for sparse search (defaults to above if unset)
- VESSEL_ADDITIONAL_DENSE_MODEL_DIRS: Comma-separated nnU-Net model paths for additional dense inference
- VESSEL_FOLDS: Segmentation folds (e.g., "0,1,2,3" or "all")
- VESSEL_DEVICE: Device for nnU-Net segmentation (e.g., "cuda:0" / "1" / "cpu")
- VESSEL_DEVICES: Comma-separated GPUs for multi-GPU runs (e.g., "0,1,3")
- VESSEL_ENABLE_ORIENTATION_CORRECTION: Enable orientation correction via sparse seg result
- VESSEL_ORIENTATION_WEIGHTS: Weights for orientation estimation (w_AP,w_LR,w_SI) as comma-separated values
  ROI re-crop margins (after dense inference):
  - VESSEL_REFINE_MARGIN_Z: z-axis margin in voxels
  - VESSEL_REFINE_MARGIN_XY: x/y-plane margin in voxels
  - VESSEL_REFINE_Z_ONLY: Limit BBox refine to z (SI) direction (ON/OFF). Defaults to True if unset.

TensorRT (speed up nnU-Net segmentation):
- VESSEL_TRT_DIR: Root directory for TensorRT engines (trainer__plans__config/fold_*). Use TensorRT when set.
- VESSEL_TRT_PRECISION: Engine precision ("fp16" | "fp32"), default "fp16".
- ROI_EXPERIMENTS: Comma-separated Hydra experiment names for ROI classification
  - ROI_EXPERIMENT is kept for backward-compatibility (single value)
- ROI_FOLDS: Folds for ROI classification (e.g., "0,1,2,3,4")
- ROI_TTA: TTA count for ROI inference (1/2/4/8). 1 means no TTA.
- RSNA_DEBUG: Print debug info during prediction (enable with "1"/"true", etc.)

Override ROI backbone (nnU-Net) directory for Kaggle runs:
- ROI_NNUNET_MODEL_DIR: When RUN_MODE=kaggle, override cfg.model.net.nnunet_model_dir with this value
  - Can be comma-separated. Match the order of `ROI_EXPERIMENTS`.
  - If a single value is given, broadcast to all experiments.
  - If unset, fallback to VESSEL_NNUNET_MODEL_DIR as before.

Override ROI classification input size:
- ROI_INPUT_SIZE: Override model input size (e.g., "96,192,192" or "96x192x192")

Checkpoint selection (env var):
- ROI_CKPT: "last" (default) or "best"
  - "last": load `last.ckpt`
  - "best": load `epoch_*.ckpt` with the highest epoch number
  - Raise an error if the target checkpoint is not found

Score output switches (for AUC diagnostics):
- RSNA_ONLY_AP: if "1", use model prediction only for AP; set locations to a fixed value
- RSNA_ONLY_LOCATIONS: if "1", use model prediction only for 13 locations; set AP to a fixed value
- RSNA_FIXED_PROB: Fixed value for non-target classes (default: 0.5)

Error monitoring (fail-safe):
- RSNA_MAX_PREDICT_ERRORS: Abort immediately if prediction exceptions exceed this count (<=0 disables)
- RSNA_ERROR_LOG: JSONL path to write prediction errors (default: /kaggle/working/predict_errors.jsonl)
- RSNA_ERROR_FALLBACK_PROBS: Comma-separated 14 probabilities returned on error (default: all 0.1)

Behavior on sparse-search failure:
- VESSEL_ABORT_ON_SPARSE_FAIL: Enable with "1". When VESSEL_NNUNET_SPARSE_MODEL_DIR is set and sparse_search fails
  (i.e., deemed to have fallen back to full FOV), raise an exception without running dense inference.
  Then predict() returns RSNA_ERROR_FALLBACK_PROBS.

Behavior on small ROI volume:
- VESSEL_ABORT_ON_SMALL_ROI: Enable with "1". If ROI volume is below threshold, raise and return
  RSNA_ERROR_FALLBACK_PROBS from predict().
- VESSEL_MIN_ROI_VOXELS: Minimum ROI volume (voxels). Disabled if unset. Absolute count only (no ratio checks).

Behavior on low union voxel count (detect seg failure):
- VESSEL_ABORT_ON_LOW_UNION: Enable with "1". If the sum over union_b (vessel union at model resolution)
  is below threshold in _prepare_inputs_for_roi(), raise and return RSNA_ERROR_FALLBACK_PROBS.
- VESSEL_MIN_UNION_SUM: Minimum union_b.sum() (voxels). Disabled if unset.

Staged diagnostics (for hidden test):
- RSNA_STAGE: Final stage to execute
  - "dicom": Only DICOM→NIfTI (then return fixed probabilities)
  - "vessel": DICOM→NIfTI + vessel segmentation (skip ROI classification and return fixed probabilities)
  - "roi" (default/unset): Run the full pipeline
- RSNA_STAGE_PLACEHOLDER: Fixed probability to return when skipping (default: 0.1)
- RSNA_VESSEL_STAGE_LIMIT: Final stage for vessel processing
  ("load"|"preprocess"|"sparse_search"|"roi_crop"|"fold_infer"|"refine"|"aggregate", default "aggregate")

Note:
- Place trained models (nnU-Net and ROI classifier) under /kaggle/input or /workspace/logs beforehand.

Additional sparse-search settings (DBSCAN-based ROI extraction):
- VESSEL_SPARSE_ROI_EXTENT_MM: ROI physical size (mm) for DBSCANAdaptiveSparsePredictor
  - Single value (e.g., "130") or comma-separated per-axis (e.g., "130,130,150")
  - Effective only when using a separate sparse model via VESSEL_NNUNET_SPARSE_MODEL_DIR
"""

from __future__ import annotations

import os
import json
import logging  # standardized logging
import time  # timing measurements
import shutil
from datetime import datetime
import tempfile
import gc
import contextlib
import re  # for extracting epoch numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import hydra
from matplotlib import pyplot as plt

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient

# Root setup
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import existing utilities/modules
from src.my_utils.rsna_dcm2niix import which_or_die, convert_dicom_to_nifti
from src.my_utils.vessel_segmentation import (
    VesselSegmentationPredictor,
    VesselSegmentationStage,
    VesselSegmentationOutput,
    parse_vessel_stage_limit,
)
from src.my_utils.kaggle_utils import load_experiment_config
from src.data.components.aneurysm_vessel_seg_dataset import (
    ANEURYSM_CLASSES,
    DET_TO_SEG_OFFSET,
    get_val_transforms,
)
from src.models.aneurysm_vessel_seg_roi_module import AneurysmVesselSegROILitModule


# ===== Constants (column names) =====
LABEL_COLS = ANEURYSM_CLASSES  # 13 locations + Aneurysm Present (last)

# Default ROI experiment name
DEFAULT_ROI_EXPERIMENT = "250903-vessel_roi-nnunet_pretrained-s96_192-lr1e-4-bs1_8"

_ALLOWED_TTA_COUNTS = (1, 2, 4, 8)


def _normalize_tta_count(raw: int) -> int:
    """Clamp TTA count to supported values."""

    if raw <= 1:
        return 1
    for v in _ALLOWED_TTA_COUNTS:
        if raw <= v:
            return v
    return _ALLOWED_TTA_COUNTS[-1]


def _build_tta_flip_flags(count: int) -> List[Tuple[bool, bool, bool]]:
    """Flip combinations (z, y, x axes) according to the requested count."""

    configs: List[Tuple[bool, bool, bool]] = [(False, False, False)]
    if count <= 1:
        return configs

    configs.append((False, False, True))  # x flip (left-right)
    if count <= 2:
        return configs[:count]

    configs.append((False, True, False))  # y flip (anterior-posterior)
    configs.append((True, False, False))  # z flip (inferior-superior)
    if count <= 4:
        return configs[:count]

    configs.extend(
        [
            (False, True, True),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ]
    )
    return configs[:count]


def _prune_unused_roi_variants(module: AneurysmVesselSegROILitModule) -> None:
    """Drop unused (EMA vs non-EMA) branches for inference-only usage."""

    base_runtime = getattr(module, "model", None)
    if base_runtime is None:
        return

    try:
        use_ema = bool(module._use_ema_for_eval())
    except Exception:
        use_ema = False

    try:
        selected_runtime = base_runtime.get_runtime_module(use_ema=use_ema)
    except Exception:
        selected_runtime = base_runtime
        use_ema = False

    module.model = selected_runtime

    if hasattr(selected_runtime, "ema_model"):
        selected_runtime.ema_model = None
        if hasattr(selected_runtime, "ema_enabled"):
            selected_runtime.ema_enabled = False

    if selected_runtime is not base_runtime and hasattr(base_runtime, "ema_model"):
        base_runtime.ema_model = None

    gc.collect()


def _apply_flip_for_tta(tensor: torch.Tensor, flags: Tuple[bool, bool, bool]) -> torch.Tensor:
    """Apply flips along specified axes (z, y, x)."""

    flip_dims: List[int] = []
    if flags[0]:
        flip_dims.append(2)
    if flags[1]:
        flip_dims.append(3)
    if flags[2]:
        flip_dims.append(4)
    if not flip_dims:
        return tensor
    return torch.flip(tensor, dims=flip_dims).contiguous()


# ===== Label Mapping =====
_DET_TO_SEG_OFFSET_TORCH = torch.tensor(DET_TO_SEG_OFFSET, dtype=torch.long)
_SEG_TO_DET_TORCH = torch.zeros(14, dtype=torch.long)
_SEG_TO_DET_TORCH[_DET_TO_SEG_OFFSET_TORCH + 1] = torch.arange(1, 14, dtype=torch.long)


# ===== Minimal Globals =====
def _resolve_device_from_env() -> torch.device:
    """Determine device for Vessel from environment variables.

    Priority:
    - If `VESSEL_DEVICE` is set, use it (e.g., "cuda:1" / "1" / "cpu").
    - Otherwise default to `cuda` for backward compatibility.
    """
    # Read environment variable (treat empty string as unset)
    raw = os.getenv("VESSEL_DEVICE", "").strip()
    if raw == "":
        # Compatibility allowance for potential alias in the future
        raw = os.getenv("VESSEL_DEVICE_ID", "").strip()

    if raw != "":
        key = raw.lower()
        # If only digits, treat as GPU index (e.g., "1" -> cuda:1)
        if re.fullmatch(r"-?\d+", key):
            try:
                idx = int(key)
                return torch.device(f"cuda:{idx}")
            except Exception:
                pass
        # Otherwise, rely on torch.device parsing (cpu/cuda/cuda:0)
        try:
            return torch.device(key)
        except Exception:
            # On parse failure, fall back to default
            pass

    # Keep legacy default (always "cuda").
    return torch.device("cuda")


_device = _resolve_device_from_env()
# Flag to check dcm2niix / gdcmconv only once
_dcm_tools_checked: bool = False


# ===== Timing Utilities =====
@contextlib.contextmanager
def time_section(label: str, enabled: bool = False):
    """Simple section timing. Logs elapsed seconds at DEBUG when enabled.

    Args:
        label: label for display
        enabled: when True, emit timing logs
    """
    logger = logging.getLogger(__name__)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            dt = time.perf_counter() - t0
            logger.debug(f"[DEBUG][{label}] {dt:.3f}s")


_logging_configured: bool = False  # flag to ensure forced config is done only once


def _configure_logging(debug: bool) -> None:
    """Initialize/update logging (reliable display in Kaggle notebooks).

    - First call: basicConfig(force=True) to override existing handlers (remove IPython defaults).
    - Subsequent calls: only update level (also align handler levels).
    - Delegate module loggers to root (propagate=True).
    """
    global _logging_configured
    level = logging.DEBUG if debug else logging.WARNING
    if not _logging_configured:
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)
        _logging_configured = True
    else:
        root = logging.getLogger()
        root.setLevel(level)
        for h in root.handlers:
            try:
                h.setLevel(level)
            except Exception:
                pass
    # Configure this module logger (no handlers; propagate to root)
    logger = logging.getLogger(__name__)
    logger.propagate = True
    logger.setLevel(level)

    # Suppress verbose matplotlib font search DEBUG logs
    mpl_level = logging.INFO if debug else logging.WARNING
    for name in ("matplotlib", "matplotlib.font_manager"):
        mpl_logger = logging.getLogger(name)
        mpl_logger.setLevel(mpl_level)
        mpl_logger.propagate = False


# ===== Centralized Configuration =====
def _clear_cuda_cache() -> None:
    """Lightweight cache clearing to mitigate CUDA memory fragmentation."""
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class PipelineOptions:
    """Pipeline-wide options (constructed from environment variables)."""

    # Common
    debug: bool = False
    rsna_stage: str = "roi"  # "dicom" | "vessel" | "roi"
    stage_placeholder: float = 0.1
    only_ap: bool = False
    only_locations: bool = False
    fixed_prob: float = 0.5
    error_fallback_probs: List[float] = field(default_factory=lambda: [0.1 for _ in range(len(LABEL_COLS))])
    # Abort without dense inference when sparse search fails (enable via env var)
    abort_on_sparse_fail: bool = False
    # Physical size (mm) guard for abort decision (do not abort if all dims < threshold). Disabled when None.
    abort_min_all_dims_mm: Optional[float] = None
    # Abort when ROI volume is too small
    abort_on_small_roi: bool = False
    # Minimum ROI volume (voxels). Only used when set.
    min_roi_voxels: Optional[int] = None
    # Minimum sum of union_b (voxels). Only used when set.
    abort_on_low_union: bool = False
    min_union_sum: Optional[int] = None

    # Vessel segmentation
    vessel_model_dir: str = (
        "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/"
        "RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusWithBackground__nnUNetResEncUNetMPlans__3d_fullres"
    )
    vessel_sparse_model_dir: Optional[str] = None
    vessel_additional_model_dirs: List[str] = field(default_factory=list)
    vessel_folds: List[int | str] = field(default_factory=lambda: [0, 1, 2])
    vessel_stage_limit: VesselSegmentationStage = VesselSegmentationStage.AGGREGATE
    # Device assignment for nnU-Net. None lets Predictor autodetect.
    vessel_devices: Optional[List[int]] = None
    # Whether to use sparse search (default: True). False forces dense over full FOV.
    vessel_use_sparse: bool = True

    # Detailed vessel inference parameters (applied only when set)
    window_count_threshold: Optional[int] = None
    sparse_downscale_factor: Optional[float] = None
    sparse_overlap: Optional[float] = None
    detection_threshold: Optional[float] = None
    sparse_bbox_margin_voxels: Optional[int] = None
    dense_overlap: Optional[float] = None
    use_gaussian: Optional[bool] = None
    perform_everything_on_device: Optional[bool] = None
    limit_si_extent: Optional[bool] = None
    max_si_extent_mm: Optional[float] = None
    min_si_extent_mm: Optional[float] = None
    min_xy_extent_mm: Optional[float] = None
    si_axis: Optional[int] = None
    refine_roi_after_dense: Optional[bool] = None
    # Restrict BBox refine to z direction only after dense inference (None uses Predictor default)
    refine_z_only: Optional[bool] = None
    enable_orientation_correction: bool = False
    orientation_solver_weights: Optional[Tuple[float, float, float]] = None
    # TensorRT (nnU-Net segmentation)
    vessel_trt_dir: Optional[str] = None
    vessel_trt_fp16: bool = True
    # ROI re-crop margins after dense inference
    refine_margin_voxels_z: Optional[int] = None
    refine_margin_voxels_xy: Optional[int] = None
    # ROI physical size (mm) for sparse-only Predictor. None uses Predictor default.
    sparse_roi_extent_mm: Optional[Union[float, Tuple[float, ...], List[float]]] = None

    # ROI classification
    roi_experiments: List[str] = field(default_factory=lambda: [DEFAULT_ROI_EXPERIMENT])
    roi_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # ROI checkpoint selection: "last" | "best"
    roi_ckpt: str = "last"
    # ROI inference TTA count
    roi_tta: int = 1
    # Optional override of ROI input size
    roi_input_size_override: Optional[Tuple[int, int, int]] = None
    # Override nnU-Net model dirs per ROI experiment (for Kaggle)
    # - None or empty list: unspecified
    # - Single value: broadcast to all experiments
    # - Multiple: aligned with ROI_EXPERIMENTS order
    roi_nnunet_model_dirs: Optional[List[str]] = None

    # Allow automatic apt install of dcm2niix / gdcmconv
    allow_apt_install: bool = False

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        v = v.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
        return default

    @classmethod
    def from_env(cls) -> "PipelineOptions":
        # Common
        debug = cls._env_flag("RSNA_DEBUG", default=cls._env_flag("DEBUG", False))
        rsna_stage = os.getenv("RSNA_STAGE", "roi").strip().lower()
        try:
            stage_placeholder = float(os.getenv("RSNA_STAGE_PLACEHOLDER", "0.1"))
        except Exception:
            stage_placeholder = 0.1
        try:
            fixed_prob = float(os.getenv("RSNA_FIXED_PROB", "0.5"))
        except Exception:
            fixed_prob = 0.5

        def _parse_error_fallback() -> Optional[List[float]]:
            """Parse per-class fallback probabilities."""
            raw = os.getenv("RSNA_ERROR_FALLBACK_PROBS", "").strip()
            if raw == "":
                return None
            tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
            if len(tokens) != len(LABEL_COLS):
                return None
            values: List[float] = []
            for token in tokens:
                try:
                    v = float(token)
                except Exception:
                    return None
                values.append(float(np.clip(v, 0.0, 1.0)))
            return values

        error_fallback = _parse_error_fallback()

        # Vessel segmentation
        vessel_model_dir = os.getenv("VESSEL_NNUNET_MODEL_DIR", cls().vessel_model_dir)
        sparse_model_env = os.getenv("VESSEL_NNUNET_SPARSE_MODEL_DIR", "").strip()
        vessel_sparse_model_dir = sparse_model_env if sparse_model_env else None
        folds_env = os.getenv("VESSEL_FOLDS", "0,1,2").replace(" ", "")
        fold_tokens = [token for token in folds_env.split(",") if token != ""]
        vessel_folds: List[int | str] = []
        if fold_tokens:
            for token in fold_tokens:
                lower = token.lower()
                if lower in {"all", "fold_all"}:
                    vessel_folds = ["all"]
                    break
                try:
                    vessel_folds.append(int(token))
                except Exception:
                    continue
        if not vessel_folds:
            vessel_folds = [0]
        additional_models_env = os.getenv("VESSEL_ADDITIONAL_DENSE_MODEL_DIRS", "").strip()
        vessel_additional_model_dirs = (
            [token for token in (t.strip() for t in additional_models_env.split(",")) if token]
            if additional_models_env
            else []
        )
        vessel_stage_limit = parse_vessel_stage_limit(os.getenv("RSNA_VESSEL_STAGE_LIMIT"))
        # Sparse ON/OFF priority: VESSEL_NO_SPARSE > VESSEL_USE_SPARSE (default True)
        vessel_use_sparse = True
        if cls._env_flag("VESSEL_NO_SPARSE", False):
            vessel_use_sparse = False
        else:
            vessel_use_sparse = cls._env_flag("VESSEL_USE_SPARSE", True)

        # Device selection (single/multiple)
        def _parse_vessel_devices() -> Optional[List[int]]:
            # Priority: VESSEL_DEVICES (comma-separated)
            raw_list = os.getenv("VESSEL_DEVICES", "").strip()
            if raw_list:
                toks = [t.strip() for t in re.split(r"[,\s]+", raw_list) if t.strip() != ""]
                out: List[int] = []
                for t in toks:
                    if re.fullmatch(r"-?\d+", t):
                        try:
                            out.append(int(t))
                        except Exception:
                            pass
                return out
            # Fallback: VESSEL_DEVICE (single; loosely accept "0,1")
            raw = os.getenv("VESSEL_DEVICE", "").strip()
            if not raw:
                raw = os.getenv("VESSEL_DEVICE_ID", "").strip()
            if not raw:
                return None
            key = raw.lower()
            # Do not return CPU/generic keywords as devices (use fallback device)
            if key in ("cpu", "cuda", "cuda:cpu"):
                return []
            # If comma-separated, interpret as multi-GPU
            if "," in key:
                toks = [t.strip() for t in key.split(",") if t.strip() != ""]
                out2: List[int] = []
                for t in toks:
                    if re.fullmatch(r"-?\d+", t):
                        try:
                            out2.append(int(t))
                        except Exception:
                            pass
                return out2
            # Single cuda:N or N
            m = re.fullmatch(r"cuda:(-?\d+)", key)
            if m:
                try:
                    return [int(m.group(1))]
                except Exception:
                    return None
            if re.fullmatch(r"-?\d+", key):
                try:
                    return [int(key)]
                except Exception:
                    return None
            return None

        enable_orientation_correction = cls._env_flag("VESSEL_ENABLE_ORIENTATION_CORRECTION", False)

        orientation_solver_weights: Optional[Tuple[float, float, float]] = None
        weights_env = os.getenv("VESSEL_ORIENTATION_WEIGHTS", "").strip()
        if weights_env:
            tokens = [t.strip() for t in weights_env.split(",") if t.strip() != ""]
            if len(tokens) == 3:
                try:
                    w0 = float(tokens[0])
                    w1 = float(tokens[1])
                    w2 = float(tokens[2])
                    orientation_solver_weights = (w0, w1, w2)
                except Exception:
                    orientation_solver_weights = None

        # Parse detailed vessel parameters
        def _get_env_float(name: str) -> Optional[float]:
            v = os.getenv(name, "").strip()
            if v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        def _get_env_int(name: str) -> Optional[int]:
            v = os.getenv(name, "").strip()
            if v == "":
                return None
            try:
                return int(float(v))
            except Exception:
                return None

        # Parse sparse ROI physical size (mm)
        def _parse_sparse_roi_extent_mm() -> Optional[Union[float, Tuple[float, ...], List[float]]]:
            raw = os.getenv("VESSEL_SPARSE_ROI_EXTENT_MM", "").strip()
            if raw == "":
                return None
            # Allowed separators: comma / space / x
            try:
                tokens = [t for t in re.split(r"[xX,\s]+", raw) if t != ""]
            except Exception:
                tokens = [raw]
            vals: List[float] = []
            for t in tokens:
                try:
                    vals.append(float(t))
                except Exception:
                    # On parse failure, treat as unspecified
                    return None
            if not vals:
                return None
            if len(vals) == 1:
                return float(vals[0])
            # Per-axis specification
            return tuple(vals)

        def _get_env_bool(name: str) -> Optional[bool]:
            v = os.getenv(name, "").strip().lower()
            if v == "":
                return None
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            return None

        # ROI
        default_roi_experiments = list(cls().roi_experiments)
        experiments_env = os.getenv("ROI_EXPERIMENTS", "").strip()
        if experiments_env:
            expr_tokens = [t.strip() for t in experiments_env.split(",") if t.strip() != ""]
        else:
            single_env = os.getenv("ROI_EXPERIMENT", "").strip()
            if single_env:
                expr_tokens = [t.strip() for t in single_env.split(",") if t.strip() != ""]
            else:
                expr_tokens = []

        if not expr_tokens:
            expr_tokens = default_roi_experiments

        roi_experiments: List[str] = []
        seen_expr = set()
        for token in expr_tokens:
            if token not in seen_expr:
                roi_experiments.append(token)
                seen_expr.add(token)

        roi_folds_env = os.getenv("ROI_FOLDS", "0,1,2,3,4").replace(" ", "")
        roi_folds = [int(x) for x in roi_folds_env.split(",") if x != ""]
        roi_ckpt = os.getenv("ROI_CKPT", "last").strip().lower()
        if roi_ckpt not in ("last", "best"):
            # Invalid value falls back to default "last"
            roi_ckpt = "last"

        def _parse_roi_tta() -> int:
            raw = os.getenv("ROI_TTA", "1").strip()
            if raw == "":
                return 1
            try:
                value = int(float(raw))
            except Exception:
                return 1
            normalized = _normalize_tta_count(value)
            if normalized not in _ALLOWED_TTA_COUNTS:
                normalized = 1
            return normalized

        roi_tta = _parse_roi_tta()

        # Override ROI nnU-Net directories (multi)
        def _parse_roi_nnunet_dirs() -> Optional[List[str]]:
            raw = os.getenv("ROI_NNUNET_MODEL_DIR", "").strip()
            if raw == "":
                return None
            toks = [t.strip() for t in raw.split(",") if t.strip() != ""]
            return toks if len(toks) > 0 else None

        # Override ROI input size and related fields
        def _parse_roi_input_size() -> Optional[Tuple[int, int, int]]:
            raw = os.getenv("ROI_INPUT_SIZE", "").strip()
            if raw == "":
                return None
            try:
                toks = [t for t in re.split(r"[xX,\s]+", raw) if t != ""]
                if len(toks) != 3:
                    return None
                sz = tuple(int(float(t)) for t in toks[:3])
                if any(int(v) <= 0 for v in sz):
                    return None
                return (int(sz[0]), int(sz[1]), int(sz[2]))
            except Exception:
                return None

        # Overrides for keep_ratio / spatial_transform / pad_multiple are not supported here

        # TensorRT settings
        trt_dir_env = os.getenv("VESSEL_TRT_DIR", "").strip()
        trt_dir = trt_dir_env if trt_dir_env else None
        trt_prec = os.getenv("VESSEL_TRT_PRECISION", "fp16").strip().lower()
        trt_fp16 = False if trt_prec == "fp32" else True
        sparse_roi_extent_mm = _parse_sparse_roi_extent_mm()

        return cls(
            debug=debug,
            rsna_stage=rsna_stage,
            stage_placeholder=float(np.clip(stage_placeholder, 0.0, 1.0)),
            only_ap=cls._env_flag("RSNA_ONLY_AP", False),
            only_locations=cls._env_flag("RSNA_ONLY_LOCATIONS", False),
            fixed_prob=float(np.clip(fixed_prob, 0.0, 1.0)),
            error_fallback_probs=(error_fallback if error_fallback is not None else [0.1] * len(LABEL_COLS)),
            abort_on_sparse_fail=cls._env_flag("VESSEL_ABORT_ON_SPARSE_FAIL", False),
            abort_min_all_dims_mm=_get_env_float("VESSEL_ABORT_MIN_ALL_DIMS_MM"),
            abort_on_small_roi=cls._env_flag("VESSEL_ABORT_ON_SMALL_ROI", False),
            min_roi_voxels=_get_env_int("VESSEL_MIN_ROI_VOXELS"),
            abort_on_low_union=cls._env_flag("VESSEL_ABORT_ON_LOW_UNION", False),
            min_union_sum=_get_env_int("VESSEL_MIN_UNION_SUM"),
            vessel_model_dir=vessel_model_dir,
            vessel_sparse_model_dir=vessel_sparse_model_dir,
            vessel_additional_model_dirs=vessel_additional_model_dirs,
            vessel_folds=vessel_folds,
            vessel_stage_limit=vessel_stage_limit,
            vessel_use_sparse=vessel_use_sparse,
            window_count_threshold=_get_env_int("VESSEL_WINDOW_COUNT_THRESHOLD"),
            sparse_downscale_factor=_get_env_float("VESSEL_SPARSE_DOWNSCALE"),
            sparse_overlap=_get_env_float("VESSEL_SPARSE_OVERLAP"),
            detection_threshold=_get_env_float("VESSEL_DETECTION_THRESHOLD"),
            sparse_bbox_margin_voxels=_get_env_int("VESSEL_SPARSE_MARGIN_VOXELS"),
            dense_overlap=_get_env_float("VESSEL_DENSE_OVERLAP"),
            use_gaussian=_get_env_bool("VESSEL_USE_GAUSSIAN"),
            perform_everything_on_device=_get_env_bool("VESSEL_PERFORM_ON_DEVICE"),
            limit_si_extent=_get_env_bool("VESSEL_LIMIT_SI_EXTENT"),
            max_si_extent_mm=_get_env_float("VESSEL_MAX_SI_EXTENT_MM"),
            min_si_extent_mm=_get_env_float("VESSEL_MIN_SI_EXTENT_MM"),
            min_xy_extent_mm=_get_env_float("VESSEL_MIN_XY_EXTENT_MM"),
            si_axis=_get_env_int("VESSEL_SI_AXIS"),
            refine_roi_after_dense=_get_env_bool("VESSEL_REFINE_ROI_AFTER_DENSE"),
            # z-only refine enable/disable (None -> Predictor default=True)
            refine_z_only=_get_env_bool("VESSEL_REFINE_Z_ONLY"),
            enable_orientation_correction=enable_orientation_correction,
            orientation_solver_weights=orientation_solver_weights,
            vessel_trt_dir=trt_dir,
            vessel_trt_fp16=trt_fp16,
            vessel_devices=_parse_vessel_devices(),
            # ROI re-crop margins (use Predictor default if unset)
            refine_margin_voxels_z=_get_env_int("VESSEL_REFINE_MARGIN_Z"),
            refine_margin_voxels_xy=_get_env_int("VESSEL_REFINE_MARGIN_XY"),
            # ROI physical extent (mm) for sparse-only Predictor (DBSCAN case)
            sparse_roi_extent_mm=sparse_roi_extent_mm,
            roi_experiments=roi_experiments,
            roi_folds=roi_folds,
            roi_ckpt=roi_ckpt,
            roi_tta=roi_tta,
            roi_input_size_override=_parse_roi_input_size(),
            roi_nnunet_model_dirs=_parse_roi_nnunet_dirs(),
            allow_apt_install=cls._env_flag("ALLOW_APT_INSTALL", False),
        )


def clear_caches(target: str = "roi") -> None:
    """Compatibility API: clear global pipeline instance caches."""
    p = _get_pipeline(create_if_none=False)
    if p is None:
        return
    p.clear_caches(target)


# ===== Helper: dcm2niix check and temporary conversion =====
def _ensure_dcm2niix_available(allow_apt_install: bool = False) -> None:
    """Ensure dcm2niix / gdcmconv availability (first time only)."""
    global _dcm_tools_checked
    if _dcm_tools_checked:
        return
    try:
        which_or_die("dcm2niix")
        which_or_die("gdcmconv")
        _dcm_tools_checked = True
        return
    except FileNotFoundError:
        if not allow_apt_install:
            raise
        import subprocess

        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "dcm2niix", "libgdcm-tools"], check=True)
        # Re-check
        which_or_die("dcm2niix")
        which_or_die("gdcmconv")
        _dcm_tools_checked = True


def convert_dicom_series_to_nifti(series_path: str, out_dir: Path, debug: bool = False) -> str:
    """Convert DICOM series to NIfTI under out_dir and return the path."""
    logger = logging.getLogger(__name__)
    tmp_root = Path(tempfile.gettempdir())
    nifti_path, _, logs, errors = convert_dicom_to_nifti(
        dir_path=Path(series_path),
        out_dir=out_dir,
        tmp_root=tmp_root,
        use_majority_size=True,
        copy_mode="auto",
        file_exts=(".dcm", ""),
        dcm2niix_flags=("-z", "n", "-b", "y", "-i", "n", "-f", "%s"),
        gdcm_first=False,
        use_slice_spacing_filter=True,
        slice_spacing_tolerance=2.0,
        convert_to_ras=False,
    )
    if debug:
        try:
            if logs:
                logger.debug("[DEBUG][STEP1] DICOM→NIfTI conversion logs:")
                for line in logs:
                    logger.debug("  %s", line)
            if errors:
                logger.debug("[DEBUG][STEP1] DICOM→NIfTI errors:")
                for line in errors:
                    logger.debug("  %s", line)
        except Exception:
            pass
    if nifti_path is None:
        raise RuntimeError("Failed to convert to NIfTI: " + ("\n".join(errors) if errors else "unknown error"))
    return str(nifti_path)


# ===== Helper: nnU-Net vessel segmentation predictor =====
class RsnaRoiPipeline:
    """RSNA ROI inference pipeline (with internal caches and options)."""

    def __init__(self, opts: PipelineOptions):
        # Options
        self.opts = opts
        # Vessel segmentation predictor
        self._vessel_predictor: Optional[VesselSegmentationPredictor] = None
        # NIfTI IO handler
        self._nifti_io_handler: Optional[SimpleITKIOWithReorient] = None
        # ROI classification models (fold ensemble)
        self._roi_models: Optional[List[Tuple[AneurysmVesselSegROILitModule, torch.device]]] = None
        self._roi_input_size: Optional[Tuple[int, int, int]] = None
        self._roi_keep_ratio: Optional[str] = None
        self._roi_spatial_transform: Optional[str] = None
        self._roi_pad_multiple: Optional[int] = None
        # Counters of calls and errors
        self._predict_error_count: int = 0
        self._predict_call_count: int = 0

    # ---- Cache operations ----
    def clear_caches(self, target: str = "roi") -> None:
        """Clear internal caches."""
        t = (target or "roi").lower().strip()
        if t in ("roi", "all"):
            self._roi_models = None
            self._roi_input_size = None
            self._roi_keep_ratio = None
            self._roi_spatial_transform = None
            self._roi_pad_multiple = None
        if t in ("vessel", "all"):
            self._vessel_predictor = None
            self._nifti_io_handler = None

    # ---- Tools verification ----
    def ensure_tools(self) -> None:
        _ensure_dcm2niix_available(allow_apt_install=self.opts.allow_apt_install)

    # ---- DICOM→NIfTI ----
    def dicom_to_nifti(self, series_path: str, out_dir: Path) -> str:
        return convert_dicom_series_to_nifti(series_path, out_dir=out_dir, debug=self.opts.debug)

    def _get_nifti_io_handler(self) -> SimpleITKIOWithReorient:
        if self._nifti_io_handler is None:
            self._nifti_io_handler = SimpleITKIOWithReorient()
        return self._nifti_io_handler

    def load_nifti_volume(self, nifti_path: str) -> Tuple[np.ndarray, Dict]:
        """Load NIfTI via SimpleITK and optionally fix missing metadata."""

        io_handler = self._get_nifti_io_handler()
        image, properties = io_handler.read_images([nifti_path], orientation="RAS")

        # Toggle load-time fix via env (default OFF)
        try:
            v = os.getenv("RSNA_FIX_MF_AT_LOAD", "0").strip().lower()
            fix_enabled = v in ("1", "true", "yes", "on")
        except Exception:
            fix_enabled = False

        if not fix_enabled:
            return image, properties

        # Obtain multi-frame MR info from JSON and adjust spacing under aggressive conditions
        try:
            meta_path = Path(nifti_path).parent / "series_metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            is_mf = bool(meta.get("IsMultiframeMR", False))

            # Quick check for suspicious current spacing (nnU-Net convention: z,y,x)
            def _bad_spacing(sp) -> bool:
                try:
                    if not isinstance(sp, (list, tuple)) or len(sp) < 3:
                        return True
                    z, y, x = float(sp[0]), float(sp[1]), float(sp[2])
                    if not np.isfinite(z) or not np.isfinite(y) or not np.isfinite(x):
                        return True
                    if z <= 0 or y <= 0 or x <= 0:
                        return True
                    # Extreme values (implausible for MR)
                    if z > 20 or y > 5 or x > 5:
                        return True
                    # All 1.0 suggests dcm2niix default fallback
                    if abs(z - 1.0) < 1e-3 and abs(y - 1.0) < 1e-3 and abs(x - 1.0) < 1e-3:
                        return True
                    return False
                except Exception:
                    return True

            cur_spacing = properties.get("spacing", None)
            need_fix = bool(is_mf and _bad_spacing(cur_spacing))

            if need_fix:
                # Decide x,y: prefer JSON PixelSpacing ([x, y]); else 0.5, 0.5
                sx = sy = 0.5
                ps = meta.get("PixelSpacing")
                if isinstance(ps, (list, tuple)) and len(ps) >= 2:
                    try:
                        sx = float(ps[0])
                        sy = float(ps[1])
                        if not (np.isfinite(sx) and np.isfinite(sy)) or sx <= 0 or sy <= 0:
                            sx, sy = 0.5, 0.5
                    except Exception:
                        sx, sy = 0.5, 0.5

                # Decide z: slice-count heuristic: <=80 → 5.0 mm, else → 0.55 mm
                n: Optional[int] = None
                approx = meta.get("ApproxSliceCount")
                try:
                    if isinstance(approx, (int, float)) and approx > 0:
                        n = int(approx)
                    else:
                        # Get Z dimension from loaded array (image: C,Z,Y,X)
                        n = int(image.shape[1]) if image.ndim >= 4 else None
                except Exception:
                    n = None
                dz = 5.0 if (n is not None and n <= 80) else 0.55

                # Rewrite properties (nnU-Net: z,y,x / SITK: x,y,z)
                properties["spacing"] = [float(dz), float(sy), float(sx)]
                sitk_stuff = properties.get("sitk_stuff", {}) or {}
                sitk_stuff["spacing"] = (float(sx), float(sy), float(dz))
                properties["sitk_stuff"] = sitk_stuff

                # Debug markers
                properties["mf_spacing_fix"] = True
                properties["mf_spacing_fix_reason"] = {
                    "IsMultiframeMR": bool(is_mf),
                    "cur_spacing": cur_spacing,
                }
        except Exception:
            # Continue inference even if reading/fixing fails
            pass

        return image, properties

    def _get_vessel_predictor(self) -> VesselSegmentationPredictor:
        if self._vessel_predictor is not None:
            return self._vessel_predictor
        # Orientation correction expects sparse-only model (4 classes).
        # If enabled without VESSEL_NNUNET_SPARSE_MODEL_DIR, disable and warn.
        effective_orientation = bool(self.opts.enable_orientation_correction)
        if effective_orientation and not self.opts.vessel_sparse_model_dir:
            logging.getLogger(__name__).warning(
                "[WARN] VESSEL_ENABLE_ORIENTATION_CORRECTION enabled but sparse-only model is not specified. Disabling."
            )
            effective_orientation = False

        vp = VesselSegmentationPredictor(
            model_path=self.opts.vessel_model_dir,
            sparse_model_path=self.opts.vessel_sparse_model_dir,
            folds=tuple(self.opts.vessel_folds),
            additional_dense_model_paths=tuple(self.opts.vessel_additional_model_dirs),
            # Fallback device when devices is empty/unset
            device=str(_device),
            use_sparse_search=bool(self.opts.vessel_use_sparse),
            use_mirroring=False,
            verbose=bool(self.opts.debug),
            # Multi-GPU via env (None/empty list forces CPU)
            devices=(tuple(self.opts.vessel_devices) if self.opts.vessel_devices is not None else None),
            window_count_threshold=self.opts.window_count_threshold,
            sparse_downscale_factor=self.opts.sparse_downscale_factor,
            sparse_overlap=self.opts.sparse_overlap,
            detection_threshold=self.opts.detection_threshold,
            sparse_bbox_margin_voxels=self.opts.sparse_bbox_margin_voxels,
            dense_overlap=self.opts.dense_overlap,
            use_gaussian=self.opts.use_gaussian,
            perform_everything_on_device=self.opts.perform_everything_on_device,
            limit_si_extent=self.opts.limit_si_extent,
            max_si_extent_mm=self.opts.max_si_extent_mm,
            min_si_extent_mm=self.opts.min_si_extent_mm,
            min_xy_extent_mm=self.opts.min_xy_extent_mm,
            si_axis=self.opts.si_axis,
            refine_roi_after_dense=self.opts.refine_roi_after_dense,
            # z-only refine (None -> Predictor default=True)
            refine_z_only=self.opts.refine_z_only,
            # ROI re-crop margins after dense inference (configurable via env)
            refine_margin_voxels_z=self.opts.refine_margin_voxels_z,
            refine_margin_voxels_xy=self.opts.refine_margin_voxels_xy,
            # ROI physical size (mm) for sparse-only Predictor (DBSCAN)
            sparse_roi_extent_mm=self.opts.sparse_roi_extent_mm,
            enable_orientation_correction=effective_orientation,
            orientation_solver_weights=self.opts.orientation_solver_weights,
            trt_dir=self.opts.vessel_trt_dir,
            trt_fp16=self.opts.vessel_trt_fp16,
        )
        self._vessel_predictor = vp
        return vp

    # ---- Segmentation with OOM fallback ----
    def vessel_predict(
        self,
        image: np.ndarray,
        properties: Dict,
    ) -> VesselSegmentationOutput:
        vessel_pred = self._get_vessel_predictor()
        orig_perform_on_dev = None
        _clear_cuda_cache()
        try:
            if hasattr(vessel_pred, "predictors") and len(vessel_pred.predictors) > 0:
                orig_perform_on_dev = bool(vessel_pred.predictors[0].perform_everything_on_device)
        except Exception:
            orig_perform_on_dev = None

        stage_limit = self.opts.vessel_stage_limit

        def _run_with_volume(img: np.ndarray, props: Dict) -> VesselSegmentationOutput:
            return vessel_pred.predict_single_volume_with_info(
                image=img,
                properties=props,
                return_probabilities=False,
                stage_limit=stage_limit,
            )

        try:
            result = _run_with_volume(image, properties)
        except RuntimeError as e:
            msg = str(e).lower()
            is_oom = "out of memory" in msg  # or ("cuda" in msg and "memory" in msg)
            if not is_oom:
                raise RuntimeError(e)
            if self.opts.debug:
                logging.getLogger(__name__).warning(
                    "[WARN] VesselSeg: CUDA OOM detected. Retrying with memory-friendly settings."
                )
            del e
            _clear_cuda_cache()
            try:
                vessel_pred.set_perform_everything_on_device(False)
                result = _run_with_volume(image, properties)
            finally:
                if orig_perform_on_dev is not None:
                    vessel_pred.set_perform_everything_on_device(bool(orig_perform_on_dev))
        return result

    # ---- ROI model loading ----
    def _available_cuda_devices(self) -> List[torch.device]:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return [torch.device(f"cuda:{i}") for i in range(n)] if n > 0 else [torch.device("cuda")]
        return [torch.device("cpu")]

    def _assign_devices_to_folds(self, folds: List[int]) -> List[torch.device]:
        devs = self._available_cuda_devices()
        if len(devs) == 1:
            return [devs[0] for _ in folds]
        out: List[torch.device] = []
        for i, _ in enumerate(folds):
            out.append(devs[i % len(devs)])
        return out

    def _find_ckpt_path(self, log_dir: Path) -> Optional[Path]:
        """Find fold checkpoint path.

        - Selected via env `ROI_CKPT` (default: "last").
        - "best": choose epoch_*.ckpt with the largest epoch.
        - "last": choose last.ckpt.
        Return None if not found.
        """
        mode = (self.opts.roi_ckpt or "last").strip().lower()
        if mode == "best":
            # Choose epoch_*.ckpt with the largest epoch number
            epoch_ckpts = list(log_dir.glob("epoch_*.ckpt"))
            if not epoch_ckpts:
                return None

            def _epoch_num(p: Path) -> int:
                # Extract epoch number from filename (e.g., epoch_014.ckpt -> 14)
                m = re.search(r"epoch[_=]?(\d+)", p.name)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        return -1
                return -1

            return max(epoch_ckpts, key=_epoch_num)
        # Default: last.ckpt
        last_ckpt = log_dir / "last.ckpt"
        return last_ckpt if last_ckpt.exists() else None

    def load_roi_models(
        self,
    ) -> Tuple[
        List[Tuple[AneurysmVesselSegROILitModule, torch.device]],
        Tuple[int, int, int],
        str,
        str,
        Optional[int],
    ]:
        if (
            self._roi_models is not None
            and self._roi_input_size is not None
            and self._roi_keep_ratio is not None
            and self._roi_spatial_transform is not None
        ):
            return (
                self._roi_models,
                self._roi_input_size,
                self._roi_keep_ratio,
                self._roi_spatial_transform,
                self._roi_pad_multiple,
            )

        experiments = list(self.opts.roi_experiments)
        folds = list(self.opts.roi_folds)

        if not experiments:
            raise RuntimeError("ROI experiments are not specified. Check environment variables.")
        if not folds:
            raise RuntimeError("ROI folds are not specified. Check environment variables.")

        model_specs: List[Tuple[str, int]] = [(exp, fold) for exp in experiments for fold in folds]
        fold_devices = self._assign_devices_to_folds([spec[1] for spec in model_specs])

        models: List[Tuple[AneurysmVesselSegROILitModule, torch.device]] = []
        cfg_cache: Dict[str, object] = {}
        missing_specs: List[Tuple[str, int]] = []  # collect missing experiment-fold specs

        for (experiment, fold), dev in zip(model_specs, fold_devices):
            try:
                if experiment not in cfg_cache:
                    config_dir = os.getenv("CONFIG_DIR", "/workspace/configs")
                    cfg_cache[experiment] = load_experiment_config(experiment, config_dir=config_dir)
                cfg_template = cfg_cache[experiment]
                cfg = cfg_template.copy() if hasattr(cfg_template, "copy") else cfg_template
                cfg.data.fold = int(fold)
                # Override ROI input size (via env)
                try:
                    if self.opts.roi_input_size_override is not None:
                        ov = tuple(int(x) for x in self.opts.roi_input_size_override)
                        if len(ov) == 3 and all(v > 0 for v in ov):
                            cfg.data.input_size = [int(ov[0]), int(ov[1]), int(ov[2])]
                except Exception:
                # Even if override fails, continue; later checks will catch issues
                    pass
                if cfg.model.compile:
                    rm_orig_mod = True
                    cfg.model.compile = False
                else:
                    rm_orig_mod = False

                if os.getenv("RUN_MODE", "local") == "kaggle":
                    # In Kaggle, allow per-experiment override of ROI backbone nnU-Net paths
                    # - If multiple ROI_NNUNET_MODEL_DIR are set, align with ROI_EXPERIMENTS order
                    # - If single, broadcast to all experiments
                    # - If unset, fall back to VESSEL_NNUNET_MODEL_DIR
                    override_dirs = self.opts.roi_nnunet_model_dirs or []
                    if len(override_dirs) == 0:
                        chosen_dir = self.opts.vessel_model_dir
                    elif len(override_dirs) == 1:
                        chosen_dir = override_dirs[0]
                    else:
                        try:
                            idx = self.opts.roi_experiments.index(experiment)
                        except Exception:
                            idx = 0
                        # If out of range, use the last one (handle minor redundancy)
                        if idx >= len(override_dirs):
                            if self.opts.debug:
                                logging.getLogger(__name__).warning(
                                    "[WARN] ROI_NNUNET_MODEL_DIR count (%d) < ROI_EXPERIMENTS (%d). Using last entry (exp=%s)",
                                    len(override_dirs),
                                    len(self.opts.roi_experiments),
                                    experiment,
                                )
                            chosen_dir = override_dirs[-1]
                        else:
                            chosen_dir = override_dirs[idx]

                    cfg.model.net.nnunet_model_dir = chosen_dir
                    if self.opts.debug:
                        logging.getLogger(__name__).debug(
                            "[DEBUG] ROI nnUNet dir (exp=%s): %s",
                            experiment,
                            cfg.model.net.nnunet_model_dir,
                        )

                model: AneurysmVesselSegROILitModule = hydra.utils.instantiate(cfg.model)

                ckpt_dir = Path(cfg.log_dir) / "checkpoints" / f"fold{fold}"
                ckpt_path = self._find_ckpt_path(ckpt_dir)
                if ckpt_path is None:
                    logging.getLogger(__name__).warning(
                        f"[WARN] checkpoint not found (exp={experiment}, fold={fold}, mode={self.opts.roi_ckpt}): {ckpt_dir}"
                    )
                    missing_specs.append((experiment, int(fold)))
                    continue

                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = state.get("state_dict", state)
                if rm_orig_mod:
                    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if len(unexpected) > 0:
                    logging.getLogger(__name__).warning(f"[WARN] Unexpected keys: {len(unexpected)}")
                if len(missing) > 0:
                    logging.getLogger(__name__).warning(f"[WARN] Missing keys: {len(missing)}")

                _prune_unused_roi_variants(model)
                model = model.to(dev).eval()
                models.append((model, dev))

                sz = getattr(cfg.data, "input_size", None)
                if sz is None or len(sz) != 3:
                    raise RuntimeError("cfg.data.input_size missing or invalid (expected length-3 array)")
                current_size = (int(sz[0]), int(sz[1]), int(sz[2]))
                if self._roi_input_size is None:
                    self._roi_input_size = current_size
                elif self._roi_input_size != current_size:
                    raise RuntimeError(
                        f"Inconsistent ROI input_size across experiments: {self._roi_input_size} vs {current_size} (exp={experiment})"
                    )

                kr = getattr(cfg.data, "keep_ratio", None)
                if kr is None:
                    raise RuntimeError("cfg.data.keep_ratio not found")
                if self._roi_keep_ratio is None:
                    self._roi_keep_ratio = kr
                elif self._roi_keep_ratio != kr:
                    raise RuntimeError(
                        f"Inconsistent ROI keep_ratio across experiments: {self._roi_keep_ratio} vs {kr} (exp={experiment})"
                    )

                spatial_tf = getattr(cfg.data, "spatial_transform", "resize")
                if spatial_tf is None:
                    spatial_tf = "resize"
                spatial_tf = str(spatial_tf)
                if self._roi_spatial_transform is None:
                    self._roi_spatial_transform = spatial_tf
                elif self._roi_spatial_transform != spatial_tf:
                    raise RuntimeError(
                        "Inconsistent ROI spatial_transform across experiments: "
                        f"{self._roi_spatial_transform} vs {spatial_tf} (exp={experiment})"
                    )

                pad_multiple_attr = getattr(cfg.data, "pad_multiple", None)
                pad_multiple: Optional[int]
                if pad_multiple_attr is not None:
                    pad_multiple = int(pad_multiple_attr)
                    if pad_multiple <= 0:
                        raise RuntimeError(
                            f"cfg.data.pad_multiple must be a positive integer: {pad_multiple_attr}"
                        )
                else:
                    pad_multiple = None

                if self._roi_spatial_transform == "pad":
                    if pad_multiple is None:
                        pad_multiple = 32
                    if self._roi_pad_multiple is None:
                        self._roi_pad_multiple = pad_multiple
                    elif self._roi_pad_multiple != pad_multiple:
                        raise RuntimeError(
                            "Inconsistent ROI pad_multiple across experiments: "
                            f"{self._roi_pad_multiple} vs {pad_multiple} (exp={experiment})"
                        )
                else:
                    pad_multiple = None
                    if self._roi_pad_multiple is None:
                        self._roi_pad_multiple = None
                    elif self._roi_pad_multiple is not None:
                        raise RuntimeError(
                            "Inconsistent ROI pad_multiple across experiments: "
                            f"Expected: None, actual: {self._roi_pad_multiple} (exp={experiment})"
                        )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"[WARN] ROI model load failed (exp={experiment}, fold={fold}): {e}"
                )
                missing_specs.append((experiment, int(fold)))
                continue

        total_expected = len(model_specs)
        if not models:
            raise RuntimeError(
                "Failed to load ROI model. Check checkpoint path / environment variables."
            )

        if len(models) != total_expected:
            miss_str = (
                ",".join(f"{exp}:fold{fold}" for exp, fold in sorted(set(missing_specs)))
                if missing_specs
                else "unknown"
            )
            raise RuntimeError(
                f"Some experiment/fold checkpoints are missing (mode={self.opts.roi_ckpt}). missing=[{miss_str}]"
            )

        if self._roi_input_size is None:
            raise RuntimeError("Failed to obtain ROI input_size")
        if self._roi_keep_ratio is None:
            raise RuntimeError("Failed to obtain ROI keep_ratio")
        if self._roi_spatial_transform is None:
            raise RuntimeError("Failed to obtain ROI spatial_transform")
        self._roi_models = models

        logging.getLogger(__name__).info(
            f"ROI model load success: {len(models)} models (experiments={experiments}, folds={folds})"
        )
        logging.getLogger(__name__).info(f"ROI input_size: {self._roi_input_size}")
        logging.getLogger(__name__).info(f"ROI keep_ratio: {self._roi_keep_ratio}")
        logging.getLogger(__name__).info(f"ROI spatial_transform: {self._roi_spatial_transform}")
        if self._roi_pad_multiple is not None:
            logging.getLogger(__name__).info(f"ROI pad_multiple: {self._roi_pad_multiple}")

        return (
            self._roi_models,
            self._roi_input_size,
            self._roi_keep_ratio,
            self._roi_spatial_transform,
            self._roi_pad_multiple,
        )


def _convert_segmentation_to_detection_label(seg_label: torch.Tensor) -> torch.Tensor:
    """Convert nnU-Net label order to detection task order."""
    if seg_label.ndim == 3:
        seg_label = seg_label.unsqueeze(0)
    if seg_label.ndim != 4 or seg_label.shape[0] != 1:
        raise ValueError(f"Invalid label map shape: {tuple(seg_label.shape)}")
    mapping = _SEG_TO_DET_TORCH.to(seg_label.device)
    return mapping[seg_label.long()]


def _prepare_inputs_for_roi(
    roi_t: torch.Tensor,
    seg_t: torch.Tensor,
    input_size: Tuple[int, int, int],
    keep_ratio: str,
    spatial_transform: str,
    pad_multiple: Optional[int],
    extra_segs: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare ROI inputs with the same preprocessing as training."""
    roi_tensor = torch.as_tensor(roi_t)
    seg_tensor = torch.as_tensor(seg_t)

    ref_device = roi_tensor.device
    seg_tensor = seg_tensor.to(ref_device)
    if extra_segs is not None:
        for i in range(len(extra_segs)):
            extra_segs[i] = extra_segs[i].to(ref_device)

    if roi_tensor.ndim == 3:
        roi_tensor = roi_tensor.unsqueeze(0)
    if roi_tensor.ndim != 4:
        raise ValueError(f"Invalid ROI shape: {tuple(roi_tensor.shape)}")
    if seg_tensor.ndim == 3:
        seg_tensor = seg_tensor.unsqueeze(0)
    if seg_tensor.ndim != 4 or seg_tensor.shape[0] != 1:
        raise ValueError(f"Invalid seg label shape: {tuple(seg_tensor.shape)}")

    roi_tensor = roi_tensor.to(torch.float32)
    det_label = _convert_segmentation_to_detection_label(seg_tensor)
    det_label = det_label.to(torch.float32)

    num_extra = int(len(extra_segs)) if extra_segs is not None else 0
    val_tf = get_val_transforms(
        tuple(int(x) for x in input_size),
        keep_ratio=keep_ratio,
        spatial_transform=spatial_transform,
        pad_multiple=pad_multiple if pad_multiple is not None else 32,
        include_extra_seg=(num_extra > 0),
        num_extra_seg=num_extra if num_extra > 0 else None,
    )
    data_in = {
        "image": roi_tensor,
        "vessel_label": det_label,
    }
    # Insert additional segs (detection-class-order labels)
    if num_extra > 0:
        for i, seg_extra in enumerate(extra_segs or []):
            seg_extra_t = torch.as_tensor(seg_extra)
            if seg_extra_t.ndim == 3:
                seg_extra_t = seg_extra_t.unsqueeze(0)
            if seg_extra_t.ndim != 4 or seg_extra_t.shape[0] != 1:
                raise ValueError(f"Invalid additional seg shape: {tuple(seg_extra_t.shape)}")
            det_label_e = _convert_segmentation_to_detection_label(seg_extra_t).to(torch.float32)
            key = "extra_vessel_label" if i == 0 else f"extra_vessel_label_{i}"
            data_in[key] = det_label_e
    data_out = val_tf(data_in)

    roi_resized = data_out["image"]
    vessel_label_resized = data_out["vessel_label"]

    if roi_resized.ndim == 3:
        roi_resized = roi_resized.unsqueeze(0)
    if vessel_label_resized.ndim == 3:
        vessel_label_resized = vessel_label_resized.unsqueeze(0)

    vessel_label_resized = vessel_label_resized.round().clamp_(0, 13).long()
    label_wo_channel = vessel_label_resized.squeeze(0)
    one_hot = F.one_hot(label_wo_channel, num_classes=14).permute(3, 0, 1, 2).contiguous()
    vessel_masks = one_hot[1:].to(torch.float32)
    vessel_union = (label_wo_channel > 0).unsqueeze(0).to(torch.float32)

    # Binarize and merge additional segs
    extra_masks_cat: Optional[torch.Tensor] = None
    if num_extra > 0:
        extra_masks_list: List[torch.Tensor] = []
        extra_union_any: Optional[torch.Tensor] = None
        for i in range(num_extra):
            key = "extra_vessel_label" if i == 0 else f"extra_vessel_label_{i}"
            if key not in data_out:
                continue
            lbl = data_out[key]
            if lbl.ndim == 3:
                lbl = lbl.unsqueeze(0)
            lbl = lbl.round().clamp_(0, 13).long()
            lbl_wo = lbl.squeeze(0)
            oh = F.one_hot(lbl_wo, num_classes=14).permute(3, 0, 1, 2).contiguous()
            mask13 = oh[1:].to(torch.float32)
            extra_masks_list.append(mask13)
            u = (lbl_wo > 0).unsqueeze(0).to(torch.float32)
            extra_union_any = u if extra_union_any is None else torch.maximum(extra_union_any, u)
        if extra_masks_list:
            extra_masks_cat = torch.cat(extra_masks_list, dim=0).contiguous()
            if extra_union_any is not None:
                vessel_union = torch.maximum(vessel_union, extra_union_any)

    roi_batch = roi_resized.unsqueeze(0)
    vessel_masks_batch = vessel_masks.unsqueeze(0)
    vessel_union_batch = vessel_union.unsqueeze(0)

    extra_batch = extra_masks_cat.unsqueeze(0) if extra_masks_cat is not None else None
    return roi_batch, vessel_masks_batch, vessel_union_batch, extra_batch


# ===== Main inference =====
def _predict_probs_with_pipeline(
    p: RsnaRoiPipeline, series_path: str
) -> Tuple[np.ndarray, VesselSegmentationOutput]:
    """Probability inference using the pipeline object."""
    p.ensure_tools()
    with tempfile.TemporaryDirectory(prefix="rsna_niix_") as td:
        tmp_dir = Path(td)
        with time_section("STEP1 DICOM→NIfTI conversion", enabled=p.opts.debug):
            nifti_path = p.dicom_to_nifti(series_path, out_dir=tmp_dir)

        with time_section("STEP2 NIfTI load", enabled=p.opts.debug):
            image, properties = p.load_nifti_volume(nifti_path)

        with time_section("STEP3 VesselSeg inference + ROI extraction", enabled=p.opts.debug):
            vessel_out = p.vessel_predict(image, properties)

        # Early abort on sparse-search failure (full-extent ROI fallback)
        # - Condition: VESSEL_ABORT_ON_SPARSE_FAIL=1 and VESSEL_NNUNET_SPARSE_MODEL_DIR is set
        # - Decision: For PREPROCESS.data_shape, SPARSE_SEARCH.roi_bbox equals [0:dim)×3
        # - Physical size guard: if VESSEL_ABORT_MIN_ALL_DIMS_MM is set and all dims computed from
        #   SPARSE_SEARCH.network_shape/network_spacing are below threshold, treat as small-volume and do not abort
        if p.opts.abort_on_sparse_fail and p.opts.vessel_sparse_model_dir:
            try:
                pre_ex = vessel_out.extras.get(VesselSegmentationStage.PREPROCESS.value)  # type: ignore[attr-defined]
                sp_ex = vessel_out.extras.get(VesselSegmentationStage.SPARSE_SEARCH.value)  # type: ignore[attr-defined]
                if isinstance(pre_ex, dict) and isinstance(sp_ex, dict):
                    # If sparse search was explicitly skipped, do not apply abort logic
                    if bool(sp_ex.get("skipped_sparse", False)):
                        pass
                    else:
                        data_shape = tuple(int(x) for x in pre_ex.get("data_shape", ()))
                        roi_bbox = sp_ex.get("roi_bbox")
                        if (
                            isinstance(data_shape, tuple)
                            and len(data_shape) >= 4
                            and isinstance(roi_bbox, (list, tuple))
                            and len(roi_bbox) == 3
                            and all(isinstance(r, (list, tuple)) and len(r) == 2 for r in roi_bbox)
                        ):
                            spatial = tuple(int(x) for x in data_shape[1:4])
                            is_full = True
                            for i, (start, stop) in enumerate(roi_bbox):
                                if int(start) != 0 or int(stop) != int(spatial[i]):
                                    is_full = False
                                    break
                            if is_full:
                                # Physical size (mm) guard (optional)
                                thr_mm = p.opts.abort_min_all_dims_mm
                                if thr_mm is not None:
                                    try:
                                        net_shape = sp_ex.get("network_shape")
                                        net_spacing = sp_ex.get("network_spacing")
                                        if (
                                            isinstance(net_shape, (list, tuple))
                                            and isinstance(net_spacing, (list, tuple))
                                            and len(net_shape) >= 3
                                            and len(net_spacing) >= 3
                                        ):
                                            # Evaluate with minimum dimension count even if inconsistent
                                            dims_mm = [
                                                float(net_shape[i]) * float(net_spacing[i])
                                                for i in range(min(3, len(net_shape), len(net_spacing)))
                                            ]
                                            all_small = all(d < float(thr_mm) for d in dims_mm)
                                            if all_small:
                                                # Full-extent due to small volume → do not abort
                                                if p.opts.debug:
                                                    logging.getLogger(__name__).info(
                                                        "[DEBUG] Full-extent ROI but all dims < %.1fmm (dims=%s). Not aborting.",
                                                        float(thr_mm),
                                                        ",".join(f"{d:.1f}" for d in dims_mm),
                                                    )
                                                pass
                                            else:
                                                raise RuntimeError(
                                                    "Sparse search returned full-extent ROI (fallback). Aborting per VESSEL_ABORT_ON_SPARSE_FAIL."
                                                )
                                        else:
                                            # If shape/spacing is unavailable, abort as before
                                            raise RuntimeError(
                                                "Sparse search returned full-extent ROI (fallback). Aborting per VESSEL_ABORT_ON_SPARSE_FAIL."
                                            )
                                    except RuntimeError:
                                        # Propagate abort
                                        raise
                                    except Exception:
                                        # Abort on unexpected exceptions (fail-safe)
                                        raise RuntimeError(
                                            "Sparse search returned full-extent ROI (fallback). Aborting per VESSEL_ABORT_ON_SPARSE_FAIL."
                                        )
                                else:
                                    # Abort as before when guard is disabled
                                    raise RuntimeError(
                                        "Sparse search returned full-extent ROI (fallback). Aborting per VESSEL_ABORT_ON_SPARSE_FAIL."
                                    )
            except Exception:
                # Let the caller handle exceptions (propagate)
                raise

        if not vessel_out.is_full:
            _clear_cuda_cache()
            if p.opts.debug:
                logging.getLogger(__name__).info(
                    "[DEBUG] VesselSeg processing stopped by stage limit: limit=%s, completed=%s, history=%s",
                    vessel_out.stage_limit.value,
                    vessel_out.completed_stage.value,
                    "->".join(stage.value for stage in vessel_out.stage_history),
                )
            placeholder = np.full((len(LABEL_COLS),), p.opts.stage_placeholder, dtype=np.float32)
            return placeholder, vessel_out

        # Minimum ROI volume check (too small → out-of-distribution; abort safely)
        if p.opts.abort_on_small_roi:
            try:
                roi_obj = vessel_out.roi
                if roi_obj is None:
                    raise RuntimeError("Vessel segmentation ROI tensor missing")
                # Obtain ROI tensor spatial shape (Z,Y,X)
                if isinstance(roi_obj, torch.Tensor):
                    roi_shape = tuple(int(x) for x in roi_obj.shape)
                else:
                    roi_shape = tuple(int(x) for x in np.asarray(roi_obj).shape)
                if len(roi_shape) == 4:
                    spatial = roi_shape[1:4]
                elif len(roi_shape) == 3:
                    spatial = roi_shape
                else:
                    raise RuntimeError(f"unexpected ROI ndim={len(roi_shape)}")
                roi_voxels = int(spatial[0]) * int(spatial[1]) * int(spatial[2])
                logging.getLogger(__name__).debug("[DEBUG] roi_voxels=%d", roi_voxels)
                if p.opts.min_roi_voxels is not None and p.opts.min_roi_voxels > 0:
                    if roi_voxels < int(p.opts.min_roi_voxels):
                        raise RuntimeError(
                            f"ROI volume too small: roi_voxels={roi_voxels}; min_roi_voxels={int(p.opts.min_roi_voxels)}"
                        )
            except Exception:
                # Exceptions here are caught by predict() which returns fallback probabilities
                raise

        if p.opts.debug:
            sparse_extras = vessel_out.extras.get(VesselSegmentationStage.SPARSE_SEARCH.value)
            if isinstance(sparse_extras, dict):
                orientation_info = sparse_extras.get("orientation")
                if isinstance(orientation_info, dict):
                    logging.getLogger(__name__).debug(
                        "[DEBUG] Orientation correction: perm=%s signs=%s score=%.3f",
                        orientation_info.get("perm"),
                        orientation_info.get("signs"),
                        float(orientation_info.get("score", float("nan"))),
                    )
            fig = None
            try:
                roi_data = vessel_out.roi
                seg_data = vessel_out.seg
                if roi_data is None or seg_data is None:
                    raise ValueError("roi/seg missing in vessel_out")
                if isinstance(roi_data, torch.Tensor):
                    roi_data = roi_data.detach().cpu().numpy()
                if isinstance(seg_data, torch.Tensor):
                    seg_data = seg_data.detach().cpu().numpy()
                roi_np = np.asarray(roi_data)
                seg_np = np.asarray(seg_data)
                if roi_np.ndim == 4:
                    roi_vol = roi_np[0]
                elif roi_np.ndim == 3:
                    roi_vol = roi_np
                else:
                    raise ValueError(f"unexpected ROI ndim={roi_np.ndim}")
                roi_vol = np.squeeze(roi_vol)
                if seg_np.ndim == 4:
                    seg_vol = seg_np[0]
                elif seg_np.ndim == 3:
                    seg_vol = seg_np
                else:
                    raise ValueError(f"unexpected seg ndim={seg_np.ndim}")
                seg_vol = np.squeeze(seg_vol)
                if roi_vol.shape != seg_vol.shape:
                    raise ValueError(f"ROI/seg shape mismatch: {roi_vol.shape} vs {seg_vol.shape}")
                seg_vol = np.nan_to_num(seg_vol)
                if seg_vol.dtype.kind in ("f", "c"):
                    seg_vol = np.rint(seg_vol).astype(np.int16, copy=False)
                else:
                    seg_vol = seg_vol.astype(np.int16, copy=False)
                coords = np.argwhere(seg_vol > 0)
                if coords.size:
                    center = np.clip(
                        np.round(coords.mean(axis=0)).astype(int), 0, np.array(seg_vol.shape) - 1
                    )
                else:
                    center = np.array(seg_vol.shape) // 2
                center_z, center_y, center_x = map(int, center.tolist())
                roi_slices = (
                    ("Axial", roi_vol[center_z, :, :], seg_vol[center_z, :, :]),
                    ("Coronal", roi_vol[:, center_y, :], seg_vol[:, center_y, :]),
                    ("Sagittal", roi_vol[:, :, center_x], seg_vol[:, :, center_x]),
                )
                fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
                axes = np.atleast_1d(axes)
                for ax, (title, img_slice, seg_slice) in zip(axes, roi_slices):
                    img_slice = np.nan_to_num(img_slice)
                    seg_mask = seg_slice > 0
                    ax.imshow(img_slice, cmap="gray", origin="lower")
                    if seg_mask.any():
                        overlay = np.ma.masked_where(~seg_mask, seg_slice)
                        ax.imshow(
                            overlay, cmap="nipy_spectral", origin="lower", alpha=0.5, interpolation="nearest"
                        )
                    ax.set_title(title)
                    ax.axis("off")
                debug_dir = Path("/workspace/logs/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                series_id = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(series_path).name or "series")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = debug_dir / f"{series_id}_{timestamp}_roi_planes.png"
                fig.savefig(out_path, dpi=150)
            except Exception as exc:
                logging.getLogger(__name__).debug("[DEBUG] Failed to save ROI slice: %s", exc)
            finally:
                if fig is not None:
                    plt.close(fig)

        seg = vessel_out.seg
        roi = vessel_out.roi
        if seg is None or roi is None:
            raise RuntimeError("Vessel segmentation output is missing tensors despite full stage execution")

        # Preserve additional dense model segmentation
        dense_outputs = getattr(vessel_out, "dense_outputs", {})
        additional_dense_segs: Dict[str, Union[torch.Tensor, np.ndarray]] = {}
        if isinstance(dense_outputs, dict):
            for key, result in dense_outputs.items():
                if key == "primary" or result is None:
                    continue
                seg_extra = getattr(result, "prediction", None)
                if seg_extra is None:
                    continue
                additional_dense_segs[key] = seg_extra.detach()
        if additional_dense_segs:
            # Store in auxiliary attribute for later stages
            setattr(vessel_out, "additional_dense_segmentations", additional_dense_segs)

        # Clear caches (mitigate fragmentation)
        _clear_cuda_cache()

        with time_section("STEP4 ROI classification inference", enabled=p.opts.debug):
            models_with_dev, input_size, keep_ratio, spatial_tf, pad_multiple = p.load_roi_models()
            # List additional (dense) segs sorted by key for stability
            extra_list: Optional[List[np.ndarray]] = None
            add_dict = getattr(vessel_out, "additional_dense_segmentations", None)
            if isinstance(add_dict, dict) and len(add_dict) > 0:
                extra_list = [add_dict[k] for k in sorted(add_dict.keys())]

            roi_b, vessel_13_b, union_b, extra_b = _prepare_inputs_for_roi(
                roi,
                seg,
                input_size,
                keep_ratio,
                spatial_tf,
                pad_multiple,
                extra_segs=extra_list,
            )

            # If union_b sum (voxels) is too small, consider seg failure and abort safely
            logging.getLogger(__name__).debug("[DEBUG] union_b=%d", union_b.sum())
            if p.opts.abort_on_low_union and p.opts.min_union_sum is not None and p.opts.min_union_sum > 0:
                try:
                    union_sum = float(torch.as_tensor(union_b).sum().item())
                except Exception:
                    # Cross-check via numpy
                    union_sum = float(np.asarray(union_b).sum())
                if union_sum < float(p.opts.min_union_sum):
                    raise RuntimeError(
                        f"union_b sum too small: sum={int(union_sum)}; min_union_sum={int(p.opts.min_union_sum)}"
                    )

            tta_flags = _build_tta_flip_flags(max(1, p.opts.roi_tta))
            cpu_inputs = _build_tta_variants(roi_b, vessel_13_b, union_b, tta_flags, extra_masks=extra_b)
            if p.opts.debug and len(tta_flags) > 1:
                logging.getLogger(__name__).debug(
                    "[DEBUG] ROI TTA variants=%d flags=%s",
                    len(tta_flags),
                    tta_flags,
                )

            by_dev_inputs = _assign_inputs_to_devices(
                models_with_dev,
                cpu_inputs,
            )

            logits_sum_cpu, counts = _forward_roi_models(
                models_with_dev,
                by_dev_inputs,
                torch.float16,
            )

        with time_section("STEP5 Fold ensemble (mean + post)", enabled=p.opts.debug):
            if logits_sum_cpu is None or counts == 0:
                raise RuntimeError("Failed to aggregate ROI inference (counts=0)")
            logits_mean = logits_sum_cpu / float(counts)
            probs = torch.sigmoid(logits_mean).squeeze(0).detach().cpu().numpy().astype(np.float32)

        if p.opts.debug:
            # Total time can be approximated by summing time_section logs
            logging.getLogger(__name__).debug(
                f"[DEBUG] probs shape={probs.shape} min={probs.min():.4f} max={probs.max():.4f}"
            )
        return probs, vessel_out


def _run_until_stage(series_path: str, stage: str, debug: bool = False) -> None:
    """Run up to the specified stage; return on success, raise on failure.

    Args:
        series_path: Path to the DICOM series
        stage: "dicom" | "vessel" (use _predict_probs() for "roi")
        debug: Enable lightweight logs when True

    Notes:
        - Does not return probabilities; only verifies stage completion
        - Useful on Kaggle hidden tests to confirm which stage causes a failure
    """
    stage = (stage or "").strip().lower()
    if stage not in ("dicom", "vessel"):
        return  # Delegate to full pipeline for "roi" etc.

    # 1) DICOM→NIfTI
    # Keep legacy stage diagnostics for compatibility
    opts = PipelineOptions.from_env()
    p = _get_pipeline(opts)
    with tempfile.TemporaryDirectory(prefix="rsna_niix_") as td:
        tmp_dir = Path(td)
        _ensure_dcm2niix_available(allow_apt_install=opts.allow_apt_install)
        nifti_path = convert_dicom_series_to_nifti(series_path, out_dir=tmp_dir, debug=debug)
        if debug:
            logging.getLogger(__name__).info(f"[STAGE] DICOM→NIfTI OK: {nifti_path}")
        if stage == "dicom":
            return
        # 2) Load NIfTI
        image, properties = p.load_nifti_volume(nifti_path)
        # 3) Vessel segmentation
        _ = p.vessel_predict(image, properties)
        if debug:
            logging.getLogger(__name__).info("[STAGE] VesselSeg OK")
        return


# ===== Kaggle Entry Point =====
def predict(series_path: str) -> pl.DataFrame:
    """
    Prediction function called by Kaggle API.

    Args:
        series_path: Path to the DICOM series

    Returns:
        pl.DataFrame: Columns = 13 locations + Aneurysm Present
    """
    # Determine debug output from environment variables
    # Prefer RSNA_DEBUG, else fall back to generic DEBUG
    opts = PipelineOptions.from_env()
    # Minimal log level configuration; use logging afterwards
    _configure_logging(opts.debug)
    debug = opts.debug
    p = _get_pipeline(opts)
    p._predict_call_count += 1
    try:
        if debug:
            logging.getLogger(__name__).debug(f"[DEBUG] series_path={series_path}")
            logging.getLogger(__name__).debug(f"[DEBUG] device={_device}")

        # Pre-control the stage (for hidden test diagnostics)
        # RSNA_STAGE: "dicom" | "vessel" | "roi" (default)
        stage = opts.rsna_stage
        if stage in ("dicom", "vessel"):
            # Execute until the specified stage; raise on failure
            _run_until_stage(series_path, stage, debug=debug)
            # If successful, return fixed probabilities (for success/failure isolation only)
            placeholder = opts.stage_placeholder
            out = np.full((len(LABEL_COLS),), placeholder, dtype=np.float32)
            if debug:
                logging.getLogger(__name__).info(
                    f"[DEBUG] RSNA_STAGE={stage} executed: returning fixed value ({placeholder})"
                )
            return pl.DataFrame([out.tolist()], schema=LABEL_COLS, orient="row")

        # When debug, show processing time per step (full pipeline)
        probs, vessel_diag = _predict_probs_with_pipeline(p, series_path)
        if vessel_diag.terminated_early and debug:
            logging.getLogger(__name__).debug(
                "[DEBUG] Vessel segmentation terminated early: stage=%s limit=%s",
                vessel_diag.completed_stage.value,
                vessel_diag.stage_limit.value,
            )

        # ===== Additional options: toggle which scores come from model vs. fixed =====
        # - RSNA_ONLY_AP enabled: use model only for AP, locations are fixed
        # - RSNA_ONLY_LOCATIONS enabled: use model only for locations, AP is fixed
        # - Fixed value is from RSNA_FIXED_PROB (default 0.5), always clamped to [0,1]
        only_ap = opts.only_ap
        only_loc = opts.only_locations
        fixed_val = opts.fixed_prob

        out = probs.copy()
        if only_ap and only_loc:
            # If both are true, prioritize AP (explicit log)
            if debug:
                logging.getLogger(__name__).debug(
                    "[DEBUG] RSNA_ONLY_AP and RSNA_ONLY_LOCATIONS both enabled. Prioritizing AP."
                )
        if only_ap:
            # Replace 13 locations (first 13) with fixed value; keep AP prediction
            out[:13] = fixed_val
            if debug:
                logging.getLogger(__name__).debug(
                    f"[DEBUG] RSNA_ONLY_AP enabled: locations=fixed({fixed_val}), AP=predicted"
                )
        elif only_loc:
            # Replace AP (last) with fixed value; keep 13 location predictions
            out[13] = fixed_val
            if debug:
                logging.getLogger(__name__).debug(
                    f"[DEBUG] RSNA_ONLY_LOCATIONS enabled: AP=fixed({fixed_val}), locations=predicted"
                )

        return pl.DataFrame([out.tolist()], schema=LABEL_COLS, orient="row")

    except Exception as e:
        return _handle_prediction_exception(e, series_path, p, opts)


# ===== Pipeline singleton =====
_pipeline_singleton: Optional[RsnaRoiPipeline] = None


def _get_pipeline(
    opts: Optional[PipelineOptions] = None, create_if_none: bool = True
) -> Optional[RsnaRoiPipeline]:
    """Keep a single global pipeline instance for reuse."""
    global _pipeline_singleton
    if _pipeline_singleton is None and create_if_none:
        _pipeline_singleton = RsnaRoiPipeline(opts or PipelineOptions.from_env())
    return _pipeline_singleton


def _build_tta_variants(
    roi_batch: torch.Tensor,
    vessel_masks: torch.Tensor,
    vessel_union: torch.Tensor,
    flags_list: Sequence[Tuple[bool, bool, bool]],
    *,
    extra_masks: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    variants: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
    for flags in flags_list:
        roi_var = _apply_flip_for_tta(roi_batch, flags)
        vessel_var = _apply_flip_for_tta(vessel_masks, flags)
        union_var = _apply_flip_for_tta(vessel_union, flags)
        extra_var = _apply_flip_for_tta(extra_masks, flags) if extra_masks is not None else None
        variants.append((roi_var, vessel_var, union_var, extra_var))
    return variants


def _assign_inputs_to_devices(
    models_with_dev: Sequence[Tuple[AneurysmVesselSegROILitModule, torch.device]],
    cpu_inputs: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
) -> Dict[torch.device, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]:
    assignments: Dict[
        torch.device, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    ] = {}
    for _, dev in models_with_dev:
        if dev in assignments:
            continue
        dev_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
        for roi_cpu, vessel_cpu, union_cpu, extra_cpu in cpu_inputs:
            dev_inputs.append(
                (
                    roi_cpu.to(dev, non_blocking=True),
                    vessel_cpu.to(dev, non_blocking=True),
                    union_cpu.to(dev, non_blocking=True),
                    (extra_cpu.to(dev, non_blocking=True) if isinstance(extra_cpu, torch.Tensor) else None),
                )
            )
        assignments[dev] = dev_inputs
    return assignments


def _forward_roi_models(
    models_with_dev: Sequence[Tuple[AneurysmVesselSegROILitModule, torch.device]],
    by_device_inputs: Dict[
        torch.device, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    ],
    autocast_dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], int]:
    per_dev_sum: Dict[torch.device, torch.Tensor] = {}
    counts = 0
    use_autocast = any(dev.type == "cuda" for _, dev in models_with_dev)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        for model, dev in models_with_dev:
            dev_inputs = by_device_inputs[dev]
            for roi_t, vessel_t, union_t, extra_t in dev_inputs:
                out = model.forward(roi_t, vessel_t, union_t, extra_vessel_seg=extra_t)
                logits_loc = out["logits_loc"]
                logit_ap = out["logit_ap"].unsqueeze(1)
                logits = torch.cat([logits_loc, logit_ap], dim=1)
                if dev not in per_dev_sum:
                    per_dev_sum[dev] = logits
                else:
                    per_dev_sum[dev] = per_dev_sum[dev] + logits
                counts += 1

    try:
        for dev in per_dev_sum.keys():
            if dev.type == "cuda":
                torch.cuda.synchronize(device=dev)
    except Exception:
        pass

    logits_sum_cpu: Optional[torch.Tensor] = None
    for tensor in per_dev_sum.values():
        tensor_cpu = tensor.detach().to("cpu")
        if logits_sum_cpu is None:
            logits_sum_cpu = tensor_cpu
        else:
            logits_sum_cpu += tensor_cpu
    return logits_sum_cpu, counts


def _handle_prediction_exception(
    exc: Exception,
    series_path: str,
    pipeline: RsnaRoiPipeline,
    opts: PipelineOptions,
) -> pl.DataFrame:
    logger = logging.getLogger(__name__)
    logger.error(f"[ERROR] prediction failed: {exc}")

    pipeline._predict_error_count += 1
    max_err = _parse_int_env("RSNA_MAX_PREDICT_ERRORS", default=0)

    _append_error_log(series_path, exc, pipeline)

    # RSNA_MAX_PREDICT_ERRORS: stop only when errors strictly exceed the limit, not when equal.
    if max_err > 0 and pipeline._predict_error_count > max_err:
        logger.error(
            "[FATAL] predict errors exceeded threshold: %d / %d. Aborting.",
            pipeline._predict_error_count,
            max_err,
        )
        raise SystemExit(2)

    fallback = _prepare_fallback_probs(opts)
    adjusted = _apply_fixed_prob_overrides(
        fallback,
        opts.only_ap,
        opts.only_locations,
        opts.fixed_prob,
    )
    return pl.DataFrame([adjusted.tolist()], schema=LABEL_COLS, orient="row")


def _parse_int_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def _append_error_log(series_path: str, exc: Exception, pipeline: RsnaRoiPipeline) -> None:
    log_path = os.getenv("RSNA_ERROR_LOG", "/kaggle/working/predict_errors.jsonl").strip()
    if not log_path:
        return
    try:
        log_path_obj = Path(log_path)
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path_obj, "a", encoding="utf-8") as f:
            import json as _json

            _json.dump(
                {
                    "series_path": series_path,
                    "error": str(exc),
                    "error_count": pipeline._predict_error_count,
                    "call_count": pipeline._predict_call_count,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    except Exception:
        pass


def _prepare_fallback_probs(opts: PipelineOptions) -> np.ndarray:
    fallback = np.asarray(opts.error_fallback_probs, dtype=np.float32)
    if fallback.shape[0] != len(LABEL_COLS):
        fallback = np.full((len(LABEL_COLS),), 0.1, dtype=np.float32)
    return fallback


def _apply_fixed_prob_overrides(
    probs: np.ndarray,
    only_ap: bool,
    only_loc: bool,
    fixed_val: float,
) -> np.ndarray:
    out = probs.copy()
    if only_ap:
        out[:13] = fixed_val
    elif only_loc:
        out[13] = fixed_val
    return out


if __name__ == "__main__":
    import glob

    os.environ["VESSEL_ABORT_ON_SPARSE_FAIL"] = "1"  # abort on sparse failure instead of dense
    os.environ["VESSEL_ABORT_MIN_ALL_DIMS_MM"] = "140"  # threshold to distinguish fallback vs small volume
    os.environ["VESSEL_ABORT_ON_SMALL_ROI"] = "1"  # abort when ROI volume too small
    os.environ["VESSEL_MIN_ROI_VOXELS"] = "1000000"
    os.environ["VESSEL_ABORT_ON_LOW_UNION"] = "1"  # abort when union voxel count too small
    os.environ["VESSEL_MIN_UNION_SUM"] = "2000"

    os.environ["RSNA_ERROR_FALLBACK_PROBS"] = (
        "0.05,0.07,0.07,0.13,0.07,0.06,0.02,0.01,0.01,0.07,0.02,0.01,0.028,0.55"
    )

    # Vessel segmentation
    os.environ["VESSEL_NNUNET_SPARSE_MODEL_DIR"] = (
        "/workspace/logs/nnUNet_results/Dataset003_VesselGrouping/RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres"
    )
    os.environ["VESSEL_ENABLE_ORIENTATION_CORRECTION"] = "1"
    os.environ["VESSEL_ORIENTATION_WEIGHTS"] = "1,1,1"
    # ROI physical size (mm)
    os.environ["VESSEL_SPARSE_ROI_EXTENT_MM"] = "140"
    os.environ["VESSEL_REFINE_Z_ONLY"] = "0"

    os.environ["VESSEL_NNUNET_MODEL_DIR"] = (
        "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/nnUNetTrainerSkeletonRecall_more_DAv3__nnUNetResEncUNetMPlans__3d_fullres"
    )
    os.environ["VESSEL_ADDITIONAL_DENSE_MODEL_DIRS"] = (
        "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres,"
    )
    os.environ["VESSEL_FOLDS"] = "all"

    os.environ["VESSEL_REFINE_MARGIN_Z"] = "15"
    os.environ["VESSEL_REFINE_MARGIN_XY"] = "30"

    os.environ["VESSEL_TRT_DIR"] = "/workspace/logs/trt"

    # ROI classification model
    os.environ["ROI_EXPERIMENTS"] = (
        "251013-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256"
    )
    os.environ["ROI_FOLDS"] = "0,1,3,4"
    os.environ["ROI_TTA"] = "2"

    os.environ["RSNA_DEBUG"] = "1"

    # Simple test (local run)
    n_test = 10
    series_paths = glob.glob(os.path.join("/workspace/data/series", "*"))
    series_paths = sorted(series_paths)

    for series_path in series_paths[:n_test]:
        # Debug display follows RSNA_DEBUG / DEBUG
        df = predict(series_path)
        logging.getLogger(__name__).info(df.to_numpy())
