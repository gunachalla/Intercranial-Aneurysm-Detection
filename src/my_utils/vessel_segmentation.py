#!/usr/bin/env python3
"""
Sample inference script for Kaggle submissions.
Brain vessel segmentation - fast inference with adaptive sparse search.

Usage:
    1. Upload model files to Kaggle
    2. Copy this script into a Kaggle notebook
    3. Set appropriate paths and run
"""

import os
import time
import json
import gc
import math
import re
import argparse
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence, Union
from tqdm import tqdm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from nnunetv2.inference.adaptive_sparse_predictor import (
    DBSCANAdaptiveSparsePredictor,
    AdaptiveSparsePredictor,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient
from src.my_utils.vessel_orientation_solver import (
    estimate_perm_signs_from_vessel3,
    OrientationEstimate,
)


class VesselSegmentationStage(str, Enum):
    """Stage types for the inference pipeline."""

    LOAD = "load"
    PREPROCESS = "preprocess"
    SPARSE_SEARCH = "sparse_search"
    ROI_CROP = "roi_crop"
    FOLD_INFER = "fold_infer"
    REFINE = "refine"
    AGGREGATE = "aggregate"


_STAGE_ORDER: List[VesselSegmentationStage] = [
    VesselSegmentationStage.LOAD,
    VesselSegmentationStage.PREPROCESS,
    VesselSegmentationStage.SPARSE_SEARCH,
    VesselSegmentationStage.ROI_CROP,
    VesselSegmentationStage.FOLD_INFER,
    VesselSegmentationStage.REFINE,
    VesselSegmentationStage.AGGREGATE,
]

_STAGE_INDEX: Dict[VesselSegmentationStage, int] = {stage: idx for idx, stage in enumerate(_STAGE_ORDER)}

_STAGE_ALIASES: Dict[str, VesselSegmentationStage] = {
    "load": VesselSegmentationStage.LOAD,
    "read": VesselSegmentationStage.LOAD,
    "preprocess": VesselSegmentationStage.PREPROCESS,
    "pre": VesselSegmentationStage.PREPROCESS,
    "prep": VesselSegmentationStage.PREPROCESS,
    "sparse_search": VesselSegmentationStage.SPARSE_SEARCH,
    "sparse": VesselSegmentationStage.SPARSE_SEARCH,
    "search": VesselSegmentationStage.SPARSE_SEARCH,
    "roi_crop": VesselSegmentationStage.ROI_CROP,
    "roi": VesselSegmentationStage.ROI_CROP,
    "crop": VesselSegmentationStage.ROI_CROP,
    "fold_infer": VesselSegmentationStage.FOLD_INFER,
    "infer": VesselSegmentationStage.FOLD_INFER,
    "inference": VesselSegmentationStage.FOLD_INFER,
    "refine": VesselSegmentationStage.REFINE,
    "refinement": VesselSegmentationStage.REFINE,
    "aggregate": VesselSegmentationStage.AGGREGATE,
    "agg": VesselSegmentationStage.AGGREGATE,
    "full": VesselSegmentationStage.AGGREGATE,
    "final": VesselSegmentationStage.AGGREGATE,
    "output": VesselSegmentationStage.AGGREGATE,
}


_DEFAULT_REFINE_THRESHOLD = 0.3
_DEFAULT_REFINE_MARGIN_Z = 15
_DEFAULT_REFINE_MARGIN_XY = 30


def parse_vessel_stage_limit(
    raw: Optional[str],
    default: VesselSegmentationStage = VesselSegmentationStage.AGGREGATE,
) -> VesselSegmentationStage:
    """Parse stage limit given as an env-style string."""

    if raw is None:
        return default
    key = raw.strip().lower()
    if key == "":
        return default
    return _STAGE_ALIASES.get(key, default)


FoldKey = Union[int, str]


def _fold_sort_key(value: FoldKey) -> Tuple[int, Union[int, str]]:
    """Build a sorting key for fold values."""

    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def _normalize_fold_token(value: FoldKey) -> FoldKey:
    """Normalize a fold token to an int or identifier."""

    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if text == "":
        raise ValueError("Empty fold specification")
    lower = text.lower()
    if lower in {"all", "fold_all"}:
        return "all"
    if lower.startswith("fold_"):
        remainder = lower[len("fold_") :]
        if remainder.isdigit() or (remainder.startswith("-") and remainder[1:].isdigit()):
            return int(remainder)
        return remainder
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return lower


def _discover_available_folds(model_dir: Path) -> List[FoldKey]:
    """Collect available folds inside a model directory."""

    folds: List[FoldKey] = []
    if not model_dir.exists():
        return folds
    for subdir in sorted(model_dir.glob("fold_*")):
        if not subdir.is_dir():
            continue
        suffix = subdir.name.split("fold_", 1)[1]
        normalized = _normalize_fold_token(suffix)
        if normalized not in folds:
            folds.append(normalized)
    folds.sort(key=_fold_sort_key)
    return folds


def _normalize_fold_list(
    folds_input: Sequence[FoldKey] | FoldKey,
    available_folds: Sequence[FoldKey],
) -> Tuple[FoldKey, ...]:
    """Normalize fold specs and validate directory presence."""

    if isinstance(folds_input, (int, str)):
        candidates: List[FoldKey] = [folds_input]
    else:
        candidates = list(folds_input)

    if len(candidates) == 0:
        raise ValueError("Empty folds specification")

    normalized: List[FoldKey] = []
    for fold in candidates:
        normalized_fold = _normalize_fold_token(fold)
        if available_folds and normalized_fold not in available_folds:
            raise ValueError(f"Specified fold {fold} not found in available folds: {available_folds}")
        normalized.append(normalized_fold)

    return tuple(normalized)


@dataclass
class DenseModelResult:
    """Bundle of outputs per dense-inference model."""

    prediction: Optional[torch.Tensor | np.ndarray]
    roi: Optional[torch.Tensor | np.ndarray]
    transform_info: Optional[dict]
    return_probabilities: bool
    refined_local_bbox: Optional[Tuple[slice, ...]] = None


@dataclass
class VesselSegmentationOutput:
    """Vessel segmentation outputs with diagnostics."""

    seg: Optional[torch.Tensor]
    roi: Optional[torch.Tensor]
    transform_info: Optional[dict]
    completed_stage: VesselSegmentationStage
    stage_limit: VesselSegmentationStage
    stage_history: List[VesselSegmentationStage] = field(default_factory=list)
    extras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dense_outputs: Dict[str, DenseModelResult] = field(default_factory=dict)

    @property
    def is_full(self) -> bool:
        """Whether the final stage has been reached and required data are present."""

        return (
            self.completed_stage == VesselSegmentationStage.AGGREGATE
            and self.seg is not None
            and self.roi is not None
        )

    @property
    def terminated_early(self) -> bool:
        """Whether the run terminated early due to a stage limit."""

        return not self.is_full


class VesselSegmentationStageTracker:
    """Helper to track progression through vessel segmentation stages."""

    def __init__(self, stage_limit: Optional[VesselSegmentationStage]):
        self.stage_limit = stage_limit or VesselSegmentationStage.AGGREGATE
        self.stage_limit_index = _STAGE_INDEX[self.stage_limit]
        self.stage_history: List[VesselSegmentationStage] = []
        self.completed_stage: Optional[VesselSegmentationStage] = None
        self.extras: Dict[str, Dict[str, Any]] = {}

    def mark(self, stage: VesselSegmentationStage, info: Optional[Dict[str, Any]] = None) -> None:
        """Record a completed stage and optionally attach info."""

        self.stage_history.append(stage)
        self.completed_stage = stage
        if info:
            self.extras[stage.value] = info

    def limit_reached(self) -> bool:
        """Whether the current stage has reached the limit stage."""

        if self.completed_stage is None:
            return False
        return _STAGE_INDEX[self.completed_stage] >= self.stage_limit_index

    def should_stop(self) -> bool:
        """Signal to stop when a stage limit is set and reached before final stage."""

        if self.stage_limit == VesselSegmentationStage.AGGREGATE:
            return False
        return self.limit_reached()

    def build_output(
        self,
        seg: Optional[torch.Tensor] = None,
        roi: Optional[torch.Tensor] = None,
        transform_info: Optional[dict] = None,
        dense_outputs: Optional[Dict[str, DenseModelResult]] = None,
    ) -> VesselSegmentationOutput:
        """Assemble the current state into a `VesselSegmentationOutput`."""

        stage = self.completed_stage or VesselSegmentationStage.LOAD
        return VesselSegmentationOutput(
            seg=seg,
            roi=roi,
            transform_info=transform_info,
            completed_stage=stage,
            stage_limit=self.stage_limit,
            stage_history=list(self.stage_history),
            extras=dict(self.extras),
            dense_outputs=dict(dense_outputs) if dense_outputs is not None else {},
        )


def _shape_tuple(shape: Any) -> Tuple[int, ...]:
    """Convert array/tensor shape to a tuple of ints."""

    return tuple(int(x) for x in shape)


def _slices_to_ranges(slices: Tuple[slice, ...]) -> List[Tuple[int, int]]:
    """Convert slice objects to [start, stop) integer ranges."""

    ranges: List[Tuple[int, int]] = []
    for s in slices:
        start = 0 if s.start is None else int(s.start)
        stop = 0 if s.stop is None else int(s.stop)
        ranges.append((start, stop))
    return ranges


@dataclass
class OrientationCorrectionResult:
    perm: Tuple[int, int, int]
    signs: Tuple[int, int, int]
    estimate: OrientationEstimate
    shape_before: Tuple[int, int, int]
    shape_after: Tuple[int, int, int]
    num_classes: int
    spacing_zyx: Tuple[float, float, float]
    roi_bbox_before: Tuple[slice, ...]
    roi_bbox_after: Tuple[slice, ...]
    used_cropped_roi: bool

    def to_metadata(
        self,
        *,
        shape_before: Optional[Sequence[int]] = None,
        shape_after: Optional[Sequence[int]] = None,
        roi_bbox_before: Optional[Tuple[slice, ...]] = None,
        roi_bbox_after: Optional[Tuple[slice, ...]] = None,
        spacing_zyx: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Convert to a JSON-friendly dict (supports overriding shapes/ROI)."""

        if shape_before is not None:
            shape_before_tuple = tuple(int(x) for x in shape_before)
        else:
            shape_before_tuple = self.shape_before

        if shape_after is not None:
            shape_after_tuple = tuple(int(x) for x in shape_after)
        elif shape_before is not None:
            shape_after_tuple = tuple(shape_before_tuple[int(idx)] for idx in self.perm)
        else:
            shape_after_tuple = self.shape_after

        if roi_bbox_before is not None:
            roi_before = roi_bbox_before
        else:
            roi_before = self.roi_bbox_before

        if roi_bbox_after is not None:
            roi_after = roi_bbox_after
        elif roi_bbox_before is not None and roi_bbox_before is not self.roi_bbox_before:
            roi_after = _apply_orientation_to_slices(
                roi_bbox_before,
                shape_before_tuple,
                self.perm,
                self.signs,
            )
        else:
            roi_after = self.roi_bbox_after

        spacing_tuple = tuple(float(x) for x in spacing_zyx) if spacing_zyx is not None else self.spacing_zyx

        return {
            "perm": [int(x) for x in self.perm],
            "signs": [int(x) for x in self.signs],
            "shape_before": [int(x) for x in shape_before_tuple],
            "shape_after": [int(x) for x in shape_after_tuple],
            "spacing_zyx": [float(x) for x in spacing_tuple],
            "num_classes": int(self.num_classes),
            "used_cropped_roi": bool(self.used_cropped_roi),
            "score": float(self.estimate.score),
            "score_terms": {k: float(v) for k, v in self.estimate.score_terms.items()},
            "reliability": {k: float(v) for k, v in self.estimate.reliability.items()},
            "chosen_index_24": int(self.estimate.chosen_index_24),
            "roi_bbox_before": _slices_to_ranges(roi_before) if roi_before is not None else None,
            "roi_bbox_after": _slices_to_ranges(roi_after) if roi_after is not None else None,
            "estimation_roi_bbox": (
                _slices_to_ranges(roi_before) if (roi_before is not None and self.used_cropped_roi) else None
            ),
        }


def _orientation_is_identity(perm: Tuple[int, int, int], signs: Tuple[int, int, int]) -> bool:
    return perm == (0, 1, 2) and signs == (1, 1, 1)


def _extract_perm_signs_from_meta(
    meta: Any,
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Safely extract perm/signs from orientation_correction metadata."""

    if not isinstance(meta, dict):
        return None
    perm_raw = meta.get("perm")
    signs_raw = meta.get("signs")
    if not isinstance(perm_raw, (list, tuple)) or not isinstance(signs_raw, (list, tuple)):
        return None
    if len(perm_raw) != 3 or len(signs_raw) != 3:
        return None
    try:
        perm_tuple = tuple(int(perm_raw[i]) for i in range(3))
        signs_tuple = tuple(int(signs_raw[i]) for i in range(3))
    except (TypeError, ValueError):
        return None
    return perm_tuple, signs_tuple


def _resolve_orientation_from_output(
    output: "VesselSegmentationOutput",
    base_result: Optional[DenseModelResult],
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Recover orientation-correction perm/signs from a VesselSegmentationOutput and related info."""

    candidates: List[Any] = []

    def _push(meta_parent: Optional[Dict[str, Any]]) -> None:
        if isinstance(meta_parent, dict):
            candidates.append(meta_parent.get("orientation_correction"))

    _push(output.transform_info if isinstance(output.transform_info, dict) else None)

    if base_result is not None and isinstance(base_result.transform_info, dict):
        _push(base_result.transform_info)

    for result in output.dense_outputs.values():
        if result is None:
            continue
        meta = getattr(result, "transform_info", None)
        if isinstance(meta, dict):
            _push(meta)

    stage_key = VesselSegmentationStage.SPARSE_SEARCH.value
    extras_meta = output.extras.get(stage_key) if isinstance(output.extras, dict) else None
    if isinstance(extras_meta, dict):
        candidates.append(extras_meta.get("orientation"))

    for meta in candidates:
        extracted = _extract_perm_signs_from_meta(meta)
        if extracted is not None:
            return extracted
    return None


def _apply_orientation_to_tensor(
    tensor: torch.Tensor, perm: Tuple[int, int, int], signs: Tuple[int, int, int]
) -> torch.Tensor:
    """Apply perm/flip to the tensor's spatial dimensions."""

    spatial_dims = 3
    if tensor.ndim < spatial_dims:
        raise ValueError(f"Orientation correction requires a tensor with >=3 dims: {tensor.shape}")
    lead = tensor.ndim - spatial_dims
    permute_order = list(range(lead)) + [lead + perm[i] for i in range(spatial_dims)]
    oriented = tensor.permute(permute_order)
    flip_dims = [lead + i for i, s in enumerate(signs) if int(s) < 0]
    if flip_dims:
        oriented = torch.flip(oriented, dims=flip_dims)
    return oriented.contiguous()


def _apply_orientation_to_slices(
    slices: Tuple[slice, ...],
    shape_before: Tuple[int, int, int],
    perm: Tuple[int, int, int],
    signs: Tuple[int, int, int],
) -> Tuple[slice, ...]:
    """Transform slices from the old to the new coordinate system."""

    new_slices: List[slice] = []
    for new_ax, old_ax in enumerate(perm):
        slc = slices[old_ax]
        length = int(shape_before[old_ax])
        start = int(slc.start)
        stop = int(slc.stop)
        if int(signs[new_ax]) < 0:
            new_start = length - stop
            new_stop = length - start
        else:
            new_start = start
            new_stop = stop
        new_slices.append(slice(new_start, new_stop))
    return tuple(new_slices)


def _invert_orientation_on_slices(
    slices: Tuple[slice, ...],
    shape_before: Tuple[int, int, int],
    perm: Tuple[int, int, int],
    signs: Tuple[int, int, int],
) -> Tuple[slice, ...]:
    """Transform slices from the new coordinate system back to the old one."""

    old_slices: List[Optional[slice]] = [None, None, None]
    for new_ax, old_ax in enumerate(perm):
        slc = slices[new_ax]
        length = int(shape_before[old_ax])
        start = int(slc.start)
        stop = int(slc.stop)
        if int(signs[new_ax]) < 0:
            old_start = length - stop
            old_stop = length - start
        else:
            old_start = start
            old_stop = stop
        old_slices[old_ax] = slice(old_start, old_stop)
    return tuple(x if x is not None else slice(0, shape_before[i]) for i, x in enumerate(old_slices))


def _apply_orientation_to_coords(
    coords: List[float],
    shape_before: Tuple[int, int, int],
    perm: Tuple[int, int, int],
    signs: Tuple[int, int, int],
) -> List[float]:
    """Transform continuous coordinates from the old to the new coordinate system."""

    transformed = [0.0, 0.0, 0.0]
    for new_ax, old_ax in enumerate(perm):
        val = float(coords[old_ax])
        if int(signs[new_ax]) < 0:
            transformed[new_ax] = float(shape_before[old_ax] - 1 - val)
        else:
            transformed[new_ax] = val
    return transformed


class VesselSegmentationPredictor:
    """
    Brain vessel segmentation inference class
    """

    def __init__(
        self,
        model_path: str,
        additional_dense_model_paths: Optional[Sequence[str]] = None,
        sparse_model_path: Optional[str] = None,
        folds: Sequence[FoldKey] | FoldKey = (0, 1, 2, 3, 4),
        device: str = "cuda",
        use_sparse_search: bool = True,
        torch_compile: bool = False,
        verbose: bool = True,
        # Respect Predictor defaults: treat None as unspecified
        use_mirroring: Optional[bool] = None,
        devices: Tuple[int, ...] | List[int] | None = None,
        # GPU memory: whether to aggregate outputs on GPU (None=auto, True=GPU, False=CPU)
        perform_everything_on_device: Optional[bool] = None,
        # Performance-related parameters (None uses Predictor defaults)
        window_count_threshold: Optional[int] = None,
        sparse_downscale_factor: Optional[float] = None,
        sparse_overlap: Optional[float] = None,
        detection_threshold: Optional[float] = None,
        # Margin (in voxels) added to ROI obtained from sparse search
        sparse_bbox_margin_voxels: Optional[int] = None,
        # Physical ROI size (mm) used by the sparse-search-only Predictor (DBSCAN)
        # If None, use Predictor defaults
        sparse_roi_extent_mm: Optional[Union[float, Sequence[float]]] = None,
        dense_overlap: Optional[float] = None,
        use_gaussian: Optional[bool] = None,
        # ROI re-cropping margins (per axis)
        refine_threshold: Optional[float] = None,
        refine_margin_voxels_z: Optional[int] = None,
        refine_margin_voxels_xy: Optional[int] = None,
        # Restrict BBox refinement after dense inference to SI (z) axis only (default: disabled)
        refine_z_only: Optional[bool] = None,
        # ROI size control (SI guard/min lengths): None uses Predictor defaults
        limit_si_extent: Optional[bool] = None,
        max_si_extent_mm: Optional[float] = None,
        min_si_extent_mm: Optional[float] = None,
        min_xy_extent_mm: Optional[float] = None,
        si_axis: Optional[int] = None,
        refine_roi_after_dense: Optional[bool] = None,
        # Root for original annotations (directory to search .nii.annotations.json)
        series_niix_dir: Optional[str | Path] = None,
        # Output format (npz: compressed as before, npy: raw array)
        save_format: str = "npz",
        # Fold aggregation: "logit_mean" (legacy), "prob_mean", "prob_max", "noisy_or"
        fold_agg: str = "logit_mean",
        # Enable foreground-first gate (applied only when labeling)
        foreground_first: bool = True,
        # Single threshold to prefer foreground (if max non-background prob >= this value)
        fg_threshold: float = 0.30,
        # Enable orientation correction using vessel-seg sparse-search result
        enable_orientation_correction: bool = True,
        orientation_solver_weights: Optional[Tuple[float, float, float]] = None,
        sparse_model_fold: str | int = "all",
        # TensorRT integration (optional). If engine root dir is provided, discover engines automatically.
        trt_dir: Optional[str] = None,
        trt_fp16: bool = True,
    ):
        """
        Args:
            model_path: Path to nnUNet model
            additional_dense_model_paths: Paths to additional nnUNet models for dense inference
            sparse_model_path: Path to nnUNet model used only for sparse search (defaults to model_path)
            folds: Fold indices or identifiers (e.g., 0, 1, "all")
            device: Device to use (single device; backward compatibility)
            use_sparse_search: Whether to use sparse search (False sets window_count_threshold=inf internally)
            verbose: Verbose logging for this class; delegated to each Predictor as well
            use_mirroring: Mirroring (TTA) during inference; None uses Predictor default
            devices: Multi-GPU indices (e.g., (0,1,2)); auto-detected when unspecified

        Notes:
            - Mutable parameters (e.g., thresholds for sparse search) are delegated to each
              Predictor's __init__ default when set to None; this class does not impose its own defaults.
        """
        self.model_dir = Path(model_path).expanduser()
        if not self.model_dir.exists():
            raise ValueError(f"Model directory does not exist: {self.model_dir}")
        self.model_path = str(self.model_dir)

        # Configure sparse-search-only model (defaults to same as aggregate model if unspecified)
        sparse_dir = Path(sparse_model_path).expanduser() if sparse_model_path is not None else self.model_dir
        self.sparse_model_path = str(sparse_dir)
        self._use_sparse_model_override = sparse_model_path is not None
        # TensorRT settings
        self.trt_dir = str(trt_dir) if trt_dir else None
        self.trt_fp16 = bool(trt_fp16)

        self.available_folds: List[FoldKey] = _discover_available_folds(self.model_dir)
        if len(self.available_folds) == 0:
            raise ValueError(f"No fold_* found in model directory: {self.model_dir}")
        if verbose:
            print(f"Available folds: {self.available_folds}")

        self.folds = _normalize_fold_list(folds, self.available_folds)
        self.sparse_model_fold = _normalize_fold_token(sparse_model_fold)
        if self._use_sparse_model_override:
            if not sparse_dir.exists():
                raise ValueError(f"Sparse-model directory does not exist: {sparse_dir}")
            sparse_available = _discover_available_folds(sparse_dir)
            if sparse_available and self.sparse_model_fold not in sparse_available:
                raise ValueError(
                    f"Fold {self.sparse_model_fold} specified for sparse model is unavailable: {sparse_available}"
                )
        self.torch_compile = torch_compile
        # Device used for main processing (preprocess and sparse search)
        self.verbose = verbose
        # Track whether sparse search is enabled (used in predict_single_volume)
        self.use_sparse_search = bool(use_sparse_search)
        self.enable_orientation_correction = bool(enable_orientation_correction)
        if orientation_solver_weights is not None and len(orientation_solver_weights) != 3:
            raise ValueError("orientation_solver_weights must be a 3-element tuple")
        self.orientation_solver_weights = (
            tuple(float(x) for x in orientation_solver_weights)
            if orientation_solver_weights is not None
            else None
        )
        self._orientation_default_weights = (0.5, 0.3, 0.2)
        # Resolve multiple GPU specification
        if devices is None:
            # Unspecified: use all available GPUs; empty list if no GPU
            if torch.cuda.is_available():
                try:
                    n = torch.cuda.device_count()
                    devices_list: List[int] = list(range(n)) if n > 0 else []
                except Exception:
                    devices_list = []
            else:
                devices_list = []
        else:
            devices_list = list(devices)

        # Main device (at least one). Use CPU when no GPU
        if len(devices_list) > 0 and torch.cuda.is_available():
            self.main_device = torch.device(f"cuda:{int(devices_list[0])}")
        else:
            # Backward compatibility: reflect device argument
            self.main_device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize Predictor
        if verbose:
            print("Initializing adaptive sparse-search Predictor...")
            if torch.cuda.is_available():
                print(f"GPUs: {','.join(str(i) for i in devices_list) if devices_list else 'auto(single)'}")
            print(f"Main device: {self.main_device}")

        # For multiple folds, create a Predictor per fold (distribute devices)
        # Respect AdaptiveSparsePredictor __init__ defaults by omitting None kwargs
        base_kwargs_common: Dict[str, object] = {}
        # Sparse search on/off
        if use_sparse_search:
            if window_count_threshold is not None:
                base_kwargs_common["window_count_threshold"] = int(window_count_threshold)
        else:
            # Disable sparse search by using effectively infinite threshold
            base_kwargs_common["window_count_threshold"] = float("inf")
        # Sparse/detection/overlap parameters
        if sparse_downscale_factor is not None:
            base_kwargs_common["sparse_downscale_factor"] = float(sparse_downscale_factor)
        if sparse_overlap is not None:
            base_kwargs_common["sparse_overlap"] = float(sparse_overlap)
        if detection_threshold is not None:
            base_kwargs_common["detection_threshold"] = float(detection_threshold)
        if sparse_bbox_margin_voxels is not None:
            base_kwargs_common["sparse_bbox_margin_voxels"] = int(sparse_bbox_margin_voxels)
        if dense_overlap is not None:
            base_kwargs_common["dense_overlap"] = float(dense_overlap)
        # ROI size control (SI guard/min length/axis)
        if limit_si_extent is not None:
            base_kwargs_common["limit_si_extent"] = bool(limit_si_extent)
        if max_si_extent_mm is not None:
            base_kwargs_common["max_si_extent_mm"] = float(max_si_extent_mm)
        if min_si_extent_mm is not None:
            base_kwargs_common["min_si_extent_mm"] = float(min_si_extent_mm)
        if min_xy_extent_mm is not None:
            base_kwargs_common["min_xy_extent_mm"] = float(min_xy_extent_mm)
        if si_axis is not None:
            base_kwargs_common["si_axis"] = int(si_axis)
        # Others
        if use_gaussian is not None:
            base_kwargs_common["use_gaussian"] = bool(use_gaussian)
        if use_mirroring is not None:
            base_kwargs_common["use_mirroring"] = bool(use_mirroring)
        # Explicitly pass logging settings
        base_kwargs_common["verbose"] = bool(verbose)
        base_kwargs_common["allow_tqdm"] = False

        # ROI re-extraction after dense inference is controlled on this class
        self.refine_roi_after_dense = True if refine_roi_after_dense is None else bool(refine_roi_after_dense)

        self.additional_dense_model_paths = (
            tuple(str(p) for p in additional_dense_model_paths)
            if additional_dense_model_paths is not None
            else tuple()
        )
        self._primary_model_key = "primary"
        self._dense_model_configs: Dict[str, Dict[str, Any]] = {}
        self._extra_model_keys: List[str] = []
        self._dense_model_devices: Dict[str, torch.device] = {}

        def _resolve_dense_device(model_index: int) -> torch.device:
            if len(devices_list) > 0 and torch.cuda.is_available():
                dev_idx = devices_list[model_index % len(devices_list)]
                return torch.device(f"cuda:{int(dev_idx)}")
            return self.main_device

        def _instantiate_dense_predictors(dense_device: torch.device) -> List[AdaptiveSparsePredictor]:
            preds: List[AdaptiveSparsePredictor] = []
            if perform_everything_on_device is None:
                pei = dense_device.type == "cuda"
            else:
                pei = bool(perform_everything_on_device)

            for _ in self.folds:
                preds.append(
                    AdaptiveSparsePredictor(
                        **base_kwargs_common,
                        perform_everything_on_device=pei,
                        device=dense_device,
                    )
                )
            return preds

        # Use AdaptiveSparsePredictor for normal per-fold inference
        primary_device = _resolve_dense_device(0)
        self.predictors = _instantiate_dense_predictors(primary_device)
        self._dense_model_configs[self._primary_model_key] = {
            "model_path": self.model_path,
            "predictors": self.predictors,
            "save_suffix": None,
            "device": primary_device,
        }
        self._dense_model_devices[self._primary_model_key] = primary_device

        if not self.predictors:
            raise RuntimeError("Failed to create Predictors for dense inference")

        self.refine_threshold = (
            float(refine_threshold) if refine_threshold is not None else _DEFAULT_REFINE_THRESHOLD
        )
        self.refine_margin_voxels_z = (
            int(refine_margin_voxels_z) if refine_margin_voxels_z is not None else _DEFAULT_REFINE_MARGIN_Z
        )
        self.refine_margin_voxels_xy = (
            int(refine_margin_voxels_xy) if refine_margin_voxels_xy is not None else _DEFAULT_REFINE_MARGIN_XY
        )
        # Flag to refine only along z (SI) axis (default True when None)
        self.refine_z_only = False if refine_z_only is None else bool(refine_z_only)
        self.si_axis = int(si_axis) if si_axis is not None else 0

        for offset, extra_path in enumerate(self.additional_dense_model_paths, start=1):
            dense_device = _resolve_dense_device(offset)
            extra_predictors = _instantiate_dense_predictors(dense_device)
            save_suffix = self._make_model_save_suffix(extra_path, offset - 1)
            key = f"extra_{offset - 1}"
            self._dense_model_configs[key] = {
                "model_path": extra_path,
                "predictors": extra_predictors,
                "save_suffix": save_suffix,
                "device": dense_device,
            }
            self._extra_model_keys.append(key)
            self._dense_model_devices[key] = dense_device

        # Build sparse-only Predictor (use fold 0 when separate path specified)
        if self._use_sparse_model_override:
            sparse_kwargs = dict(base_kwargs_common)
            # Explicitly disable sparse rescale
            sparse_kwargs["sparse_downscale_factor"] = 1.0
            # If physical ROI size (mm) given, pass to DBSCAN args
            if sparse_roi_extent_mm is not None:
                sparse_kwargs["roi_extent_mm"] = sparse_roi_extent_mm
            sparse_device = self.main_device
            if perform_everything_on_device is None:
                sparse_pei = sparse_device.type == "cuda"
            else:
                sparse_pei = bool(perform_everything_on_device)
            # Use DBSCANAdaptiveSparsePredictor for sparse-only
            self.sparse_predictor = DBSCANAdaptiveSparsePredictor(
                **sparse_kwargs,
                perform_everything_on_device=sparse_pei,
                device=sparse_device,
            )
        else:
            self.sparse_predictor = self.predictors[0] if self.predictors else None

        # Initialize SimpleITKIO
        self.io_handler = SimpleITKIOWithReorient()

        # Load models for each fold
        self._load_model()

        # Normalize formats
        def _normalize_format(value: str, name: str) -> str:
            fmt = value.lower()
            if fmt not in {"npz", "npy"}:
                raise ValueError(f"{name} must be 'npz' or 'npy': {value}")
            return fmt

        self.save_format = _normalize_format(save_format, "save_format")

        # Normalize fold aggregation method
        def _normalize_fold_agg(value: str) -> str:
            v = value.lower()
            if v not in {"logit_mean", "prob_mean", "prob_max", "noisy_or"}:
                raise ValueError(
                    f"fold_agg must be one of 'logit_mean'|'prob_mean'|'prob_max'|'noisy_or': {value}"
                )
            return v

        self.fold_agg = _normalize_fold_agg(fold_agg)
        # Labeling behavior (foreground-first gate)
        self.foreground_first = bool(foreground_first)
        self.fg_threshold = float(fg_threshold)

        # Root for original annotations (napari-style default)
        # Fallback to legacy default if unspecified
        self.series_niix_dir = (
            Path(series_niix_dir) if series_niix_dir is not None else Path("/workspace/data/series_niix")
        )

        # Filenames (constants)
        self.FN_PROB = "prob.npz" if self.save_format == "npz" else "prob.npy"
        self.FN_SEG = "seg.npz" if self.save_format == "npz" else "seg.npy"
        self.FN_ROI = "roi_data.npz" if self.save_format == "npz" else "roi_data.npy"
        self.FN_XFORM = "transform.json"
        self.FN_ROI_ANNS = "roi_annotations.json"
        self.FN_ORIENTATION_CASES = "orientation_corrections.json"

    # ===== Internal utilities =====
    def _case_dir(self, save_dir: str, base_name: str) -> Path:
        """Return case-level output directory path."""
        return Path(save_dir) / base_name

    @staticmethod
    def _make_model_save_suffix(model_path: str, index: int) -> str:
        """Generate identifier suffix from model path for extra models."""

        name = Path(model_path).name
        if not name:
            name = f"model_{index}"
        sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", name)
        if not sanitized:
            sanitized = f"model_{index}"
        return sanitized

    def _extra_seg_filename(self, suffix: str) -> str:
        """Generate segmentation filename for extra models."""

        base = f"seg_{suffix or 'extra'}"
        if self.save_format == "npz":
            return f"{base}.npz"
        return f"{base}.npy"

    def _extract_spacing_for_predictor(
        self, predictor: AdaptiveSparsePredictor, preprocessed: Dict[str, Any]
    ) -> Optional[Tuple[float, float, float]]:
        """Obtain spatial resolution expected by the Predictor."""

        if not isinstance(preprocessed, dict):
            return None
        props = preprocessed.get("data_properties")
        if not isinstance(props, dict):
            return None
        spacing_prop = props.get("spacing")
        transpose_forward = list(getattr(predictor.plans_manager, "transpose_forward", (0, 1, 2)))
        if isinstance(spacing_prop, (list, tuple)) and len(spacing_prop) >= len(transpose_forward):
            return tuple(float(spacing_prop[int(idx)]) for idx in transpose_forward[:3])
        return None

    def _already_processed(self, case_dir: Path, return_probabilities: bool) -> bool:
        """Decide whether case is already processed based on required files."""
        if not case_dir.exists():
            return False
        required = [case_dir / self.FN_ROI, case_dir / self.FN_XFORM]
        if return_probabilities:
            required.append(case_dir / self.FN_PROB)
        else:
            required.append(case_dir / self.FN_SEG)
        for p in required:
            if not p.exists():
                return False

        for key in self._extra_model_keys:
            cfg = self._dense_model_configs.get(key)
            if not cfg:
                continue
            predictors = cfg.get("predictors", [])
            if not predictors:
                continue
            suffix = cfg.get("save_suffix")
            seg_path = case_dir / self._extra_seg_filename(suffix or key)
            if not seg_path.exists():
                return False

        return True

    @staticmethod
    def _load_orientation_perm_signs_from_transform(
        transform_path: Path,
    ) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Extract orientation-correction perm/signs from transform.json."""

        if not transform_path.exists():
            return None
        try:
            with open(transform_path, "r", encoding="utf-8") as f:
                transform_data = json.load(f)
        except Exception:
            return None
        if not isinstance(transform_data, dict):
            return None
        meta = transform_data.get("orientation_correction")
        return _extract_perm_signs_from_meta(meta)

    def _save_model_output(
        self,
        *,
        target_dir: Path,
        base_name: str,
        result: DenseModelResult,
        annotations: List[Dict],
        collect_annotation_stats: bool,
        cases_with_annotations_outside_roi: Optional[List[Dict[str, int]]],
    ) -> None:
        """Save inference results for a single model."""

        prediction = result.prediction
        roi_data = result.roi
        transform_info = result.transform_info
        return_probabilities = bool(result.return_probabilities)

        if prediction is None or roi_data is None or transform_info is None:
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        if return_probabilities:
            prob_path = target_dir / self.FN_PROB
            if isinstance(prediction, torch.Tensor):
                prediction_np = prediction.cpu().numpy()
            else:
                prediction_np = prediction
            prob_to_save = prediction_np.astype(np.float16, copy=False)
            if self.save_format == "npz":
                np.savez_compressed(prob_path, probabilities=prob_to_save)
            else:
                np.save(prob_path, prob_to_save)
        else:
            seg_path = target_dir / self.FN_SEG
            if isinstance(prediction, torch.Tensor):
                prediction_np = prediction.cpu().numpy()
            else:
                prediction_np = prediction
            seg_arr = prediction_np.astype(np.uint8, copy=False)
            if self.save_format == "npz":
                np.savez_compressed(seg_path, segmentation=seg_arr)
            else:
                np.save(seg_path, seg_arr)

        roi_path = target_dir / self.FN_ROI
        if isinstance(roi_data, torch.Tensor):
            roi_np = roi_data.cpu().numpy()
        else:
            roi_np = roi_data
        if isinstance(roi_np, np.ndarray) and roi_np.dtype == np.float32:
            roi_to_save = roi_np.astype(np.float16, copy=False)
        else:
            roi_to_save = roi_np
        if self.save_format == "npz":
            np.savez_compressed(roi_path, roi=roi_to_save)
        else:
            np.save(roi_path, roi_to_save)

        xform_path = target_dir / self.FN_XFORM
        with open(xform_path, "w") as f:
            json.dump(transform_info, f, indent=2)

        if annotations:
            roi_annotations, total_cnt, inside_cnt = self._transform_annotations_to_roi(
                annotations, transform_info
            )
            ann_out_path = target_dir / self.FN_ROI_ANNS
            with open(ann_out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "case_id": base_name,
                        "total_annotations": total_cnt,
                        "roi_annotations_count": inside_cnt,
                        "roi_annotations": roi_annotations,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            if (
                collect_annotation_stats
                and cases_with_annotations_outside_roi is not None
                and inside_cnt < total_cnt
            ):
                cases_with_annotations_outside_roi.append(
                    {
                        "case_id": base_name,
                        "total": total_cnt,
                        "inside_roi": inside_cnt,
                        "outside_roi": total_cnt - inside_cnt,
                    }
                )

        if isinstance(result.prediction, torch.Tensor):
            result.prediction = None
        if isinstance(result.roi, torch.Tensor):
            result.roi = None

    def _dense_result_to_segmentation(self, result: DenseModelResult) -> Optional[np.ndarray]:
        """Extract only the segmentation array from a DenseModelResult."""

        prediction = result.prediction
        if prediction is None:
            return None

        if isinstance(prediction, torch.Tensor):
            tensor = prediction.detach().cpu()
        else:
            tensor = torch.as_tensor(prediction)

        if result.return_probabilities:
            probs = tensor.float()
            if probs.dim() < 1:
                return None
            if self.foreground_first:
                non_bg = probs[1:]
                fg_prob, fg_idx = torch.max(non_bg, dim=0)
                seg_tensor = torch.where(
                    fg_prob >= self.fg_threshold,
                    (fg_idx + 1).to(dtype=torch.long),
                    torch.zeros_like(fg_idx, dtype=torch.long),
                )
            else:
                seg_tensor = torch.argmax(probs, dim=0)
        else:
            seg_tensor = tensor.to(dtype=torch.long)

        seg_tensor = seg_tensor.to(dtype=torch.uint8)
        return seg_tensor.cpu().numpy()

    def _save_extra_segmentation(
        self,
        *,
        target_dir: Path,
        suffix: str,
        result: DenseModelResult,
    ) -> None:
        """Save segmentation only for an additional dense model."""

        seg_np = self._dense_result_to_segmentation(result)
        if seg_np is None:
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._extra_seg_filename(suffix)
        seg_path = target_dir / file_name
        if self.save_format == "npz":
            np.savez_compressed(seg_path, segmentation=seg_np)
        else:
            np.save(seg_path, seg_np)

    # ===== Annotation utilities (napari-equivalent coordinate transform) =====
    def _load_annotations_for_case(self, case_id: str) -> List[Dict]:
        """Collect annotations from *.nii.annotations.json for the given case_id."""
        annotations: List[Dict] = []
        series_dir = self.series_niix_dir / case_id.replace("_0000", "")
        if not series_dir.exists():
            return annotations
        for json_file in series_dir.glob("*.nii.annotations.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "annotations" in data:
                        anns = data.get("annotations", [])
                        if isinstance(anns, list):
                            annotations.extend(anns)
            except Exception:
                # Silently skip failures
                continue
        return annotations

    def _transform_annotations_to_roi(
        self, annotations: List[Dict], transform: Dict
    ) -> Tuple[List[Dict], int, int]:
        """Transform annotations to the ROI space and keep those inside the ROI.

        Follows the procedure of napari_vessel_segmentation_visualizer.py.

        Returns:
            (roi_annotations, total_count, inside_count)
        """
        if not annotations or not transform:
            return annotations, len(annotations), len(annotations)

        roi_annotations: List[Dict] = []

        transpose_forward = transform.get("transpose_forward", [0, 1, 2])
        scale_factors = transform.get("scale_factors_orig2net", [1.0, 1.0, 1.0])

        roi_bbox_network = transform.get("roi_bbox_network_refined") or transform.get("roi_bbox_network", [])
        roi_offset_network = transform.get("roi_offset_network_refined") or transform.get(
            "roi_offset_network", [0, 0, 0]
        )
        bbox_used_for_cropping = transform.get("bbox_used_for_cropping", [])

        if not roi_bbox_network:
            # If conversion info is insufficient, return as-is
            return annotations, len(annotations), 0

        roi_shape = [roi_bbox_network[i][1] - roi_bbox_network[i][0] for i in range(3)]

        for ann in annotations:
            # .annotations.json stores (nifti_x, nifti_y, nifti_z)
            # Align to nnU-Net array dims (Z, Y, X)
            orig_coords = [
                float(ann.get("nifti_z", 0)),
                float(ann.get("nifti_y", 0)),
                float(ann.get("nifti_x", 0)),
            ]

            # 1) Transpose
            transposed_coords = [orig_coords[transpose_forward[i]] for i in range(3)]

            # 2) Subtract crop offset if present
            if bbox_used_for_cropping:
                for i in range(3):
                    if i < len(bbox_used_for_cropping):
                        transposed_coords[i] -= bbox_used_for_cropping[i][0]

            # 3) Resample (after transpose -> network space)
            network_coords = [float(transposed_coords[i] + 0.5) * float(scale_factors[i]) for i in range(3)]

            orientation_meta = transform.get("orientation_correction")
            if orientation_meta:
                perm_list = orientation_meta.get("perm", [])
                signs_list = orientation_meta.get("signs", [])
                shape_before_list = orientation_meta.get("shape_before", [])
                if (
                    isinstance(perm_list, list)
                    and isinstance(signs_list, list)
                    and isinstance(shape_before_list, list)
                    and len(perm_list) == 3
                    and len(signs_list) == 3
                    and len(shape_before_list) == 3
                ):
                    network_coords = _apply_orientation_to_coords(
                        network_coords,
                        tuple(int(x) for x in shape_before_list),
                        tuple(int(x) for x in perm_list),
                        tuple(int(x) for x in signs_list),
                    )

            # 4) Subtract ROI offset (network -> ROI space)
            roi_coords = [float(network_coords[i]) - float(roi_offset_network[i]) for i in range(3)]

            # Check inside ROI
            inside = all(0 <= roi_coords[i] < roi_shape[i] for i in range(3))
            if inside:
                roi_ann = ann.copy()
                roi_ann["roi_z"] = roi_coords[0]
                roi_ann["roi_y"] = roi_coords[1]
                roi_ann["roi_x"] = roi_coords[2]
                roi_annotations.append(roi_ann)

        return roi_annotations, len(annotations), len(roi_annotations)

    @staticmethod
    def _roi_bbox_to_original_bounds(
        transform_info: Dict[str, Any], roi_bbox: Tuple[slice, ...]
    ) -> List[Tuple[float, float]]:
        """Convert ROI slices from sparse search to continuous ranges in the original array space."""

        orientation_meta = transform_info.get("orientation_correction")
        adjusted_roi = roi_bbox

        transpose_forward = transform_info.get("transpose_forward", list(range(len(roi_bbox))))
        bbox_crop = transform_info.get("bbox_used_for_cropping", []) or []
        scale_factors = transform_info.get("scale_factors_orig2net", [])

        if orientation_meta:
            perm_list = orientation_meta.get("perm", [])
            bbox_crop = [bbox_crop[perm] for perm in perm_list]
            scale_factors = [scale_factors[perm] for perm in perm_list]

        bounds_transposed: List[Tuple[float, float]] = []
        for axis, slc in enumerate(adjusted_roi):
            scale = float(scale_factors[axis]) if axis < len(scale_factors) else 1.0
            if scale == 0.0:
                scale = 1.0
            crop_start = 0.0
            if axis < len(bbox_crop) and bbox_crop[axis]:
                crop_start = float(bbox_crop[axis][0])
            start = float(slc.start) / scale + crop_start
            stop = float(slc.stop) / scale + crop_start
            bounds_transposed.append((start, stop))

        dim = len(roi_bbox)
        bounds_original: List[Tuple[float, float]] = [(0.0, 0.0)] * dim
        for axis in range(dim):
            orig_axis = transpose_forward[axis] if axis < len(transpose_forward) else axis
            bounds_original[orig_axis] = bounds_transposed[axis]

        return bounds_original

    @staticmethod
    def _original_bounds_to_roi(
        transform_info: Dict[str, Any],
        bounds_original: List[Tuple[float, float]],
        data_shape: Tuple[int, ...],
    ) -> Tuple[slice, ...]:
        """Convert ranges in the original array space to ROI slices for the given plan."""

        dim = len(bounds_original)
        transpose_forward = transform_info.get("transpose_forward", list(range(dim)))
        bbox_crop = transform_info.get("bbox_used_for_cropping", []) or []
        scale_factors = transform_info.get("scale_factors_orig2net", [])

        orientation_meta = transform_info.get("orientation_correction")
        if orientation_meta:
            perm_list = orientation_meta.get("perm", list(range(dim)))
            bbox_crop = [bbox_crop[perm] for perm in perm_list]
            scale_factors = [scale_factors[perm] for perm in perm_list]

        roi_slices: List[slice] = []
        for axis in range(dim):
            orig_axis = transpose_forward[axis] if axis < len(transpose_forward) else axis
            bound = bounds_original[orig_axis] if orig_axis < len(bounds_original) else (0.0, 0.0)

            crop_start = 0.0
            if axis < len(bbox_crop) and bbox_crop[axis]:
                crop_start = float(bbox_crop[axis][0])

            start_cropped = bound[0] - crop_start
            stop_cropped = bound[1] - crop_start

            scale = float(scale_factors[axis]) if axis < len(scale_factors) else 1.0
            start_net = start_cropped * scale
            stop_net = stop_cropped * scale

            start_idx = int(math.floor(start_net))
            stop_idx = int(math.ceil(stop_net))

            max_len = data_shape[axis + 1] if axis + 1 < len(data_shape) else data_shape[-1]
            start_idx = max(0, min(start_idx, max_len - 1))
            stop_idx = max(start_idx + 1, min(stop_idx, max_len))

            roi_slices.append(slice(start_idx, stop_idx))

        return tuple(roi_slices)

    def _maybe_apply_orientation_correction(
        self,
        sparse_pred: AdaptiveSparsePredictor,
        sparse_preprocessed: Dict[str, Any],
        sparse_context: Optional[Dict[str, Any]],
        roi_bbox: Tuple[slice, ...],
        min_score: float = 4.3,
    ) -> Optional[OrientationCorrectionResult]:
        """Apply orientation correction from sparse segmentation results and return details.

        Notes:
            - If the estimated score is <= `min_score`, skip correction (return None).
            - Default `min_score` is 4.0.
        """

        if not self.enable_orientation_correction:
            return None
        if sparse_context is None:
            raise RuntimeError("Orientation correction requires sparse-search context")

        num_classes = int(sparse_context.get("num_classes", 0))
        if num_classes != 4:
            raise RuntimeError(
                "Orientation correction requires the sparse-search model to output 4 classes (incl. background)"
            )

        seg_tensor = sparse_context.get("segmentation_highres")
        if seg_tensor is None:
            raise RuntimeError("Failed to obtain sparse-search segmentation")
        seg_tensor = torch.as_tensor(seg_tensor)
        seg_np_full = seg_tensor.detach().cpu().numpy()
        if seg_np_full.ndim != 3:
            raise RuntimeError("Orientation correction requires a 3D segmentation")

        crop_slices_list: List[slice] = []
        for axis, s in enumerate(roi_bbox):
            start = 0 if s.start is None else int(s.start)
            stop = seg_np_full.shape[axis] if s.stop is None else int(s.stop)
            crop_slices_list.append(slice(start, stop))
        crop_slices = tuple(crop_slices_list)
        use_cropped_roi = True
        seg_np_for_orientation = seg_np_full
        try:
            seg_np_crop = seg_np_full[crop_slices]
            if (
                seg_np_crop.size == 0
                or seg_np_crop.shape[0] == 0
                or seg_np_crop.shape[1] == 0
                or seg_np_crop.shape[2] == 0
                or not np.any(seg_np_crop > 0)
            ):
                use_cropped_roi = False
            else:
                seg_np_for_orientation = seg_np_crop
        except Exception:
            use_cropped_roi = False

        props = (
            sparse_preprocessed.get("data_properties", {}) if isinstance(sparse_preprocessed, dict) else {}
        )
        transpose_forward = list(getattr(sparse_pred.plans_manager, "transpose_forward", (0, 1, 2)))

        spacing_zyx: Optional[Tuple[float, float, float]] = None
        spacing_ctx = sparse_context.get("network_spacing") if isinstance(sparse_context, dict) else None
        if isinstance(spacing_ctx, (list, tuple)) and len(spacing_ctx) >= 3:
            spacing_zyx = tuple(float(spacing_ctx[i]) for i in range(3))
        else:
            cfg_spacing = getattr(getattr(sparse_pred, "configuration_manager", None), "spacing", None)
            if isinstance(cfg_spacing, (list, tuple)) and len(cfg_spacing) >= 3:
                spacing_zyx = tuple(float(cfg_spacing[i]) for i in range(3))

        if spacing_zyx is None:
            spacing_prop = props.get("spacing") if isinstance(props, dict) else None
            if isinstance(spacing_prop, (list, tuple)) and len(spacing_prop) >= len(transpose_forward):
                spacing_zyx = tuple(float(spacing_prop[int(idx)]) for idx in transpose_forward[:3])
            else:
                spacing_zyx = (1.0, 1.0, 1.0)

        weights = self.orientation_solver_weights or self._orientation_default_weights
        estimate = estimate_perm_signs_from_vessel3(
            seg_np_for_orientation,
            spacing_zyx=spacing_zyx,
            weights=weights,
        )

        # If score is below threshold, skip correction (safe fallback)
        if not np.isfinite(estimate.score) or float(estimate.score) <= float(min_score):
            perm = (0, 1, 2)
            signs = (1, 1, 1)
        else:
            perm = tuple(int(x) for x in estimate.perm)
            signs = tuple(int(x) for x in estimate.signs)

        shape_before = tuple(int(seg_np_full.shape[i]) for i in range(seg_np_full.ndim))
        shape_after = tuple(shape_before[perm[i]] for i in range(len(perm)))
        roi_bbox_after = _apply_orientation_to_slices(roi_bbox, shape_before, perm, signs)

        if not _orientation_is_identity(perm, signs):
            data_tensor = sparse_preprocessed.get("data") if isinstance(sparse_preprocessed, dict) else None
            if not isinstance(data_tensor, torch.Tensor):
                raise RuntimeError("Sparse preprocessed data is not a Tensor")
            if data_tensor.shape[1:] != shape_before:
                raise RuntimeError("Sparse preprocessed data shape does not match for orientation estimation")
            sparse_preprocessed["data"] = _apply_orientation_to_tensor(data_tensor, perm, signs)

        return OrientationCorrectionResult(
            perm=perm,
            signs=signs,
            estimate=estimate,
            shape_before=shape_before,
            shape_after=shape_after,
            num_classes=num_classes,
            spacing_zyx=spacing_zyx,
            roi_bbox_before=tuple(roi_bbox),
            roi_bbox_after=roi_bbox_after,
            used_cropped_roi=bool(use_cropped_roi),
        )

    @staticmethod
    def _safe_clear_cuda_cache() -> None:
        """Safely clear CUDA cache and run garbage collection."""

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    def _prepare_dense_inputs(
        self,
        *,
        ref_pred: AdaptiveSparsePredictor,
        image: np.ndarray,
        properties: Dict,
        bounds_original: List[Tuple[float, float]],
        roi_bbox_sparse: Tuple[slice, ...],
        orientation_result: Optional[OrientationCorrectionResult],
        sparse_pred: AdaptiveSparsePredictor,
        sparse_preprocessed: Optional[Dict[str, Any]],
        sparse_data: Optional[torch.Tensor],
        shared_cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, Tuple[slice, ...], Optional[Tuple[int, int, int]]]:
        """Prepare preprocessed outputs and ROI info for dense inference."""

        if shared_cache is not None:
            dense_preprocessed = shared_cache.get("preprocessed")
            dense_data = shared_cache.get("data")
            dense_roi_bbox = shared_cache.get("roi_bbox")
            dense_shape_before = shared_cache.get("shape_before")

            if (
                not isinstance(dense_preprocessed, dict)
                or not isinstance(dense_data, torch.Tensor)
                or not isinstance(dense_roi_bbox, tuple)
            ):
                raise RuntimeError("Invalid format of shared dense preprocessed cache")

            return dense_preprocessed, dense_data, dense_roi_bbox, dense_shape_before

        reuse_sparse = ref_pred is sparse_pred and sparse_preprocessed is not None and sparse_data is not None

        if reuse_sparse:
            dense_preprocessed = sparse_preprocessed
            dense_data = sparse_data
            dense_roi_bbox = roi_bbox_sparse
            if orientation_result is not None:
                dense_shape_before = orientation_result.shape_before
            else:
                dense_tensor_cur = (
                    dense_preprocessed.get("data") if isinstance(dense_preprocessed, dict) else None
                )
                if isinstance(dense_tensor_cur, torch.Tensor):
                    dense_shape_before = tuple(
                        int(dense_tensor_cur.shape[i + 1]) for i in range(dense_tensor_cur.dim() - 1)
                    )
                else:
                    dense_shape_before = None
            return dense_preprocessed, dense_data, dense_roi_bbox, dense_shape_before

        dense_preprocessed = ref_pred._preprocess_data(image, properties, seg_from_prev_stage=None)
        dense_tensor_raw = dense_preprocessed.get("data") if isinstance(dense_preprocessed, dict) else None
        if not isinstance(dense_tensor_raw, torch.Tensor):
            raise RuntimeError("Dense preprocessed data is not a Tensor")

        dense_shape_before = tuple(
            int(dense_tensor_raw.shape[i + 1]) for i in range(dense_tensor_raw.dim() - 1)
        )

        if orientation_result is not None and not _orientation_is_identity(
            orientation_result.perm, orientation_result.signs
        ):
            dense_preprocessed["data"] = _apply_orientation_to_tensor(
                dense_tensor_raw,
                orientation_result.perm,
                orientation_result.signs,
            )
            dense_tensor_raw = dense_preprocessed["data"]
            if not isinstance(dense_tensor_raw, torch.Tensor):
                raise RuntimeError("Dense preprocessed data is not a Tensor")

        dense_data = dense_tensor_raw.to(dtype=torch.half)
        dim = dense_data.dim() - 1
        dense_full_roi = tuple(slice(0, int(dense_data.shape[i + 1])) for i in range(dim))
        transform_dense_base = ref_pred._build_transform_info(
            preprocessed=dense_preprocessed,
            data=dense_data,
            roi_bbox=dense_full_roi,
            refined_local_bbox=None,
        )

        if orientation_result is not None and dense_shape_before is not None:
            dense_full_roi_before = tuple(
                slice(0, int(dense_shape_before[i])) for i in range(len(dense_shape_before))
            )
            spacing_override = self._extract_spacing_for_predictor(ref_pred, dense_preprocessed)
            transform_dense_base["orientation_correction"] = orientation_result.to_metadata(
                shape_before=dense_shape_before,
                roi_bbox_before=dense_full_roi_before,
                spacing_zyx=spacing_override,
            )

        dense_roi_bbox = self._original_bounds_to_roi(
            transform_dense_base,
            bounds_original,
            tuple(int(x) for x in dense_data.shape),
        )

        return dense_preprocessed, dense_data, dense_roi_bbox, dense_shape_before

    def _run_dense_inference(
        self,
        *,
        predictors: Sequence[AdaptiveSparsePredictor],
        dense_preprocessed: Dict[str, Any],
        dense_data: torch.Tensor,
        dense_roi_bbox: Tuple[slice, ...],
        dense_shape_before: Optional[Tuple[int, int, int]],
        orientation_result: Optional[OrientationCorrectionResult],
        return_probabilities: bool,
        tracker: Optional[VesselSegmentationStageTracker],
        refined_local_bbox_override: Optional[Tuple[slice, ...]] = None,
    ) -> Tuple[Optional[DenseModelResult], bool]:
        """Run dense inference step and return results; set stop flag when needed."""

        if not predictors:
            raise RuntimeError("Predictors for dense inference are not initialized")

        primary_pred = predictors[0]

        roi_slices = (slice(None),) + dense_roi_bbox
        roi_data_cpu = dense_data[roi_slices]
        roi_data = roi_data_cpu.to(device=primary_pred.device)

        if tracker is not None:
            tracker.mark(
                VesselSegmentationStage.ROI_CROP,
                {
                    "roi_shape": _shape_tuple(roi_data.shape),
                    "roi_bbox_dense": _slices_to_ranges(dense_roi_bbox),
                },
            )
            if tracker.should_stop():
                del dense_data
                del roi_data
                del dense_preprocessed
                self._safe_clear_cuda_cache()
                return None, True

        refined_local_bbox = refined_local_bbox_override

        if refined_local_bbox is not None:
            slices_override = (slice(None),) + refined_local_bbox
            roi_data = roi_data[slices_override]
            roi_data_cpu = roi_data_cpu[slices_override]

        roi_for_inference = roi_data

        logits_sum: Optional[torch.Tensor] = None
        per_fold_logits: List[torch.Tensor] = []

        for idx, pred in enumerate(predictors):
            if roi_for_inference.device == pred.device:
                roi_input = roi_for_inference
            else:
                roi_input = roi_for_inference.to(pred.device)

            logits = pred._predict_logits(roi_input)
            lg = logits.detach()

            per_fold_logits.append(lg)
            if logits_sum is None:
                logits_sum = lg
            else:
                logits_sum = logits_sum + lg

            if roi_input is not roi_for_inference:
                del roi_input

        avg_logits = logits_sum / max(1, len(predictors))

        if tracker is not None:
            tracker.mark(
                VesselSegmentationStage.FOLD_INFER,
                {"fold_count": len(predictors), "logit_shape": _shape_tuple(avg_logits.shape)},
            )
            if tracker.should_stop():
                del dense_data
                del roi_data
                del dense_preprocessed
                self._safe_clear_cuda_cache()
                return None, True

        if refined_local_bbox_override is None and self.refine_roi_after_dense and refined_local_bbox is None:
            dim = avg_logits.dim() - 1
            si_ax = int(self.si_axis) if self.si_axis is not None else 0
            xy_margin = int(self.refine_margin_voxels_xy)
            z_margin = int(self.refine_margin_voxels_z)
            margins = [xy_margin] * dim
            if 0 <= si_ax < dim:
                margins[si_ax] = z_margin
            margin_tuple = tuple(margins)

            refined_local_bbox = primary_pred._refine_local_bbox_from_logits(
                avg_logits.float(), float(self.refine_threshold), margin_tuple
            )

            # Option: refine only along z (SI), keep XY as original ROI range
            if refined_local_bbox is not None and self.refine_z_only:
                # Fix XY to full range based on spatial dims of avg_logits (excluding C)
                full_sizes = [int(avg_logits.shape[1 + i]) for i in range(dim)]
                refined_list: List[slice] = []
                for ax in range(dim):
                    if ax == si_ax and 0 <= si_ax < dim:
                        # Apply detection for z (SI) axis
                        refined_list.append(refined_local_bbox[ax])
                    else:
                        # XY axes cover whole original ROI
                        refined_list.append(slice(0, full_sizes[ax]))
                refined_local_bbox = tuple(refined_list)

        refined_applied = refined_local_bbox is not None

        if refined_local_bbox is not None and refined_local_bbox_override is None:
            # If override exists, ROI already trimmed; do not reapply here
            slices_refined = (slice(None),) + refined_local_bbox
            avg_logits = avg_logits[slices_refined]
            roi_data = roi_data[slices_refined]
            roi_data_cpu = roi_data_cpu[slices_refined]
            per_fold_logits = [lg[slices_refined] for lg in per_fold_logits]

        if tracker is not None:
            tracker.mark(
                VesselSegmentationStage.REFINE,
                {
                    "applied": refined_applied,
                    "local_bbox": _slices_to_ranges(refined_local_bbox) if refined_local_bbox else None,
                },
            )
            if tracker.should_stop():
                del dense_data
                del roi_data
                del dense_preprocessed
                self._safe_clear_cuda_cache()
                return None, True

        probs_agg: Optional[torch.Tensor] = None
        fa = self.fold_agg
        if fa == "logit_mean":
            probs_agg = torch.softmax(avg_logits.float(), dim=0)
        elif fa in ("prob_mean", "prob_max", "noisy_or"):
            running: Optional[torch.Tensor] = None
            for lg in per_fold_logits:
                p = torch.softmax(lg.float(), dim=0)
                if fa == "prob_mean":
                    running = p if running is None else running + p
                elif fa == "prob_max":
                    running = p if running is None else torch.maximum(running, p)
                elif fa == "noisy_or":
                    if running is None:
                        running = 1.0 - (1.0 - p)
                    else:
                        running = 1.0 - (1.0 - running) * (1.0 - p)
            if fa == "prob_mean" and running is not None:
                probs_agg = running / max(1, len(per_fold_logits))
            else:
                probs_agg = running
        else:
            probs_agg = torch.softmax(avg_logits.float(), dim=0)

        if return_probabilities:
            pred = probs_agg
        else:
            if self.foreground_first:
                if probs_agg is None:
                    probs_agg = torch.softmax(avg_logits.float(), dim=0)
                non_bg = probs_agg[1:]
                fg_prob, fg_idx = torch.max(non_bg, dim=0)
                pred = torch.where(
                    fg_prob >= self.fg_threshold,
                    (fg_idx + 1).to(dtype=torch.long),
                    torch.zeros_like(fg_idx, dtype=torch.long),
                )
            else:
                pred = torch.argmax(avg_logits, dim=0)

        roi_data = (roi_data - roi_data.mean()) / (roi_data.std() + 1e-6)

        transform_info = primary_pred._build_transform_info(
            preprocessed=dense_preprocessed,
            data=dense_data,
            roi_bbox=dense_roi_bbox,
            refined_local_bbox=(refined_local_bbox if refined_applied else None),
        )

        if orientation_result is not None and dense_shape_before is not None:
            dense_roi_bbox_before = _invert_orientation_on_slices(
                dense_roi_bbox,
                dense_shape_before,
                orientation_result.perm,
                orientation_result.signs,
            )
            spacing_override = self._extract_spacing_for_predictor(primary_pred, dense_preprocessed)
            transform_info["orientation_correction"] = orientation_result.to_metadata(
                shape_before=dense_shape_before,
                roi_bbox_before=dense_roi_bbox_before,
                roi_bbox_after=dense_roi_bbox,
                spacing_zyx=spacing_override,
            )

        result = DenseModelResult(
            prediction=pred,
            roi=roi_data,
            transform_info=transform_info,
            return_probabilities=return_probabilities,
            refined_local_bbox=(refined_local_bbox if refined_applied else None),
        )

        del avg_logits
        del per_fold_logits
        del roi_data_cpu

        self._safe_clear_cuda_cache()

        return result, False

    # ===== Helper: run multiple models sequentially/parallel =====
    def _run_models_concurrently(
        self,
        *,
        keys: List[str],
        dense_preprocessed: Dict[str, Any],
        dense_data: torch.Tensor,
        dense_roi_bbox: Tuple[slice, ...],
        dense_shape_before: Optional[Tuple[int, int, int]],
        orientation_result: Optional[OrientationCorrectionResult],
        return_probabilities: bool,
        stage_tracker: Optional["VesselSegmentationStageTracker"],
    ) -> Tuple[Optional[DenseModelResult], Dict[str, DenseModelResult], bool]:
        """Run additional models after the primary one and align downstream on a common ROI."""

        results: Dict[str, DenseModelResult] = {}
        primary_key = self._primary_model_key
        model_cfgs: Dict[str, Dict[str, Any]] = {k: self._dense_model_configs.get(k, {}) for k in keys}

        primary_cfg = model_cfgs.get(primary_key)
        if not primary_cfg:
            raise RuntimeError("Primary dense model configuration not found")

        primary_preds = primary_cfg.get("predictors", [])
        primary_result, stop_requested = self._run_dense_inference(
            predictors=primary_preds,
            dense_preprocessed=dense_preprocessed,
            dense_data=dense_data,
            dense_roi_bbox=dense_roi_bbox,
            dense_shape_before=dense_shape_before,
            orientation_result=orientation_result,
            return_probabilities=return_probabilities,
            tracker=stage_tracker,
        )

        if primary_result is not None:
            results[primary_key] = primary_result

        if stop_requested:
            return primary_result, results, True

        refined_override = None
        if primary_result is not None and self.refine_roi_after_dense:
            refined_override = primary_result.refined_local_bbox

        for key in keys:
            if key == primary_key:
                continue
            cfg = model_cfgs.get(key)
            if not cfg:
                continue
            preds = cfg.get("predictors", [])
            if not preds:
                continue
            res, _ = self._run_dense_inference(
                predictors=preds,
                dense_preprocessed=dense_preprocessed,
                dense_data=dense_data,
                dense_roi_bbox=dense_roi_bbox,
                dense_shape_before=dense_shape_before,
                orientation_result=orientation_result,
                return_probabilities=return_probabilities,
                tracker=None,
                refined_local_bbox_override=refined_override,
            )
            if res is not None:
                results[key] = res

        return primary_result, results, False

    def _load_model(self):
        """Load models for each fold (first call only)."""
        start_time = time.time()

        # Helper to auto-resolve TensorRT engines
        def _try_enable_trt(pred, fold_value):
            try:
                if self.trt_dir is None:
                    return
                # Defensive: check attributes
                trainer_name = getattr(pred, "trainer_name", None)
                configuration_name = getattr(pred, "configuration_name", None)
                plans_name = getattr(getattr(pred, "plans_manager", None), "plans_name", None)
                if not (trainer_name and configuration_name and plans_name):
                    return
                identifier = f"{trainer_name}__{plans_name}__{configuration_name}"
                engine_name = f"model_{'fp16' if self.trt_fp16 else 'fp32'}.engine"
                engine_path = os.path.join(self.trt_dir, identifier, f"fold_{fold_value}", engine_name)
                if os.path.isfile(engine_path):
                    pred.enable_tensorrt(engine_path, enforce_half_output=True)
                else:
                    if self.verbose:
                        print(f"TensorRT engine not found: {engine_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to enable TensorRT: {e}")

        for key, cfg in self._dense_model_configs.items():
            model_dir = cfg["model_path"]
            predictors = cfg["predictors"]
            for pred, f in zip(predictors, self.folds):
                pred.load_model_for_inference(
                    model_training_output_dir=model_dir,
                    fold=f,
                    checkpoint_name="checkpoint_final.pth",
                    torch_compile=self.torch_compile,
                )
                _try_enable_trt(pred, f)
        if self._use_sparse_model_override and self.sparse_predictor is not None:
            self.sparse_predictor.load_model_for_inference(
                model_training_output_dir=self.sparse_model_path,
                fold=self.sparse_model_fold,
                checkpoint_name="checkpoint_final.pth",
                torch_compile=self.torch_compile,
            )
            _try_enable_trt(self.sparse_predictor, self.sparse_model_fold)
        elif not self._use_sparse_model_override and self.sparse_predictor is None and self.predictors:
            # As fallback, assign existing Predictor as sparse reference
            self.sparse_predictor = self.predictors[0]
        load_time = time.time() - start_time
        if self.verbose:
            print(f"Models loaded: {load_time:.2f}s, folds: {self.folds}")
            if self._use_sparse_model_override:
                print(f"Sparse model: {self.sparse_model_path} (fold={self.sparse_model_fold})")
            primary_cfg = self._dense_model_configs.get(self._primary_model_key, {})
            primary_device = primary_cfg.get("device")
            if primary_device is not None:
                print(f"Dense model [primary]: {self.model_path} (device={primary_device})")
            for key in self._extra_model_keys:
                cfg = self._dense_model_configs.get(key, {})
                model_dir = cfg.get("model_path")
                device = cfg.get("device")
                if model_dir:
                    if device is not None:
                        print(f"Additional dense model [{key}]: {model_dir} (device={device})")
                    else:
                        print(f"Additional dense model [{key}]: {model_dir}")

    def set_perform_everything_on_device(self, perform_everything_on_device: bool):
        """Switch aggregation device and toggle nnUNet preprocessing resampling between CPU/GPU.

        Notes:
            - When falling back to CPU (False), also pass `device="cpu"` to resampling to limit GPU memory.
            - When restoring (True), set `device="cuda"` (ignored if GPU is unavailable).
        """

        def _apply(pred: AdaptiveSparsePredictor) -> None:
            # nnUNet sliding-window aggregation device (GPU/CPU)
            pred.perform_everything_on_device = perform_everything_on_device

            # Switch device for nnUNet preprocessing (resampling)
            try:
                cfg = pred.configuration_manager.configuration  # type: ignore[attr-defined]
                # Always set CPU when GPU is unavailable
                if perform_everything_on_device and torch.cuda.is_available():
                    resample_dev = "cuda"
                else:
                    resample_dev = "cpu"
                for k in (
                    "resampling_fn_data_kwargs",
                    "resampling_fn_seg_kwargs",
                    "resampling_fn_probabilities_kwargs",
                ):
                    if k in cfg and isinstance(cfg[k], dict):
                        # This "device" is passed to resample_torch_fornnunet
                        cfg[k]["device"] = resample_dev
            except Exception:
                # Defensive ignore (e.g., not loaded yet)
                pass

        updated: set[int] = set()
        for cfg in self._dense_model_configs.values():
            predictors = cfg.get("predictors", [])
            for pred in predictors:
                if id(pred) in updated:
                    continue
                _apply(pred)
                updated.add(id(pred))
        if self.sparse_predictor is not None and id(self.sparse_predictor) not in updated:
            _apply(self.sparse_predictor)

    def load_nifti_volume(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Read a NIfTI file using SimpleITKIO.

        Args:
            file_path: Path to the NIfTI file

        Returns:
            image: Image array (C, H, W, D) in nnUNet format
            properties: Metadata dict
        """
        # Read via SimpleITKIO
        image, properties = self.io_handler.read_images([file_path], orientation="RAS")

        return image, properties

    def predict_single_volume(
        self,
        image: np.ndarray,
        properties: Dict,
        return_probabilities: bool = True,
        stage_tracker: Optional[VesselSegmentationStageTracker] = None,
    ) -> VesselSegmentationOutput:
        """Inference on a single volume (with intermediate-stage diagnostics)."""

        tracker = stage_tracker or VesselSegmentationStageTracker(None)

        if not self.predictors:
            raise RuntimeError("Predictors for inference are not initialized")

        # Get sparse-search Predictor and dense Predictor
        ref_pred = self.predictors[0]
        sparse_pred = self.sparse_predictor or ref_pred

        # Preprocess using the sparse model's configuration
        sparse_preprocessed = sparse_pred._preprocess_data(image, properties, seg_from_prev_stage=None)
        sparse_data = sparse_preprocessed["data"].to(dtype=torch.half)

        tracker.mark(
            VesselSegmentationStage.PREPROCESS,
            {
                "data_shape": _shape_tuple(sparse_data.shape),
                "device": str(sparse_pred.device),
            },
        )
        if tracker.should_stop():
            del sparse_data
            del sparse_preprocessed
            self._safe_clear_cuda_cache()
            return tracker.build_output()

        # Enable autocast only on CUDA
        if sparse_pred.device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda")
        else:
            from contextlib import nullcontext

            autocast_ctx = nullcontext()

        sparse_context: Optional[Dict[str, Any]] = None
        orientation_result: Optional[OrientationCorrectionResult] = None
        if not self.use_sparse_search:
            # Skip sparse search; use full image as ROI for dense inference
            roi_bbox_sparse = tuple(slice(0, int(s)) for s in sparse_data.shape[1:])
            # Auxiliary info (explicitly set skip flag)
            sparse_stage_info: Dict[str, Any] = {
                "roi_bbox": _slices_to_ranges(roi_bbox_sparse),
                "skipped_sparse": True,
            }
            try:
                network_shape = list([int(x) for x in sparse_data.shape[1:]])
                spacing_cfg = getattr(sparse_pred.configuration_manager, "spacing", None)
                network_spacing = [float(x) for x in spacing_cfg] if spacing_cfg is not None else []
                sparse_stage_info["network_shape"] = network_shape
                sparse_stage_info["network_spacing"] = network_spacing
            except Exception:
                pass
        else:
            with torch.inference_mode(), autocast_ctx:
                if self.enable_orientation_correction:
                    roi_bbox_sparse, sparse_context = sparse_pred.sparse_search(
                        sparse_data, return_context=True
                    )
                else:
                    roi_bbox_sparse = sparse_pred.sparse_search(sparse_data)

            if self.enable_orientation_correction:
                orientation_result = self._maybe_apply_orientation_correction(
                    sparse_pred=sparse_pred,
                    sparse_preprocessed=sparse_preprocessed,
                    sparse_context=sparse_context,
                    roi_bbox=roi_bbox_sparse,
                )
                sparse_data = sparse_preprocessed["data"].to(dtype=torch.half)
                if orientation_result is not None:
                    roi_bbox_sparse = orientation_result.roi_bbox_after

            # Store additional metadata for sparse stage
            # - roi_bbox: slice ranges of ROI in network space
            # - network_shape: spatial shape in network space after preprocessing (excluding C)
            # - network_spacing: voxel spacing (mm) in network space
            sparse_stage_info = {"roi_bbox": _slices_to_ranges(roi_bbox_sparse)}
            try:
                # Record shape/spacing if available (used for later safety checks)
                network_shape = list([int(x) for x in sparse_data.shape[1:]])
                spacing_cfg = getattr(sparse_pred.configuration_manager, "spacing", None)
                if spacing_cfg is not None:
                    network_spacing = [float(x) for x in spacing_cfg]
                else:
                    network_spacing = []
                sparse_stage_info["network_shape"] = network_shape
                sparse_stage_info["network_spacing"] = network_spacing
            except Exception:
                # Continue inference even if unavailable (optional info)
                pass
            if orientation_result is not None:
                sparse_stage_info["orientation"] = orientation_result.to_metadata()

        tracker.mark(
            VesselSegmentationStage.SPARSE_SEARCH,
            sparse_stage_info,
        )
        if tracker.should_stop():
            del sparse_data
            del sparse_preprocessed
            self._safe_clear_cuda_cache()
            return tracker.build_output()

        # Convert sparse ROI to shared coordinates; determine ROI for dense models
        transform_sparse = sparse_pred._build_transform_info(
            preprocessed=sparse_preprocessed,
            data=sparse_data,
            roi_bbox=roi_bbox_sparse,
            refined_local_bbox=None,
        )
        if orientation_result is not None:
            transform_sparse["orientation_correction"] = orientation_result.to_metadata()

        bounds_original = self._roi_bbox_to_original_bounds(transform_sparse, roi_bbox_sparse)

        dense_preprocessed, dense_data, dense_roi_bbox, dense_shape_before = self._prepare_dense_inputs(
            ref_pred=ref_pred,
            image=image,
            properties=properties,
            bounds_original=bounds_original,
            roi_bbox_sparse=roi_bbox_sparse,
            orientation_result=orientation_result,
            sparse_pred=sparse_pred,
            sparse_preprocessed=sparse_preprocessed,
            sparse_data=sparse_data,
        )

        self._safe_clear_cuda_cache()

        # Run primary and additional models in parallel under same scheduler
        all_keys: List[str] = [self._primary_model_key] + list(self._extra_model_keys)
        primary_result, all_results, stopped = self._run_models_concurrently(
            keys=all_keys,
            dense_preprocessed=dense_preprocessed,
            dense_data=dense_data,
            dense_roi_bbox=dense_roi_bbox,
            dense_shape_before=dense_shape_before,
            orientation_result=orientation_result,
            return_probabilities=return_probabilities,
            stage_tracker=tracker,
        )
        if stopped:
            # All launched jobs have joined (synced); stop early
            return tracker.build_output()
        if primary_result is None:
            raise RuntimeError("Failed to obtain dense inference result")

        pred = primary_result.prediction
        roi_data = primary_result.roi
        transform_info = primary_result.transform_info

        dense_outputs: Dict[str, DenseModelResult] = dict(all_results)

        tracker.mark(
            VesselSegmentationStage.AGGREGATE,
            {
                "fold_agg": self.fold_agg,
                "return_probabilities": bool(return_probabilities),
                "additional_models": len(self._extra_model_keys),
            },
        )

        return tracker.build_output(
            pred,
            roi_data,
            transform_info,
            dense_outputs=dense_outputs,
        )

    def predict_from_file_with_info(
        self,
        file_path: str,
        return_probabilities: bool = True,
        stage_limit: Optional[VesselSegmentationStage] = None,
    ) -> VesselSegmentationOutput:
        """Predict for a single file and return with stage info."""

        if self.verbose:
            print(f"Processing: {os.path.basename(file_path)}")

        image, properties = self.load_nifti_volume(file_path)

        tracker = VesselSegmentationStageTracker(stage_limit)
        tracker.mark(
            VesselSegmentationStage.LOAD,
            {
                "file_name": os.path.basename(file_path),
                "image_shape": _shape_tuple(image.shape),
                "dtype": str(image.dtype),
            },
        )
        if tracker.should_stop():
            return tracker.build_output()

        result = self.predict_single_volume(
            image=image,
            properties=properties,
            return_probabilities=return_probabilities,
            stage_tracker=tracker,
        )

        return result

    def predict_single_volume_with_info(
        self,
        image: np.ndarray,
        properties: Dict,
        return_probabilities: bool = True,
        stage_limit: Optional[VesselSegmentationStage] = None,
    ) -> VesselSegmentationOutput:
        """Predict for a single in-memory volume and return with stage info."""
        tracker = VesselSegmentationStageTracker(stage_limit)
        tracker.mark(
            VesselSegmentationStage.LOAD,
            {
                "image_shape": _shape_tuple(image.shape),
                "dtype": str(image.dtype),
            },
        )
        if tracker.should_stop():
            return tracker.build_output()

        result = self.predict_single_volume(
            image=image,
            properties=properties,
            return_probabilities=return_probabilities,
            stage_tracker=tracker,
        )

        return result

    def predict_from_file(
        self,
        file_path: str,
        return_probabilities: bool = True,
        stage_limit: Optional[VesselSegmentationStage] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
        """Compatibility API: provide legacy return values ignoring stage info."""

        if isinstance(stage_limit, str):
            stage_limit = parse_vessel_stage_limit(stage_limit)

        out = self.predict_from_file_with_info(
            file_path=file_path,
            return_probabilities=return_probabilities,
            stage_limit=stage_limit,
        )
        return out.seg, out.roi, out.transform_info

    def predict_batch(
        self,
        file_paths: List[str],
        save_dir: str = None,
        return_probabilities: bool = True,
        resume: bool = False,
    ) -> None:
        """
        Batch inference for multiple files.

        Args:
            file_paths: List of input file paths
            save_dir: Directory to save results (None to disable)
            return_probabilities: Whether to return probabilities
            resume: Whether to skip already-saved cases and resume

        Returns:
            List of predictions
        """
        cases_with_annotations_outside_roi: List[Dict[str, int]] = []
        cases_with_orientation_correction: List[Dict[str, Any]] = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Filter files to process
        files_to_process = []
        skipped_count = 0

        for file_path in file_paths:
            base_name = os.path.basename(file_path).replace(".nii.gz", "").replace("_0000", "")

            # In resume mode, check for already processed cases
            if resume and save_dir:
                case_dir = self._case_dir(save_dir, base_name)
                if self._already_processed(case_dir, return_probabilities):
                    skipped_count += 1
                    if self.verbose:
                        print(f"Skip: {base_name} (already processed)")
                    # Collect existing annotations_outside_roi if present
                    roi_ann_path = case_dir / self.FN_ROI_ANNS
                    if roi_ann_path.exists():
                        try:
                            with open(roi_ann_path, "r", encoding="utf-8") as f:
                                roi_ann_data = json.load(f)
                                total_cnt = roi_ann_data.get("total_annotations", 0)
                                inside_cnt = roi_ann_data.get("roi_annotations_count", 0)
                                if inside_cnt < total_cnt:
                                    cases_with_annotations_outside_roi.append(
                                        {
                                            "case_id": base_name,
                                            "total": total_cnt,
                                            "inside_roi": inside_cnt,
                                            "outside_roi": total_cnt - inside_cnt,
                                        }
                                    )
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Failed to load roi_annotations.json for {base_name}: {e}")
                    orientation_info = self._load_orientation_perm_signs_from_transform(
                        case_dir / self.FN_XFORM
                    )
                    if orientation_info is not None:
                        perm_existing, signs_existing = orientation_info
                        if not _orientation_is_identity(perm_existing, signs_existing):
                            cases_with_orientation_correction.append(
                                {
                                    "case_id": base_name,
                                    "perm": [int(x) for x in perm_existing],
                                    "signs": [int(x) for x in signs_existing],
                                }
                            )
                    continue

            files_to_process.append(file_path)

        if resume and self.verbose:
            print(f"\nResume mode: skipped {skipped_count} cases")
            print(f"To process: {len(files_to_process)} cases\n")

        for file_path in tqdm(files_to_process, desc="Running inference", disable=not self.verbose):
            base_name = os.path.basename(file_path).replace(".nii.gz", "").replace("_0000", "")
            try:
                # Inference
                output = self.predict_from_file_with_info(
                    file_path=file_path, return_probabilities=return_probabilities
                )

                dense_outputs = dict(output.dense_outputs)
                base_result = dense_outputs.get(self._primary_model_key)
                if base_result is None and output.seg is not None and output.roi is not None:
                    base_result = DenseModelResult(
                        prediction=output.seg,
                        roi=output.roi,
                        transform_info=output.transform_info,
                        return_probabilities=return_probabilities,
                    )
                    dense_outputs[self._primary_model_key] = base_result

                orientation_pair = _resolve_orientation_from_output(output, base_result)

                if orientation_pair is not None and not _orientation_is_identity(*orientation_pair):
                    orientation_perm_case, orientation_signs_case = orientation_pair
                    cases_with_orientation_correction.append(
                        {
                            "case_id": base_name,
                            "perm": [int(x) for x in orientation_perm_case],
                            "signs": [int(x) for x in orientation_signs_case],
                        }
                    )

                # Save results
                if save_dir:
                    case_dir = self._case_dir(save_dir, base_name)
                    annotations = self._load_annotations_for_case(base_name)

                    if base_result is not None:
                        self._save_model_output(
                            target_dir=case_dir,
                            base_name=base_name,
                            result=base_result,
                            annotations=annotations,
                            collect_annotation_stats=True,
                            cases_with_annotations_outside_roi=cases_with_annotations_outside_roi,
                        )

                    for key in self._extra_model_keys:
                        cfg = self._dense_model_configs.get(key)
                        if not cfg:
                            continue
                        result_extra = dense_outputs.get(key)
                        if result_extra is None:
                            continue
                        suffix = cfg.get("save_suffix") or key
                        self._save_extra_segmentation(
                            target_dir=case_dir,
                            suffix=suffix,
                            result=result_extra,
                        )
            except Exception as e:
                if self.verbose:
                    print(f"Error: Failed to infer/save {base_name}: {e}")

        # Save list of cases with annotations outside ROI (includes all cases in resume mode)
        if save_dir:
            list_path = os.path.join(save_dir, "annotations_outside_roi.json")

            # Merge with existing file if present
            existing_cases = {}
            if resume and os.path.exists(list_path):
                try:
                    with open(list_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        for case in existing_data.get("cases", []):
                            existing_cases[case["case_id"]] = case
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to load existing annotations_outside_roi.json: {e}")

            # Merge old and new case info
            for case in cases_with_annotations_outside_roi:
                existing_cases[case["case_id"]] = case

            # Sort by case_id and convert to list
            final_cases = sorted(existing_cases.values(), key=lambda x: x["case_id"])

            # Save to file
            if final_cases:
                with open(list_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "count": len(final_cases),
                            "cases": final_cases,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                if self.verbose and resume:
                    print(f"\nUpdated annotations_outside_roi: {len(final_cases)} cases")

        if save_dir:
            orientation_path = os.path.join(save_dir, self.FN_ORIENTATION_CASES)
            existing_orientation_cases: Dict[str, Dict[str, Any]] = {}
            if resume and os.path.exists(orientation_path):
                try:
                    with open(orientation_path, "r", encoding="utf-8") as f:
                        orientation_data = json.load(f)
                        for case in orientation_data.get("cases", []):
                            case_id = case.get("case_id")
                            if isinstance(case_id, str):
                                existing_orientation_cases[case_id] = case
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to load existing {self.FN_ORIENTATION_CASES}: {e}")

            for case in cases_with_orientation_correction:
                existing_orientation_cases[case["case_id"]] = case

            final_orientation_cases = sorted(existing_orientation_cases.values(), key=lambda x: x["case_id"])

            if final_orientation_cases:
                with open(orientation_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "count": len(final_orientation_cases),
                            "cases": final_orientation_cases,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                if self.verbose and resume:
                    print(f"\nUpdated orientation correction info: {len(final_orientation_cases)} cases")


def main():
    # Build command-line argument parser
    parser = argparse.ArgumentParser(description="Brain vessel segmentation inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/nnUNetTrainerSkeletonRecall_more_DAv3__nnUNetResEncUNetMPlans__3d_fullres",
        help="Path to the nnUNet model",
    )
    parser.add_argument(
        "--additional-dense-model-paths",
        type=str,
        nargs="*",
        default=[
            "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres",
            "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres",
        ],
        help="Paths for additional dense models (space-separated)",
    )
    parser.add_argument(
        "--sparse-model-path",
        type=str,
        default="/workspace/logs/nnUNet_results/Dataset003_VesselGrouping/RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres",
        help="Path for the nnUNet model used only for sparse search (defaults to --model-path)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/workspace/data/nnUNet_inference/imagesTs",
        help="Test data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/data/nnUNet_inference/predictions_v4_margin15_30",
        help="Output directory",
    )
    parser.add_argument(
        "--folds",
        type=str,
        nargs="+",
        default=["all"],
        help="Fold indices or identifiers to use (e.g., 0 1 all)",
    )
    parser.add_argument(
        "--series-niix-dir",
        type=str,
        default="/workspace/data/series_niix",
        help="Root directory for annotations (where *.nii.annotations.json are searched)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume and skip already saved cases",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse search",
    )
    parser.add_argument(
        "--use-mirroring",
        action="store_true",
        help="Enable mirroring (TTA) (default: disabled)",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["npz", "npy"],
        default="npz",
        help="Save format for inference results (npz: compressed, npy: uncompressed)",
    )
    # Root directory to auto-discover TensorRT engines
    parser.add_argument(
        "--trt-dir",
        type=str,
        default="/workspace/logs/trt",
        help="Root dir for TensorRT engines (trainer__plans__config/fold_* structure)",
    )
    parser.add_argument(
        "--trt-fp16",
        action="store_true",
        help="Prefer FP16 engine (model_fp16.engine)",
    )
    parser.add_argument(
        "--no-trt-fp16",
        dest="trt_fp16",
        action="store_false",
        help="Use FP32 engine (model_fp32.engine)",
    )
    parser.set_defaults(trt_fp16=True)
    parser.add_argument(
        "--fold-agg",
        type=str,
        choices=["logit_mean", "prob_mean", "prob_max", "noisy_or"],
        default="logit_mean",
        help="Fold aggregation method (logit_mean: legacy, prob_mean/prob_max/noisy_or)",
    )
    parser.add_argument(
        "--fg-threshold",
        type=float,
        default=0.30,
        help="Single threshold for foreground-first gate (0-1). Prefer foreground over background when exceeded",
    )
    # Provide both enable (default) and disable flags
    parser.add_argument(
        "--fg-first",
        dest="fg_first",
        action="store_true",
        help="Enable foreground-first gate (during labeling)",
    )
    parser.add_argument(
        "--no-fg-first",
        dest="fg_first",
        action="store_false",
        help="Disable foreground-first gate",
    )
    parser.set_defaults(fg_first=True)
    parser.add_argument(
        "--return-probabilities",
        action="store_true",
        help="Output probabilities (labels otherwise)",
    )
    # Restrict BBox refine to z only (default disabled)
    parser.add_argument(
        "--refine-z-only",
        dest="refine_z_only",
        action="store_true",
        help="Restrict BBox refinement after dense inference to z (SI) only",
    )
    parser.add_argument(
        "--no-refine-z-only",
        dest="refine_z_only",
        action="store_false",
        help="Disable z-only refinement (refine XY as well)",
    )
    parser.set_defaults(refine_z_only=False)

    args = parser.parse_args()

    # Settings
    MODEL_PATH = args.model_path
    SPARSE_MODEL_PATH = args.sparse_model_path
    TEST_DIR = args.test_dir
    OUTPUT_DIR = args.output_dir
    FOLDS = tuple(args.folds)
    SERIES_NIIX_DIR = args.series_niix_dir

    # Show GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Initialize Predictor (load models once)
    print("=" * 50)
    print("Brain vessel segmentation inference system")
    print("=" * 50)

    if args.resume:
        print("[Resume mode] Skipping already-saved cases")

    predictor = VesselSegmentationPredictor(
        model_path=MODEL_PATH,
        additional_dense_model_paths=args.additional_dense_model_paths,
        sparse_model_path=SPARSE_MODEL_PATH,
        folds=FOLDS,  # ensemble of specified folds
        device="cuda",
        use_sparse_search=not args.no_sparse,  # enable/disable sparse search
        use_mirroring=args.use_mirroring,  # enable/disable TTA (default: False)
        torch_compile=True,  # enable locally
        verbose=True,
        series_niix_dir=SERIES_NIIX_DIR,
        save_format=args.save_format,
        fold_agg=args.fold_agg,
        foreground_first=args.fg_first,
        fg_threshold=args.fg_threshold,
        refine_z_only=args.refine_z_only,
        trt_dir=args.trt_dir,
        trt_fp16=args.trt_fp16,
        enable_orientation_correction=True if SPARSE_MODEL_PATH is not None else False,
    )

    # Collect test files
    test_files = sorted(Path(TEST_DIR).glob("*.nii.gz"))
    print(f"\nNumber of test files: {len(test_files)}")

    # Run batch inference
    print("\nStarting inference...")
    start_time = time.time()

    predictor.predict_batch(
        file_paths=[str(f) for f in test_files],
        save_dir=OUTPUT_DIR,
        return_probabilities=args.return_probabilities,  # labels by default; probabilities if specified
        resume=args.resume,  # resume mode from CLI
    )

    total_time = time.time() - start_time

    # Compute average time over actually processed cases (files_to_process in resume mode)
    processed_count = len([p for p in test_files])
    avg_time = total_time / processed_count if processed_count > 0 else 0

    print(f"\nInference complete!")
    print(f"Total time: {total_time:.2f}s")
    if avg_time > 0:
        print(f"Average time: {avg_time:.2f}s/volume")


if __name__ == "__main__":
    main()
