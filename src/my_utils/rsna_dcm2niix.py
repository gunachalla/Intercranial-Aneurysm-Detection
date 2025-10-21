#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM to NIfTI conversion script (with QC and summary CSVs).

Approach:
 1) Convert to NIfTI (+JSON sidecar) using dcm2niix
 2) Pre-conversion audit: when multiple sizes are mixed in a folder (scout/localizer),
    extract only the majority (Rows, Cols) into a temp directory before conversion
 3) On failure/decoding errors, retry after running gdcmconv --raw ("decompress")
 4) Use -i n to allow DERIVED/MPR and set flags heuristically post-conversion
 5) Scan outputs, collect meta/QC for each NIfTI into meta_summary.csv; only flagged
    items are extracted into suspects.csv

Requirements:
  - dcm2niix (must be on PATH)
  - gdcmconv (must be on PATH; fallback)

Python packages:
  - pydicom, nibabel, numpy, pandas, tqdm

Example:
  python dicom_to_nifti_converter.py \
    --dicom-root /path/to/dicom_root \
    --out-root   /path/to/out_root \
    --tmp-root   /path/to/tmp (use fast disk with enough space) \
    --num-workers 20 \
    --use-majority-size \
    --min-slices 80 --max-dz 1.5

Tips:
  - For large conversions, set --num-workers to 16-20; prefer tmpfs or NVMe for --tmp-root.
  - On Kaggle, install dcm2niix and gdcmconv via: !apt-get install -y dcm2niix libgdcm-tools
  - When passing a folder, dcm2niix will automatically split by series.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import os
import gc
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Literal
import rootutils

import numpy as np
import pandas as pd
from tqdm import tqdm

# Lightweight header reads only; do not load pixel data
import pydicom
from pydicom.dataset import Dataset as DcmDataset
from pydicom.sequence import Sequence as DcmSequence

import nibabel as nib
import ast
import SimpleITK as sitk

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import for RAS conversion
from src.my_utils.rsna_utils import load_nifti_and_convert_to_ras  # noqa: E402
from src.my_utils.dicom_metadata import extract_metadata_from_dataset


# =============================
# Configuration
# =============================
@dataclass
class QCConfig:
    min_slices: int = 80  # threshold for long volume (Nz >= min_slices)
    max_dz_mm: float = 1.5  # upper bound for near-isotropic/thin slices (dz <= max_dz_mm)
    xy_aniso_tol: float = 0.12  # XY anisotropy tolerance |dx-dy|/mean(dx,dy) <= tol
    slab_slices: int = 32  # threshold for a thin slab (Nz < slab_slices)
    z_span_min_mm: float = 30.0  # below this, flag as slab
    intensity_hi_clip: float = 99.5
    intensity_lo_clip: float = 0.5


@dataclass
class RunConfig:
    dicom_root: Path
    out_root: Path
    tmp_root: Path
    num_workers: int = 16
    use_majority_size: bool = True
    copy_mode: str = "auto"  # "symlink"|"copy"|"auto"
    keep_intermediate: bool = False
    file_exts: Tuple[str, ...] = (".dcm", "")  # Some DICOM files have no extension
    dcm2niix_flags: Tuple[str, ...] = ("-z", "y", "-b", "y", "-i", "n", "-f", "%s")
    gdcm_first: bool = False  # Whether to run gdcmconv first
    annotations_csv: Optional[Path] = None  # Path to annotations CSV
    use_slice_spacing_filter: bool = True  # Apply slice-spacing filtering
    slice_spacing_tolerance: float = 2.0  # Slice-spacing tolerance (multiplier of median)
    multiframe_only: bool = False  # Process only multiframe DICOMs with annotations
    convert_to_ras: bool = True  # Convert to RAS orientation after conversion (False keeps original)


# Metadata JSON settings
SERIES_METADATA_FILENAME = "series_metadata.json"
METADATA_VERSION = 1


# =============================
# Utilities
# =============================


def which_or_die(cmd: str) -> str:
    path = shutil.which(cmd)
    if not path:
        logging.error(f"Required command not found: {cmd}")
        raise FileNotFoundError(f"'{cmd}' not found in PATH")
    return path


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean from environment variables ("1/true/yes/on" -> True)."""
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def is_dicom_file(path: Path) -> bool:
    if not path.is_file():
        return False
    # Quick extension check
    if path.suffix.lower() not in {".dcm", ""}:
        return False
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        _ = ds.get("SOPInstanceUID", None)
        return True
    except Exception:
        return False


def scan_dicom_files(dir_path: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            # Skip non-target extensions (allow files without extension)
            continue
        if is_dicom_file(p):
            files.append(p)
    return files


def series_key(ds: pydicom.Dataset) -> Tuple[str, str]:
    """Loose identifier key (PatientID, SeriesInstanceUID) for logging.
    Useful when multiple series are mixed within one folder.
    """
    return (
        str(ds.get("PatientID", "")),
        str(ds.get("SeriesInstanceUID", "")),
    )


@dataclass
class FolderAudit:
    total_files: int
    dicom_files: int
    majority_key: Optional[Tuple[int, int, Optional[float], Optional[float]]]  # (rows, cols, ps_row, ps_col)
    series_map: Dict[Tuple[str, str], int]  # key -> count

    @property
    def majority_rows_cols(self) -> Optional[Tuple[int, int]]:
        """Compatibility helper that returns only (rows, cols)."""
        if self.majority_key:
            return (self.majority_key[0], self.majority_key[1])
        return None


def audit_folder(dir_path: Path, file_exts: Tuple[str, ...], pixel_spacing_precision: int = 2) -> FolderAudit:
    """Audit a folder to find the majority image size and PixelSpacing.

    Args:
        dir_path: Directory to audit
        file_exts: Target file extensions
        pixel_spacing_precision: Rounding precision (fractional digits) for PixelSpacing
    """
    counts = Counter()
    series_counts = Counter()
    dicom_files = 0
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if file_exts and p.suffix.lower() not in file_exts:
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            dicom_files += 1
            rows = int(ds.get("Rows", -1))
            cols = int(ds.get("Columns", -1))

            # Obtain PixelSpacing and round
            pixel_spacing = ds.get("PixelSpacing", None)
            if pixel_spacing and len(pixel_spacing) >= 2:
                # Round to the specified precision (default: 2 digits)
                ps_row = round(float(pixel_spacing[0]), pixel_spacing_precision)
                ps_col = round(float(pixel_spacing[1]), pixel_spacing_precision)
                key = (rows, cols, ps_row, ps_col)
            else:
                # Use None if PixelSpacing is absent
                key = (rows, cols, None, None)

            counts[key] += 1
            series_counts[series_key(ds)] += 1
        except Exception:
            continue
    majority = counts.most_common(1)[0][0] if counts else None
    return FolderAudit(
        total_files=sum(1 for _ in dir_path.iterdir()),
        dicom_files=dicom_files,
        majority_key=majority,
        series_map=dict(series_counts),
    )


def audit_folder_with_orientation(
    dir_path: Path, file_exts: Tuple[str, ...], pixel_spacing_precision: int = 2
) -> FolderAudit:
    """Audit a folder and find the majority (Rows, Cols, PixelSpacing, Orientation).

    Args:
        dir_path: Directory to audit
        file_exts: Target file extensions
        pixel_spacing_precision: Rounding precision for PixelSpacing (fractional digits)
    """
    counts = Counter()
    series_counts = Counter()
    dicom_files = 0
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if file_exts and p.suffix.lower() not in file_exts:
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            dicom_files += 1
            rows = int(ds.get("Rows", -1))
            cols = int(ds.get("Columns", -1))

            # Obtain PixelSpacing and round
            pixel_spacing = ds.get("PixelSpacing", None)
            if pixel_spacing and len(pixel_spacing) >= 2:
                # Round to the specified precision (default: 2 digits)
                ps_row = round(float(pixel_spacing[0]), pixel_spacing_precision)
                ps_col = round(float(pixel_spacing[1]), pixel_spacing_precision)
            else:
                # Use None when PixelSpacing is missing
                ps_row, ps_col = None, None

            # Obtain ImageOrientationPatient and round (consider direction vectors)
            orientation = ds.get("ImageOrientationPatient", None)
            if orientation and len(orientation) == 6:
                # Round direction vector to 3 decimals and make a tuple
                orient_tuple = tuple(round(float(x), 3) for x in orientation)
            else:
                orient_tuple = None

            # Include size, PixelSpacing, and direction vector in the key
            key = (rows, cols, ps_row, ps_col, orient_tuple)

            counts[key] += 1
            series_counts[series_key(ds)] += 1
        except Exception:
            continue

    # Get the majority for the extended key including direction vector
    majority_extended = counts.most_common(1)[0][0] if counts else None

    return FolderAudit(
        total_files=sum(1 for _ in dir_path.iterdir()),
        dicom_files=dicom_files,
        majority_key=majority_extended,
        series_map=dict(series_counts),
    )


def filter_by_slice_spacing(
    files_with_metadata: List[Tuple[Path, pydicom.Dataset, Optional[List], str]],
    src_dir: Path,
    tolerance: float = 2.0,
) -> Tuple[List[Tuple[Path, pydicom.Dataset, Optional[List], str]], Optional[str]]:
    """Filter out slices with abnormal inter-slice spacing.

    Args:
        files_with_metadata: List of (Path, Dataset, ImagePositionPatient, SOPInstanceUID)
        tolerance: Allowed multiplier from the median (e.g., 2.0 means 0.5x-2x of median)

    Returns:
        Tuple of (filtered file list, log message or None)
    """
    # If possible, evaluate continuity along the slice-normal vector (fallback: Z)
    normals = []
    for _, ds, _, _ in files_with_metadata:
        orientation = ds.get("ImageOrientationPatient", None)
        if orientation and len(orientation) >= 6:
            try:
                r = np.array(
                    [
                        float(orientation[0]),
                        float(orientation[1]),
                        float(orientation[2]),
                    ],
                    dtype=float,
                )
                c = np.array(
                    [
                        float(orientation[3]),
                        float(orientation[4]),
                        float(orientation[5]),
                    ],
                    dtype=float,
                )
                n = np.cross(r, c)
                norm = np.linalg.norm(n)
                if norm > 1e-6:
                    normals.append(n / norm)
            except Exception:
                continue
    if normals:
        # Determine the dominant axis per normal vector
        dominant_axes = []
        for n in normals:
            abs_n = np.abs(n)
            dominant_axis = int(np.argmax(abs_n))
            dominant_axes.append(dominant_axis)

        # Majority vote for the axis
        axis_counts = {0: 0, 1: 0, 2: 0}
        for axis in dominant_axes:
            axis_counts[axis] += 1

        # Pick the most frequent axis
        axis_idx = max(axis_counts, key=axis_counts.get)

    # Consider only files with ImagePositionPatient
    files_with_position = []
    for p, ds, position, sop_uid in files_with_metadata:
        if position and len(position) >= 3:
            try:
                s_val = float(position[axis_idx])
                files_with_position.append((p, ds, position, sop_uid, s_val))
            except (ValueError, TypeError):
                continue

    if len(files_with_position) <= 2:
        # Not enough to compute spacing; return original list
        return files_with_metadata, None

    # Sort along the dominant axis (fallback: Z)
    files_with_position.sort(key=lambda x: x[4])

    # Compute inter-slice spacings
    spacings = []
    for i in range(1, len(files_with_position)):
        spacing = abs(files_with_position[i][4] - files_with_position[i - 1][4])
        if spacing > 0.001:  # exclude near-identical slice positions
            spacings.append(spacing)

    if not spacings:
        return files_with_metadata, None

    # Compute median spacing
    median_spacing = np.median(spacings)

    # Set acceptable range
    min_spacing = median_spacing / tolerance
    max_spacing = median_spacing * tolerance

    # Filtering with continuity preserved
    filtered_files = []
    prev_z = None
    outlier_count = 0
    total_outliers = 0

    for i, (p, ds, position, sop_uid, z_pos) in enumerate(files_with_position):
        if prev_z is None:
            # Always include the first slice
            filtered_files.append((p, ds, position, sop_uid))
            prev_z = z_pos
        else:
            spacing = abs(z_pos - prev_z)

            # Check if spacing is within tolerance
            if min_spacing <= spacing <= max_spacing:
                filtered_files.append((p, ds, position, sop_uid))
                prev_z = z_pos
                outlier_count = 0  # reset counter when a normal slice is found
            else:
                # Abnormal spacing
                total_outliers += 1
                outlier_count += 1

                # If 3 or more consecutive outliers, treat as the start of a new sequence
                if outlier_count >= 3:
                    # Keep it (possible new sequence) instead of skipping
                    filtered_files.append((p, ds, position, sop_uid))
                    prev_z = z_pos
                    outlier_count = 0

    # Build a log message
    log_message = None
    if total_outliers > 0:
        log_message = (
            f"[Slice Spacing Filter] {src_dir} : Removed {total_outliers} slices with abnormal spacing. "
            f"Median spacing: {median_spacing:.2f}mm, Range: [{min_spacing:.2f}, {max_spacing:.2f}]mm"
        )
        print(log_message)

    return filtered_files, log_message


def prepare_majority_subset_with_orientation(
    src_dir: Path,
    dst_dir: Path,
    majority_key: Tuple[int, int, Optional[float], Optional[float], Optional[Tuple[float, ...]]],
    copy_mode: str = "auto",
    pixel_spacing_precision: int = 2,
    slice_spacing_filter: bool = True,
    slice_spacing_tolerance: float = 2.0,
) -> Tuple[int, Optional[str]]:
    """Copy DICOM files matching majority (size, PixelSpacing, orientation).
    Optionally filter out slices with abnormal inter-slice spacing.

    Args:
        src_dir: Source directory
        dst_dir: Output directory
        majority_key: Tuple (rows, cols, ps_row, ps_col, orient_tuple)
        copy_mode: Copy mode
        pixel_spacing_precision: Rounding precision for PixelSpacing comparison
        slice_spacing_filter: Whether to filter by inter-slice spacing
        slice_spacing_tolerance: Allowed multiplier from median spacing
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # First, collect files matching majority size and orientation
    matching_files = []
    rows_mj, cols_mj, ps_row_mj, ps_col_mj, orient_mj = majority_key

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            rows = int(ds.get("Rows", -1))
            cols = int(ds.get("Columns", -1))

            # Obtain PixelSpacing for comparison
            pixel_spacing = ds.get("PixelSpacing", None)
            if pixel_spacing and len(pixel_spacing) >= 2:
                ps_row = round(float(pixel_spacing[0]), pixel_spacing_precision)
                ps_col = round(float(pixel_spacing[1]), pixel_spacing_precision)
            else:
                ps_row, ps_col = None, None

            # Obtain ImageOrientationPatient for comparison
            orientation = ds.get("ImageOrientationPatient", None)
            if orientation and len(orientation) == 6:
                orient_tuple = tuple(round(float(x), 3) for x in orientation)
            else:
                orient_tuple = None

            # Keep only when size, PixelSpacing, and orientation match
            if (rows, cols, ps_row, ps_col, orient_tuple) == (
                rows_mj,
                cols_mj,
                ps_row_mj,
                ps_col_mj,
                orient_mj,
            ):
                # Also obtain ImagePositionPatient and SOPInstanceUID
                position = ds.get("ImagePositionPatient", None)
                sop_uid = str(ds.get("SOPInstanceUID", ""))
                matching_files.append((p, ds, position, sop_uid))
        except Exception:
            continue

    # Inter-slice spacing filtering
    slice_filter_log = None
    if slice_spacing_filter and len(matching_files) > 2:
        filtered_files, slice_filter_log = filter_by_slice_spacing(
            matching_files, src_dir, slice_spacing_tolerance
        )
    else:
        filtered_files = matching_files

    # Copy/symlink filtered files
    copied = 0
    prefer_symlink = (copy_mode == "symlink") or (copy_mode == "auto" and os.name != "nt")

    for p, _, _, _ in filtered_files:
        target = dst_dir / p.name
        if prefer_symlink:
            try:
                target.symlink_to(p)
            except Exception:
                shutil.copy2(p, target)
        else:
            shutil.copy2(p, target)
        copied += 1

    return copied, slice_filter_log


def prepare_majority_subset(
    src_dir: Path,
    dst_dir: Path,
    majority_key: Tuple[int, int, Optional[float], Optional[float]],
    copy_mode: str = "auto",
    pixel_spacing_precision: int = 2,
    slice_spacing_filter: bool = True,
    slice_spacing_tolerance: float = 2.0,
) -> Tuple[int, Optional[str]]:
    """Copy/symlink only DICOM files matching majority size and PixelSpacing.
    Optionally filter slices with abnormal inter-slice spacing.

    Args:
        src_dir: Source directory
        dst_dir: Output directory
        majority_key: Tuple (rows, cols, ps_row, ps_col)
        copy_mode: Copy mode
        pixel_spacing_precision: Rounding precision for PixelSpacing comparison
        slice_spacing_filter: Whether to filter by inter-slice spacing
        slice_spacing_tolerance: Allowed multiplier from median spacing
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # First, collect files matching the majority size
    matching_files = []
    rows_mj, cols_mj, ps_row_mj, ps_col_mj = majority_key

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            rows = int(ds.get("Rows", -1))
            cols = int(ds.get("Columns", -1))

            # Obtain PixelSpacing for comparison
            pixel_spacing = ds.get("PixelSpacing", None)
            if pixel_spacing and len(pixel_spacing) >= 2:
                ps_row = round(float(pixel_spacing[0]), pixel_spacing_precision)
                ps_col = round(float(pixel_spacing[1]), pixel_spacing_precision)
            else:
                ps_row, ps_col = None, None

            # Keep only when size and PixelSpacing match
            if (rows, cols, ps_row, ps_col) == (rows_mj, cols_mj, ps_row_mj, ps_col_mj):
                # Also obtain ImagePositionPatient and SOPInstanceUID
                position = ds.get("ImagePositionPatient", None)
                sop_uid = str(ds.get("SOPInstanceUID", ""))
                matching_files.append((p, ds, position, sop_uid))
        except Exception:
            continue

    # Inter-slice spacing filtering
    slice_filter_log = None
    if slice_spacing_filter and len(matching_files) > 2:
        filtered_files, slice_filter_log = filter_by_slice_spacing(
            matching_files, src_dir, slice_spacing_tolerance
        )
    else:
        filtered_files = matching_files

    # Copy/symlink filtered files
    copied = 0
    prefer_symlink = (copy_mode == "symlink") or (copy_mode == "auto" and os.name != "nt")

    for p, _, _, _ in filtered_files:
        target = dst_dir / p.name
        if prefer_symlink:
            try:
                target.symlink_to(p)
            except Exception:
                shutil.copy2(p, target)
        else:
            shutil.copy2(p, target)
        copied += 1

    return copied, slice_filter_log


def _detect_multiframe_mr_missing_meta(src_dir: Path) -> Dict[str, object]:
    """Heuristically detect multiframe MR with missing PixelSpacing/IOP.

    Conditions:
        * NumberOfFrames > 1 or single DICOM (multiframe assumed)
        * Modality == 'MR'
        * PixelSpacing or ImageOrientationPatient missing in both SharedFG and top-level

    Returns:
        dict: {
            'is_multiframe_mr': bool,
            'missing_pixel_spacing': bool,
            'missing_orientation': bool,
            'approx_slice_count': Optional[int],
        }
    """
    is_multiframe = False
    is_mr = False
    missing_ps = False
    missing_iop = False
    approx_slice_count: Optional[int] = None

    try:
        # Count DICOM files in directory
        dicom_files: List[Path] = []
        for p in src_dir.iterdir():
            if not p.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                dicom_files.append(p)
            except Exception:
                continue

        only_one_dicom = len(dicom_files) == 1
        # Inspect header on a representative DICOM (first one)
        for p in dicom_files[:1]:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            except Exception:
                break
            # Multiframe/modality
            has_nf = "NumberOfFrames" in ds
            n_frames: Optional[int] = None
            if has_nf:
                try:
                    n_frames = int(ds.get("NumberOfFrames"))
                    if n_frames and n_frames > 1:
                        approx_slice_count = n_frames
                except Exception:
                    n_frames = None
            modality = str(ds.get("Modality", "")).upper()
            is_mr = modality == "MR"

            # Check for PixelSpacing (SharedFG -> top-level)
            ps_top = ds.get("PixelSpacing", None)
            ps_shared = None
            try:
                sfgs = getattr(ds, "SharedFunctionalGroupsSequence", None)
                if sfgs and len(sfgs) > 0:
                    pm = getattr(sfgs[0], "PixelMeasuresSequence", None)
                    if pm and len(pm) > 0:
                        ps_shared = getattr(pm[0], "PixelSpacing", None)
            except Exception:
                ps_shared = None
            missing_ps = (ps_top is None) and (ps_shared is None)

            # Presence of ImageOrientationPatient (SharedFG -> top-level)
            iop_top = ds.get("ImageOrientationPatient", None)
            iop_shared = None
            try:
                sfgs = getattr(ds, "SharedFunctionalGroupsSequence", None)
                if sfgs and len(sfgs) > 0:
                    po = getattr(sfgs[0], "PlaneOrientationSequence", None)
                    if po and len(po) > 0:
                        iop_shared = getattr(po[0], "ImageOrientationPatient", None)
            except Exception:
                iop_shared = None
            missing_iop = (iop_top is None) and (iop_shared is None)

            # Estimate slice count from length of PerFrameFunctionalGroupsSequence
            try:
                per_frame_seq = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
                if per_frame_seq is not None:
                    pf_len = len(per_frame_seq)
                    if pf_len and (approx_slice_count is None or pf_len > approx_slice_count):
                        approx_slice_count = pf_len
            except Exception:
                pass

            # Criteria: multiple NumberOfFrames or single DICOM file
            is_multiframe = bool(((has_nf and (n_frames or 0) > 1) or only_one_dicom))

            # One successful read is enough
            break
    except Exception:
        # If detection is not possible, conservatively return False
        pass

    return {
        "is_multiframe_mr": bool(is_multiframe and is_mr),
        "missing_pixel_spacing": bool(missing_ps),
        "missing_orientation": bool(missing_iop),
        "approx_slice_count": approx_slice_count,
    }


def _estimate_slice_count_from_nifti(nifti_path: Path) -> Optional[int]:
    """Estimate number of slices along Z from a NIfTI file."""
    try:
        img = nib.load(str(nifti_path), mmap=False)
    except Exception:
        return None

    shape = img.shape
    if not shape:
        return None

    # Assume 3D/4D volume; use the 3rd axis as slice count
    if len(shape) >= 3:
        return int(shape[2])

    return None


def _infer_multiframe_default_spacing(slice_count: Optional[int]) -> Tuple[Tuple[float, float, float], str]:
    """Decide default spacing for multiframe interpolation based on slice count."""
    sx = 0.5
    sy = 0.5

    if slice_count is None:
        return (sx, sy, 0.55), "unknown_slice_count"
    if slice_count <= 80:
        return (sx, sy, 5.0), "slice_count_le_80"
    return (sx, sy, 0.55), "slice_count_gt_80"


def _apply_multiframe_defaults_to_nifti(nifti_path: Path, *, spacing_xyz: Tuple[float, float, float]) -> None:
    """Apply default spacing/orientation (RAS) to a NIfTI without rotating data.

    - Do not use nib.as_closest_canonical; directly set an identity-rotation affine with (sx,sy,sz)
    - Set qform/sform codes to clarify world coordinates
    """
    img = nib.load(str(nifti_path), mmap=False)
    # Keep data array as-is (choose safe default axes when IOP is unknown)
    data = np.asanyarray(img.dataobj)
    dtype = img.get_data_dtype()
    if data.dtype != dtype:
        try:
            data = data.astype(dtype, copy=False)
        except Exception:
            data = data.astype(np.float32, copy=False)

    sx, sy, sz = [float(x) for x in spacing_xyz]
    affine = np.eye(4, dtype=float)
    affine[0, 0] = sx
    affine[1, 1] = sy
    affine[2, 2] = sz  # TODO consider -sz or flipping data along z
    affine[0:3, 3] = 0.0

    hdr = img.header.copy()
    try:
        hdr.set_xyzt_units(xyz=2)
    except Exception:
        pass
    try:
        hdr.set_zooms((sx, sy, sz))
    except Exception:
        pass

    new_img = nib.Nifti1Image(data, affine, header=hdr)
    nib.save(new_img, str(nifti_path))

    del img, new_img, data
    gc.collect()


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    import subprocess

    proc = subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


# =============================
# dcm2niix execution
# =============================


def run_dcm2niix(src_dir: Path, out_dir: Path, flags: Iterable[str]) -> Tuple[int, str, str]:
    dcm2niix_path = which_or_die("dcm2niix")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [dcm2niix_path, *flags, "-o", str(out_dir), str(src_dir)]
    return run_cmd(cmd)


def _apply_multiframe_defaults_to_dicom_dir(
    src_dir: Path,
    *,
    spacing_xyz: Tuple[float, float, float],
    default_iop: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
) -> Tuple[int, int, List[str]]:
    """Preprocess to add missing multiframe MR meta (PixelSpacing/IOP) on the DICOM side.

    Notes:
      - Apply only to a temporary copy before conversion (do not modify originals).
      - Use defer_size to keep PixelData referenced from the original when saving with pydicom.

    Returns:
      (num_processed_files, num_patched_files, log_messages)
    """
    changed = 0
    total = 0
    logs: List[str] = []

    sx, sy, sz = [float(x) for x in spacing_xyz]

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        try:
            # Use defer_size to keep PixelData referenced
            ds = pydicom.dcmread(str(p), force=True, defer_size="1 KB")
        except Exception:
            continue

        total += 1

        try:
            modality = str(ds.get("Modality", "")).upper()
            # Multiframe check (NumberOfFrames>1 or presence of PerFrameFunctionalGroupsSequence)
            n_frames = None
            if "NumberOfFrames" in ds:
                try:
                    n_frames = int(ds.get("NumberOfFrames"))
                except Exception:
                    n_frames = None
            has_per_frame = hasattr(ds, "PerFrameFunctionalGroupsSequence")
            is_multiframe = (n_frames is not None and n_frames > 1) or has_per_frame

            if modality != "MR" or not is_multiframe:
                continue

            # Check both top-level and SharedFG
            ps_top = getattr(ds, "PixelSpacing", None)
            iop_top = getattr(ds, "ImageOrientationPatient", None)

            ps_shared = None
            iop_shared = None
            sfgs = getattr(ds, "SharedFunctionalGroupsSequence", None)
            if sfgs and len(sfgs) > 0:
                try:
                    pm = getattr(sfgs[0], "PixelMeasuresSequence", None)
                    if pm and len(pm) > 0:
                        ps_shared = getattr(pm[0], "PixelSpacing", None)
                except Exception:
                    ps_shared = None
                try:
                    po = getattr(sfgs[0], "PlaneOrientationSequence", None)
                    if po and len(po) > 0:
                        iop_shared = getattr(po[0], "ImageOrientationPatient", None)
                except Exception:
                    iop_shared = None

            need_ps = (ps_top is None) and (ps_shared is None)
            need_iop = (iop_top is None) and (iop_shared is None)

            if not (need_ps or need_iop):
                continue

            # Set top-level fields (dcm2niix prefers top-level)
            if need_ps:
                try:
                    ds.PixelSpacing = [sx, sy]
                    # Provide SliceThickness/SpacingBetweenSlices as reference values
                    ds.SliceThickness = sz
                    ds.SpacingBetweenSlices = sz
                except Exception:
                    pass
            if need_iop:
                try:
                    ds.ImageOrientationPatient = list(default_iop)
                except Exception:
                    pass

            # Also fill the SharedFunctionalGroupsSequence minimally
            try:
                if sfgs is None or len(sfgs) == 0:
                    item = DcmDataset()
                    sfgs = DcmSequence([item])
                    ds.SharedFunctionalGroupsSequence = sfgs
                # PixelMeasuresSequence
                if need_ps:
                    pm = getattr(sfgs[0], "PixelMeasuresSequence", None)
                    if not pm or len(pm) == 0:
                        pm = DcmSequence([DcmDataset()])
                        sfgs[0].PixelMeasuresSequence = pm
                    pm_item = pm[0]
                    try:
                        pm_item.PixelSpacing = [sx, sy]
                    except Exception:
                        pass
                    try:
                        pm_item.SliceThickness = sz
                    except Exception:
                        pass
                    try:
                        pm_item.SpacingBetweenSlices = sz
                    except Exception:
                        pass
                # PlaneOrientationSequence
                if need_iop:
                    po = getattr(sfgs[0], "PlaneOrientationSequence", None)
                    if not po or len(po) == 0:
                        po = DcmSequence([DcmDataset()])
                        sfgs[0].PlaneOrientationSequence = po
                    po_item = po[0]
                    try:
                        po_item.ImageOrientationPatient = list(default_iop)
                    except Exception:
                        pass
            except Exception as e:
                logs.append(f"SharedFG patch failed for {p.name}: {e}")

            # Save with normalized write (write_like_original=False)
            try:
                ds.save_as(str(p), write_like_original=False)
                changed += 1
                logs.append(
                    f"Patched DICOM {p.name}: set PixelSpacing=({sx:.3f},{sy:.3f}), "
                    f"SliceThickness/SpacingBetweenSlices={sz:.3f}, IOP={'set' if need_iop else 'skip'}"
                )
            except Exception as e:
                logs.append(f"Failed to save patched DICOM {p.name}: {e}")

        except Exception as e:
            logs.append(f"Patch error for {p.name}: {e}")

    return total, changed, logs


def _nib_convert_multiframe_missingmeta(
    src_dir: Path,
    out_dir: Path,
    approx_slice_count: Optional[int] = None,
) -> Tuple[Optional[Path], List[str], List[str]]:
    """Convert suspected multiframe MR (with missing meta) to NIfTI via nibabel without dcm2niix.

    Plan:
      - Read DICOM as a volume with SimpleITK (do not trust direction/spacing)
      - Convert to numpy (Z,Y,X) -> transpose to (X,Y,Z)
      - Infer default spacing (slice-count heuristic) and set diagonal affine (RAS)
      - Save as SeriesInstanceUID.nii.gz under out_dir
    """
    logs: List[str] = []
    errors: List[str] = []

    # Get SeriesInstanceUID (fallback to directory name on failure)
    series_uid = None
    rep_ds = None
    try:
        rep_dcm_path = next(p for p in src_dir.iterdir() if is_dicom_file(p))
        rep_ds = pydicom.dcmread(str(rep_dcm_path), stop_before_pixels=True, force=True)
        series_uid = str(rep_ds.get("SeriesInstanceUID", "")).strip() or None
    except StopIteration:
        pass
    except Exception as e:
        logs.append(f"Warning: Failed to read representative DICOM: {e}")
    if not series_uid:
        series_uid = src_dir.name

    # Build volume via SimpleITK
    try:
        img_sitk, _ = build_dicom_volume_by_series_uid(src_dir, series_uid)
    except Exception as e:
        errors.append(f"SITK volume build failed: {e}")
        return None, logs, errors

    # Decide default spacing based on slice count
    if approx_slice_count is None:
        try:
            size = img_sitk.GetSize()  # (X,Y,Z)
            approx_slice_count = int(size[2])
        except Exception:
            approx_slice_count = None
    spacing_xyz, spacing_rule = _infer_multiframe_default_spacing(approx_slice_count)
    logs.append(f"Nib convert spacing rule={spacing_rule}; slices={approx_slice_count}")

    # Convert to numpy array (SITK: Z,Y,X) -> (X,Y,Z)
    try:
        arr_zyx = sitk.GetArrayFromImage(img_sitk)  # (Z,Y,X)
        if arr_zyx is None:
            raise RuntimeError("Empty image array")
        if arr_zyx.ndim == 4 and arr_zyx.shape[0] == 1:
            arr_zyx = arr_zyx[0]
        if arr_zyx.ndim != 3:
            raise RuntimeError(f"Unexpected ndim={arr_zyx.ndim}")
        arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    except Exception as e:
        errors.append(f"Array conversion failed: {e}")
        return None, logs, errors

    # Save NIfTI (diagonal affine in RAS convention)
    try:
        sx, sy, sz = [float(x) for x in spacing_xyz]
        affine = np.eye(4, dtype=float)
        affine[0, 0] = sx
        affine[1, 1] = sy
        affine[2, 2] = sz
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{series_uid}.nii.gz"
        nib.save(nib.Nifti1Image(arr_xyz, affine), str(out_path))
        logs.append(f"Nib-converted NIfTI written: {out_path.name}")
        return out_path, logs, errors
    except Exception as e:
        errors.append(f"Failed to save NIfTI via nibabel: {e}")
        return None, logs, errors


def run_gdcmconv_raw(src_dir: Path, dst_dir: Path) -> Tuple[int, str, str]:
    gdcm_path = which_or_die("gdcmconv")
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Re-save all DICOMs with --raw
    processed = 0
    errs = []
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        outp = dst_dir / p.name
        code, out, err = run_cmd([gdcm_path, "--raw", str(p), str(outp)])
        if code == 0:
            processed += 1
        else:
            errs.append((p.name, err))
    if processed == 0:
        return 1, "", "gdcmconv processed 0 files: " + "; ".join(e for _, e in errs)
    return 0, f"gdcmconv processed {processed} files", "\n".join(f"{n}: {e}" for n, e in errs)


# =============================
# Annotation coordinate conversion
# =============================
CoordMode = Literal[
    "pixel_corner_0",  # top-left pixel corner is (0,0) (common in GUIs)
    "pixel_center_0",  # top-left pixel center is (0,0)
    "pixel_corner_1",  # top-left corner is (1,1)
    "pixel_center_1",  # top-left center is (1,1)
    "norm_corner",  # normalized [0..1] with corner convention (scale by Cols/Rows)
    "norm_center",  # normalized [0..1] with center convention
]


def _uv_from_xy(x: float, y: float, rows: int, cols: int, mode: CoordMode) -> Tuple[float, float]:
    # Align u=column(x)=i, v=row(y)=j to ITK's index space (pixel center at integer)
    if mode == "pixel_corner_0":  # corner-origin -> shift to center by +0.5
        return x + 0.5, y + 0.5
    if mode == "pixel_center_0":  # center-origin -> keep as-is
        return x, y
    if mode == "pixel_corner_1":  # corner at (1,1) -> shift by -0.5
        return x - 0.5, y - 0.5
    if mode == "pixel_center_1":  # center at (1,1) -> shift by -1.0
        return x - 1.0, y - 1.0
    if mode == "norm_corner":  # normalized corner -> scale then +0.5
        return x * cols + 0.5, y * rows + 0.5
    if mode == "norm_center":  # normalized center -> scale only
        return x * cols, y * rows
    raise ValueError(mode)


def build_dicom_volume_by_series_uid(dicom_root: Path, series_uid: str) -> Tuple[sitk.Image, Dict[str, int]]:
    """Build a 3D DICOM image for the given SeriesInstanceUID.
    Returns the image and a mapping SOPInstanceUID -> slice index (k).
    """
    dicom_root = Path(dicom_root)
    rdr = sitk.ImageSeriesReader()
    rdr.MetaDataDictionaryArrayUpdateOn()
    rdr.LoadPrivateTagsOn()

    # Select the target series UID from the available list
    series_ids = list(rdr.GetGDCMSeriesIDs(str(dicom_root)))
    if series_uid not in series_ids:
        # Some datasets are expanded in subfolders; search recursively
        for sub in dicom_root.rglob("*"):
            if not sub.is_dir():
                continue
            try:
                ids = list(rdr.GetGDCMSeriesIDs(str(sub)))
            except Exception:
                continue
            if series_uid in ids:
                dicom_root = sub
                break

    file_list = rdr.GetGDCMSeriesFileNames(str(dicom_root), series_uid)
    if len(file_list) == 0:
        raise RuntimeError(f"No DICOM files for SeriesInstanceUID={series_uid}")

    rdr.SetFileNames(file_list)
    img = rdr.Execute()  # Image with origin/spacing/direction in ITK LPS world

    # Convert 4D image to 3D if needed
    if img.GetDimension() == 4:
        size = list(img.GetSize())
        if size[3] == 1:
            # When T-dim is 1, extract to 3D
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize([size[0], size[1], size[2], 0])
            extractor.SetIndex([0, 0, 0, 0])
            img3d = extractor.Execute(img)
        else:
            # Error when the 4th dimension is greater than 1
            raise RuntimeError(f"Unexpected 4D volume with T={size[3]} for SeriesInstanceUID={series_uid}")
    else:
        img3d = img

    # Build mapping from SOPInstanceUID to slice index (k)
    sop2k: Dict[str, int] = {}
    for i, f in enumerate(file_list):
        try:
            sop = rdr.GetMetaData(i, "0008|0018")  # SOPInstanceUID
        except Exception:
            # Rarely missing; fall back to pydicom
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            sop = str(ds.get("SOPInstanceUID", ""))
        sop2k[sop] = i
    return img3d, sop2k


def convert_dicom_to_nifti_coords(
    dicom_x: float,
    dicom_y: float,
    sop_uid: str,
    series_uid: str,
    dicom_root: Path,
    nifti_path: Path,
    dicom_f: int = -1,  # frame index for multiframe DICOM (0-based)
    coord_mode: CoordMode = "pixel_corner_0",
) -> Tuple[float, float, float]:
    """
    Convert DICOM (x,y on a SOP slice) to NIfTI continuous voxel (i,j,k).
    All transforms via SimpleITK (delegate LPS/RAS handling to SITK).

    For multiframe DICOMs:
    - dicom_f >= 0: use as frame index (0-based)
    - dicom_f < 0: use standard SOPInstanceUID-based slice
    """
    # 1) Build 3D DICOM image and get SOP -> k
    img_dcm, sop2k = build_dicom_volume_by_series_uid(dicom_root, series_uid)

    # For multiframe DICOM, use the frame index directly (0-based)
    if dicom_f >= 0:
        k = dicom_f
    else:
        if sop_uid not in sop2k:
            raise KeyError(f"SOPInstanceUID not found in series: {sop_uid}")
        k = sop2k[sop_uid]

    # 2) Align (x,y) to ITK index space (pixel centers are integers)
    size = img_dcm.GetSize()  # (X,Y,Z)
    cols, rows = size[0], size[1]
    u, v = _uv_from_xy(dicom_x, dicom_y, rows=rows, cols=cols, mode=coord_mode)

    # 3) DICOM index -> physical point (LPS, mm)
    #    Use TransformContinuousIndexToPhysicalPoint for continuous indices
    P_lps = img_dcm.TransformContinuousIndexToPhysicalPoint((u, v, float(k)))  # (x,y,z) in mm (LPS)

    # 4) Read NIfTI with SimpleITK and map physical point -> continuous voxel index
    img_nii = sitk.ReadImage(str(nifti_path))  # loaded in ITK-style LPS world
    ijk = img_nii.TransformPhysicalPointToContinuousIndex(P_lps)  # (i,j,k) in voxel (continuous)
    return float(ijk[0]), float(ijk[1]), float(ijk[2])


def load_annotations(csv_path: Path) -> pd.DataFrame:
    """Load the annotation CSV and parse coordinates."""
    try:
        df = pd.read_csv(csv_path)
        # Convert 'coordinates' column to dicts
        df["coords_dict"] = df["coordinates"].apply(lambda x: ast.literal_eval(x))
        df["dicom_x"] = df["coords_dict"].apply(lambda x: x["x"])
        df["dicom_y"] = df["coords_dict"].apply(lambda x: x["y"])
        df["dicom_f"] = df["coords_dict"].apply(lambda x: int(x.get("f", -1)))
        return df
    except Exception as e:
        logging.error(f"Failed to read annotation CSV: {e}")
        return pd.DataFrame()


def get_multiframe_series_with_annotations(annotations_df: pd.DataFrame) -> set:
    """Return the set of SeriesInstanceUIDs that are multiframe MR with annotations."""
    if annotations_df.empty:
        return set()

    # dicom_f >= 0 indicates multiframe DICOM annotation (0-based)
    multiframe_annots = annotations_df[annotations_df["dicom_f"] >= 0]
    return set(multiframe_annots["SeriesInstanceUID"].unique())


# =============================
# QC / metadata aggregation
# ===============================


def load_json_sidecar(json_path: Path) -> Dict:
    if not json_path.exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_series_metadata_json(
    series_dir: Path, dataset: pydicom.Dataset, extra: Optional[Dict] = None
) -> None:
    """Write a metadata JSON into the series directory.

    Notes:
      - Merge DICOM-derived metadata with `extra` when provided.
    """

    metadata = extract_metadata_from_dataset(
        dataset,
        metadata_version=METADATA_VERSION,
    )
    if not metadata:
        metadata = {}

    # Merge additional info (expect JSON-serializable primitives)
    if extra:
        try:
            for k, v in extra.items():
                # Minimal normalization for numpy scalar-like values
                if hasattr(v, "item"):
                    try:
                        metadata[k] = v.item()
                        continue
                    except Exception:
                        pass
                metadata[k] = v
        except Exception:
            # Even if extra is malformed, still write the base metadata
            pass

    metadata_path = series_dir / SERIES_METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def gather_qc_for_nifti(nii_path: Path, cfg: QCConfig) -> Dict:
    rec: Dict = {
        "nifti_path": str(nii_path),
        "json_path": (
            str(nii_path.with_suffix(nii_path.suffix + ".json"))
            if nii_path.suffix != ".gz"
            else str(nii_path.with_suffix("").with_suffix(".json"))
        ),
        "ok": False,
        "flags": [],
    }
    try:
        img = nib.load(str(nii_path))
        hdr = img.header
        shape = tuple(int(x) for x in img.shape[:3])  # first 3 axes of 3D/4D volume
        zooms = hdr.get_zooms()[:3]
        dtype = str(hdr.get_data_dtype())
        data = img.get_fdata(dtype=np.float32)  # real values (scl_slope/inter applied)
        # Robust intensity stats
        lo = np.percentile(data, cfg.intensity_lo_clip)
        hi = np.percentile(data, cfg.intensity_hi_clip)
        mn = float(np.nanmin(data))
        mx = float(np.nanmax(data))
        med = float(np.nanmedian(data))
        rec.update(
            {
                "Nx": shape[0],
                "Ny": shape[1],
                "Nz": shape[2],
                "dx": float(zooms[0]),
                "dy": float(zooms[1]),
                "dz": float(zooms[2]),
                "dtype": dtype,
                "min": mn,
                "p0.5": float(lo),
                "median": med,
                "p99.5": float(hi),
                "max": mx,
            }
        )
        flags = []
        # slab (thin volume)
        if shape[2] < cfg.slab_slices or (shape[2] * zooms[2]) < cfg.z_span_min_mm:
            flags.append("slab")
        # long volume
        if shape[2] >= cfg.min_slices and zooms[2] <= cfg.max_dz_mm:
            rec["long_volume"] = True
        else:
            rec["long_volume"] = False
        # XY near-isotropy
        aniso = abs(zooms[0] - zooms[1]) / max(1e-6, (0.5 * (zooms[0] + zooms[1])))
        rec["xy_aniso"] = float(aniso)
        if aniso > cfg.xy_aniso_tol:
            flags.append("xy_anisotropic")
        # Derived checks from JSON, etc.
        json_path = Path(rec["json_path"]) if rec["json_path"] else None
        if not json_path or not json_path.exists():
            flags.append("json_missing")
            json_meta = {}
        else:
            json_meta = load_json_sidecar(json_path)
            img_type = [x.upper() for x in json_meta.get("ImageType", [])]
            if "DERIVED" in img_type and "ORIGINAL" not in img_type:
                flags.append("derived_only")
        # Light anomaly detection on intensity range (coarse; adjust as needed)
        if np.isfinite(mn) and np.isfinite(mx) and (mx - mn) < 1e-3:
            flags.append("flat_intensity")
        rec["flags"] = flags
        rec["ok"] = True
    except Exception as e:
        rec["error"] = str(e)
        rec["flags"] = rec.get("flags", []) + ["load_failed"]
    return rec


# =============================
# Standalone DICOM-to-NIfTI conversion helpers
# =============================


def convert_dicom_to_nifti(
    dir_path: Path,
    out_dir: Path,
    tmp_root: Path,
    use_majority_size: bool = True,
    copy_mode: str = "auto",
    file_exts: Tuple[str, ...] = (".dcm", ""),
    dcm2niix_flags: Tuple[str, ...] = ("-z", "y", "-b", "y", "-i", "n", "-f", "%s"),
    gdcm_first: bool = False,
    tmp_subdir: Optional[Path] = None,  # allow external temporary folder
    use_slice_spacing_filter: bool = True,
    slice_spacing_tolerance: float = 2.0,
    convert_to_ras: bool = True,
) -> Tuple[Optional[Path], Path, List[str], List[str]]:
    """
    Generate a NIfTI file from a DICOM directory.
    Decoupled to reuse in Kaggle submission runtime.

    Args:
        dir_path: Directory containing DICOM files
        out_dir: Output directory for NIfTI
        tmp_root: Directory for temporary files
        use_majority_size: Use only DICOMs with the majority size
        copy_mode: File copy mode ("auto"|"symlink"|"copy")
        file_exts: Target DICOM file extensions
        dcm2niix_flags: Flags for dcm2niix command
        gdcm_first: Whether to use gdcmconv first
        tmp_subdir: External temp folder (for majority subset)
        use_slice_spacing_filter: Whether to filter by slice spacing
        slice_spacing_tolerance: Tolerance multiplier for slice spacing filtering
        convert_to_ras: Whether to normalize to RAS (False keeps original orientation)

    Returns:
        (single generated NIfTI path or None, actual DICOM directory used, logs, errors)
    """
    nifti_files = []
    logs = []
    errors = []
    use_dir = dir_path  # default to the original directory
    pre_dir_created_here: Optional[Path] = None
    # Track conversion path and applied mitigations
    conversion_method = "dcm2niix"
    used_nib_conversion = False
    mf_defaults_pre_applied = False
    mf_defaults_post_applied = False
    orientation_aware_reconvert = False
    # Quick record: number of source DICOMs and MF detection
    source_dicom_file_count: Optional[int] = None
    mf_detect_info: Optional[Dict[str, object]] = None

    try:
        # Audit DICOM folder
        audit = audit_folder(dir_path, file_exts)
        logs.append(
            f"Audit {dir_path}: dicom={audit.dicom_files}, "
            f"majority={audit.majority_key}, series={len(audit.series_map)}"
        )

        if audit.dicom_files == 0:
            logs.append("No DICOM files. Skipped.")
            return nifti_files, dir_path, logs, errors

        # Decide source directory for input
        use_dir = dir_path
        tmp_subdir_created_here = False  # created within this function

        if use_majority_size and audit.majority_key is not None:
            # Create only if not provided externally
            if tmp_subdir is None:
                tmp_subdir = Path(tempfile.mkdtemp(dir=tmp_root))
                tmp_subdir_created_here = True
            n, slice_filter_log = prepare_majority_subset(
                dir_path,
                tmp_subdir,
                audit.majority_key,
                copy_mode=copy_mode,
                slice_spacing_filter=use_slice_spacing_filter,
                slice_spacing_tolerance=slice_spacing_tolerance,
            )
            logs.append(f"Prepared majority subset: {n} files -> {tmp_subdir} (with slice spacing filter)")
            if slice_filter_log:
                logs.append(slice_filter_log)
            if n > 0:
                use_dir = tmp_subdir

        # Pre-conversion meta patch (fill missing MF MR meta)
        try:
            if _env_flag("RSNA_ENABLE_MF_DEFAULTS_PRECONVERT", False):
                mf_info_pre = _detect_multiframe_mr_missing_meta(use_dir)
                if mf_info_pre.get("is_multiframe_mr") and (
                    mf_info_pre.get("missing_pixel_spacing") or mf_info_pre.get("missing_orientation")
                ):
                    # Create temp patch dir and copy all DICOMs
                    pre_dir = Path(tempfile.mkdtemp(dir=tmp_root))
                    pre_dir_created_here = pre_dir
                    copied = 0
                    for p in use_dir.iterdir():
                        if p.is_file() and is_dicom_file(p):
                            shutil.copy2(p, pre_dir / p.name)
                            copied += 1
                    if copied == 0:
                        logs.append("Preconvert patch skipped: no DICOM files to copy")
                    else:
                        slice_count = mf_info_pre.get("approx_slice_count")
                        spacing_xyz, spacing_rule = _infer_multiframe_default_spacing(slice_count)
                        total, changed, patch_logs = _apply_multiframe_defaults_to_dicom_dir(
                            pre_dir, spacing_xyz=spacing_xyz
                        )
                        logs.extend(patch_logs)
                        logs.append(
                            f"Preconvert MF defaults applied: files={total}, changed={changed}, rule={spacing_rule}"
                        )
                        if changed > 0:
                            use_dir = pre_dir
                            mf_defaults_pre_applied = True
        except Exception as e:
            logs.append(f"Warning: preconvert MF patch failed: {e}")

        # Record auxiliary info based on the directory actually used
        try:
            cnt = 0
            for p in use_dir.iterdir():
                if p.is_file() and is_dicom_file(p):
                    cnt += 1
            source_dicom_file_count = cnt
        except Exception:
            source_dicom_file_count = None
        try:
            mf_detect_info = _detect_multiframe_mr_missing_meta(use_dir)
        except Exception:
            mf_detect_info = None

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Helper for conversion
        def convert_with_dcm2niix(src: Path, dst: Path) -> Tuple[bool, str]:
            code, out, err = run_dcm2niix(src, dst, dcm2niix_flags)
            ok = (code == 0) and any(
                str(dst).endswith(".nii") or str(dst).endswith(".nii.gz") for dst in dst.glob("*.nii*")
            )
            msg = out + "\n" + err
            return ok, msg

        # ========== Optional: direct conversion via nibabel (for MF MR missing meta) ==========
        selected_nifti: Optional[Path] = None
        try:
            if _env_flag("RSNA_ENABLE_MF_NIB_CONVERT", False):
                mf_info_nib = _detect_multiframe_mr_missing_meta(use_dir)
                if mf_info_nib.get("is_multiframe_mr") and (
                    mf_info_nib.get("missing_pixel_spacing") or mf_info_nib.get("missing_orientation")
                ):
                    sc = mf_info_nib.get("approx_slice_count")
                    nib_path, nib_logs, nib_errs = _nib_convert_multiframe_missingmeta(
                        use_dir, out_dir, approx_slice_count=sc
                    )
                    logs.extend(nib_logs)
                    errors.extend(nib_errs)
                    if nib_path is not None:
                        selected_nifti = nib_path
                        logs.append("Used nibabel conversion for MF missing meta")
                        used_nib_conversion = True
                        conversion_method = "nibabel"
        except Exception as e:
            logs.append(f"Warning: nibabel conversion path failed: {e}")

        if selected_nifti is None:
            # ========== Normal: convert with dcm2niix ==========
            if gdcm_first:
                raw_dir = Path(tempfile.mkdtemp(dir=tmp_root))
                code, out, err = run_gdcmconv_raw(use_dir, raw_dir)
                logs.append(out)
                if code != 0:
                    errors.append("gdcmconv failed before dcm2niix: " + err)
                ok, msg = convert_with_dcm2niix(raw_dir, out_dir)
                if not ok:
                    errors.append("dcm2niix failed after gdcmconv: \n" + msg)
            else:
                ok, msg = convert_with_dcm2niix(use_dir, out_dir)
                if not ok:
                    logs.append("dcm2niix direct failed; trying gdcmconv --raw -> dcm2niix")
                    raw_dir = Path(tempfile.mkdtemp(dir=tmp_root))
                    code, out, err = run_gdcmconv_raw(use_dir, raw_dir)
                    logs.append(out)
                    if code == 0:
                        ok2, msg2 = convert_with_dcm2niix(raw_dir, out_dir)
                        ok = ok2
                        msg += "\n" + msg2
                    if not ok:
                        errors.append("Both direct and fallback conversions failed.\n" + msg)

            # Collect generated NIfTI files
            for nii in out_dir.glob("*.nii*"):
                nifti_files.append(nii)

            # Select one among multiple files (e.g., Dual Energy CT)
            selected_nifti, select_logs, select_errors = select_primary_nifti_file(nifti_files)
            logs.extend(select_logs)
            errors.extend(select_errors)

            if selected_nifti is None:
                # No files generated or selection failed
                return None, use_dir, logs, errors

        # Normalize to 3D (single file only)
        try:
            result_3d = force_3d_nii(selected_nifti)
            if not result_3d:
                # If force_3d_nii returns empty, treat as error
                errors.append(
                    f"force_3d_nii failed for {selected_nifti}: 4D with T>1 or unsupported dimension"
                )
                return None, use_dir, logs, errors
            # force_3d_nii should return exactly one file
            selected_nifti = result_3d[0]

        except RuntimeError as e:
            if "OrthonormalError" in str(e):
                # Fallback when an OrthonormalError is detected
                logs.append(f"OrthonormalError detected: {e}")
                logs.append("Retrying with orientation-aware majority subset...")

                # Orientation-aware majority audit
                audit_with_orient = audit_folder_with_orientation(dir_path, file_exts)
                if audit_with_orient.majority_key is None:
                    errors.append("No majority found even with orientation filtering")
                    return None, use_dir, logs, errors

                logs.append(f"Orientation-aware majority: {audit_with_orient.majority_key}")

                # Clean up existing files in tmp_subdir (reuse directory)
                if tmp_subdir.exists():
                    for file in tmp_subdir.glob("*"):
                        if file.is_file():
                            file.unlink()
                    logs.append(f"Cleaned up temporary directory: {tmp_subdir}")

                # Create orientation-aware majority subset in the temp directory
                n_orient, slice_filter_log = prepare_majority_subset_with_orientation(
                    dir_path,
                    tmp_subdir,
                    audit_with_orient.majority_key,
                    copy_mode=copy_mode,
                    slice_spacing_filter=use_slice_spacing_filter,
                    slice_spacing_tolerance=slice_spacing_tolerance,
                )
                logs.append(
                    (
                        "Orientation-filtered subset: "
                        f"{n_orient} files -> {tmp_subdir} "
                        "(with slice spacing filter)"
                    )
                )
                if slice_filter_log:
                    logs.append(slice_filter_log)

                if n_orient == 0:
                    errors.append("No files selected by orientation-aware filtering")
                    return None, use_dir, logs, errors

                # Re-convert using files filtered by orientation
                # Remove any existing NIfTI files
                if selected_nifti.exists():
                    selected_nifti.unlink()
                json_file = selected_nifti.with_suffix("").with_suffix(".json")
                if json_file.exists():
                    json_file.unlink()

                # Perform re-conversion
                ok, msg = convert_with_dcm2niix(tmp_subdir, out_dir)
                if not ok:
                    errors.append("Orientation-aware re-conversion failed: " + msg)
                    return None, use_dir, logs, errors

                # Get files generated by re-conversion
                new_nifti_files = list(out_dir.glob("*.nii*"))
                if not new_nifti_files:
                    errors.append("No NIfTI files generated by orientation-aware re-conversion")
                    return None, use_dir, logs, errors

                # Select the primary file again
                new_selected_nifti, select_logs, select_errors = select_primary_nifti_file(new_nifti_files)
                logs.extend(select_logs)
                errors.extend(select_errors)

                if new_selected_nifti is None:
                    errors.append("Failed to select primary file after orientation-aware re-conversion")
                    return None, use_dir, logs, errors

                # Retry 3D normalization
                result_3d = force_3d_nii(new_selected_nifti)
                if not result_3d:
                    errors.append("force_3d_nii failed even after orientation filtering")
                    return None, use_dir, logs, errors

                selected_nifti = result_3d[0]
                use_dir = tmp_subdir  # update used directory
                orientation_aware_reconvert = True
                logs.append("Successfully recovered from OrthonormalError with orientation filtering")

            else:
                # Other RuntimeError
                errors.append(f"Unexpected RuntimeError in force_3d_nii: {e}")
                return None, use_dir, logs, errors

        # ===== Apply fallback defaults for missing multiframe MR metadata =====
        try:
            if _env_flag("RSNA_ENABLE_MF_DEFAULTS", False):
                mf_info = _detect_multiframe_mr_missing_meta(use_dir)
                if mf_info["is_multiframe_mr"] and (
                    mf_info["missing_pixel_spacing"] or mf_info["missing_orientation"]
                ):
                    slice_count = mf_info.get("approx_slice_count")
                    if slice_count is None:
                        slice_count = _estimate_slice_count_from_nifti(selected_nifti)

                    spacing_xyz, spacing_rule = _infer_multiframe_default_spacing(slice_count)

                    _apply_multiframe_defaults_to_nifti(
                        selected_nifti,
                        spacing_xyz=spacing_xyz,
                    )
                    sx, sy, sz = spacing_xyz
                    mf_defaults_post_applied = True

                    # Do not update JSON sidecar to avoid I/O instability in submission env

                    logs.append(
                        "Applied MF defaults: spacing(x,y,z)="
                        f"({sx:.3f},{sy:.3f},{sz:.3f}); rule={spacing_rule}; "
                        f"slice_count={slice_count}"
                    )
        except Exception as e:
            logs.append(f"Warning: MF defaults application failed: {e}")

        if convert_to_ras:
            # Convert to RAS orientation (single file only)
            ras_success, ras_logs, ras_errors = convert_to_ras_orientation(selected_nifti)
            logs.extend(ras_logs)
            errors.extend(ras_errors)

            if not ras_success:
                # Treat RAS conversion failure as error
                return None, use_dir, logs, errors

        # Obtain representative DICOM metadata and save series JSON with extra info
        try:
            ds_demo: Optional[pydicom.Dataset] = None
            try:
                rep_dcm_path = next(p for p in use_dir.iterdir() if is_dicom_file(p))
                ds_demo = pydicom.dcmread(str(rep_dcm_path), stop_before_pixels=True, force=True)
            except StopIteration:
                ds_demo = None
            except Exception as e:
                logs.append(f"Warning: Failed to load metadata: {e}")

            if ds_demo is not None:
                try:
                    extra_meta: Dict[str, object] = {
                        "IsMultiframeMR": (
                            bool(mf_detect_info.get("is_multiframe_mr", False)) if mf_detect_info else False
                        ),
                        "MissingPixelSpacing": (
                            bool(mf_detect_info.get("missing_pixel_spacing", False))
                            if mf_detect_info
                            else False
                        ),
                        "MissingOrientation": (
                            bool(mf_detect_info.get("missing_orientation", False))
                            if mf_detect_info
                            else False
                        ),
                        "ApproxSliceCount": (
                            mf_detect_info.get("approx_slice_count") if mf_detect_info else None
                        ),
                        "SourceDICOMFileCount": source_dicom_file_count,
                        "ConvertedWith": conversion_method,
                        "AppliedMFDefaultsPreconvert": mf_defaults_pre_applied,
                        "AppliedMFDefaultsPost": mf_defaults_post_applied,
                        "OrientationAwareReconvert": orientation_aware_reconvert,
                    }
                    write_series_metadata_json(selected_nifti.parent, ds_demo, extra=extra_meta)
                except Exception as e:
                    logs.append(f"Warning: Failed to write metadata JSON: {e}")
        except Exception:
            pass

        # Clean up temp directory only if created here
        if tmp_subdir_created_here and tmp_subdir and tmp_subdir.exists():
            shutil.rmtree(tmp_subdir, ignore_errors=True)
        if pre_dir_created_here and pre_dir_created_here.exists():
            shutil.rmtree(pre_dir_created_here, ignore_errors=True)

    except Exception as e:
        errors.append(f"Unhandled error for {dir_path}: {e}")
        return None, use_dir, logs, errors

    # Return final single NIfTI file
    return selected_nifti, use_dir, logs, errors


def select_primary_nifti_file(nifti_files: List[Path]) -> Tuple[Optional[Path], List[str], List[str]]:
    """
    Select a primary NIfTI file among multiple (e.g., Dual Energy CT).

    Args:
        nifti_files: List of NIfTI files

    Returns:
        (selected NIfTI file, log messages, error messages)
    """
    logs = []
    errors = []

    if len(nifti_files) == 0:
        return None, logs, errors
    elif len(nifti_files) == 1:
        return nifti_files[0], logs, errors

    # Handle case with multiple files
    logs.append(f"Multiple NIfTI files detected ({len(nifti_files)} files). " f"Selecting primary file.")

    # Sort by file name (lexicographic)
    nifti_files_sorted = sorted(nifti_files, key=lambda x: x.name)

    # Separate files without underscores from those with underscores
    files_without_underscore = []
    files_with_underscore = []

    for nii in nifti_files_sorted:
        if "_" in nii.stem:
            files_with_underscore.append(nii)
        else:
            files_without_underscore.append(nii)

    # Select primary file
    if files_without_underscore:
        # If any file without underscores exists, select the first
        selected_nifti = files_without_underscore[0]
        files_to_remove = files_with_underscore + files_without_underscore[1:]
    else:
        # Otherwise, select the first after sorting
        selected_nifti = files_with_underscore[0]
        files_to_remove = files_with_underscore[1:]

    # Remove files to be deleted
    for nii in files_to_remove:
        try:
            # Remove NIfTI file
            if nii.exists():
                nii.unlink()
                logs.append(f"Removed duplicate file: {nii.name}")

            # Remove corresponding JSON file
            json_file = nii.with_suffix("").with_suffix(".json")
            if json_file.exists():
                json_file.unlink()
                logs.append(f"Removed associated JSON: {json_file.name}")

        except Exception as e:
            errors.append(f"Failed to remove {nii}: {e}")

    logs.append(f"Selected primary file: {selected_nifti.name}")
    return selected_nifti, logs, errors


def convert_to_ras_orientation(nifti_path: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Convert a single NIfTI file to RAS orientation.

    Args:
        nifti_path: Path to the NIfTI to convert

    Returns:
        (success flag, logs, errors)
    """
    logs = []
    errors = []

    try:
        # Load NIfTI and convert to RAS orientation
        _, _, ras_nifti = load_nifti_and_convert_to_ras(nifti_path)

        # Overwrite NIfTI at the same path after RAS conversion
        nib.save(ras_nifti, str(nifti_path))

        # If JSON sidecar also needs to be updated
        json_path = (
            nifti_path.with_suffix(".json")
            if nifti_path.suffix != ".gz"
            else nifti_path.with_suffix("").with_suffix(".json")
        )
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                # Record that RAS conversion was applied
                json_data["Orientation"] = "RAS"
                json_data["OrientationConverted"] = True

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logs.append(f"Warning: Failed to update JSON sidecar for RAS orientation: {e}")

        logs.append(f"Converted to RAS orientation: {nifti_path}")
        return True, logs, errors

    except Exception as e:
        errors.append(f"Failed to convert {nifti_path} to RAS orientation: {e}")
        return False, logs, errors


def force_3d_nii(nifti_path: Path) -> list[Path]:
    """
    Normalize a NIfTI to 3D.
      - 3D: no-op
      - 4D with T==1: extract 3D and overwrite same name
      - 4D with T>1: unsupported; return empty list (error)
    Returns: output NIfTI paths (when overwriting: [nifti_path]); empty on failure
    """
    try:
        img = sitk.ReadImage(str(nifti_path))
    except Exception as e:
        # Check SimpleITK read errors
        if "orthonormal direction cosines" in str(e):
            # Re-raise as specific error on orthonormal failure
            raise RuntimeError(f"OrthonormalError: {e}")
        else:
            # Other errors
            print(f"[force_3d_nii] SimpleITK read error: {e}")
            return []

    dim = img.GetDimension()
    if dim == 3:
        return [nifti_path]

    if dim != 4:
        print(f"[force_3d_nii] Unsupported dim={dim}: {nifti_path}")
        return []  # return empty to indicate failure

    sx, sy, sz, st = img.GetSize()
    out_paths: list[Path] = []

    if st == 1:
        # Keep as 3D and overwrite
        img3 = sitk.Extract(img, size=[sx, sy, sz, 0], index=[0, 0, 0, 0])
        sitk.WriteImage(img3, str(nifti_path))
        out_paths.append(nifti_path)
    else:
        # 4D with T>1 unsupported
        print(f"[force_3d_nii] ERROR: 4D with T={st}>1 not supported: {nifti_path}")
        return []  # return empty to indicate error

    return out_paths


# =============================
# Main process
# =============================


def enumerate_candidate_dirs(root: Path, multiframe_series: Optional[set] = None) -> List[Path]:
    """Enumerate candidate subdirectories (recursive) that contain DICOM files.
    * Pass each subdirectory as a unit to dcm2niix
    * Exclude directories without any DICOM files
    * Sort by directory name (usually the UID)
    * If multiframe_series is given, return only directories for those SeriesInstanceUIDs
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)

        # When multiframe_series is specified, check if dirname contains the SeriesInstanceUID
        if multiframe_series is not None:
            # Often, the directory name is the SeriesInstanceUID
            if p.name not in multiframe_series:
                continue

        # Check that the directory directly contains a DICOM
        has_dicom = False
        for name in filenames:
            fp = p / name
            if is_dicom_file(fp):
                has_dicom = True
                break
        if has_dicom:
            candidates.append(p)
    # Sort by directory name (usually SeriesInstanceUID)
    candidates.sort(key=lambda x: x.name)
    return candidates


def process_one_dir(
    dir_path: Path,
    run_cfg: RunConfig,
    qc_cfg: QCConfig,
    out_root: Path,
    tmp_root: Path,
    annotations_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """Process one directory and return QC records for generated NIfTI (dict of lists).
    Swallow exceptions here and return logs to the caller.
    """
    result = {
        "records": [],
        "logs": [],
        "errors": [],
    }

    tmp_subdir = None  # temporary folder (also used for annotation conversion)

    try:
        # Compute output directory
        rel = dir_path.relative_to(run_cfg.dicom_root)
        out_dir = out_root / rel

        # If using majority size, prepare a temporary folder first
        if run_cfg.use_majority_size:
            # Audit the folder and ensure we have a majority key
            audit = audit_folder(dir_path, run_cfg.file_exts)
            if audit.majority_key is not None:
                tmp_subdir = Path(tempfile.mkdtemp(dir=tmp_root))
                result["logs"].append(f"Created tmp_subdir for majority subset: {tmp_subdir}")

        # Call independent conversion function
        nifti_file, used_dicom_dir, logs, errors = convert_dicom_to_nifti(
            dir_path=dir_path,
            out_dir=out_dir,
            tmp_root=tmp_root,
            use_majority_size=run_cfg.use_majority_size,
            copy_mode=run_cfg.copy_mode,
            file_exts=run_cfg.file_exts,
            dcm2niix_flags=run_cfg.dcm2niix_flags,
            gdcm_first=run_cfg.gdcm_first,
            tmp_subdir=tmp_subdir,  # pass the temporary folder
            use_slice_spacing_filter=run_cfg.use_slice_spacing_filter,
            slice_spacing_tolerance=run_cfg.slice_spacing_tolerance,
            convert_to_ras=run_cfg.convert_to_ras,
        )

        result["logs"].extend(logs)
        result["errors"].extend(errors)

        # On conversion failure
        if nifti_file is None:
            if not errors:
                result["logs"].append("No NIfTI files generated.")

            # Create an error record for CSV
            error_rec = {
                "nifti_path": "",
                "json_path": "",
                "ok": False,
                "flags": ["conversion_failed"],
                "InputDir": str(dir_path),
                "OutputDir": str(out_dir),
                "error": "; ".join(errors) if errors else "No NIfTI files generated",
            }

            # Obtain DICOM info if possible
            try:
                rep_dcm = next(p for p in dir_path.iterdir() if is_dicom_file(p))
                ds = pydicom.dcmread(str(rep_dcm), stop_before_pixels=True, force=True)
                error_rec.update(
                    {
                        "PatientID": str(ds.get("PatientID", "")),
                        "StudyInstanceUID": str(ds.get("StudyInstanceUID", "")),
                        "SeriesInstanceUID": str(ds.get("SeriesInstanceUID", "")),
                    }
                )
            except Exception:
                error_rec.update(
                    {
                        "PatientID": "",
                        "StudyInstanceUID": "",
                        "SeriesInstanceUID": "",
                    }
                )

            result["records"].append(error_rec)
            return result

        # QC collection (single file only)
        rec = gather_qc_for_nifti(nifti_file, qc_cfg)
        # Attach light input series info (PatientID/SeriesInstanceUID, etc.)
        series_uid = ""
        try:
            # Take from the representative DICOM (first in dir_path)
            rep_dcm = next(p for p in dir_path.iterdir() if is_dicom_file(p))
            ds = pydicom.dcmread(str(rep_dcm), stop_before_pixels=True, force=True)
            series_uid = str(ds.get("SeriesInstanceUID", ""))
            rec.update(
                {
                    "PatientID": str(ds.get("PatientID", "")),
                    "StudyInstanceUID": str(ds.get("StudyInstanceUID", "")),
                    "SeriesInstanceUID": series_uid,
                    "InputDir": str(dir_path),
                    "OutputDir": str(out_dir),
                }
            )
        except Exception:
            rec.update(
                {
                    "PatientID": "",
                    "StudyInstanceUID": "",
                    "SeriesInstanceUID": "",
                    "InputDir": str(dir_path),
                    "OutputDir": str(out_dir),
                }
            )

        # Annotation coordinate conversion
        if annotations_df is not None and series_uid:
            series_annot = annotations_df[annotations_df["SeriesInstanceUID"] == series_uid]
            if not series_annot.empty:
                annot_list = []

                for _, annot_row in series_annot.iterrows():
                    # For multiframe DICOMs, use dicom_f (0-based)
                    dicom_f = annot_row.get("dicom_f", -1)

                    nifti_coords = convert_dicom_to_nifti_coords(
                        annot_row["dicom_x"],
                        annot_row["dicom_y"],
                        annot_row["SOPInstanceUID"],
                        series_uid,
                        used_dicom_dir,  # DICOM dir actually used for conversion
                        nifti_file,  # single NIfTI file
                        dicom_f,  # multiframe DICOM frame index (0-based)
                    )

                    if nifti_coords:
                        annot_list.append(
                            {
                                "SOPInstanceUID": annot_row["SOPInstanceUID"],
                                "location": annot_row["location"],
                                "dicom_x": annot_row["dicom_x"],
                                "dicom_y": annot_row["dicom_y"],
                                "dicom_f": (
                                    dicom_f if dicom_f >= 0 else None
                                ),  # also record frame index (0-based)
                                "nifti_x": nifti_coords[0],
                                "nifti_y": nifti_coords[1],
                                "nifti_z": nifti_coords[2],
                            }
                        )

                if annot_list:
                    rec["annotations"] = annot_list
                    result["logs"].append(f"Series {series_uid}: converted {len(annot_list)} annotations")

                    # Save per-file annotation info into JSON
                    annot_json_path = nifti_file.with_suffix(".annotations.json")
                    try:
                        with open(annot_json_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "SeriesInstanceUID": series_uid,
                                    "nifti_file": nifti_file.name,
                                    "annotations": annot_list,
                                },
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )
                        result["logs"].append(f"Saved annotation JSON: {annot_json_path.name}")
                    except Exception as e:
                        result["errors"].append(f"Failed to save annotation JSON: {e}")

        result["records"].append(rec)

    except Exception as e:
        result["errors"].append(f"Unhandled error for {dir_path}: {e}")

        # On error, remove the output directory as well
        if out_dir.exists():
            try:
                shutil.rmtree(out_dir, ignore_errors=True)
                result["logs"].append(f"Cleaned up output directory after error: {out_dir}")
            except Exception as cleanup_e:
                result["errors"].append(f"Failed to clean up output directory: {cleanup_e}")

    finally:
        # Cleanup temporary folder
        if tmp_subdir and tmp_subdir.exists() and not run_cfg.keep_intermediate:
            try:
                shutil.rmtree(tmp_subdir, ignore_errors=True)
                result["logs"].append(f"Cleaned up tmp_subdir: {tmp_subdir}")
            except Exception as e:
                result["errors"].append(f"Failed to clean up tmp_subdir: {e}")
    return result


def main():
    parser = argparse.ArgumentParser(description="DICOM to NIfTI batch conversion (with QC)")
    parser.add_argument("--dicom-root", type=Path, default=Path("/workspace/data/series"))
    parser.add_argument("--out-root", type=Path, default=Path("/workspace/data/series_niix"))
    parser.add_argument("--tmp-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--copy-mode", choices=["auto", "symlink", "copy"], default="auto")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--min-slices", type=int, default=80)
    parser.add_argument("--max-dz", type=float, default=1.5)
    parser.add_argument("--xy-aniso-tol", type=float, default=0.12)
    parser.add_argument("--slab-slices", type=int, default=32)
    parser.add_argument("--z-span-min", type=float, default=30.0)
    parser.add_argument("--gdcm-first", action="store_true", help="Run gdcmconv --raw before conversion")
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("/workspace/data/train_localizers.csv"),
        help="Path to annotations CSV file",
    )
    parser.add_argument(
        "--no-slice-spacing-filter",
        action="store_true",
        help="Disable slice spacing filtering",
    )
    parser.add_argument(
        "--slice-spacing-tolerance",
        type=float,
        default=2.0,
        help="Slice spacing tolerance (multiplier from median, default 2.0)",
    )
    parser.add_argument(
        "--multiframe-only",
        action="store_true",
        help="Process only multiframe DICOM series with annotations",
    )
    parser.add_argument(
        "--no-ras",
        action="store_true",
        help="Disable conversion to RAS (keep original orientation)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Ensure required commands exist
    try:
        which_or_die("dcm2niix")
        which_or_die("gdcmconv")
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    qc_cfg = QCConfig(
        min_slices=args.min_slices,
        max_dz_mm=args.max_dz,
        xy_aniso_tol=args.xy_aniso_tol,
        slab_slices=args.slab_slices,
        z_span_min_mm=args.z_span_min,
    )
    run_cfg = RunConfig(
        dicom_root=args.dicom_root.resolve(),
        out_root=args.out_root.resolve(),
        tmp_root=args.tmp_root.resolve(),
        num_workers=args.num_workers,
        copy_mode=args.copy_mode,
        keep_intermediate=args.keep_intermediate,
        gdcm_first=args.gdcm_first,
        annotations_csv=args.annotations_csv.resolve() if args.annotations_csv else None,
        use_slice_spacing_filter=not args.no_slice_spacing_filter,
        slice_spacing_tolerance=args.slice_spacing_tolerance,
        multiframe_only=args.multiframe_only,
        convert_to_ras=not args.no_ras,
    )

    # When multiframe-only is enabled, add suffix to output directory
    if run_cfg.multiframe_only:
        run_cfg.out_root = run_cfg.out_root.parent / f"{run_cfg.out_root.name}_multiframe"
        logging.info(f"Multiframe-only output directory: {run_cfg.out_root}")

    run_cfg.out_root.mkdir(parents=True, exist_ok=True)
    run_cfg.tmp_root.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations_df = None
    multiframe_series = None

    if run_cfg.annotations_csv and run_cfg.annotations_csv.exists():
        annotations_df = load_annotations(run_cfg.annotations_csv)
        if not annotations_df.empty:
            logging.info(f"Loaded annotations: {len(annotations_df)} rows")

            # If multiframe-only, collect the target series
            if run_cfg.multiframe_only:
                multiframe_series = get_multiframe_series_with_annotations(annotations_df)
                logging.info(f"Number of multiframe DICOM series with annotations: {len(multiframe_series)}")
                if not multiframe_series:
                    logging.warning("No multiframe DICOM series with annotations found.")
                    sys.exit(0)

    candidates = enumerate_candidate_dirs(run_cfg.dicom_root, multiframe_series)
    if not candidates:
        if run_cfg.multiframe_only:
            logging.error("No specified multiframe DICOM directories found.")
        else:
            logging.error("No DICOM-containing subdirectories found.")
        sys.exit(2)

    if run_cfg.multiframe_only:
        logging.info(f"Found {len(candidates)} multiframe DICOM directories with annotations.")
    else:
        logging.info(f"Found {len(candidates)} candidate directories.")

    all_records: List[Dict] = []
    all_logs: List[str] = []
    all_errors: List[str] = []

    try:
        with futures.ThreadPoolExecutor(max_workers=run_cfg.num_workers) as ex:
            jobs = [
                ex.submit(
                    process_one_dir, p, run_cfg, qc_cfg, run_cfg.out_root, run_cfg.tmp_root, annotations_df
                )
                for p in candidates
            ]
            for job in tqdm(futures.as_completed(jobs), total=len(jobs), desc="Converting"):
                res = job.result()
                all_records.extend(res.get("records", []))
                all_logs.extend(res.get("logs", []))
                all_errors.extend(res.get("errors", []))

        # Build DataFrame
        df = pd.DataFrame(all_records)
        # 'flags' is a list; stringify it
        if not df.empty and "flags" in df.columns:
            df["flags"] = df["flags"].apply(lambda x: ",".join(x) if isinstance(x, list) else str(x))

            # Drop 'annotations' column to avoid JSON serialization issues
            if "annotations" in df.columns:
                df = df.drop(columns=["annotations"])

        summary_csv = run_cfg.out_root / "meta_summary.csv"
        df.to_csv(summary_csv, index=False)

        # suspects: ok=False
        if not df.empty:
            suspects = df[df["ok"] == False]  # noqa: E712
        else:
            suspects = pd.DataFrame()
        suspects_csv = run_cfg.out_root / "suspects.csv"
        suspects.to_csv(suspects_csv, index=False)

        logging.info(f"Done. Summary: {summary_csv}")
        logging.info(f"Suspects: {suspects_csv}")

    except Exception as e:
        logging.error(f"Error occurred during processing: {str(e)}")
        all_errors.append(f"Main process error: {str(e)}")
        raise

    finally:
        # Always save logs and errors even on failure
        try:
            (run_cfg.out_root / "logs.txt").write_text("\n".join(all_logs), encoding="utf-8")
            logging.info(f"Saved logs: {run_cfg.out_root / 'logs.txt'}")
        except Exception as e:
            logging.error(f"Failed to save logs: {str(e)}")

        try:
            (run_cfg.out_root / "errors.txt").write_text("\n".join(all_errors), encoding="utf-8")
            if all_errors:
                logging.warning(f"Saved error log: {run_cfg.out_root / 'errors.txt'}")
        except Exception as e:
            logging.error(f"Failed to save error log: {str(e)}")

        # Save a summary JSON when possible
        try:
            report = {
                "total_dirs": len(candidates),
                "total_outputs": len(all_records),
                "num_errors": len(all_errors),
                "out_root": str(run_cfg.out_root),
            }
            (run_cfg.out_root / "conversion_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logging.info(f"Saved report: {run_cfg.out_root / 'conversion_report.json'}")
        except Exception as e:
            logging.error(f"Failed to save report: {str(e)}")


if __name__ == "__main__":
    main()
