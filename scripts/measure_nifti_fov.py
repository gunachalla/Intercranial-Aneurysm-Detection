#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure physical field-of-view (FOV) of NIfTI volumes and export CSV/JSON.

Expected input:
  - Directory layout: /workspace/data/series_niix/{SeriesInstanceUID}/*.nii.gz
  - Volumes are assumed to be RAS-oriented (axis codes are recorded otherwise)

Outputs:
  - CSV: Per-file shape, voxel spacing, physical size (mm), etc.
  - JSON: Dataset-level summary statistics (min/percentiles/max)

Usage example:
  python scripts/measure_nifti_fov.py \
    --input-dir /workspace/data/series_niix \
    --output-csv /workspace/outputs/orientation_size_summary/fov_sizes.csv \
    --output-json /workspace/outputs/orientation_size_summary/fov_sizes_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from nibabel.affines import voxel_sizes as affine_voxel_sizes
from nibabel.orientations import aff2axcodes


def list_nifti_files(root: Path) -> List[Path]:
    """Recursively list NIfTI files under the input directory."""
    patterns = ("*.nii.gz", "*.nii")
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    # 安定した順序で処理
    files = sorted(files)
    return files


def analyze_nifti(nifti_path: Path) -> Dict:
    """Extract shape/spacing/physical size from NIfTI header and affine.

    Notes:
      - If the shape is 4D or higher, only the first three axes are analyzed.
      - Spacing is computed from the affine (header zooms are also recorded).
    """
    img = nib.load(str(nifti_path))
    shape = img.shape
    if len(shape) < 3:
        raise ValueError(f"NIfTI with fewer than 3 dimensions is not supported: {nifti_path} (shape={shape})")

    # 4D以上は先頭3軸のみ
    x, y, z = shape[:3]

    affine = img.affine
    # アフィンから各軸のボクセル間隔（mm）を取得
    try:
        sx, sy, sz = affine_voxel_sizes(affine)
    except Exception:
        # フォールバックとしてヘッダのzoomsを使用
        zooms = img.header.get_zooms()[:3]
        sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])

    # 実寸サイズ（mm）: ボクセル数 × ボクセル間隔
    size_x_mm = float(sx * x)
    size_y_mm = float(sy * y)
    size_z_mm = float(sz * z)

    # Reference: header zooms (differences can help investigation)
    header_zooms = tuple(float(v) for v in img.header.get_zooms()[:3])

    # Axis codes (e.g., (R, A, S))
    try:
        codes = aff2axcodes(affine)
        axcodes = "".join(codes)
    except Exception:
        axcodes = "UNKNOWN"

    # SeriesInstanceUID equivalent (use parent directory name as UID)
    series_uid = nifti_path.parent.name

    info = {
        "series_uid": series_uid,
        "file": str(nifti_path),
        "axcodes": axcodes,
        # Shape (nibabel arrays are in (X, Y, Z) order)
        "shape_x": int(x),
        "shape_y": int(y),
        "shape_z": int(z),
        # Voxel spacing (mm)
        "spacing_x_mm": float(sx),
        "spacing_y_mm": float(sy),
        "spacing_z_mm": float(sz),
        # Physical size (mm)
        "size_x_mm": size_x_mm,
        "size_y_mm": size_y_mm,
        "size_z_mm": size_z_mm,
        # 参考情報
        "header_zooms_x": float(header_zooms[0]),
        "header_zooms_y": float(header_zooms[1]),
        "header_zooms_z": float(header_zooms[2]),
        "ndim": int(len(shape)),
        "extra_dims": list(shape[3:]) if len(shape) > 3 else [],
    }
    return info


def summarize_stats(rows: List[Dict]) -> Dict:
    """Compute dataset-level summary statistics."""
    def col(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=float)
        q = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100]).tolist()
        return {
            "min": float(q[0]),
            "p05": float(q[1]),
            "p25": float(q[2]),
            "p50": float(q[3]),
            "p75": float(q[4]),
            "p95": float(q[5]),
            "max": float(q[6]),
        }

    stats = {
        "count_files": len(rows),
        "size_x_mm": col([r["size_x_mm"] for r in rows]),
        "size_y_mm": col([r["size_y_mm"] for r in rows]),
        "size_z_mm": col([r["size_z_mm"] for r in rows]),
        "spacing_x_mm": col([r["spacing_x_mm"] for r in rows]),
        "spacing_y_mm": col([r["spacing_y_mm"] for r in rows]),
        "spacing_z_mm": col([r["spacing_z_mm"] for r in rows]),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Measure physical FOV of NIfTI volumes and save CSV/JSON")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/data/series_niix"),
        help="Root directory containing NIfTI files (recursively searched).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/workspace/outputs/orientation_size_summary/fov_sizes.csv"),
        help="CSV file to save per-file measurements.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/workspace/outputs/orientation_size_summary/fov_sizes_stats.json"),
        help="JSON file to save dataset-level statistics.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    out_csv: Path = args.output_csv
    out_json: Path = args.output_json

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    files = list_nifti_files(input_dir)
    print(f"Scan target: {input_dir}  NIfTI count: {len(files)}")
    if not files:
        print("No NIfTI files found. Aborting.")
        return

    rows: List[Dict] = []
    errors: List[Tuple[str, str]] = []

    for i, f in enumerate(files, 1):
        try:
            info = analyze_nifti(f)
            rows.append(info)
        except Exception as e:
            errors.append((str(f), str(e)))
            continue
        if i % 50 == 0 or i == len(files):
            print(f"Progress: {i}/{len(files)} processed")

    # CSV保存
    fieldnames = [
        "series_uid",
        "file",
        "axcodes",
        "shape_x",
        "shape_y",
        "shape_z",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "size_x_mm",
        "size_y_mm",
        "size_z_mm",
        "header_zooms_x",
        "header_zooms_y",
        "header_zooms_z",
        "ndim",
        "extra_dims",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Save statistics as JSON
    stats = summarize_stats(rows)
    result = {
        "input_dir": str(input_dir),
        "n_files": len(rows),
        "n_errors": len(errors),
        "stats": stats,
        "errors": errors,
    }
    with out_json.open("w", encoding="utf-8") as fj:
        json.dump(result, fj, ensure_ascii=False, indent=2)

    # Console output (highlights only)
    print("\n=== Summary (excerpt) ===")
    for k in ("size_x_mm", "size_y_mm", "size_z_mm"):
        s = stats.get(k, {})
        if not s:
            continue
        print(
            f"{k}: min={s['min']:.1f}  p05={s['p05']:.1f}  p50={s['p50']:.1f}  p95={s['p95']:.1f}  max={s['max']:.1f}"
        )
    if errors:
        print(f"\nWarning: {len(errors)} files failed (see JSON for details)")


if __name__ == "__main__":
    main()
