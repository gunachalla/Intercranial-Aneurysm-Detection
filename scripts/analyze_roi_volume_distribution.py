#!/usr/bin/env python
"""Utility to summarize ROI volume distributions under a vessel_pred_dir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze ROI volume distributions")
    parser.add_argument(
        "--vessel_pred_dir",
        type=Path,
        required=True,
        help="Inference results directory containing roi_data.npz/transform.json",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Path to save per-case statistics as CSV",
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
        help="Path to save aggregated summary as JSON",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
        help="Quantiles to output (specified in 0-1)",
    )
    return parser.parse_args()


def load_roi_array(case_dir: Path) -> np.ndarray:
    """Load the roi_data array file."""
    npz_path = case_dir / "roi_data.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            if "roi" not in data:
                raise KeyError(f"Missing 'roi' key in {npz_path}")
            return data["roi"]
    npy_path = case_dir / "roi_data.npy"
    if npy_path.exists():
        return np.load(npy_path)
    raise FileNotFoundError("roi_data.npz/roi_data.npy not found")


def load_transform(case_dir: Path) -> Dict:
    """Load transform.json as a dictionary."""
    transform_path = case_dir / "transform.json"
    if not transform_path.exists():
        raise FileNotFoundError("transform.json not found")
    with transform_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_spacing(transform: Dict) -> Optional[Sequence[float]]:
    """Select voxel spacing in ROI space."""
    spacing = transform.get("spacing_after_resampling")
    if spacing:
        return spacing
    return transform.get("spacing_original")


def ensure_dims(array: np.ndarray) -> Sequence[int]:
    """Extract (Z, Y, X) dimensions from the ROI array."""
    if array.ndim == 4 and array.shape[0] == 1:
        return array.shape[1:]
    if array.ndim == 3:
        return array.shape
    if array.ndim == 4:
        return array.shape[1:]
    raise ValueError(f"Unexpected array shape: {array.shape}")


def summarize_series(name: str, values: pd.Series, quantiles: Iterable[float]) -> Dict[str, float]:
    """Extract summary statistics from a Series."""
    stats = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }
    for q in quantiles:
        if 0.0 <= q <= 1.0:
            stats[f"q{int(q*100):02d}"] = float(values.quantile(q))
    return stats


def main() -> None:
    """Main routine."""
    args = parse_args()
    vessel_pred_dir: Path = args.vessel_pred_dir
    if not vessel_pred_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {vessel_pred_dir}")

    case_dirs = sorted([d for d in vessel_pred_dir.iterdir() if d.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No subdirectories found: {vessel_pred_dir}")

    records: List[Dict] = []
    skipped: List[str] = []

    for case_dir in case_dirs:
        try:
            roi = load_roi_array(case_dir)
            dims = ensure_dims(roi)
            transform = load_transform(case_dir)
            spacing = select_spacing(transform)

            dims_array = np.asarray(dims, dtype=np.int64)
            voxel_volume = int(np.prod(dims_array))
            volume_mm3: Optional[float] = None
            if spacing is not None:
                spacing_arr = np.asarray(list(spacing)[: len(dims_array)], dtype=np.float64)
                volume_mm3 = float(np.prod(spacing_arr) * voxel_volume)

            record = {
                "series_uid": case_dir.name,
                "dim_z": int(dims_array[0]),
                "dim_y": int(dims_array[1]),
                "dim_x": int(dims_array[2]),
                "voxels": voxel_volume,
            }
            if volume_mm3 is not None:
                record["volume_mm3"] = volume_mm3
            bbox = transform.get("roi_bbox_network_refined") or transform.get("roi_bbox_network")
            if bbox:
                record["roi_bbox_start_z"] = int(bbox[0][0])
                record["roi_bbox_start_y"] = int(bbox[1][0])
                record["roi_bbox_start_x"] = int(bbox[2][0])
            records.append(record)
        except Exception as exc:
            skipped.append(f"{case_dir.name}: {exc}")

    if not records:
        raise RuntimeError("No analyzable cases found")

    df = pd.DataFrame(records)

    print(f"Number of analyzable cases: {len(df)}")
    if skipped:
        print("\nCases failed to load:")
        for msg in skipped:
            print(f"  - {msg}")

    quantiles = tuple(sorted({q for q in args.quantiles if 0.0 <= q <= 1.0}))

    stats_dims = {
        axis: summarize_series(axis, df[f"dim_{axis[-1]}"], quantiles) for axis in ("dim_z", "dim_y", "dim_x")
    }
    stats_vox = summarize_series("voxels", df["voxels"], quantiles)
    stats_mm3: Optional[Dict[str, float]] = None
    outlier_info: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    print("\nVoxel count per axis:")
    for axis_key, stats in stats_dims.items():
        print(f"  {axis_key}:")
        for k, v in stats.items():
            print(f"    {k:>4}: {v:.3f}")

    print("\nROI volume (voxels):")
    for k, v in stats_vox.items():
        print(f"  {k:>4}: {v:.3f}")

    if "volume_mm3" in df.columns:
        stats_mm3 = summarize_series("volume_mm3", df["volume_mm3"], quantiles)
        print("\nROI volume (mm^3):")
        for k, v in stats_mm3.items():
            print(f"  {k:>4}: {v:.3f}")

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved per-case statistics: {args.save_csv}")

    # Extract outliers at quantile endpoints
    if quantiles:
        q_low = quantiles[0]
        q_high = quantiles[-1]
        metric_columns = [
            ("dim_z", "dim_z"),
            ("dim_y", "dim_y"),
            ("dim_x", "dim_x"),
            ("voxels", "voxels"),
        ]
        if "volume_mm3" in df.columns:
            metric_columns.append(("volume_mm3", "volume_mm3"))

        for json_key, col in metric_columns:
            series = df[col]
            low_thr = float(series.quantile(q_low))
            high_thr = float(series.quantile(q_high))
            low_df = df[series <= low_thr][["series_uid", col]]
            high_df = df[series >= high_thr][["series_uid", col]]
            outlier_info[json_key] = {
                "low": [
                    {"series_uid": str(row.series_uid), "value": float(getattr(row, col))}
                    for row in low_df.itertuples(index=False)
                ],
                "high": [
                    {"series_uid": str(row.series_uid), "value": float(getattr(row, col))}
                    for row in high_df.itertuples(index=False)
                ],
            }

    if args.save_summary is not None:
        summary = {
            "cases": len(df),
            "stats_dim_z": stats_dims["dim_z"],
            "stats_dim_y": stats_dims["dim_y"],
            "stats_dim_x": stats_dims["dim_x"],
            "stats_voxels": stats_vox,
        }
        if stats_mm3 is not None:
            summary["stats_volume_mm3"] = stats_mm3
        if outlier_info:
            summary["outliers"] = outlier_info
        args.save_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.save_summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary statistics: {args.save_summary}")


if __name__ == "__main__":
    main()
