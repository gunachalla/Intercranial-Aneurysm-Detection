#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test utility: duplicate a DICOM series while removing spacing/orientation tags.

Use cases:
  - For multi-frame DICOMs (especially MR), remove PixelSpacing and
    ImageOrientationPatient from SharedFunctionalGroupsSequence, etc., to
    reproduce missing metadata at inference time.

Usage:
  python scripts/sanitize_dicom_remove_spacing_orientation.py \
    --src /path/to/src_series \
    --dst /path/to/dst_series \
    --remove-thickness 1

Notes:
  - PixelData is kept as is (load with pydicom and re-save).
  - Only delete tags if present; skip silently if not found.
  - Search comprehensively across top-level, Shared FG, and Per-Frame FG items.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import pydicom


def _maybe_del(ds, name: str) -> bool:
    """Delete the specified tag if present and return True if deleted."""
    if hasattr(ds, name):
        try:
            delattr(ds, name)
            return True
        except Exception:
            pass
    return False


def _remove_top_level(ds, *, remove_thickness: bool) -> Tuple[int, int]:
    """Remove top-level PixelSpacing / ImageOrientationPatient / (optional thickness tags)."""
    removed_spacing = 0
    removed_orient = 0
    if _maybe_del(ds, "PixelSpacing"):
        removed_spacing += 1
    if remove_thickness:
        if _maybe_del(ds, "SliceThickness"):
            removed_spacing += 1
        if _maybe_del(ds, "SpacingBetweenSlices"):
            removed_spacing += 1
    if _maybe_del(ds, "ImageOrientationPatient"):
        removed_orient += 1
    return removed_spacing, removed_orient


def _remove_shared_functional_groups(ds, *, remove_thickness: bool) -> Tuple[int, int]:
    """Remove PixelSpacing/IOP from SharedFunctionalGroupsSequence if present."""
    removed_spacing = 0
    removed_orient = 0
    try:
        sfgs = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if sfgs and len(sfgs) > 0:
            item = sfgs[0]
            # PixelMeasuresSequence -> PixelSpacing, SliceThickness, SpacingBetweenSlices
            try:
                pm = getattr(item, "PixelMeasuresSequence", None)
                if pm and len(pm) > 0:
                    pm0 = pm[0]
                    if _maybe_del(pm0, "PixelSpacing"):
                        removed_spacing += 1
                    if remove_thickness:
                        if _maybe_del(pm0, "SliceThickness"):
                            removed_spacing += 1
                        if _maybe_del(pm0, "SpacingBetweenSlices"):
                            removed_spacing += 1
            except Exception:
                pass
            # PlaneOrientationSequence -> ImageOrientationPatient
            try:
                po = getattr(item, "PlaneOrientationSequence", None)
                if po and len(po) > 0:
                    po0 = po[0]
                    if _maybe_del(po0, "ImageOrientationPatient"):
                        removed_orient += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed_spacing, removed_orient


def _remove_per_frame_functional_groups(ds, *, remove_thickness: bool) -> Tuple[int, int]:
    """Remove PixelSpacing/IOP from PerFrameFunctionalGroupsSequence if present."""
    removed_spacing = 0
    removed_orient = 0
    try:
        pfgs = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        if not pfgs:
            return 0, 0
        for pf in pfgs:
            # PixelMeasuresSequence（あれば）
            try:
                pm = getattr(pf, "PixelMeasuresSequence", None)
                if pm and len(pm) > 0:
                    pm0 = pm[0]
                    if _maybe_del(pm0, "PixelSpacing"):
                        removed_spacing += 1
                    if remove_thickness:
                        if _maybe_del(pm0, "SliceThickness"):
                            removed_spacing += 1
                        if _maybe_del(pm0, "SpacingBetweenSlices"):
                            removed_spacing += 1
            except Exception:
                pass
            # PlaneOrientationSequence -> ImageOrientationPatient（あれば）
            try:
                po = getattr(pf, "PlaneOrientationSequence", None)
                if po and len(po) > 0:
                    po0 = po[0]
                    if _maybe_del(po0, "ImageOrientationPatient"):
                        removed_orient += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed_spacing, removed_orient


def process_series(src: Path, dst: Path, *, remove_thickness: bool = True) -> None:
    """Duplicate DICOM files in a series folder while removing specified tags."""
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    rm_space = 0
    rm_orient = 0
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        # Skip non-DICOM files
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=True)
        except Exception:
            continue
        total += 1

        a, b = _remove_top_level(ds, remove_thickness=remove_thickness)
        rm_space += a
        rm_orient += b

        a, b = _remove_shared_functional_groups(ds, remove_thickness=remove_thickness)
        rm_space += a
        rm_orient += b

        a, b = _remove_per_frame_functional_groups(ds, remove_thickness=remove_thickness)
        rm_space += a
        rm_orient += b

        outp = dst / p.name
        ds.save_as(str(outp), write_like_original=False)

    print(
        f"Done: files={total}, removed_spacing_tags={rm_space}, removed_orientation_tags={rm_orient}"
    )


def main():
    parser = argparse.ArgumentParser(description="Duplicate a DICOM series after removing spacing/orientation headers")
    parser.add_argument("--src", type=Path, required=True, help="Input series directory")
    parser.add_argument("--dst", type=Path, required=True, help="Output series directory")
    parser.add_argument(
        "--remove-thickness", type=int, default=1, help="Also remove SliceThickness/SpacingBetweenSlices"
    )
    args = parser.parse_args()

    if not args.src.exists() or not args.src.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.src}")

    process_series(args.src, args.dst, remove_thickness=bool(args.remove_thickness))


if __name__ == "__main__":
    main()
