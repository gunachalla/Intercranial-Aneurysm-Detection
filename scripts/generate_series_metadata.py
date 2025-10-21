#!/usr/bin/env python3
"""Generate auxiliary metadata JSON from DICOM headers for existing NIfTI series.

This script scans per-series NIfTI folders, reads a representative DICOM from the
corresponding DICOM series, extracts key header fields, and writes a
`series_metadata.json` alongside the NIfTI files.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pydicom
from tqdm import tqdm

from src.my_utils.dicom_metadata import (
    convert_age_to_years,
    extract_metadata_from_dataset,
)

SERIES_METADATA_FILENAME = "series_metadata.json"
METADATA_VERSION = 1


@dataclass
class SeriesProcessResult:
    """Summary of a single series processing result."""

    updated: int
    failed: int
    missing_json: int
    missing_dicom: int
    no_metadata: int
    age: Optional[str]
    sex: Optional[str]


def load_representative_dataset(dicom_dir: Path) -> Optional[pydicom.Dataset]:
    """Load a representative DICOM within a series (header only)."""
    for dcm_path in sorted(dicom_dir.iterdir()):
        if not dcm_path.is_file():
            continue
        try:
            return pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        except Exception:
            continue
    return None


def process_series(
    series_dir: Path,
    dicom_root: Path,
    dry_run: bool = False,
) -> SeriesProcessResult:
    """Generate metadata JSON for a single series directory."""
    json_paths = [p for p in series_dir.glob("*.json") if not p.name.endswith(".annotations.json")]
    if not json_paths:
        return SeriesProcessResult(0, 0, 1, 0, 0, None, None)

    dicom_dir = dicom_root / series_dir.name
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        return SeriesProcessResult(0, 0, 0, 1, 0, None, None)

    dataset = load_representative_dataset(dicom_dir)
    if dataset is None:
        return SeriesProcessResult(0, 0, 0, 0, 1, None, None)

    metadata = extract_metadata_from_dataset(
        dataset,
        metadata_version=METADATA_VERSION,
    )
    age = metadata.get("PatientAge")
    sex = metadata.get("PatientSex")

    if not metadata:
        return SeriesProcessResult(0, 0, 0, 0, 1, age, sex)

    metadata_path = series_dir / SERIES_METADATA_FILENAME

    if dry_run:
        return SeriesProcessResult(1, 0, 0, 0, 0, age, sex)

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception:
        logging.exception("Exception occurred while saving metadata JSON: %s", metadata_path)
        return SeriesProcessResult(0, 1, 0, 0, 0, age, sex)

    return SeriesProcessResult(1, 0, 0, 0, 0, age, sex)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niix-root",
        type=Path,
        default=Path("data/series_niix"),
        help="Root directory containing per-series NIfTI and JSON files.",
    )
    parser.add_argument(
        "--dicom-root",
        type=Path,
        default=Path("data/series"),
        help="Root directory containing corresponding DICOM series.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing files; report statistics only.",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Process only the first N series (for sampling).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    niix_root: Path = args.niix_root
    dicom_root: Path = args.dicom_root

    if not niix_root.exists() or not niix_root.is_dir():
        raise FileNotFoundError(f"NIfTI root not found: {niix_root}")
    if not dicom_root.exists() or not dicom_root.is_dir():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")

    series_dirs = sorted([p for p in niix_root.iterdir() if p.is_dir()])
    if args.max_series is not None:
        series_dirs = series_dirs[: args.max_series]

    total = len(series_dirs)
    total_updated = 0
    total_failed = 0
    total_missing_json = 0
    total_missing_dicom = 0
    total_no_metadata = 0

    age_counter: Counter[str] = Counter()
    sex_counter: Counter[str] = Counter()
    age_years_values: List[float] = []
    age_missing_count = 0
    age_unparsable_count = 0
    sex_missing_count = 0

    logging.info("Number of target series: %d", total)
    if args.dry_run:
        logging.info("Running in dry-run mode. Metadata JSON will not be written.")

    for series_dir in tqdm(series_dirs, desc="Processing", unit="series"):
        result = process_series(
            series_dir,
            dicom_root,
            dry_run=args.dry_run,
        )

        total_updated += result.updated
        total_failed += result.failed
        total_missing_json += result.missing_json
        total_missing_dicom += result.missing_dicom
        total_no_metadata += result.no_metadata

        stats_count = result.updated + result.failed
        if stats_count > 0:
            if result.age:
                age_counter[result.age] += stats_count
                age_years = convert_age_to_years(result.age)
                if age_years is not None:
                    age_years_values.extend([age_years] * stats_count)
                else:
                    age_unparsable_count += stats_count
            else:
                age_missing_count += stats_count

            if result.sex:
                sex_counter[result.sex] += stats_count
            else:
                sex_missing_count += stats_count

    logging.info("Completed: %d JSON written", total_updated)
    if total_failed > 0:
        logging.info("Failed to write: %d JSON", total_failed)
    logging.info("Skipped (no JSON found): %d series", total_missing_json)
    logging.info("Skipped (no DICOM found): %d series", total_missing_dicom)
    logging.info("Skipped (metadata unavailable): %d series", total_no_metadata)

    logging.info("--- Age distribution ---")
    if age_counter:
        for age_value, count in age_counter.most_common(20):
            logging.info("Age %s: %d series", age_value, count)
    else:
        logging.info("Could not obtain age information")
    logging.info("Age missing: %d series", age_missing_count)
    logging.info("Age parsing failed: %d series", age_unparsable_count)
    if age_years_values:
        min_age = min(age_years_values)
        max_age = max(age_years_values)
        mean_age = sum(age_years_values) / len(age_years_values)
        logging.info(
            "Age stats (in years): min=%.2f, max=%.2f, mean=%.2f, n=%d",
            min_age,
            max_age,
            mean_age,
            len(age_years_values),
        )

    logging.info("--- Sex distribution ---")
    if sex_counter:
        for sex_value, count in sex_counter.most_common():
            logging.info("Sex %s: %d series", sex_value, count)
    else:
        logging.info("Could not obtain sex information")
    logging.info("Sex missing: %d series", sex_missing_count)


if __name__ == "__main__":
    main()
