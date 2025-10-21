#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to move erroneous data into a backup directory.

Moves SeriesInstanceUIDs listed in error_data.yaml from
series_niix and/or _series_niix to a backup directory.
"""

import yaml
import shutil
from pathlib import Path
from typing import List, Set
import logging
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_error_series_ids(yaml_path: Path) -> Set[str]:
    """Load erroneous SeriesInstanceUIDs from error_data.yaml."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        # Load YAML (list expected)
        data = yaml.safe_load(f)

    # Convert to a set if it's a list
    if isinstance(data, list):
        series_ids = set(data)
    else:
        series_ids = set()
        logger.warning("YAML file format is not as expected")

    logger.info(f"Error series count: {len(series_ids)}")
    return series_ids


def move_error_data(series_ids: Set[str], source_dirs: List[Path], backup_root: Path):
    """Move erroneous data to the backup directory."""

    moved_count = 0
    not_found_count = 0

    for series_id in tqdm(series_ids, desc="Moving error data"):
        found = False

        for source_dir in source_dirs:
            source_path = source_dir / series_id

            if source_path.exists():
                # Create backup directory
                relative_path = source_dir.name
                backup_path = backup_root / relative_path / series_id
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                # Move the data
                try:
                    shutil.move(str(source_path), str(backup_path))
                    logger.debug(f"Moved: {source_path} -> {backup_path}")
                    found = True
                except Exception as e:
                    logger.error(f"Move failed {series_id}: {e}")

        if found:
            moved_count += 1
        else:
            not_found_count += 1
            logger.warning(f"Data not found: {series_id}")

    logger.info(f"Moved: {moved_count}")
    logger.info(f"Not found: {not_found_count}")

    return moved_count, not_found_count


def verify_backup(backup_root: Path, series_ids: Set[str]):
    """Verify moved data in the backup directory."""
    verified_count = 0

    for backup_dir in backup_root.glob("*/"):
        for series_dir in backup_dir.glob("*/"):
            if series_dir.name in series_ids:
                verified_count += 1

    logger.info(f"Backup verified: {verified_count}/{len(series_ids)}")
    return verified_count


def main():
    """Main entry point."""
    # Paths
    yaml_path = Path("/workspace/data/error_data.yaml")
    data_root = Path("/workspace/data")
    backup_root = data_root / "series_niix_error_data_backup"

    # Target directories
    source_dirs = [
        data_root / "series_niix",
    ]

    # Load error SeriesInstanceUIDs
    series_ids = load_error_series_ids(yaml_path)

    if not series_ids:
        logger.warning("No error series found")
        return

    # Prepare backup directory
    backup_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backup root: {backup_root}")

    # Move error data
    moved_count, not_found_count = move_error_data(
        series_ids=series_ids, source_dirs=source_dirs, backup_root=backup_root
    )

    # Verify backup
    verify_backup(backup_root, series_ids)

    # Summary
    print("\n" + "=" * 50)
    print("Finished moving error data")
    print("=" * 50)
    print(f"Total targets: {len(series_ids)}")
    print(f"Moved: {moved_count}")
    print(f"Not found: {not_found_count}")
    print(f"Backup root: {backup_root}")
    print("=" * 50)


if __name__ == "__main__":
    main()
