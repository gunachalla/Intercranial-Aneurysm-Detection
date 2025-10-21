#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create nnUNet inference dataset.

Expose NIfTI files converted by rsna_dcm2niix.py in the nnUNet inference
format (filename convention: {uid}_0000.nii.gz). Since nnUNet handles
normalization, this script does not re-save images; instead it creates a
symlink to the original NIfTI (with automatic copy fallback if symlinks are
not allowed). The old normalization + re-save path and brain BBox cropping
were removed.

Input:
- Brain NIfTI: /workspace/data/series_niix/{uid}/*.nii.gz

Output:
- nnUNet inference format: /workspace/nnUNet_inference/imagesTs/
  (for each {uid}, create a symlink named {uid}_0000.nii.gz)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import shutil
from tqdm import tqdm
import argparse
import rootutils
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Removed: dependencies related to BBox cropping


# Removed: BBox JSON IO/cropping related code


def _safe_symlink(src: Path, dst: Path, absolute: bool = True) -> None:
    """
    Create a symbolic link utility.

    - Remove and recreate if an existing link/file is present.
    - If `absolute=True`, create an absolute symlink (robust to workspace moves).
    """
    # Remove existing output, if any
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    target = src.resolve() if absolute else src
    dst.symlink_to(target)


# Removed: normalization + re-save helper functions are no longer needed


def process_case(
    uid: str,
    input_dir: Path,
    output_dir: Path,
    case_id: int,
) -> Tuple[bool, str]:
    """
    Process a single case and expose it in nnUNet inference format.

    Args:
        uid: Case UID
        input_dir: Input NIfTI directory
        output_dir: Output directory
        case_id: Case ID (sequential)

    Returns:
        (success flag, message)
    """
    try:
        # Build input path
        case_dir = input_dir / uid

        # Find NIfTI files (usually a single .nii.gz)
        nifti_files = list(case_dir.glob("*.nii.gz"))
        if not nifti_files:
            return False, f"No NIfTI files found in {case_dir}"

        # Use the first NIfTI file (typically there is only one)
        image_path = nifti_files[0]

        # nnUNet filename for inference: {uid}_0000.nii.gz
        output_filename = output_dir / f"{uid}_0000.nii.gz"

        # Create symlink (absolute path)
        try:
            _safe_symlink(image_path, output_filename, absolute=True)
            return True, f"Linked {uid} -> {output_filename.name}"
        except OSError as e:
            # Fallback to copy if symlinks are not permitted
            try:
                shutil.copyfile(image_path, output_filename)
                return True, f"Copied {uid} -> {output_filename.name} (symlink fallback)"
            except Exception as ce:
                return False, f"Symlink and copy failed for {uid}: {str(ce)}"

    except Exception as e:
        return False, f"Error processing {uid}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Create nnUNet inference dataset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/data/series_niix"),
        help="NIfTI directory converted by rsna_dcm2niix.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/nnUNet_inference/imagesTs"),
        help="Output directory for inference data",
    )
    # Note: This script prefers symlinks (falls back to copy if not allowed)
    # Removed: BBox-related arguments
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Max number of cases to process (debug)",
    )
    parser.add_argument(
        "--start-case-id",
        type=int,
        default=1,
        help="Starting case ID number",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of workers for parallel processing",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect case directories from input
    case_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir()])

    if args.max_cases:
        case_dirs = case_dirs[: args.max_cases]

    print(f"Number of cases to process: {len(case_dirs)}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel workers: {args.num_workers}")

    # Removed: logs related to BBox

    # Process each case (in parallel)
    successful_cases = 0
    failed_cases = []
    processed_uids = []

    # Create partial for process_case (fix common args)
    process_case_partial = partial(
        process_case,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    # Create task list (UID, case_id)
    case_tasks = [(case_dir.name, args.start_case_id + idx) for idx, case_dir in enumerate(case_dirs)]

    # Run in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit each task
        future_to_case = {}
        for uid, case_id in case_tasks:
            future = executor.submit(process_case_partial, uid=uid, case_id=case_id)
            future_to_case[future] = (uid, case_id)

        # Collect results (with progress bar)
        for future in tqdm(as_completed(future_to_case), total=len(case_tasks), desc="Processing"):
            uid, case_id = future_to_case[future]
            try:
                success, message = future.result()
                if success:
                    successful_cases += 1
                    processed_uids.append((uid, f"{uid}_0000.nii.gz"))
                else:
                    failed_cases.append((uid, message))
            except Exception as e:
                failed_cases.append((uid, f"Parallel processing error: {str(e)}"))

    # Summary
    print("\nDone:")
    print(f"  Success: {successful_cases}/{len(case_dirs)} cases")

    if failed_cases:
        print(f"  Failed: {len(failed_cases)} cases")
        for uid, msg in failed_cases[:5]:
            print(f"    - {uid}: {msg}")
        if len(failed_cases) > 5:
            print(f"    ... and {len(failed_cases) - 5} more cases")

    # Create mapping file (UID to filename)
    if successful_cases > 0:
        mapping_file = args.output_dir.parent / "case_mapping.json"
        mapping_data = {
            "total_cases": successful_cases,
            "mappings": [{"uid": uid, "filename": filename} for uid, filename in processed_uids],
        }

        with open(mapping_file, "w") as f:
            json.dump(mapping_data, f, indent=2)

        print(f"\nCase mapping file: {mapping_file}")
        print("\nNext steps:")
        print("1. Ensure nnUNet environment variables are set:")
        print("   export nnUNet_raw=/workspace/nnUNet_raw")
        print("   export nnUNet_preprocessed=/workspace/nnUNet_preprocessed")
        print("   export nnUNet_results=/workspace/nnUNet_results")
        print("2. Run inference:")
        print(f"   nnUNetv2_predict -i {args.output_dir} -o /workspace/nnUNet_inference/predictions \\")
        print("     -d [DATASET_ID] -c 3d_fullres -f [FOLD or 'all']")
        print("\nNote: Set DATASET_ID and FOLD according to the trained model")


if __name__ == "__main__":
    main()
