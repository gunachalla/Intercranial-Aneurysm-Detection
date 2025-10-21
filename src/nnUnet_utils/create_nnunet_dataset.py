#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset creation script for nnUNet.

Creates training data in nnUNet format from brain images and vessel
segmentation masks. (Note: the old brain BBox cropping feature has been
removed.)

Inputs:
- Brain image: /workspace/data/segmentations/{uid}.nii
- Vessel segmentation: /workspace/data/segmentations/{uid}_cowseg.nii

Output:
- nnUNet-formatted dataset: /workspace/nnUNet_raw/Dataset001_VesselSegmentation/
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import pydicom
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# BBox crop feature removed


# Dataset-specific configuration by ID (more IDs may be added)
DATASET_CONFIGS: Dict[int, Dict[str, Any]] = {
    1: {
        "folder_name": "Dataset001_VesselSegmentation",
        "binary_vessel": False,
        "label_mapping": None,
        "labels": {
            "background": 0,
            "Other Posterior Circulation": 1,
            "Basilar Tip": 2,
            "Right Posterior Communicating Artery": 3,
            "Left Posterior Communicating Artery": 4,
            "Right Infraclinoid Internal Carotid Artery": 5,
            "Left Infraclinoid Internal Carotid Artery": 6,
            "Right Supraclinoid Internal Carotid Artery": 7,
            "Left Supraclinoid Internal Carotid Artery": 8,
            "Right Middle Cerebral Artery": 9,
            "Left Middle Cerebral Artery": 10,
            "Right Anterior Cerebral Artery": 11,
            "Left Anterior Cerebral Artery": 12,
            "Anterior Communicating Artery": 13,
        },
    },
    2: {
        "folder_name": "Dataset002_VesselROI",
        "binary_vessel": True,
        "label_mapping": None,
        "labels": {
            "background": 0,
            "vessel": 1,
        },
    },
    3: {
        "folder_name": "Dataset003_VesselGrouping",
        "binary_vessel": False,
        "label_mapping": {
            0: 0,
            1: 1,
            2: 1,
            3: 3,
            4: 3,
            5: 3,
            6: 3,
            7: 3,
            8: 3,
            9: 2,
            10: 2,
            11: 3,
            12: 3,
            13: 3,
        },
        "labels": {
            "background": 0,
            "Posterior_Circulation_and_Basilar": 1,
            "Middle_Cerebral_Arteries": 2,
            "Other_Locations": 3,
        },
    },
}


def get_dataset_config(dataset_id: int) -> Dict[str, Any]:
    """Return configuration for a given dataset ID."""

    if dataset_id not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset ID: {dataset_id}. Please add the configuration to DATASET_CONFIGS"
        )
    return DATASET_CONFIGS[dataset_id]


def get_modality_from_dicom(uid: str, series_base_dir: Path) -> str:
    """
    Extract modality from DICOM.

    Args:
        uid: Case UID
        series_base_dir: Base path of the series directory

    Returns:
        Modality (CT, MR, etc.) or "Unknown".
    """
    series_dir = series_base_dir / uid
    if not series_dir.exists():
        return "Unknown"

    # Read the first DICOM file
    dcm_files = list(series_dir.glob("*.dcm"))
    if not dcm_files:
        return "Unknown"

    try:
        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        return getattr(ds, "Modality", "Unknown")
    except Exception:
        return "Unknown"






def normalize_image(img: np.ndarray, modality: str) -> np.ndarray:
    """Normalize image intensities (simple percentile clipping + scaling)."""
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img -= img.min()
    img /= img.max()
    return img


def binarize_vessel_labels(label_data: np.ndarray) -> np.ndarray:
    """
    Binarize labels (background 0 / vessel 1).

    Args:
        label_data: Original label data (0=background, 1-13=vessel classes)

    Returns:
        Binarized labels (0/1).
    """
    # 0 is background; everything else becomes vessel (1)
    bin_label = (label_data > 0).astype(np.uint8)
    return bin_label


def process_case(
    uid: str,
    segmentation_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    series_base_dir: Path,
    binary_vessel: bool = False,
    label_mapping: Optional[Dict[int, int]] = None,
) -> Tuple[bool, str]:
    """
    Process a single case and save in nnUNet format.

    Args:
        uid: Case UID
        segmentation_dir: Path to the segmentation directory
        output_images_dir: Output directory for images
        output_labels_dir: Output directory for labels
        series_base_dir: Base path for the series directory
        binary_vessel: Whether to binarize labels
        label_mapping: Label mapping for class grouping (None to disable)

    Returns:
        (success flag, modality)
    """
    try:
        # File paths (new structure: no per-UID subdir)
        image_path = segmentation_dir / f"{uid}.nii"
        label_path = segmentation_dir / f"{uid}_cowseg.nii"

        # Get modality information
        modality = get_modality_from_dicom(uid, series_base_dir)

        # Check file existence
        if not image_path.exists():
            print(f"  Image file not found: {image_path}")
            return False, modality
        if not label_path.exists():
            print(f"  Label file not found: {label_path}")
            return False, modality

        # Load NIfTI files
        img_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        # Enforce RAS orientation
        img_nii = nib.as_closest_canonical(img_nii)
        label_nii = nib.as_closest_canonical(label_nii)

        # Extract arrays
        img_data = img_nii.get_fdata()
        label_data = label_nii.get_fdata().astype(np.uint8)

        cropped_img = img_data
        cropped_label = label_data
        affine = img_nii.affine.copy()

        # Apply class remapping if provided
        if label_mapping:
            # Validate unexpected labels
            unknown_labels = [val for val in np.unique(cropped_label) if val not in label_mapping]
            if unknown_labels:
                raise ValueError(f"Undefined label values were found: {unknown_labels}")
            remapped_label = np.zeros_like(cropped_label, dtype=np.uint8)
            for src_value, dst_value in label_mapping.items():
                remapped_label[cropped_label == src_value] = dst_value
            cropped_label = remapped_label

        # Optional binarization
        if binary_vessel:
            cropped_label = binarize_vessel_labels(cropped_label)

        # Normalize
        cropped_img = normalize_image(cropped_img, modality)

        # Create new NIfTI objects
        cropped_img_nii = nib.Nifti1Image(cropped_img, affine)
        cropped_label_nii = nib.Nifti1Image(cropped_label, affine)

        # nnUNet-style filenames (use UID)
        img_filename = output_images_dir / f"{uid}_0000.nii.gz"
        label_filename = output_labels_dir / f"{uid}.nii.gz"

        # Save
        nib.save(cropped_img_nii, str(img_filename))
        nib.save(cropped_label_nii, str(label_filename))

        return True, modality

    except Exception as e:
        print(f"  Error occurred: {e}")
        return False, "Unknown"


def create_dataset_json(
    output_dir: Path,
    num_training: int,
    labels: Dict[str, int],
    modality_counts: Dict[str, int],
) -> None:
    """
    Create dataset.json.

    Args:
        output_dir: Output directory
        num_training: Number of training cases
        labels: Label definitions
        modality_counts: Counts per modality
    """
    # Configure channel_names according to modality
    if len(modality_counts) == 1:
        # Single modality
        modality = list(modality_counts.keys())[0]
        if modality == "CT":
            channel_names = {"0": "CT"}  # Normalization mode for CT (clamp + z-score)
        elif modality == "MR":
            channel_names = {"0": "MR"}  # Normalization mode for MR (percentile + z-score)
        else:
            channel_names = {"0": "CT"}  # Default to CT
    else:
        # Mixed modalities
        channel_names = {"0": "MR"}  # Use MR mode for mixed cases
        print(f"Multiple modalities detected: {modality_counts}")
        print("Using MR normalization mode")

    dataset_info = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "modality_distribution": modality_counts,  # debug info
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Created dataset.json: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Create nnUNet dataset")
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("/workspace/data/segmentations"),
        help="Path to segmentation directory",
    )
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=Path("/workspace/data/series"),
        help="Path to series directory (for modality discovery)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. If not specified, set automatically by dataset ID",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        help="Dataset ID to generate (1=multi-class, 2=binary, 3=class grouping; more may be added)",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Max number of cases to process (for debugging)")

    args = parser.parse_args()

    dataset_config = get_dataset_config(args.dataset_id)

    # If output is not specified, use default path based on ID
    if args.output_dir is None:
        default_raw_root = Path("/workspace/data/nnUNet/nnUNet_raw")
        args.output_dir = default_raw_root / dataset_config["folder_name"]

    binary_status = "Enabled" if dataset_config["binary_vessel"] else "Disabled"
    remap_status = "Enabled" if dataset_config.get("label_mapping") else "Disabled"
    print(
        f"Using dataset ID {args.dataset_id} (binarization: {binary_status} / class remapping: {remap_status})"
    )
    print(f"Output directory: {args.output_dir}")

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "imagesTr"
    labels_dir = args.output_dir / "labelsTr"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Build UID list from segmentation directory (new structure: {uid}.nii)
    nii_files = sorted([f for f in args.segmentation_dir.glob("*.nii") if not f.name.endswith("_cowseg.nii")])
    uids = [f.stem for f in nii_files]
    if args.max_cases:
        uids = uids[: args.max_cases]

    print(f"Number of cases to process: {len(uids)}")

    # Process each case
    successful_cases = 0
    failed_uids = []
    modality_counts = {}
    processed_uids = []  # For mapping UID to filenames

    for uid in tqdm(uids, desc="Processing"):
        success, modality = process_case(
            uid=uid,
            segmentation_dir=args.segmentation_dir,
            output_images_dir=images_dir,
            output_labels_dir=labels_dir,
            series_base_dir=args.series_dir,
            binary_vessel=dataset_config["binary_vessel"],
            label_mapping=dataset_config.get("label_mapping"),
        )

        if success:
            successful_cases += 1
            # Count modality
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            # Record UID and filenames
            processed_uids.append({"uid": uid, "image": f"{uid}_0000.nii.gz", "label": f"{uid}.nii.gz"})
        else:
            failed_uids.append(uid)

    # Summary
    print(f"\nDone:")
    print(f"  Success: {successful_cases}/{len(uids)} cases")
    print(f"  Modality distribution: {modality_counts}")
    if failed_uids:
        print(f"  Failed: {len(failed_uids)} cases")
        print(f"    {failed_uids[:5]}{'...' if len(failed_uids) > 5 else ''}")

    # Create dataset.json
    if successful_cases > 0:
        labels = dataset_config["labels"]

        create_dataset_json(
            output_dir=args.output_dir,
            num_training=successful_cases,
            labels=labels,
            modality_counts=modality_counts,
        )

        # Create case mapping file
        mapping_file = args.output_dir / "case_mapping.json"
        mapping_data = {
            "total_cases": successful_cases,
            "files": processed_uids,
        }
        with open(mapping_file, "w") as f:
            json.dump(mapping_data, f, indent=2)

        print(f"\nCreated dataset at: {args.output_dir}")
        print(f"Case mapping file: {mapping_file}")
        print("\nNext steps:")
        print("1. Set nnUNet environment variables:")
        print("   export nnUNet_raw=/workspace/nnUNet_raw")
        print("   export nnUNet_preprocessed=/workspace/nnUNet_preprocessed")
        print("   export nnUNet_results=/workspace/nnUNet_results")
        print("2. Run preprocessing:")
        dataset_id_str = str(args.dataset_id)
        print(f"   nnUNetv2_plan_and_preprocess -d {dataset_id_str} --verify_dataset_integrity")
        print("3. Start training:")
        print(f"   nnUNetv2_train {dataset_id_str} 3d_fullres 0")


if __name__ == "__main__":
    main()
