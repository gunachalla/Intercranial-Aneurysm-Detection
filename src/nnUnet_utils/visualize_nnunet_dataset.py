#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet dataset visualization script.

Utility to inspect images and labels in a generated nnUNet dataset.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from tqdm import tqdm

def load_dataset_info(dataset_dir: Path) -> Dict:
    """Load dataset information from dataset.json."""
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "r") as f:
        return json.load(f)

def load_case(case_id: int, images_dir: Path, labels_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load image and label for the specified case."""
    case_name = f"case_{case_id:03d}"
    
    # Load image
    img_path = images_dir / f"{case_name}_0000.nii.gz"
    img_nii = nib.load(str(img_path))
    img_data = img_nii.get_fdata()
    
    # Load label
    label_path = labels_dir / f"{case_name}.nii.gz"
    label_nii = nib.load(str(label_path))
    label_data = label_nii.get_fdata()
    
    print(f"Case {case_id}:")
    print(f"  Image shape: {img_data.shape}")
    print(f"  Label shape: {label_data.shape}")
    print(f"  Image intensity range: [{img_data.min():.2f}, {img_data.max():.2f}]")
    print(f"  Unique labels: {np.unique(label_data)}")
    print(f"  voxel spacing: {img_nii.header.get_zooms()}")
    
    return img_data, label_data

def visualize_case(
    img_data: np.ndarray, 
    label_data: np.ndarray, 
    case_id: int,
    output_dir: Optional[Path] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None
) -> None:
    """Overlay visualization of image and label."""
    
    # Determine slice indices (choose central slices)
    if slice_indices is None:
        z_idx = img_data.shape[0] // 2
        y_idx = img_data.shape[1] // 2  
        x_idx = img_data.shape[2] // 2
    else:
        z_idx, y_idx, x_idx = slice_indices
    
    # Create slices for three axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Case {case_id:03d} - Image and Label Overlay', fontsize=16)
    
    # Axial view (Z-axis)
    img_axial = img_data[z_idx, :, :]
    label_axial = label_data[z_idx, :, :]
    
    axes[0, 0].imshow(img_axial, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Axial Image (Z={z_idx})')
    axes[0, 0].axis('off')
    
    # Overlay label (background label 0 is transparent)
    label_axial_masked = np.ma.masked_where(label_axial == 0, label_axial)
    axes[1, 0].imshow(img_axial, cmap='gray', origin='lower')
    axes[1, 0].imshow(label_axial_masked, cmap='tab10', alpha=0.6, origin='lower')
    axes[1, 0].set_title(f'Axial Overlay (Z={z_idx})')
    axes[1, 0].axis('off')
    
    # Sagittal view (X-axis)
    img_sagittal = img_data[:, :, x_idx]
    label_sagittal = label_data[:, :, x_idx]
    
    axes[0, 1].imshow(img_sagittal, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Sagittal Image (X={x_idx})')
    axes[0, 1].axis('off')
    
    label_sagittal_masked = np.ma.masked_where(label_sagittal == 0, label_sagittal)
    axes[1, 1].imshow(img_sagittal, cmap='gray', origin='lower')
    axes[1, 1].imshow(label_sagittal_masked, cmap='tab10', alpha=0.6, origin='lower')
    axes[1, 1].set_title(f'Sagittal Overlay (X={x_idx})')
    axes[1, 1].axis('off')
    
    # Coronal view (Y-axis)
    img_coronal = img_data[:, y_idx, :]
    label_coronal = label_data[:, y_idx, :]
    
    axes[0, 2].imshow(img_coronal, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'Coronal Image (Y={y_idx})')
    axes[0, 2].axis('off')
    
    label_coronal_masked = np.ma.masked_where(label_coronal == 0, label_coronal)
    axes[1, 2].imshow(img_coronal, cmap='gray', origin='lower')
    axes[1, 2].imshow(label_coronal_masked, cmap='tab10', alpha=0.6, origin='lower')
    axes[1, 2].set_title(f'Coronal Overlay (Y={y_idx})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / f"case_{case_id:03d}_visualization.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {output_path}")
        plt.close()  # explicitly close to prevent memory leak
    else:
        plt.show()

def analyze_label_distribution(labels_dir: Path, dataset_info: Dict) -> None:
    """Analyze label distribution statistics."""
    print("\n=== Label Distribution Analysis ===")
    
    all_labels = []
    label_counts = {}
    
    # Inspect labels across all cases
    for label_file in sorted(labels_dir.glob("*.nii.gz")):
        label_nii = nib.load(str(label_file))
        label_data = label_nii.get_fdata()
        unique_labels = np.unique(label_data)
        all_labels.extend(unique_labels)
        
        # Count voxel numbers per label
        for label in unique_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += np.sum(label_data == label)
    
    # Show summary statistics
    unique_labels = sorted(set(all_labels))
    print(f"Detected labels: {unique_labels}")
    print(f"Expected labels: {list(dataset_info['labels'].values())}")
    
    print("\nVoxel counts per label:")
    for label in unique_labels:
        label_name = None
        for name, value in dataset_info['labels'].items():
            if value == int(label):
                label_name = name
                break
        print(f"  Label {int(label)} ({label_name}): {label_counts[label]:,} voxels")

def main():
    parser = argparse.ArgumentParser(description="Visualize nnUNet dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/workspace/nnUNet_raw/Dataset001_VesselSegmentation"),
        help="nnUNet dataset directory"
    )
    parser.add_argument(
        "--case-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Case IDs to visualize"
    )
    parser.add_argument(
        "--visualize-all",
        action="store_true",
        help="Visualize all cases (ignore --case-ids)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory to save visualizations"
    )
    parser.add_argument(
        "--analyze-labels",
        action="store_true",
        help="Run label distribution analysis"
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=None,
        help="Max number of visualizations (only with --visualize-all)"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.dataset_dir.exists():
        print(f"Error: dataset directory not found: {args.dataset_dir}")
        return
    
    images_dir = args.dataset_dir / "imagesTr"
    labels_dir = args.dataset_dir / "labelsTr"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: imagesTr or labelsTr directory not found")
        return
    
    # Create output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset info
    dataset_info = load_dataset_info(args.dataset_dir)
    print(f"Dataset: {args.dataset_dir.name}")
    print(f"Number of training cases: {dataset_info['numTraining']}")
    print(f"Channels: {dataset_info['channel_names']}")
    print(f"Labels: {dataset_info['labels']}")
    
    # Label distribution analysis
    if args.analyze_labels:
        analyze_label_distribution(labels_dir, dataset_info)
    
    # Decide which cases to visualize
    if args.visualize_all:
        # Collect all cases
        image_files = sorted(images_dir.glob("case_*_0000.nii.gz"))
        case_ids = []
        for img_file in image_files:
            # Extract 001 from case_001_0000.nii.gz
            case_num_str = img_file.stem.split('_')[1]
            case_ids.append(int(case_num_str))
        # Limit the number of cases
        if args.max_visualizations and len(case_ids) > args.max_visualizations:
            case_ids = case_ids[:args.max_visualizations]
            print(f"Visualizing the first {len(case_ids)} cases")
        else:
            print(f"Visualizing all {len(case_ids)} cases")
        
        # If output directory is not provided, set default
        if args.output_dir is None:
            args.output_dir = args.dataset_dir / "visualizations"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving visualizations to: {args.output_dir}")
    else:
        case_ids = args.case_ids
    
    # Visualize cases
    print(f"\n=== Case Visualization ===")
    successful_visualizations = 0
    failed_cases = []
    
    # Progress bar
    case_iterator = tqdm(case_ids, desc="Visualizing") if args.visualize_all else case_ids
    
    for case_id in case_iterator:
        try:
            img_data, label_data = load_case(case_id, images_dir, labels_dir)
            visualize_case(img_data, label_data, case_id, args.output_dir)
            successful_visualizations += 1
            if not args.visualize_all:  # Suppress detailed output when visualizing all
                print()
        except FileNotFoundError as e:
            if not args.visualize_all:
                print(f"Case {case_id} not found: {e}")
            failed_cases.append(case_id)
        except Exception as e:
            if not args.visualize_all:
                print(f"Error while processing case {case_id}: {e}")
            failed_cases.append(case_id)
    
    # Summary when visualizing all
    if args.visualize_all:
        print(f"\n=== Visualization Summary ===")
        print(f"Success: {successful_visualizations}/{len(case_ids)} cases")
        if failed_cases:
            print(f"Failed: {len(failed_cases)} cases ({failed_cases[:5]}{'...' if len(failed_cases) > 5 else ''})")
        if args.output_dir:
            print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
