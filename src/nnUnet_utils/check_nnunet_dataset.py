#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic nnUNet dataset check script.
"""

import json
from pathlib import Path
import numpy as np
import nibabel as nib


def check_dataset_basic(dataset_dir: Path):
    """Check basic nnUNet dataset structure."""
    print(f"Dataset directory: {dataset_dir}")
    print(f"Exists: {dataset_dir.exists()}")
    
    # Check dataset.json
    json_path = dataset_dir / "dataset.json"
    if json_path.exists():
        with open(json_path) as f:
            dataset_info = json.load(f)
        print(f"Number of training cases: {dataset_info['numTraining']}")
        print(f"Labels: {list(dataset_info['labels'].keys())}")
    else:
        print("dataset.json not found")
        return
    
    # Check directories
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    
    print(f"imagesTr: {images_dir.exists()}")
    print(f"labelsTr: {labels_dir.exists()}")
    
    if images_dir.exists():
        image_files = list(images_dir.glob("*.nii.gz"))
        print(f"Number of image files: {len(image_files)}")
        
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.nii.gz"))
        print(f"Number of label files: {len(label_files)}")
    
    # Detailed check for a few files
    print("\n=== Detailed File Check ===")
    for i in range(1, min(4, len(image_files) + 1)):
        try:
            case_name = f"case_{i:03d}"
            img_path = images_dir / f"{case_name}_0000.nii.gz"
            label_path = labels_dir / f"{case_name}.nii.gz"
            
            if img_path.exists() and label_path.exists():
                img = nib.load(str(img_path))
                label = nib.load(str(label_path))
                
                img_data = img.get_fdata()
                label_data = label.get_fdata()
                
                print(f"\nCase {i}:")
                print(f"  Image file: {img_path.name}")
                print(f"  Image shape: {img_data.shape}")
                print(f"  Image intensity range: [{img_data.min():.1f}, {img_data.max():.1f}]")
                print(f"  voxel spacing: {img.header.get_zooms()}")
                print(f"  Label shape: {label_data.shape}")
                print(f"  Number of unique labels: {len(np.unique(label_data))}")
                print(f"  Label values: {sorted(np.unique(label_data))}")
                
                # Check shape consistency
                if img_data.shape != label_data.shape:
                    print(f"  Warning: image and label shapes do not match")
                else:
                    print(f"  OK: image and label shapes match")
                
            else:
                print(f"\nCase {i}: files not found")
                
        except Exception as e:
            print(f"\nCase {i}: error - {e}")


if __name__ == "__main__":
    dataset_dir = Path("/workspace/nnUNet_raw/Dataset001_VesselSegmentation")
    check_dataset_basic(dataset_dir)
