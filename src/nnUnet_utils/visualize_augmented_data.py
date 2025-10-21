#!/usr/bin/env python3
"""
Script to visualize nnUNet data augmentations.
Use nnUNet trainers (e.g., nnUNetTrainer_onlyMirror01) to inspect training-time
augmentations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict
import argparse
import importlib
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from nnUNet.nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnUNet.nnunetv2.paths import nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import load_json, join


def visualize_batch(batch: Dict, batch_idx: int, save_dir: Path = None):
    """
    Visualize a batch (axial/sagittal/coronal views).

    Args:
        batch: Batch from the dataloader
        batch_idx: Batch index
        save_dir: Output directory (if None, only show)
    """
    # Get image and segmentation
    data = batch["data"]  # (batch_size, channels, d, h, w) or (batch_size, channels, h, w)
    # Prefer target[0] for segmentation; fallback to seg
    seg = None
    if "target" in batch and batch["target"] is not None:
        seg = batch["target"][0]  # (batch_size, 1, d, h, w) or (batch_size, 1, h, w)
    elif "seg" in batch:
        seg = batch["seg"]

    batch_size = data.shape[0]
    n_channels = data.shape[1]

    # 2D data
    if len(data.shape) == 4:  # 2D data
        # Limit number of samples (max 4)
        n_samples = min(batch_size, 4)
        # Visualize per channel (up to 3 channels)
        for ch_idx in range(min(n_channels, 3)):
            fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f"Batch {batch_idx} - Channel {ch_idx} (2D)", fontsize=16)

            for sample_idx in range(n_samples):
                # Image
                img = data[sample_idx, ch_idx].cpu().numpy()
                # Segmentation
                seg_img = None
                if seg is not None:
                    seg_img = seg[sample_idx, 0].cpu().numpy()

                # Stats
                img_min, img_max = np.min(img), np.max(img)
                img_mean, img_std = np.mean(img), np.std(img)

                # 1. Image (+ segmentation overlay)
                ax = axes[sample_idx, 0]
                im = ax.imshow(img, cmap="gray", vmin=img_min, vmax=img_max)
                if seg_img is not None:
                    # Overlay segmentation (with alpha)
                    mask = seg_img > 0
                    if np.any(mask):
                        ax.imshow(seg_img, cmap="tab20", alpha=0.3, interpolation="nearest")
                ax.set_title(f"Sample {sample_idx}: Image + Segmentation")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 2. Histogram
                ax = axes[sample_idx, 1]
                ax.hist(img.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black")
                ax.set_title(
                    f"Histogram\nmin={img_min:.2f}, max={img_max:.2f}\nmean={img_mean:.2f}, std={img_std:.2f}"
                )
                ax.set_xlabel("Intensity")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_dir is not None:
                save_path = save_dir / f"batch_{batch_idx:03d}_channel_{ch_idx}_2D.png"
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                print(f"  Saved image: {save_path}")

            plt.show()
            plt.close()
        return

    # 3D data
    if len(data.shape) == 5:  # 3D data
        # Center indices
        mid_d = data.shape[2] // 2
        mid_h = data.shape[3] // 2
        mid_w = data.shape[4] // 2

        # Limit number of samples (max 4)
        n_samples = min(batch_size, 4)

        # Visualize per channel (up to 3 channels)
        for ch_idx in range(min(n_channels, 3)):
            fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(f"Batch {batch_idx} - Channel {ch_idx} - 3 orthogonal views", fontsize=16)

            for sample_idx in range(n_samples):
                # 3D image
                img_3d = data[sample_idx, ch_idx].cpu().numpy()
                # 3D segmentation
                seg_3d = None
                if seg is not None:
                    seg_3d = seg[sample_idx, 0].cpu().numpy()

                # Extract three orthogonal slices
                # Axial (transverse): central slice along Z
                axial_img = img_3d[mid_d, :, :]
                axial_seg = seg_3d[mid_d, :, :] if seg_3d is not None else None

                # Sagittal: central slice along X
                sagittal_img = img_3d[:, :, mid_w]
                sagittal_seg = seg_3d[:, :, mid_w] if seg_3d is not None else None

                # Coronal: central slice along Y
                coronal_img = img_3d[:, mid_h, :]
                coronal_seg = seg_3d[:, mid_h, :] if seg_3d is not None else None

                # Stats
                img_min, img_max = np.min(img_3d), np.max(img_3d)
                img_mean, img_std = np.mean(img_3d), np.std(img_3d)

                views = [
                    (axial_img, axial_seg, "Axial"),
                    (sagittal_img, sagittal_seg, "Sagittal"),
                    (coronal_img, coronal_seg, "Coronal"),
                ]

                # 1-3. Orthogonal views
                for view_idx, (view_img, view_seg, view_name) in enumerate(views):
                    ax = axes[sample_idx, view_idx]
                    im = ax.imshow(view_img, cmap="gray", vmin=img_min, vmax=img_max)
                    # Overlay segmentation
                    if view_seg is not None:
                        mask = view_seg > 0
                        if np.any(mask):
                            ax.imshow(view_seg, cmap="tab20", alpha=0.3, interpolation="nearest")
                    ax.set_title(f"Sample {sample_idx}: {view_name}")
                    ax.axis("off")
                    if view_idx == 0:  # add colorbar to the first view only
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 4. Histogram
                ax = axes[sample_idx, 3]
                ax.hist(img_3d.flatten(), bins=50, alpha=0.7, color="blue", edgecolor="black")
                ax.set_title(
                    f"Histogram\nmin={img_min:.2f}, max={img_max:.2f}\nmean={img_mean:.2f}, std={img_std:.2f}"
                )
                ax.set_xlabel("Intensity")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_dir is not None:
                save_path = save_dir / f"batch_{batch_idx:03d}_channel_{ch_idx}_3D.png"
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                print(f"  Saved image: {save_path}")

            plt.show()
            plt.close()


def compare_augmentations(dataloader, n_batches: int = 5, save_dir: Path = None):
    """
    Compare multiple augmentation draws on the same data (orthogonal views).

    Args:
        dataloader: nnUNet dataloader
        n_batches: Number of batches to compare
        save_dir: Output directory
    """
    print(f"\n{'='*60}")
    print(f"Start augmentation comparison ({n_batches} batches)")
    print(f"{'='*60}\n")

    # Fetch the first batch multiple times
    iterator = iter(dataloader)

    # Retrieve the same index multiple times
    first_batches = []
    for _ in range(min(n_batches, 3)):
        # Reset the dataloader
        iterator = iter(dataloader)
        batch = next(iterator)
        first_batches.append(batch)

    # Visualization for comparison
    if len(first_batches) > 1:
        data_list = [b["data"] for b in first_batches]

        # 2D case
        if len(data_list[0].shape) == 4:
            data_2d_list = [d[0, 0, :, :].cpu().numpy() for d in data_list]
            # Comparison plot
            fig, axes = plt.subplots(1, len(data_2d_list), figsize=(5 * len(data_2d_list), 5))
            if len(data_2d_list) == 1:
                axes = [axes]

            fig.suptitle("Comparison of Different Augmentations on Same Data (2D)", fontsize=14)

            for idx, (ax, img) in enumerate(zip(axes, data_2d_list)):
                ax.imshow(img, cmap="gray")
                ax.set_title(f"Augmentation {idx+1}\nmin={np.min(img):.2f}, max={np.max(img):.2f}")
                ax.axis("off")

            plt.tight_layout()

            if save_dir is not None:
                save_path = save_dir / "augmentation_comparison_2D.png"
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                print(f"Saved comparison image: {save_path}")

            plt.show()
            plt.close()

        # 3D case
        elif len(data_list[0].shape) == 5:
            # Get center indices
            mid_d = data_list[0].shape[2] // 2
            mid_h = data_list[0].shape[3] // 2
            mid_w = data_list[0].shape[4] // 2

            # Extract orthogonal slices
            views = ["Axial", "Sagittal", "Coronal"]
            for view_idx, view_name in enumerate(views):
                fig, axes = plt.subplots(1, len(data_list), figsize=(5 * len(data_list), 5))
                if len(data_list) == 1:
                    axes = [axes]

                fig.suptitle(f"Comparison of Different Augmentations ({view_name} View)", fontsize=14)

                for aug_idx, data in enumerate(data_list):
                    img_3d = data[0, 0].cpu().numpy()
                    # Get slice for each view
                    if view_idx == 0:  # Axial
                        view_img = img_3d[mid_d, :, :]
                    elif view_idx == 1:  # Sagittal
                        view_img = img_3d[:, :, mid_w]
                    else:  # Coronal
                        view_img = img_3d[:, mid_h, :]

                    ax = axes[aug_idx]
                    ax.imshow(view_img, cmap="gray")
                    ax.set_title(
                        f"Augmentation {aug_idx+1}\n"
                        f"min={np.min(view_img):.2f}, max={np.max(view_img):.2f}"
                    )
                    ax.axis("off")

                plt.tight_layout()

                if save_dir is not None:
                    save_path = save_dir / f"augmentation_comparison_{view_name}.png"
                    plt.savefig(save_path, dpi=100, bbox_inches="tight")
                    print(f"Saved comparison image: {save_path}")

                plt.show()
                plt.close()


def get_trainer_class(trainer_class_name: str):
    """
    Resolve trainer class object by class name.

    Args:
        trainer_class_name: Trainer class name (e.g., "nnUNetTrainer", "nnUNetTrainer_onlyMirror01")

    Returns:
        Trainer class
    """
    # Common locations for trainer classes
    trainer_locations = [
        "nnUNet.nnunetv2.training.nnUNetTrainer.nnUNetTrainer",
        "nnUNet.nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring",
        "nnUNet.nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerBN",
        "nnUNet.nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision",
        "nnUNet.nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DA",
        "nnUNet.nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv2",
        "nnUNet.nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv3",
        "nnUNet.nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv4",
        "nnUNet.nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv5",
    ]

    for location in trainer_locations:
        try:
            module = importlib.import_module(location)
            if hasattr(module, trainer_class_name):
                trainer_class = getattr(module, trainer_class_name)
                print(f"Loaded trainer class {trainer_class_name} from {location}")
                return trainer_class
        except (ImportError, AttributeError):
            continue

    # Not found
    raise ValueError(
        f"Trainer class '{trainer_class_name}' not found.\nChecked locations: {trainer_locations}"
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize nnUNet data augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset_id", "-d", type=int, default=1, help="Dataset ID")

    parser.add_argument("--configuration", "-c", type=str, default="3d_fullres", help="nnUNet configuration name")

    parser.add_argument("--fold", "-f", type=int, default=0, help="Cross-validation fold index")

    parser.add_argument("--plans_name", "-p", type=str, default="nnUNetResEncUNetMPlans", help="Plans name")

    parser.add_argument(
        "--trainer_class_name",
        "-tr",
        type=str,
        default="RSNA2025Trainer_moreDAv5",
        help="Trainer class name",
    )

    parser.add_argument("--n_batches", "-n", type=int, default=5, help="Number of batches to visualize")

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/workspace/outputs/nnUNet_augmentation_visualization",
        help="Base path for output directory",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse CLI args
    args = parse_arguments()

    # Settings
    dataset_id = args.dataset_id
    configuration = args.configuration
    fold = args.fold
    plans_name = args.plans_name
    trainer_class_name = args.trainer_class_name
    n_batches_to_visualize = args.n_batches

    # Output directory
    output_root = Path(args.output_dir)
    output_dir = output_root / f"d{dataset_id}_{configuration}_{fold}_{plans_name}_{trainer_class_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Resolve trainer class
    try:
        trainer_class = get_trainer_class(trainer_class_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nExamples of trainer classes:")
        print("  - nnUNetTrainer")
        print("  - nnUNetTrainer_onlyMirror01")
        print("  - nnUNetTrainerNoMirroring")
        print("  - RSNA2025Trainer_moreDA")
        return

    # Dataset name
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    print(f"\nDataset: {dataset_name}")
    print(f"Configuration: {configuration}")
    print(f"Plans: {plans_name}")
    print(f"Trainer: {trainer_class_name}")

    # Path to plans file
    plans_file = join(nnUNet_preprocessed, dataset_name, f"{plans_name}.json")
    if not os.path.exists(plans_file):
        print(f"Error: plans file not found: {plans_file}")
        return

    # Path to dataset.json
    dataset_json_file = join(nnUNet_preprocessed, dataset_name, "dataset.json")
    if not os.path.exists(dataset_json_file):
        print(f"Error: dataset.json not found: {dataset_json_file}")
        return

    # Load plans and dataset info
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)

    print("\nPlans info:")
    print(f"  - Configurations: {list(plans['configurations'].keys())}")
    # print(f"  - Dataset fingerprint: {plans['dataset_fingerprint']}")

    # Initialize trainer
    print(f"\nInitializing {trainer_class_name} ...")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create trainer instance
    trainer = trainer_class(
        plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device
    )

    # Initialize
    trainer.initialize()

    print("\nTrainer settings:")
    print(f"  - Batch size: {trainer.batch_size}")
    print(f"  - Patch size: {trainer.configuration_manager.patch_size}")
    print(f"  - Mirroring axes: {trainer.inference_allowed_mirroring_axes}")

    # Dataloader setup
    print("\nSetting up dataloaders ...")
    trainer.num_iterations_per_epoch = 10  # set small for testing
    trainer.num_val_iterations_per_epoch = 5

    # Create dataloaders
    trainer._do_i_compile()  # compilation setup
    trainer.set_deep_supervision_enabled(True)  # deep supervision
    trainer.get_dataloaders()

    # Training dataloader
    train_loader, _ = trainer.get_dataloaders()

    print("\nDataloader info:")
    print(f"  - Batch size: {train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'N/A'}")
    print(f"  - Num workers: {train_loader.num_workers if hasattr(train_loader, 'num_workers') else 'N/A'}")

    # Visualize data augmentations
    print(f"\n{'='*60}")
    print("Start augmentation visualization")
    print(f"{'='*60}\n")

    # Fetch a few batches and visualize
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches_to_visualize:
            break

        print(f"\nProcessing batch {batch_idx + 1}/{n_batches_to_visualize} ...")
        print(f"  - Batch keys: {list(batch.keys())}")
        print(f"  - Data shape: {batch['data'].shape}")
        if "seg" in batch:
            print(f"  - Segmentation shape: {batch['seg'].shape}")
        if "target" in batch:
            print(f"  - Target shape: {batch['target'][0].shape}")

        # Visualize batch
        visualize_batch(batch, batch_idx, save_dir=output_dir)

    # Compare multiple augmentations for the same data
    compare_augmentations(train_loader, n_batches=3, save_dir=output_dir)

    print(f"\n{'='*60}")
    print("Augmentation visualization finished")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
