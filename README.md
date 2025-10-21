# RSNA Intracranial Aneurysm Detection: 1st Place

## Environment
This solution was developed using the following Docker image:
 - gcr.io/kaggle-gpu-images/python:v161

All required Python packages can be found in `pip_packages/requirements.txt`.

### Hardware Used
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores / 32 threads, x86_64)
- CPU cores: 16 physical cores, 32 logical CPUs (2 threads/core)
- Memory: 256 GiB RAM
- GPU: 1 × NVIDIA GeForce RTX 4090 (24 GB)

## Dev Container
- Use the provided VS Code Dev Container for a fully reproducible setup.
- Requirements: Docker with NVIDIA Container Toolkit, VS Code with Dev Containers extension.
- Open this repository in VS Code and choose "Reopen in Container".
- The container image installs required tools (dcm2niix/GDCM, MONAI, TensorRT, etc.), performs `pip install -e ./nnUNet`, and exports `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.
- After the container is up, follow the commands below (Data → nnU‑Net Training → Vessel Segmentation Inference → ROI Classification → TensorRT Conversion).

## Data
```
# 0) Place competition data under /workspace/data
#    Required: train.csv, train_localizers.csv, series/, segmentations/ 

# 1) Convert DICOM to NIfTI (outputs to /workspace/data/series_niix)
python src/my_utils/rsna_dcm2niix.py

# 2) Move erroneous series listed in /workspace/data/error_data.yaml
python src/my_utils/move_error_data.py

# 3) Create nnU-Net training datasets (IDs 1 & 3)
python src/nnUnet_utils/create_nnunet_dataset.py
python src/nnUnet_utils/create_nnunet_dataset.py --dataset-id 3

# 4) nnU-Net v2 environment and preprocessing
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM
nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity -pl nnUNetPlannerResEncMForcedLowres -overwrite_target_spacing 1.0 1.0 1.0 -c 3d_fullres

# 4.1) In Dataset003 plans.json, set patch_size to [128, 128, 128] 
# 4.2) Re-preprocess Dataset003 after patch_size change
nnUNetv2_preprocess -d 3 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres

# 5) Create nnU-Net inference set (imagesTs for segmentation inference)
python src/nnUnet_utils/create_nnunet_inference_dataset.py
```

## nnU-Net Training
```
# Dataset 001 — VesselSegmentation (3d_fullres, 5 folds)
nnUNetv2_train 1 3d_fullres all -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecall_more_DAv3
nnUNetv2_train 1 3d_fullres all -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07
nnUNetv2_train 1 3d_fullres all -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07

# Dataset 003 — VesselGrouping (3d_fullres, 5 folds)
nnUNetv2_train 3 3d_fullres all -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv7
```

## Vessel Segmentation Inference (ROI Preprocessing)
```
python src/my_utils/vessel_segmentation.py 
```

## ROI Classification Training
```
# 5-fold cross-validation (edit experiment_list in src/train_cv.py if needed)
python src/train_cv.py

# Single-fold training/evaluation (example: fold 0)
python src/train.py \
  experiment=251013-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256 \
  data.fold=0
```

## TensorRT Conversion
```
# Local (nnU‑Net model -> ONNX + TensorRT)
python scripts/convert_nnunet_to_onnx_trt.py \
  --model-dir /workspace/logs/nnUNet_results/Dataset003_VesselGrouping/RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres \
  --fold all \
  --out-dir /workspace/logs/trt \
  --fp16 --device cuda

# Kaggle (nnU-Net models -> TensorRT)
# Open and run: /workspace/rsna2025-convert-trt.ipynb
# Engines saved under: /kaggle/working/trt/{TRAINER_PLANS_CONFIG}/fold_*/model_fp16.engine
#   e.g., RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres/fold_all/model_fp16.engine
```

## Acknowledgements
We extend our sincere gratitude to the creators of the following repositories and notebooks, whose outstanding work significantly contributed to our project:

  - [ashleve/lightning-hydra-template: PyTorch Lightning + Hydra. A very user-friendly template for ML experimentation. ⚡🔥⚡](https://github.com/ashleve/lightning-hydra-template)
  - [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
  - [MIC-DKFZ/Skeleton-Recall: Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures](https://github.com/MIC-DKFZ/Skeleton-Recall)
  - [Project-MONAI/MONAI: AI Toolkit for Healthcare Imaging](https://github.com/Project-MONAI/MONAI)
