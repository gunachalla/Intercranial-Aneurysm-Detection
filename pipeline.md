# Training & Inference Pipeline (RSNA 2025 Brain Aneurysm Detection)

Purpose:
- Extract vessel regions and ROI using vessel segmentation (nnUNet)
- Multi‑label classification of 13 locations + overall Aneurysm Present using the ROI image and location‑wise vessel masks
- Evaluation: 0.5 × (AUC of Aneurysm Present + mean AUC over 13 locations)

This note summarizes the steps required for production: preprocessing → segmentation inference → detection training → visualization, with the main scripts, settings, and example commands.

---

## 0. Data Directories
- Assumptions
  - DICOM: `/workspace/data/series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm`
  - Label CSV: `/workspace/data/train.csv`
  - Segmentations: `/workspace/data/segmentations/*.nii`
- Preprocessing outputs
  - NIfTI outputs: `/workspace/data/series_niix/{SeriesInstanceUID}/*.nii.gz`
  - nnUNet inference input: `/workspace/data/nnUNet_inference/imagesTs/*.nii.gz`
  - nnUNet inference outputs (saved by this project): `/workspace/data/nnUNet_inference/predictions/{SeriesInstanceUID}/`

---

## 1. DICOM → NIfTI Conversion (QC/Coordinate Consistency)

- Script: `/workspace/src/my_utils/rsna_dcm2niix.py`
- Overview:
  - Convert with dcm2niix. Pre‑audit mixed slice sizes and keep only the majority size for robustness
  - On failure, retry using `gdcmconv`
  - After conversion, enforce RAS orientation and generate QC summaries (`meta_summary.csv`, `suspects.csv`)
  - Convert point annotations in `train_localizers.csv` into NIfTI coordinates (outputs napari‑compatible JSON)
- Output: `/workspace/data/series_niix/{SeriesInstanceUID}/*.nii.gz` with accompanying JSON
- Example:
  - `python src/my_utils/rsna_dcm2niix.py --dicom-root /workspace/data/series --out-root /workspace/data/series_niix --tmp-root /workspace/tmp`
- Reference discussion: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/598083

---

## 2. Move Out Erroneous Data

- Script: `/workspace/src/my_utils/move_error_data.py`
- Overview: Move SeriesInstanceUIDs listed in `/workspace/data/error_data.yaml` out of `series_niix/`
- Output: `/workspace/data/series_niix_error_data_backup/`
- Example: `python src/my_utils/move_error_data.py`

---

## 3. nnUNet Data Preparation (Training/Inference)

- Create training sets: `/workspace/src/nnUnet_utils/create_nnunet_dataset.py`
  - Required: create training sets for dataset IDs `1` and `3` (specify via `-d/--dataset-id`)
  - Input: converted NIfTI
  - Output: `nnUNet_raw/Dataset001_VesselSegmentation/` and `nnUNet_raw/Dataset003_VesselGrouping/`
- Create inference set: `/workspace/src/nnUnet_utils/create_nnunet_inference_dataset.py`
  - Input: `/workspace/data/series_niix/`
  - Output: `/workspace/data/nnUNet_inference/imagesTs/{SeriesInstanceUID}_0000.nii.gz`
  - Example: `python src/nnUnet_utils/create_nnunet_inference_dataset.py --input-dir /workspace/data/series_niix --output-dir /workspace/data/nnUNet_inference/imagesTs`

---

## 4. nnUNet Vessel Segmentation Training (external)

- Representative commands (see `nnUNet_command.txt`)
  - `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM`
  - `nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecall_more_DAv3`
  - `nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity -pl nnUNetPlannerResEncMForcedLowres -overwrite_target_spacing 1.0 1.0 1.0 -c 3d_fullres`
  - Modify patch_size in the plan
    - `nnUNetv2_preprocess -d 3 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres`
    - `nnUNetv2_train 3 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr RSNA2025Trainer_moreDAv7`

- Example model output:
  - `/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv3_FocalTverskyPlusPlusWithBackground__nnUNetResEncUNetMPlans__3d_fullres`

---

## 5. nnUNet Inference + ROI Extraction (Adaptive Sparse Search)

- Script: `/workspace/src/my_utils/vessel_segmentation.py`
- Core: `nnUNet/nnunetv2/inference/adaptive_sparse_predictor.py`
  - For large volumes: coarse sparse search → single ROI selection → fine inference within ROI → (optional) recrop in two stages
  - Complete the preprocessing/interpolation/morphology on GPU for speed and memory efficiency
  - Suppress excessive ROI extent along the SI (Z) direction (mm cap) and add margins
- Output (per‑case subfolder): `/workspace/outputs/nnUNet_inference/predictions/{SeriesInstanceUID}/`
  - `seg.npz` or `seg.npy`: label map (0: background, 1..13: nnUNet output order, uint8)
  - `roi_data.npz`: preprocessed ROI image (C×D×H×W, fp16)
  - `transform.json`: coordinate transforms between original and network space, ROI metadata
  - `roi_annotations.json`: annotations that fell inside the ROI (if any)
  - Root file `annotations_outside_roi.json`: aggregated list of cases with annotations left outside the ROI
- Example execution:
  - `python src/my_utils/vessel_segmentation.py --model-path /workspace/logs/nnUNet_results/.../3d_fullres --test-dir /workspace/data/nnUNet_inference/imagesTs --output-dir /workspace/data/nnUNet_inference/predictions`

---

## 6. Aneurysm Detection Model (Multi‑label classification with ROI + region masks)

### Input Dataset

- Implementation: `/workspace/src/data/components/aneurysm_vessel_seg_dataset.py`
- Required inputs (per case):
  - `seg.npz` (or label map `seg.npy`) / `roi_data.npz` (ROI image) / `transform.json`
  - If `roi_annotations.json` is present, generate spherical GT
- Channel reordering:
  - Normalize nnUNet output (background + 13 locations) to detection label order (13 locations) using `DET_TO_SEG_OFFSET`
  - Additionally, create a 1‑channel union of all vessels from labels > 0 (binary 1‑ch)
- Extras:
  - `sphere_radius`: generate spherical GT around point annotations
- Transforms:
  - CPU: unify size with `Resize3DFitPadCropd`, light noise/contrast
  - GPU: affine/flip/low‑resolution simulation

### DataModule

- Implementation: `/workspace/src/data/rsna_aneurysm_vessel_seg_datamodule.py`
- Functions:
  - Collect usable cases from `train.csv` and `vessel_pred_dir`
  - Save/reuse K‑Fold splits (`cv_split_*.json`)

### Model (Backbone + Region Masked Pooling)

- LightningModule: `/workspace/src/models/aneurysm_vessel_seg_roi_module.py`
- Recommended backbone: `AneurysmRoiBackboneNnUNet`
  - Implementation: `/workspace/src/models/components/anet_roi_net.py`
  - Reuse features from a pretrained nnUNet encoder/last decoder stage
  - Outputs: `feat` (high‑resolution features), `logits_sphere` (auxiliary spherical task)
- Pooling: `RegionMaskedPooling3D` (`/workspace/src/models/components/region_mask_pooling.py`)
  - Concatenate region‑weighted average with GAP (per‑region vector: 2C dimensions)
- Classifier:
  - 13 locations: shared MLP → 1 output (applied per region)
  - Aneurysm Present: separate head over the union vessel mask
- Losses:
  - Locations/overall: BCEWithLogits
  - Spherical auxiliary: BalancedBCE + FocalTversky++ (only for positive samples)
- Validation metric (optimization target):
  - `final_score = 0.5 × (AUC_AP + mean(AUC over 13 locations))`

### Config Files and Example Commands

- Final experiment config: `/workspace/configs/experiment/251018-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256-test.yaml`
  - `model.net.nnunet_model_dir`: `/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres`
  - `data.vessel_pred_dir`: `/workspace/data/nnUNet_inference/predictions_v4_margin15_30`
  - `data.input_size`: `[128, 256, 256]`
- Single‑fold training/eval:
  - `python src/train.py experiment=251018-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256-test data.fold=0`
- 5‑fold CV:
  - `python src/train_cv.py` (set the experiment name above in `experiment_list`)

## 7. Submission

- Entry script: `/workspace/scripts/rsna_submission_roi.py`
- Kaggle inference calls `predict(series_path)`. Main behavior is controlled via environment variables.
- Typical submission settings:
  - `ROI_EXPERIMENTS`: `251018-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256-test`
  - `ROI_FOLDS`: `0,1,3,4`
  - `ROI_CKPT`: `last`
  - `VESSEL_NNUNET_MODEL_DIR`: `/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres`
  - `VESSEL_FOLDS`: `all`
  - `ROI_NNUNET_MODEL_DIR`: set this equal if overriding `cfg.model.net.nnunet_model_dir` on Kaggle
  - Optional (example of final production config): `VESSEL_REFINE_MARGIN_Z=15`, `VESSEL_REFINE_MARGIN_XY=30`, `VESSEL_ENABLE_ORIENTATION_CORRECTION=1`

- Notes:
  - The top‑level docstring of `rsna_submission_roi.py` documents all environment variables. Set `ROI_TTA` or adjust `ROI_CKPT` as needed.
  - Replace model paths with your own storage locations.
