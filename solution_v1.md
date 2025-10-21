I thank RSNA, the organizers, the contributing radiologists and institutions, and the Kaggle team and community for hosting this impactful challenge. This write‑up summarizes the solution that placed 1st on the leaderboard. I focus on a robust, coarse‑to‑fine vessel‑segmentation and ROI‑based classification pipeline designed to generalize across modalities while producing location‑aware predictions.

*Note: This is an initial summary. I will add detailed information and code in a revised version to be released later.*

## Data Preparation
- I excluded about 60 series with various issues, including orientation or label anomalies, corrupted DICOM files, implausible interslice spacing, and other quality problems.
- I used multilabel‑stratified 5‑fold cross‑validation.

## Pipeline

### 1. DICOM → NIfTI
- Within each series, retain only the majority image configuration (`Rows×Cols`, `PixelSpacing`) and filter outliers in interslice spacing.
- Convert with dcm2niix. If it fails, first run `gdcmconv --raw`, then reconvert. [1]

### 2. nnUNet Segmentation + ROI Extraction (Coarse‑to‑Fine)
- Model setup (nnUNet v2 [2]):
  1) spacing = (1.0, 1.0, 1.0); vessels grouped into three classes (Posterior_Circulation_and_Basilar / Middle_Cerebral_Arteries / Other_Locations)
  2) spacing = (0.80, 0.45, 0.44); loss = Dice + CE + SkeletonRecall (weight=1) [3]
  3) spacing = (0.80, 0.45, 0.44); loss = Tversky + CE + SkeletonRecall (weight=3; recall‑oriented)
- Augmentation notes: disable left–right mirroring for Models 2 and 3; use stronger intensity/geometry augmentations; add transforms that mimic thicker slices.
- Inference (two stages):
  - Coarse scan: run Model 1 with sliding windows (overlap=0.2) to propose a single vessel ROI.
  - Fine inference: run Models 2 and 3 only inside the ROI (overlap=0.3).
- The fine‑stage segmentations are used downstream as masks.

### 3. ROI Classification (13 locations + Aneurysm Present)
- Backbone: reuse an nnUNet trained for segmentation; training is faster and accuracy is better than with timm 2.5D/3D baselines.
- Decoder simplification: remove the last decoder block (computationally heavy; no observed performance drop).
- Input size: 128×256×256.
- Auxiliary detection task: from decoder features, reconstruct a binary sphere (radius=5 px) centered on annotated aneurysm points to learn a detection signal.
- Per‑location features: mask decoder features with the vessel mask and pool to obtain per‑location embeddings.
- Fusion and classification: concatenate encoder GAP features with the masked features; model inter‑vessel relations with a lightweight transformer; classify the 13 locations with an MLP head (multi‑label).
- AP head: concatenate features pooled over the union vessel mask with encoder GAP features, then use a dedicated MLP to predict “Aneurysm Present”.
- Output design: each head is an binary classifier to mitigate the challenge of class imbalance.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F15217057%2F98e18ca93230a3f26bf2181021ce481a%2Fmodel_overview.png?generation=1760490733871010&alt=media)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F15217057%2F187c3d0327547a692d8e922dc5f1d5cd%2Fvessel_pooling.png?generation=1760490747217372&alt=media" width="80%">

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F15217057%2F0dd935cbc51a7d50bc69779c8f12d918%2Ftransformer.png?generation=1760490761936793&alt=media" width="60%">

### 4. Others
- Test‑time augmentations (TTA).
- Fail‑safe: if anomalies occur during segmentation or ROI extraction, I do not force an aneurysm prediction; I fall back to fixed output values.


## References
[1] https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/598083
[2] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
[3] Kirchhoff, Yannick, et al. "Skeleton recall loss for connectivity conserving and resource efficient segmentation of thin tubular structures." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.
