#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained ROI model (Hydra experiment) by calling predict from
scripts/rsna_submission_roi.py on the training-time validation data, and
compute AUCs using labels from /workspace/data/train.csv.

Overview:
- Load training config from a Hydra experiment name and obtain per-fold
  validation UIDs from the saved CV split JSON.
- Detect available checkpoints (fold directories) and set ROI_FOLDS
  automatically.
- Run predict(series_path) on each SeriesInstanceUID DICOM series.
- Compute AUCs for 14 labels (13 locations + Aneurysm Present).
- Final score is (AP_AUC + mean location AUC)/2 in accordance with the task.

Example:
    python scripts/eval_val_with_predict.py \
        --experiment 250907-vessel_roi-nnunet_pretrained-s64_128-lr1e-4-bs1_8 \
        --folds auto \
        --limit 50

Notes:
- Inference is heavy (nnUNet vessel seg -> ROI classification). Mind runtime
  and GPU memory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm  # progress bar

# Root setup (enable local imports)
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import scripts.rsna_submission_roi as rsna_submission_roi  # Includes Kaggle-compatible predict()
from src.my_utils.kaggle_utils import load_experiment_config
from src.data.components.aneurysm_vessel_seg_dataset import ANEURYSM_CLASSES


def _load_cv_val_list(vessel_pred_dir: str, n_folds: int, split_seed: int, fold: int) -> List[str]:
    """Load validation UID list from the saved CV split JSON used during training.

    Args:
        vessel_pred_dir: Training-time `data.vessel_pred_dir`
        n_folds: Number of folds used during training
        split_seed: Split seed
        fold: Target fold index
    Returns:
        val_uids: List of SeriesInstanceUIDs for the validation split of the fold
    """
    # Split file is stored at the parent of vessel_pred_dir (per DataModule)
    split_path = Path(vessel_pred_dir).parent / f"cv_split_seg_{n_folds}fold_seed{split_seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"CV split file not found: {split_path}")

    with open(split_path, "r") as f:
        data = json.load(f)

    # Support both dict keys and list
    fold_splits = data.get("fold_splits")
    if fold_splits is None:
        raise ValueError(f"Invalid split file format: {split_path}")

    cur = (
        fold_splits[str(fold)]
        if isinstance(fold_splits, dict) and str(fold) in fold_splits
        else fold_splits[fold]
    )
    val_list: List[str] = list(cur["val"])  # 型明示
    return val_list


def _detect_available_folds(log_dir: Path) -> List[int]:
    """Search under checkpoints/fold*/ and return available folds."""
    ckpt_root = log_dir / "checkpoints"
    if not ckpt_root.exists():
        return []
    folds: List[int] = []
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.startswith("fold"):
            try:
                f = int(p.name.replace("fold", ""))
                folds.append(f)
            except Exception:
                continue
    return sorted(set(folds))


def _auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Safe AUC: return 0.5 when either class is missing."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_prob = np.asarray(y_prob).astype(np.float32)
    # 片側クラスのみの場合、sklearnは例外になるため0.5を返す
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


class ParticipantVisibleError(Exception):
    """Same error type as the official notebook (for display)."""

    pass


def _weighted_multilabel_auc(
    y_true: np.ndarray, y_scores: np.ndarray, class_weights: Optional[List[float]] = None
) -> float:
    """
    Compute weighted multi-label AUC consistent with the official notebook
    (mean-weighted-columnwise-aucroc.ipynb).

    Notes:
    - If any class lacks positive or negative samples, the official code raises
      an exception. This function mirrors that by raising ParticipantVisibleError.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]

    # クラス別AUC（sklearnへ委譲）
    try:
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
    except ValueError:
        # 公式実装と同様のメッセージで例外化
        raise ParticipantVisibleError("AUC could not be calculated from given predictions.")

    # 重み処理（正規化）
    if class_weights is None:
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)

    if len(weights_array) != n_classes:
        raise ValueError(
            f"Number of weights ({len(weights_array)}) must match number of classes ({n_classes})"
        )
    if np.any(weights_array < 0):
        raise ValueError("All class weights must be non-negative")
    if np.sum(weights_array) == 0:
        raise ValueError("At least one class weight must be positive")

    weights_array = weights_array / np.sum(weights_array)
    return float(np.sum(individual_aucs * weights_array))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    """Compute per-label AUCs and the final score (official weighted AUC).

    Args:
      y_true: (N, 14) binary array
      y_pred: (N, 14) predicted probabilities
      label_names: Label names (column order matches model output/CSV)
    Returns:
      metrics: {label_name: AUC, 'final_score': final weighted AUC}
    """
    metrics: Dict[str, float] = {}

    # Per-label AUCs (fallback to 0.5 for single-class cases)
    n_labels = y_true.shape[1]
    for i in range(n_labels):
        metrics[label_names[i]] = _auc_safe(y_true[:, i], y_pred[:, i])

    # Use official metric (weighted AUC) for final score
    # Weights: 13 locations=1, AP=13
    ap_idx = label_names.index("Aneurysm Present")
    weights = [1.0] * n_labels
    weights[ap_idx] = 13.0

    try:
        final_official = _weighted_multilabel_auc(y_true, y_pred, class_weights=weights)
        metrics["final_score"] = float(final_official)
    except ParticipantVisibleError:
        # If official requirements are not met, fallback to (AP + mean of 13 locations)/2
        loc_indices = [i for i in range(n_labels) if i != ap_idx]
        ap_auc = metrics[label_names[ap_idx]]
        loc_auc_mean = float(np.mean([metrics[label_names[i]] for i in loc_indices]))
        metrics["final_score"] = 0.5 * (ap_auc + loc_auc_mean)

    return metrics


def main() -> None:
    # Arguments
    parser = argparse.ArgumentParser(description="Validation inference + AUC evaluation")
    parser.add_argument(
        "--experiment",
        type=str,
        default="251011-seg_tf-v4-nnunet_truncate1-pretrained_1e-3_e30-ex_dav6w3-m32g64-w1_1_01",
        # default="250909-vessel_roi-nnunet_pretrained-s96_192-lr1e-4-bs1_8-e15",
        help="Hydra experiment name (configs/experiment/{name}.yaml)",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="auto",
        help='Folds to evaluate (e.g., "0,1,2" / "auto" to detect trained ones only)',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of series to evaluate per fold (for debugging)",
    )
    args = parser.parse_args()

    # Load experiment config (for log paths and data settings)
    cfg = load_experiment_config(args.experiment, config_dir="/workspace/configs")

    # Detect available folds or parse user-specified ones
    if args.folds == "auto":
        available_folds = _detect_available_folds(Path(cfg.log_dir))
    else:
        available_folds = [int(x) for x in args.folds.replace(" ", "").split(",") if x != ""]
    if len(available_folds) == 0:
        raise RuntimeError("No available folds found (no checkpoints detected)")

    # ROI_EXPERIMENT is fixed. Set ROI_FOLDS per fold and clear caches explicitly.
    os.environ["ROI_EXPERIMENTS"] = args.experiment
    # Local run: RUN_MODE=local (default)
    os.environ.setdefault("RUN_MODE", "local")

    os.environ["VESSEL_ABORT_ON_SPARSE_FAIL"] = "1"  # Abort instead of dense inference on sparse failure
    os.environ["VESSEL_ABORT_MIN_ALL_DIMS_MM"] = "140"
    os.environ["VESSEL_ABORT_ON_SMALL_ROI"] = "1"  # Abort when ROI volume is too small
    os.environ["VESSEL_MIN_ROI_VOXELS"] = "1000000"
    os.environ["VESSEL_ABORT_ON_LOW_UNION"] = "1"  # Abort when vessel union voxel count is too small
    os.environ["VESSEL_MIN_UNION_SUM"] = "2000"

    os.environ["RSNA_ERROR_FALLBACK_PROBS"] = (
        "0.05,0.07,0.07,0.13,0.07,0.06,0.02,0.01,0.01,0.07,0.02,0.01,0.028,0.55"
    )

    os.environ["VESSEL_NNUNET_SPARSE_MODEL_DIR"] = (
        "/workspace/logs/nnUNet_results/Dataset003_VesselGrouping/RSNA2025Trainer_moreDAv7__nnUNetResEncUNetMPlans__3d_fullres"
    )
    os.environ["VESSEL_ENABLE_ORIENTATION_CORRECTION"] = "0"
    os.environ["VESSEL_ORIENTATION_WEIGHTS"] = "1,1,1"
    os.environ["VESSEL_SPARSE_ROI_EXTENT_MM"] = "140"
    os.environ["VESSEL_REFINE_Z_ONLY"] = "0"

    os.environ["VESSEL_NNUNET_MODEL_DIR"] = (
        "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/nnUNetTrainerSkeletonRecall_more_DAv3__nnUNetResEncUNetMPlans__3d_fullres"
        # "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/nnUNetTrainerSkeletonRecall_more_DAv3_ep800__nnUNetResEncUNetMPlans__3d_fullres"
    )
    os.environ["VESSEL_ADDITIONAL_DENSE_MODEL_DIRS"] = (
        "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_SkeletonRecallW3TverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres,"
        # "/workspace/logs/nnUNet_results/Dataset001_VesselSegmentation/RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres"
    )
    os.environ["VESSEL_FOLDS"] = "all"

    os.environ["VESSEL_REFINE_MARGIN_Z"] = "15"
    os.environ["VESSEL_REFINE_MARGIN_XY"] = "30"

    os.environ["VESSEL_TRT_DIR"] = "/workspace/logs/trt"

    os.environ["ROI_TTA"] = "1"

    os.environ["RSNA_ONLY_AP"] = "0"
    os.environ["RSNA_ONLY_LOCATIONS"] = "0"
    os.environ["RSNA_DEBUG"] = "1"

    # Load label CSV (for GT)
    train_csv_path = Path(cfg.data.train_csv)
    df_train = pd.read_csv(train_csv_path)
    # Column order follows ANEURYSM_CLASSES (predict outputs the same order)
    label_cols = ANEURYSM_CLASSES
    df_train = df_train.set_index("SeriesInstanceUID")

    # For each fold, obtain validation list and evaluate
    all_metrics: Dict[str, Dict[str, float]] = {}
    # Count inference errors (exceptions) per fold
    fold_error_counts: Dict[int, int] = {}
    # Save UIDs that raised exceptions during inference
    fold_exception_uids: Dict[int, List[str]] = {}
    # Collect y_true/y_pred across all folds to report overall AUC
    global_y_true_list: List[np.ndarray] = []
    global_y_pred_list: List[np.ndarray] = []
    for f in available_folds:
        # Obtain validation UIDs from the training-time CV split
        val_list = _load_cv_val_list(
            vessel_pred_dir=cfg.data.vessel_pred_dir,
            n_folds=int(cfg.data.n_folds),
            split_seed=int(cfg.data.split_seed),
            fold=int(f),
        )

        # Restrict to this fold via env var and clear ROI model caches
        # Subsequent predict() calls will load only the fold-specific model
        os.environ["ROI_FOLDS"] = str(int(f))
        rsna_submission_roi.clear_caches(target="roi")

        # Map SeriesInstanceUIDs to DICOM series paths
        series_root = Path("/workspace/data/series")
        series_paths: List[Tuple[str, Path]] = []  # (uid, path)
        for uid in val_list:
            series_dir = series_root / uid
            if series_dir.exists() and series_dir.is_dir():
                series_paths.append((uid, series_dir))
        if len(series_paths) == 0:
            print(f"[WARN] fold{f}: No target series found. Skipping.")
            continue

        # Debug: limit number of cases
        if args.limit is not None and args.limit > 0:
            series_paths = series_paths[: int(args.limit)]

        # Run inference and collect results
        y_true_list: List[np.ndarray] = []
        y_pred_list: List[np.ndarray] = []

        print(f"=== fold{f}: running inference on {len(series_paths)} cases ===")
        # Error counter within the fold
        fold_err = 0

        for uid, sdir in tqdm(series_paths, desc=f"fold{f}", unit="case"):
            # Get GT labels (skip if missing)
            if uid not in df_train.index:
                continue
            y_true = df_train.loc[uid, label_cols].astype(int).to_numpy().astype(np.uint8)

            # Inference via Kaggle-compatible function (returns a Polars DataFrame, 1x14)
            try:
                df_pred = rsna_submission_roi.predict(str(sdir))
                # Polars -> Pandas -> NumPy (column order matches label_cols)
                y_pred = df_pred.to_pandas()[label_cols].iloc[0].to_numpy(dtype=np.float32)
            except Exception as e:
                # Count this case as an inference error
                fold_err += 1
                # Track exception UIDs
                fold_exception_uids.setdefault(f, []).append(uid)
                print(f"[WARN] Inference failed: {uid}: {e}")
                continue

            # predict() may return a constant vector (default 0.1) on error.
            # Count it as an error but keep it in evaluation per request.
            try:
                # Candidate constant (consider stage-diagnostic placeholder)
                placeholder_env = os.getenv("RSNA_STAGE_PLACEHOLDER", "")
                placeholder = float(placeholder_env) if placeholder_env != "" else 0.1
            except Exception:
                placeholder = 0.1

            is_all_const = bool(np.allclose(y_pred, placeholder, atol=1e-6)) or bool(
                np.allclose(y_pred, 0.1, atol=1e-6)
            )
            if is_all_const:
                # As requested, include constant-return cases in scoring while counting as errors.
                fold_err += 1
                print(f"[WARN] Detected constant prediction (counted as error, included in eval): {uid}")

            # Accumulate with consistent shapes
            y_true_list.append(y_true.reshape(1, -1))
            y_pred_list.append(y_pred.reshape(1, -1))

        if len(y_true_list) == 0:
            print(f"[WARN] fold{f}: No valid inference results. Skipping.")
            continue

        y_true_arr = np.concatenate(y_true_list, axis=0)
        y_pred_arr = np.concatenate(y_pred_list, axis=0)

        # Accumulate for overall aggregation
        global_y_true_list.append(y_true_arr)
        global_y_pred_list.append(y_pred_arr)

        # Compute AUCs
        metrics = _compute_metrics(y_true_arr, y_pred_arr, label_cols)
        all_metrics[f"fold{f}"] = metrics

        # Save per-fold error counts
        fold_error_counts[f] = fold_err

        # Console output
        ap_auc = metrics["Aneurysm Present"]
        loc_auc_mean = float(np.mean([metrics[c] for c in label_cols if c != "Aneurysm Present"]))
        print(
            f"fold{f}: cases={y_true_arr.shape[0]} AP_AUC={ap_auc:.4f} "
            f"LOC_MEAN={loc_auc_mean:.4f} FINAL={metrics['final_score']:.4f}"
        )

    # Aggregate across folds (optional display)
    if all_metrics:
        finals = [v["final_score"] for v in all_metrics.values()]
        print("=== Summary ===")
        for k, v in all_metrics.items():
            print(f"  {k}: final={v['final_score']:.4f}")
        print(f"  mean={np.mean(finals):.4f} std={np.std(finals):.4f}")

        # Extra 1: summary of inference error counts
        total_err = int(np.sum(list(fold_error_counts.values()))) if fold_error_counts else 0
        print("=== Error Summary ===")
        if fold_error_counts:
            by_fold_str = ", ".join([f"fold{f}={cnt}" for f, cnt in sorted(fold_error_counts.items())])
            print(f"  total_errors={total_err} ({by_fold_str})")
        else:
            print("  total_errors=0")

        # List UIDs that raised exceptions
        print("=== Exception UIDs ===")
        if fold_exception_uids:
            for fold_idx, uids in sorted(fold_exception_uids.items()):
                joined = ", ".join(uids)
                print(f"  fold{fold_idx}: {joined}")
        else:
            print("  No exceptions recorded")

        # Extra 2: per-class AUC details (overall across folds)
        try:
            overall_y_true = np.concatenate(global_y_true_list, axis=0)
            overall_y_pred = np.concatenate(global_y_pred_list, axis=0)
            overall_metrics = _compute_metrics(overall_y_true, overall_y_pred, label_cols)
            print("=== Per-class AUC (overall) ===")
            for name in label_cols:
                print(f"  {name}: {overall_metrics[name]:.4f}")
            print(f"  FINAL(overall): {overall_metrics['final_score']:.4f}")
        except Exception:
            # Fallback for impossible overall aggregation
            print("=== Per-class AUC (overall) ===")
            print("  Aggregation failed (insufficient data or exception)")
    else:
        print("No evaluation results.")


if __name__ == "__main__":
    main()
