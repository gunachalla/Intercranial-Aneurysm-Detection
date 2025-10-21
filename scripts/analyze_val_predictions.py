"""
Validation error analysis using preprocessed ROIs.

- Does not reprocess from DICOM. Uses the preprocessed ROI artifacts handled by
  `src/data/rsna_aneurysm_vessel_seg_datamodule.py` (`prob.npz`, `roi_data.npz`,
  `transform.json`, and optionally `roi_annotations.json`).
- For GT localization, use spheres/ann_points derived from preprocessed data
  (`roi_annotations.json`) instead of `train_localizers.csv` when available.
- Checkpoints are loaded from `/workspace/logs/train/runs/{experiment}/checkpoints/fold{fold}/`.
- Threshold optimization uses Youden's J (maximize TPR - FPR).

Outputs (example):
- /workspace/logs/train/runs/{experiment}/analysis/fold{fold}/tables/oof_val_predictions.parquet:
  14 prediction columns + 14 GT columns + meta
- In the same folder: errors_fp.csv, errors_fn.csv (FP/FN lists per label)
- In the same folder: thresholds.json (per-label Youden thresholds)
- In the same folder: metrics_summary.json (AUC/AP/weighted averages)

Usage:
python scripts/analyze_val_predictions.py \
  --experiment 250909-vessel_roi-nnunet_pretrained-s96_192-lr1e-4-bs1_8-e15 \
  --fold 0

Notes:
- CPU cannot use float16 arithmetic; on CPU, tensors are converted to float32.
"""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    multilabel_confusion_matrix,
)
from sklearn.calibration import calibration_curve
import matplotlib

matplotlib.use("Agg")  # For saving in non-GUI environments
import matplotlib.pyplot as plt

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Use data modules/models
from hydra.utils import instantiate

from src.data.rsna_aneurysm_vessel_seg_datamodule import RSNAAneurysmVesselSegDataModule
from src.data.components.aneurysm_vessel_seg_dataset import (
    ANEURYSM_CLASSES,
    AneurysmVesselSegDataset,
    get_val_transforms,
    normalize_extra_seg_suffixes,
)
from src.models.aneurysm_vessel_seg_roi_module import AneurysmVesselSegROILitModule
from src.my_utils.kaggle_utils import load_experiment_config


# =============== Utilities ===============


def find_fold_checkpoint(log_root: Path, experiment: str, fold: int, prefer: str = "epoch") -> Path:
    """Resolve and return the checkpoint path.

    Args:
        log_root: Root logs directory (/workspace/logs/train/runs)
        experiment: Experiment directory name
        fold: Fold index
        prefer: "epoch" | "last". If multiple epoch_* exist, choose the largest epoch.
    """
    exp_dir = log_root / experiment / "checkpoints" / f"fold{fold}"
    if not exp_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {exp_dir}")

    # Prefer epoch_*.ckpt
    epoch_ckpts = sorted(exp_dir.glob("epoch_*.ckpt"))
    if prefer == "epoch" and len(epoch_ckpts) > 0:
        # Choose the file with the highest epoch number
        def _epoch_num(p: Path) -> int:
            try:
                stem = p.stem  # epoch_014
                return int(stem.split("_")[1])
            except Exception:
                return -1

        best = sorted(epoch_ckpts, key=_epoch_num)[-1]
        return best

    # Fallback: last.ckpt
    last = exp_dir / "last.ckpt"
    if last.exists():
        return last

    # Otherwise, return the first available file
    any_ckpt = list(exp_dir.glob("*.ckpt"))
    if any_ckpt:
        return any_ckpt[0]

    raise FileNotFoundError(f"No checkpoint found: {exp_dir}")


def to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move batch tensors to device with appropriate dtypes.

    - Use float32 on CPU and float16 on CUDA to match training setup.
    - Keep uint8/long dtypes where required for loss computations.
    """
    fp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        if k in ("image", "vessel_seg", "vessel_union"):
            out[k] = v.to(device=device, dtype=fp_dtype, non_blocking=True)
        elif k in ("labels",):
            out[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)
        elif k in ("vessel_label",):
            out[k] = v.to(device=device, dtype=torch.long, non_blocking=True)
        elif k in ("sphere_mask",):
            out[k] = v.to(device=device, dtype=torch.uint8, non_blocking=True)
        elif k in ("selected_loc_idx",):
            out[k] = v.to(device=device, dtype=torch.long, non_blocking=True)
        elif k in ("ann_points",):
            out[k] = v.to(device=device, dtype=torch.float32, non_blocking=True)
        elif k in ("ann_points_valid",):
            out[k] = v.to(device=device, dtype=torch.uint8, non_blocking=True)
        else:
            # Leave others as-is
            out[k] = v.to(device=device, non_blocking=True)
    return out


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC that returns 0.5 when only one class is present."""
    try:
        if np.unique(y_true).size < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the threshold maximizing Youden's J statistic.
    If ROC cannot be computed (e.g., single-class), return 0.5.
    """
    try:
        if np.unique(y_true).size < 2:
            return 0.5
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        i = int(np.argmax(j))
        return float(thr[i])
    except Exception:
        return 0.5


def compute_competition_score(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Final score: 0.5 * (AUC_AP + mean AUC over 13 locations)."""
    assert y_true.shape[1] >= 14 and y_prob.shape[1] >= 14
    loc_idx = list(range(13))
    ap_idx = 13
    auc_loc = [safe_auc(y_true[:, i], y_prob[:, i]) for i in loc_idx]
    auc_ap = safe_auc(y_true[:, ap_idx], y_prob[:, ap_idx])
    mean_loc = float(np.mean(auc_loc)) if len(auc_loc) > 0 else 0.5
    final = 0.5 * (auc_ap + mean_loc)
    out = {"final_score": final, "auc_ap": auc_ap, "auc_loc_mean": mean_loc}
    for i, a in enumerate(auc_loc):
        out[f"auc_loc_{i}"] = a
    return out


# =============== Prediction & Aggregation Logic ===============


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=torch.cuda.is_available())
def run_val_prediction(
    experiment: str,
    fold: int,
    outdir: Path,
    prefer_ckpt: str = "epoch",
    batch_size_override: int | None = None,
) -> Dict[str, Path]:
    """Run validation inference for the given experiment/fold and save outputs.

    Returns: A dictionary of generated output paths.
    """
    out_tables = outdir / "tables"
    out_plots = outdir / "plots"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    # 1) Resolve checkpoint
    log_root = Path("/workspace/logs/train/runs")
    ckpt_path = find_fold_checkpoint(log_root, experiment, fold, prefer=prefer_ckpt)

    # 2) Load Hydra config and instantiate components
    cfg = load_experiment_config(experiment)
    cfg.data.fold = int(fold)
    if batch_size_override is not None:
        cfg.data.batch_size = int(batch_size_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm: RSNAAneurysmVesselSegDataModule = instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("validate")
    val_loader = dm.val_dataloader()

    data_cfg = cfg.data
    model_cfg = cfg.model

    model: AneurysmVesselSegROILitModule = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if getattr(cfg.model, "compile", False):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    load_res = model.load_state_dict(state_dict, strict=False)
    if getattr(load_res, "missing_keys", []):
        print(f"[analyze_val_predictions] Warning: Keys missing in checkpoint: {load_res.missing_keys}")
    if getattr(load_res, "unexpected_keys", []):
        print(f"[analyze_val_predictions] Warning: Unexpected keys for model: {load_res.unexpected_keys}")
    model.eval()
    model.to(device)
    forward_params = inspect.signature(model.forward).parameters
    requires_masks = "vessel_seg" in forward_params

    # 5) Inference loop
    rows: List[Dict] = []
    for batch in val_loader:
        # Preprocess similar to Lightning's validation_step (no aug, re-generate spheres from points).
        # _gpu_augment_batch keeps half internally, so convert to float32 on CPU.
        batch = to_device_batch(batch, device)

        # Batch processing including sphere re-generation
        batch_proc = model._gpu_augment_batch(batch, training=False)

        # Build kwargs based on forward signature (support extra seg/meta)
        fwd_kwargs = {}
        if "vessel_seg" in forward_params and "vessel_seg" in batch_proc:
            fwd_kwargs["vessel_seg"] = batch_proc["vessel_seg"]
        if "vessel_union" in forward_params and "vessel_union" in batch_proc:
            fwd_kwargs["vessel_union"] = batch_proc["vessel_union"]
        if "extra_vessel_seg" in forward_params and "extra_vessel_seg" in batch_proc:
            fwd_kwargs["extra_vessel_seg"] = batch_proc.get("extra_vessel_seg")
        # Embed metadata if used during training so that inference matches
        if "metadata_numeric" in forward_params and "metadata_numeric" in batch_proc:
            fwd_kwargs["metadata_numeric"] = batch_proc.get("metadata_numeric")
        if "metadata_numeric_missing" in forward_params and "metadata_numeric_missing" in batch_proc:
            fwd_kwargs["metadata_numeric_missing"] = batch_proc.get("metadata_numeric_missing")
        if "metadata_categorical" in forward_params and "metadata_categorical" in batch_proc:
            fwd_kwargs["metadata_categorical"] = batch_proc.get("metadata_categorical")

        out = model.forward(batch_proc["image"], **fwd_kwargs)  # dict(feat, logits_sphere)
        loss = model._compute_losses(batch_proc, out)

        logits_loc = loss["logits_loc"]  # (B,13)
        logit_ap = loss["logit_ap"].unsqueeze(1)  # (B,1)
        logits_all = torch.cat([logits_loc, logit_ap], dim=1)
        probs_all = torch.sigmoid(logits_all).detach().cpu().numpy()

        gt = batch["labels"].detach().cpu().numpy()
        series_list = batch["series_uid"]  # List[str]

        # Some meta (transform info from preprocessing and annotation counts)
        metas = batch.get("meta", None)
        ann_counts = [int(m.get("roi_annotations_count", 0)) if isinstance(m, dict) else 0 for m in metas]

        # Extract per-image meta (e.g., spacing)
        spc_z = []
        spc_y = []
        spc_x = []
        for m in metas:
            if isinstance(m, dict) and isinstance(m.get("transform_info"), dict):
                sp = m["transform_info"].get("spacing_original") or m["transform_info"].get(
                    "spacing_after_resampling"
                )
                if isinstance(sp, (list, tuple)) and len(sp) >= 3:
                    spc_z.append(float(sp[0]))
                    spc_y.append(float(sp[1]))
                    spc_x.append(float(sp[2]))
                else:
                    spc_z.append(np.nan)
                    spc_y.append(np.nan)
                    spc_x.append(np.nan)
            else:
                spc_z.append(np.nan)
                spc_y.append(np.nan)
                spc_x.append(np.nan)

        for i in range(probs_all.shape[0]):
            row = {"SeriesInstanceUID": series_list[i]}
            # Predicted probabilities
            for j, name in enumerate(ANEURYSM_CLASSES):
                row[f"prob_{name}"] = float(probs_all[i, j])
                row[f"gt_{name}"] = int(gt[i, j])
            row["ann_points_count"] = ann_counts[i]
            row["spacing_z"] = spc_z[i]
            row["spacing_y"] = spc_y[i]
            row["spacing_x"] = spc_x[i]
            rows.append(row)

    pred_df = pd.DataFrame(rows)

    # Additional meta: bring Modality/Age/Sex from train.csv
    train_csv = Path(data_cfg.get("train_csv", "/workspace/data/train.csv"))
    if train_csv.exists():
        gt_df = pd.read_csv(train_csv)
        keep_cols = [
            "SeriesInstanceUID",
            "Modality",
            "PatientAge",
            "PatientSex",
        ]
        gt_meta = gt_df[keep_cols].drop_duplicates()
        pred_df = pred_df.merge(gt_meta, on="SeriesInstanceUID", how="left")

    # Save outputs
    oof_path = out_tables / "oof_val_predictions.parquet"
    pred_df.to_parquet(oof_path, index=False)

    # 5) Metrics and Youden thresholds
    y_true = np.stack([pred_df[f"gt_{n}"].values for n in ANEURYSM_CLASSES], axis=1)
    y_prob = np.stack([pred_df[f"prob_{n}"].values for n in ANEURYSM_CLASSES], axis=1)

    metrics = compute_competition_score(y_true, y_prob)
    metrics_path = out_tables / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    thresholds: Dict[str, float] = {}
    for j, name in enumerate(ANEURYSM_CLASSES):
        thr = youden_threshold(y_true[:, j], y_prob[:, j])
        thresholds[name] = thr
    thr_path = out_tables / "thresholds.json"
    with open(thr_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # Compute confusion matrices for the 13 location classes
    loc_names = ANEURYSM_CLASSES[:13]
    loc_thr = np.array([thresholds[name] for name in loc_names], dtype=float)
    y_true_loc = y_true[:, : len(loc_names)]
    y_pred_loc = (y_prob[:, : len(loc_names)] >= loc_thr).astype(int)
    cm = multilabel_confusion_matrix(y_true_loc, y_pred_loc)
    cm_rows = []
    for idx, name in enumerate(loc_names):
        tn, fp, fn, tp = cm[idx].ravel()
        cm_rows.append({"label": name, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    cm_df = pd.DataFrame(cm_rows)
    cm_path = out_tables / "confusion_matrix_locations.csv"
    cm_df.to_csv(cm_path, index=False)

    # Aggregate co-occurrence patterns where a true label is detected as another label
    mis_counts = np.zeros((len(loc_names), len(loc_names)), dtype=np.int64)
    fn_totals = np.zeros(len(loc_names), dtype=np.int64)
    for sample_idx in range(y_true_loc.shape[0]):
        gt_pos = np.where(y_true_loc[sample_idx] == 1)[0]
        pred_pos = np.where(y_pred_loc[sample_idx] == 1)[0]
        for gt_idx in gt_pos:
            if y_pred_loc[sample_idx, gt_idx] == 1:
                continue
            fn_totals[gt_idx] += 1
            mis_pred = [p for p in pred_pos if p != gt_idx and y_true_loc[sample_idx, p] == 0]
            for pred_idx in mis_pred:
                mis_counts[gt_idx, pred_idx] += 1
    # Render heatmap for misdetection patterns
    mis_plot_path = out_plots / "misclassification_heatmap.png"
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_matrix = np.where(fn_totals[:, None] > 0, mis_counts / fn_totals[:, None], 0.0)
    max_rate = float(np.nanmax(rate_matrix)) if rate_matrix.size > 0 else 0.0
    vmax = max_rate if max_rate > 0 else 1.0
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(rate_matrix, cmap="OrRd", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(loc_names)))
    ax.set_yticks(range(len(loc_names)))
    ax.set_xticklabels(loc_names, rotation=45, ha="right")
    ax.set_yticklabels(loc_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground truth label")
    ax.set_title("Misclassification heatmap (rate per FN)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rate per FN")
    for i in range(len(loc_names)):
        for j in range(len(loc_names)):
            cnt = int(mis_counts[i, j])
            if cnt <= 0:
                continue
            if fn_totals[i] > 0:
                rate_percent = rate_matrix[i, j] * 100.0
                text = f"{cnt}\n({rate_percent:.1f}%)"
            else:
                text = f"{cnt}"
            text_color = "white" if rate_matrix[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9)
    plt.tight_layout()
    plt.savefig(mis_plot_path, dpi=150)
    plt.close(fig)

    # 6) Extract misclassifications (FP/FN)
    err_rows_fp: List[Dict] = []
    err_rows_fn: List[Dict] = []
    for idx, r in pred_df.iterrows():
        for name in ANEURYSM_CLASSES:
            p = float(r[f"prob_{name}"])
            g = int(r[f"gt_{name}"])
            pred = int(p >= thresholds[name])
            if pred == 1 and g == 0:
                err_rows_fp.append(
                    {
                        "SeriesInstanceUID": r["SeriesInstanceUID"],
                        "label": name,
                        "prob": p,
                        "gt": g,
                        "Modality": r.get("Modality", None),
                        "PatientAge": r.get("PatientAge", None),
                        "PatientSex": r.get("PatientSex", None),
                        "ann_points_count": r.get("ann_points_count", 0),
                    }
                )
            elif pred == 0 and g == 1:
                err_rows_fn.append(
                    {
                        "SeriesInstanceUID": r["SeriesInstanceUID"],
                        "label": name,
                        "prob": p,
                        "gt": g,
                        "Modality": r.get("Modality", None),
                        "PatientAge": r.get("PatientAge", None),
                        "PatientSex": r.get("PatientSex", None),
                        "ann_points_count": r.get("ann_points_count", 0),
                    }
                )

    fp_df = pd.DataFrame(err_rows_fp).sort_values(["label", "prob"], ascending=[True, False])
    fn_df = pd.DataFrame(err_rows_fn).sort_values(["label", "prob"], ascending=[True, True])
    fp_path = out_tables / "errors_fp.csv"
    fn_path = out_tables / "errors_fn.csv"
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)

    # 7) Slice analysis (simple): AUC by Modality/Age bins/spacing_z (Present/AP only)
    try:
        # By modality
        rows_mod = []
        for mod, g in pred_df.groupby("Modality"):
            y_t = g[f"gt_{ANEURYSM_CLASSES[13]}"]
            y_p = g[f"prob_{ANEURYSM_CLASSES[13]}"]
            rows_mod.append({"Modality": mod, "AUC_AP": safe_auc(y_t.values, y_p.values), "n": int(len(g))})
        pd.DataFrame(rows_mod).to_csv(out_tables / "slice_auc_modality.csv", index=False)

        # By age bins
        if pred_df["PatientAge"].notna().any():
            pred_df["AgeBin"] = pd.cut(
                pred_df["PatientAge"].astype(float), bins=[0, 40, 60, 80, 200], right=False
            )
            rows_age = []
            for ageb, g in pred_df.groupby("AgeBin"):
                y_t = g[f"gt_{ANEURYSM_CLASSES[13]}"]
                y_p = g[f"prob_{ANEURYSM_CLASSES[13]}"]
                rows_age.append(
                    {"AgeBin": str(ageb), "AUC_AP": safe_auc(y_t.values, y_p.values), "n": int(len(g))}
                )
            pd.DataFrame(rows_age).to_csv(out_tables / "slice_auc_agebin.csv", index=False)

        # By spacing_z bins (mm)
        if pred_df["spacing_z"].notna().any():
            pred_df["SpcZBin"] = pd.cut(
                pred_df["spacing_z"].astype(float), bins=[0, 0.6, 0.8, 1.2, 5.0], right=False
            )
            rows_spc = []
            for zb, g in pred_df.groupby("SpcZBin"):
                y_t = g[f"gt_{ANEURYSM_CLASSES[13]}"]
                y_p = g[f"prob_{ANEURYSM_CLASSES[13]}"]
                rows_spc.append(
                    {"SpcZBin": str(zb), "AUC_AP": safe_auc(y_t.values, y_p.values), "n": int(len(g))}
                )
            pd.DataFrame(rows_spc).to_csv(out_tables / "slice_auc_spcz.csv", index=False)
    except Exception:
        pass

    return {
        "oof": oof_path,
        "metrics": metrics_path,
        "thresholds": thr_path,
        "confusion_matrix": cm_path,
        "misclassification_plot": mis_plot_path,
        "errors_fp": fp_path,
        "errors_fn": fn_path,
    }


# =============== Visualization ===============


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_roc_pr_calibration(pred_df: pd.DataFrame, outdir: Path) -> None:
    """Save ROC/PR/Calibration plots (AP and ROC grid for all labels)."""
    plots_dir = outdir / "plots"
    _ensure_dir(plots_dir)

    # y_true/y_score arrays
    y_true = np.stack([pred_df[f"gt_{n}"].values for n in ANEURYSM_CLASSES], axis=1)
    y_prob = np.stack([pred_df[f"prob_{n}"].values for n in ANEURYSM_CLASSES], axis=1)

    # 1) ROC/PR/Calibration for AP
    ap_idx = 13
    y_t = y_true[:, ap_idx]
    y_p = y_prob[:, ap_idx]

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_ap = safe_auc(y_t, y_p)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc_ap:.3f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC - Aneurysm Present")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_ap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # PR
    try:
        prec, rec, _ = precision_recall_curve(y_t, y_p)
        ap = average_precision_score(y_t, y_p)
        plt.figure(figsize=(5, 5))
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR - Aneurysm Present")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(plots_dir / "pr_ap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Calibration
    try:
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=10, strategy="uniform")
        plt.figure(figsize=(5, 5))
        plt.plot(prob_pred, prob_true, marker="o", label="Aneurysm Present")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("Predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration - Aneurysm Present")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(plots_dir / "calibration_ap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # 2) ROC grid for all labels
    try:
        n = len(ANEURYSM_CLASSES)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.ravel()
        for i, name in enumerate(ANEURYSM_CLASSES):
            ax = axes[i]
            yi = y_true[:, i]
            pi = y_prob[:, i]
            if np.unique(yi).size >= 2:
                fpr, tpr, _ = roc_curve(yi, pi)
                auc_i = safe_auc(yi, pi)
                ax.plot(fpr, tpr, label=f"AUC={auc_i:.2f}")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                ax.legend(loc="lower right", fontsize=8)
            ax.set_title(name, fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        # Hide empty subplots
        for j in range(i + 1, rows * cols):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_grid.png", dpi=150)
        plt.close()
    except Exception:
        pass


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=torch.cuda.is_available())
def render_error_thumbnails(
    experiment: str,
    fold: int,
    pred_df: pd.DataFrame,
    thresholds: Dict[str, float],
    outdir: Path,
    n_examples_per_label: int = 8,
) -> None:
    """
    Generate Z-MIP thumbnails for error cases (FP/FN).
    - Background: ROI image
    - Overlays: vessel union, GT spheres, predicted spheres (from model forward)
    """
    qual_dir = outdir / "qualitative"
    _ensure_dir(qual_dir)

    # Load error lists
    tables_dir = outdir / "tables"
    fp_path = tables_dir / "errors_fp.csv"
    fn_path = tables_dir / "errors_fn.csv"
    if not fp_path.exists() or not fn_path.exists():
        return
    fp_df = pd.read_csv(fp_path)
    fn_df = pd.read_csv(fn_path)

    # Select labels: AP + top frequent error labels
    labels = ["Aneurysm Present"]
    all_err = pd.concat([fp_df, fn_df], ignore_index=True)
    if not all_err.empty:
        top = all_err[all_err["label"] != "Aneurysm Present"]["label"].value_counts().head(3).index.tolist()
        labels.extend(top)
    labels = list(dict.fromkeys(labels))  # Remove duplicates

    # Collect target UIDs
    sel_uids: List[str] = []
    per_label_cases: Dict[Tuple[str, str], List[str]] = {}  # (label, type=fp/fn) -> uids
    for lab in labels:
        fp_lab = fp_df[fp_df["label"] == lab].sort_values("prob", ascending=False).head(n_examples_per_label)
        fn_lab = fn_df[fn_df["label"] == lab].sort_values("prob", ascending=True).head(n_examples_per_label)
        u_fp = fp_lab["SeriesInstanceUID"].tolist()
        u_fn = fn_lab["SeriesInstanceUID"].tolist()
        per_label_cases[(lab, "fp")] = u_fp
        per_label_cases[(lab, "fn")] = u_fn
        sel_uids.extend(u_fp)
        sel_uids.extend(u_fn)
    sel_uids = list(dict.fromkeys(sel_uids))

    # Load Hydra config
    cfg = load_experiment_config(experiment)
    cfg.data.fold = int(fold)
    data_cfg = cfg.data
    model_cfg = cfg.model

    # Prepare dataset (consistent with training config; reflect extra seg/meta)
    # Normalize extra seg settings
    extra_suffixes = normalize_extra_seg_suffixes(data_cfg.get("extra_seg_suffix", None))
    include_extra_seg = bool(extra_suffixes)
    num_extra_seg = (len(extra_suffixes) if extra_suffixes is not None else None)

    dataset = AneurysmVesselSegDataset(
        vessel_pred_dir=data_cfg.get("vessel_pred_dir", "/workspace/data/nnUNet_inference/predictions"),
        train_csv=data_cfg.get("train_csv", "/workspace/data/train.csv"),
        series_list=sel_uids,
        transform=get_val_transforms(
            input_size=tuple(data_cfg.get("input_size", [128, 224, 224])),
            keep_ratio=str(data_cfg.get("keep_ratio", "z-xy")),
            spatial_transform=str(data_cfg.get("spatial_transform", "resize")),
            pad_multiple=int(data_cfg.get("pad_multiple", 32)),
            include_extra_seg=include_extra_seg,
            num_extra_seg=num_extra_seg,
        ),
        cache_data=True,
        metadata_root=data_cfg.get("metadata_root", None),
        include_metadata=bool(data_cfg.get("include_metadata", True)),
        extra_seg_suffix=data_cfg.get("extra_seg_suffix", None),
    )

    # Restore model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_fold_checkpoint(Path("/workspace/logs/train/runs"), experiment, fold, prefer="epoch")
    model: AneurysmVesselSegROILitModule = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if getattr(cfg.model, "compile", False):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    load_res = model.load_state_dict(state_dict, strict=False)
    if getattr(load_res, "missing_keys", []):
        print(f"[analyze_val_predictions] Warning: Keys missing in checkpoint: {load_res.missing_keys}")
    if getattr(load_res, "unexpected_keys", []):
        print(f"[analyze_val_predictions] Warning: Unexpected keys for model: {load_res.unexpected_keys}")
    model.eval().to(device)
    forward_params = inspect.signature(model.forward).parameters
    requires_masks = "vessel_seg" in forward_params

    # UID -> sample data cache
    uid_to_sample: Dict[str, Dict[str, np.ndarray]] = {}

    def _load_and_predict(uid: str) -> Dict[str, np.ndarray]:
        # Find index
        try:
            idx = dataset.cases.index(uid)
        except ValueError:
            return {}
        sample = dataset[idx]
        img_tensor = sample["image"]  # (C,D,H,W)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_np = img_tensor.numpy()[0]  # (D,H,W)

        label_tensor = sample["vessel_label"]
        if label_tensor.dim() == 3:
            label_tensor = label_tensor.unsqueeze(0)
        if label_tensor.dim() == 4:
            label_tensor = label_tensor.unsqueeze(0)
        label_batch = label_tensor.long()

        num_loc = int(getattr(model.hparams, "num_location_classes", 13))
        vessel_one_hot = F.one_hot(label_batch.squeeze(1), num_classes=num_loc + 1)
        vessel_seg_tensor = vessel_one_hot[..., 1:].permute(0, 4, 1, 2, 3).contiguous()
        vessel_seg_tensor = vessel_seg_tensor.to(dtype=img_tensor.dtype)
        vessel_union_tensor = (label_batch > 0).to(img_tensor.dtype)
        v_union = vessel_union_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        # Concatenate extra vessel segs (if any) in 13-chunks
        extra_seg_cat = None
        if "extra_vessel_label" in sample or any(k.startswith("extra_vessel_label_") for k in sample.keys()):
            extra_keys: List[str] = []
            if "extra_vessel_label" in sample:
                extra_keys.append("extra_vessel_label")
            # Sort numbered suffixes ascending
            extra_keys.extend(sorted([k for k in sample.keys() if k.startswith("extra_vessel_label_")]))
            seg_list: List[torch.Tensor] = []
            for k in extra_keys:
                lab = sample[k]
                if lab.dim() == 3:
                    lab = lab.unsqueeze(0).unsqueeze(0)
                elif lab.dim() == 4:
                    lab = lab.unsqueeze(0)
                lab = lab.long()
                oh = F.one_hot(lab.squeeze(1), num_classes=num_loc + 1)
                seg = oh[..., 1:].permute(0, 4, 1, 2, 3).contiguous().to(dtype=img_tensor.dtype)
                seg_list.append(seg)
            if seg_list:
                extra_seg_cat = torch.cat(seg_list, dim=1)

        # GT spheres: regenerate from annotation points
        ann_pts = sample.get("ann_points")
        ann_valid = sample.get("ann_points_valid")
        sphere_radius_cfg = model_cfg.get(
            "sphere_radius", data_cfg.get("sphere_radius", getattr(model.hparams, "sphere_radius", 10))
        )
        sphere_radius = int(sphere_radius_cfg)
        if ann_pts is not None and ann_valid is not None and ann_valid.sum() > 0:
            pts = ann_pts.unsqueeze(0).to(dtype=torch.float32, device=device)
            valid = ann_valid.unsqueeze(0).to(dtype=torch.float32, device=device)
            out_shape = torch.Size((1, 1, *img_tensor.shape[-3:]))
            sphere_tensor = model._rasterize_spheres_from_points(pts, valid, out_shape, sphere_radius)
            s_gt = sphere_tensor.squeeze(0).squeeze(0).cpu().numpy()
        else:
            s_gt = np.zeros_like(v_union, dtype=np.uint8)

        # Predicted spheres (logits_sphere -> sigmoid -> upsample)
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            seg_for_model = vessel_seg_tensor.to(device=device, dtype=x.dtype)
            union_for_model = vessel_union_tensor.to(device=device, dtype=x.dtype)
            # Build kwargs to match forward signature
            fwd_params = inspect.signature(model.forward).parameters
            fwd_kwargs = {}
            if "vessel_seg" in fwd_params:
                fwd_kwargs["vessel_seg"] = seg_for_model
            if "vessel_union" in fwd_params:
                fwd_kwargs["vessel_union"] = union_for_model
            if extra_seg_cat is not None and "extra_vessel_seg" in fwd_params:
                fwd_kwargs["extra_vessel_seg"] = extra_seg_cat.to(device=x.device, dtype=x.dtype)
            # Metadata (if present/used)
            if "metadata_numeric" in sample and "metadata_numeric" in fwd_params:
                fwd_kwargs["metadata_numeric"] = sample["metadata_numeric"].unsqueeze(0).to(device=x.device)
            if "metadata_numeric_missing" in sample and "metadata_numeric_missing" in fwd_params:
                fwd_kwargs["metadata_numeric_missing"] = sample["metadata_numeric_missing"].unsqueeze(0).to(
                    device=x.device
                )
            if "metadata_categorical" in sample and "metadata_categorical" in fwd_params:
                fwd_kwargs["metadata_categorical"] = sample["metadata_categorical"].unsqueeze(0).to(device=x.device)

            out = model.forward(x, **fwd_kwargs)
            logit = out["logits_sphere"][0:1]
            pr = torch.sigmoid(
                F.interpolate(logit, size=img_np.shape, mode="trilinear", align_corners=False)
            )[0, 0]
            s_pred = pr.detach().cpu().numpy()

        return {"image": img_np, "vessel_union": v_union, "sphere_gt": s_gt, "sphere_pred": s_pred}

    def _norm01(x: np.ndarray) -> np.ndarray:
        # Percentile normalization
        vmin, vmax = np.percentile(x, 1.0), np.percentile(x, 99.0)
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = x - x.min()
            den = x.max() if x.max() > 0 else 1.0
            x = x / den
        return np.clip(x, 0.0, 1.0)

    def _mip(arr: np.ndarray) -> np.ndarray:
        # Z-axis MIP
        return np.max(arr, axis=0)

    for (lab, typ), uids in per_label_cases.items():
        out_dir = qual_dir / lab.replace("/", "-") / typ
        _ensure_dir(out_dir)
        for uid in uids:
            if uid not in uid_to_sample:
                uid_to_sample[uid] = _load_and_predict(uid)
            data = uid_to_sample.get(uid, {})
            if not data:
                continue
            img = _norm01(data["image"])  # (D,H,W) 0-1
            vu = data["vessel_union"]  # (D,H,W)
            sgt = data["sphere_gt"]  # (D,H,W)
            spr = data["sphere_pred"]  # (D,H,W)

            # MIP
            img_mip = _mip(img)
            vu_mip = _mip(vu.astype(float))
            sgt_mip = _mip(sgt.astype(float))
            spr_mip = _mip(spr.astype(float))

            # Render
            plt.figure(figsize=(5, 5))
            plt.imshow(img_mip, cmap="gray", vmin=0, vmax=1)
            # Vessel union (faint)
            if np.any(vu_mip > 0):
                plt.imshow(vu_mip, cmap="Blues", alpha=0.25, interpolation="nearest")
            # GT spheres
            if np.any(sgt_mip > 0):
                plt.imshow(sgt_mip, cmap="Reds", alpha=0.5, interpolation="nearest")
            # Predicted sphere probabilities
            if np.any(spr_mip > 0):
                plt.imshow(spr_mip, cmap="YlOrBr", alpha=0.35, interpolation="nearest")

            p = float(pred_df.loc[pred_df["SeriesInstanceUID"] == uid, f"prob_{lab}"].values[0])
            g = int(pred_df.loc[pred_df["SeriesInstanceUID"] == uid, f"gt_{lab}"].values[0])
            thr = thresholds.get(lab, 0.5)
            plt.title(f"{lab} | {typ.upper()} | p={p:.2f} thr={thr:.2f} gt={g} | {uid[:12]}…", fontsize=8)
            plt.axis("off")
            plt.tight_layout()
            save_path = out_dir / f"{uid}.png"
            plt.savefig(save_path, dpi=120)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validation error analysis (preprocessed ROI / Youden threshold)")
    parser.add_argument("--experiment", required=True, help="Experiment directory name (under logs/train/runs)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (single)")
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated folds to evaluate (e.g., 0,1,2). Overrides --fold when given.",
    )
    parser.add_argument(
        "--prefer_ckpt",
        choices=["epoch", "last"],
        default="last",
        help="Preference for checkpoint selection (epoch_* or last)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for inference (optional)")
    args = parser.parse_args()

    log_root = Path("/workspace/logs/train/runs")
    # Parse multiple folds option
    folds: List[int]
    if args.folds is not None and str(args.folds).strip():
        try:
            folds = [int(x) for x in str(args.folds).replace(" ", "").split(",") if x != ""]
        except Exception:
            raise ValueError("--folds must be comma-separated integers (e.g., 0,1,2)")
    else:
        folds = [int(args.fold)]

    # Single-fold evaluation or multi-fold aggregation
    if len(folds) == 1:
        fold = folds[0]
        outdir = log_root / args.experiment / "analysis" / f"fold{fold}"
        outdir.mkdir(parents=True, exist_ok=True)

        paths = run_val_prediction(
            experiment=args.experiment,
            fold=fold,
            outdir=outdir,
            prefer_ckpt=args.prefer_ckpt,
            batch_size_override=args.batch_size,
        )

        # Visualization
        pred_df = pd.read_parquet(paths["oof"])
        plot_roc_pr_calibration(pred_df, outdir)
        with open(paths["thresholds"], "r") as f:
            thresholds = json.load(f)
        render_error_thumbnails(
            experiment=args.experiment,
            fold=fold,
            pred_df=pred_df,
            thresholds=thresholds,
            outdir=outdir,
            n_examples_per_label=6,
        )

        print("=== Generated outputs ===")
        for k, v in paths.items():
            print(f"{k}: {v}")
        print(f"plots: {outdir/'plots'}")
        print(f"qualitative: {outdir/'qualitative'}")
    else:
        # Evaluate multiple folds and aggregate
        folds_sorted = sorted(set(int(f) for f in folds))
        folds_tag = "_".join(str(f) for f in folds_sorted)
        agg_outdir = log_root / args.experiment / "analysis" / f"folds_{folds_tag}"
        (agg_outdir / "tables").mkdir(parents=True, exist_ok=True)
        (agg_outdir / "plots").mkdir(parents=True, exist_ok=True)

        per_fold_paths: List[Dict[str, Path]] = []
        for fold in folds_sorted:
            outdir_f = log_root / args.experiment / "analysis" / f"fold{fold}"
            outdir_f.mkdir(parents=True, exist_ok=True)
            paths = run_val_prediction(
                experiment=args.experiment,
                fold=fold,
                outdir=outdir_f,
                prefer_ckpt=args.prefer_ckpt,
                batch_size_override=args.batch_size,
            )
            per_fold_paths.append(paths)

        # Concatenate per-fold predictions and evaluate on the merged set
        dfs = [pd.read_parquet(p["oof"]) for p in per_fold_paths]
        df_all = pd.concat(dfs, ignore_index=True)
        oof_all_path = agg_outdir / "tables" / "oof_val_predictions_all.parquet"
        df_all.to_parquet(oof_all_path, index=False)

        # Aggregated metrics (re-evaluate on concatenated data)
        y_true = np.stack([df_all[f"gt_{n}"].values for n in ANEURYSM_CLASSES], axis=1)
        y_prob = np.stack([df_all[f"prob_{n}"].values for n in ANEURYSM_CLASSES], axis=1)
        metrics_all = compute_competition_score(y_true, y_prob)
        with open(agg_outdir / "tables" / "metrics_summary_merged.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

        # Youden: optimal thresholds on concatenated data (reference)
        thresholds_merged: Dict[str, float] = {}
        for j, name in enumerate(ANEURYSM_CLASSES):
            thresholds_merged[name] = youden_threshold(y_true[:, j], y_prob[:, j])
        with open(agg_outdir / "tables" / "thresholds_merged.json", "w") as f:
            json.dump(thresholds_merged, f, indent=2)

        # Youden: mean threshold across folds
        thr_list = []
        for p in per_fold_paths:
            with open(p["thresholds"], "r") as f:
                thr_list.append(json.load(f))
        # Per-label average
        thr_mean: Dict[str, float] = {}
        thr_by_fold_rows = []
        for name in ANEURYSM_CLASSES:
            vals = [float(t.get(name, 0.5)) for t in thr_list]
            if len(vals) > 0:
                thr_mean[name] = float(np.mean(vals))
            else:
                thr_mean[name] = 0.5
        with open(agg_outdir / "tables" / "thresholds_mean.json", "w") as f:
            json.dump(thr_mean, f, indent=2)

        # Reference: thresholds per fold CSV
        for i, thr in enumerate(thr_list):
            row = {"fold": folds_sorted[i]}
            for name in ANEURYSM_CLASSES:
                row[name] = float(thr.get(name, 0.5))
            thr_by_fold_rows.append(row)
        if thr_by_fold_rows:
            pd.DataFrame(thr_by_fold_rows).to_csv(agg_outdir / "tables" / "thresholds_by_fold.csv", index=False)

        # Using concatenated data + mean thresholds, extract misclassifications / confusion matrices
        loc_names = ANEURYSM_CLASSES[:13]
        loc_thr = np.array([thr_mean[name] for name in loc_names], dtype=float)
        y_true_loc = y_true[:, : len(loc_names)]
        y_pred_loc = (y_prob[:, : len(loc_names)] >= loc_thr).astype(int)
        cm = multilabel_confusion_matrix(y_true_loc, y_pred_loc)
        cm_rows = []
        for idx, name in enumerate(loc_names):
            tn, fp, fn, tp = cm[idx].ravel()
            cm_rows.append({"label": name, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        pd.DataFrame(cm_rows).to_csv(agg_outdir / "tables" / "confusion_matrix_locations_mean_thr.csv", index=False)

        # Misclassification lists (using mean thresholds)
        err_rows_fp: List[Dict] = []
        err_rows_fn: List[Dict] = []
        for idx, r in df_all.iterrows():
            for name in ANEURYSM_CLASSES:
                p = float(r[f"prob_{name}"])
                g = int(r[f"gt_{name}"])
                thr = thr_mean.get(name, 0.5)
                pred = int(p >= thr)
                if pred == 1 and g == 0:
                    err_rows_fp.append(
                        {
                            "SeriesInstanceUID": r["SeriesInstanceUID"],
                            "label": name,
                            "prob": p,
                            "gt": g,
                            "Modality": r.get("Modality", None),
                            "PatientAge": r.get("PatientAge", None),
                            "PatientSex": r.get("PatientSex", None),
                            "ann_points_count": r.get("ann_points_count", 0),
                        }
                    )
                elif pred == 0 and g == 1:
                    err_rows_fn.append(
                        {
                            "SeriesInstanceUID": r["SeriesInstanceUID"],
                            "label": name,
                            "prob": p,
                            "gt": g,
                            "Modality": r.get("Modality", None),
                            "PatientAge": r.get("PatientAge", None),
                            "PatientSex": r.get("PatientSex", None),
                            "ann_points_count": r.get("ann_points_count", 0),
                        }
                    )
        pd.DataFrame(err_rows_fp).sort_values(["label", "prob"], ascending=[True, False]).to_csv(
            agg_outdir / "tables" / "errors_fp_mean_thr.csv", index=False
        )
        pd.DataFrame(err_rows_fn).sort_values(["label", "prob"], ascending=[True, True]).to_csv(
            agg_outdir / "tables" / "errors_fn_mean_thr.csv", index=False
        )

        # Visualization (on concatenated data)
        plot_roc_pr_calibration(df_all, agg_outdir)

        print("=== Aggregated results ===")
        print(f"oof_all: {oof_all_path}")
        print(f"thresholds_mean: {agg_outdir/'tables'/'thresholds_mean.json'}")
        print(f"thresholds_merged: {agg_outdir/'tables'/'thresholds_merged.json'}")
        print(f"metrics_merged: {agg_outdir/'tables'/'metrics_summary_merged.json'}")
        print(f"confusion(mean_thr): {agg_outdir/'tables'/'confusion_matrix_locations_mean_thr.csv'}")
        print(f"errors_fp(mean_thr): {agg_outdir/'tables'/'errors_fp_mean_thr.csv'}")
        print(f"errors_fn(mean_thr): {agg_outdir/'tables'/'errors_fn_mean_thr.csv'}")


if __name__ == "__main__":
    main()
