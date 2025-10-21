#!/usr/bin/env python
"""Training script for cross-validation."""

import os
import numpy as np
import subprocess
import wandb
from omegaconf import OmegaConf
from hydra import initialize, compose
import rootutils
from pathlib import Path
import time
import json

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.my_utils.kaggle_utils import get_ckpt_name


def run_cross_validation(
    experiment: str,
    n_folds: int = 5,
    except_folds: list = None,
    resume_folds: list = None,
    use_wandb: bool = True,
):
    """
    Run cross-validation.

    Args:
        experiment: Experiment name
        n_folds: Number of folds
        except_folds: List of folds to skip
        resume_folds: List of folds to resume from checkpoint
        use_wandb: Whether to log results to Weights & Biases
    """
    if except_folds is None:
        except_folds = []
    if resume_folds is None:
        resume_folds = []

    print(f"=== Starting cross-validation: {experiment} ===")
    print(f"Num folds: {n_folds}")
    print(f"Excluded folds: {except_folds}")
    print(f"Resume folds: {resume_folds}")

    # Store results for each fold
    fold_results = {}

    # Run training for each fold
    for fold in range(n_folds):
        if fold in except_folds:
            print(f"\nFold {fold}: skipped")
            continue

        print(f"\n=== Starting fold {fold}/{n_folds} ===")

        # Build command
        command = [
            "python",
            "/workspace/src/train.py",
            f"experiment={experiment}",
            f"data.fold={fold}",
            f"data.n_folds={n_folds}",
            f"seed={fold}",  # use fold index as seed
        ]

        # If resuming, set checkpoint path
        if fold in resume_folds:
            ckpt_name = get_ckpt_name("last")
            ckpt_path = os.path.join(
                "/workspace/logs/train/runs", experiment, "checkpoints", f"fold{fold}", ckpt_name
            )
            if os.path.exists(ckpt_path):
                print(f"Resuming from checkpoint: {ckpt_path}")
                command.append(f"ckpt_path={ckpt_path}")
            else:
                print(f"Warning: checkpoint not found: {ckpt_path}")

        # Execute command
        print("Command:", " ".join(command))
        start_time = time.time()

        try:
            result = subprocess.run(command, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"Error: execution failed for fold {fold}")
                continue
        except Exception as e:
            print(f"Error: exception during fold {fold} execution: {e}")
            continue

        elapsed_time = time.time() - start_time
        print(f"Fold {fold} finished (elapsed: {elapsed_time:.2f}s)")

    # Load config (for log paths)
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        cfg.paths.output_dir = "${hydra.runtime.output_dir}"
        cfg.paths.work_dir = "${hydra.runtime.cwd}"
        cfg.hydra.run.dir = cfg.log_dir
        cfg.hydra.runtime.output_dir = cfg.hydra.run.dir

    # Collect metric values
    save_dir = Path(cfg.log_dir) / "metric_values"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load results for each fold
    for fold in range(n_folds):
        if fold in except_folds:
            continue

        metric_file = save_dir / f"metric_value_{fold}.txt"
        if metric_file.exists():
            try:
                with open(metric_file, "r") as f:
                    result = float(f.read())
                    fold_results[f"fold_{fold}"] = result
                    print(f"Fold {fold} score: {result:.4f}")
            except Exception as e:
                print(f"Warning: failed to read metric for fold {fold}: {e}")

    # Compute CV mean/std
    if fold_results:
        scores = [fold_results[f"fold_{fold}"] for fold in range(n_folds) if f"fold_{fold}" in fold_results]
        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        fold_results["cv_mean"] = cv_mean
        fold_results["cv_std"] = cv_std

        print(f"\n=== Cross-validation results ===")
        print(f"Mean score: {cv_mean:.4f} ± {cv_std:.4f}")

        # Save results as JSON
        results_file = save_dir / "cv_results.json"
        with open(results_file, "w") as f:
            json.dump(fold_results, f, indent=2)
        print(f"Saved results: {results_file}")
    else:
        print("Warning: no valid results found")
        cv_mean = 0.0
        cv_std = 0.0

    # Log to Weights & Biases
    if use_wandb:
        try:
            run = wandb.init(
                project="RSNA_BrainBBox_CV",
                name=f"{cfg.experiment_name}_CV",
                dir="/workspace/logs/wandb_cv",
                config=OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False),
            )

            # Log fold results and CV mean
            run.log(fold_results)

            # Create a summary table
            if fold_results:
                table_data = []
                for fold in range(n_folds):
                    if f"fold_{fold}" in fold_results:
                        table_data.append([fold, fold_results[f"fold_{fold}"]])

                table = wandb.Table(columns=["Fold", "Score"], data=table_data)
                run.log({"cv_results_table": table})

            run.finish()
            print("Finished logging to Weights & Biases")
        except Exception as e:
            print(f"Weights & Biases logging error: {e}")

    return cv_mean, cv_std, fold_results


if __name__ == "__main__":
    # Experiment list
    experiment_list = [
        "251013-seg_tf-v4-nnunet_truncate1_preV6_1-ex_dav6w3-m32g64-e25-w01_005_1-s128_256_256",
    ]

    # Parameters
    n_folds = 5
    except_folds = []  # folds to skip
    resume_folds = []  # folds to resume

    # Run CV for each experiment
    for experiment in experiment_list:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment}")
        print(f"{'='*60}")

        cv_mean, cv_std, fold_results = run_cross_validation(
            experiment=experiment,
            n_folds=n_folds,
            except_folds=except_folds,
            resume_folds=resume_folds,
            use_wandb=False,
        )

        print(f"\nFinal results - {experiment}:")
        print(f"  CV mean: {cv_mean:.4f} ± {cv_std:.4f}")
