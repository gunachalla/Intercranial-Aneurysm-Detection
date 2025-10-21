"""
Lightning DataModule for aneurysm detection based on labeled vessel segmentations (seg.*).

It loads seg.npz / seg.npy produced by vessel_segmentation.py and provides
train/val DataLoaders for Lightning training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import rootutils
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.aneurysm_vessel_seg_dataset import (
    AneurysmVesselSegDataset,
    get_train_transforms,
    get_val_transforms,
    normalize_extra_seg_suffixes,
    ANEURYSM_CLASSES,
)


class RSNAAneurysmVesselSegDataModule(LightningDataModule):
    """Lightning DataModule for aneurysm detection using seg.* artifacts"""

    def __init__(
        self,
        vessel_pred_dir: str,
        train_csv: str,
        batch_size: int = 2,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        # Cross-validation settings
        n_folds: int = 5,
        fold: int = 0,
        split_seed: int = 42,
        split_file: Optional[str] = None,
        cache_data: bool = False,
        input_size: Tuple[int, int, int] = (128, 224, 224),
        train_transform_version: int = 1,
        keep_ratio: str = "z-xy",
        spatial_transform: str = "resize",
        pad_multiple: int = 32,
        metadata_root: Optional[str] = None,
        include_metadata: bool = True,
        metadata_numeric_dropout_prob: float = 0.0,
        metadata_categorical_dropout_prob: float = 0.0,
        # Additional segmentation suffixes (str/sequence; priority order). None to disable.
        extra_seg_suffix: Optional[Any] = None,
    ) -> None:
        super().__init__()
        # Allow multiple suffixes (comma/space/plus/semicolon separated)
        extra_seg_suffix = normalize_extra_seg_suffixes(extra_seg_suffix)
        self.save_hyperparameters(logger=False)
        self._extra_seg_suffixes: Optional[tuple[str, ...]] = extra_seg_suffix

        if not 0.0 <= metadata_numeric_dropout_prob <= 1.0:
            raise ValueError("metadata_numeric_dropout_prob must be in [0.0, 1.0]")
        if not 0.0 <= metadata_categorical_dropout_prob <= 1.0:
            raise ValueError("metadata_categorical_dropout_prob must be in [0.0, 1.0]")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self._prefetch_factor = int(prefetch_factor)

    # ---------- CV split helpers ----------

    def _get_split_file_path(self) -> Path:
        if self.hparams.split_file:
            return Path(self.hparams.split_file)
        vessel_path = Path(self.hparams.vessel_pred_dir)
        return (
            vessel_path.parent / f"cv_split_seg_{self.hparams.n_folds}fold_seed{self.hparams.split_seed}.json"
        )

    def _save_cv_split(self, fold_splits: Dict[int, Dict[str, List[str]]]) -> None:
        split_file = self._get_split_file_path()
        payload = {
            "n_folds": int(self.hparams.n_folds),
            "split_seed": int(self.hparams.split_seed),
            "fold_splits": fold_splits,
        }
        with open(split_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved CV split: {split_file}")

    def _load_cv_split(self) -> Optional[Dict[int, Dict[str, List[str]]]]:
        split_file = self._get_split_file_path()
        if not split_file.exists():
            return None
        try:
            with open(split_file, "r") as f:
                data = json.load(f)
        except Exception:
            return None
        if (
            data.get("n_folds") == int(self.hparams.n_folds)
            and data.get("split_seed") == int(self.hparams.split_seed)
            and "fold_splits" in data
        ):
            return data["fold_splits"]
        return None

    # ---------- Multilabel Stratified K-Fold ----------

    @staticmethod
    def _multilabel_stratified_kfold(
        series_uids: List[str],
        labels_df: pd.DataFrame,
        n_splits: int,
        random_state: int = 42,
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Split so that the multilabel positive rates are as balanced across folds as possible.

        Reference: Iterative Stratification (simplified). Iterate samples in the
        order of label rarity and greedily assign each to the fold that minimizes
        the deviation from target label counts (total_positives / n_splits).

        Args:
            series_uids: List of target SeriesInstanceUIDs (detection order)
            labels_df: DataFrame with index=SeriesInstanceUID and columns=ANEURYSM_CLASSES
            n_splits: Number of folds
            random_state: Random seed

        Returns:
            fold_splits: {fold_idx: {"train": [...], "val": [...], "n_train": int, "n_val": int}}
        """
        assert n_splits >= 2, "n_splits must be >= 2"
        rng = np.random.RandomState(int(random_state))

        # Compose label matrix in ANEURYSM_CLASSES order (missing -> 0)
        label_cols = list(ANEURYSM_CLASSES)
        y_list: List[np.ndarray] = []
        for uid in series_uids:
            if uid not in labels_df.index:
                # Unexpected; fill with all zeros for robustness
                y_list.append(np.zeros(len(label_cols), dtype=np.int64))
                continue
            row = labels_df.loc[uid]
            vals = []
            for name in label_cols:
                try:
                    v = int(row.get(name, 0))
                except Exception:
                    v = 0
                vals.append(1 if v else 0)
            y_list.append(np.asarray(vals, dtype=np.int64))
        Y = np.vstack(y_list)  # (N, C)
        N, C = Y.shape

        # Target samples per fold / target positives per label
        target_fold_size = np.full(n_splits, N / n_splits, dtype=np.float64)
        label_totals = Y.sum(axis=0).astype(np.float64)
        target_label_counts = label_totals / n_splits  # (C,)

        # Current fold state
        fold_sizes = np.zeros(n_splits, dtype=np.int64)
        fold_label_counts = np.zeros((n_splits, C), dtype=np.int64)
        assigned = -np.ones(N, dtype=np.int64)

        # Process labels from rarest to most frequent
        label_order = np.argsort(label_totals)  # ascending (rare -> frequent)
        for l in label_order:
            # Unassigned samples that are positive for label l
            cand_idx = np.where((Y[:, l] > 0) & (assigned < 0))[0]
            rng.shuffle(cand_idx)
            for i in cand_idx:
                L = np.where(Y[i] > 0)[0]
                if L.size == 0:
                    continue
                # Compute score per fold: label deviation + size deviation
                scores = []
                for f in range(n_splits):
                    # Label deviation (sum of current/target ratios)
                    denom = np.maximum(target_label_counts[L], 1e-6)
                    label_score = float((fold_label_counts[f, L] / denom).sum())
                    # Size deviation
                    size_score = float(fold_sizes[f] / max(target_fold_size[f], 1e-6))
                    # Overall score (weights can be tuned empirically)
                    score = label_score + 0.5 * size_score
                    scores.append(score)
                # Assign to the fold with the minimum score (tie -> smaller index)
                f_best = int(np.argmin(scores))
                assigned[i] = f_best
                fold_sizes[f_best] += 1
                fold_label_counts[f_best, L] += 1

        # Assign remaining all-zero (or unassigned) samples to the smallest fold
        remaining = np.where(assigned < 0)[0]
        if remaining.size > 0:
            rng.shuffle(remaining)
            for i in remaining:
                f_best = int(np.argmin(fold_sizes))
                assigned[i] = f_best
                fold_sizes[f_best] += 1

        # Convert to per-fold UID lists
        fold_to_val_indices: Dict[int, List[int]] = {f: [] for f in range(n_splits)}
        for idx, f in enumerate(assigned.tolist()):
            fold_to_val_indices[int(f)].append(int(idx))

        fold_splits: Dict[int, Dict[str, List[str]]] = {}
        for f in range(n_splits):
            val_idx = fold_to_val_indices[f]
            val_set = {series_uids[i] for i in val_idx}
            train_uids = [u for u in series_uids if u not in val_set]
            val_uids = [series_uids[i] for i in val_idx]
            fold_splits[f] = {
                "train": train_uids,
                "val": val_uids,
                "n_train": len(train_uids),
                "n_val": len(val_uids),
            }
        return fold_splits

    def _collect_error_cases(self) -> Set[str]:
        """Collect SeriesInstanceUIDs to exclude from error reports under vessel_pred_dir"""

        vessel_path = Path(self.hparams.vessel_pred_dir)
        error_cases: Set[str] = set()
        for json_path in sorted(vessel_path.glob("*.json")):
            try:
                with open(json_path, "r") as f:
                    payload = json.load(f)
            except Exception:
                continue
            extracted = self._extract_case_ids_from_error_json(payload)
            if extracted:
                error_cases.update(extracted)
        return error_cases

    @staticmethod
    def _extract_case_ids_from_error_json(payload: Any) -> Set[str]:
        """Extract candidate SeriesInstanceUIDs from an error report JSON"""

        case_ids: Set[str] = set()

        def _add_case_id(value: Any) -> None:
            if isinstance(value, str) and value:
                case_ids.add(value)

        candidates: List[Any] = []
        if isinstance(payload, dict):
            cases_block = payload.get("cases")
            if isinstance(cases_block, list):
                candidates.extend(cases_block)
            case_ids_block = payload.get("case_ids")
            if isinstance(case_ids_block, list):
                candidates.extend(case_ids_block)
            single_case = payload.get("case_id") or payload.get("SeriesInstanceUID")
            _add_case_id(single_case)
        elif isinstance(payload, list):
            candidates.extend(payload)

        for candidate in candidates:
            if isinstance(candidate, dict):
                for key in ("case_id", "caseId", "SeriesInstanceUID", "series_uid"):
                    if key in candidate:
                        _add_case_id(candidate[key])
                        break
            else:
                _add_case_id(candidate)

        return case_ids

    # ---------- Lightning hooks ----------

    def prepare_data(self) -> None:
        vessel_path = Path(self.hparams.vessel_pred_dir)
        if not vessel_path.exists():
            raise ValueError(f"Vessel segmentation predictions not found: {vessel_path}")

        csv_path = Path(self.hparams.train_csv)
        if not csv_path.exists():
            raise ValueError(f"Label CSV not found: {csv_path}")

        labels_df = pd.read_csv(csv_path)
        labels_df.set_index("SeriesInstanceUID", inplace=True)

        available_cases: List[str] = []
        for case_dir in vessel_path.iterdir():
            if not case_dir.is_dir():
                continue
            seg_npz = case_dir / "seg.npz"
            seg_npy = case_dir / "seg.npy"
            roi_npz = case_dir / "roi_data.npz"
            roi_npy = case_dir / "roi_data.npy"
            transform_file = case_dir / "transform.json"
            if not (transform_file.exists() and (seg_npz.exists() or seg_npy.exists())):
                continue
            if not (roi_npz.exists() or roi_npy.exists()):
                continue
            if self._extra_seg_suffixes:
                # Accept only if all specified suffixes exist
                ok_all = True
                for suffix in self._extra_seg_suffixes:
                    seg_file_npz = case_dir / f"seg_{suffix}.npz"
                    seg_file_npy = case_dir / f"seg_{suffix}.npy"
                    if not (seg_file_npz.exists() or seg_file_npy.exists()):
                        ok_all = False
                        break
                if not ok_all:
                    continue
            uid = case_dir.name
            if uid in labels_df.index:
                available_cases.append(uid)

        if not available_cases:
            raise ValueError(f"No available cases found under: {vessel_path}")

        error_cases = self._collect_error_cases()
        if error_cases:
            before_count = len(available_cases)
            available_cases = [uid for uid in available_cases if uid not in error_cases]
            removed_count = before_count - len(available_cases)
            if removed_count > 0:
                print(f"Excluded {removed_count} cases due to error reports")
            if not available_cases:
                raise ValueError("No available cases remain after excluding error cases")

        series_uids = sorted(available_cases)
        print(f"Available cases: {len(series_uids)}")

        fold_splits = self._load_cv_split()
        if fold_splits is None and self.hparams.n_folds > 1:
            # Split by multilabel stratified K-Fold based on label distribution
            # labels_df has SeriesInstanceUID in index and label names in columns
            strat_splits = self._multilabel_stratified_kfold(
                series_uids=series_uids,
                labels_df=labels_df,
                n_splits=int(self.hparams.n_folds),
                random_state=int(self.hparams.split_seed),
            )
            fold_splits = strat_splits
            self._save_cv_split(fold_splits)

        if self.hparams.n_folds > 1:
            key = int(self.hparams.fold)
            cur = (
                fold_splits[str(key)]
                if isinstance(fold_splits, dict) and str(key) in fold_splits
                else fold_splits[key]
            )
            train_uids = [u for u in cur["train"] if u in series_uids]
            val_uids = [u for u in cur["val"] if u in series_uids]
        else:
            n_train = int(len(series_uids) * 0.8)
            train_uids = series_uids[:n_train]
            val_uids = series_uids[n_train:]

        self.train_series_list = train_uids
        self.val_series_list = val_uids
        print(
            f"Data split (seg): fold={self.hparams.fold}/{self.hparams.n_folds} "
            f"train {len(train_uids)}, val {len(val_uids)}"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", "test"):
            if self.data_train is None and self.data_val is None:
                include_extra_seg = bool(self._extra_seg_suffixes)

                transforms_train = get_train_transforms(
                    input_size=self.hparams.input_size,
                    keep_ratio=self.hparams.keep_ratio,
                    spatial_transform=self.hparams.spatial_transform,
                    pad_multiple=self.hparams.pad_multiple,
                    include_extra_seg=include_extra_seg,
                    num_extra_seg=(len(self._extra_seg_suffixes) if self._extra_seg_suffixes else 0),
                    version=self.hparams.train_transform_version,
                )
                transforms_val = get_val_transforms(
                    input_size=self.hparams.input_size,
                    keep_ratio=self.hparams.keep_ratio,
                    spatial_transform=self.hparams.spatial_transform,
                    pad_multiple=self.hparams.pad_multiple,
                    include_extra_seg=include_extra_seg,
                    num_extra_seg=(len(self._extra_seg_suffixes) if self._extra_seg_suffixes else 0),
                )

                self.data_train = AneurysmVesselSegDataset(
                    vessel_pred_dir=self.hparams.vessel_pred_dir,
                    train_csv=self.hparams.train_csv,
                    series_list=self.train_series_list,
                    transform=transforms_train,
                    cache_data=self.hparams.cache_data,
                    metadata_root=self.hparams.metadata_root,
                    include_metadata=self.hparams.include_metadata,
                    metadata_numeric_dropout_prob=self.hparams.metadata_numeric_dropout_prob,
                    metadata_categorical_dropout_prob=self.hparams.metadata_categorical_dropout_prob,
                    extra_seg_suffix=self._extra_seg_suffixes,
                )
                self.data_val = AneurysmVesselSegDataset(
                    vessel_pred_dir=self.hparams.vessel_pred_dir,
                    train_csv=self.hparams.train_csv,
                    series_list=self.val_series_list,
                    transform=transforms_val,
                    cache_data=self.hparams.cache_data,
                    metadata_root=self.hparams.metadata_root,
                    include_metadata=self.hparams.include_metadata,
                    metadata_numeric_dropout_prob=0.0,
                    metadata_categorical_dropout_prob=0.0,
                    extra_seg_suffix=self._extra_seg_suffixes,
                )
                self.data_test = self.data_val
        elif stage == "predict":
            raise NotImplementedError("predict stage is not implemented for seg datamodule")

    # ---------- loaders ----------

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
            raise RuntimeError("Call setup('fit') before calling train_dataloader")
        return DataLoader(
            self.data_train,
            batch_size=int(self.hparams.batch_size),
            shuffle=True,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=bool(self.hparams.persistent_workers),
            prefetch_factor=self._prefetch_factor,
            collate_fn=self._collate_with_meta,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.data_val is None:
            raise RuntimeError("Call setup('fit') before calling val_dataloader")
        return DataLoader(
            self.data_val,
            batch_size=int(self.hparams.batch_size),
            shuffle=False,
            num_workers=int(self.hparams.num_workers),
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=bool(self.hparams.persistent_workers),
            prefetch_factor=self._prefetch_factor,
            collate_fn=self._collate_with_meta,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", "test", None):
            self.data_train = None
            self.data_val = None
            self.data_test = None

    # ---------- Utilities ----------

    @property
    def num_classes(self) -> int:
        return len(self.data_train[0]["labels"]) if self.data_train is not None else 14

    @property
    def example_input_array(self) -> torch.Tensor:
        if self.data_train is None:
            raise RuntimeError("Call setup('fit') before retrieving example_input_array")
        batch = next(iter(self.train_dataloader()))
        return batch["image"]

    def on_before_zero_grad(self, *args: Any, **kwargs: Any) -> None:
        return None

    @staticmethod
    def _collate_with_meta(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function.
        Keep meta and series_uid as lists; apply default collate to others.
        Supports presence/absence of sphere_mask.
        If optional keys like ann_points/ann_points_valid are missing in some
        samples, fill zero placeholders so the batch is consistent.

        Args:
            batch: list of sample dicts

        Returns:
            collated batch dict
        """
        from torch.utils.data._utils.collate import default_collate

        # Extract meta and series_uid
        metas = [b.get("meta") for b in batch] if ("meta" in batch[0]) else None
        series_uids = [b.get("series_uid") for b in batch] if ("series_uid" in batch[0]) else None

        # Optional keys to complete: ann_points, ann_points_valid
        optional_point_keys = ("ann_points", "ann_points_valid")
        # Estimate the max number of points for ann_points from present samples
        max_pts = 0
        for b in batch:
            if "ann_points" in b and isinstance(b["ann_points"], (list, tuple)):
                # Handle rare case of list-type ann_points
                try:
                    max_pts = max(max_pts, int(len(b["ann_points"])))
                except Exception:
                    pass
            elif "ann_points" in b and hasattr(b["ann_points"], "shape"):
                try:
                    max_pts = max(max_pts, int(b["ann_points"].shape[0]))
                except Exception:
                    pass

        # Temporarily remove meta/series_uid and complete optional keys
        batch_wo_meta = []
        for b in batch:
            b2 = dict(b)
            if "meta" in b2:
                b2.pop("meta")
            if "series_uid" in b2:
                b2.pop("series_uid")
            # Fill missing ann_points/ann_points_valid.
            # If max_pts == 0, prepare (0,3)/(0,) tensors.
            if "ann_points" not in b2:
                if max_pts > 0:
                    b2["ann_points"] = torch.zeros((max_pts, 3), dtype=torch.float32)
                else:
                    b2["ann_points"] = torch.zeros((0, 3), dtype=torch.float32)
            if "ann_points_valid" not in b2:
                if max_pts > 0:
                    b2["ann_points_valid"] = torch.zeros((max_pts,), dtype=torch.uint8)
                else:
                    b2["ann_points_valid"] = torch.zeros((0,), dtype=torch.uint8)
            batch_wo_meta.append(b2)

        # Apply default collate
        collated = default_collate(batch_wo_meta)

        # Add meta and series_uid back
        if metas is not None:
            collated["meta"] = metas
        if series_uids is not None:
            collated["series_uid"] = series_uids

        return collated
