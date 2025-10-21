"""
Dataset for aneurysm detection using labeled vessel segmentations (seg.*).

Loads seg.npz / seg.npy saved by vessel_segmentation.py (background=0, foreground=1..13),
reorders labels into the detection task order, and builds training batches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    EnsureTyped,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianSharpend,
)

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.custom_transforms import (
    Resize3DFitPadCropd,
    RandInvertAroundMeanD,
    PadToMultipleOfd,
    Resize3DMaxSized,
    Resize3DFitWithinPadToSized,
    Resize3DXYLongSideZDownOnlyPadToSized,
)
from src.my_utils.dicom_metadata import (
    DEFAULT_MANUFACTURER_CATEGORY,
    DEFAULT_SEX_PLACEHOLDER,
    MAJOR_MANUFACTURERS,
)

# ------------------------------------------------------------------------------------
# Constants used for aneurysm detection
# ------------------------------------------------------------------------------------

# 13 locations + overall label names (match the metric weighting order)
ANEURYSM_CLASSES = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]

# Mapping to reorder nnUNet vessel segmentation channels to detection-task order
DET_TO_SEG_OFFSET = [
    5,  # 0: Left Infraclinoid ICA        <- seg 6
    4,  # 1: Right Infraclinoid ICA       <- seg 5
    7,  # 2: Left Supraclinoid ICA        <- seg 8
    6,  # 3: Right Supraclinoid ICA       <- seg 7
    9,  # 4: Left MCA                     <- seg 10
    8,  # 5: Right MCA                    <- seg 9
    12,  # 6: AComA                        <- seg 13
    11,  # 7: Left ACA                     <- seg 12
    10,  # 8: Right ACA                    <- seg 11
    3,  # 9: Left PComA                   <- seg 4
    2,  # 10: Right PComA                 <- seg 3
    1,  # 11: Basilar Tip                 <- seg 2
    0,  # 12: Other Posterior Circulation <- seg 1
]

# Max number of ROI annotation points when keeping a fixed-length array
MAX_ANN_POINTS: int = 8

# Mapping table: seg order (0:background, 1..13) -> detection order (0:background, 1..13)
_SEG_TO_DET = np.zeros(14, dtype=np.uint8)
for det_idx, seg_offset in enumerate(DET_TO_SEG_OFFSET):
    _SEG_TO_DET[int(seg_offset) + 1] = det_idx + 1


METADATA_NUMERIC_PERCENTILES = {
    "PatientAgeYears": {"p02": 25.0, "p50": 60.0, "p98": 85.0},
    "SliceThickness": {"p02": 0.5, "p50": 1.0, "p98": 5.0},
    "SpacingBetweenSlices": {"p02": 0.4, "p50": 0.799999, "p98": 24.0},
    "ReconstructionDiameter": {"p02": 180.0, "p50": 230.0, "p98": 319.86},
    "KVP": {"p02": 80.0, "p50": 120.0, "p98": 120.0},
    "XRayTubeCurrent": {"p02": 144.0, "p50": 328.0, "p98": 791.0},
    "ExposureTime": {"p02": 250.0, "p50": 500.0, "p98": 1000.0},
    "RepetitionTime": {"p02": 6.09392, "p50": 30.0, "p98": 8441.88},
    "EchoTime": {"p02": 2.132, "p50": 7.0, "p98": 129.0},
    "FlipAngle": {"p02": 8.0, "p50": 25.0, "p98": 160.1276},
    "PixelSpacingX": {"p02": 0.247396, "p50": 0.449219, "p98": 0.9375},
    "PixelSpacingY": {"p02": 0.247396, "p50": 0.449219, "p98": 0.9375},
}

METADATA_NUMERIC_FEATURES: Tuple[str, ...] = tuple(METADATA_NUMERIC_PERCENTILES.keys())
METADATA_NUMERIC_DEFAULTS: Dict[str, float] = {
    name: stats["p50"] for name, stats in METADATA_NUMERIC_PERCENTILES.items()
}

METADATA_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "PatientSex",
    "Manufacturer",
    "Modality",
)

METADATA_CATEGORICAL_DEFAULTS: Dict[str, str] = {
    "PatientSex": DEFAULT_SEX_PLACEHOLDER,
    "Manufacturer": DEFAULT_MANUFACTURER_CATEGORY,
    "Modality": "UNKNOWN",
}

KNOWN_CATEGORICAL_VALUES: Dict[str, List[str]] = {
    "PatientSex": ["F", "M", DEFAULT_SEX_PLACEHOLDER],
    "Manufacturer": sorted({*MAJOR_MANUFACTURERS, DEFAULT_MANUFACTURER_CATEGORY}),
    "Modality": ["CT", "MR", "UNKNOWN"],
}


def normalize_extra_seg_suffixes(
    extra_seg_suffix: Optional[Union[str, Sequence[str]]],
) -> Optional[Tuple[str, ...]]:
    """
    Normalize additional segmentation suffix specification and return a prioritized tuple.

    - If a string: split by comma/space/plus/semicolon
    - If a list/tuple: keep the original order
    - Remove empty tokens; do not de-duplicate (first match wins)
    """

    if extra_seg_suffix is None:
        return None

    tokens: List[str] = []
    if isinstance(extra_seg_suffix, str):
        text = extra_seg_suffix.strip()
        if not text:
            return None
        # Split by comma/semicolon/plus and then by whitespace
        tmp = []
        for sep in [",", ";", "+"]:
            if sep in text:
                # Normalize separators then split by whitespace
                text = text.replace(sep, " ")
        tmp = [t.strip() for t in text.split()]
        tokens = [t for t in tmp if t]
    else:
        try:
            tokens = [str(t).strip() for t in extra_seg_suffix if str(t).strip()]
        except TypeError:
            # Treat non-iterables as a single element
            single = str(extra_seg_suffix).strip()
            tokens = [single] if single else []

    if not tokens:
        return None
    return tuple(tokens)


def _convert_label_map(label_map: np.ndarray) -> np.ndarray:
    """Convert a seg-order label map to the detection-class order."""
    lbl = label_map
    if lbl.ndim == 3:
        lbl = lbl[np.newaxis, ...]
    if lbl.shape[0] != 1:
        raise ValueError(f"The first dimension of label_map must be 1: {lbl.shape}")
    converted = _SEG_TO_DET[lbl.astype(np.uint8)]
    return converted


class AneurysmVesselSegDataset(Dataset):
    """Dataset for aneurysm detection using seg.* inputs"""

    def __init__(
        self,
        vessel_pred_dir: Union[str, Path],
        train_csv: Union[str, Path],
        series_list: Optional[List[str]] = None,
        transform: Optional[Compose] = None,
        cache_data: bool = False,
        metadata_root: Optional[Union[str, Path]] = None,
        include_metadata: bool = True,
        metadata_numeric_dropout_prob: float = 0.1,
        metadata_categorical_dropout_prob: float = 0.1,
        # Additional segmentations: str or sequence (priority order). None to disable.
        extra_seg_suffix: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        self.vessel_pred_dir = Path(vessel_pred_dir)
        self.train_csv = Path(train_csv)
        self.transform = transform
        self.cache_data = cache_data
        if not 0.0 <= metadata_numeric_dropout_prob <= 1.0:
            raise ValueError("metadata_numeric_dropout_prob must be in [0.0, 1.0]")
        if not 0.0 <= metadata_categorical_dropout_prob <= 1.0:
            raise ValueError("metadata_categorical_dropout_prob must be in [0.0, 1.0]")
        self.metadata_numeric_dropout_prob = float(metadata_numeric_dropout_prob)
        self.metadata_categorical_dropout_prob = float(metadata_categorical_dropout_prob)
        # Normalize additional segmentation suffixes (priority order)
        self._extra_seg_suffixes = normalize_extra_seg_suffixes(extra_seg_suffix)
        self._extra_seg_required = self._extra_seg_suffixes is not None
        self._num_extra_seg = len(self._extra_seg_suffixes) if self._extra_seg_suffixes is not None else 0
        # Backward-compat: remember the first suffix to store in meta["extra_vessel_suffix"]
        self._chosen_extra_suffix: Dict[str, str] = {}

        resolved_metadata_root: Optional[Path] = None
        metadata_candidates: List[Path] = []
        if metadata_root is not None:
            metadata_candidates.append(Path(metadata_root))
        else:
            metadata_candidates.append(self.vessel_pred_dir / "series_niix")
            metadata_candidates.append(self.vessel_pred_dir.parent / "series_niix")
            try:
                metadata_candidates.append(self.vessel_pred_dir.parent.parent / "series_niix")
            except Exception:
                pass
            metadata_candidates.append(Path("/workspace/data/series_niix"))

        for candidate in metadata_candidates:
            if candidate is None:
                continue
            if candidate.exists():
                resolved_metadata_root = candidate
                break

        self._metadata_root = resolved_metadata_root
        self._metadata_enabled = bool(include_metadata)

        self.labels_df = pd.read_csv(self.train_csv)
        self.labels_df.set_index("SeriesInstanceUID", inplace=True)

        available_cases: List[str] = []
        case_dir_map: Dict[str, Path] = {}
        missing_extra_cases: List[str] = []
        for case_dir in self.vessel_pred_dir.iterdir():
            if not case_dir.is_dir():
                continue
            seg_npz = case_dir / "seg.npz"
            seg_npy = case_dir / "seg.npy"
            roi_npz = case_dir / "roi_data.npz"
            roi_npy = case_dir / "roi_data.npy"
            transform_file = case_dir / "transform.json"
            if not transform_file.exists():
                continue
            if not (seg_npz.exists() or seg_npy.exists()):
                continue
            if not (roi_npz.exists() or roi_npy.exists()):
                continue
            series_uid = case_dir.name
            if series_uid not in self.labels_df.index:
                continue
            if series_list is not None and series_uid not in series_list:
                continue
            if self._extra_seg_required:
                assert self._extra_seg_suffixes is not None
                # Require all specified suffixes to exist
                all_found = True
                for suffix in self._extra_seg_suffixes:
                    seg_file_npz = case_dir / f"seg_{suffix}.npz"
                    seg_file_npy = case_dir / f"seg_{suffix}.npy"
                    if not (seg_file_npz.exists() or seg_file_npy.exists()):
                        all_found = False
                        break
                if not all_found:
                    missing_extra_cases.append(series_uid)
                    continue
                # Keep the first suffix for backward compatibility
                self._chosen_extra_suffix[series_uid] = self._extra_seg_suffixes[0]
            available_cases.append(series_uid)
            case_dir_map[series_uid] = case_dir

        if not available_cases:
            raise ValueError(f"No available cases found: {self.vessel_pred_dir}")

        if missing_extra_cases:
            sample_items = ", ".join(missing_extra_cases[:5])
            raise FileNotFoundError(
                "Additional segmentations are missing: "
                f"suffixes={self._extra_seg_suffixes} total_missing={len(missing_extra_cases)} "
                f"examples={sample_items}"
            )

        self.cases = sorted(available_cases)
        self._case_dirs = {uid: case_dir_map[uid] for uid in self.cases}
        print(f"AneurysmVesselSegDataset: {len(self.cases)} cases")

        self.aneurysm_classes = ANEURYSM_CLASSES
        self.aneurysm_class_to_index = {c: i for i, c in enumerate(self.aneurysm_classes)}

        self.cache: Optional[Dict[int, Dict]] = {} if cache_data else None

        self.metadata_numeric_features: Tuple[str, ...] = METADATA_NUMERIC_FEATURES
        self.metadata_categorical_features: Tuple[str, ...] = METADATA_CATEGORICAL_FEATURES
        self.metadata_numeric_values: Dict[str, np.ndarray] = {}
        self.metadata_numeric_missing: Dict[str, np.ndarray] = {}
        self.metadata_categorical_indices: Dict[str, np.ndarray] = {}
        self.metadata_categorical_vocab: Dict[str, Dict[str, int]] = {}

        if self._metadata_enabled:
            self._prepare_metadata_features()

    def __len__(self) -> int:
        return len(self.cases)

    def _resolve_metadata_path(self, series_uid: str) -> Optional[Path]:
        """Resolve search path to series_metadata.json"""

        case_dir = self._case_dirs.get(series_uid)
        if case_dir is not None:
            candidate = case_dir / "series_metadata.json"
            if candidate.exists():
                return candidate

        if self._metadata_root is not None:
            candidate = self._metadata_root / series_uid / "series_metadata.json"
            if candidate.exists():
                return candidate

        return None

    def _load_series_metadata(self, series_uid: str) -> Dict[str, Any]:
        """Load per-series metadata JSON"""

        metadata_path = self._resolve_metadata_path(series_uid)
        if metadata_path is None:
            return {}

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return {}

        pixel_spacing = metadata.get("PixelSpacing")
        if isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 2:
            try:
                metadata.setdefault("PixelSpacingX", float(pixel_spacing[0]))
                metadata.setdefault("PixelSpacingY", float(pixel_spacing[1]))
            except (TypeError, ValueError):
                metadata.setdefault("PixelSpacingX", None)
                metadata.setdefault("PixelSpacingY", None)

        return metadata

    @staticmethod
    def _extract_numeric_value(metadata: Dict[str, Any], key: str) -> Optional[float]:
        """Safely convert numeric metadata to float"""

        value = metadata.get(key)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _prepare_metadata_features(self) -> None:
        """Precompute per-case metadata features"""

        numeric_features = self.metadata_numeric_features
        categorical_features = self.metadata_categorical_features

        categorical_values: Dict[str, set] = {}
        for feature in categorical_features:
            base_values = set(KNOWN_CATEGORICAL_VALUES.get(feature, []))
            base_values.add(METADATA_CATEGORICAL_DEFAULTS[feature])
            categorical_values[feature] = base_values

        per_case_numeric: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        per_case_categorical_raw: Dict[str, Dict[str, str]] = {}

        for series_uid in self.cases:
            metadata = self._load_series_metadata(series_uid)

            numeric_vals: List[float] = []
            numeric_missing: List[float] = []
            for feature in numeric_features:
                stats = METADATA_NUMERIC_PERCENTILES[feature]
                value = self._extract_numeric_value(metadata, feature)
                is_missing = value is None
                if is_missing:
                    value = METADATA_NUMERIC_DEFAULTS[feature]
                lower = stats["p02"]
                upper = stats["p98"]
                value = float(np.clip(value, lower, upper))
                denom = max(upper - lower, 1e-6)
                centered = value - METADATA_NUMERIC_DEFAULTS[feature]
                normalized = centered / denom
                numeric_vals.append(float(normalized))
                numeric_missing.append(1.0 if is_missing else 0.0)

            categorical_raw: Dict[str, str] = {}
            for feature in categorical_features:
                default_value = METADATA_CATEGORICAL_DEFAULTS[feature]
                value = metadata.get(feature, default_value)
                if value is None or str(value).strip() == "":
                    value = default_value
                value_str = str(value)
                categorical_values[feature].add(value_str)
                categorical_raw[feature] = value_str

            per_case_numeric[series_uid] = (
                np.asarray(numeric_vals, dtype=np.float32),
                np.asarray(numeric_missing, dtype=np.float32),
            )
            per_case_categorical_raw[series_uid] = categorical_raw

        vocabularies: Dict[str, Dict[str, int]] = {}
        for feature in categorical_features:
            default_value = METADATA_CATEGORICAL_DEFAULTS[feature]
            candidates = categorical_values[feature]
            ordered = [default_value] + sorted(v for v in candidates if v != default_value)
            vocabularies[feature] = {value: idx for idx, value in enumerate(ordered)}

        self.metadata_categorical_vocab = vocabularies

        for series_uid in self.cases:
            numeric_vals, numeric_missing = per_case_numeric[series_uid]
            categorical_raw = per_case_categorical_raw[series_uid]
            categorical_indices = [
                vocabularies[feature][categorical_raw[feature]] for feature in categorical_features
            ]
            self.metadata_numeric_values[series_uid] = numeric_vals
            self.metadata_numeric_missing[series_uid] = numeric_missing
            self.metadata_categorical_indices[series_uid] = np.asarray(categorical_indices, dtype=np.int64)

    def _load_np_archive(self, case_dir: Path, stem: str, key: str) -> np.ndarray:
        """Utility to load from npz/npy uniformly"""
        npz_path = case_dir / f"{stem}.npz"
        if npz_path.exists():
            with np.load(npz_path) as f:
                if key not in f:
                    raise KeyError(f"{key} not found in {npz_path}")
                return f[key]
        npy_path = case_dir / f"{stem}.npy"
        if npy_path.exists():
            return np.load(npy_path)
        raise FileNotFoundError(f"{stem}.npz / {stem}.npy not found in {case_dir}")

    def _make_label_vector(self, labels: pd.Series) -> np.ndarray:
        vec = np.zeros(len(self.aneurysm_classes), dtype=np.float32)
        for cls_name in self.aneurysm_classes:
            if cls_name in labels.index:
                vec[self.aneurysm_class_to_index[cls_name]] = float(labels[cls_name])
        return vec

    def _load_case_data(self, idx: int) -> Dict:
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        series_uid = self.cases[idx]
        case_dir = self.vessel_pred_dir / series_uid

        vessel_label_seg = self._load_np_archive(case_dir, "seg", "segmentation")
        roi_data = self._load_np_archive(case_dir, "roi_data", "roi")
        if roi_data.ndim == 3:
            roi_data = roi_data[np.newaxis, ...]

        # Load all additional segmentations
        extra_seg_maps: List[np.ndarray] = []
        if self._extra_seg_required:
            assert self._extra_seg_suffixes is not None
            for suffix in self._extra_seg_suffixes:
                stem = f"seg_{suffix}"
                arr = self._load_np_archive(case_dir, stem, "segmentation")
                extra_seg_maps.append(arr)

        with open(case_dir / "transform.json", "r") as f:
            transform_info = json.load(f)

        roi_annotations: List[Dict] = []
        ann_file = case_dir / "roi_annotations.json"
        if ann_file.exists():
            with open(ann_file, "r") as f:
                ann_data = json.load(f)
                roi_annotations = ann_data.get("roi_annotations", [])

        labels = self.labels_df.loc[series_uid]

        data = {
            "series_uid": series_uid,
            "vessel_label_seg": vessel_label_seg,
            "roi_data": roi_data,
            "transform_info": transform_info,
            "roi_annotations": roi_annotations,
            "labels": labels,
        }

        if extra_seg_maps:
            # Store first one under the BC key, then index the rest
            data["extra_vessel_label_seg"] = extra_seg_maps[0]
            for i in range(1, len(extra_seg_maps)):
                data[f"extra_vessel_label_seg_{i}"] = extra_seg_maps[i]

        if self._metadata_enabled:
            numeric_vals = self.metadata_numeric_values.get(series_uid)
            numeric_missing = self.metadata_numeric_missing.get(series_uid)
            categorical_indices = self.metadata_categorical_indices.get(series_uid)
            if numeric_vals is not None and numeric_missing is not None:
                data["metadata_numeric"] = numeric_vals.copy()
                data["metadata_numeric_missing"] = numeric_missing.copy()
            if categorical_indices is not None:
                data["metadata_categorical"] = categorical_indices.copy()

        if self.cache is not None:
            self.cache[idx] = data

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._load_case_data(idx)

        image = data["roi_data"]
        labels_vec = self._make_label_vector(data["labels"])
        vessel_label = _convert_label_map(data["vessel_label_seg"])

        # Additional segmentations: convert all to detection-order maps
        extra_labels_converted: List[np.ndarray] = []
        if self._extra_seg_required:
            seg_arr0 = data.get("extra_vessel_label_seg")
            if seg_arr0 is not None:
                extra_labels_converted.append(_convert_label_map(seg_arr0))
            # Subsequent extra maps
            for i in range(1, self._num_extra_seg):
                seg_arr_i = data.get(f"extra_vessel_label_seg_{i}")
                if seg_arr_i is not None:
                    extra_labels_converted.append(_convert_label_map(seg_arr_i))

        sample: Dict[str, Union[np.ndarray, float, int, Dict]] = {
            "image": image,
            "vessel_label": vessel_label,
            "labels": labels_vec,
            "series_uid": data["series_uid"],
            "meta": {
                "transform_info": data["transform_info"],
                "roi_annotations_count": len(data["roi_annotations"]),
            },
        }

        if self._extra_seg_suffixes is not None:
            sample["meta"]["extra_vessel_suffix_candidates"] = list(self._extra_seg_suffixes)
            # Store the first suffix for backward compatibility
            chosen = self._chosen_extra_suffix.get(data["series_uid"]) if self._extra_seg_required else None
            if chosen is not None:
                sample["meta"]["extra_vessel_suffix"] = chosen

        # Expand additional segmentations into the sample (first uses the BC key, others are indexed)
        if extra_labels_converted:
            sample["extra_vessel_label"] = extra_labels_converted[0]
            for i in range(1, len(extra_labels_converted)):
                sample[f"extra_vessel_label_{i}"] = extra_labels_converted[i]

        if self._metadata_enabled:
            if "metadata_numeric" in data and "metadata_numeric_missing" in data:
                sample["metadata_numeric"] = data["metadata_numeric"]
                sample["metadata_numeric_missing"] = data["metadata_numeric_missing"]
            if "metadata_categorical" in data:
                sample["metadata_categorical"] = data["metadata_categorical"]
            self._apply_metadata_dropout(sample)

        if data["roi_annotations"]:
            pts = np.zeros((MAX_ANN_POINTS, 3), dtype=np.float32)
            msk = np.zeros((MAX_ANN_POINTS,), dtype=np.uint8)
            for i, ann in enumerate(data["roi_annotations"][:MAX_ANN_POINTS]):
                pts[i] = [
                    float(ann.get("roi_z", 0.0)),
                    float(ann.get("roi_y", 0.0)),
                    float(ann.get("roi_x", 0.0)),
                ]
                msk[i] = 1
            sample["ann_points"] = pts
            sample["ann_points_valid"] = msk

        if self.transform is not None:
            sample = self.transform(sample)

        out: Dict[str, torch.Tensor] = {
            "image": torch.as_tensor(sample["image"], dtype=torch.float16),
            "vessel_label": torch.as_tensor(sample["vessel_label"], dtype=torch.int64),
            "labels": torch.as_tensor(sample["labels"], dtype=torch.float32),
            "series_uid": sample["series_uid"],  # type: ignore[assignment]
            "meta": sample["meta"],  # type: ignore[assignment]
        }

        if "ann_points" in sample:
            out["ann_points"] = torch.as_tensor(sample["ann_points"], dtype=torch.float32)
            out["ann_points_valid"] = torch.as_tensor(sample["ann_points_valid"], dtype=torch.uint8)

        if "extra_vessel_label" in sample:
            out["extra_vessel_label"] = torch.as_tensor(sample["extra_vessel_label"], dtype=torch.int64)
        # Additional extras (indexed)
        if self._num_extra_seg > 1:
            for i in range(1, self._num_extra_seg):
                key = f"extra_vessel_label_{i}"
                if key in sample:
                    out[key] = torch.as_tensor(sample[key], dtype=torch.int64)

        if self._metadata_enabled:
            if "metadata_numeric" in sample and "metadata_numeric_missing" in sample:
                out["metadata_numeric"] = torch.as_tensor(sample["metadata_numeric"], dtype=torch.float32)
                out["metadata_numeric_missing"] = torch.as_tensor(
                    sample["metadata_numeric_missing"], dtype=torch.float32
                )
            if "metadata_categorical" in sample:
                out["metadata_categorical"] = torch.as_tensor(
                    sample["metadata_categorical"], dtype=torch.int64
                )

        return out

    def _apply_metadata_dropout(self, sample: Dict[str, Union[np.ndarray, float, int, Dict]]) -> None:
        """Stochastically mask metadata as missing according to dropout probs"""

        if self.metadata_numeric_dropout_prob > 0.0:
            if "metadata_numeric" in sample and "metadata_numeric_missing" in sample:
                numeric = sample["metadata_numeric"]
                missing = sample["metadata_numeric_missing"]
                if isinstance(numeric, np.ndarray) and isinstance(missing, np.ndarray):
                    drop_mask = np.random.rand(numeric.shape[0]) < self.metadata_numeric_dropout_prob
                    if drop_mask.any():
                        # As augmentation during training, overwrite with defaults and missing flags
                        numeric = numeric.copy()
                        missing = missing.copy()
                        numeric[drop_mask] = 0.0
                        missing[drop_mask] = 1.0
                        sample["metadata_numeric"] = numeric
                        sample["metadata_numeric_missing"] = missing

        if self.metadata_categorical_dropout_prob > 0.0:
            if "metadata_categorical" in sample:
                categorical = sample["metadata_categorical"]
                if isinstance(categorical, np.ndarray):
                    drop_mask = np.random.rand(categorical.shape[0]) < self.metadata_categorical_dropout_prob
                    if drop_mask.any():
                        # Replace with the ID for the missing category token
                        categorical = categorical.copy()
                        for idx, should_drop in enumerate(drop_mask):
                            if not should_drop:
                                continue
                            feature_name = self.metadata_categorical_features[idx]
                            default_token = METADATA_CATEGORICAL_DEFAULTS[feature_name]
                            default_idx = self.metadata_categorical_vocab[feature_name][default_token]
                            categorical[idx] = default_idx
                        sample["metadata_categorical"] = categorical

    @property
    def metadata_numeric_dim(self) -> int:
        return len(self.metadata_numeric_features) if self._metadata_enabled else 0

    @property
    def metadata_categorical_dim(self) -> int:
        return len(self.metadata_categorical_features) if self._metadata_enabled else 0

    @property
    def metadata_info(self) -> Dict[str, Any]:
        if not self._metadata_enabled:
            return {}
        return {
            "numeric_features": self.metadata_numeric_features,
            "categorical_features": self.metadata_categorical_features,
            "categorical_vocab": self.metadata_categorical_vocab,
            "numeric_percentiles": METADATA_NUMERIC_PERCENTILES,
        }


def get_train_transforms(
    input_size: Tuple[int, int, int],
    keep_ratio: str = "z-xy",
    spatial_transform: str = "resize",
    pad_multiple: int = 32,
    include_extra_seg: bool = False,
    num_extra_seg: Optional[int] = None,
    version: int = 1,
) -> Compose:
    """Training-time CPU transforms for the labeled dataset"""

    # Dynamically compose keys for additional segmentations
    mask_keys = ["vessel_label"]
    _num_extra = int(num_extra_seg) if (num_extra_seg is not None) else (1 if include_extra_seg else 0)
    if _num_extra > 0:
        mask_keys.append("extra_vessel_label")
        for i in range(1, _num_extra):
            mask_keys.append(f"extra_vessel_label_{i}")
    spatial_transform = spatial_transform.lower()

    transforms: List = [
        EnsureTyped(keys=["image", "labels", "vessel_label"], allow_missing_keys=True),
    ]

    if spatial_transform == "resize":
        transforms.append(
            Resize3DFitPadCropd(
                keys=["image", *mask_keys],
                target_size=input_size,
                keep_ratio=keep_ratio,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                scale_clip=(0.5, 2.0),
            )
        )
    elif spatial_transform == "resize_xy_long_z_downonly":
        if input_size[1] != input_size[2]:
            raise ValueError(
                f"For resize_xy_long_z_downonly, XY must be square (Ht==Wt). Got={input_size[1]}x{input_size[2]}"
            )
        transforms.append(
            Resize3DXYLongSideZDownOnlyPadToSized(
                keys=["image", *mask_keys],
                target_size=input_size,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
            )
        )
    elif spatial_transform == "pad":
        transforms.append(
            Resize3DMaxSized(
                keys=["image", *mask_keys],
                max_size=input_size,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                point_valid_keys={"ann_points": "ann_points_valid"},
            )
        )
        transforms.append(
            PadToMultipleOfd(
                keys=["image", *mask_keys],
                multiple=pad_multiple,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                point_valid_keys={"ann_points": "ann_points_valid"},
            )
        )
    elif spatial_transform == "pad_to_size":
        transforms.append(
            Resize3DFitWithinPadToSized(
                keys=["image", *mask_keys],
                target_size=input_size,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                point_valid_keys={"ann_points": "ann_points_valid"},
            )
        )
    else:
        raise ValueError(f"Unknown spatial_transform specified: {spatial_transform}")

    if version == 1:
        transforms.extend(
            [
                RandGaussianSmoothd(
                    keys=["image"], prob=0.2, sigma_x=(0.2, 0.7), sigma_y=(0.2, 0.7), sigma_z=(0.2, 0.7)
                ),
                RandGaussianNoised(keys=["image"], prob=0.3, std=0.2),
                RandShiftIntensityd(keys=["image"], offsets=(-0.5, 0.5), prob=0.1),
                RandScaleIntensityd(keys=["image"], factors=(0.75, 1.25), prob=0.3),
                RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.6, 2.0)),
                RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.6, 2.0), invert_image=True),
                RandGaussianSharpend(keys=["image"], prob=0.2),
                RandInvertAroundMeanD(keys=["image"], prob=0.2),
            ]
        )
    elif version == 2:
        transforms.extend(
            [
                RandGaussianNoised(keys=["image"], prob=0.4, std=0.25),
                RandShiftIntensityd(keys=["image"], offsets=(-0.5, 0.5), prob=0.2),
                RandScaleIntensityd(keys=["image"], factors=(0.75, 1.25), prob=0.3),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.6, 2.0)),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.6, 2.0), invert_image=True),
                RandGaussianSharpend(keys=["image"], prob=0.2),
                RandInvertAroundMeanD(
                    keys=["image"], prob=0.2, p_per_channel=1.0, p_synchronize_channels=1.0
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown version specified: {version}")

    return Compose(transforms)


def get_val_transforms(
    input_size: Tuple[int, int, int],
    keep_ratio: str = "z-xy",
    spatial_transform: str = "resize",
    pad_multiple: int = 32,
    include_extra_seg: bool = False,
    num_extra_seg: Optional[int] = None,
) -> Compose:
    """Validation-time CPU transforms for the labeled dataset"""

    mask_keys = ["vessel_label"]
    _num_extra = int(num_extra_seg) if (num_extra_seg is not None) else (1 if include_extra_seg else 0)
    if _num_extra > 0:
        mask_keys.append("extra_vessel_label")
        for i in range(1, _num_extra):
            mask_keys.append(f"extra_vessel_label_{i}")
    spatial_transform = spatial_transform.lower()

    transforms: List = [EnsureTyped(keys=["image", "labels", "vessel_label"], allow_missing_keys=True)]

    if spatial_transform == "resize":
        transforms.append(
            Resize3DFitPadCropd(
                keys=["image", *mask_keys],
                target_size=input_size,
                keep_ratio=keep_ratio,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
            )
        )
    elif spatial_transform == "resize_xy_long_z_downonly":
        if input_size[1] != input_size[2]:
            raise ValueError(
                f"For resize_xy_long_z_downonly, XY must be square (Ht==Wt). Got={input_size[1]}x{input_size[2]}"
            )
        transforms.append(
            Resize3DXYLongSideZDownOnlyPadToSized(
                keys=["image", *mask_keys],
                target_size=input_size,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
            )
        )
    elif spatial_transform == "pad":
        transforms.append(
            Resize3DMaxSized(
                keys=["image", *mask_keys],
                max_size=input_size,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                point_valid_keys={"ann_points": "ann_points_valid"},
            )
        )
        transforms.append(
            PadToMultipleOfd(
                keys=["image", *mask_keys],
                multiple=pad_multiple,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                point_valid_keys={"ann_points": "ann_points_valid"},
            )
        )
    elif spatial_transform == "pad_to_size":
        transforms.append(
            Resize3DFitWithinPadToSized(
                keys=["image", *mask_keys],
                target_size=input_size,
                image_keys=("image",),
                mask_keys=tuple(mask_keys),
                point_keys=("ann_points",),
                point_valid_keys={"ann_points": "ann_points_valid"},
            )
        )
    else:
        raise ValueError(f"Unknown spatial_transform specified: {spatial_transform}")

    return Compose(transforms)
