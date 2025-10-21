"""Utilities to extract auxiliary metadata from DICOM headers."""

from __future__ import annotations

from decimal import Decimal
import re
from typing import Any, Dict, Optional

import pydicom
from pydicom.multival import MultiValue
from pydicom.valuerep import DSdecimal, DSfloat, IS

# Mapping of DICOM tags to output keys
METADATA_FIELDS = [
    ("Modality", "Modality"),
    ("Manufacturer", "Manufacturer"),
    ("SliceThickness", "SliceThickness"),
    ("SpacingBetweenSlices", "SpacingBetweenSlices"),
    ("PixelSpacing", "PixelSpacing"),
    ("ReconstructionDiameter", "ReconstructionDiameter"),
    ("ReconstructionAlgorithm", "ReconstructionAlgorithm"),
    ("KVP", "KVP"),
    ("XRayTubeCurrent", "XRayTubeCurrent"),
    ("ExposureTime", "ExposureTime"),
    ("RepetitionTime", "RepetitionTime"),
    ("EchoTime", "EchoTime"),
    ("FlipAngle", "FlipAngle"),
    ("PatientPosition", "PatientPosition"),
    ("ImageOrientationPatient", "ImageOrientationPatient"),
]

DEFAULT_SEX_PLACEHOLDER = "O"
DEFAULT_MANUFACTURER_CATEGORY = "Other"
MAJOR_MANUFACTURERS = {
    "GE MEDICAL SYSTEMS",
    "SIEMENS",
    "TOSHIBA",
    "Philips",
}

# Normalization table for manufacturer names (keys are uppercased/canonicalized)
MANUFACTURER_NORMALIZATION_MAP = {
    "GE MEDICAL SYSTEMS": "GE MEDICAL SYSTEMS",
    "GE MEDICAL SYSTEMS LLC": "GE MEDICAL SYSTEMS",
    "SIEMENS": "SIEMENS",
    "SIEMENS HEALTHINEERS": "SIEMENS",
    "SIEMENS HEALTHCARE": "SIEMENS",
    "SIEMENS HEALTHCARE GMBH": "SIEMENS",
    "SIEMENS HEALTHCARE DIAGNOSTICS": "SIEMENS",
    "SIEMENS AG": "SIEMENS",
    "TOSHIBA": "TOSHIBA",
    "TOSHIBA MEC": "TOSHIBA",
    "CANON MEC": "TOSHIBA",
    "CANON MEDICAL SYSTEMS": "TOSHIBA",
    "CANON MEDICAL SYSTEMS CORPORATION": "TOSHIBA",
    "PHILIPS": "Philips",
    "PHILIPS MEDICAL SYSTEMS": "Philips",
    "PHILIPS HEALTHCARE": "Philips",
    "PHILIPS MEDICAL SYSTEMS NEDERLAND B V": "Philips",
}

# Keyword-based fallback rules when the normalization table has no direct match
MANUFACTURER_KEYWORD_RULES = (
    ("SIEMENS", ("SIEMENS",)),
    ("TOSHIBA", ("TOSHIBA", "CANON")),
    ("GE MEDICAL SYSTEMS", ("GE MEDICAL", "GENERAL ELECTRIC", "GE HEALTHCARE")),
    ("Philips", ("PHILIPS",)),
)

# Normalization table for modality strings (keys are uppercased/canonicalized)
MODALITY_NORMALIZATION_MAP = {
    "CT": "CT",
    "CTA": "CT",
    "CT ANGIOGRAPHY": "CT",
    "MR": "MR",
    "MRI": "MR",
    "MRA": "MR",
}


def convert_age_to_years(age: str) -> Optional[float]:
    """Convert a DICOM-style age string to years as a float."""

    if not age:
        return None
    age_clean = age.strip().upper()
    if len(age_clean) < 2:
        return None
    unit = age_clean[-1]
    value_str = age_clean[:-1]
    try:
        value = float(value_str)
    except ValueError:
        return None

    if unit == "Y":
        factor = 1.0
    elif unit == "M":
        factor = 1.0 / 12.0
    elif unit == "W":
        factor = 1.0 / 52.0
    elif unit == "D":
        factor = 1.0 / 365.0
    else:
        return None

    return value * factor


def _to_serializable(value: Any) -> Optional[Any]:
    """Normalize a value to a JSON-serializable representation."""

    if value is None:
        return None

    if isinstance(value, MultiValue):
        normalized = [_to_serializable(v) for v in value]
        normalized = [v for v in normalized if v is not None]
        return normalized if normalized else None

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = value.decode("latin-1", "ignore").strip()
        return text or None

    if isinstance(value, (DSfloat, DSdecimal, float)):
        try:
            return float(value)
        except Exception:
            return None

    if isinstance(value, (IS, int)):
        try:
            return int(str(value))
        except Exception:
            return None

    if isinstance(value, Decimal):
        return float(value)

    text = str(value).strip()
    return text or None


def _normalize_manufacturer(value: Optional[Any]) -> str:
    """Normalize a manufacturer string to predefined categories."""

    if isinstance(value, list):
        manufacturer_value = str(value[0]).strip() if value else ""
    else:
        manufacturer_value = str(value).strip() if value is not None else ""

    canonical = _canonicalize_manufacturer(manufacturer_value)
    if canonical is not None:
        return canonical
    return DEFAULT_MANUFACTURER_CATEGORY


def _canonicalize_manufacturer(raw_value: str) -> Optional[str]:
    """Map manufacturer names to major categories using the normalization table."""

    if not raw_value:
        return None

    key = _sanitize_manufacturer_key(raw_value)
    normalized = MANUFACTURER_NORMALIZATION_MAP.get(key)
    if normalized is not None:
        return normalized

    return _match_manufacturer_keywords(key)


def _sanitize_manufacturer_key(raw_value: str) -> str:
    """Sanitize manufacturer text to a canonical lookup key."""

    upper = raw_value.strip().upper()
    replaced = re.sub(r"[\s\-_]+", " ", upper)
    cleaned = re.sub(r"[,.;:]+", "", replaced)
    return re.sub(r"\s+", " ", cleaned).strip()


def _match_manufacturer_keywords(key: str) -> Optional[str]:
    """Infer a major manufacturer category using keyword heuristics."""

    for canonical, keywords in MANUFACTURER_KEYWORD_RULES:
        if any(keyword in key for keyword in keywords):
            return canonical
    return None


def _normalize_modality(value: Optional[Any]) -> Optional[str]:
    """Normalize a modality string to known categories when possible."""

    if isinstance(value, list):
        modality_value = str(value[0]).strip() if value else ""
    else:
        modality_value = str(value).strip() if value is not None else ""

    if not modality_value:
        return None

    canonical = _canonicalize_modality(modality_value)
    if canonical is not None:
        return canonical
    return modality_value


def _canonicalize_modality(raw_value: str) -> Optional[str]:
    """Map modality strings using the normalization table."""

    if not raw_value:
        return None

    key = _sanitize_modality_key(raw_value)
    return MODALITY_NORMALIZATION_MAP.get(key)


def _sanitize_modality_key(raw_value: str) -> str:
    """Sanitize modality text to a canonical lookup key."""

    upper = raw_value.strip().upper()
    return re.sub(r"[\s\-_]+", " ", upper)


def extract_metadata_from_dataset(
    ds: pydicom.Dataset,
    *,
    metadata_version: int,
) -> Dict[str, Any]:
    """Build an auxiliary metadata dict from a DICOM dataset."""

    metadata: Dict[str, Any] = {}

    age_raw = ds.get("PatientAge", None)
    if age_raw:
        age_text = str(age_raw).strip()
        if age_text:
            metadata["PatientAge"] = age_text
            age_years = convert_age_to_years(age_text)
            if age_years is not None:
                metadata["PatientAgeYears"] = age_years

    sex_raw = ds.get("PatientSex", None)
    sex_text = str(sex_raw).strip() if sex_raw else ""
    if sex_text:
        metadata["PatientSex"] = sex_text
    else:
        metadata["PatientSex"] = DEFAULT_SEX_PLACEHOLDER

    for out_key, dicom_key in METADATA_FIELDS:
        value = ds.get(dicom_key, None)
        normalized = _to_serializable(value)

        if out_key == "Modality":
            modality_value = _normalize_modality(normalized)
            if modality_value is None:
                continue
            metadata[out_key] = modality_value
            continue

        if out_key == "Manufacturer":
            metadata[out_key] = _normalize_manufacturer(normalized)
            continue

        if normalized is None:
            continue
        metadata[out_key] = normalized

    if "Manufacturer" not in metadata:
        metadata["Manufacturer"] = DEFAULT_MANUFACTURER_CATEGORY

    metadata["MetadataSource"] = "dicom_header"
    metadata["MetadataVersion"] = metadata_version

    return metadata


__all__ = [
    "DEFAULT_MANUFACTURER_CATEGORY",
    "DEFAULT_SEX_PLACEHOLDER",
    "MAJOR_MANUFACTURERS",
    "METADATA_FIELDS",
    "convert_age_to_years",
    "extract_metadata_from_dataset",
]
