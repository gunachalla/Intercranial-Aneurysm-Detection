#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organized loader implementation for the RSNA dataset.

Extract required utilities from rsna_utils.py and reimplement them with
GeometricDicomLoader in a cleaner architecture.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import SimpleITK as sitk

from .dicom_loader import GeometricDicomLoader
from .rsna_utils import SphereMaskGenerator


@dataclass
class DicomAnnotationResult:
    """Container for loaded DICOM data and annotations."""

    volume_data: Optional[np.ndarray]
    sop_to_slice_map: Optional[Dict[str, int]]
    voxel_spacing: Optional[Tuple[float, float, float]]
    case_info: Dict
    annotations_3d: Optional[List[Dict]]
    sphere_masks: Optional[np.ndarray]
    vessel_segmentation: Optional[np.ndarray]
    series_uid: Optional[str] = None
    slice_info: Optional[Dict] = None

    def __getitem__(self, key):
        """Allow dict-style access."""
        return getattr(self, key)

    def get(self, key, default=None):
        """Allow dict-style access with default."""
        return getattr(self, key, default)

    def keys(self):
        """Expose keys for dict-style access."""
        return [
            "volume_data",
            "sop_to_slice_map",
            "voxel_spacing",
            "case_info",
            "annotations_3d",
            "sphere_masks",
            "vessel_segmentation",
            "series_uid",
            "slice_info",
        ]


class RSNALoader:
    """
    Integrated loader for RSNA dataset.
    Loads DICOM data via GeometricDicomLoader and integrates annotations.
    """

    def __init__(self, data_dir: str, default_mask_label: int = 14):
        """
        Initialize the loader.

        Args:
            data_dir (str): Path to the dataset root directory
            default_mask_label (int): Default mask label value
        """
        self.data_dir = Path(data_dir)
        self.series_dir = self.data_dir / "series"
        self.segmentation_dir = self.data_dir / "segmentations"
        self.train_csv = self.data_dir / "train.csv"
        self.localizers_csv = self.data_dir / "train_localizers.csv"

        # Use GeometricDicomLoader
        self.dicom_loader = GeometricDicomLoader(
            data_dir=str(data_dir), exclude_localizer=True, default_mask_label=default_mask_label
        )

        # Init SphereMaskGenerator
        self.mask_generator = SphereMaskGenerator(default_mask_label)

        # Load metadata
        self.metadata = None
        if self.train_csv.exists():
            self.metadata = pd.read_csv(self.train_csv)
            print(f"Loaded metadata: {len(self.metadata)} rows")

        # Load annotation positions
        self.localizers = None
        if self.localizers_csv.exists():
            self.localizers = pd.read_csv(self.localizers_csv)
            print(f"Loaded localizer annotations: {len(self.localizers)} rows")

    def get_case_info(self, series_uid: str) -> Dict:
        """
        Retrieve case info for a given series.

        Args:
            series_uid (str): SeriesInstanceUID

        Returns:
            Dict: Case info
        """
        if self.metadata is None:
            return {"series_uid": series_uid}

        case_info = self.metadata[self.metadata["SeriesInstanceUID"] == series_uid]
        if len(case_info) == 0:
            return {"series_uid": series_uid}

        return case_info.iloc[0].to_dict()

    def get_annotations_3d(
        self,
        series_uid: str,
        sop_to_slice_map: Dict[str, int],
        volume_shape: Tuple[int, int, int],
        slice_info: Optional[Dict],
    ) -> List[Dict]:
        """
        Fetch annotations and convert them directly to 3D coordinates.

        Args:
            series_uid (str): SeriesInstanceUID
            sop_to_slice_map: SOPInstanceUID -> slice index map
            volume_shape: Volume shape (Z, Y, X)
            slice_info: DICOM slice info (including coordinate transform info)

        Returns:
            List[Dict]: List of annotations with 3D coordinates
        """
        if self.localizers is None:
            return []

        annotations = self.localizers[self.localizers["SeriesInstanceUID"] == series_uid]
        if len(annotations) == 0:
            return []

        annotations_3d = []
        z_max, y_max, x_max = volume_shape

        for _, ann in annotations.iterrows():
            try:
                # Parse coordinates
                coords_str = str(ann["coordinates"]).strip()

                if coords_str.startswith("{") and coords_str.endswith("}"):
                    # Parse JSON/dict-style coordinates
                    import ast

                    coords_dict = ast.literal_eval(coords_str)
                    x_2d = float(coords_dict["x"])
                    y_2d = float(coords_dict["y"])
                elif "," in coords_str:
                    # Comma-separated format
                    coords_str = (
                        coords_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                    )
                    x_str, y_str = coords_str.split(",")
                    x_2d = float(x_str.strip())
                    y_2d = float(y_str.strip())
                else:
                    print(f"Coordinate parse error: {coords_str}")
                    continue

                # Get slice index from SOPInstanceUID
                sop_uid = ann["SOPInstanceUID"]
                if sop_uid not in sop_to_slice_map:
                    print(f"Warning: SOPInstanceUID {sop_uid} not found in volume")
                    continue

                slice_idx = sop_to_slice_map[sop_uid]

                # Apply coordinate transform
                if "_coordinate_transform" in slice_info:
                    z_3d, y_3d, x_3d = self._transform_coordinates_to_lps(x_2d, y_2d, slice_idx, slice_info)
                else:
                    # Fallback without transform information
                    z_3d, y_3d, x_3d = slice_idx, int(y_2d), int(x_2d)

                # Check boundary of coordinates
                if not (0 <= z_3d < z_max):
                    print(f"Warning: Z coordinate out of range: {z_3d} (0-{z_max-1})")
                    continue
                if not (0 <= x_3d < x_max) or not (0 <= y_3d < y_max):
                    print(f"Warning: XY coordinates out of range: ({x_3d}, {y_3d}) (0-{x_max-1}, 0-{y_max-1})")
                    continue

                # Append 3D annotation
                annotations_3d.append(
                    {
                        "sop_uid": sop_uid,
                        "x_2d": x_2d,
                        "y_2d": y_2d,
                        "z": z_3d,
                        "y": y_3d,
                        "x": x_3d,
                        "location": ann.get("location", ""),
                        "coordinates_raw": ann["coordinates"],
                    }
                )

            except Exception as e:
                print(f"Annotation parse error: {e}, coordinates: {ann['coordinates']}")

        return annotations_3d

    def generate_sphere_masks(
        self,
        annotations_3d: List[Dict],
        volume_shape: Tuple[int, int, int],
        voxel_spacing: Tuple[float, float, float],
        radius_mm: float = 3.0,
        mask_label: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate spherical masks from annotations.

        Args:
            annotations_3d: Annotations with 3D coordinates
            volume_shape: Volume shape (Z, Y, X)
            voxel_spacing: Voxel spacing (Z, Y, X)
            radius_mm: Sphere radius (mm)
            mask_label: Mask label value

        Returns:
            np.ndarray: Spherical mask
        """
        combined_mask = np.zeros(volume_shape, dtype=np.uint8)

        for ann in annotations_3d:
            center = (ann["z"], ann["y"], ann["x"])
            sphere_mask = self.mask_generator.create_sphere_mask(
                center=center,
                radius_mm=radius_mm,
                volume_shape=volume_shape,
                voxel_spacing=voxel_spacing,
                label=mask_label,
            )

            # Merge masks (logical OR)
            combined_mask = np.logical_or(combined_mask, sphere_mask).astype(np.uint8)
            if mask_label is not None:
                combined_mask[combined_mask > 0] = mask_label

        return combined_mask

    def load_vessel_segmentation(self, series_uid: str) -> Optional[np.ndarray]:
        """
        Load vessel segmentation (_cowseg.nii) and convert to LPS orientation.

        Args:
            series_uid (str): SeriesInstanceUID

        Returns:
            Optional[np.ndarray]: Vessel segmentation (Z,Y,X) or None if not found
        """
        seg_path = self.segmentation_dir / series_uid / f"{series_uid}_cowseg.nii"

        if not seg_path.exists():
            print(f"Vessel segmentation not found: {seg_path}")
            return None

        try:
            print(f"Loading vessel segmentation: {seg_path}")
            seg_nii = nib.load(str(seg_path))
            seg_data, _ = reorient_nifti_to_lps(seg_nii)
            print(f"Loaded vessel segmentation: {seg_data.shape}")
            return seg_data.astype(np.uint8)
        except Exception as e:
            print(f"Error loading vessel segmentation: {e}")
            return None

    def load_dicom_with_annotations(
        self,
        series_uid: str,
        generate_masks: bool = False,
        radius_mm: float = 3.0,
        mask_label: Optional[int] = None,
        load_vessel_segmentation: bool = True,
    ) -> DicomAnnotationResult:
        """
        Load a DICOM series and annotations together.

        Args:
            series_uid (str): SeriesInstanceUID
            generate_masks (bool): Whether to generate spherical masks
            radius_mm (float): Sphere radius (mm)
            mask_label (int, optional): Mask label value
            load_vessel_segmentation (bool): Whether to load vessel segmentation

        Returns:
            DicomAnnotationResult: Combined load result
        """
        # 1. Load DICOM volume data
        volume_data, sop_to_slice_map, voxel_spacing, slice_info = self.dicom_loader.load_dicom_series(
            series_uid, convert_to_lps=True
        )

        if volume_data is None:
            return DicomAnnotationResult(
                volume_data=None,
                sop_to_slice_map=None,
                voxel_spacing=None,
                case_info=self.get_case_info(series_uid),
                annotations_3d=None,
                sphere_masks=None,
                vessel_segmentation=None,
                series_uid=series_uid,
                slice_info=None,
            )

        # 2. Get case info
        case_info = self.get_case_info(series_uid)

        # 3. Fetch annotations and convert to 3D
        annotations_3d = self.get_annotations_3d(series_uid, sop_to_slice_map, volume_data.shape, slice_info)

        # 4. Generate spherical masks (if requested)
        sphere_masks = None
        if generate_masks and annotations_3d:
            sphere_masks = self.generate_sphere_masks(
                annotations_3d=annotations_3d,
                volume_shape=volume_data.shape,
                voxel_spacing=voxel_spacing,
                radius_mm=radius_mm,
                mask_label=mask_label,
            )

        # 5. Load vessel segmentation (if requested)
        vessel_segmentation = None
        if load_vessel_segmentation:
            vessel_segmentation = self.load_vessel_segmentation(series_uid)

        return DicomAnnotationResult(
            volume_data=volume_data,
            sop_to_slice_map=sop_to_slice_map,
            voxel_spacing=voxel_spacing,
            case_info=case_info,
            annotations_3d=annotations_3d,
            sphere_masks=sphere_masks,
            vessel_segmentation=vessel_segmentation,
            series_uid=series_uid,
            slice_info=slice_info,
        )

    def get_short_location_name(self, location: str) -> str:
        """
        Convert anatomical location names to short tokens.

        Args:
            location (str): Full location name

        Returns:
            str: Shortened token
        """
        short_names = {
            "Left Infraclinoid Internal Carotid Artery": "L_ICA_IC",
            "Right Infraclinoid Internal Carotid Artery": "R_ICA_IC",
            "Left Supraclinoid Internal Carotid Artery": "L_ICA_SC",
            "Right Supraclinoid Internal Carotid Artery": "R_ICA_SC",
            "Left Middle Cerebral Artery": "L_MCA",
            "Right Middle Cerebral Artery": "R_MCA",
            "Left Anterior Cerebral Artery": "L_ACA",
            "Right Anterior Cerebral Artery": "R_ACA",
            "Anterior Communicating Artery": "AComA",
            "Left Posterior Communicating Artery": "L_PComA",
            "Right Posterior Communicating Artery": "R_PComA",
            "Basilar Tip": "Basilar",
            "Other Posterior Circulation": "Other_PC",
        }
        return short_names.get(location, location[:8])

    def _transform_coordinates_to_lps(
        self,
        x_2d: float,
        y_2d: float,
        slice_idx: int,
        slice_info: Dict,
    ) -> Tuple[int, int, int]:
        """
        Convert 2D coordinates in the original space to 3D LPS coordinates.
        Uses SimpleITK coordinate transforms.

        Args:
            x_2d, y_2d: Original 2D image coordinates (pixels)
            slice_idx: Original slice index
            slice_info: DICOM slice info (with transform data)
            sop_uid: SOPInstanceUID

        Returns:
            Tuple[int, int, int]: (z, y, x) coordinates in LPS space
        """
        # Get transform info
        transform_info = slice_info.get("_coordinate_transform", {})
        if not transform_info or not transform_info.get("converted_to_lps", False):
            # If not converted to LPS, return as-is
            return slice_idx, int(y_2d), int(x_2d)

        try:

            # Original image space info
            original_direction = transform_info["original_direction"]
            original_origin = transform_info["original_origin"]
            original_spacing = transform_info["original_spacing"]

            # LPS image space info
            lps_direction = transform_info["lps_direction"]
            lps_origin = transform_info["lps_origin"]
            lps_spacing = transform_info["lps_spacing"]

            # Original image
            original_img = sitk.Image(transform_info["original_size"], sitk.sitkFloat32)
            original_img.SetDirection(original_direction)
            original_img.SetOrigin(original_origin)
            original_img.SetSpacing(original_spacing)

            # LPS image
            lps_img = sitk.Image(transform_info["lps_size"], sitk.sitkFloat32)
            lps_img.SetDirection(lps_direction)
            lps_img.SetOrigin(lps_origin)
            lps_img.SetSpacing(lps_spacing)

            # 3D index in the original image
            # Note: SimpleITK uses (x, y, z) while numpy uses (z, y, x)
            index_3d = [int(x_2d), int(y_2d), slice_idx]

            # Index -> physical point
            physical_point = original_img.TransformIndexToPhysicalPoint(index_3d)

            # Physical point -> index in LPS image
            index_lps = lps_img.TransformPhysicalPointToIndex(physical_point)

            # Convert (X,Y,Z) from SimpleITK to (Z,Y,X) for numpy
            return index_lps[2], index_lps[1], index_lps[0]

        except Exception as e:
            print(f"Coordinate transform error: {e}")
            # On error, return original coordinates
            return slice_idx, int(y_2d), int(x_2d)
