#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DicomLoader with strict geometric filtering.

Implements majority-vote slice filtering:
- Group by geometric metadata (Rows, Cols, PixelSpacing, ImageOrientation)
- Keep only the subgroup with the largest number of effective slices
- Automatically exclude minority subgroups
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pydicom
import SimpleITK as sitk


class GeometricDicomLoader:
    """DicomLoader that applies geometric filtering."""

    def __init__(self, data_dir: str, exclude_localizer: bool = True, default_mask_label: int = 14):
        """
        Initialize the loader.

        Args:
            data_dir (str): Path to the dataset root directory
            exclude_localizer (bool): Whether to exclude LOCALIZER images
            default_mask_label (int): Default mask label value
        """
        self.data_dir = Path(data_dir)
        self.series_dir = self.data_dir / "series"
        self.exclude_localizer = exclude_localizer
        self.default_mask_label = default_mask_label

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def _geo_signature(self, ds) -> Tuple[str, str, str, str]:
        """
        Extract a geometric signature from a DICOM dataset.

        Args:
            ds: pydicom.Dataset

        Returns:
            Tuple: (Rows, Cols, PixelSpacing, ImageOrientationPatient)
        """
        rows = str(getattr(ds, "Rows", ""))
        cols = str(getattr(ds, "Columns", ""))

        # Get PixelSpacing as a string
        pixel_spacing = getattr(ds, "PixelSpacing", None)
        if pixel_spacing is not None:
            if hasattr(pixel_spacing, "__iter__") and len(pixel_spacing) >= 2:
                pixel_spacing_str = f"{pixel_spacing[0]}\\{pixel_spacing[1]}"
            else:
                pixel_spacing_str = str(pixel_spacing)
        else:
            pixel_spacing_str = ""

        # Get ImageOrientationPatient as a string
        orientation = getattr(ds, "ImageOrientationPatient", None)
        if orientation is not None and hasattr(orientation, "__iter__"):
            orientation_str = "\\".join(str(x) for x in orientation)
        else:
            orientation_str = str(orientation) if orientation else ""

        return (rows, cols, pixel_spacing_str, orientation_str)

    def _looks_localizer(self, ds) -> bool:
        """
        Heuristically determine if a slice is a LOCALIZER.

        Args:
            ds: pydicom.Dataset

        Returns:
            bool: True if LOCALIZER
        """
        image_type = getattr(ds, "ImageType", "")
        if isinstance(image_type, (list, tuple)):
            image_type = "\\".join(str(x) for x in image_type)
        return "LOCALIZER" in str(image_type).upper()

    def _effective_counts(self, datasets: List[pydicom.Dataset]) -> Tuple[int, int, int]:
        """
        Compute effective slice count and tie-break info.

        Args:
            datasets: List of DICOM datasets

        Returns:
            Tuple: (effective_slices, singleframe_count, pixel_count)
        """
        eff = 0
        sf = 0

        for ds in datasets:
            num_frames = getattr(ds, "NumberOfFrames", 1)
            if isinstance(num_frames, str) and num_frames.isdigit():
                num_frames = int(num_frames)
            elif not isinstance(num_frames, int):
                num_frames = 1

            if num_frames > 1:
                eff += num_frames
            else:
                eff += 1
                sf += 1

        # Compute pixel count (Rows*Columns) for tie-breaking
        try:
            if datasets:
                rows = int(getattr(datasets[0], "Rows", 0))
                cols = int(getattr(datasets[0], "Columns", 0))
                pix = rows * cols
            else:
                pix = 0
        except (ValueError, TypeError):
            pix = 0

        return eff, sf, pix

    def _discover_geometric_groups(
        self, series_uid: str
    ) -> Dict[Tuple[str, str, str, str], List[pydicom.Dataset]]:
        """
        Group a DICOM series into geometric subgroups.

        Args:
            series_uid (str): SeriesInstanceUID

        Returns:
            Dict: geometric signature -> list of DICOM datasets
        """
        series_path = self.series_dir / series_uid
        if not series_path.exists():
            self.logger.warning(f"Series not found: {series_uid}")
            return {}

        dicom_files = list(series_path.glob("*.dcm"))
        if not dicom_files:
            self.logger.warning(f"No DICOM files found: {series_uid}")
            return {}

        self.logger.info(f"Analyzing DICOM files: {len(dicom_files)}")

        geometric_groups = {}

        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)

                # Keep file path on the dataset for later path resolution
                ds.filename = str(dcm_file)

                # Exclude LOCALIZER if configured
                if self.exclude_localizer and self._looks_localizer(ds):
                    self.logger.info(f"Skipping LOCALIZER: {dcm_file.name}")
                    continue

                # Compute geometric signature
                geo_sig = self._geo_signature(ds)

                if geo_sig not in geometric_groups:
                    geometric_groups[geo_sig] = []

                geometric_groups[geo_sig].append(ds)

            except Exception as e:
                self.logger.warning(f"Failed to read DICOM file {dcm_file.name}: {e}")
                continue

        self.logger.info(f"Number of discovered subgroups: {len(geometric_groups)}")
        for geo_sig, datasets in geometric_groups.items():
            self.logger.info(f"  Subgroup {geo_sig}: {len(datasets)} files")

        return geometric_groups

    def _select_majority_group(
        self, geometric_groups: Dict[Tuple[str, str, str, str], List[pydicom.Dataset]]
    ) -> Optional[List[pydicom.Dataset]]:
        """
        Select the majority subgroup.

        Args:
            geometric_groups: Dict of geometric subgroups

        Returns:
            Optional[List[pydicom.Dataset]]: Selected dataset list
        """
        if not geometric_groups:
            return None

        majority_geo = None
        majority_stats = (-1, -1, -1)  # (effective_slices, singleframe_count, pixel_count)

        for geo_sig, datasets in geometric_groups.items():
            stats = self._effective_counts(datasets)
            self.logger.info(
                f"Subgroup {geo_sig}: effective_slices={stats[0]}, SF={stats[1]}, pixels={stats[2]}"
            )

            if stats > majority_stats:
                majority_geo = geo_sig
                majority_stats = stats

        if majority_geo is None:
            self.logger.warning("No valid subgroup found")
            return None

        # Log selection/exclusion
        self.logger.info(f"Selected subgroup: {majority_geo} (effective_slices={majority_stats[0]})")
        for geo_sig, datasets in geometric_groups.items():
            if geo_sig == majority_geo:
                self.logger.info(f"  -> selected: {geo_sig} ({len(datasets)})")
            else:
                self.logger.info(f"  -> excluded: {geo_sig} ({len(datasets)})")

        return geometric_groups[majority_geo]

    def _sort_slices_robust(self, slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
        """
        Robustly sort slices by Z-position and InstanceNumber.

        Args:
            slices: List of DICOM slices

        Returns:
            List[pydicom.Dataset]: Sorted slices
        """

        def get_z_position(ds):
            """Get Z coordinate."""
            image_pos = getattr(ds, "ImagePositionPatient", None)
            if image_pos and len(image_pos) >= 3:
                return float(image_pos[2])
            return 0.0

        def get_instance_number(ds):
            """Get InstanceNumber."""
            instance_num = getattr(ds, "InstanceNumber", 0)
            return (
                int(instance_num)
                if isinstance(instance_num, (int, str)) and str(instance_num).isdigit()
                else 0
            )

        # Sort by Z position and InstanceNumber
        try:
            sorted_slices = sorted(slices, key=lambda s: (get_z_position(s), get_instance_number(s)))
            self.logger.info(f"Slice sorting done: {len(sorted_slices)} slices")
            return sorted_slices
        except Exception as e:
            self.logger.warning(f"Slice sorting error: {e}")
            return slices

    def _create_volume_from_slices(
        self, slices: List[pydicom.Dataset]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        """
        Build a 3D volume from DICOM slices.

        Args:
            slices: Sorted DICOM slices

        Returns:
            Tuple: (3D volume, voxel_spacing)
        """
        if not slices:
            return None, None

        try:
            # Get basic info from the first slice
            first_slice = slices[0]

            # Collect pixel data from all slices
            pixel_arrays = []
            for slice_ds in slices:
                pixel_data = slice_ds.pixel_array
                if len(pixel_data.shape) == 2:
                    pixel_arrays.append(pixel_data)
                else:
                    self.logger.warning(f"Unexpected pixel data shape: {pixel_data.shape}")
                    continue

            if not pixel_arrays:
                self.logger.error("No valid pixel data found")
                return None, None

            # Stack into a 3D volume
            volume = np.stack(pixel_arrays, axis=0)
            self.logger.info(f"3D volume created: {volume.shape}")

            # Compute voxel spacing
            pixel_spacing = getattr(first_slice, "PixelSpacing", [1.0, 1.0])
            if len(pixel_spacing) >= 2:
                x_spacing = float(pixel_spacing[1])  # Column spacing
                y_spacing = float(pixel_spacing[0])  # Row spacing
            else:
                x_spacing = y_spacing = 1.0

            # Compute Z spacing
            z_spacing = 1.0
            if len(slices) > 1:
                try:
                    pos1 = getattr(slices[0], "ImagePositionPatient", None)
                    pos2 = getattr(slices[1], "ImagePositionPatient", None)
                    if pos1 and pos2 and len(pos1) >= 3 and len(pos2) >= 3:
                        z_spacing = abs(float(pos2[2]) - float(pos1[2]))
                        if z_spacing < 1e-6:  # Near zero: fallback to SliceThickness
                            slice_thickness = getattr(first_slice, "SliceThickness", None)
                            if slice_thickness:
                                z_spacing = float(slice_thickness)
                            else:
                                z_spacing = 1.0
                except (ValueError, TypeError, IndexError):
                    z_spacing = 1.0

            voxel_spacing = (z_spacing, y_spacing, x_spacing)
            self.logger.info(f"Voxel spacing: {voxel_spacing}")

            return volume, voxel_spacing

        except Exception as e:
            self.logger.error(f"Volume creation error: {e}")
            return None, None

    def load_dicom_series(
        self, series_uid: str, convert_to_lps: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Tuple], Optional[Dict]]:
        """
        Load a DICOM series and convert it to volume data, with majority filtering (SimpleITK-based).

        Args:
            series_uid (str): SeriesInstanceUID
            convert_to_lps (bool): Whether to convert to LPS coordinate system

        Returns:
            Tuple: (3D volume data, SOPInstanceUID->slice index map,
                   voxel_spacing, DICOM slice info)
        """
        self.logger.info(f"Starting series load: {series_uid}")

        # 1. Obtain file paths after majority filtering
        majority_file_paths = self.get_majority_filtered_file_paths(series_uid)
        if not majority_file_paths:
            return None, None, None, None

        try:
            # 2. Read DICOM series via SimpleITK
            reader = sitk.ImageSeriesReader()
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            reader.SetFileNames(majority_file_paths)

            sitk_img = reader.Execute()

            # Save original size
            original_size = sitk_img.GetSize()
            if len(original_size) == 4:
                if original_size[-1] == 1:
                    original_size = original_size[:-1]
                else:
                    raise ValueError(f"Last axis of 4D data is not 1: {original_size}")

            # Save original direction/origin/spacing
            original_direction = sitk_img.GetDirection()
            original_origin = sitk_img.GetOrigin()
            original_spacing = sitk_img.GetSpacing()

            # Reorient to LPS+ with SimpleITK
            if convert_to_lps:
                sitk_img_lps = sitk.DICOMOrient(sitk_img, "LPS")
                # Save direction/origin after LPS reorientation
                lps_direction = sitk_img_lps.GetDirection()
                lps_origin = sitk_img_lps.GetOrigin()
                sitk_img = sitk_img_lps

            # 3. Convert SimpleITK image to numpy array
            volume = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X)
            if volume.ndim == 4:
                if volume.shape[0] == 1:
                    volume = volume[0, ...]
                else:
                    raise ValueError(f"First axis of 4D data is not 1: {volume.shape}")
            sitk_spacing = sitk_img.GetSpacing()  # (X, Y, Z)

            # Convert voxel spacing to (Z, Y, X)
            voxel_spacing = (sitk_spacing[2], sitk_spacing[1], sitk_spacing[0])

            self.logger.info(f"Loaded via SimpleITK: shape={volume.shape}, voxel_spacing={voxel_spacing}")

            # 4. Build SOPInstanceUID -> slice index map
            sop_to_slice_map = {}
            slice_info = {}

            for i, file_path in enumerate(majority_file_paths):
                try:
                    # Read metadata only
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    sop_uid = getattr(ds, "SOPInstanceUID", f"unknown_{i}")
                    sop_to_slice_map[sop_uid] = i

                    # Save per-slice info
                    slice_info[sop_uid] = {
                        "PatientAge": getattr(ds, "PatientAge", None),
                        "PatientSex": getattr(ds, "PatientSex", None),
                        "Modality": getattr(ds, "Modality", None),
                        "StudyDescription": getattr(ds, "StudyDescription", None),
                        "SeriesDescription": getattr(ds, "SeriesDescription", None),
                        "SliceThickness": getattr(ds, "SliceThickness", None),
                        "ImagePositionPatient": getattr(ds, "ImagePositionPatient", None),
                        "ImageOrientationPatient": getattr(ds, "ImageOrientationPatient", None),
                        "PixelSpacing": getattr(ds, "PixelSpacing", None),
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata {file_path}: {e}")
                    sop_to_slice_map[f"unknown_{i}"] = i

            self.logger.info(
                f"Load complete: volume_shape={volume.shape}, SOP map entries={len(sop_to_slice_map)}"
            )

            # Attach coordinate transform info
            if convert_to_lps:
                slice_info["_coordinate_transform"] = {
                    "original_direction": original_direction,
                    "original_origin": original_origin,
                    "original_spacing": original_spacing,
                    "lps_direction": lps_direction,
                    "lps_origin": lps_origin,
                    "lps_spacing": sitk_spacing,
                    "original_size": original_size,
                    "lps_size": sitk_img_lps.GetSize(),
                    "converted_to_lps": True,
                }
            else:
                slice_info["_coordinate_transform"] = {
                    "original_direction": original_direction,
                    "original_origin": original_origin,
                    "original_spacing": original_spacing,
                    "original_size": original_size,
                    "converted_to_lps": False,
                }

            return volume, sop_to_slice_map, voxel_spacing, slice_info

        except Exception as e:
            self.logger.error(f"SimpleITK read error: {e}")
            self.logger.info("Retrying with legacy loader")
            return self._load_dicom_series_legacy(series_uid)

    # def _load_dicom_series_legacy(
    #     self, series_uid: str
    # ) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Tuple], Optional[Dict]]:
    #     """
    #     Legacy pydicom-based loader (fallback).

    #     Args:
    #         series_uid (str): SeriesInstanceUID

    #     Returns:
    #         Tuple: (3D volume, SOPInstanceUID->slice index map,
    #                voxel_spacing, DICOM slice info)
    #     """
    #     self.logger.info(f"Loading series with legacy path: {series_uid}")

    #     # 1. Discover geometric groups
    #     geometric_groups = self._discover_geometric_groups(series_uid)
    #     if not geometric_groups:
    #         return None, None, None, None

    #     # 2. Select majority subgroup
    #     selected_datasets = self._select_majority_group(geometric_groups)
    #     if not selected_datasets:
    #         return None, None, None, None

    #     # 3. Sort slices
    #     sorted_slices = self._sort_slices_robust(selected_datasets)

    #     # 4. Build 3D volume
    #     volume, voxel_spacing = self._create_volume_from_slices(sorted_slices)
    #     if volume is None:
    #         return None, None, None, None

    #     # 5. Build SOPInstanceUID -> slice index mapping
    #     sop_to_slice_map = {}
    #     for i, slice_ds in enumerate(sorted_slices):
    #         sop_uid = getattr(slice_ds, "SOPInstanceUID", f"unknown_{i}")
    #         sop_to_slice_map[sop_uid] = i

    #     # 6. Build per-slice info dict
    #     slice_info = {}
    #     for slice_ds in sorted_slices:
    #         sop_uid = getattr(slice_ds, "SOPInstanceUID", None)
    #         if sop_uid:
    #             slice_info[sop_uid] = {
    #                 "PatientAge": getattr(slice_ds, "PatientAge", None),
    #                 "PatientSex": getattr(slice_ds, "PatientSex", None),
    #                 "Modality": getattr(slice_ds, "Modality", None),
    #                 "StudyDescription": getattr(slice_ds, "StudyDescription", None),
    #                 "SeriesDescription": getattr(slice_ds, "SeriesDescription", None),
    #                 "SliceThickness": getattr(slice_ds, "SliceThickness", None),
    #                 "ImagePositionPatient": getattr(slice_ds, "ImagePositionPatient", None),
    #                 "ImageOrientationPatient": getattr(slice_ds, "ImageOrientationPatient", None),
    #                 "PixelSpacing": getattr(slice_ds, "PixelSpacing", None),
    #             }

    #     self.logger.info(
    #         f"Loaded via legacy path: volume_shape={volume.shape}, SOP_map_entries={len(sop_to_slice_map)}"
    #     )

    #     return volume, sop_to_slice_map, voxel_spacing, slice_info

    def get_majority_filtered_file_paths(self, series_uid: str) -> Optional[List[str]]:
        """
        Apply majority filtering and return the sorted list of DICOM file paths.

        Args:
            series_uid (str): SeriesInstanceUID

        Returns:
            Optional[List[str]]: Sorted DICOM file paths (majority only)
        """
        self.logger.info(f"Starting majority filtering: {series_uid}")

        # 1. Discover geometric groups
        geometric_groups = self._discover_geometric_groups(series_uid)
        if not geometric_groups:
            return None

        # 2. Select majority subgroup
        selected_datasets = self._select_majority_group(geometric_groups)
        if not selected_datasets:
            return None

        # 3. Sort slices
        sorted_slices = self._sort_slices_robust(selected_datasets)

        # 4. Build the list of file paths
        file_paths = []
        for slice_ds in sorted_slices:
            # Get the file path from the DICOM dataset (Dataset.filename)
            if hasattr(slice_ds, "filename") and slice_ds.filename:
                file_paths.append(str(slice_ds.filename))
            else:
                # If filename is missing, try to deduce it from SOPInstanceUID
                sop_uid = getattr(slice_ds, "SOPInstanceUID", None)
                if sop_uid:
                    series_path = self.series_dir / series_uid
                    dcm_file = series_path / f"{sop_uid}.dcm"
                    if dcm_file.exists():
                        file_paths.append(str(dcm_file))
                    else:
                        # As a fallback, find the file that contains the same SOPInstanceUID
                        for dcm_file in series_path.glob("*.dcm"):
                            try:
                                temp_ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                                if getattr(temp_ds, "SOPInstanceUID", None) == sop_uid:
                                    file_paths.append(str(dcm_file))
                                    break
                            except Exception:
                                continue

        if not file_paths:
            self.logger.warning(f"Failed to retrieve file paths: {series_uid}")
            return None

        self.logger.info(f"Majority filtering done: {len(file_paths)} files")
        return file_paths
