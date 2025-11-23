"""
NIfTI visualization tool — view NIfTI files converted by rsna_dcm2niix.py and annotations in napari.
"""

import numpy as np
import napari
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
import nibabel as nib
import pandas as pd
import json
import ast
import yaml
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.my_utils.rsna_utils import load_nifti_and_convert_to_ras


class NapariNiftiVisualizer:
    """Visualize NIfTI files and annotations in napari."""

    def __init__(
        self,
        nifti_dir: str = "/workspace/data/series_niix",
        segmentation_dir: str = "/workspace/data/segmentations",
        segmentation_only: bool = False,
        uid_filter_file: Optional[str] = None,
    ):
        """Initialize visualizer.

        Args:
            nifti_dir: Directory of NIfTI files
            segmentation_dir: Directory of segmentation files
            segmentation_only: Restrict to cases with vessel segmentation
            uid_filter_file: YAML path for UID filter
        """
        self.nifti_dir = Path(nifti_dir)
        self.segmentation_dir = Path(segmentation_dir)
        self.segmentation_only = segmentation_only
        self.uid_filter = self._load_uid_filter(uid_filter_file) if uid_filter_file else None
        self.viewer = None
        self.nifti_list = self._get_nifti_list()
        self.current_index = 0

        # Annotation state
        self.annotations_df = self._load_annotations()
        self.current_annotations = []
        self.current_annotation_index = 0

    def _load_uid_filter(self, filter_file: str) -> Set[str]:
        """Load UID filter from YAML.

        Args:
            filter_file: YAML path

        Returns:
            Set[str]: Set of UIDs to include
        """
        try:
            with open(filter_file, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    # If simple list format
                    uid_set = set(uid for uid in data if isinstance(uid, str) and uid.strip())
                else:
                    # For nested structures with comments, extract UIDs
                    uid_set = set()
                    for item in data if isinstance(data, list) else []:
                        if isinstance(item, str) and item.strip() and not item.startswith("#"):
                            uid_set.add(item.strip())

            print(f"Loaded UID filter: {len(uid_set)} UIDs")
            return uid_set
        except Exception as e:
            print(f"⚠ Failed to load UID filter: {e}")
            return set()

    def _get_nifti_list(self) -> List[Path]:
        """List available NIfTI files (sorted)."""
        if not self.nifti_dir.exists():
            print(f"NIfTI directory not found: {self.nifti_dir}")
            return []

        # Find .nii.gz and .nii files (exclude .annotations.json)
        nifti_files = []
        for pattern in ["**/*.nii.gz", "**/*.nii"]:
            for f in self.nifti_dir.glob(pattern):
                if not f.name.endswith(".annotations.json"):
                    series_uid = self._get_series_uid_from_path(f)

                    # Apply UID filter if present
                    if self.uid_filter and series_uid not in self.uid_filter:
                        continue

                    # In segmentation_only mode, check for segmentation existence
                    if self.segmentation_only:
                        if series_uid:
                            # New layout: segmentation files live directly under segmentations dir
                            seg_paths = [
                                self.segmentation_dir / f"{series_uid}_cowseg.nii",
                                self.segmentation_dir / f"{series_uid}_cowseg.nii.gz",
                            ]

                            if any(seg_path.exists() for seg_path in seg_paths):
                                nifti_files.append(f)
                        # Skip when SeriesUID is unavailable
                    else:
                        nifti_files.append(f)

        nifti_files.sort()

        # Summarize after filter
        if self.uid_filter:
            print(f"After UID filter: {len(nifti_files)} NIfTI files")

        if self.segmentation_only:
            print(f"Found {len(nifti_files)} NIfTI files with vessel segmentation")
        else:
            print(f"Found {len(nifti_files)} NIfTI files")

        return nifti_files

    def _load_annotations(self) -> Optional[pd.DataFrame]:
        """Load annotation CSV file in nifti_dir if present."""
        annotations_csv = self.nifti_dir / "nifti_annotations.csv"
        if annotations_csv.exists():
            try:
                df = pd.read_csv(annotations_csv)
                print(f"Loaded {len(df)} annotations from {annotations_csv}")
                return df
            except Exception as e:
                print(f"Failed to load annotations CSV: {e}")
        return None

    def _load_annotations_json(self, nifti_path: Path) -> List[Dict]:
        """Load per-file annotations JSON next to NIfTI file."""
        # Construct JSON filename appropriately for .nii.gz and .nii
        if nifti_path.suffix == ".gz":
            json_path = nifti_path.with_suffix(".annotations.json")
        else:
            json_path = nifti_path.with_suffix(".annotations.json")

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("annotations", [])
            except Exception as e:
                print(f"Failed to load annotations JSON: {e}")
        return []

    def _get_series_uid_from_path(self, nifti_path: Path) -> Optional[str]:
        """Infer SeriesInstanceUID from parent directory name."""
        return nifti_path.parent.name

    def _load_modality_from_json(self, nifti_path: Path) -> Optional[str]:
        """Read Modality from sidecar JSON next to NIfTI (if exists)."""
        # Build JSON path next to NIfTI filename
        if nifti_path.suffix == ".gz":
            # For .nii.gz
            json_path = nifti_path.with_suffix("").with_suffix(".json")
        else:
            # For .nii
            json_path = nifti_path.with_suffix(".json")

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    modality = data.get("Modality", None)
                    if modality:
                        print(f"✓ Modality: {modality}")
                        return modality
            except Exception as e:
                print(f"⚠ Failed to load sidecar JSON: {e}")
        else:
            print(f"⚠ Sidecar JSON not found: {json_path}")
        return None

    def _load_vessel_segmentation(self, series_uid: str) -> Optional[np.ndarray]:
        """Load vessel segmentation by SeriesInstanceUID if available."""
        # New layout: files in segmentations dir
        # Pattern 1: SeriesUID_cowseg.nii
        seg_path = self.segmentation_dir / f"{series_uid}_cowseg.nii"

        # Pattern 2: SeriesUID_cowseg.nii.gz
        if not seg_path.exists():
            seg_path = self.segmentation_dir / f"{series_uid}_cowseg.nii.gz"

        if seg_path.exists():
            try:
                seg_data, _, _ = load_nifti_and_convert_to_ras(seg_path)
                print(f"Loaded vessel segmentation: {seg_path.name}")
                return seg_data
            except Exception as e:
                print(f"Failed to load vessel segmentation: {e}")
        return None

    def _setup_keyboard_bindings(self):
        """Setup keyboard events"""

        @self.viewer.bind_key("n")
        def next_case(viewer):
            """Move to next case (n key)"""
            if len(self.nifti_list) == 0:
                print("No NIfTI files available")
                return

            self.current_index = (self.current_index + 1) % len(self.nifti_list)
            current_file = self.nifti_list[self.current_index]
            series_uid = self._get_series_uid_from_path(current_file)
            print(f"\n=== Next Case ({self.current_index + 1}/{len(self.nifti_list)}) ===")
            print(f"File: {current_file.name}")
            if series_uid:
                print(f"SeriesInstanceUID: {series_uid}")
            self._load_and_display_case(current_file)

        @self.viewer.bind_key("p")
        def previous_case(viewer):
            """Move to previous case (p key)"""
            if len(self.nifti_list) == 0:
                print("No NIfTI files available")
                return

            self.current_index = (self.current_index - 1) % len(self.nifti_list)
            current_file = self.nifti_list[self.current_index]
            series_uid = self._get_series_uid_from_path(current_file)
            print(f"\n=== Previous Case ({self.current_index + 1}/{len(self.nifti_list)}) ===")
            print(f"File: {current_file.name}")
            if series_uid:
                print(f"SeriesInstanceUID: {series_uid}")
            self._load_and_display_case(current_file)

        @self.viewer.bind_key("c")
        def reset_camera(viewer):
            """Reset camera (c key)"""
            print("Camera reset")
            viewer.reset_view()

        @self.viewer.bind_key("m")
        def toggle_annotations(viewer):
            """Toggle annotation related layers display (m key)"""
            self._toggle_annotation_layers()

        @self.viewer.bind_key("k")
        def next_annotation(viewer):
            """Move to next annotation position (k key)"""
            self._navigate_to_next_annotation()

        @self.viewer.bind_key("j")
        def previous_annotation(viewer):
            """Move to previous annotation position (j key)"""
            self._navigate_to_previous_annotation()

        print("Keyboard bindings set up:")
        print("  'n' - Next case")
        print("  'p' - Previous case")
        print("  'c' - Reset camera")
        print("  'm' - Toggle annotations")
        print("  'k' - Next annotation")
        print("  'j' - Previous annotation")

    def _toggle_annotation_layers(self):
        """Toggle visibility of annotation related layers"""
        annotation_layer_names = ["Annotation Points", "Vessel Segmentation"]

        # Check current display state
        visible_count = 0
        total_count = 0

        for layer_name in annotation_layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                total_count += 1
                if layer.visible:
                    visible_count += 1

        # Toggle logic
        new_state = visible_count < total_count
        action = "Show" if new_state else "Hide"

        # Change visibility of each layer
        for layer_name in annotation_layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                layer.visible = new_state

        print(f"Annotation-related layers: {action}")

    def _navigate_to_next_annotation(self):
        """Move to next annotation position"""
        if not self.current_annotations:
            print("⚠ No annotation points")
            return

        self.current_annotation_index = (self.current_annotation_index + 1) % len(self.current_annotations)
        self._move_to_current_annotation("Next")

    def _navigate_to_previous_annotation(self):
        """Move to previous annotation position"""
        if not self.current_annotations:
            print("⚠ No annotation points")
            return

        self.current_annotation_index = (self.current_annotation_index - 1) % len(self.current_annotations)
        self._move_to_current_annotation("Prev")

    def _move_to_current_annotation(self, direction: str = ""):
        """Move to slice corresponding to current annotation index"""
        if not self.current_annotations or not self.viewer:
            return

        current_ann = self.current_annotations[self.current_annotation_index]
        z_slice = int(current_ann.get("nifti_z", 0))
        location = current_ann.get("location", "Unknown")

        # Adjust napari viewer slice position
        if hasattr(self.viewer.dims, "current_step"):
            try:
                # Get volume layer shape
                volume_layer = None
                for layer in self.viewer.layers:
                    if hasattr(layer, "data") and len(layer.data.shape) == 3:
                        volume_layer = layer
                        break

                if volume_layer is not None:
                    volume_shape = volume_layer.data.shape
                    max_z = volume_shape[0] - 1

                    if 0 <= z_slice <= max_z:
                        new_step = [int(z_slice), volume_shape[1] // 2, volume_shape[2] // 2]

                        self.viewer.dims.current_step = new_step
                        print(f"✓ {direction} annotation: {location} (slice {z_slice}) "
                              f"[{self.current_annotation_index + 1}/{len(self.current_annotations)}]")
                    else:
                        print(f"⚠ Slice index out of range: {z_slice} (max: {max_z})")

            except Exception as e:
                print(f"⚠ Failed to move annotation: {e}")

    def _get_layer_by_name(self, name_pattern: str):
        """Find first layer whose name contains the pattern."""
        if not self.viewer:
            return None
        for layer in self.viewer.layers:
            if name_pattern in layer.name:
                return layer
        return None

    def _update_or_create_layer(self, layer_name: str, data=None, layer_type="image", **kwargs):
        """Update data if layer exists, otherwise create new

        Args:
            layer_name: Layer name
            data: Layer data (None hides the layer if present)
            layer_type: One of "image", "labels", "points"
            **kwargs: Additional parameters passed to layer creation
        """
        if not self.viewer:
            return

        # Find existing layer
        existing_layer = self._get_layer_by_name(layer_name)

        if data is None:
            # Hide existing layer when no data provided
            if existing_layer is not None:
                existing_layer.visible = False
            return

        if existing_layer is not None:
            # Update existing layer data
            existing_layer.data = data
            existing_layer.visible = True

            # Update other attributes when provided
            if "scale" in kwargs:
                existing_layer.scale = kwargs["scale"]
            if "opacity" in kwargs:
                existing_layer.opacity = kwargs["opacity"]
            if "colormap" in kwargs and hasattr(existing_layer, "colormap"):
                existing_layer.colormap = kwargs["colormap"]
            if "properties" in kwargs and hasattr(existing_layer, "properties"):
                existing_layer.properties = kwargs["properties"]
            if "text" in kwargs and hasattr(existing_layer, "text"):
                existing_layer.text = kwargs["text"]
            if "size" in kwargs and hasattr(existing_layer, "size"):
                existing_layer.size = kwargs["size"]
            if "face_color" in kwargs and hasattr(existing_layer, "face_color"):
                existing_layer.face_color = kwargs["face_color"]
        else:
            # Create new layer
            if layer_type == "image":
                self.viewer.add_image(data, name=layer_name, **kwargs)
            elif layer_type == "labels":
                self.viewer.add_labels(data, name=layer_name, **kwargs)
            elif layer_type == "points":
                self.viewer.add_points(data, name=layer_name, **kwargs)

    def _load_and_display_case(self, nifti_path: Path):
        """Load and display the specified NIfTI file."""
        try:
            # Load NIfTI file
            nii_img = nib.load(str(nifti_path))
            volume_xyz = nii_img.get_fdata()  # shape: (X, Y, Z)
            zooms_xyz = nii_img.header.get_zooms()[:3]  # (dx, dy, dz)

            # Transpose as napari expects (Z, Y, X)
            volume_zyx = np.transpose(volume_xyz, (2, 1, 0))  # (Z, Y, X)
            voxel_spacing_zyx = (zooms_xyz[2], zooms_xyz[1], zooms_xyz[0])  # (dz, dy, dx)

            # Read Modality from JSON
            modality = self._load_modality_from_json(nifti_path)

            # Normalize volume (considering Modality)
            volume_normalized = self._normalize_volume(volume_zyx, modality)

            # Update or create volume layer
            self._update_or_create_layer(
                "NIfTI Volume",
                volume_normalized,
                layer_type="image",
                colormap="gray",
                scale=voxel_spacing_zyx,
                opacity=0.8,
            )

            print(f"✓ NIfTI Volume: {volume_zyx.shape}, Spacing: {voxel_spacing_zyx}")

            # Get SeriesInstanceUID
            series_uid = self._get_series_uid_from_path(nifti_path)

            # Display SeriesInstanceUID
            if series_uid:
                print(f"📋 SeriesInstanceUID: {series_uid}")
            else:
                print("⚠ Failed to determine SeriesInstanceUID")

            # Load vessel segmentation
            vessel_seg_data = None
            if series_uid:
                vessel_seg = self._load_vessel_segmentation(series_uid)
                if vessel_seg is not None:
                    # Handle segmentation resampling if needed
                    if vessel_seg.shape != volume_xyz.shape:
                        print(f"⚠ Segmentation shape mismatch: {vessel_seg.shape} != {volume_xyz.shape}")
                    else:
                        vessel_seg_data = np.transpose(vessel_seg, (2, 1, 0)).astype(np.uint32)
                        unique_labels = np.unique(vessel_seg)
                        print(f"✓ Vessel segmentation: {vessel_seg.shape}, labels: {unique_labels}")

            # Update or create vessel segmentation layer (hide if no data)
            self._update_or_create_layer(
                "Vessel Segmentation",
                vessel_seg_data,
                layer_type="labels",
                opacity=0.6,
                scale=voxel_spacing_zyx,
            )

            # Load annotations
            annotations = self._load_annotations_json(nifti_path)
            self.current_annotations = annotations
            self.current_annotation_index = 0

            points_data = None
            properties = None

            if annotations:
                # Extract annotation points
                points = []
                properties = {"location": [], "sop_uid": []}

                for ann in annotations:
                    if "nifti_x" in ann and "nifti_y" in ann and "nifti_z" in ann:
                        points.append([ann["nifti_z"], ann["nifti_y"], ann["nifti_x"]])
                        properties["location"].append(ann.get("location", "Unknown"))
                        sop_uid = ann.get("SOPInstanceUID", "")
                        properties["sop_uid"].append(sop_uid[:12] + "..." if len(sop_uid) > 12 else sop_uid)

                if points:
                    points_data = np.array(points)
                    print(f"✓ Annotation points: {len(points)}")

                    # Sort annotations by Z coordinate
                    self.current_annotations = sorted(
                        annotations, key=lambda x: x.get("nifti_z", float("inf"))
                    )

                    print("Annotation list:")
                    for i, ann in enumerate(self.current_annotations):
                        location = ann.get("location", "Unknown")
                        z_slice = ann.get("nifti_z", 0)
                        print(f"  {i+1}. {location}: Slice {z_slice:.1f}")

            # Update or create annotation layer (hide if no data)
            text_parameters = None
            if properties:
                text_parameters = {
                    "string": "location",
                    "anchor": "upper_left",  # Text anchor position
                    "translation": [0, 0, -20],  # Offset from point (z, y, x) - 3D support
                    "size": 10,  # Text size
                    "color": "yellow",  # Text color
                }

            self._update_or_create_layer(
                "Annotation Points",
                points_data,
                layer_type="points",
                size=5.0,
                face_color="red",
                properties=properties,
                text=text_parameters,
                scale=voxel_spacing_zyx,
            )

            # Update viewer title
            title = f"NIfTI Viewer - Case {self.current_index + 1}/{len(self.nifti_list)} - {nifti_path.name}"
            self.viewer.title = title

            # Reset camera
            self.viewer.reset_view()

            # Move to first annotation position
            if self.current_annotations:
                self.current_annotation_index = 0
                self._move_to_current_annotation("Initial")

            print(f"=== Data Info ===")
            print(f"File: {nifti_path.name}")
            print(f"SeriesInstanceUID: {series_uid if series_uid else 'N/A'}")
            print(f"Modality: {modality if modality else 'N/A'}")
            print(f"Shape: {volume_zyx.shape}")
            print(f"DType: {volume_zyx.dtype}")
            print(f"Value range: {volume_zyx.min():.1f} - {volume_zyx.max():.1f}")
            print(f"Voxel spacing: {voxel_spacing_zyx}")
            print(f"Annotation count: {len(annotations)}")

        except Exception as e:
            print(f"Error loading case {nifti_path}: {e}")

    def _normalize_volume(self, volume: np.ndarray, modality: Optional[str] = None) -> np.ndarray:
        """Normalize volume data (Process according to Modality)"""
        volume_float = volume.astype(np.float32)

        # if modality == "CT":
        #     # For CT: Window normalization (center=50, width=350)
        #     # window center = 50 HU, window width = 350 HU
        #     # window_min = center - width/2 = 50 - 175 = -125 HU
        #     # window_max = center + width/2 = 50 + 175 = 225 HU
        #     window_min = 50 - 350 / 2  # -125
        #     window_max = 50 + 350 / 2  # 225
        #
        #     print(f"✓ CT Window normalization applied: center={50}, width={350} (Range: {window_min} - {window_max})")
        #
        #     # Clip and normalize within window range
        #     volume_normalized = np.clip(volume_float, window_min, window_max)
        #     volume_normalized = (volume_normalized - window_min) / (window_max - window_min)
        # else:
        # Other Modality (MR etc.): Percentile-based normalization
        p1 = np.percentile(volume_float, 1)
        p99 = np.percentile(volume_float, 99)

        if p99 > p1:
            volume_normalized = np.clip(volume_float, p1, p99)
            volume_normalized = (volume_normalized - p1) / (p99 - p1)
        else:
            # Fallback: Normalize with min and max
            if volume_float.max() > volume_float.min():
                volume_normalized = (volume_float - volume_float.min()) / (
                    volume_float.max() - volume_float.min()
                )
            else:
                volume_normalized = volume_float

        return volume_normalized

    def visualize(self, start_index: int = 0, series_uid: Optional[str] = None) -> napari.Viewer:
        """
        Launch the napari viewer.

        Args:
            start_index: Starting index
            series_uid: Specific SeriesInstanceUID to start

        Returns:
            napari.Viewer
        """
        if len(self.nifti_list) == 0:
            print("No NIfTI files found")
            return None

        # If specific UID is specified, find its index
        if series_uid:
            uid_found = False
            for i, nifti_path in enumerate(self.nifti_list):
                path_uid = self._get_series_uid_from_path(nifti_path)
                if path_uid == series_uid:
                    start_index = i
                    uid_found = True
                    print(f"Found specified UID at index {i + 1}: {series_uid}")
                    break

            if not uid_found:
                print(f"⚠ Specified UID '{series_uid}' not found")
                print("Available UIDs:")
                for i, nifti_path in enumerate(self.nifti_list[:10]):  # Show first 10
                    path_uid = self._get_series_uid_from_path(nifti_path)
                    if path_uid:
                        print(f"  {i + 1}. {path_uid}")
                if len(self.nifti_list) > 10:
                    print(f"  ... and {len(self.nifti_list) - 10} more")
                print("Starting at default index")

        # Adjust start index
        start_index = max(0, min(start_index, len(self.nifti_list) - 1))
        self.current_index = start_index

        # First file
        first_file = self.nifti_list[start_index]
        print(f"Starting with file {start_index + 1}/{len(self.nifti_list)}: {first_file.name}")

        # Create napari viewer
        self.viewer = napari.Viewer(title=f"NIfTI Viewer - {first_file.name}")

        # Setup keyboard bindings
        self._setup_keyboard_bindings()

        # Display first case
        self._load_and_display_case(first_file)

        return self.viewer


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Napari NIfTI Visualizer")
    parser.add_argument(
        "--nifti-dir",
        type=str,
        default="/workspace/data/series_niix",
        help="Directory containing NIfTI files",
    )
    parser.add_argument(
        "--seg-dir",
        type=str,
        default="/workspace/data/segmentations",
        help="Directory containing vessel segmentation files",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Starting index")
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Specific SeriesInstanceUID to start with",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Only show cases with vessel segmentation",
    )
    parser.add_argument(
        "--uid-filter",
        type=str,
        default=None,
        help="YAML file containing UIDs to filter (e.g., /workspace/data/error_data.yaml)",
    )

    args = parser.parse_args()

    print("=== Napari NIfTI Visualizer ===")
    print(f"NIfTI directory: {args.nifti_dir}")
    print(f"Segmentation directory: {args.seg_dir}")
    if args.segmentation_only:
        print("Mode: Showing only cases with vessel segmentation")

    # Check UID filter file
    if args.uid_filter:
        filter_path = Path(args.uid_filter)
        if filter_path.exists():
            print(f"Using UID filter file: {args.uid_filter}")
        else:
            print(f"⚠ UID filter file not found: {args.uid_filter}")
            args.uid_filter = None

    # Create visualizer
    visualizer = NapariNiftiVisualizer(
        nifti_dir=args.nifti_dir,
        segmentation_dir=args.seg_dir,
        segmentation_only=args.segmentation_only,
        uid_filter_file=args.uid_filter,
    )

    # Launch viewer
    viewer = visualizer.visualize(start_index=args.start_index, series_uid=args.uid)

    if viewer:
        print("\n=== Napari viewer opened ===")
        print("Mouse:")
        print("- Wheel: zoom")
        print("- Right-drag: rotate")
        print("- Left-drag: pan")
        print("- Sliders: adjust slice per axis")
        print("\nKeyboard:")
        print("- 'n': next case")
        print("- 'p': previous case")
        print("- 'c': reset camera")
        print("- 'm': toggle annotations")
        print("- 'k': next annotation")
        print("- 'j': previous annotation")

        napari.run()
    else:
        print("Visualization failed")


if __name__ == "__main__":
    main()
