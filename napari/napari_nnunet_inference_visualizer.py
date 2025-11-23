"""
Visualize nnU-Net inference outputs (.npz probability maps + .nii.gz segmentation) in napari.
Press 'v' to toggle the displayed class.

Example:
  python napari_nnunet_inference_visualizer.py \
    --pred-dir /workspace/logs/nnUNet_results/.../inference_debug \
    --images-dir /workspace/data/nnUNet_inference/imagesTs_debug

Key bindings:
  - n: next case
  - p: previous case
  - v: toggle view (per-class probability -> full segmentation)
  - c: reset camera
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import json
import numpy as np
import nibabel as nib
import napari

try:
    import rootutils

    # Set project root (same as validation scripts)
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.my_utils.rsna_utils import load_nifti_and_convert_to_ras
except Exception:
    # Fallback when rootutils is unavailable: read via nibabel
    rootutils = None

    def load_nifti_and_convert_to_ras(path: Path):  # type: ignore
        img = nib.load(str(path))
        data = img.get_fdata()
        # Return as-is here (implement RAS conversion if needed)
        affine = img.affine
        header = img.header
        return data, affine, header


class NapariNnunetInferenceVisualizer:
    """Class to visualize nnU-Net inference outputs with napari."""

    def __init__(
        self,
        pred_dir: str,
        images_dir: Optional[str] = None,
    ) -> None:
        self.pred_dir = Path(pred_dir)
        if not self.pred_dir.exists():
            raise ValueError(f"Prediction directory not found: {self.pred_dir}")

        self.images_dir = Path(images_dir) if images_dir else None

        # Case IDs (basename without extension)
        self.case_ids = self._collect_case_ids()
        if len(self.case_ids) == 0:
            raise ValueError(f"No cases found under: {self.pred_dir}")

        # Load label names from dataset.json (optional)
        self.label_names = self._load_label_names()

        # napari state
        self.viewer: Optional[napari.Viewer] = None
        self.current_index = 0
        self.current_class = 0  # class ID including background
        self.view_index = 0  # 0..(C-1): per-class prob, C: full segmentation

        # Per-case cache
        self._cache: Dict[str, Dict[str, object]] = {}

    # ---------- Setup ----------
    def _collect_case_ids(self) -> List[str]:
        ids = set()
        for p in sorted(self.pred_dir.glob("*.npz")):
            ids.add(p.stem)
        for p in sorted(self.pred_dir.glob("*.nii.gz")):
            ids.add(p.name.replace(".nii.gz", ""))
        # Skip .pkl-only entries
        return sorted(ids)

    def _load_label_names(self) -> Dict[int, str]:
        dataset_json = self.pred_dir / "dataset.json"
        names: Dict[int, str] = {}
        if dataset_json.exists():
            try:
                with open(dataset_json, "r") as f:
                    d = json.load(f)
                labels = d.get("labels") or d.get("label_names") or {}
                # Expected format: {"0":"background","1":"class1",...}
                for k, v in labels.items():
                    try:
                        names[int(k)] = str(v)
                    except Exception:
                        continue
            except Exception:
                pass
        # If empty, fallback to IDs
        if not names:
            # Open any npz to infer class count
            any_npz = next(iter(self.pred_dir.glob("*.npz")), None)
            if any_npz is not None:
                arr = np.load(any_npz)
                c = int(arr["probabilities"].shape[0])
                names = {i: ("background" if i == 0 else f"class_{i}") for i in range(c)}
        return names

    # ---------- File paths ----------
    def _npz_path(self, case_id: str) -> Optional[Path]:
        p = self.pred_dir / f"{case_id}.npz"
        return p if p.exists() else None

    def _nii_path(self, case_id: str) -> Optional[Path]:
        p = self.pred_dir / f"{case_id}.nii.gz"
        return p if p.exists() else None

    def _image_path(self, case_id: str) -> Optional[Path]:
        if self.images_dir is None:
            return None
        p = self.images_dir / f"{case_id}_0000.nii.gz"
        return p if p.exists() else None

    # ---------- Load Data ----------
    def _load_case(self, case_id: str) -> Dict[str, object]:
        if case_id in self._cache:
            return self._cache[case_id]

        out: Dict[str, object] = {}

        # Probability map
        npz_path = self._npz_path(case_id)
        if npz_path is None:
            raise FileNotFoundError(f".npz not found for case: {case_id}")
        npz = np.load(npz_path)
        probs = npz["probabilities"]  # (C, Z, Y, X)
        out["probs"] = probs

        # Segmentation (Final label)
        nii_path = self._nii_path(case_id)
        if nii_path is None:
            raise FileNotFoundError(f".nii.gz not found for case: {case_id}")
        seg_data, _, _ = load_nifti_and_convert_to_ras(nii_path)
        # Transpose as napari uses (Z, Y, X) convention
        seg_zyx = np.transpose(seg_data, (2, 1, 0)).astype(np.uint32)
        out["seg"] = seg_zyx

        # Image (Optional)
        img_path = self._image_path(case_id)
        if img_path is not None:
            img_data, _, _ = load_nifti_and_convert_to_ras(img_path)
            img_zyx = np.transpose(img_data, (2, 1, 0))
            # Normalization
            img_norm = self._normalize(img_zyx)
            out["img"] = img_norm

            # Scale (Voxel spacing)
            nii_img = nib.load(str(img_path))
            zooms_xyz = nii_img.header.get_zooms()[:3]
            out["scale"] = (zooms_xyz[2], zooms_xyz[1], zooms_xyz[0])
        else:
            # Use segmentation voxel spacing if image is missing (if exists)
            try:
                nii_seg = nib.load(str(nii_path))
                zooms_xyz = nii_seg.header.get_zooms()[:3]
                out["scale"] = (zooms_xyz[2], zooms_xyz[1], zooms_xyz[0])
            except Exception:
                out["scale"] = (1.0, 1.0, 1.0)

        self._cache[case_id] = out
        return out

    # ---------- Visualization Logic ----------
    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32)
        p1, p99 = np.percentile(vol, [1, 99])
        if p99 > p1:
            vol = np.clip(vol, p1, p99)
            vol = (vol - p1) / (p99 - p1)
        else:
            vmin, vmax = vol.min(), vol.max()
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
        return vol

    def _update_layers_for_case(self, case_id: str):
        d = self._load_case(case_id)
        probs: np.ndarray = d["probs"]  # (C, Z, Y, X)
        seg: np.ndarray = d["seg"]  # (Z, Y, X)
        scale: Tuple[float, float, float] = d.get("scale", (1.0, 1.0, 1.0))  # type: ignore
        img: Optional[np.ndarray] = d.get("img")  # type: ignore

        # Number of classes
        num_classes = probs.shape[0]
        # Image layer
        if img is not None:
            self._add_or_update_image("Image", img, scale=scale, colormap="gray", opacity=0.8)
        
        # Create/update layers first (toggle visibility later)
        # Probability layer (create with current class name)
        cur_c = min(self.current_class, num_classes - 1)
        lname_prob = self._class_label_name(cur_c)
        self._add_or_update_image(
            f"Prob: {lname_prob}", probs[cur_c], scale=scale, colormap="magma", opacity=0.7
        )
        # Segmentation (Whole)
        self._add_or_update_labels("Seg(Whole)", seg.astype(np.uint32), scale=scale, opacity=0.5)

        # Display control: Image always visible. Toggle probability (cycle classes) and whole seg with 'v'
        prob_layer = self._get_layer("Prob:")
        seg_layer = self._get_layer("Seg(Whole)")
        if self.view_index < num_classes:
            # Probability display mode
            self.current_class = self.view_index
            new_name = self._class_label_name(self.current_class)
            if prob_layer is not None:
                prob_layer.name = f"Prob: {new_name}"
                prob_layer.data = probs[self.current_class]
                prob_layer.visible = True
            if seg_layer is not None:
                seg_layer.visible = False
            self.viewer.title = (
                f"nnUNet Inference Viewer - Case {self.current_index + 1}/{len(self.case_ids)} - "
                f"Prob Class {self.current_class}/{num_classes - 1} ({new_name})"
            )
            if img is not None and (probs[self.current_class].shape != img.shape or seg.shape != img.shape):
                print(
                    f"[Warning] shape mismatch: img{img.shape}, prob{probs[self.current_class].shape}, seg{seg.shape}. "
                    "There might be differences in coordinate system or cropping."
                )
        else:
            # Segmentation display mode
            if prob_layer is not None:
                prob_layer.visible = False
            if seg_layer is not None:
                seg_layer.visible = True
            self.viewer.title = (
                f"nnUNet Inference Viewer - Case {self.current_index + 1}/{len(self.case_ids)} - Segmentation"
            )
        # Do not create per-class segmentation layers (requirement)

    # ---------- napari layer helper ----------
    def _get_layer(self, name_contains: str):
        for layer in self.viewer.layers:  # type: ignore
            if name_contains in layer.name:
                return layer
        return None

    def _add_or_update_image(self, name: str, data: np.ndarray, **kwargs):
        lyr = self._get_layer(name)
        if lyr is None:
            self.viewer.add_image(data, name=name, **kwargs)  # type: ignore
        else:
            lyr.data = data
            if "scale" in kwargs:
                lyr.scale = kwargs["scale"]
            if "opacity" in kwargs:
                lyr.opacity = kwargs["opacity"]
            if "colormap" in kwargs and hasattr(lyr, "colormap"):
                lyr.colormap = kwargs["colormap"]

    def _add_or_update_labels(self, name: str, data: np.ndarray, **kwargs):
        lyr = self._get_layer(name)
        if lyr is None:
            self.viewer.add_labels(data, name=name, **kwargs)  # type: ignore
        else:
            lyr.data = data
            if "scale" in kwargs:
                lyr.scale = kwargs["scale"]
            if "opacity" in kwargs:
                lyr.opacity = kwargs["opacity"]

    def _class_label_name(self, cid: int) -> str:
        label = self.label_names.get(cid, f"class_{cid}")
        return f"{label} ({cid})"

    # ---------- Key Bindings ----------
    def _setup_keybindings(self):
        @self.viewer.bind_key("n")
        def _next_case(v):
            self.current_index = (self.current_index + 1) % len(self.case_ids)
            case_id = self.case_ids[self.current_index]
            print(f"\n=== Next Case {self.current_index + 1}/{len(self.case_ids)}: {case_id}")
            self._update_layers_for_case(case_id)

        @self.viewer.bind_key("p")
        def _prev_case(v):
            self.current_index = (self.current_index - 1) % len(self.case_ids)
            case_id = self.case_ids[self.current_index]
            print(f"\n=== Previous Case {self.current_index + 1}/{len(self.case_ids)}: {case_id}")
            self._update_layers_for_case(case_id)

        @self.viewer.bind_key("v")
        def _next_class(v):
            # Probability class -> ... -> Whole seg -> Return to start
            case_id = self.case_ids[self.current_index]
            probs = self._load_case(case_id)["probs"]  # type: ignore
            num_classes = probs.shape[0]
            self.view_index = (self.view_index + 1) % (num_classes + 1)
            if self.view_index < num_classes:
                cname = self._class_label_name(self.view_index)
                print(f"Switch view -> Probability: {cname}")
            else:
                print("Switch view -> Segmentation")
            self._update_layers_for_case(case_id)

        @self.viewer.bind_key("c")
        def _reset_camera(v):
            print("Camera reset")
            self.viewer.reset_view()

        print("\nKeyboard bindings:")
        print("  'n' - Next case")
        print("  'p' - Previous case")
        print("  'v' - Cycle: Prob(class 0..N) -> Seg")
        print("  'c' - Reset camera")

    # ---------- Execution ----------
    def run(self, start_index: int = 0, start_class: Optional[int] = None) -> Optional[napari.Viewer]:
        start_index = max(0, min(start_index, len(self.case_ids) - 1))
        self.current_index = start_index
        if start_class is not None:
            self.current_class = max(0, start_class)
            self.view_index = max(0, start_class)
        else:
            self.view_index = 0

        first_case = self.case_ids[self.current_index]
        print(f"Found {len(self.case_ids)} cases under: {self.pred_dir}")
        print(f"Starting with case[{self.current_index + 1}]: {first_case}")

        self.viewer = napari.Viewer(title=f"nnUNet Inference Viewer - {first_case}")
        self._setup_keybindings()
        self._update_layers_for_case(first_case)
        return self.viewer


def main():
    parser = argparse.ArgumentParser(description="nnUNet Inference Visualizer")
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Path to prediction output directory (contains .npz, .nii.gz)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/workspace/data/nnUNet_inference/imagesTs_debug",
        help="Optional path to original images (for background view)",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-class", type=int, default=None)

    args = parser.parse_args()

    viz = NapariNnunetInferenceVisualizer(pred_dir=args.pred_dir, images_dir=args.images_dir)
    viewer = viz.run(start_index=args.start_index, start_class=args.start_class)
    if viewer is not None:
        napari.run()


if __name__ == "__main__":
    main()
