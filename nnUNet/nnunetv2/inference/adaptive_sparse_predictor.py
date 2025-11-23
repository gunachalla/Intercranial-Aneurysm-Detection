# -*- coding: utf-8 -*-
import numpy as np
import torch

# scikit-learn is only required for DBSCAN-based sparse search (DBSCANAdaptiveSparsePredictor).
# To avoid ImportError when unused (e.g., VESSEL_NNUNET_SPARSE_MODEL_DIR not specified),
# make the import optional here.
try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN  # type: ignore
except Exception:
    SklearnDBSCAN = None  # type: ignore[assignment]
from typing import Tuple, List, Optional, Dict, Any, Union
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
from monai.inferers import SlidingWindowInferer
from src.my_utils.trt_runner import TRTRunner  # TensorRT execution wrapper


class AdaptiveSparsePredictor(nnUNetPredictor):
    # nnUNet inference class using adaptive sparse search
    # Performs 2-stage processing (sparse search -> detailed inference) for large volumes,
    # and executes normal inference for small volumes.

    # Constant definitions
    MORPHOLOGY_KERNEL_SIZE = 2
    BINARY_MASK_THRESHOLD = 0.7

    def __init__(
        self,
        # Sparse search decision parameters
        window_count_threshold: int = 100,
        # Sparse search parameters
        sparse_downscale_factor: float = 2.0,
        sparse_overlap: float = 0.2,
        detection_threshold: float = 0.3,
        # ROI margin used in sparse search
        sparse_bbox_margin_voxels: int = 20,
        # Overlap rate for dense search (sliding within ROI)
        dense_overlap: float = 0.3,
        # ROI vertical (SI) direction control
        limit_si_extent: bool = True,
        max_si_extent_mm: float = 150.0,
        si_axis: Optional[int] = 0,
        # ROI minimum length (mm) guarantee
        min_si_extent_mm: float = 100.0,
        min_xy_extent_mm: float = 120.0,
        # Normal parameters
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = False,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = False,
    ):
        # Args:
        #     window_count_threshold: Threshold for sliding window splits to start sparse search
        #     sparse_downscale_factor: Downsampling rate during sparse search
        #     sparse_overlap: Overlap rate during sparse search
        #     detection_threshold: Threshold for vessel region detection
        #     sparse_bbox_margin_voxels: Margin (in voxels) added to the detection region (BBox) obtained from sparse search
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )

        self.window_count_threshold = window_count_threshold
        self.sparse_downscale_factor = sparse_downscale_factor
        self.sparse_overlap = sparse_overlap
        self.detection_threshold = detection_threshold
        # BBox margin for sparse search
        self.sparse_bbox_margin_voxels = int(sparse_bbox_margin_voxels)
        # Dense search overlap rate (converted to tile_step_size = 1 - dense_overlap for internal use)
        self.dense_overlap = float(dense_overlap)
        # ROI vertical (SI) direction control
        self.limit_si_extent = limit_si_extent
        self.max_si_extent_mm = max_si_extent_mm
        self.si_axis = si_axis
        # ROI minimum length (mm) setting (specified separately for SI axis and XY axis)
        self.min_si_extent_mm = float(min_si_extent_mm)
        self.min_xy_extent_mm = float(min_xy_extent_mm)
        # Holds network weights (state_dict) for a single fold
        # Ensemble is expected to be performed by the caller
        self.network_weights = None
        # TensorRT integration: Runner and control flags
        self.trt_runner: Optional[TRTRunner] = None
        self.trt_enforce_half_output: bool = True
        # Holds training configuration name for identification (used for engine search, etc.)
        self.trainer_name: Optional[str] = None
        self.configuration_name: Optional[str] = None

    # ===== Helpers (aggregation of duplicate processing) =====
    @staticmethod
    def _slices_to_tuples(slices: Tuple[slice, ...]) -> List[Tuple[int, int]]:
        # Convert slice tuple to (start, stop) array (for JSON saving)
        tuples: List[Tuple[int, int]] = []
        for s in slices:
            if isinstance(s, slice):
                tuples.append((int(s.start), int(s.stop)))
            elif isinstance(s, (list, tuple)) and len(s) == 2:
                tuples.append((int(s[0]), int(s[1])))
            else:
                raise TypeError(f"Unsupported bbox element type: {type(s)}")
        return tuples

    @staticmethod
    def _add_slices(a: Tuple[slice, ...], b: Tuple[slice, ...]) -> Tuple[slice, ...]:
        # Add two slices and return the global coordinate slice
        out: List[slice] = []
        for s1, s2 in zip(a, b):
            out.append(slice(int(s1.start) + int(s2.start), int(s1.start) + int(s2.stop)))
        return tuple(out)

    @staticmethod
    def _bbox_from_mask(
        mask: torch.Tensor, margin: int | Tuple[int, ...], full_size: torch.Tensor
    ) -> Optional[Tuple[slice, ...]]:
        # Extract BBox from binary mask and apply margin and boundary clipping
        # margin accepts a single value or a tuple for each axis
        if not torch.any(mask):
            return None
        coords = torch.nonzero(mask, as_tuple=False)
        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values
        # Margin normalization
        if isinstance(margin, (tuple, list)):
            if len(margin) != len(mins):
                # If element counts do not match, fill the shortage with the last value
                mm = list(margin) + [int(margin[-1])] * (len(mins) - len(margin))
            else:
                mm = list(margin)
            margin_t = torch.tensor(mm, device=mins.device, dtype=mins.dtype)
        else:
            margin_t = torch.tensor([int(margin)] * len(mins), device=mins.device, dtype=mins.dtype)
        if torch.any(margin_t > 0):
            mins = torch.clamp(mins - margin_t, min=0)
            maxs = torch.clamp(maxs + margin_t, max=full_size - 1)
        return tuple(slice(int(mins[i].item()), int(maxs[i].item()) + 1) for i in range(len(mins)))

    def _build_transform_info(
        self,
        preprocessed: dict,
        data: torch.Tensor,
        roi_bbox: Tuple[slice, ...],
        refined_local_bbox: Optional[Tuple[slice, ...]],
    ) -> dict:
        # Generate coordinate transformation info dictionary (original space -> network space)
        props = preprocessed["data_properties"]
        transpose_forward = list(self.plans_manager.transpose_forward)
        transpose_backward = list(self.plans_manager.transpose_backward)
        spacing_transposed = [props["spacing"][i] for i in transpose_forward]
        dim = len(roi_bbox)
        target_spacing = list(self.configuration_manager.spacing)
        if len(target_spacing) < dim:
            target_spacing = [spacing_transposed[0]] + target_spacing
        scale_factors = [spacing_transposed[i] / target_spacing[i] for i in range(dim)]

        roi_bbox_network = roi_bbox
        if refined_local_bbox is not None:
            roi_bbox_network_refined = self._add_slices(roi_bbox_network, refined_local_bbox)
        else:
            roi_bbox_network_refined = roi_bbox_network

        return {
            "transpose_forward": transpose_forward,
            "transpose_backward": transpose_backward,
            "spacing_original": list(props["spacing"]),
            "spacing_after_resampling": target_spacing,
            "scale_factors_orig2net": scale_factors,
            "shape_before_cropping": list(props.get("shape_before_cropping", [])),
            "shape_after_cropping_and_before_resampling": list(
                props.get("shape_after_cropping_and_before_resampling", [])
            ),
            "bbox_used_for_cropping": self._slices_to_tuples(props.get("bbox_used_for_cropping", [])),
            "network_shape": list(data.shape[1:]),
            "roi_bbox_network": self._slices_to_tuples(roi_bbox_network),
            "roi_offset_network": [int(s.start) for s in roi_bbox_network],
            "roi_bbox_network_refined": self._slices_to_tuples(roi_bbox_network_refined),
            "roi_offset_network_refined": [int(s.start) for s in roi_bbox_network_refined],
        }

    def _refine_local_bbox_from_logits(
        self,
        prediction_logits: torch.Tensor,
        threshold: float,
        refine_margin_voxels: int | Tuple[int, ...],
    ) -> Optional[Tuple[slice, ...]]:
        # Re-extract ROI using non-background class probabilities from dense search logits
        probs = torch.softmax(prediction_logits, dim=0)
        if probs.shape[0] > 1:
            vessel_prob = torch.max(probs[1:], dim=0)[0]
        else:
            vessel_prob = probs[0]
        mask = vessel_prob > float(threshold)
        full_size = torch.tensor(prediction_logits.shape[1:], device=prediction_logits.device)
        return self._bbox_from_mask(mask, refine_margin_voxels, full_size)

    def should_use_sparse_search(self, image_size: Tuple[int, ...]) -> bool:
        # Determine whether to use sparse search from image size and patch size
        # Args:
        #     image_size: Input image size assumed to be (C, H, W, D) format
        # Returns:
        #     True if using sparse search
        # False if configuration is not loaded
        if self.configuration_manager is None:
            return False

        # Spatial dimension size excluding channel dimension
        spatial_size = image_size[1:] if len(image_size) == 4 else image_size

        # Check number of sliding window splits
        patch_size = self.configuration_manager.patch_size
        steps = compute_steps_for_sliding_window(spatial_size, patch_size, self.tile_step_size)
        total_windows = np.prod([len(s) for s in steps])

        if total_windows > self.window_count_threshold:
            if self.verbose:
                print(f"Window split count {total_windows} > {self.window_count_threshold}, using sparse search")
            return True

        if self.verbose:
            print(f"Window split count {total_windows} <= {self.window_count_threshold}, executing normal inference")
        return False

    @torch.inference_mode()
    def sparse_search(
        self, input_image: torch.Tensor, return_context: bool = False
    ) -> Union[Tuple[slice, ...], Tuple[Tuple[slice, ...], Dict[str, Any]]]:
        # Sparse search phase: Fast scan with downsampled image,
        # complete preprocessing on GPU and return a single BBox
        # Args:
        #     input_image: Input image tensor (C, H, W, D)
        # Returns:
        #     roi_bbox: Tuple of slices indicating a single ROI
        if self.verbose:
            print("Starting sparse search phase...")

        # Downsampling
        downscale_factors = [1] + [1 / self.sparse_downscale_factor] * (input_image.ndim - 1)
        downsampled = torch.nn.functional.interpolate(
            input_image.unsqueeze(0),
            scale_factor=downscale_factors[1:],
            mode="trilinear" if input_image.ndim == 4 else "bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.verbose:
            print(f"Downsampling: {input_image.shape} -> {downsampled.shape}")

        # Get original resolution size in advance
        target_size = tuple(int(x) for x in input_image.shape[1:])

        # Temporary parameter setting for sparse search
        original_tile_step_size = self.tile_step_size
        try:
            self.tile_step_size = 1.0 - self.sparse_overlap  # Convert overlap rate to step size

            # Sliding window inference (sparse)
            # Prefer fp16 compute on CUDA with autocast; logits will be cast to fp32 for softmax later
            if self.device.type == "cuda":
                downsampled = downsampled.to(device=self.device, dtype=torch.half)
                with torch.autocast(device_type="cuda"):
                    sparse_logits = self.predict_sliding_window_return_logits(downsampled)
            else:
                sparse_logits = self.predict_sliding_window_return_logits(downsampled)
        finally:
            # Always restore parameters
            self.tile_step_size = original_tile_step_size

        # Apply softmax to convert to probability
        sparse_probs = torch.softmax(sparse_logits, dim=0)

        # Create sparse search context info if necessary
        sparse_context: Optional[Dict[str, Any]] = None
        if return_context:
            seg_lowres = torch.argmax(sparse_probs, dim=0)
            seg_lowres_cpu = seg_lowres.detach().to(device="cpu", dtype=torch.int16)
            seg_highres = torch.nn.functional.interpolate(
                seg_lowres[None, None].to(dtype=torch.float32),
                size=target_size,
                mode="nearest",
            )[0, 0]
            seg_highres_cpu = seg_highres.detach().to(device="cpu", dtype=torch.int16)
            sparse_context = {
                "segmentation_lowres": seg_lowres_cpu,
                "segmentation_highres": seg_highres_cpu,
                "num_classes": int(sparse_logits.shape[0]),
                "input_shape": tuple(int(x) for x in input_image.shape),
                "downsampled_shape": tuple(int(x) for x in downsampled.shape),
                "network_spacing": tuple(
                    float(x) for x in getattr(self.configuration_manager, "spacing", [])
                ),
            }
            del seg_lowres, seg_highres

        # Get max probability of non-background classes (vessel region detection)
        if sparse_probs.shape[0] > 1:
            vessel_probs = torch.max(sparse_probs[1:], dim=0)[0]  # Max value of non-background classes
        else:
            vessel_probs = sparse_probs[0]

        # Create binary mask with thresholding (executed on GPU)
        # To save memory, first perform denoising on the low-resolution (downsampled) mask,
        # then change the order to upsample to original resolution
        binary_mask = (vessel_probs > self.detection_threshold).to(dtype=torch.half)

        # Denoising (morphological opening): erosion -> dilation (GPU approximation)
        # Execute while low-resolution to suppress memory usage
        k_open = int(self.MORPHOLOGY_KERNEL_SIZE)
        if k_open > 1:
            pad_open = k_open // 2
            inv = 1.0 - binary_mask
            if binary_mask.ndim == 3:
                inv_dil = torch.nn.functional.max_pool3d(
                    inv[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
                eroded = 1.0 - inv_dil
                opened = torch.nn.functional.max_pool3d(
                    eroded[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
            elif binary_mask.ndim == 2:
                inv_dil = torch.nn.functional.max_pool2d(
                    inv[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
                eroded = 1.0 - inv_dil
                opened = torch.nn.functional.max_pool2d(
                    eroded[None, None], kernel_size=k_open, stride=1, padding=pad_open
                )[0, 0]
            else:
                raise RuntimeError(f"Unsupported mask ndim: {binary_mask.ndim}")
        else:
            opened = binary_mask

        cleaned_lowres = opened if torch.any(opened > 0.5) else binary_mask

        # Upsample to original resolution (nearest neighbor). Use mask after opening
        if cleaned_lowres.ndim == 3:
            cleaned = torch.nn.functional.interpolate(
                cleaned_lowres[None, None], size=target_size, mode="nearest"
            )[0, 0]
        elif cleaned_lowres.ndim == 2:
            cleaned = torch.nn.functional.interpolate(
                cleaned_lowres[None, None], size=target_size, mode="nearest"
            )[0, 0]
        else:
            raise RuntimeError(f"Unsupported mask ndim: {cleaned_lowres.ndim}")

        # Re-binarize to 0/1 with threshold just in case
        cleaned = (cleaned > self.BINARY_MASK_THRESHOLD).to(dtype=torch.half)

        if self.verbose:
            vessel_ratio = ((cleaned > 0.5).sum() / cleaned.numel()).item()
            print(f"Sparse search complete: Vessel region {vessel_ratio*100:.1f}%")

        # Extract single BBox (whole if empty)
        full_size = torch.tensor(target_size, device=cleaned.device)
        # Create BBox from binary mask obtained by sparse search and add sparse search margin
        bbox = self._bbox_from_mask(cleaned > 0.5, int(self.sparse_bbox_margin_voxels), full_size)
        if bbox is None:
            return tuple(slice(0, s) for s in target_size)
        mins = torch.tensor([s.start for s in bbox], device=cleaned.device)
        maxs = torch.tensor([s.stop - 1 for s in bbox], device=cleaned.device)

        # From here: Suppress excessive ROI vertical (SI) expansion
        # - If scan range includes shoulders, false detections cause ROI to extend too much vertically, making dense search heavy
        # - If exceeding limit (mm), trim around mask center of mass (COM)
        if self.limit_si_extent and cleaned.ndim == 3:
            # Get spacing (network space). No need for 2D config compatibility (3D only)
            try:
                target_spacing = list(self.configuration_manager.spacing)
            except Exception:
                target_spacing = [1.0, 1.0, 1.0]
            # Dimension adjustment (duplicate to head if length doesn't match)
            while len(target_spacing) < cleaned.ndim:
                target_spacing = [target_spacing[0]] + target_spacing

            # Determine SI axis
            if self.si_axis is not None and 0 <= int(self.si_axis) < cleaned.ndim:
                si_ax = int(self.si_axis)
            else:
                # After nnU-Net preprocessing transposition, the first spatial axis (0th) is assumed to be basically the SI axis
                # (Because high resolution = small spacing axis comes later). Does not depend on spacing magnitude.
                si_ax = 0

            # Current ROI SI length (mm)
            extent_vox = (maxs - mins + 1).to(dtype=torch.int64)
            extent_mm = float(extent_vox[si_ax].item()) * float(target_spacing[si_ax])

            if extent_mm > float(self.max_si_extent_mm):
                # Occupancy profile per slice (A_z): Mask ratio of each slice along SI axis (calculated on GPU)
                mask_bool = cleaned > 0.5
                inplane_axes = [ax for ax in range(3) if ax != si_ax]
                A = mask_bool.float().mean(dim=inplane_axes)  # shape=(Z,)

                # Calculate trim range (mm limit) based on center of mass (COM) on GPU
                idxs = torch.arange(A.shape[0], device=A.device, dtype=A.dtype)
                mass = A.sum()
                if mass > 0:
                    com_t = (A * idxs).sum() / mass
                else:
                    com_t = (mins[si_ax].to(A.dtype) + maxs[si_ax].to(A.dtype)) * 0.5

                half_extent_vox = int(
                    np.ceil(0.5 * float(self.max_si_extent_mm) / float(target_spacing[si_ax]))
                )
                z_center = int(torch.round(com_t).item())
                z0 = max(0, z_center - half_extent_vox)
                z1 = min(int(full_size[si_ax].item()) - 1, z_center + half_extent_vox)

                # Intersection with original ROI (take intersection first to prevent excessive cutting)
                new_min_z = max(int(mins[si_ax].item()), z0)
                new_max_z = min(int(maxs[si_ax].item()), z1)
                if new_max_z <= new_min_z:
                    # Adopt COM-centered range if intersection is extremely small
                    new_min_z, new_max_z = z0, z1

                mins = mins.clone()
                maxs = maxs.clone()
                mins[si_ax] = int(new_min_z)
                maxs[si_ax] = int(new_max_z)

                if self.verbose:
                    after_extent_mm = (maxs[si_ax] - mins[si_ax] + 1).item() * float(target_spacing[si_ax])
                    print(
                        f"SI guard applied: Axis{si_ax}, {extent_mm:.1f}mm -> {after_extent_mm:.1f}mm (Limit {self.max_si_extent_mm}mm)"
                    )

        # From here: Guarantee ROI minimum length (mm) (expand just before BBox confirmation)
        # - Ensure minimum min_si_extent_mm for SI axis and min_xy_extent_mm for other spatial axes
        # - Expand symmetrically to keep center as much as possible, and distribute remainder restricted by image boundary to the opposite side
        try:
            target_spacing_min = list(self.configuration_manager.spacing)
        except Exception:
            # Assume 1mm if spacing cannot be obtained
            target_spacing_min = [1.0] * cleaned.ndim
        while len(target_spacing_min) < cleaned.ndim:
            target_spacing_min = [target_spacing_min[0]] + target_spacing_min

        # Redetermine SI axis (3D only). Treat all axes as XY for 2D
        if cleaned.ndim == 3:
            if self.si_axis is not None and 0 <= int(self.si_axis) < cleaned.ndim:
                si_ax_for_min = int(self.si_axis)
            else:
                si_ax_for_min = 0
        else:
            si_ax_for_min = None

        for ax in range(cleaned.ndim):
            # Target minimum length per axis (mm)
            min_len_mm = (
                self.min_si_extent_mm
                if (si_ax_for_min is not None and ax == si_ax_for_min)
                else self.min_xy_extent_mm
            )
            # Convert to required voxel count
            desired_vox = int(np.ceil(float(min_len_mm) / float(target_spacing_min[ax])))
            current_vox = int((maxs[ax] - mins[ax] + 1).item())
            if desired_vox <= 1 or current_vox >= desired_vox:
                continue

            # Expand left/right based on center
            need = desired_vox - current_vox
            left = need // 2
            right = need - left
            new_min = int(max(0, int(mins[ax].item()) - left))
            new_max = int(min(int(full_size[ax].item()) - 1, int(maxs[ax].item()) + right))

            # Redistribute shortage due to boundary restriction to opposite side
            final_extent = new_max - new_min + 1
            if final_extent < desired_vox:
                remaining = desired_vox - final_extent
                # Shift to left as much as possible
                shift_left = min(remaining, new_min)
                new_min = new_min - shift_left
                remaining -= shift_left
                if remaining > 0:
                    # Shift to right
                    max_allow = int(full_size[ax].item()) - 1
                    available_right = max_allow - new_max
                    shift_right = min(remaining, available_right)
                    new_max = new_max + shift_right

            # Update
            mins[ax] = int(new_min)
            maxs[ax] = int(new_max)

        roi_bbox = tuple(slice(int(mins[i].item()), int(maxs[i].item()) + 1) for i in range(len(mins)))

        if return_context:
            return roi_bbox, sparse_context if sparse_context is not None else {}
        return roi_bbox

    def load_model_for_inference(
        self,
        model_training_output_dir: str,
        fold: Optional[int] = 0,
        checkpoint_name: str = "checkpoint_final.pth",
        torch_compile: bool = False,
    ) -> None:
        # For Kaggle: Load only single fold model
        # Args:
        #     model_training_output_dir: Directory where model is saved
        #     fold: Fold number to use
        #     checkpoint_name: Checkpoint filename
        if fold is None:
            fold = 0
        print(f"Loading model (fold: {fold})...")

        # Load dataset and plans
        self.dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        self.plans_manager = PlansManager(plans)

        # Load parameters (single fold)
        checkpoint = torch.load(
            join(model_training_output_dir, f"fold_{fold}", checkpoint_name),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        trainer_name = checkpoint["trainer_name"]
        configuration_name = checkpoint["init_args"]["configuration"]
        # Save identifier name to instance as well
        self.trainer_name = str(trainer_name)
        self.configuration_name = str(configuration_name)
        self.allowed_mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes", None)
        self.network_weights = checkpoint["network_weights"]

        self.configuration_manager = self.plans_manager.get_configuration(configuration_name)

        # Execute resampling with torch gpu
        self.configuration_manager.configuration["resampling_fn_data"] = "resample_torch_fornnunet"
        self.configuration_manager.configuration["resampling_fn_seg"] = "resample_torch_fornnunet"
        self.configuration_manager.configuration["resampling_fn_probabilities"] = "resample_torch_fornnunet"
        self.configuration_manager.configuration["resampling_fn_data_kwargs"] = {
            "is_seg": False,
            "device": "cuda",
            "force_separate_z": None,
        }
        self.configuration_manager.configuration["resampling_fn_seg_kwargs"] = {
            "is_seg": True,
            "device": "cuda",
            "force_separate_z": None,
        }
        self.configuration_manager.configuration["resampling_fn_probabilities_kwargs"] = {
            "is_seg": False,
            "device": "cuda",
            "force_separate_z": None,
        }

        # Initialize label manager
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)

        # Restore network
        num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            "nnunetv2.training.nnUNetTrainer",
        )

        if trainer_class is None:
            raise RuntimeError(f"Trainer class '{trainer_name}' not found")

        self.network = trainer_class.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            self.label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )

        self.network = self.network.to(self.device)
        # Enable fp16 network weights for CUDA inference to reduce memory/IO
        if self.device.type == "cuda":
            self.network = self.network.half()

        # Load network weights (into half-precision GPU optimized network)
        self.network.load_state_dict(self.network_weights)

        if torch_compile:
            # Small effect in nnUNet, but enabled locally
            # Disabled in Kaggle environment as it causes errors
            self.network = torch.compile(self.network)

        print("Model loading complete: Single fold")

    def enable_tensorrt(self, engine_path: str, *, enforce_half_output: bool = True) -> None:
        # Load TensorRT engine and switch inference to TRT
        # Args:
        #     engine_path: Path to .engine file
        #     enforce_half_output: Return output converted to half precision (consistent with nnUNet internal aggregation dtype)
        # Bind engine according to execution device
        dev = self.device
        try:
            # If torch.device("cuda"), index is None -> get current_device
            if dev.type == "cuda" and dev.index is None:
                dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        except Exception:
            pass
        self.trt_runner = TRTRunner(engine_path, device=dev)
        self.trt_enforce_half_output = bool(enforce_half_output)
        if self.verbose:
            print(f"TensorRT engine enabled: {engine_path}")

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        # Replace network call considering mirroring (TTA) if TRT is enabled
        if self.trt_runner is None:
            # Conventional PyTorch path
            return super()._internal_maybe_mirror_and_predict(x)

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        # First, raw inference
        prediction = self.trt_runner.run(x, enforce_half_output=self.trt_enforce_half_output)

        if mirror_axes is not None:
            # x is 4D (2D image) or 5D (3D image). Mirror axes are spatial axes (0-based), so
            # add +2 to match tensor dimensions and skip N, C
            assert len(x.shape) in (4, 5), "Input tensor is expected to be 4D or 5D"
            assert max(mirror_axes) <= x.ndim - 3, "mirror_axes does not match the dimension of the input!"
            axes_adj = [m + 2 for m in mirror_axes]

            import itertools

            axes_combinations = [
                c for i in range(len(axes_adj)) for c in itertools.combinations(axes_adj, i + 1)
            ]
            for axes in axes_combinations:
                x_flipped = torch.flip(x, axes)
                y = self.trt_runner.run(x_flipped, enforce_half_output=self.trt_enforce_half_output)
                prediction += torch.flip(y, axes)
            prediction /= len(axes_combinations) + 1
        return prediction

    def _preprocess_data(
        self,
        image: np.ndarray,
        properties: dict,
        seg_from_prev_stage: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Execute data preprocessing
        # Args:
        #     image: Input image
        #     properties: Image properties
        #     seg_from_prev_stage: Segmentation from previous stage
        # Returns:
        #     data: Preprocessed data
        #     preprocessed: Preprocessing result
        preprocessor = PreprocessAdapterFromNpy(
            [image],
            [seg_from_prev_stage] if seg_from_prev_stage is not None else None,
            [properties],
            [None],  # output_file_truncated
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_threads_in_multithreaded=1,
            verbose=False,
        )

        preprocessed = next(preprocessor)

        return preprocessed

    def _determine_roi_from_sparse_search(self, data: torch.Tensor) -> Optional[Tuple[slice, ...]]:
        # Determine ROI using sparse search
        # Args:
        #     data: Input data
        # Returns:
        #     roi_bbox: ROI bounding box (None if not present)
        if not self.should_use_sparse_search(data.shape):
            return None

        # Execute sparse search and get single ROI
        roi_bbox = self.sparse_search(data)

        if self.verbose and roi_bbox is not None:
            roi_size = tuple(s.stop - s.start for s in roi_bbox)
            print(f"ROI determined: Size {roi_size}")

        return roi_bbox

    def _predict_logits(
        self, data: torch.Tensor, roi_bbox: Optional[Tuple[slice, ...]] = None
    ) -> torch.Tensor:
        # Inference with single fold (detailed inference only for that region if ROI is specified)
        if roi_bbox is not None:
            results_device = self.device if self.perform_everything_on_device else torch.device("cpu")
            pred_full = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=(torch.half if self.device.type == "cuda" else torch.float32),
                device=results_device,
            )
            roi_slices = (slice(None),) + roi_bbox
            roi_data = data[roi_slices]
            # Convert overlap rate to step size even during dense search
            original_tile_step_size = self.tile_step_size
            try:
                self.tile_step_size = 1.0 - float(self.dense_overlap)
                if self.device.type == "cuda":
                    roi_data = roi_data.half()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        roi_logits = self.predict_sliding_window_return_logits(roi_data)
                else:
                    roi_logits = self.predict_sliding_window_return_logits(roi_data)
            finally:
                self.tile_step_size = original_tile_step_size
            pred_full[(slice(None),) + roi_bbox] = roi_logits
            return pred_full
        else:
            original_tile_step_size = self.tile_step_size
            try:
                self.tile_step_size = 1.0 - float(self.dense_overlap)
                if self.device.type == "cuda":
                    data_cast = data.half()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.predict_sliding_window_return_logits(data_cast)
                else:
                    logits = self.predict_sliding_window_return_logits(data)
            finally:
                self.tile_step_size = original_tile_step_size
        return logits


class DBSCANAdaptiveSparsePredictor(AdaptiveSparsePredictor):
    """Sparse search inference class introducing ROI extraction by DBSCAN"""

    def __init__(
        self,
        *args,
        dbscan_eps_voxels: float = 20.0,
        dbscan_min_samples: int = 100,
        dbscan_max_points: int = 60000,
        roi_extent_mm: Union[float, Tuple[float, ...], List[float]] = (130.0, 130.0, 130.0),
        **kwargs,
    ) -> None:
        # Args:
        #     dbscan_eps_voxels: DBSCAN epsilon (in voxels)
        #     dbscan_min_samples: Minimum neighbors for core point determination
        #     dbscan_max_points: Maximum points to cluster
        #     roi_extent_mm: ROI physical size (mm). Specified as single value or sequence per axis
        super().__init__(*args, **kwargs)
        self.dbscan_eps_voxels = float(dbscan_eps_voxels)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.dbscan_max_points = int(dbscan_max_points)
        if isinstance(roi_extent_mm, (tuple, list)):
            self.dbscan_roi_extent_mm = tuple(float(x) for x in roi_extent_mm)
        else:
            self.dbscan_roi_extent_mm = (float(roi_extent_mm),)

    @staticmethod
    def _dbscan_labels(coords: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Get cluster labels with DBSCAN"""
        # DBSCAN cannot be used in environments where scikit-learn is not installed.
        # Throw error explicitly only when using DBSCAN path.
        if SklearnDBSCAN is None:  # type: ignore[truthy-function]
            raise ImportError(
                "scikit-learn not found. This is fine if not using DBSCAN-based sparse search."
                "Please install scikit-learn if using DBSCAN."
            )
        if coords.size == 0:
            return np.empty((0,), dtype=np.int32)

        clustering = SklearnDBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = clustering.fit_predict(coords.astype(np.float32, copy=False))
        return labels.astype(np.int32, copy=False)

    @torch.inference_mode()
    def sparse_search(
        self, input_image: torch.Tensor, return_context: bool = False
    ) -> Union[Tuple[slice, ...], Tuple[Tuple[slice, ...], Dict[str, Any]]]:
        # roi_bbox = tuple(slice(0, s) for s in tuple(int(x) for x in input_image.shape[1:]))
        # if return_context:
        #     return roi_bbox, {}
        # return roi_bbox

        # Fixed-size ROI sparse search using DBSCAN cluster centroids
        if self.verbose:
            print("Starting sparse search (DBSCAN) phase...")

        downscale_factors = [1] + [1 / self.sparse_downscale_factor] * (input_image.ndim - 1)
        downsampled = torch.nn.functional.interpolate(
            input_image.unsqueeze(0),
            scale_factor=downscale_factors[1:],
            mode="trilinear" if input_image.ndim == 4 else "bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.verbose:
            print(f"Downsampling: {input_image.shape} -> {downsampled.shape}")

        target_size = tuple(int(x) for x in input_image.shape[1:])
        original_tile_step_size = self.tile_step_size

        try:
            self.tile_step_size = 1.0 - self.sparse_overlap
            if self.device.type == "cuda":
                downsampled = downsampled.to(device=self.device, dtype=torch.half)
                with torch.autocast(device_type="cuda"):
                    sparse_logits = self.predict_sliding_window_return_logits(downsampled)
            else:
                sparse_logits = self.predict_sliding_window_return_logits(downsampled)
        finally:
            self.tile_step_size = original_tile_step_size

        sparse_probs = torch.softmax(sparse_logits, dim=0)

        # SI direction guard (applied only within DBSCAN sparse search)
        # - If SI axis physical length is largest among axes and exceeds max_si_extent_mm,
        #   keep only the top vox_allow of the SI axis and treat the rest as background
        if self.si_axis is not None and sparse_probs.dim() >= 2:
            try:
                si_ax = int(self.si_axis)
            except Exception:
                si_ax = -1
            spatial_nd = sparse_probs.dim() - 1
            if 0 <= si_ax < spatial_nd:
                try:
                    spacing = list(self.configuration_manager.spacing)  # type: ignore[attr-defined]
                except Exception:
                    spacing = [1.0] * spatial_nd
                while len(spacing) < spatial_nd:
                    spacing = [spacing[0]] + spacing
                spacing = spacing[:spatial_nd]

                extents_mm = [float(target_size[i]) * float(spacing[i]) for i in range(spatial_nd)]
                si_extent_mm = float(extents_mm[si_ax])
                max_extent_mm = float(max(extents_mm)) if extents_mm else 0.0

                if si_extent_mm >= max_extent_mm - 1e-6 and si_extent_mm > float(self.max_si_extent_mm):
                    eff_spacing_lowres = float(spacing[si_ax]) * float(self.sparse_downscale_factor)
                    vox_allow = int(np.ceil(float(self.max_si_extent_mm) / max(1e-6, eff_spacing_lowres)))
                    full_len = int(sparse_probs.shape[1 + si_ax])
                    if 0 < vox_allow < full_len:
                        # Keep original probability for restoration (low resolution). Restore within COM-centered allowable window eventually
                        sparse_probs_backup = sparse_probs.clone()
                        keep_start = full_len - vox_allow
                        sl = [slice(None)] * sparse_probs.dim()
                        sl[1 + si_ax] = slice(0, keep_start)
                        sparse_probs[(slice(1, None),) + tuple(sl[1:])] = 0.0
                        sparse_probs[(0,) + tuple(sl[1:])] = 1.0
                        if self.verbose:
                            print(
                                f"DBSCAN-sparse SI guard: Axis{si_ax}, physical length {si_extent_mm:.1f}mm (max {max_extent_mm:.1f}mm) > limit {float(self.max_si_extent_mm):.1f}mm\n"
                                f" -> Set lower {keep_start} voxels to background in low resolution"
                            )

                        # Additional safety measure: Trim with central window using center of mass in SI direction
                        # - Keep only max_si_extent_mm range around center of mass (COM) and set others to background
                        try:
                            if sparse_probs.shape[0] > 1:
                                vp = torch.max(sparse_probs[1:], dim=0)[0]
                            else:
                                vp = sparse_probs[0]
                            # Sum axes other than SI axis to 1D distribution
                            reduce_axes = tuple(i for i in range(vp.dim()) if i != si_ax)
                            prof = vp.sum(dim=reduce_axes)
                            mass = prof.sum()
                            if mass > 0:
                                idxs = torch.arange(full_len, device=vp.device, dtype=vp.dtype)
                                com_t = (prof * idxs).sum() / mass
                            else:
                                com_t = torch.tensor((full_len - 1) * 0.5, device=vp.device, dtype=vp.dtype)

                            half_extent = int(
                                np.ceil(0.5 * float(self.max_si_extent_mm) / max(1e-6, eff_spacing_lowres))
                            )
                            c_idx = int(torch.round(com_t).item())
                            z0 = max(0, c_idx - half_extent)
                            z1 = min(full_len - 1, c_idx + half_extent)
                            if z1 <= z0:
                                z0 = max(0, (full_len // 2) - half_extent)
                                z1 = min(full_len - 1, z0 + 2 * half_extent)

                            # Set outside of COM-centered window to background ([0:z0) and (z1+1:full_len))
                            if z0 > 0:
                                sl_pre = [slice(None)] * sparse_probs.dim()
                                sl_pre[1 + si_ax] = slice(0, z0)
                                sparse_probs[(slice(1, None),) + tuple(sl_pre[1:])] = 0.0
                                sparse_probs[(0,) + tuple(sl_pre[1:])] = 1.0
                            if z1 + 1 < full_len:
                                sl_post = [slice(None)] * sparse_probs.dim()
                                sl_post[1 + si_ax] = slice(z1 + 1, full_len)
                                sparse_probs[(slice(1, None),) + tuple(sl_post[1:])] = 0.0
                                sparse_probs[(0,) + tuple(sl_post[1:])] = 1.0

                            # Restore original sparse search probability within COM allowable window [z0:z1] (restore even if cut at lower side earlier)
                            restore_sl = [slice(None)] * sparse_probs.dim()
                            restore_sl[1 + si_ax] = slice(z0, z1 + 1)
                            sparse_probs[(slice(None),) + tuple(restore_sl[1:])] = sparse_probs_backup[
                                (slice(None),) + tuple(restore_sl[1:])
                            ]

                            if self.verbose:
                                print(
                                    f"DBSCAN-sparse SI guard (COM adjustment): COM={c_idx}, range[{z0}:{z1}] vox, half-width {half_extent}vox — Restore allowable window to original prediction"
                                )
                        except Exception:
                            # Continue inference even if COM calculation fails
                            pass

        sparse_context: Optional[Dict[str, Any]] = None
        if return_context:
            seg_lowres = torch.argmax(sparse_probs, dim=0)
            seg_lowres_cpu = seg_lowres.detach().to(device="cpu", dtype=torch.int16)
            seg_highres = torch.nn.functional.interpolate(
                seg_lowres[None, None].to(dtype=torch.float32),
                size=target_size,
                mode="nearest",
            )[0, 0]
            seg_highres_cpu = seg_highres.detach().to(device="cpu", dtype=torch.int16)
            sparse_context = {
                "segmentation_lowres": seg_lowres_cpu,
                "segmentation_highres": seg_highres_cpu,
                "num_classes": int(sparse_logits.shape[0]),
                "input_shape": tuple(int(x) for x in input_image.shape),
                "downsampled_shape": tuple(int(x) for x in downsampled.shape),
                "network_spacing": tuple(
                    float(x) for x in getattr(self.configuration_manager, "spacing", [])
                ),
            }
            del seg_lowres, seg_highres

        if sparse_probs.shape[0] > 1:
            vessel_probs = torch.max(sparse_probs[1:], dim=0)[0]
        else:
            vessel_probs = sparse_probs[0]

        mask_bool_lowres = vessel_probs > 0.8

        if mask_bool_lowres.sum() == 0:
            roi_bbox = tuple(slice(0, s) for s in target_size)
            if return_context:
                return roi_bbox, sparse_context if sparse_context is not None else {}
            return roi_bbox

        coords = torch.nonzero(mask_bool_lowres, as_tuple=False)
        vessel_probs_cpu = vessel_probs.detach().to(device="cpu")
        coords_cpu = coords.detach().to(device="cpu")
        point_values = vessel_probs_cpu[tuple(coords_cpu.t())]

        total_points = coords_cpu.shape[0]
        if total_points > self.dbscan_max_points:
            topk = torch.topk(point_values, k=self.dbscan_max_points)
            coords_cpu = coords_cpu[topk.indices]
            point_values = topk.values

        coords_np = coords_cpu.numpy().astype(np.float32, copy=False)
        values_np = point_values.numpy().astype(np.float32, copy=False)

        labels_np = self._dbscan_labels(coords_np, self.dbscan_eps_voxels, self.dbscan_min_samples)
        unique_labels = [lab for lab in np.unique(labels_np) if lab >= 0]

        cluster_summaries: List[Dict[str, Any]] = []
        for lab in unique_labels:
            member_idx = np.where(labels_np == lab)[0]
            if member_idx.size == 0:
                continue
            m_coords = coords_np[member_idx]
            m_vals = values_np[member_idx]
            wsum = float(np.sum(m_vals))
            if wsum > 0.0:
                centroid_lowres = (m_coords * m_vals[:, None]).sum(axis=0) / wsum
            else:
                centroid_lowres = m_coords.mean(axis=0)
            score = float(np.sum(m_vals))
            cluster_summaries.append(
                {
                    "label": int(lab),
                    "score": score,
                    "count": int(member_idx.size),
                    "centroid_lowres": centroid_lowres,
                }
            )

        if not cluster_summaries:
            selected_highres = torch.nn.functional.interpolate(
                mask_bool_lowres.float()[None, None], size=target_size, mode="nearest"
            )[0, 0]
            selected_mask_highres = (selected_highres > 0.5).to(dtype=torch.half)
            full_size = torch.tensor(target_size, device=selected_mask_highres.device)
            bbox = self._bbox_from_mask(
                selected_mask_highres > 0.5, int(self.sparse_bbox_margin_voxels), full_size
            )
            roi_bbox = bbox if bbox is not None else tuple(slice(0, s) for s in target_size)
            if return_context:
                return roi_bbox, sparse_context if sparse_context is not None else {}
            return roi_bbox

        # Orientation-independent cluster selection: Prioritize density, confidence, size in order
        def _cluster_metrics(item: Dict[str, Any]) -> Tuple[float, float, float]:
            lab = int(item["label"])  # Cluster label
            member_idx = np.where(labels_np == lab)[0]
            if member_idx.size == 0:
                return (0.0, 0.0, 0.0)
            pts = coords_np[member_idx]
            vals = values_np[member_idx]
            # Average confidence of top K (prioritize locally dense clusters over over-dispersed large clusters)
            k = int(min(256, vals.shape[0]))
            conf = float(np.mean(np.partition(vals, -k)[-k:])) if k > 0 else 0.0
            size = float(pts.shape[0])
            return (conf, size)

        best_cluster = max(cluster_summaries, key=_cluster_metrics)
        centroid_lowres = best_cluster["centroid_lowres"]

        lowres_shape = np.array(mask_bool_lowres.shape, dtype=np.float32)
        scale = np.divide(
            np.array(target_size, dtype=np.float32),
            lowres_shape,
            out=np.ones_like(lowres_shape),
            where=lowres_shape > 0,
        )
        center_highres = centroid_lowres * scale
        center_highres = np.clip(center_highres, 0.0, np.array(target_size, dtype=np.float32) - 1.0)

        try:
            spacing = list(self.configuration_manager.spacing)
        except Exception:
            spacing = [1.0] * len(target_size)
        while len(spacing) < len(target_size):
            spacing = [spacing[0]] + spacing
        spacing = spacing[: len(target_size)]

        if len(self.dbscan_roi_extent_mm) == 1:
            extents_mm: List[float] = [self.dbscan_roi_extent_mm[0]] * len(target_size)
        else:
            extents_mm = list(self.dbscan_roi_extent_mm[: len(target_size)])
            if len(extents_mm) < len(target_size):
                extents_mm.extend([extents_mm[-1]] * (len(target_size) - len(extents_mm)))

        roi_slices: List[slice] = []
        for axis, size_axis in enumerate(target_size):
            spacing_axis = float(spacing[axis]) if axis < len(spacing) else 1.0
            desired_mm = float(extents_mm[axis]) if axis < len(extents_mm) else float(extents_mm[-1])
            if desired_mm <= 0.0 or spacing_axis <= 0.0:
                desired_vox = size_axis
            else:
                desired_vox = int(np.ceil(desired_mm / spacing_axis))
            desired_vox = max(1, min(desired_vox, size_axis))

            center_vox = float(center_highres[axis])
            start = int(round(center_vox - desired_vox / 2.0))
            max_start = max(0, size_axis - desired_vox)
            start = max(0, min(start, max_start))
            end = start + desired_vox
            if end > size_axis:
                end = size_axis
                start = max(0, end - desired_vox)
            roi_slices.append(slice(start, end))

        roi_bbox = tuple(roi_slices)

        if return_context:
            cluster_meta = {
                "score": best_cluster["score"],
                "count": best_cluster["count"],
                "center_vox": [float(center_highres[i]) for i in range(len(center_highres))],
            }
            if sparse_context is None:
                sparse_context = {}
            sparse_context["dbscan_cluster"] = cluster_meta
            # Zero out predictions other than selected cluster (orientation independent)
            try:
                best_lab = int(best_cluster.get("label", -1))
                # Create low-resolution cluster mask
                selected_idx = np.where(labels_np == best_lab)[0]
                if selected_idx.size > 0:
                    # Keep mask as CPU tensor
                    sel_mask_low = torch.zeros(mask_bool_lowres.shape, dtype=torch.bool, device="cpu")
                    sel_coords = coords_cpu[selected_idx].long()
                    if sel_coords.numel() > 0:
                        sel_mask_low[tuple(sel_coords.t())] = True

                    # Apply zeroing if segmentation_lowres exists
                    if isinstance(sparse_context.get("segmentation_lowres"), torch.Tensor):
                        seg_lr = sparse_context["segmentation_lowres"].clone()
                        seg_lr[~sel_mask_low] = 0
                        sparse_context["segmentation_lowres"] = seg_lr

                    # Nearest neighbor upsampling to high-resolution mask and zeroing
                    if isinstance(sparse_context.get("segmentation_highres"), torch.Tensor):
                        high_shape = tuple(int(x) for x in target_size)
                        sel_mask_high = torch.nn.functional.interpolate(
                            sel_mask_low.float()[None, None], size=high_shape, mode="nearest"
                        )[0, 0].to(dtype=torch.bool)
                        seg_hr = sparse_context["segmentation_highres"].clone()
                        seg_hr[~sel_mask_high] = 0
                        sparse_context["segmentation_highres"] = seg_hr
            except Exception:
                # Continue inference even if failed
                pass
            return roi_bbox, sparse_context
        return roi_bbox

    def predict_single_npy_array(self) -> np.ndarray:
        raise NotImplementedError("predict_single_npy_array is not supported")

    def predict_from_list_of_npy_arrays(self) -> List[np.ndarray]:
        raise NotImplementedError("predict_from_list_of_npy_arrays is not supported")
