from typing import Sequence, Mapping, MutableMapping, Tuple, Union, Dict, Optional, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor
from monai.transforms import MapTransform, Randomizable, MedianSmoothd

# If you add more above, adjust the imports accordingly.

PairInt = Tuple[int, int]
RangeOrPerAxis = Union[PairInt, Sequence[PairInt]]


class RandMedianSmoothdVaried(MapTransform, Randomizable):
    """
    Apply MedianSmoothd to keys with probability p.
    The radius is sampled per axis.

    Args:
        keys: target keys (e.g., ["image"]).
        prob: apply probability (0.0–1.0).
        radius_range: per-axis radius ranges. Accepts either:
            - (low, high): use this for all spatial axes
            - [(l0,h0), (l1,h1), ...]: range per axis (len == spatial dims)
            low/high are non-negative ints with low <= high.
        allow_missing_keys: whether to ignore missing keys.
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.5,
        radius_range: RangeOrPerAxis = (0, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        Randomizable.__init__(self)
        self.prob = float(prob)
        # Keep the user-specified ranges (validated at call time)
        self.radius_range = radius_range
        self._chosen_radius: Tuple[int, ...] | None = None

    def _validate_and_expand_ranges(self, spatial_dim: int) -> Sequence[PairInt]:
        # If a single pair is given, copy to all axes; if a sequence, check length
        if (
            isinstance(self.radius_range, tuple)
            and len(self.radius_range) == 2
            and isinstance(self.radius_range[0], int)
        ):
            return [self.radius_range] * spatial_dim
        # Treat as a sequence
        rngs = list(self.radius_range)  # type: ignore
        if len(rngs) != spatial_dim:
            raise ValueError(
                f"radius_range length ({len(rngs)}) must match spatial dim ({spatial_dim}) or be a single (low,high) pair."
            )
        for i, pr in enumerate(rngs):
            if not (
                isinstance(pr, tuple) and len(pr) == 2 and isinstance(pr[0], int) and isinstance(pr[1], int)
            ):
                raise ValueError(f"radius_range[{i}] must be a pair of ints (low, high). got: {pr}")
            if pr[0] < 0 or pr[1] < pr[0]:
                raise ValueError(f"invalid range for axis {i}: {pr}")
        return rngs

    def randomize(self, spatial_dim: int) -> None:
        # Decide whether to apply
        self._do_transform = self.R.random() < self.prob
        self._chosen_radius = None
        if not self._do_transform:
            return
        # Determine per-axis ranges and sample
        rngs = self._validate_and_expand_ranges(spatial_dim)
        chosen = []
        for low, high in rngs:
            if low == high:
                chosen.append(low)
            else:
                # numpy RandomState randint uses exclusive high -> add 1
                chosen.append(int(self.R.randint(low, high + 1)))
        self._chosen_radius = tuple(chosen)

    def __call__(self, data: Mapping[str, np.ndarray]) -> MutableMapping[str, np.ndarray]:
        d = dict(data)
        # Determine spatial dims from the first present key
        found_key = None
        for key in self.keys:
            if key in d:
                found_key = key
                break
        if found_key is None:
            return d  # Behavior consistent with allow_missing_keys (handled by MapTransform)

        img = d[found_key]
        # Expected: (C,H,W) or (C,D,H,W)
        if img.ndim < 3:
            # Minimal handling for rare cases without an explicit channel dim
            spatial_dim = img.ndim - 1 if img.ndim >= 1 else 0
        else:
            spatial_dim = img.ndim - 1

        # Decide application and sample radii
        self.randomize(spatial_dim)
        if not self._do_transform or self._chosen_radius is None:
            return d

        # Create MONAI MedianSmoothd on the fly and run (pass tuple radius)
        smoother = MedianSmoothd(
            keys=self.keys, radius=self._chosen_radius, allow_missing_keys=self.allow_missing_keys
        )
        return smoother(d)


class InvertImageTransform(MapTransform, Randomizable):
    """
    Image intensity inversion.

    Args:
        keys: keys to transform
        prob: probability to apply to the whole image
        p_per_channel: probability per channel
        p_synchronize_channels: if True, invert all channels synchronously
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.2,
        p_per_channel: float = 0.5,
        p_synchronize_channels: float = 0.5,
    ) -> None:
        super().__init__(keys)
        self.prob = float(prob)
        self.p_per_channel = float(p_per_channel)
        self.p_sync = float(p_synchronize_channels)

    def randomize(self) -> None:
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.keys:
            if key not in d:
                continue
            img = d[key]
            if img.ndim != 4:
                continue
            out = img.astype(np.float32, copy=True)
            c = out.shape[0]

            if self.R.random() < self.p_sync:
                # Decide for all channels together
                if self.R.random() < self.p_per_channel:
                    pmin = float(np.min(out))
                    pmax = float(np.max(out))
                    out = (pmax + pmin) - out
            else:
                # Decide per channel
                for ch in range(c):
                    if self.R.random() < self.p_per_channel:
                        pmin = float(out[ch].min())
                        pmax = float(out[ch].max())
                        out[ch] = (pmax + pmin) - out[ch]

            d[key] = out
        return d


class RandInvertAroundMeanD(MapTransform, Randomizable):
    """
    Stochastic transform that inverts around the per-channel mean.

    Args:
        keys: keys to transform
        prob: probability to apply the transform
        p_per_channel: probability per channel
        p_synchronize_channels: if True, share the decision across channels
        allow_missing_keys: whether to ignore missing keys
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.2,
        p_per_channel: float = 0.5,
        p_synchronize_channels: float = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        Randomizable.__init__(self)
        self.prob = float(prob)
        self.p_per_channel = float(p_per_channel)
        self.p_sync = float(p_synchronize_channels)

    def randomize(self) -> None:
        self._do_transform = self.R.random() < self.prob

    def _select_channels(self, num_channels: int) -> List[int]:
        if num_channels <= 0:
            return []
        if self.R.random() < self.p_sync:
            return list(range(num_channels)) if self.R.random() < self.p_per_channel else []
        mask = self.R.random_sample(num_channels) < self.p_per_channel
        return [idx for idx, flag in enumerate(mask) if flag]

    def _invert_tensor(self, tensor: torch.Tensor, channels: List[int]) -> torch.Tensor:
        if not channels:
            return tensor
        work = tensor.to(torch.float32)
        for ch in channels:
            slice_view = work[ch]
            mean_val = slice_view.mean()
            work[ch] = (2.0 * mean_val) - slice_view
        return work.to(tensor.dtype)

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d

        for key in self.keys:
            if key not in d:
                if not self.allow_missing_keys:
                    raise KeyError(f"key '{key}' not found in input dict.")
                continue
            img = d[key]
            if not torch.is_tensor(img):
                tensor = torch.as_tensor(img)
                is_meta = False
            else:
                tensor = img
                is_meta = isinstance(img, MetaTensor)

            if tensor.ndim < 2:
                continue
            channels = tensor.shape[0]
            target_channels = self._select_channels(channels)
            if not target_channels:
                continue

            updated = self._invert_tensor(tensor, target_channels)
            if isinstance(img, np.ndarray):
                d[key] = updated.cpu().numpy().astype(img.dtype, copy=False)
            elif is_meta:
                img.copy_(updated)
                d[key] = img
            elif torch.is_tensor(img):
                img.copy_(updated)
                d[key] = img
            else:
                d[key] = updated

        return d


class Resize3DFitPadCropd(MapTransform):
    """
    Fit a 3D volume to the target size, then standardize by padding/cropping.
    - keep_ratio="z-xy": separate scales for Z and (Y,X), preserve in-plane ratio (recommended)
    - keep_ratio="uniform": same scale across Z,Y,X (preserves volumetric ratio)
    - keep_ratio="per-axis": scale each axis independently (easiest sizing, most distortion)
    """

    def __init__(
        self,
        keys: Sequence[str],
        target_size: Tuple[int, int, int],  # (D,H,W)
        keep_ratio: str = "z-xy",
        image_keys: Optional[Sequence[str]] = ("image",),
        mask_keys: Optional[Sequence[str]] = ("sphere_mask",),
        point_keys: Optional[Sequence[str]] = None,
        pad_value_image: float = 0.0,
        pad_value_mask: int = 0,
        scale_clip: Optional[Tuple[float, float]] = None,  # e.g., (0.5, 2.0)
    ):
        super().__init__(keys)
        self.target = target_size
        self.keep_ratio = keep_ratio
        self.image_keys = set(image_keys or [])
        self.mask_keys = set(mask_keys or [])
        # Keys that hold 3D point coordinates (N,3) in (z,y,x)
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()
        self.pad_value_image = float(pad_value_image)
        self.pad_value_mask = int(pad_value_mask)
        self.scale_clip = scale_clip

    @staticmethod
    def _interp(x: torch.Tensor, size, mode):
        if mode in ("trilinear", "linear", "bilinear"):
            return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return F.interpolate(x, size=size, mode="nearest")

    def _compute_new_size(self, shp: Tuple[int, int, int]) -> Tuple[int, int, int]:
        D, H, W = shp
        Dt, Ht, Wt = self.target
        if self.keep_ratio == "uniform":
            s = min(Dt / D, Ht / H, Wt / W)
            if self.scale_clip:
                s = float(np.clip(s, *self.scale_clip))
            nD, nH, nW = max(1, int(round(D * s))), max(1, int(round(H * s))), max(1, int(round(W * s)))
        elif self.keep_ratio == "z-xy":
            sz = Dt / D
            sxy = min(Ht / H, Wt / W)
            if self.scale_clip:
                sz = float(np.clip(sz, *self.scale_clip))
                sxy = float(np.clip(sxy, *self.scale_clip))
            nD, nH, nW = max(1, int(round(D * sz))), max(1, int(round(H * sxy))), max(1, int(round(W * sxy)))
        else:  # per-axis
            nD, nH, nW = Dt, Ht, Wt
        return (nD, nH, nW)

    @staticmethod
    def _center_pad_or_crop(x: torch.Tensor, target: Tuple[int, int, int], pad_value: float) -> torch.Tensor:
        # x: (N,C,D,H,W)
        _, _, D, H, W = x.shape
        Dt, Ht, Wt = target
        # Pad (add missing voxels on both sides)
        pd = max(0, Dt - D)
        ph = max(0, Ht - H)
        pw = max(0, Wt - W)
        if pd or ph or pw:
            x = F.pad(
                x,
                (
                    pw // 2,
                    pw - pw // 2,  # W_left, W_right
                    ph // 2,
                    ph - ph // 2,  # H_left, H_right
                    pd // 2,
                    pd - pd // 2,
                ),  # D_left, D_right
                mode="constant",
                value=pad_value,
            )
            _, _, D, H, W = x.shape
        # Crop (center crop the excess)
        sd = max(0, (D - Dt) // 2)
        sh = max(0, (H - Ht) // 2)
        sw = max(0, (W - Wt) // 2)
        x = x[:, :, sd : sd + Dt, sh : sh + Ht, sw : sw + Wt]
        return x

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        # Union of target keys (image and mask)
        keys = set(self.keys) | self.image_keys | self.mask_keys
        # Find a representative key to check shape
        rep_key = None
        for k in keys:
            if k in d:
                rep_key = k
                break
        if rep_key is None:
            return d
        arr_rep = d[rep_key]
        is_tensor_rep = torch.is_tensor(arr_rep)
        xrep = arr_rep if is_tensor_rep else torch.as_tensor(arr_rep)
        if xrep.ndim == 3:
            xrep = xrep.unsqueeze(0)
        assert xrep.ndim == 4, f"{rep_key} must be (C,D,H,W) or (D,H,W)"
        _, D0, H0, W0 = xrep.shape
        nD, nH, nW = self._compute_new_size((D0, H0, W0))

        # Pre-compute pad/crop offsets
        pd = max(0, self.target[0] - nD)
        ph = max(0, self.target[1] - nH)
        pw = max(0, self.target[2] - nW)
        # Crop amounts
        sd = max(0, (nD - self.target[0]) // 2)
        sh = max(0, (nH - self.target[1]) // 2)
        sw = max(0, (nW - self.target[2]) // 2)

        # Scale factors (split z and yx depending on keep_ratio)
        if self.keep_ratio == "uniform":
            s = min(self.target[0] / D0, self.target[1] / H0, self.target[2] / W0)
            if self.scale_clip:
                s = float(np.clip(s, *self.scale_clip))
            sz = s
            sy = s
            sx = s
        elif self.keep_ratio == "z-xy":
            sz = self.target[0] / D0
            sxy = min(self.target[1] / H0, self.target[2] / W0)
            if self.scale_clip:
                sz = float(np.clip(sz, *self.scale_clip))
                sxy = float(np.clip(sxy, *self.scale_clip))
            sy = sxy
            sx = sxy
        else:  # per-axis
            sz = self.target[0] / D0
            sy = self.target[1] / H0
            sx = self.target[2] / W0

        for k in keys:
            if k not in d:
                continue
            arr = d[k]
            is_tensor = torch.is_tensor(arr)
            x = arr if is_tensor else torch.as_tensor(arr)
            # Ensure shape is (C,D,H,W)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            assert x.ndim == 4, f"{k} must be (C,D,H,W) or (D,H,W)"
            C, D, H, W = x.shape
            nD, nH, nW = self._compute_new_size((D, H, W))
            # Interpolate in (N,C,D,H,W)
            x = x.unsqueeze(0)
            mode = "nearest" if (k in self.mask_keys) else "trilinear"
            x = self._interp(x, size=(nD, nH, nW), mode=mode)
            pad_val = self.pad_value_mask if (k in self.mask_keys) else self.pad_value_image
            x = self._center_pad_or_crop(x, self.target, pad_val)
            x = x.squeeze(0)  # C,D,H,W
            d[k] = x if is_tensor else x.numpy()

        # Adjust coordinates for point keys (assume (N,3) in (z,y,x))
        for k in self.point_keys:
            if k not in d:
                continue
            pts = d[k]
            is_tensor = torch.is_tensor(pts)
            pts_t = pts if is_tensor else torch.as_tensor(pts)
            if pts_t.numel() == 0:
                continue
            # Allow (...,3) with arbitrary leading dims
            assert pts_t.shape[-1] == 3, f"{k} must have shape (...,3)"
            pts_f = pts_t.to(torch.float32)
            # Scale (coordinates in voxel units)
            z_new = pts_f[..., 0] * sz
            y_new = pts_f[..., 1] * sy
            x_new = pts_f[..., 2] * sx
            # Padding (even on both sides)
            z_new = z_new + (pd // 2)
            y_new = y_new + (ph // 2)
            x_new = x_new + (pw // 2)
            # Crop (center)
            z_new = z_new - sd
            y_new = y_new - sh
            x_new = x_new - sw
            # Clip to 0..target-1
            z_new = z_new.clamp(0, self.target[0] - 1)
            y_new = y_new.clamp(0, self.target[1] - 1)
            x_new = x_new.clamp(0, self.target[2] - 1)
            pts_out = torch.stack([z_new, y_new, x_new], dim=-1)
            d[k] = pts_out if is_tensor else pts_out.numpy()

        # Optionally record used scales and padding in meta
        try:
            meta = d.get("meta", {})
            meta["size_unify"] = {"target": self.target, "keep_ratio": self.keep_ratio}
            d["meta"] = meta
        except Exception:
            pass
        return d


class Resize3DMaxSized(MapTransform):
    """Resize uniformly to fit within the given max size."""

    def __init__(
        self,
        keys: Sequence[str],
        max_size: Tuple[int, int, int],
        image_keys: Optional[Sequence[str]] = ("image",),
        mask_keys: Optional[Sequence[str]] = ("vessel_label",),
        point_keys: Optional[Sequence[str]] = None,
        point_valid_keys: Optional[Mapping[str, str]] = None,
        meta_key: str = "meta",
        meta_resize_key: str = "resize_to_max",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if len(max_size) != 3:
            raise ValueError("max_size must be a 3-tuple (D,H,W)")
        if any(v <= 0 for v in max_size):
            raise ValueError(f"each element of max_size must be positive: {max_size}")
        self.max_size = tuple(int(v) for v in max_size)
        self.image_keys = set(image_keys or [])
        self.mask_keys = set(mask_keys or [])
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()
        self.point_valid_keys = dict(point_valid_keys) if point_valid_keys else {}
        self.meta_key = meta_key
        self.meta_resize_key = meta_resize_key

    @staticmethod
    def _prepare_tensor(
        arr: Union[np.ndarray, torch.Tensor, MetaTensor],
    ) -> Tuple[torch.Tensor, bool, torch.device, torch.dtype]:
        added_channel = False
        if isinstance(arr, MetaTensor):
            tensor = arr.as_tensor()
        elif torch.is_tensor(arr):
            tensor = arr
        else:
            tensor = torch.as_tensor(arr)

        device = tensor.device
        dtype = tensor.dtype

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            added_channel = True
        elif tensor.ndim != 4:
            raise ValueError(f"Unexpected tensor rank: {tensor.shape}")

        return tensor, added_channel, device, dtype

    @staticmethod
    def _is_float_dtype(dtype: torch.dtype) -> bool:
        if hasattr(dtype, "is_floating_point"):
            return dtype.is_floating_point
        return dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)

    @staticmethod
    def _restore_tensor(
        original: Union[np.ndarray, torch.Tensor, MetaTensor],
        resized: torch.Tensor,
        added_channel: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Union[np.ndarray, torch.Tensor, MetaTensor]:
        out = resized
        if added_channel:
            out = out.squeeze(0)
        if isinstance(original, MetaTensor):
            meta = original.meta.copy() if hasattr(original, "meta") else None
            ops = list(original.applied_operations) if hasattr(original, "applied_operations") else []
            reshaped = MetaTensor(out.to(device=device, dtype=dtype), meta=meta)
            if ops:
                reshaped.applied_operations = ops
            return reshaped
        if torch.is_tensor(original):
            return out.to(device=device, dtype=dtype)
        np_out = out.to(device="cpu", dtype=dtype).numpy()
        return np_out.astype(original.dtype, copy=False)

    def __call__(self, data: Mapping[str, Any]) -> MutableMapping[str, Any]:  # type: ignore[name-defined]
        d = dict(data)

        target_keys = set(self.keys) | self.image_keys | self.mask_keys
        rep_key = next((k for k in target_keys if k in d), None)
        if rep_key is None:
            return d

        rep_tensor, _, _, _ = self._prepare_tensor(d[rep_key])
        size_d, size_h, size_w = rep_tensor.shape[-3:]
        scale = min(
            1.0,
            self.max_size[0] / size_d,
            self.max_size[1] / size_h,
            self.max_size[2] / size_w,
        )

        if scale >= 1.0:
            return d

        new_size = (
            max(1, int(round(size_d * scale))),
            max(1, int(round(size_h * scale))),
            max(1, int(round(size_w * scale))),
        )

        for key in target_keys:
            if key not in d:
                continue
            tensor, added_channel, device, dtype = self._prepare_tensor(d[key])
            tensor = tensor.unsqueeze(0)
            mode = "nearest" if key in self.mask_keys else "trilinear"
            work = tensor.to(dtype=torch.float32)
            resized = F.interpolate(
                work, size=new_size, mode=mode, align_corners=False if mode != "nearest" else None
            )
            if key in self.mask_keys:
                resized = resized.round()
                if not self._is_float_dtype(dtype):
                    resized = resized.to(dtype=dtype)
            else:
                resized = resized.to(dtype=dtype if self._is_float_dtype(dtype) else torch.float32)
            resized = resized.squeeze(0)
            target_dtype = dtype if self._is_float_dtype(dtype) else dtype
            d[key] = self._restore_tensor(d[key], resized, added_channel, device, target_dtype)

        if self.point_keys:
            for key in self.point_keys:
                if key not in d:
                    continue
                pts = d[key]
                valid = None
                valid_key = self.point_valid_keys.get(key)
                if valid_key and valid_key in d:
                    mask_val = d[valid_key]
                    valid = (
                        mask_val.astype(bool)
                        if isinstance(mask_val, np.ndarray)
                        else mask_val.to(dtype=torch.bool)
                    )
                if isinstance(pts, np.ndarray):
                    pts = pts.copy()
                    if valid is not None:
                        pts[valid] *= scale
                    else:
                        pts *= scale
                    d[key] = pts
                else:
                    tensor_pts = pts if torch.is_tensor(pts) else torch.as_tensor(pts)
                    if valid is not None:
                        mask_tensor = (
                            valid
                            if torch.is_tensor(valid)
                            else torch.as_tensor(valid, device=tensor_pts.device)
                        )
                        tensor_pts = tensor_pts.clone()
                        tensor_pts[mask_tensor] = tensor_pts[mask_tensor] * scale
                    else:
                        tensor_pts = tensor_pts * scale
                    if isinstance(pts, MetaTensor):
                        meta = pts.meta.copy() if hasattr(pts, "meta") else None
                        ops = list(pts.applied_operations) if hasattr(pts, "applied_operations") else []
                        out = MetaTensor(tensor_pts, meta=meta)
                        if ops:
                            out.applied_operations = ops
                        d[key] = out
                    elif torch.is_tensor(pts):
                        d[key] = tensor_pts
                    else:
                        d[key] = tensor_pts.cpu().numpy()

        meta = d.get(self.meta_key)
        if isinstance(meta, dict):
            meta[self.meta_resize_key] = {
                "scale": float(scale),
                "target": self.max_size,
                "result": new_size,
            }

        return d


class Resize3DFitWithinPadToSized(MapTransform):
    """Pad to the specified size (optionally with uniform resampling)."""

    def __init__(
        self,
        keys: Sequence[str],
        target_size: Tuple[int, int, int],
        image_keys: Optional[Sequence[str]] = ("image",),
        mask_keys: Optional[Sequence[str]] = ("vessel_label",),
        point_keys: Optional[Sequence[str]] = None,
        point_valid_keys: Optional[Mapping[str, str]] = None,
        pad_value_image: float = 0.0,
        pad_value_mask: float = 0.0,
        meta_key: str = "meta",
        meta_entry_key: str = "pad_to_size",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if len(target_size) != 3:
            raise ValueError("target_size must be a 3-tuple (D,H,W)")
        if any(v <= 0 for v in target_size):
            raise ValueError(f"each element of target_size must be positive: {target_size}")
        self.target_size = tuple(int(v) for v in target_size)
        self.image_keys = set(image_keys or [])
        self.mask_keys = set(mask_keys or [])
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()
        self.point_valid_keys = dict(point_valid_keys) if point_valid_keys else {}
        self.pad_value_image = float(pad_value_image)
        self.pad_value_mask = float(pad_value_mask)
        self.meta_key = meta_key
        self.meta_entry_key = meta_entry_key

    @staticmethod
    def _prepare_tensor(
        arr: Union[np.ndarray, torch.Tensor, MetaTensor],
    ) -> Tuple[torch.Tensor, bool, torch.device, torch.dtype]:
        added_channel = False
        if isinstance(arr, MetaTensor):
            tensor = arr.as_tensor()
        elif torch.is_tensor(arr):
            tensor = arr
        else:
            tensor = torch.as_tensor(arr)

        device = tensor.device
        dtype = tensor.dtype

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            added_channel = True
        elif tensor.ndim != 4:
            raise ValueError(f"Unexpected tensor rank: {tensor.shape}")

        return tensor, added_channel, device, dtype

    @staticmethod
    def _is_float_dtype(dtype: torch.dtype) -> bool:
        if hasattr(dtype, "is_floating_point"):
            return dtype.is_floating_point
        return dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)

    @staticmethod
    def _restore_tensor(
        original: Union[np.ndarray, torch.Tensor, MetaTensor],
        tensor: torch.Tensor,
        added_channel: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Union[np.ndarray, torch.Tensor, MetaTensor]:
        out = tensor
        if added_channel:
            out = out.squeeze(0)
        if isinstance(original, MetaTensor):
            meta = original.meta.copy() if hasattr(original, "meta") else None
            ops = list(original.applied_operations) if hasattr(original, "applied_operations") else []
            reshaped = MetaTensor(out.to(device=device, dtype=dtype), meta=meta)
            if ops:
                reshaped.applied_operations = ops
            return reshaped
        if torch.is_tensor(original):
            return out.to(device=device, dtype=dtype)
        np_out = out.to(device="cpu", dtype=dtype).numpy()
        return np_out.astype(original.dtype, copy=False)

    def __call__(self, data: Mapping[str, Any]) -> MutableMapping[str, Any]:  # type: ignore[name-defined]
        d = dict(data)

        target_keys = set(self.keys) | self.image_keys | self.mask_keys
        rep_key = next((k for k in target_keys if k in d), None)
        if rep_key is None:
            return d

        rep_tensor, _, _, _ = self._prepare_tensor(d[rep_key])
        cur_size = rep_tensor.shape[-3:]
        scale = min(
            1.0,
            self.target_size[0] / cur_size[0],
            self.target_size[1] / cur_size[1],
            self.target_size[2] / cur_size[2],
        )

        scaled_size = (
            max(1, min(self.target_size[0], int(np.floor(cur_size[0] * scale)))),
            max(1, min(self.target_size[1], int(np.floor(cur_size[1] * scale)))),
            max(1, min(self.target_size[2], int(np.floor(cur_size[2] * scale)))),
        )

        pad_before = []
        pad_after = []

        for key in target_keys:
            if key not in d:
                continue
            tensor, added_channel, device, dtype = self._prepare_tensor(d[key])
            tensor = tensor.unsqueeze(0)

            if scale < 1.0:
                mode = "nearest" if key in self.mask_keys else "trilinear"
                work = tensor.to(dtype=torch.float32)
                tensor = F.interpolate(
                    work,
                    size=scaled_size,
                    mode=mode,
                    align_corners=False if mode != "nearest" else None,
                )
                if key in self.mask_keys and not self._is_float_dtype(dtype):
                    tensor = tensor.round().to(dtype=dtype)
            else:
                tensor = tensor

            tensor = tensor.squeeze(0)
            size_after = tensor.shape[-3:]
            pads = []
            for tgt, cur in zip(self.target_size, size_after):
                remaining = max(0, tgt - cur)
                before = remaining // 2
                after = remaining - before
                pads.append((before, after))
            # convert to torch padding order (W, H, D)
            pad_tuple = (
                pads[2][0],
                pads[2][1],
                pads[1][0],
                pads[1][1],
                pads[0][0],
                pads[0][1],
            )

            pad_val = self.pad_value_mask if key in self.mask_keys else self.pad_value_image
            padded = F.pad(tensor.unsqueeze(0), pad_tuple, mode="constant", value=pad_val)
            padded = padded.squeeze(0)

            out_dtype = dtype if self._is_float_dtype(dtype) or key in self.mask_keys else torch.float32
            d[key] = self._restore_tensor(d[key], padded, added_channel, device, out_dtype)

            if not pad_before:
                pad_before = [pads[0][0], pads[1][0], pads[2][0]]
                pad_after = [pads[0][1], pads[1][1], pads[2][1]]

        pad_before = tuple(int(v) for v in pad_before)
        pad_after = tuple(int(v) for v in pad_after)

        if self.point_keys:
            for key in self.point_keys:
                if key not in d:
                    continue
                pts = d[key]
                valid_key = self.point_valid_keys.get(key)
                valid_mask = None
                if valid_key and valid_key in d:
                    val = d[valid_key]
                    valid_mask = val.astype(bool) if isinstance(val, np.ndarray) else val.to(dtype=torch.bool)
                if isinstance(pts, np.ndarray):
                    pts = pts.copy()
                    if scale < 1.0:
                        pts *= scale
                    if valid_mask is not None:
                        pts[valid_mask] += np.array(pad_before, dtype=np.float32)
                    else:
                        pts += np.array(pad_before, dtype=np.float32)
                    d[key] = pts
                else:
                    tensor_pts = pts if torch.is_tensor(pts) else torch.as_tensor(pts)
                    if scale < 1.0:
                        tensor_pts = tensor_pts * scale
                    offset = torch.as_tensor(pad_before, device=tensor_pts.device, dtype=tensor_pts.dtype)
                    if valid_mask is not None:
                        mask_tensor = (
                            valid_mask
                            if torch.is_tensor(valid_mask)
                            else torch.as_tensor(valid_mask, device=tensor_pts.device)
                        )
                        tensor_pts = tensor_pts.clone()
                        tensor_pts[mask_tensor] = tensor_pts[mask_tensor] + offset
                    else:
                        tensor_pts = tensor_pts + offset
                    if isinstance(pts, MetaTensor):
                        meta = pts.meta.copy() if hasattr(pts, "meta") else None
                        ops = list(pts.applied_operations) if hasattr(pts, "applied_operations") else []
                        out_pts = MetaTensor(tensor_pts, meta=meta)
                        if ops:
                            out_pts.applied_operations = ops
                        d[key] = out_pts
                    elif torch.is_tensor(pts):
                        d[key] = tensor_pts
                    else:
                        d[key] = tensor_pts.cpu().numpy()

        meta = d.get(self.meta_key)
        if isinstance(meta, dict):
            meta[self.meta_entry_key] = {
                "target": self.target_size,
                "scale": float(scale),
                "pad_before": pad_before,
                "pad_after": pad_after,
            }

        return d


class Resize3DXYLongSideZDownOnlyPadToSized(MapTransform):
    """
    Resize 3D data so that XY preserves aspect ratio and matches the target XY long side,
    Z is only downsampled when larger than the target Z (no upsampling if smaller), and
    finally standardize (D,H,W) via center padding/cropping to the target size.

    - XY is uniformly scaled (preserve in-plane ratio) so that max(H', W') = Ht (=Wt).
    - Z uses scale 1.0 if D <= Dt; downsample to Dt only when D > Dt.
    - Then center pad/center crop each axis to target_size=(Dt,Ht,Wt).

    Requirement: XY must be square (Ht==Wt). Images use trilinear; labels use nearest.
    Coordinate keys are assumed to be in (z,y,x).
    """

    def __init__(
        self,
        keys: Sequence[str],
        target_size: Tuple[int, int, int],
        image_keys: Optional[Sequence[str]] = ("image",),
        mask_keys: Optional[Sequence[str]] = ("vessel_label",),
        point_keys: Optional[Sequence[str]] = None,
        pad_value_image: float = 0.0,
        pad_value_mask: float = 0.0,
        meta_key: str = "meta",
        meta_entry_key: str = "resize_xy_long_zdown_pad_to_size",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if len(target_size) != 3:
            raise ValueError("target_size must be a 3-tuple (D,H,W)")
        if any(v <= 0 for v in target_size):
            raise ValueError(f"each element of target_size must be positive: {target_size}")
        self.target_size = tuple(int(v) for v in target_size)
        # XY must be square
        if self.target_size[1] != self.target_size[2]:
            raise ValueError(
                f"Resize3DXYLongSideZDownOnlyPadToSized requires XY to be square (Ht==Wt). "
                f"Got={self.target_size[1]}x{self.target_size[2]}"
            )
        self._target_xy = self.target_size[1]
        self.image_keys = set(image_keys or [])
        self.mask_keys = set(mask_keys or [])
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()
        self.pad_value_image = float(pad_value_image)
        self.pad_value_mask = float(pad_value_mask)
        self.meta_key = meta_key
        self.meta_entry_key = meta_entry_key

    @staticmethod
    def _prepare_tensor(
        arr: Union[np.ndarray, torch.Tensor, MetaTensor],
    ) -> Tuple[torch.Tensor, bool, torch.device, torch.dtype]:
        # Normalize input to (C,D,H,W)
        added_channel = False
        if isinstance(arr, MetaTensor):
            tensor = arr.as_tensor()
        elif torch.is_tensor(arr):
            tensor = arr
        else:
            tensor = torch.as_tensor(arr)

        device = tensor.device
        dtype = tensor.dtype

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            added_channel = True
        elif tensor.ndim != 4:
            raise ValueError(f"Unexpected tensor rank: {tensor.shape}")

        return tensor, added_channel, device, dtype

    @staticmethod
    def _is_float_dtype(dtype: torch.dtype) -> bool:
        if hasattr(dtype, "is_floating_point"):
            return dtype.is_floating_point
        return dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)

    @staticmethod
    def _restore_tensor(
        original: Union[np.ndarray, torch.Tensor, MetaTensor],
        out_tensor: torch.Tensor,
        added_channel: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Union[np.ndarray, torch.Tensor, MetaTensor]:
        out = out_tensor
        if added_channel:
            out = out.squeeze(0)
        if isinstance(original, MetaTensor):
            meta = original.meta.copy() if hasattr(original, "meta") else None
            ops = list(original.applied_operations) if hasattr(original, "applied_operations") else []
            reshaped = MetaTensor(out.to(device=device, dtype=dtype), meta=meta)
            if ops:
                reshaped.applied_operations = ops
            return reshaped
        if torch.is_tensor(original):
            return out.to(device=device, dtype=dtype)
        np_out = out.to(device="cpu", dtype=dtype).numpy()
        return np_out.astype(original.dtype, copy=False)

    @staticmethod
    def _center_pad_then_crop(
        x: torch.Tensor,
        target: Tuple[int, int, int],
        pad_value: float,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
        # x: (N,C,D,H,W)
        _, _, D, H, W = x.shape
        Dt, Ht, Wt = target
        # Padding amounts (only when lacking)
        pd = max(0, Dt - D)
        ph = max(0, Ht - H)
        pw = max(0, Wt - W)
        if pd or ph or pw:
            x = F.pad(
                x,
                (
                    pw // 2,
                    pw - pw // 2,
                    ph // 2,
                    ph - ph // 2,
                    pd // 2,
                    pd - pd // 2,
                ),
                mode="constant",
                value=pad_value,
            )
            _, _, D, H, W = x.shape

        # Crop amounts (only when exceeding; center crop)
        sd = max(0, (D - Dt) // 2)
        sh = max(0, (H - Ht) // 2)
        sw = max(0, (W - Wt) // 2)
        x = x[:, :, sd : sd + Dt, sh : sh + Ht, sw : sw + Wt]
        return x, (pd, ph, pw), (sd, sh, sw)

    def __call__(self, data: Mapping[str, Any]) -> MutableMapping[str, Any]:  # type: ignore[name-defined]
        d = dict(data)

        # Set of target keys (include image and mask)
        target_keys = set(self.keys) | self.image_keys | self.mask_keys
        rep_key = next((k for k in target_keys if k in d), None)
        if rep_key is None:
            return d

        rep_tensor, _, _, _ = self._prepare_tensor(d[rep_key])
        size_d, size_h, size_w = rep_tensor.shape[-3:]

        Dt, Ht, Wt = self.target_size
        target_long_xy = self._target_xy  # XY is square
        cur_long_xy = max(size_h, size_w)
        # Match XY long side while preserving aspect ratio
        s_xy = target_long_xy / float(cur_long_xy) if cur_long_xy > 0 else 1.0
        # Z only downsamples (1.0 when smaller)
        s_z = 1.0 if size_d <= Dt else (Dt / float(size_d))

        new_size = (
            max(1, int(round(size_d * s_z))),
            max(1, int(round(size_h * s_xy))),
            max(1, int(round(size_w * s_xy))),
        )

        # Actual transform (images/labels)
        for key in target_keys:
            if key not in d:
                continue
            tensor, added_channel, device, dtype = self._prepare_tensor(d[key])
            tensor = tensor.unsqueeze(0)  # (1,C,D,H,W)
            mode = "nearest" if key in self.mask_keys else "trilinear"

            work = tensor.to(dtype=torch.float32)
            resized = F.interpolate(
                work,
                size=new_size,
                mode=mode,
                # align_corners=False if mode != "nearest" else None,
            )

            # For masks, keep integer types
            if key in self.mask_keys:
                # resized = resized.round()
                if not self._is_float_dtype(dtype):
                    resized = resized.to(dtype=dtype)
            else:
                resized = resized.to(dtype=dtype if self._is_float_dtype(dtype) else torch.float32)

            # Center pad then center crop to target_size
            pad_val = self.pad_value_mask if key in self.mask_keys else self.pad_value_image
            out, pad_before, crop_starts = self._center_pad_then_crop(resized, self.target_size, pad_val)
            out = out.squeeze(0)
            d[key] = self._restore_tensor(d[key], out, added_channel, device, dtype)

            # Keep pad/crop info from the first key (for updating points)
            if key == rep_key:
                last_pad_before = pad_before
                last_crop_starts = crop_starts

        # Adjust coordinate keys ((N,3) in (z,y,x))
        if self.point_keys:
            pd, ph, pw = last_pad_before  # type: ignore[name-defined]
            sd, sh, sw = last_crop_starts  # type: ignore[name-defined]
            for key in self.point_keys:
                if key not in d:
                    continue
                pts = d[key]
                is_tensor = torch.is_tensor(pts)
                pts_t = pts if is_tensor else torch.as_tensor(pts)
                if pts_t.numel() == 0:
                    continue
                assert pts_t.shape[-1] == 3, f"{key} must have shape (...,3)"
                pts_f = pts_t.to(torch.float32)

                z_new = pts_f[..., 0] * s_z
                y_new = pts_f[..., 1] * s_xy
                x_new = pts_f[..., 2] * s_xy

                z_new = z_new + (pd // 2)
                y_new = y_new + (ph // 2)
                x_new = x_new + (pw // 2)

                z_new = z_new - sd
                y_new = y_new - sh
                x_new = x_new - sw

                z_new = z_new.clamp(0, Dt - 1)
                y_new = y_new.clamp(0, Ht - 1)
                x_new = x_new.clamp(0, Wt - 1)

                pts_out = torch.stack([z_new, y_new, x_new], dim=-1)
                d[key] = pts_out if is_tensor else pts_out.numpy()

        # Store meta information (optional)
        meta = d.get(self.meta_key)
        if isinstance(meta, dict):
            meta[self.meta_entry_key] = {
                "target": self.target_size,
                "scale_xy": float(s_xy),
                "scale_z": float(s_z),
                "new_size": new_size,
            }

        return d


class PadToMultipleOfd(MapTransform):
    """Zero-pad spatial dimensions to a specified multiple."""

    def __init__(
        self,
        keys: Sequence[str],
        multiple: int = 32,
        image_keys: Optional[Sequence[str]] = ("image",),
        mask_keys: Optional[Sequence[str]] = ("vessel_label",),
        point_keys: Optional[Sequence[str]] = None,
        point_valid_keys: Optional[Mapping[str, str]] = None,
        pad_value_image: float = 0.0,
        pad_value_mask: float = 0.0,
        meta_key: str = "meta",
        meta_pad_key: str = "pad_to_multiple",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if multiple is None or int(multiple) <= 0:
            raise ValueError(f"multiple must be a positive integer: {multiple}")
        self.multiple = int(multiple)
        self.image_keys = set(image_keys or [])
        self.mask_keys = set(mask_keys or [])
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()
        self.point_valid_keys = dict(point_valid_keys) if point_valid_keys else {}
        self.pad_value_image = float(pad_value_image)
        self.pad_value_mask = float(pad_value_mask)
        self.meta_key = meta_key
        self.meta_pad_key = meta_pad_key

    @staticmethod
    def _prepare_tensor(
        arr: Union[np.ndarray, torch.Tensor, MetaTensor],
    ) -> Tuple[torch.Tensor, bool, torch.device, torch.dtype]:
        added_channel = False
        tensor: torch.Tensor
        if isinstance(arr, MetaTensor):
            tensor = arr.as_tensor()
        elif torch.is_tensor(arr):
            tensor = arr
        else:
            tensor = torch.as_tensor(arr)

        device = tensor.device
        dtype = tensor.dtype

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            added_channel = True
        elif tensor.ndim != 4:
            raise ValueError(f"Unexpected tensor rank: {tensor.shape}")

        return tensor, added_channel, device, dtype

    @staticmethod
    def _restore_tensor(
        original: Union[np.ndarray, torch.Tensor, MetaTensor],
        padded: torch.Tensor,
        added_channel: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Union[np.ndarray, torch.Tensor, MetaTensor]:
        out = padded
        if added_channel:
            out = out.squeeze(0)
        if isinstance(original, MetaTensor):
            meta = original.meta.copy() if hasattr(original, "meta") else None
            ops = list(original.applied_operations) if hasattr(original, "applied_operations") else []
            reshaped = MetaTensor(out.to(device=device, dtype=dtype), meta=meta)
            if ops:
                reshaped.applied_operations = ops
            return reshaped
        if torch.is_tensor(original):
            return out.to(device=device, dtype=dtype)
        np_out = out.to(device="cpu", dtype=dtype).numpy()
        return np_out.astype(original.dtype, copy=False)

    def __call__(self, data: Mapping[str, Any]) -> MutableMapping[str, Any]:  # type: ignore[name-defined]
        d = dict(data)
        if self.multiple <= 1:
            return d

        # Determine a representative key
        search_keys = list(self.keys) + list(self.image_keys) + list(self.mask_keys)
        rep_key = next((k for k in search_keys if k in d), None)
        if rep_key is None:
            return d

        rep_tensor, added_channel, _, _ = self._prepare_tensor(d[rep_key])
        spatial = rep_tensor.shape[-3:]
        pad_before: List[int] = []
        pad_after: List[int] = []

        for size in spatial:
            remainder = size % self.multiple
            if remainder == 0:
                pad_before.append(0)
                pad_after.append(0)
            else:
                total = self.multiple - remainder
                before = total // 2
                after = total - before
                pad_before.append(before)
                pad_after.append(after)

        if not any(pad_before) and not any(pad_after):
            return d

        pad_tuple = (
            pad_before[2],
            pad_after[2],
            pad_before[1],
            pad_after[1],
            pad_before[0],
            pad_after[0],
        )

        target_keys = set(self.keys) | self.image_keys | self.mask_keys
        for key in target_keys:
            if key not in d:
                continue
            tensor, added_channel_key, device, dtype = self._prepare_tensor(d[key])
            tensor = tensor.unsqueeze(0)
            pad_val = self.pad_value_mask if key in self.mask_keys else self.pad_value_image
            padded = F.pad(tensor, pad_tuple, mode="constant", value=pad_val)
            padded = padded.squeeze(0)
            d[key] = self._restore_tensor(d[key], padded, added_channel_key, device, dtype)

        # Translate point sets accordingly
        if self.point_keys and any(pad_before):
            offsets = np.array(pad_before, dtype=np.float32)
            for key in self.point_keys:
                if key not in d:
                    continue
                pts = d[key]
                if isinstance(pts, np.ndarray):
                    pts = pts.copy()
                    mask = None
                    valid_key = self.point_valid_keys.get(key)
                    if valid_key and valid_key in d:
                        mask = d[valid_key]
                        mask = mask.astype(bool)
                    if mask is not None:
                        pts[mask] += offsets
                    else:
                        pts += offsets
                    d[key] = pts
                else:
                    tensor_pts = pts if torch.is_tensor(pts) else torch.as_tensor(pts)
                    valid_key = self.point_valid_keys.get(key)
                    mask = None
                    if valid_key and valid_key in d:
                        v = d[valid_key]
                        mask = (
                            v.to(dtype=torch.bool)
                            if torch.is_tensor(v)
                            else torch.as_tensor(v).to(dtype=torch.bool)
                        )
                    if mask is not None:
                        tensor_pts = tensor_pts.clone()
                        tensor_pts[mask] = tensor_pts[mask] + torch.as_tensor(
                            offsets, device=tensor_pts.device, dtype=tensor_pts.dtype
                        )
                    else:
                        tensor_pts = tensor_pts + torch.as_tensor(
                            offsets, device=tensor_pts.device, dtype=tensor_pts.dtype
                        )
                    if isinstance(pts, MetaTensor):
                        meta = pts.meta.copy() if hasattr(pts, "meta") else None
                        ops = list(pts.applied_operations) if hasattr(pts, "applied_operations") else []
                        out = MetaTensor(tensor_pts, meta=meta)
                        if ops:
                            out.applied_operations = ops
                        d[key] = out
                    elif torch.is_tensor(pts):
                        d[key] = tensor_pts
                    else:
                        d[key] = tensor_pts.cpu().numpy()

        # Record padding amounts in meta
        meta = d.get(self.meta_key)
        if isinstance(meta, dict):
            meta[self.meta_pad_key] = {
                "multiple": self.multiple,
                "pad_before": tuple(int(v) for v in pad_before),
                "pad_after": tuple(int(v) for v in pad_after),
            }

        return d


# ============================
# Batch-friendly, custom torch transforms
# ============================


class BatchedRandSimulateLowResolutiond(Randomizable):
    """
    Apply a probabilistic low-resolution simulation (downsample -> upsample back) to
    batch tensors (B,C,D,H,W).

    - zoom_range: (min, max) or [(min_z, max_z), (min_y, max_y), (min_x, max_x)]
      Randomly sample shrink ratios per axis (<1.0 shrinks).
    - Input is a dict; only process keys listed in `keys`.
    - Intended for images with continuous values (trilinear interpolation).

    Note: This is a lightweight replacement for MONAI RandSimulateLowResolutiond.
    Sampling/blur details may differ; prioritizes fast batched processing.
    """

    def __init__(
        self,
        keys: Sequence[str],
        zoom_range: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (0.5, 1.0),
        prob: float = 0.3,
        allow_missing_keys: bool = True,
        mode: str = "trilinear",
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        self.zoom_range = zoom_range
        self.prob = float(prob)
        self.allow_missing_keys = allow_missing_keys
        self.mode = mode

    def _sample_zoom_per_axis(self, R: np.random.RandomState, spatial_dim: int) -> Tuple[float, float, float]:
        # If zoom_range is a single pair -> share across axes; otherwise per-axis
        if (
            isinstance(self.zoom_range, tuple)
            and len(self.zoom_range) == 2
            and isinstance(self.zoom_range[0], (int, float))
        ):
            low, high = float(self.zoom_range[0]), float(self.zoom_range[1])
            z = float(R.uniform(low, high))
            if spatial_dim == 3:
                return (z, z, z)
            elif spatial_dim == 2:
                return (z, z, 1.0)
            else:
                return (z, 1.0, 1.0)
        # Per-axis ranges
        rngs = list(self.zoom_range)  # type: ignore
        if len(rngs) < 3:
            # Pad missing axes with 1.0
            rngs = rngs + [(1.0, 1.0)] * (3 - len(rngs))
        out: List[float] = []
        for i in range(3):
            low, high = rngs[i]
            low = float(low)
            high = float(high)
            out.append(float(R.uniform(low, high)))
        return (out[0], out[1], out[2])

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)
        # Use Randomizable.R as seeds
        R = self.R
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"key '{key}' not found in input dict.")
            x = d[key]
            if not torch.is_tensor(x):
                # Skip numpy arrays; this implementation assumes GPU/torch tensors
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{key} must be a torch.Tensor.")
            # Normalize to shape (B,C,D,H,W)
            if x.ndim == 4:
                x_in = x.unsqueeze(0)
            elif x.ndim == 5:
                x_in = x
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{key} must be 4D or 5D. shape={tuple(x.shape)}")

            B, C, D, H, W = x_in.shape
            device = x_in.device
            dtype = x_in.dtype

            # Output tensor
            out = x_in
            # Decide per-batch application and target size
            for b in range(B):
                if R.random_sample() >= self.prob:
                    continue  # skip
                z, y, xz = self._sample_zoom_per_axis(R, 3)
                nD = max(1, int(round(D * z)))
                nH = max(1, int(round(H * y)))
                nW = max(1, int(round(W * xz)))

                if nD == D and nH == H and nW == W:
                    continue
                # Downsample -> upsample (trilinear)
                tmp = F.interpolate(out[b : b + 1].to(dtype), size=(nD, nH, nW), mode=self.mode)
                out[b : b + 1] = F.interpolate(tmp, size=(D, H, W), mode=self.mode)

            # Restore original dimensionality
            d[key] = out if x.ndim == 5 else out.squeeze(0)
        return d


class BatchedRandAffined(Randomizable):
    """
    Apply a 3D affine (rotation + scale + shear) stochastically to batch tensors (B,C,D,H,W).

    - rotate_range: (rx, ry, rz) max rotation per axis in radians; sample uniformly in [-r, +r].
    - scale_range: float or (sx, sy, sz); scale = 1 + U(-r, r).
    - shear_range: like MONAI RandAffine; each element is a scalar or (min, max).
    - mode: str or per-key sequence (e.g., ["trilinear", "nearest", ...]).
    - padding_mode: grid_sample's padding_mode ("zeros"/"border"/"reflection").

    Note: no translation is applied; rotation center is the volume center (affine_grid convention).
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.6,
        rotate_range: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale_range: Union[float, Tuple[float, float, float]] = 0.0,
        shear_range: Optional[Union[float, Sequence[Union[float, Tuple[float, float], None]]]] = None,
        mode: Union[str, Sequence[str]] = "trilinear",
        padding_mode: str = "zeros",
        allow_missing_keys: bool = True,
        point_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        self.prob = float(prob)
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.mode = mode
        self.padding_mode = padding_mode
        self.allow_missing_keys = allow_missing_keys
        # Keys for point sets of shape (B,M,3) in (z,y,x)
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()

    @staticmethod
    def _is_range_sequence(val: Any) -> bool:
        return isinstance(val, Sequence) and not isinstance(val, (str, bytes))

    def _normalize_shear_range(self) -> Optional[List[Tuple[float, float]]]:
        """Normalize shear_range into a list of 6 (low, high) pairs."""
        if self.shear_range is None:
            return None

        def to_pair(item: Any) -> Tuple[float, float]:
            if item is None:
                return (0.0, 0.0)
            if self._is_range_sequence(item):
                seq = list(item)
                if len(seq) != 2:
                    raise ValueError(f"Each element of shear_range must be a 2-tuple: {item}")
                low, high = float(seq[0]), float(seq[1])
            else:
                mag = float(item)
                low, high = -mag, mag
            if low > high:
                raise ValueError(f"shear_range must satisfy low <= high: {(low, high)}")
            return (low, high)

        if isinstance(self.shear_range, (float, int)):
            pairs: List[Tuple[float, float]] = [(-float(self.shear_range), float(self.shear_range))] * 6
        elif self._is_range_sequence(self.shear_range):
            raw_seq = list(self.shear_range)
            pairs = [to_pair(v) for v in raw_seq]
        else:
            raise TypeError("shear_range must be a scalar or a sequence.")

        if len(pairs) < 6:
            pairs.extend([(0.0, 0.0)] * (6 - len(pairs)))
        else:
            pairs = pairs[:6]
        return pairs

    @staticmethod
    def _rotation_matrix(rx: float, ry: float, rz: float) -> torch.Tensor:
        """Create a 3x3 rotation matrix (Rz @ Ry @ Rx)."""
        sx, cx = torch.sin(torch.tensor(rx)), torch.cos(torch.tensor(rx))
        sy, cy = torch.sin(torch.tensor(ry)), torch.cos(torch.tensor(ry))
        sz, cz = torch.sin(torch.tensor(rz)), torch.cos(torch.tensor(rz))
        Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
        Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
        R = Rz @ Ry @ Rx
        return R

    def _get_mode_for_key(self, key: str, idx: int) -> str:
        if isinstance(self.mode, str):
            return self.mode
        # For sequences, pick per key order (use last if lengths mismatch)
        if idx < len(self.mode):
            return self.mode[idx]
        return self.mode[-1]

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)
        Rng = self.R

        # Normalize scale range
        if isinstance(self.scale_range, (int, float)):
            srx = sry = srz = float(self.scale_range)
        else:
            tmp = list(self.scale_range)
            if len(tmp) < 3:
                tmp = tmp + [0.0] * (3 - len(tmp))
            srx, sry, srz = float(tmp[0]), float(tmp[1]), float(tmp[2])

        rx_max, ry_max, rz_max = self.rotate_range

        # Keep forward matrices (inverse from output to input) for later point updates
        thetas_per_key: Dict[str, torch.Tensor] = {}
        Ms_fwd_per_key: Dict[str, torch.Tensor] = {}

        # 1) Find reference key and sample transforms once per batch
        ref_key = None
        for key in self.keys:
            if key in d and torch.is_tensor(d[key]):
                ref_key = key
                break
        if ref_key is not None:
            xref = d[ref_key]
            if xref.ndim == 4:
                xref_in = xref.unsqueeze(0)
            elif xref.ndim == 5:
                xref_in = xref
            else:
                raise ValueError(f"{ref_key} must be 4D or 5D. shape={tuple(xref.shape)}")
            B, _, _, _, _ = xref_in.shape
            device = xref_in.device

            thetas: List[torch.Tensor] = []
            Ms_fwd: List[torch.Tensor] = []
            shear_ranges = self._normalize_shear_range()
            for b in range(B):
                if Rng.random_sample() >= self.prob:
                    thetas.append(torch.eye(3, 4, dtype=torch.float32))
                    Ms_fwd.append(torch.eye(3, dtype=torch.float32))
                    continue
                rx = float(Rng.uniform(-rx_max, rx_max)) if rx_max != 0 else 0.0
                ry = float(Rng.uniform(-ry_max, ry_max)) if ry_max != 0 else 0.0
                rz = float(Rng.uniform(-rz_max, rz_max)) if rz_max != 0 else 0.0
                sx = 1.0 + (float(Rng.uniform(-srx, srx)) if srx != 0 else 0.0)
                sy = 1.0 + (float(Rng.uniform(-sry, sry)) if sry != 0 else 0.0)
                sz = 1.0 + (float(Rng.uniform(-srz, srz)) if srz != 0 else 0.0)

                if shear_ranges is not None:
                    shear_vals: List[float] = []
                    for low, high in shear_ranges:
                        if low == high:
                            shear_vals.append(float(low))
                        else:
                            shear_vals.append(float(Rng.uniform(low, high)))
                    # 3D shear uses 6 components
                    shear_vals = (shear_vals + [0.0] * 6)[:6]
                    shear_mat = torch.tensor(
                        [
                            [1.0, shear_vals[0], shear_vals[1]],
                            [shear_vals[2], 1.0, shear_vals[3]],
                            [shear_vals[4], shear_vals[5], 1.0],
                        ],
                        dtype=torch.float32,
                    )
                    shear_mat_inv = torch.inverse(shear_mat)
                else:
                    shear_mat = torch.eye(3, dtype=torch.float32)
                    shear_mat_inv = torch.eye(3, dtype=torch.float32)

                S_inv = torch.diag(torch.tensor([1.0 / sx, 1.0 / sy, 1.0 / sz], dtype=torch.float32))
                Rm = self._rotation_matrix(rx, ry, rz).to(dtype=torch.float32)
                A = Rm @ shear_mat_inv @ S_inv
                theta = torch.zeros((3, 4), dtype=torch.float32)
                theta[:, :3] = A
                thetas.append(theta)

                R_inv = torch.inverse(Rm)
                S_fw = torch.diag(torch.tensor([sx, sy, sz], dtype=torch.float32))
                M_fwd = S_fw @ shear_mat @ R_inv
                Ms_fwd.append(M_fwd)

            theta_b = torch.stack(thetas, dim=0).to(device)
            M_fwd_b = torch.stack(Ms_fwd, dim=0).to(device)
        else:
            theta_b = None
            M_fwd_b = None

        # 2) Apply the same thetas to all keys (choose interpolation mode per key)
        for i, key in enumerate(self.keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"key '{key}' not found in input dict.")
            x = d[key]
            if not torch.is_tensor(x):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{key} must be a torch.Tensor.")

            if x.ndim == 4:
                x_in = x.unsqueeze(0)
            elif x.ndim == 5:
                x_in = x
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{key} must be a 4D or 5D tensor. shape={tuple(x.shape)}")

            if theta_b is None:
                # No transform needed (e.g., missing keys)
                d[key] = x
                continue

            grid = F.affine_grid(theta_b, size=x_in.shape, align_corners=False)
            mode_key = self._get_mode_for_key(key, i)
            if mode_key in ("trilinear", "linear", "bilinear"):
                mode_torch = "bilinear"
            elif mode_key == "nearest":
                mode_torch = "nearest"
            else:
                mode_torch = "nearest"

            # grid_sample requires floating point -> convert to float32
            orig_dtype = x_in.dtype
            x_proc = x_in.to(torch.float32)
            y = F.grid_sample(
                x_proc, grid, mode=mode_torch, padding_mode=self.padding_mode, align_corners=False
            )
            # Restore original dtype (round for integer masks, etc.)
            if orig_dtype == torch.uint8:
                y = y.round().clamp_(0, 1).to(torch.uint8)
            elif orig_dtype == torch.float16:
                y = y.to(torch.float16)
            # Keep float32/float64 as-is

            d[key] = y if x.ndim == 5 else y.squeeze(0)

        # If point keys are specified, update coordinates with the same affine
        for pk in self.point_keys:
            if pk not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"point key '{pk}' not found in input dict.")
            pts = d[pk]
            if not torch.is_tensor(pts):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{pk} must be a torch.Tensor.")
            # Expect shape (B,M,3); if (M,3), assume B=1
            if pts.ndim == 2 and pts.shape[-1] == 3:
                pts_in = pts.unsqueeze(0)
            elif pts.ndim == 3 and pts.shape[-1] == 3:
                pts_in = pts
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{pk} must be (B,M,3) or (M,3). shape={tuple(pts.shape)}")

            Bp, Mp, _ = pts_in.shape
            # Get (D,H,W) from the representative key (first image key)
            ref_key = None
            for key in self.keys:
                if key in d:
                    ref_key = key
                    break
            if ref_key is None:
                return d
            ref = d[ref_key]
            if ref.ndim == 4:
                ref = ref.unsqueeze(0)
            _, _, D, H, W = ref.shape

            # Convert to normalized coordinates (align_corners=False)
            P = pts_in.to(torch.float32)
            xn = (P[..., 2] + 0.5) / W * 2.0 - 1.0
            yn = (P[..., 1] + 0.5) / H * 2.0 - 1.0
            zn = (P[..., 0] + 0.5) / D * 2.0 - 1.0
            vec = torch.stack([xn, yn, zn], dim=-1)  # (B,M,3)

            if M_fwd_b is None:
                d[pk] = pts
                continue
            vec_out = torch.einsum("bij,bmj->bmi", M_fwd_b, vec)

            # Convert back to voxel coordinates (align_corners=False)
            x_out = ((vec_out[..., 0] + 1.0) * 0.5) * W - 0.5
            y_out = ((vec_out[..., 1] + 1.0) * 0.5) * H - 0.5
            z_out = ((vec_out[..., 2] + 1.0) * 0.5) * D - 0.5
            pts_out = torch.stack([z_out, y_out, x_out], dim=-1)
            # Clip to valid ranges
            pts_out[..., 0] = pts_out[..., 0].clamp(0, D - 1)
            pts_out[..., 1] = pts_out[..., 1].clamp(0, H - 1)
            pts_out[..., 2] = pts_out[..., 2].clamp(0, W - 1)
            d[pk] = pts_out if pts.ndim == 3 else pts_out.squeeze(0)

        return d


class BatchedRandGridDistortiond(Randomizable):
    """
    Stochastic MONAI RandGridDistortion-like deformation on batch tensors (B,C,[D],H,W).

    - num_cells: number of grid cells per axis (int or per-axis sequence)
    - distort_limit: distortion range; scalar implies ±distort_limit
    - point_keys: also approximately apply the same deformation to point sets (B,M,3)[z,y,x]
    - Since grid warping is non-linear, update point sets via per-axis inverse mapping with linear interp
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.6,
        num_cells: Union[int, Sequence[int]] = 5,
        distort_limit: Union[float, Tuple[float, float]] = (-0.03, 0.03),
        mode: Union[str, Sequence[str]] = "bilinear",
        padding_mode: str = "border",
        allow_missing_keys: bool = True,
        point_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        self.prob = float(prob)
        self.num_cells = num_cells
        if isinstance(distort_limit, (float, int)):
            limit = float(distort_limit)
            self.distort_low = -abs(limit)
            self.distort_high = abs(limit)
        else:
            low, high = float(distort_limit[0]), float(distort_limit[1])
            self.distort_low = min(low, high)
            self.distort_high = max(low, high)
        self.mode = mode
        self.padding_mode = padding_mode
        self.allow_missing_keys = allow_missing_keys
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()

    @staticmethod
    def _is_sequence(val: Any) -> bool:
        return isinstance(val, Sequence) and not isinstance(val, (str, bytes))

    def _normalized_num_cells(self, spatial_dim: int) -> Tuple[int, ...]:
        if isinstance(self.num_cells, int):
            if self.num_cells <= 0:
                raise ValueError("num_cells must be a positive integer.")
            base = [int(self.num_cells)] * spatial_dim
        elif self._is_sequence(self.num_cells):
            seq = list(self.num_cells)
            if not seq:
                raise ValueError("num_cells sequence cannot be empty.")
            seq = [int(max(1, v)) for v in seq]
            if len(seq) < spatial_dim:
                seq.extend([seq[-1]] * (spatial_dim - len(seq)))
            base = tuple(seq[:spatial_dim])
        else:
            raise TypeError("num_cells must be an int or a sequence.")
        return tuple(int(v) for v in base[:spatial_dim])

    def _get_mode_for_key(self, key: str, idx: int) -> str:
        if isinstance(self.mode, str):
            return self.mode
        if idx < len(self.mode):
            return self.mode[idx]
        return self.mode[-1]

    def _sample_axis_steps(self, num_cells: int) -> List[float]:
        rand = self.R.uniform(self.distort_low, self.distort_high, size=num_cells + 1)
        return [1.0 + float(v) for v in rand]

    @staticmethod
    def _compute_axis_ranges(
        dim_size: int,
        num_cells: int,
        steps: Sequence[float],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_cells <= 0:
            raise ValueError("num_cells must be positive for each axis.")
        cell_size = dim_size // num_cells
        if cell_size <= 0:
            raise ValueError("num_cells is too large; cell size becomes 0.")
        steps = list(steps)
        if len(steps) < num_cells + 1:
            steps.extend([1.0] * (num_cells + 1 - len(steps)))

        raw = torch.zeros(dim_size, dtype=torch.float32, device=device)
        prev = 0.0
        for idx in range(num_cells + 1):
            start = int(idx * cell_size)
            end = start + cell_size
            if end > dim_size:
                end = dim_size
                cur = float(dim_size)
            else:
                cur = prev + cell_size * float(steps[idx])
            if end > start:
                raw[start:end] = torch.linspace(prev, cur, end - start, device=device)
            prev = cur
        centered = raw - (dim_size - 1.0) / 2.0
        return centered, raw

    def _build_grid(
        self,
        axis_centered: Sequence[torch.Tensor],
        spatial_shape: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        axes = [ax.to(device) for ax in axis_centered]
        mesh = torch.meshgrid(*axes, indexing="ij")
        stacked = torch.stack(mesh, dim=0)  # (spatial_dim, ...)
        perm = list(range(len(axes) - 1, -1, -1))
        grid_t = torch.stack([stacked[p] for p in perm], dim=-1)
        for idx, dim in enumerate(reversed(spatial_shape)):
            grid_t[..., idx] *= 2.0 / max(2, dim)
        return grid_t.unsqueeze(0)  # 1 x ... x spatial_dim

    @staticmethod
    def _warp_axis_coords(coords: torch.Tensor, axis_raw: torch.Tensor, dim_size: int) -> torch.Tensor:
        coords_clamped = coords.clamp_(0.0, float(dim_size - 1))
        idx = torch.searchsorted(axis_raw, coords_clamped, right=True)
        max_idx = axis_raw.shape[0] - 1
        idx0 = torch.clamp(idx - 1, min=0, max=max_idx)
        idx1 = torch.clamp(idx, min=0, max=max_idx)
        val0 = axis_raw[idx0]
        val1 = axis_raw[idx1]
        denom = (val1 - val0).abs()
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        t = (coords_clamped - val0) / denom
        warped = idx0.to(coords.dtype) + t.to(coords.dtype)
        return warped

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)
        Rng = self.R

        # Get shape from representative key
        ref_key = None
        for key in self.keys:
            if key in d and torch.is_tensor(d[key]):
                ref_key = key
                break
        if ref_key is None:
            return d

        ref = d[ref_key]
        if ref.ndim == 4:
            ref_in = ref.unsqueeze(0)
        elif ref.ndim == 5:
            ref_in = ref
        else:
            raise ValueError(f"{ref_key} must be 4D or 5D. shape={tuple(ref.shape)}")

        B = ref_in.shape[0]
        spatial_shape = tuple(ref_in.shape[2:])
        spatial_dim = len(spatial_shape)
        num_cells_per_axis = self._normalized_num_cells(spatial_dim)

        params_per_batch: List[Optional[Dict[str, Any]]] = []
        for b in range(B):
            if Rng.random_sample() >= self.prob:
                params_per_batch.append(None)
                continue
            axis_centered: List[torch.Tensor] = []
            axis_raw: List[torch.Tensor] = []
            for dim_size, n_cell in zip(spatial_shape, num_cells_per_axis):
                steps = self._sample_axis_steps(n_cell)
                centered, raw = self._compute_axis_ranges(dim_size, n_cell, steps, device=torch.device("cpu"))
                axis_centered.append(centered)
                axis_raw.append(raw)
            params_per_batch.append(
                {
                    "axis_centered": axis_centered,
                    "axis_raw": axis_raw,
                }
            )

        # Grid cache (batch_idx, device) -> grid tensor
        grid_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        for idx_key, key in enumerate(self.keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"key '{key}' not found in input dict.")
            x = d[key]
            if not torch.is_tensor(x):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{key} must be a torch.Tensor.")

            if x.ndim == 4:
                x_in = x.unsqueeze(0)
            elif x.ndim == 5:
                x_in = x
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{key} must be 4D or 5D. shape={tuple(x.shape)}")

            Bx = x_in.shape[0]
            if Bx != B or tuple(x_in.shape[2:]) != spatial_shape:
                raise ValueError(
                    f"{key} shape {tuple(x_in.shape)} does not match ref shape {(B, x_in.shape[1], *spatial_shape)}."
                )

            out = x_in.clone()
            mode_key = self._get_mode_for_key(key, idx_key)
            if mode_key in ("bilinear", "trilinear", "linear"):
                mode_torch = "bilinear"
            elif mode_key == "nearest":
                mode_torch = "nearest"
            else:
                mode_torch = "bilinear"

            for b in range(B):
                params = params_per_batch[b]
                if params is None:
                    continue
                cache_key = (b, out.device)
                if cache_key not in grid_cache:
                    grid_cache[cache_key] = self._build_grid(
                        params["axis_centered"], spatial_shape, device=out.device
                    )
                grid = grid_cache[cache_key]

                orig_dtype = out.dtype
                sample = out[b : b + 1].to(torch.float32)
                warped = F.grid_sample(
                    sample,
                    grid,
                    mode=mode_torch,
                    padding_mode=self.padding_mode,
                    align_corners=False,
                )
                if orig_dtype == torch.uint8:
                    warped = warped.round().clamp_(0, 1).to(torch.uint8)
                elif orig_dtype == torch.float16:
                    warped = warped.to(torch.float16)
                else:
                    warped = warped.to(orig_dtype)
                out[b : b + 1] = warped

            d[key] = out if x.ndim == 5 else out.squeeze(0)

        # Update point sets as well
        for pk in self.point_keys:
            if pk not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"point key '{pk}' not found in input dict.")
            pts = d[pk]
            if not torch.is_tensor(pts):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{pk} must be a torch.Tensor.")

            if pts.ndim == 2 and pts.shape[-1] == spatial_dim:
                pts_in = pts.unsqueeze(0)
            elif pts.ndim == 3 and pts.shape[-1] == spatial_dim:
                pts_in = pts
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{pk} must be (B,M,{spatial_dim}). shape={tuple(pts.shape)}")

            pts_out = pts_in.clone()
            for b in range(B):
                params = params_per_batch[b]
                if params is None:
                    continue
                axis_raw = [ar.to(pts_out.device) for ar in params["axis_raw"]]
                coord = pts_out[b]
                for axis_idx, (raw_axis, dim_size) in enumerate(zip(axis_raw, spatial_shape)):
                    coord[:, axis_idx] = self._warp_axis_coords(coord[:, axis_idx], raw_axis, dim_size)
                for axis_idx, dim_size in enumerate(spatial_shape):
                    coord[:, axis_idx].clamp_(0.0, float(dim_size - 1))
                pts_out[b] = coord

            d[pk] = pts_out if pts.ndim == 3 else pts_out.squeeze(0)

        return d


class BatchedRandFlipd(Randomizable):
    """
    Apply probabilistic flips along specified axes to batch tensors (B,C,D,H,W).

    - spatial_axis: candidate axes to flip (z=0, y=1, x=2). int or sequence.
    - prob: probability per axis (per-sample in batch).
    - keys: dict keys to flip (images/masks).
    - point_keys: update point sets (B,M,3)[z,y,x] with the same flips.
    - allow_missing_keys: if True, skip missing keys.

    Notes:
    - Uses torch.flip (no interpolation; exact values).
    - Expects 4D (C,D,H,W) or 5D (B,C,D,H,W) tensors.
    """

    def __init__(
        self,
        keys: Sequence[str],
        spatial_axis: Union[int, Sequence[int]] = (0, 1, 2),
        prob: float = 0.5,
        allow_missing_keys: bool = True,
        point_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        if isinstance(spatial_axis, int):
            spatial_axis = (spatial_axis,)
        self.spatial_axis = tuple(int(a) for a in spatial_axis)
        self.prob = float(prob)
        self.allow_missing_keys = allow_missing_keys
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)
        Rng = self.R

        # Get batch dimension and spatial size from a representative key
        ref_key = None
        for k in self.keys:
            if k in d and torch.is_tensor(d[k]):
                ref_key = k
                break
        if ref_key is None:
            # If none of the target keys exist, do nothing
            return d

        ref = d[ref_key]
        if ref.ndim == 4:
            ref_in = ref.unsqueeze(0)
        elif ref.ndim == 5:
            ref_in = ref
        else:
            raise ValueError(f"{ref_key} must be 4D or 5D. shape={tuple(ref.shape)}")

        B, _, D, H, W = ref_in.shape

        # Sample flip flags per batch and per axis
        # flips[b, a] = True means flip along axis=a
        axes = self.spatial_axis
        flips = [[False, False, False] for _ in range(B)]
        for b in range(B):
            for a in axes:
                if Rng.random_sample() < self.prob:
                    flips[b][a] = True

        # Apply to image/mask keys
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"key '{key}' not found in input dict.")
            x = d[key]
            if not torch.is_tensor(x):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{key} must be a torch.Tensor.")

            if x.ndim == 4:
                x_in = x.unsqueeze(0)
            elif x.ndim == 5:
                x_in = x
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{key} must be 4D or 5D. shape={tuple(x.shape)}")

            out = x_in
            # Flip along required axes per sample
            for b in range(B):
                # out[b] has shape (C, D, H, W);
                # use relative indices [-3 (D), -2 (H), -1 (W)] for spatial axes
                dims: list[int] = []
                if flips[b][0]:
                    dims.append(-3)  # D axis
                if flips[b][1]:
                    dims.append(-2)  # H axis
                if flips[b][2]:
                    dims.append(-1)  # W axis
                if dims:
                    out[b] = torch.flip(out[b], dims=tuple(dims))

            d[key] = out if x.ndim == 5 else out.squeeze(0)

        # Update point keys (B,M,3)[z,y,x] with the same flips
        for pk in self.point_keys:
            if pk not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"point key '{pk}' not found in input dict.")
            pts = d[pk]
            if not torch.is_tensor(pts):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{pk} must be a torch.Tensor.")

            # (B,M,3) or (M,3)
            if pts.ndim == 2 and pts.shape[-1] == 3:
                pts_in = pts.unsqueeze(0)
            elif pts.ndim == 3 and pts.shape[-1] == 3:
                pts_in = pts
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{pk} must be (B,M,3) or (M,3). shape={tuple(pts.shape)}")

            # Use representative key size (as in ref_in)
            out_pts = pts_in.to(torch.float32).clone()
            for b in range(B):
                if flips[b][0]:  # flip z
                    out_pts[b, :, 0] = (D - 1) - out_pts[b, :, 0]
                if flips[b][1]:  # flip y
                    out_pts[b, :, 1] = (H - 1) - out_pts[b, :, 1]
                if flips[b][2]:  # flip x
                    out_pts[b, :, 2] = (W - 1) - out_pts[b, :, 2]

            d[pk] = out_pts if pts.ndim == 3 else out_pts.squeeze(0)

        return d


class BatchedRandAxisSwapd(Randomizable):
    """
    Dictionary transform that stochastically permutes the three spatial axes (z,y,x)
    for batch tensors (B,C,D,H,W).

    - keys: tensor keys (images/masks). Expects 4D (C,D,H,W) or 5D (B,C,D,H,W).
    - prob: probability to apply to the entire batch (same permutation for all samples).
    - point_keys: point-key names with (B,M,3)[z,y,x] to update with the same permutation.
    - allow_missing_keys: if True, skip missing keys.
    - permutations: allowed axis orders (tuples of (pi_z, pi_y, pi_x)). If None, use all 6 S3 perms.
    - ensure_same_shape: if True and the chosen permutation would change (D,H,W), skip (no-op).

    Notes:
    - No flips are performed (combine with BatchedRandFlipd to cover 6×8=48 variants).
    - Since shape dims change order, downstream code must tolerate variable shapes.
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.5,
        allow_missing_keys: bool = True,
        point_keys: Optional[Sequence[str]] = None,
        permutations: Optional[Sequence[Tuple[int, int, int]]] = None,
        ensure_same_shape: bool = False,
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        self.prob = float(prob)
        self.allow_missing_keys = allow_missing_keys
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()
        self.ensure_same_shape = bool(ensure_same_shape)
        if permutations is None:
            # All S3 permutations (new (z,y,x) = old axis indices)
            self.perms: List[Tuple[int, int, int]] = [
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ]
        else:
            # Validate user-provided permutations
            tmp: List[Tuple[int, int, int]] = []
            for p in permutations:
                if not (isinstance(p, tuple) and len(p) == 3):
                    raise ValueError(f"invalid permutation spec: {p}")
                if set(p) != {0, 1, 2}:
                    raise ValueError(f"permutation must be a permutation of (0,1,2). got={p}")
                tmp.append((int(p[0]), int(p[1]), int(p[2])))
            self.perms = tmp

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)
        Rng = self.R

        # Find a representative key and get batch/spatial sizes
        ref_key = None
        for k in self.keys:
            if k in d and torch.is_tensor(d[k]):
                ref_key = k
                break
        if ref_key is None:
            return d

        ref = d[ref_key]
        if ref.ndim == 4:
            ref_in = ref.unsqueeze(0)
        elif ref.ndim == 5:
            ref_in = ref
        else:
            raise ValueError(f"{ref_key} must be a 4D or 5D tensor. shape={tuple(ref.shape)}")

        B, _, D, H, W = ref_in.shape

        # Decide per-batch whether to apply and choose permutation
        do_apply = Rng.random_sample() < self.prob
        perm_idx = int(Rng.randint(0, len(self.perms))) if do_apply else -1
        if perm_idx >= 0 and self.ensure_same_shape:
            pz, py, px = self.perms[perm_idx]
            # If the shape would change (e.g., H!=W and swapping y/x), skip (no-op)
            new_D, new_H, new_W = ((D, H, W)[pz], (D, H, W)[py], (D, H, W)[px])
            if (new_D, new_H, new_W) != (D, H, W):
                perm_idx = -1

        # Process image/mask keys
        for key in self.keys:
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"key '{key}' not found in input dict.")
            x = d[key]
            if not torch.is_tensor(x):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{key} must be a torch.Tensor.")

            if x.ndim == 4:
                x_in = x.unsqueeze(0)
            elif x.ndim == 5:
                x_in = x
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{key} must be a 4D or 5D tensor. shape={tuple(x.shape)}")

            out_list: List[torch.Tensor] = []
            for b in range(B):
                xb = x_in[b]
                idx = perm_idx
                if idx < 0:
                    out_list.append(xb)
                    continue
                pz, py, px = self.perms[idx]
                # Swap spatial axes (1,2,3) -> [pz,py,px] for (C,D,H,W)
                xb_perm = xb.permute(0, 1 + pz, 1 + py, 1 + px).contiguous()
                out_list.append(xb_perm)

            # Shapes may differ per sample; avoid cat and accumulate a list.
            # In this project we assume identical sizes, so torch.stack is possible.
            try:
                out = torch.stack(out_list, dim=0)
            except Exception:
                # Do not attempt complex handling for mismatched shapes between samples
                raise RuntimeError("Shapes differ across samples after axis swap. Check prior resizing.")

            d[key] = out if x.ndim == 5 else out.squeeze(0)

        # Swap point keys (B,M,3)[z,y,x] with the same order
        for pk in self.point_keys:
            if pk not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"point key '{pk}' not found in input dict.")
            pts = d[pk]
            if not torch.is_tensor(pts):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{pk} must be a torch.Tensor.")

            if pts.ndim == 2 and pts.shape[-1] == 3:
                pts_in = pts.unsqueeze(0)
            elif pts.ndim == 3 and pts.shape[-1] == 3:
                pts_in = pts
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{pk} must be (B,M,3) or (M,3). shape={tuple(pts.shape)}")

            out_pts = pts_in.to(torch.float32).clone()
            for b in range(B):
                idx = perm_idx
                if idx < 0:
                    continue
                pz, py, px = self.perms[idx]
                # New (z,y,x) = old coords (pz,py,px)
                old = out_pts[b].clone()
                out_pts[b, :, 0] = old[:, pz]
                out_pts[b, :, 1] = old[:, py]
                out_pts[b, :, 2] = old[:, px]

            d[pk] = out_pts if pts.ndim == 3 else out_pts.squeeze(0)

        return d


class BatchedRandShrinkPadToOriginald(Randomizable):
    """
    Randomly shrink along Z and XY independently for batch tensors (B,C,D,H,W),
    then center-pad back to the original (D,H,W). Dictionary-style transform.

    - Shrink ratios for z and xy are sampled independently (xy uses y=x same ratio).
    - Resampling uses `F.interpolate` (trilinear for images, nearest for masks).
    - Applied per sample with probability; otherwise acts as identity.

    Args:
        keys: target keys (images, labels, etc.).
        prob: probability to apply per sample.
        max_shrink_ratio: maximum shrink ratio (e.g., 0.9 -> min scale 0.1).
        mode: interpolation mode, string or per-key sequence ("trilinear"/"nearest").
        pad_value: constant value to use for padding.
        allow_missing_keys: if True, skip missing keys.
        point_keys: point-key names for (B,M,3)[z,y,x]; coordinates updated accordingly.
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.3,
        max_shrink_ratio: float = 0.1,
        mode: str | Sequence[str] = "nearest",
        pad_value: float = 0.0,
        allow_missing_keys: bool = True,
        point_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.keys = list(keys)
        self.prob = float(prob)
        self.max_shrink_ratio = float(max_shrink_ratio)
        self.mode = mode
        self.pad_value = float(pad_value)
        self.allow_missing_keys = bool(allow_missing_keys)
        self.point_keys = tuple(point_keys) if point_keys is not None else tuple()

    def _get_mode_for_key(self, idx: int) -> str:
        if isinstance(self.mode, str):
            return self.mode
        if idx < len(self.mode):
            return self.mode[idx]
        return self.mode[-1]

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        d = dict(data)

        # Get a representative key to determine batch and spatial sizes
        ref_key = None
        for k in self.keys:
            if k in d and torch.is_tensor(d[k]):
                ref_key = k
                break
        if ref_key is None:
            return d

        ref = d[ref_key]
        if ref.ndim == 4:
            ref_in = ref.unsqueeze(0)
        elif ref.ndim == 5:
            ref_in = ref
        else:
            if self.allow_missing_keys:
                return d
            raise ValueError(f"{ref_key} must be 4D or 5D. shape={tuple(ref.shape)}")

        B, _, D, H, W = ref_in.shape

        # For each sample decide application and sample scales
        apply = []
        scale_z = []
        scale_xy = []
        for _ in range(B):
            if self.R.random_sample() < self.prob:
                # Shrink ratio r ~ U(0, max), scale s = 1 - r
                rz = float(self.R.uniform(0.0, self.max_shrink_ratio))
                rxy = float(self.R.uniform(0.0, self.max_shrink_ratio))
                sz = max(1e-6, 1.0 - rz)
                sxy = max(1e-6, 1.0 - rxy)
                apply.append(True)
                scale_z.append(sz)
                scale_xy.append(sxy)
            else:
                apply.append(False)
                scale_z.append(1.0)
                scale_xy.append(1.0)

        # Process each image/label key
        for idx_key, key in enumerate(self.keys):
            if key not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"key '{key}' not found in input dict.")
            x = d[key]
            if not torch.is_tensor(x):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{key} must be a torch.Tensor.")

            if x.ndim == 4:
                x_in = x.unsqueeze(0)
            elif x.ndim == 5:
                x_in = x
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{key} must be 4D or 5D. shape={tuple(x.shape)}")

            mode_key = self._get_mode_for_key(idx_key)
            if mode_key in ("bilinear", "linear", "trilinear"):
                interp_mode = "trilinear"
                align = True  # align_corners is bool; not used for nearest
            else:
                interp_mode = "nearest"
                align = False

            # Allocate output buffer
            out = torch.empty_like(x_in)

            for b in range(B):
                if not apply[b]:
                    out[b] = x_in[b]
                    continue
                sz = scale_z[b]
                sxy = scale_xy[b]
                nD = max(1, int(round(D * sz)))
                nH = max(1, int(round(H * sxy)))
                nW = max(1, int(round(W * sxy)))

                pd = max(0, D - nD)
                ph = max(0, H - nH)
                pw = max(0, W - nW)

                # Resample (process one sample at a time with B=1)
                xb = x_in[b : b + 1]
                orig_dtype = xb.dtype
                xb32 = xb.to(torch.float32)
                y = F.interpolate(
                    xb32,
                    size=(nD, nH, nW),
                    mode=interp_mode,
                    align_corners=False if interp_mode != "nearest" else None,
                )
                # Center padding
                pad_tuple = (
                    pw // 2,
                    pw - pw // 2,
                    ph // 2,
                    ph - ph // 2,
                    pd // 2,
                    pd - pd // 2,
                )
                y = F.pad(y, pad_tuple, mode="constant", value=self.pad_value)
                # Restore dtype (round integer types)
                if orig_dtype == torch.uint8:
                    y = y.round().clamp_(0, 255).to(torch.uint8)
                elif orig_dtype == torch.float16:
                    y = y.to(torch.float16)
                else:
                    y = y.to(orig_dtype)
                out[b] = y

            d[key] = out if x.ndim == 5 else out.squeeze(0)

        # Update points (B,M,3)[z,y,x]
        for pk in self.point_keys:
            if pk not in d:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"point key '{pk}' not found in input dict.")
            pts = d[pk]
            if not torch.is_tensor(pts):
                if self.allow_missing_keys:
                    continue
                raise TypeError(f"{pk} must be a torch.Tensor.")
            if pts.ndim == 2 and pts.shape[-1] == 3:
                P = pts.unsqueeze(0).to(torch.float32)
            elif pts.ndim == 3 and pts.shape[-1] == 3:
                P = pts.to(torch.float32)
            else:
                if self.allow_missing_keys:
                    continue
                raise ValueError(f"{pk} must be (B,M,3) or (M,3). shape={tuple(pts.shape)}")

            Bp, Mp, _ = P.shape
            if Bp != B:
                # If batch size mismatches, return without transform
                d[pk] = pts
                continue

            out_pts = P.clone()
            for b in range(B):
                if not apply[b]:
                    continue
                sz = scale_z[b]
                sxy = scale_xy[b]
                nD = max(1, int(round(D * sz)))
                nH = max(1, int(round(H * sxy)))
                nW = max(1, int(round(W * sxy)))
                pd = max(0, D - nD)
                ph = max(0, H - nH)
                pw = max(0, W - nW)
                z = out_pts[b, :, 0] * sz + (pd // 2)
                y = out_pts[b, :, 1] * sxy + (ph // 2)
                x = out_pts[b, :, 2] * sxy + (pw // 2)
                out_pts[b, :, 0] = z.clamp(0, D - 1)
                out_pts[b, :, 1] = y.clamp(0, H - 1)
                out_pts[b, :, 2] = x.clamp(0, W - 1)

            d[pk] = out_pts if pts.ndim == 3 else out_pts.squeeze(0)

        return d
