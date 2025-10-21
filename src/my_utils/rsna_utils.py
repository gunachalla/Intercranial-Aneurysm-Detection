from typing import Tuple, Optional
import numpy as np
from pathlib import Path
import nibabel as nib
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)


def reorient_nifti_to_lps(nii_img, return_zyx=True):
    """
    Reorient a NIfTI image to LPS+ orientation.

    Args:
        nii_img: nibabel.Nifti1Image
        return_zyx (bool): Whether to reorder axes (X,Y,Z)->(Z,Y,X)

    Returns:
        np.ndarray: Image data reoriented to LPS+
        Tuple: Reoriented voxel spacing (LPS+)
    """
    # Current orientation
    ornt_src = io_orientation(nii_img.affine)
    # Desired LPS+ orientation codes
    desired = axcodes2ornt(("L", "P", "S"))
    # Compute orientation transform
    transform = ornt_transform(ornt_src, desired)
    # Apply transform to data
    data_aligned = apply_orientation(nii_img.get_fdata(), transform)
    # Collapse 4D to 3D if needed
    if data_aligned.ndim == 4:
        print(f"Converting 4D to 3D: {data_aligned.shape}")
        if data_aligned.shape[-1] == 1:
            data_aligned = data_aligned.squeeze(-1)
        else:
            raise ValueError(f"Last axis of 4D data is not 1: {data_aligned.shape}")

    if return_zyx:
        # Reorder axes (X,Y,Z)->(Z,Y,X)
        data_zyx = np.transpose(data_aligned, (2, 1, 0))
    else:
        data_zyx = data_aligned

    # Get voxel spacing directly from NIfTI header
    original_spacing = nii_img.header.get_zooms()[:3]

    # Reorder spacing according to the orientation transform
    # Use the index of the max absolute value per row of the transform
    spacing_mapping = []
    for i in range(3):
        # Column index with the maximum absolute value in row i
        max_idx = np.argmax(np.abs(transform[i, :3]))
        spacing_mapping.append(max_idx)

    # Reorder original spacing to match transformed axes
    transformed_spacing = tuple(original_spacing[idx] for idx in spacing_mapping)

    if return_zyx:
        # Finally change axis order (X,Y,Z)->(Z,Y,X)
        final_spacing = (transformed_spacing[2], transformed_spacing[1], transformed_spacing[0])
    else:
        final_spacing = transformed_spacing

    return data_zyx, final_spacing


def load_nifti(nifti_path: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI file.

    Returns:
        volume: Volume (X,Y,Z)
        spacing: Voxel spacing (X,Y,Z) in mm
        nifti_obj: NIfTI object
    """
    nii = nib.load(str(nifti_path))

    data = nii.get_fdata(dtype=np.float32)
    spacing = np.array(nii.header.get_zooms())

    return data, spacing, nii


def load_nifti_and_convert_to_ras(nifti_path: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI file and convert to RAS orientation.

    Returns:
        volume: Volume in RAS orientation (X,Y,Z)
        spacing: Voxel spacing (X,Y,Z) in mm
        nifti_obj: NIfTI object after RAS conversion
    """
    nii = nib.load(str(nifti_path))

    ras_nifti = nib.as_closest_canonical(nii)
    data = ras_nifti.get_fdata(dtype=np.float32)
    spacing = np.array(ras_nifti.header.get_zooms())

    return data, spacing, ras_nifti


class SphereMaskGenerator:
    """Generate spherical masks."""

    def __init__(self, default_label: int = 14):
        """
        Initialize the generator.

        Args:
            default_label (int): Default mask label value
        """
        self.default_label = default_label

    def create_sphere_mask(
        self,
        center: Tuple[int, int, int],
        radius_mm: float,
        volume_shape: Tuple[int, int, int],
        voxel_spacing: Tuple[float, float, float],
        label: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create a spherical mask.

        Args:
            center (Tuple[int, int, int]): Center coordinate (z, y, x)
            radius_mm (float): Radius in millimeters
            volume_shape (Tuple[int, int, int]): Volume shape (Z, Y, X)
            voxel_spacing (Tuple[float, float, float]): Voxel spacing (Z, Y, X)
            label (int, optional): Mask label value

        Returns:
            np.ndarray: Spherical mask
        """
        if label is None:
            label = self.default_label

        z_center, y_center, x_center = center
        z_max, y_max, x_max = volume_shape
        z_spacing, y_spacing, x_spacing = voxel_spacing

        # Avoid division by zero
        if abs(z_spacing) < 1e-6:
            print(
                f"Warning: Z-axis spacing is near zero ({z_spacing}). Using X-axis spacing ({x_spacing})."
            )
            z_spacing = x_spacing

        if abs(x_spacing) < 1e-6:
            print("Error: X-axis spacing is also near zero. Cannot create spherical mask.")
            return np.zeros(volume_shape, dtype=np.uint8)

        # Create 3D coordinate meshgrid
        z_coords, y_coords, x_coords = np.mgrid[0:z_max, 0:y_max, 0:x_max]

        # Compute distance in mm from each voxel center to the specified center
        distances = np.sqrt(
            ((z_coords - z_center) * z_spacing) ** 2
            + ((y_coords - y_center) * y_spacing) ** 2
            + ((x_coords - x_center) * x_spacing) ** 2
        )

        # Set label for voxels within the specified radius
        mask = np.zeros(volume_shape, dtype=np.uint8)
        mask[distances <= radius_mm] = label

        return mask
