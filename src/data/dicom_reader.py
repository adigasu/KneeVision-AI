"""
RSNA Knee Abnormality Detection - DICOM Series Reader & Volume Preprocessor.
Handles sorting by 3D spatial position, decompression of varied transfer syntaxes,
intensity normalization/windowing, and slice resampling.
"""

import os
import glob
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_dicom_slice(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a single DICOM slice and extracts key metadata.

    Args:
        file_path: Path to the .dcm file.

    Returns:
        pixel_array: 2D float32 numpy array.
        metadata: Dictionary with slice spatial and acquisition tags.
    """
    dcm = pydicom.dcmread(file_path, stop_before_pixels=False)
    
    # Extract metadata tags
    meta = {
        "SOPInstanceUID": getattr(dcm, "SOPInstanceUID", os.path.basename(file_path).replace(".dcm", "")),
        "InstanceNumber": int(getattr(dcm, "InstanceNumber", 0)),
        "SliceLocation": float(getattr(dcm, "SliceLocation", 0.0)),
        "ImagePositionPatient": [float(x) for x in getattr(dcm, "ImagePositionPatient", [0.0, 0.0, 0.0])],
        "ImageOrientationPatient": [float(x) for x in getattr(dcm, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
        "PixelSpacing": [float(x) for x in getattr(dcm, "PixelSpacing", [1.0, 1.0])],
        "SliceThickness": float(getattr(dcm, "SliceThickness", 1.0)),
        "RescaleSlope": float(getattr(dcm, "RescaleSlope", 1.0)),
        "RescaleIntercept": float(getattr(dcm, "RescaleIntercept", 0.0)),
        "PhotometricInterpretation": getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"),
    }

    # Apply VOI LUT if available, otherwise raw rescale
    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception:
        arr = dcm.pixel_array.astype(np.float32)
        arr = arr * meta["RescaleSlope"] + meta["RescaleIntercept"]

    arr = arr.astype(np.float32)

    # Invert if MONOCHROME1 (where 0 is white and max is black)
    if meta["PhotometricInterpretation"] == "MONOCHROME1":
        arr = np.max(arr) - arr

    return arr, meta


def load_series_volume(
    series_dir: str,
    target_slices: Optional[int] = 32,
    normalize: str = "minmax",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads all DICOM slices in a series folder, sorts them along the slice normal,
    normalizes intensities, and optionally resamples to a fixed slice depth.

    Args:
        series_dir: Path to directory containing .dcm files for one series.
        target_slices: If specified, resamples/interpolates along depth to this exact count.
        normalize: 'minmax' (scales to [0, 1]) or 'zscore' (mean 0, std 1) or 'raw'.

    Returns:
        volume: 3D float32 array of shape (Depth, Height, Width).
        metadata_list: List of metadata dictionaries for each slice.
    """
    dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    # Read all slices and metadata
    slice_data = []
    for f in dcm_files:
        try:
            arr, meta = read_dicom_slice(f)
            slice_data.append((arr, meta))
        except Exception as e:
            continue

    if not slice_data:
        raise ValueError(f"Failed to decode any DICOM files from {series_dir}")

    # Sort slices spatially using projection onto patient slice normal or ImagePositionPatient[2]
    # Slice normal = cross product of row and column direction cosines
    try:
        iop = slice_data[0][1]["ImageOrientationPatient"]
        row_vec = np.array(iop[:3])
        col_vec = np.array(iop[3:])
        normal_vec = np.cross(row_vec, col_vec)
        # Compute projection along normal
        slice_data.sort(key=lambda s: np.dot(np.array(s[1]["ImagePositionPatient"]), normal_vec))
    except Exception:
        # Fallback to ImagePositionPatient[2] or InstanceNumber
        slice_data.sort(key=lambda s: s[1]["ImagePositionPatient"][2] if "ImagePositionPatient" in s[1] else s[1]["InstanceNumber"])

    arrays = [s[0] for s in slice_data]
    metas = [s[1] for s in slice_data]
    volume = np.stack(arrays, axis=0).astype(np.float32)  # (D, H, W)

    # Uniform slice resampling if target_slices is set
    if target_slices is not None and volume.shape[0] != target_slices:
        current_depth = volume.shape[0]
        if current_depth < target_slices:
            # Replicate or linearly interpolate
            indices = np.linspace(0, current_depth - 1, target_slices).round().astype(int)
            volume = volume[indices]
            metas = [metas[i] for i in indices]
        else:
            # Subsample evenly
            indices = np.linspace(0, current_depth - 1, target_slices).round().astype(int)
            volume = volume[indices]
            metas = [metas[i] for i in indices]

    # Intensity normalization
    if normalize == "minmax":
        v_min, v_max = np.percentile(volume, (0.5, 99.5))
        if v_max > v_min:
            volume = np.clip((volume - v_min) / (v_max - v_min), 0.0, 1.0)
        else:
            volume = np.zeros_like(volume)
    elif normalize == "zscore":
        mean = np.mean(volume)
        std = np.std(volume) + 1e-6
        volume = (volume - mean) / std

    return volume, metas
