"""
RSNA Knee Abnormality Detection - High-Performance Volume Preprocessor.
Performs 3D spatial slice sorting along slice normal, intensity windowing/normalization,
in-plane resizing, and 2.5D multi-slice channel stacking.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_dicom_slice_raw(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a single DICOM file and extracts raw pixel values along with spatial tags.
    """
    dcm = pydicom.dcmread(file_path, stop_before_pixels=False)

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

    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception:
        arr = dcm.pixel_array.astype(np.float32)
        arr = arr * meta["RescaleSlope"] + meta["RescaleIntercept"]

    arr = arr.astype(np.float32)

    if meta["PhotometricInterpretation"] == "MONOCHROME1":
        arr = np.max(arr) - arr

    return arr, meta


def stack_25d_channels(volume_3d: np.ndarray) -> np.ndarray:
    """
    Converts (D, H, W) 3D volume into (D, 3, H, W) 2.5D slice tensors.
    Channel 0: slice z-1
    Channel 1: slice z
    Channel 2: slice z+1
    With boundary clamping at z=0 and z=D-1.
    """
    D, H, W = volume_3d.shape
    stacked = np.zeros((D, 3, H, W), dtype=volume_3d.dtype)

    for z in range(D):
        z_prev = max(0, z - 1)
        z_next = min(D - 1, z + 1)
        stacked[z, 0] = volume_3d[z_prev]
        stacked[z, 1] = volume_3d[z]
        stacked[z, 2] = volume_3d[z_next]

    return stacked


def process_single_series(
    series_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    normalize: str = "minmax",
    stack_25d: bool = True,
    target_slices: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Loads all DICOM slices in a series folder, sorts them in true 3D spatial anatomical order,
    normalizes intensities, resizes in-plane, and creates 2.5D stacked channels.

    Args:
        series_dir: Path to directory containing .dcm files for a series.
        target_size: (Height, Width) for in-plane spatial resizing.
        normalize: 'minmax' (percentile 0.5% - 99.5% scaled to [0, 1]) or 'zscore' or 'raw'.
        stack_25d: If True, outputs (D, 3, H, W) tensor.
        target_slices: Optional fixed depth resampling (if None, keeps all native slices).

    Returns:
        dict containing:
            'volume': np.ndarray of shape (D, 3, H, W) if stack_25d else (D, H, W)
            'num_slices': int
            'relative_depths': np.ndarray of shape (D,) with values in [0, 1]
            'meta': dict of series-level metadata
    """
    dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    slice_data = []
    for f in dcm_files:
        try:
            arr, meta = read_dicom_slice_raw(f)
            slice_data.append((arr, meta))
        except Exception:
            continue

    if not slice_data:
        raise ValueError(f"Failed to read any valid DICOM files from {series_dir}")

    # 3D Spatial Sorting along Patient Normal Vector
    try:
        iop = slice_data[0][1]["ImageOrientationPatient"]
        row_vec = np.array(iop[:3])
        col_vec = np.array(iop[3:])
        normal_vec = np.cross(row_vec, col_vec)
        # Sort along dot product projection
        slice_data.sort(key=lambda s: np.dot(np.array(s[1]["ImagePositionPatient"]), normal_vec))
    except Exception:
        # Fallback to ImagePositionPatient[2] or InstanceNumber
        slice_data.sort(key=lambda s: s[1]["ImagePositionPatient"][2] if "ImagePositionPatient" in s[1] else s[1]["InstanceNumber"])

    arrays = [s[0] for s in slice_data]
    metas = [s[1] for s in slice_data]
    raw_volume = np.stack(arrays, axis=0).astype(np.float32)  # (D, orig_H, orig_W)

    # Uniform slice depth resampling if specified
    if target_slices is not None and raw_volume.shape[0] != target_slices:
        current_depth = raw_volume.shape[0]
        indices = np.linspace(0, current_depth - 1, target_slices).round().astype(int)
        raw_volume = raw_volume[indices]
        metas = [metas[i] for i in indices]

    num_slices = raw_volume.shape[0]

    # Intensity normalization
    if normalize == "minmax":
        v_min, v_max = np.percentile(raw_volume, (0.5, 99.5))
        if v_max > v_min:
            norm_volume = np.clip((raw_volume - v_min) / (v_max - v_min), 0.0, 1.0)
        else:
            norm_volume = np.zeros_like(raw_volume)
    elif normalize == "zscore":
        mean = np.mean(raw_volume)
        std = np.std(raw_volume) + 1e-6
        norm_volume = (raw_volume - mean) / std
    else:
        norm_volume = raw_volume

    # In-plane spatial resizing
    target_h, target_w = target_size
    resized_slices = []
    for z in range(num_slices):
        sl = norm_volume[z]
        orig_h, orig_w = sl.shape
        if (orig_h, orig_w) != (target_h, target_w):
            interp = cv2.INTER_AREA if (orig_h >= target_h and orig_w >= target_w) else cv2.INTER_LINEAR
            sl_res = cv2.resize(sl, (target_w, target_h), interpolation=interp)
        else:
            sl_res = sl
        resized_slices.append(sl_res)

    volume_resized = np.stack(resized_slices, axis=0).astype(np.float32)  # (D, target_h, target_w)

    # 2.5D Stacking
    if stack_25d:
        final_volume = stack_25d_channels(volume_resized)  # (D, 3, target_h, target_w)
    else:
        final_volume = volume_resized  # (D, target_h, target_w)

    relative_depths = np.linspace(0.0, 1.0, num_slices, dtype=np.float32)

    return {
        "volume": final_volume,
        "num_slices": num_slices,
        "relative_depths": relative_depths,
        "meta": {
            "orig_shape": (len(slice_data), arrays[0].shape[0], arrays[0].shape[1]),
            "pixel_spacing": metas[0]["PixelSpacing"],
            "slice_thickness": metas[0]["SliceThickness"],
            "orientation": metas[0]["ImageOrientationPatient"],
        },
    }
