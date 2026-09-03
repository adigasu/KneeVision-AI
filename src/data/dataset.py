"""
RSNA Knee Abnormality Detection - PyTorch Dataset for Multi-Sequence Knee MRI.
Loads DICOM series per study, applies slice-level and volumetric transforms,
and formats tensors for MIL / 2.5D / 3D vision encoders.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dicom_reader import load_series_volume
from src.metrics.auc_metrics import TARGET_COLUMNS


class KneeMRIDataset(Dataset):
    """
    Dataset for loading Knee MRI studies with multiple sequences and 12 abnormality targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_df: pd.DataFrame,
        series_dir: str,
        target_slices: int = 24,
        image_size: Tuple[int, int] = (256, 256),
        transforms: Optional[Callable] = None,
        is_training: bool = False,
        preferred_plane: Optional[str] = None,
    ):
        """
        Args:
            df: DataFrame containing StudyInstanceUID and (optionally) the 12 target columns.
            series_df: DataFrame containing StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, etc.
            series_dir: Path to directory containing DICOM series.
            target_slices: Number of slices to sample/resample per series.
            image_size: (Height, Width) to resize each slice.
            transforms: Albumentations compose transform pipeline.
            is_training: If True, enables random slice sampling or training behaviors.
            preferred_plane: If set, filters to a specific Anatomical_Plane ('Sagittal', 'Coronal', 'Axial').
        """
        self.df = df.reset_index(drop=True)
        self.series_df = series_df
        self.series_dir = series_dir
        self.target_slices = target_slices
        self.image_size = image_size
        self.transforms = transforms
        self.is_training = is_training
        self.preferred_plane = preferred_plane

        # Pre-group series by StudyInstanceUID for O(1) lookup
        self.study_to_series = self.series_df.groupby("StudyInstanceUID")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # Find series belonging to this study
        try:
            study_series = self.study_to_series.get_group(study_uid)
            if self.preferred_plane is not None and "Anatomical_Plane" in study_series.columns:
                plane_filtered = study_series[study_series["Anatomical_Plane"] == self.preferred_plane]
                if len(plane_filtered) > 0:
                    study_series = plane_filtered
            
            series_uid = study_series.iloc[0]["SeriesInstanceUID"]
            series_path = os.path.join(self.series_dir, study_uid, series_uid)
            if not os.path.isdir(series_path):
                # Check direct series directory structure
                series_path = os.path.join(self.series_dir, series_uid)
        except Exception:
            series_path = os.path.join(self.series_dir, study_uid)

        # Load 3D volume (D, H, W)
        try:
            volume, _ = load_series_volume(
                series_path,
                target_slices=self.target_slices,
                normalize="minmax",
            )
        except Exception:
            # Fallback zero tensor if DICOM reading fails
            volume = np.zeros((self.target_slices, self.image_size[0], self.image_size[1]), dtype=np.float32)

        # Apply transforms slice-by-slice
        transformed_slices = []
        for s_idx in range(volume.shape[0]):
            slice_2d = volume[s_idx]
            if self.transforms is not None:
                res = self.transforms(image=slice_2d)
                slice_2d = res["image"]
            transformed_slices.append(slice_2d)

        # Shape: (Depth, 1, Height, Width)
        vol_tensor = torch.tensor(np.stack(transformed_slices, axis=0), dtype=torch.float32).unsqueeze(1)

        # Targets extraction
        has_targets = all(col in row for col in TARGET_COLUMNS)
        if has_targets:
            target_vals = row[TARGET_COLUMNS].values.astype(np.float32)
            targets = torch.tensor(target_vals, dtype=torch.float32)
        else:
            targets = torch.full((len(TARGET_COLUMNS),), float("nan"), dtype=torch.float32)

        return {
            "study_uid": study_uid,
            "images": vol_tensor,
            "targets": targets,
        }
