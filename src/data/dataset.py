"""
RSNA Knee Abnormality Detection - PyTorch Dataset for Multi-Sequence Knee MRI.
Supports loading from fast preprocessed 2.5D disk cache (.npy) as well as Parquet/CSV manifests.
Provides single-plane and tri-planar multi-view study collation for MIL models.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.metrics.auc_metrics import TARGET_COLUMNS


def load_dataframe_auto(source: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Helper to load a DataFrame from a Parquet/CSV path or return directly if already a DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.reset_index(drop=True)
    elif isinstance(source, str):
        if source.endswith(".parquet") or os.path.exists(source.replace(".csv", ".parquet")):
            parquet_path = source if source.endswith(".parquet") else source.replace(".csv", ".parquet")
            return pd.read_parquet(parquet_path)
        return pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported data source type: {type(source)}")


class KneeMRIDataset(Dataset):
    """
    High-Performance PyTorch Dataset for Knee MRI 2.5D MIL volumes.
    Loads preprocessed (D, 3, H, W) slice tensors from disk cache with on-the-fly augmentations.
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        cache_dir: str = "./data/preprocessed_256",
        series_df: Optional[Union[pd.DataFrame, str]] = None,
        target_slices: Optional[int] = None,
        preferred_plane: Optional[str] = None,
        transforms: Optional[Callable] = None,
        is_training: bool = False,
    ):
        """
        Args:
            df: DataFrame or path to .parquet/.csv containing StudyInstanceUID and target labels.
            cache_dir: Path to directory containing preprocessed {study_uid}/{series_uid}.npy.
            series_df: DataFrame or path to .parquet/.csv with SeriesInstanceUID and Anatomical_Plane mappings.
            target_slices: If set, resamples slices to fixed depth; if None, keeps all native slices (for MIL).
            preferred_plane: If set, filters to specific plane ('Sagittal', 'Coronal', 'Axial').
            transforms: Albumentations compose transform pipeline applied per slice.
            is_training: Boolean flag for training vs validation.
        """
        self.df = load_dataframe_auto(df)
        self.cache_dir = cache_dir
        self.target_slices = target_slices
        self.preferred_plane = preferred_plane
        self.transforms = transforms
        self.is_training = is_training

        # Load series mapping (checks Parquet first, then CSV)
        if series_df is None:
            manifest_parquet = os.path.join(os.path.dirname(cache_dir), "preprocessed_index.parquet")
            manifest_csv = os.path.join(os.path.dirname(cache_dir), "preprocessed_index.csv")
            if os.path.exists(manifest_parquet):
                self.series_df = pd.read_parquet(manifest_parquet)
            elif os.path.exists(manifest_csv):
                self.series_df = pd.read_csv(manifest_csv)
            else:
                self.series_df = pd.DataFrame()
        else:
            self.series_df = load_dataframe_auto(series_df)

        if not self.series_df.empty and "StudyInstanceUID" in self.series_df.columns:
            self.study_to_series = self.series_df.groupby("StudyInstanceUID")
        else:
            self.study_to_series = None

    def __len__(self) -> int:
        return len(self.df)

    def _find_series_path(self, study_uid: str) -> Optional[str]:
        study_dir = os.path.join(self.cache_dir, study_uid)
        if not os.path.isdir(study_dir):
            return None

        npy_files = [f for f in os.listdir(study_dir) if f.endswith(".npy")]
        if not npy_files:
            return None

        # If preferred plane is requested, filter via series_df
        if self.preferred_plane is not None and self.study_to_series is not None:
            try:
                group = self.study_to_series.get_group(study_uid)
                plane_matches = group[group["Anatomical_Plane"] == self.preferred_plane]
                if len(plane_matches) > 0:
                    preferred_ser_uid = plane_matches.iloc[0]["SeriesInstanceUID"]
                    cand_path = os.path.join(study_dir, f"{preferred_ser_uid}.npy")
                    if os.path.exists(cand_path):
                        return cand_path
            except Exception:
                pass

        # Default: return the first available series in the study
        return os.path.join(study_dir, npy_files[0])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        series_path = self._find_series_path(study_uid)
        if series_path is not None and os.path.exists(series_path):
            try:
                # Load uint8 volume: shape (D, 3, H, W)
                volume = np.load(series_path)
            except Exception:
                volume = np.zeros((24, 3, 256, 256), dtype=np.uint8)
        else:
            volume = np.zeros((24, 3, 256, 256), dtype=np.uint8)

        # Depth resampling if fixed slice count is requested
        if self.target_slices is not None and volume.shape[0] != self.target_slices:
            cur_d = volume.shape[0]
            indices = np.linspace(0, cur_d - 1, self.target_slices).round().astype(int)
            volume = volume[indices]

        num_slices = volume.shape[0]

        # Apply transforms slice-by-slice
        transformed_slices = []
        for s in range(num_slices):
            slice_img = volume[s].transpose(1, 2, 0)  # (H, W, 3) uint8
            if self.transforms is not None:
                res = self.transforms(image=slice_img)
                slice_tensor = res["image"]  # (3, H, W) float tensor
                if not isinstance(slice_tensor, torch.Tensor):
                    slice_tensor = torch.tensor(slice_tensor.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                elif slice_tensor.dtype == torch.uint8:
                    slice_tensor = slice_tensor.float() / 255.0
            else:
                slice_tensor = torch.tensor(volume[s], dtype=torch.float32) / 255.0

            transformed_slices.append(slice_tensor)

        # Stack into (Depth, 3, Height, Width)
        vol_tensor = torch.stack(transformed_slices, dim=0)

        # Relative depths array p_z in [0, 1]
        relative_depths = torch.tensor(np.linspace(0.0, 1.0, num_slices, dtype=np.float32))

        # Extract 12 target binary labels
        target_cols_present = [col for col in TARGET_COLUMNS if col in row and pd.notnull(row[col])]
        if len(target_cols_present) == len(TARGET_COLUMNS):
            target_vals = row[TARGET_COLUMNS].values.astype(np.float32)
            targets = torch.tensor(target_vals, dtype=torch.float32)
        else:
            targets = torch.full((len(TARGET_COLUMNS),), float("nan"), dtype=torch.float32)

        return {
            "study_uid": study_uid,
            "images": vol_tensor,  # (D, 3, H, W)
            "relative_depths": relative_depths,  # (D,)
            "num_slices": num_slices,
            "targets": targets,  # (12,)
        }
