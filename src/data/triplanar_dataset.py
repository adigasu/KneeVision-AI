"""
RSNA Knee Abnormality Detection - Tri-Planar Multi-View PyTorch Dataset.
Simultaneously loads aligned Sagittal, Coronal, and Axial 2.5D MRI volumes per study
along with 12-target Tri-State labels (+1.0 = positive, 0.0 = normal/negated, NaN = unmentioned).
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.metrics.auc_metrics import TARGET_COLUMNS
from src.data.dataset import load_dataframe_auto


class TriPlanarKneeDataset(Dataset):
    """
    Simultaneous 3-view dataset for Knee MRI:
    1. Sagittal Volume (D_s, 3, H, W)
    2. Coronal Volume (D_c, 3, H, W)
    3. Axial Volume (D_a, 3, H, W)
    4. 12 Tri-State Target Labels (y in {1.0, 0.0, NaN})
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        cache_dir: Optional[str] = None,
        series_df: Optional[Union[pd.DataFrame, str]] = None,
        target_slices: int = 24,
        transforms: Optional[Callable] = None,
        is_training: bool = False,
    ):
        self.df = load_dataframe_auto(df)
        if cache_dir is None:
            try:
                from src.config import resolve_cache_dir
                cache_dir = str(resolve_cache_dir())
            except Exception:
                cache_dir = "./data/cached_series_384"

        if not os.path.exists(cache_dir):
            fallback_256 = os.path.join(os.path.dirname(cache_dir), "preprocessed_256")
            if os.path.exists(fallback_256):
                import logging
                logging.getLogger(__name__).warning(
                    f"Cache dir '{cache_dir}' not found. Falling back to '{fallback_256}'."
                )
                cache_dir = fallback_256
            else:
                raise FileNotFoundError(
                    f"Cache directory '{cache_dir}' does not exist. Please check configs/env.yaml or run preprocessing."
                )

        self.cache_dir = cache_dir
        self.target_slices = target_slices
        self.transforms = transforms
        self.is_training = is_training

        # Load series mapping
        if series_df is None:
            manifest_parquet = os.path.join(os.path.dirname(cache_dir), "preprocessed_index.parquet")
            self.series_df = pd.read_parquet(manifest_parquet) if os.path.exists(manifest_parquet) else pd.DataFrame()
        else:
            self.series_df = load_dataframe_auto(series_df)

        if not self.series_df.empty and "StudyInstanceUID" in self.series_df.columns:
            self.study_to_series = self.series_df.groupby("StudyInstanceUID")
        else:
            self.study_to_series = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_plane_series_path(self, study_uid: str, plane: str) -> Optional[str]:
        study_dir = os.path.join(self.cache_dir, study_uid)
        if not os.path.isdir(study_dir):
            return None

        if self.study_to_series is not None:
            try:
                group = self.study_to_series.get_group(study_uid)
                plane_matches = group[group["Anatomical_Plane"] == plane]
                if len(plane_matches) > 0:
                    ser_uid = plane_matches.iloc[0]["SeriesInstanceUID"]
                    cand_path = os.path.join(study_dir, f"{ser_uid}.npy")
                    if os.path.exists(cand_path):
                        return cand_path
            except Exception:
                pass

        # Fallback to any file if specific plane not found
        npy_files = [f for f in os.listdir(study_dir) if f.endswith(".npy")]
        return os.path.join(study_dir, npy_files[0]) if npy_files else None

    def _load_and_transform_plane(self, series_path: Optional[str], study_uid: str = "", plane: str = "") -> torch.Tensor:
        if series_path is None or not os.path.exists(series_path):
            raise FileNotFoundError(
                f"Missing cached volume for StudyInstanceUID '{study_uid}' ({plane} plane) in '{self.cache_dir}'. "
                f"Please ensure preprocessing has completed or verify configs/env.yaml data.cache_dir."
            )
        try:
            volume = np.load(series_path)
        except Exception as e:
            raise IOError(
                f"Corrupted cache file for StudyInstanceUID '{study_uid}' ({plane} plane) at '{series_path}': {e}"
            )

        # Depth Resampling to target_slices
        cur_d = volume.shape[0]
        if cur_d != self.target_slices:
            indices = np.linspace(0, cur_d - 1, self.target_slices).round().astype(int)
            volume = volume[indices]

        # Apply transforms
        transformed = []
        for s in range(self.target_slices):
            slice_img = volume[s].transpose(1, 2, 0)
            if self.transforms is not None:
                res = self.transforms(image=slice_img)
                slice_tensor = res["image"]
                if not isinstance(slice_tensor, torch.Tensor):
                    slice_tensor = torch.tensor(slice_tensor.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                elif slice_tensor.dtype == torch.uint8:
                    slice_tensor = slice_tensor.float() / 255.0
            else:
                slice_tensor = torch.tensor(volume[s], dtype=torch.float32) / 255.0
            transformed.append(slice_tensor)

        return torch.stack(transformed, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])

        # 1. Load 3 views
        sag_path = self._get_plane_series_path(study_uid, "Sagittal")
        cor_path = self._get_plane_series_path(study_uid, "Coronal")
        ax_path = self._get_plane_series_path(study_uid, "Axial")

        sag_tensor = self._load_and_transform_plane(sag_path, study_uid=study_uid, plane="Sagittal")  # (D, 3, H, W)
        cor_tensor = self._load_and_transform_plane(cor_path, study_uid=study_uid, plane="Coronal")   # (D, 3, H, W)
        ax_tensor = self._load_and_transform_plane(ax_path, study_uid=study_uid, plane="Axial")       # (D, 3, H, W)

        # 2. Extract 12 Tri-State Targets (supports +1.0, 0.0, NaN)
        target_vals = []
        for c in TARGET_COLUMNS:
            if c in row and pd.notnull(row[c]):
                target_vals.append(float(row[c]))
            else:
                target_vals.append(float("nan"))

        targets = torch.tensor(target_vals, dtype=torch.float32)

        is_gold = bool(row["is_gold"]) if "is_gold" in row and pd.notnull(row["is_gold"]) else False

        return {
            "study_uid": study_uid,
            "sagittal": sag_tensor,
            "coronal": cor_tensor,
            "axial": ax_tensor,
            "targets": targets,
            "is_gold": is_gold,
        }
