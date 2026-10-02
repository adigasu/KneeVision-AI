"""
RSNA Knee Abnormality Detection - Tri-Planar Multi-View PyTorch Dataset (Phase 13).
Simultaneously loads aligned Sagittal, Coronal, and Axial 2.5D MRI volumes per study (48 slices total: 16 Sag + 16 Cor + 16 Ax)
with learned plane embeddings, sequence priority matching (Fluid-Sensitive & Fat-Suppressed first),
and 12-target Tri-State / Consensus-Denoised label loading.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.metrics.auc_metrics import TARGET_COLUMNS
from src.data.dataset import load_dataframe_auto


class TriPlanarKneeDataset(Dataset):
    """
    Simultaneous 3-view dataset for Knee MRI:
    - Sagittal: 16 slices (plane 0)
    - Coronal:  16 slices (plane 1)
    - Axial:    16 slices (plane 2)
    Total: 48 slices per study of shape (48, 3, H, W).
    """

    PLANES = ["Sagittal", "Coronal", "Axial"]

    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        cache_dir: Optional[str] = None,
        slices_per_plane: int = 16,
        transforms: Optional[Callable] = None,
        is_training: bool = False,
    ):
        self.df = load_dataframe_auto(df).reset_index(drop=True)
        if cache_dir is None:
            cache_dir = "./data/cached_series_384"
            if not os.path.exists(cache_dir):
                fallback = "./data/preprocessed_256"
                if os.path.exists(fallback):
                    cache_dir = fallback

        self.cache_dir = cache_dir
        self.slices_per_plane = slices_per_plane
        self.total_slices = slices_per_plane * 3
        self.transforms = transforms
        self.is_training = is_training

        # Precompute lookup for (StudyInstanceUID, Anatomical_Plane) -> cache_path
        # Prioritizing: Fluid_Sensitive=1, Fat_Suppression=1, num_slices desc
        manifest_parquet = os.path.join(os.path.dirname(self.cache_dir), "preprocessed_index.parquet")
        self.lookup: Dict[Tuple[str, str], str] = {}
        if os.path.exists(manifest_parquet):
            idx_df = pd.read_parquet(manifest_parquet)
            sort_cols = [c for c in ["Fluid_Sensitive", "Fat_Suppression", "num_slices"] if c in idx_df.columns]
            if sort_cols:
                idx_df = idx_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
            best_per_plane = idx_df.drop_duplicates(subset=["StudyInstanceUID", "Anatomical_Plane"])
            for _, row in best_per_plane.iterrows():
                self.lookup[(str(row["StudyInstanceUID"]), str(row["Anatomical_Plane"]))] = str(row["cache_path"])

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_plane_path(self, study_uid: str, plane: str) -> Optional[str]:
        # Fast dictionary lookup
        key = (study_uid, plane)
        if key in self.lookup:
            cand = self.lookup[key]
            if os.path.exists(cand):
                return cand
            # If relative path
            alt = os.path.join(os.path.dirname(self.cache_dir), cand) if not cand.startswith("/") else cand
            if os.path.exists(alt):
                return alt

        # Fallback to study directory inspection
        study_dir = os.path.join(self.cache_dir, study_uid)
        if os.path.isdir(study_dir):
            npy_files = [f for f in os.listdir(study_dir) if f.endswith(".npy")]
            if npy_files:
                return os.path.join(study_dir, npy_files[0])
        return None

    def _load_plane_slices(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        """
        Loads (D, 3, H, W) for one plane.
        Returns:
            slices_tensor: (slices_per_plane, 3, H, W)
            is_valid: bool (True if loaded from real cache, False if dummy zero-filled)
        """
        if path is None or not os.path.exists(path):
            dummy = torch.zeros((self.slices_per_plane, 3, 288, 288), dtype=torch.float32)
            return dummy, False

        try:
            vol = np.load(path)  # (D, 3, H, W)
        except Exception:
            dummy = torch.zeros((self.slices_per_plane, 3, 288, 288), dtype=torch.float32)
            return dummy, False

        # Resample depth to slices_per_plane
        cur_d = vol.shape[0]
        if cur_d != self.slices_per_plane:
            indices = np.linspace(0, cur_d - 1, self.slices_per_plane).round().astype(int)
            vol = vol[indices]

        # Apply transforms volume-consistently across slices of this plane
        transformed = []
        replay_dict = None
        for s in range(self.slices_per_plane):
            img = vol[s].transpose(1, 2, 0)  # (H, W, 3)
            if self.transforms is not None:
                if hasattr(self.transforms, "replay") and hasattr(self.transforms, "__call__"):
                    if s == 0:
                        res = self.transforms(image=img)
                        replay_dict = res.get("replay", None)
                    else:
                        res = self.transforms.replay(replay_dict, image=img) if replay_dict else self.transforms(image=img)
                else:
                    res = self.transforms(image=img)

                t = res["image"]
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                elif t.dtype == torch.uint8:
                    t = t.float() / 255.0
            else:
                t = torch.tensor(vol[s], dtype=torch.float32) / 255.0
            transformed.append(t)

        return torch.stack(transformed, dim=0), True

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool]]:
        row = self.df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])

        plane_tensors = []
        plane_masks = []
        plane_id_list = []

        for p_idx, plane in enumerate(self.PLANES):
            path = self._resolve_plane_path(study_uid, plane)
            plane_tensor, is_valid = self._load_plane_slices(path)
            plane_tensors.append(plane_tensor)
            plane_masks.extend([is_valid] * self.slices_per_plane)
            plane_id_list.extend([p_idx] * self.slices_per_plane)

        images = torch.cat(plane_tensors, dim=0)
        plane_ids = torch.tensor(plane_id_list, dtype=torch.long)
        mask = torch.tensor(plane_masks, dtype=torch.bool)

        hard_vals = [
            float(row[f"hard_{c}"]) if f"hard_{c}" in row and pd.notnull(row[f"hard_{c}"])
            else (float(row[c]) if c in row and pd.notnull(row[c]) else float("nan"))
            for c in TARGET_COLUMNS
        ]
        soft_vals = [
            float(row[f"soft_{c}"]) if f"soft_{c}" in row and pd.notnull(row[f"soft_{c}"]) else 0.50
            for c in TARGET_COLUMNS
        ]

        is_gold = bool(row["is_gold"]) if "is_gold" in row and pd.notnull(row["is_gold"]) else False

        return {
            "study_uid": study_uid,
            "images": images,                         # (48, 3, H, W)
            "plane_ids": plane_ids,                   # (48,)
            "mask": mask,                             # (48,)
            "hard_targets": torch.tensor(hard_vals, dtype=torch.float32),
            "soft_targets": torch.tensor(soft_vals, dtype=torch.float32),
            "is_gold": is_gold,
        }


def triplanar_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["images"] for b in batch])
    plane_ids = torch.stack([b["plane_ids"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    hard_targets = torch.stack([b["hard_targets"] for b in batch])
    soft_targets = torch.stack([b["soft_targets"] for b in batch])
    study_uids = [b["study_uid"] for b in batch]
    is_gold = torch.tensor([b["is_gold"] for b in batch], dtype=torch.bool)

    return {
        "images": images,
        "plane_ids": plane_ids,
        "mask": mask,
        "hard_targets": hard_targets,
        "soft_targets": soft_targets,
        "study_uids": study_uids,
        "is_gold": is_gold,
    }
