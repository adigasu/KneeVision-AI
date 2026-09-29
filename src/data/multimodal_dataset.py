"""
RSNA Knee Abnormality Detection - Multimodal Dataset (MRI Volume + Spanish Radiology Report).
Provides aligned 2.5D MRI volumes, HuggingFace tokenized radiology reports, and 12-target pseudo labels
for self-supervised vision-language contrastive pretraining.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.metrics.auc_metrics import TARGET_COLUMNS
from src.data.dataset import load_dataframe_auto


class MultimodalKneeDataset(Dataset):
    """
    Dual-stream dataset loading:
    1. Preprocessed 2.5D MRI Volume (D, 3, H, W)
    2. Tokenized Spanish Clinical Radiology Report (input_ids, attention_mask)
    3. 12 Abnormality Pseudo-Labels (y_pseudo in {0, 1})
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        pseudo_df: Optional[Union[pd.DataFrame, str]] = None,
        cache_dir: str = "./data/preprocessed_256",
        series_df: Optional[Union[pd.DataFrame, str]] = None,
        tokenizer_name: str = "dccuchile/bert-base-spanish-wwm-cased",
        max_seq_length: int = 256,
        target_slices: Optional[int] = 32,
        preferred_plane: str = "Sagittal",
        transforms: Optional[Callable] = None,
        is_training: bool = True,
    ):
        """
        Args:
            df: DataFrame containing StudyInstanceUID, Report, and split info.
            pseudo_df: DataFrame containing 12 extracted pseudo labels per StudyInstanceUID.
            cache_dir: Directory containing cached .npy series files.
            series_df: DataFrame mapping SeriesInstanceUID to Anatomical_Plane.
            tokenizer_name: HuggingFace model identifier for Spanish tokenizer.
            max_seq_length: Maximum token length for report encoding.
            target_slices: Resampled depth for uniform batching during pretraining.
            preferred_plane: Target plane for volume selection (default 'Sagittal').
            transforms: Albumentations slice augmentations.
            is_training: Whether data is in training mode.
        """
        self.df = load_dataframe_auto(df)
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.target_slices = target_slices
        self.preferred_plane = preferred_plane
        self.transforms = transforms
        self.is_training = is_training

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Merge pseudo labels if provided
        if pseudo_df is not None:
            pdf = load_dataframe_auto(pseudo_df)
            if "StudyInstanceUID" in pdf.columns:
                cols_to_merge = ["StudyInstanceUID"] + [c for c in TARGET_COLUMNS if c in pdf.columns]
                self.df = pd.merge(self.df, pdf[cols_to_merge], on="StudyInstanceUID", how="left", suffixes=("", "_pseudo"))

        # Load series mapping
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

        return os.path.join(study_dir, npy_files[0])

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])

        # 1. Load Volumetric MRI
        series_path = self._find_series_path(study_uid)
        if series_path is None or not os.path.exists(series_path):
            raise FileNotFoundError(
                f"Missing cached volume for StudyInstanceUID '{study_uid}' in '{self.cache_dir}'. "
                f"Please ensure preprocessing has completed or verify configs/env.yaml data.cache_dir."
            )
        try:
            volume = np.load(series_path)
        except Exception as e:
            raise IOError(
                f"Corrupted cache file for StudyInstanceUID '{study_uid}' at {series_path}: {e}"
            )

        # Depth Resampling
        if self.target_slices is not None and volume.shape[0] != self.target_slices:
            cur_d = volume.shape[0]
            indices = np.linspace(0, cur_d - 1, self.target_slices).round().astype(int)
            volume = volume[indices]

        num_slices = volume.shape[0]

        # Apply slice transforms
        transformed_slices = []
        for s in range(num_slices):
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
            transformed_slices.append(slice_tensor)

        vol_tensor = torch.stack(transformed_slices, dim=0)

        # 2. Tokenize Radiology Report Text
        report_text = str(row.get("Report", "")) if pd.notnull(row.get("Report", "")) else "Sin hallazgos significativos."
        encoded_text = self.tokenizer(
            report_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 3. Extract 12-target Pseudo-Labels
        target_vals = []
        for c in TARGET_COLUMNS:
            col_key = c if c in row else f"{c}_pseudo"
            val = float(row[col_key]) if (col_key in row and pd.notnull(row[col_key])) else 0.0
            target_vals.append(val)
        pseudo_targets = torch.tensor(target_vals, dtype=torch.float32)

        return {
            "study_uid": study_uid,
            "images": vol_tensor,                             # (D, 3, H, W)
            "input_ids": encoded_text["input_ids"].squeeze(0),         # (Seq_Len,)
            "attention_mask": encoded_text["attention_mask"].squeeze(0), # (Seq_Len,)
            "pseudo_targets": pseudo_targets,                  # (12,)
            "num_slices": num_slices,
        }
