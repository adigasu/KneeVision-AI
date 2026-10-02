import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.metrics.auc_metrics import TARGET_COLUMNS

class MRIAwareKneeDataset(Dataset):
    """
    Dataset supporting laterality normalization (L -> R canonical mirror)
    and dual hard/soft target loading for mixed supervision.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str = 'data/cached_series_384',
        target_slices: int = 32,
        preferred_plane: str = 'Sagittal',
        transforms=None,
        is_training: bool = False,
        return_raw: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.return_raw = return_raw
        self.cache_dir = cache_dir
        self.target_slices = target_slices
        self.preferred_plane = preferred_plane
        self.transforms = transforms
        self.is_training = is_training

        # Load series manifest
        manifest_p = os.path.join(os.path.dirname(cache_dir), 'preprocessed_index.parquet')
        if os.path.exists(manifest_p):
            self.series_df = pd.read_parquet(manifest_p)
            self.study_to_series = self.series_df.groupby('StudyInstanceUID')
        else:
            self.series_df = pd.DataFrame()
            self.study_to_series = None

    def __len__(self):
        return len(self.df)

    def _find_series_path(self, study_uid: str) -> Optional[str]:
        study_dir = os.path.join(self.cache_dir, study_uid)
        if not os.path.isdir(study_dir):
            return None

        if self.preferred_plane and self.study_to_series is not None:
            try:
                group = self.study_to_series.get_group(study_uid)
                plane_matches = group[group['Anatomical_Plane'] == self.preferred_plane]
                if len(plane_matches) > 0:
                    cand = os.path.join(study_dir, f"{plane_matches.iloc[0]['SeriesInstanceUID']}.npy")
                    if os.path.exists(cand):
                        return cand
            except Exception:
                pass

        npy_files = [f for f in os.listdir(study_dir) if f.endswith('.npy')]
        return os.path.join(study_dir, npy_files[0]) if npy_files else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        study_uid = row['StudyInstanceUID']

        series_path = self._find_series_path(study_uid)
        if series_path is None or not os.path.exists(series_path):
            raise FileNotFoundError(f'Missing volume for {study_uid}')

        # (D, 3, H, W)
        volume = np.load(series_path)

        # Depth resampling
        if self.target_slices is not None and volume.shape[0] != self.target_slices:
            cur_d = volume.shape[0]
            indices = np.linspace(0, cur_d - 1, self.target_slices).round().astype(int)
            volume = volume[indices]

        num_slices = volume.shape[0]
        transformed = []
        
        # Apply consistent 3D volume transforms across ALL slices
        replay_dict = None
        for s in range(num_slices):
            img = volume[s].transpose(1, 2, 0) # (H, W, 3)
            if self.transforms is not None:
                if hasattr(self.transforms, 'replay') and hasattr(self.transforms, '__call__'):
                    if s == 0:
                        res = self.transforms(image=img)
                        replay_dict = res.get('replay', None)
                    else:
                        res = self.transforms.replay(replay_dict, image=img) if replay_dict else self.transforms(image=img)
                else:
                    res = self.transforms(image=img)
                t = res['image']
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t.transpose(2, 0, 1), dtype=torch.float32) / 255.0
                elif t.dtype == torch.uint8:
                    t = t.float() / 255.0
            else:
                t = torch.tensor(volume[s], dtype=torch.float32) / 255.0
            transformed.append(t)

        vol_tensor = torch.stack(transformed, dim=0) # (D, 3, H, W)

        # Targets
        hard_vals = [float(row[f'hard_{c}']) if f'hard_{c}' in row and pd.notnull(row[f'hard_{c}']) else float('nan') for c in TARGET_COLUMNS]
        soft_vals = [float(row[f'soft_{c}']) if f'soft_{c}' in row and pd.notnull(row[f'soft_{c}']) else 0.50 for c in TARGET_COLUMNS]

        hard_tensor = torch.tensor(hard_vals, dtype=torch.float32)
        soft_tensor = torch.tensor(soft_vals, dtype=torch.float32)

        out_dict = {
            'study_uid': study_uid,
            'images': vol_tensor,
            'hard_targets': hard_tensor,
            'soft_targets': soft_tensor,
            'hard_labels': hard_tensor,
            'soft_labels': soft_tensor,
            'is_gold': bool(row.get('is_gold', False)),
        }
        if self.return_raw:
            raw_slices = [torch.tensor(volume[s], dtype=torch.float32) / 255.0 for s in range(num_slices)]
            out_dict['raw_images'] = torch.stack(raw_slices, dim=0)
        return out_dict


def mri_aware_collate(batch):
    images = torch.stack([b['images'] for b in batch])
    hard_targets = torch.stack([b['hard_targets'] for b in batch])
    soft_targets = torch.stack([b['soft_targets'] for b in batch])
    study_uids = [b['study_uid'] for b in batch]
    is_gold = torch.tensor([b['is_gold'] for b in batch], dtype=torch.bool)
    batch_dict = {
        'images': images,
        'hard_targets': hard_targets,
        'soft_targets': soft_targets,
        'hard_labels': hard_targets,
        'soft_labels': soft_targets,
        'study_uids': study_uids,
        'study_uid': study_uids,
        'is_gold': is_gold,
    }
    if 'raw_images' in batch[0]:
        batch_dict['raw_images'] = torch.stack([b['raw_images'] for b in batch])
    return batch_dict
