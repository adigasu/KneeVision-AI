"""
Unit and integration tests for Preprocessing, Caching, and KneeMRIDataset loading with Parquet & CSV.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import KneeMRIDataset
from src.data.transforms import get_training_transforms, get_validation_transforms


def test_preprocessing_and_dataset():
    # 1. Test Splits Loading (Parquet & CSV)
    splits_parquet = "data/splits_5fold.parquet"
    splits_csv = "data/splits_5fold.csv"
    assert os.path.exists(splits_parquet), f"{splits_parquet} must exist"
    assert os.path.exists(splits_csv), f"{splits_csv} must exist"

    df_splits = pd.read_parquet(splits_parquet)
    labeled_df = df_splits[df_splits["ACL"].notnull()].copy()
    assert len(labeled_df) == 58

    # 2. Test Manifest Loading
    index_parquet = "data/preprocessed_index.parquet"
    assert os.path.exists(index_parquet), f"{index_parquet} must exist"
    df_index = pd.read_parquet(index_parquet)
    assert len(df_index) >= 336

    # 3. Test Dataset Initialization directly from Parquet path
    transforms = get_training_transforms(image_size=256)
    dataset = KneeMRIDataset(
        df=splits_parquet,
        cache_dir="./data/preprocessed_256",
        series_df=index_parquet,
        preferred_plane="Sagittal",
        transforms=transforms,
        is_training=True,
    )
    # 58 labeled + 4349 unlabeled = 4407 total in master splits
    assert len(dataset) == 4407

    # Load labeled item
    item = dataset[0]
    assert "study_uid" in item
    assert "images" in item
    assert "targets" in item
    assert "relative_depths" in item

    images = item["images"]
    assert images.ndim == 4  # (D, 3, H, W)
    assert images.shape[1] == 3
    assert images.shape[2] == 256
    assert images.shape[3] == 256
    assert images.min() >= 0.0
    assert images.max() <= 1.0

    targets = item["targets"]
    assert targets.shape[0] == 12

    # 4. Test DataLoader with MIL Padding Collate Function on labeled subset
    labeled_dataset = KneeMRIDataset(
        df=labeled_df,
        cache_dir="./data/preprocessed_256",
        series_df=df_index,
        preferred_plane="Sagittal",
        transforms=transforms,
        is_training=True,
    )

    def mil_collate(batch):
        depths = [it["images"].shape[0] for it in batch]
        max_d = max(depths)
        b_size = len(batch)

        padded_images = torch.zeros(b_size, max_d, 3, 256, 256, dtype=torch.float32)
        masks = torch.zeros(b_size, max_d, dtype=torch.bool)
        targets_stack = torch.stack([it["targets"] for it in batch])

        for i, it in enumerate(batch):
            d = it["images"].shape[0]
            padded_images[i, :d] = it["images"]
            masks[i, :d] = True

        return {
            "images": padded_images,
            "mask": masks,
            "targets": targets_stack,
            "study_uids": [it["study_uid"] for it in batch],
        }

    loader = DataLoader(labeled_dataset, batch_size=4, shuffle=False, collate_fn=mil_collate)
    batch = next(iter(loader))

    assert batch["images"].shape[0] == 4
    assert batch["images"].shape[2] == 3
    assert batch["mask"].shape[0] == 4
    assert batch["targets"].shape == (4, 12)
    print("\n✓ test_preprocessing_and_dataset with Parquet PASSED successfully!")


if __name__ == "__main__":
    test_preprocessing_and_dataset()
