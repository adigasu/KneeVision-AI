#!/usr/bin/env python3
"""
Extract embeddings for RadImageNet, DINOv3, and MedSigLIP in high-speed parallel workers.
"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.foundation_extractors import get_foundation_extractor
from src.data.dataset import KneeMRIDataset, load_dataframe_auto

def single_study_collate(batch):
    return batch[0]

def extract_for_model(model_name: str, device: str = "cuda:0", labels_path: str = "data/dense_labels_master.parquet",
                      cache_dir: str = "data/preprocessed_256", output_dir: str = "artifacts/experiments/benchmark_foundation_models/embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    out_file = Path(output_dir) / f"{model_name}_embeddings.pt"

    if out_file.exists():
        print(f"⏩ Found existing embeddings for {model_name} at {out_file}. Skipping.")
        return

    print(f"\n================================================================")
    print(f"🚀 Initializing Extractor: {model_name} on {device}")
    print(f"================================================================")

    extractor = get_foundation_extractor(model_name, device=device)
    target_img_size = extractor.img_size
    print(f"Model: {model_name} | Embed Dim: {extractor.embed_dim} | Target Image Size: {target_img_size}px")

    df = load_dataframe_auto(labels_path)
    dataset = KneeMRIDataset(df=df, cache_dir=cache_dir, transforms=None, is_training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=single_study_collate, pin_memory=True)
    
    print(f"Found {len(dataset)} studies in dataset. Extracting on {device}...")

    embeddings_dict = {}
    t_start = time.time()

    with torch.no_grad():
        for item in tqdm(loader, desc=f"Extracting {model_name}"):
            study_uid = item["study_uid"]
            images = item["images"].to(device, non_blocking=True)

            D, C, H, W = images.shape
            if H != target_img_size or W != target_img_size:
                images = F.interpolate(images, size=(target_img_size, target_img_size),
                                       mode="bilinear", align_corners=False)

            with torch.amp.autocast("cuda", enabled=("cuda" in device)):
                slice_feats = extractor.extract_slices(images)

            embeddings_dict[study_uid] = slice_feats.half().cpu()

    elapsed = time.time() - t_start
    print(f"✅ Extracted embeddings for {len(embeddings_dict)} studies in {elapsed:.1f}s ({len(embeddings_dict)/elapsed:.1f} studies/s)")

    save_payload = {
        "model_name": model_name,
        "embed_dim": extractor.embed_dim,
        "img_size": target_img_size,
        "embeddings": embeddings_dict,
    }
    torch.save(save_payload, out_file)
    print(f"💾 Saved {model_name} embeddings to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")

def main():
    models_to_run = ["radimagenet", "dinov3", "medsiglip"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for m in models_to_run:
        extract_for_model(m, device=device)

if __name__ == "__main__":
    main()
