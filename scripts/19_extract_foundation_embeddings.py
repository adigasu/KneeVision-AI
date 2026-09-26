"""
RSNA Knee Abnormality Detection - High-Throughput Foundation Model Embedding Extraction.
Extracts frozen slice representations for all studies and saves compact FP16 tensors to disk.
Supported Models:
- biomedclip (512-dim)
- radimagenet (2048-dim)
- dinov2_small (384-dim), dinov2 (768-dim), dinov2_large (1024-dim)
- dinov3_small (384-dim), dinov3 (768-dim), dinov3_large (1024-dim)
- siglip (768-dim), medsiglip (1152-dim)
- convnext_small (768-dim), convnext_tiny (768-dim), swin (1024-dim)
"""

import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.benchmarks.foundation_extractors import get_foundation_extractor
from src.data.dataset import KneeMRIDataset, load_dataframe_auto


def main():
    parser = argparse.ArgumentParser(description="Extract foundation model slice embeddings")
    parser.add_argument("--model", type=str, default="radimagenet",
                        choices=["radimagenet", "biomedclip", "dinov2_small", "dinov2", "dinov2_large",
                                 "dinov3_small", "dinov3", "dinov3_large",
                                 "siglip", "medsiglip", "convnext_small", "convnext_tiny", "swin"],
                        help="Foundation model name")
    parser.add_argument("--labels_path", type=str, default="data/dense_labels_master.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/preprocessed_256")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="artifacts/experiments/benchmark_foundation_models/embeddings")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if embeddings file exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = Path(args.output_dir) / f"{args.model}_embeddings.pt"

    if out_file.exists() and not args.force:
        print(f"⏩ Found existing embeddings for {args.model} at {out_file}. Skipping extraction (use --force to re-extract).")
        return

    print(f"================================================================")
    print(f"🚀 Initializing Extractor: {args.model} on {args.device}")
    print(f"================================================================")

    extractor = get_foundation_extractor(args.model, device=args.device)
    target_img_size = extractor.img_size
    print(f"Model Embed Dim: {extractor.embed_dim} | Target Image Size: {target_img_size}px")

    df = load_dataframe_auto(args.labels_path)
    dataset = KneeMRIDataset(df=df, cache_dir=args.cache_dir, transforms=None, is_training=False)
    
    print(f"Found {len(dataset)} studies in dataset.")

    embeddings_dict = {}
    t_start = time.time()

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Extracting {args.model}"):
            item = dataset[i]
            study_uid = item["study_uid"]
            images = item["images"].to(args.device) # (D, 3, H, W)

            # Resize if needed
            D, C, H, W = images.shape
            if H != target_img_size or W != target_img_size:
                images = F.interpolate(images, size=(target_img_size, target_img_size),
                                       mode="bilinear", align_corners=False)

            with torch.amp.autocast("cuda", enabled=("cuda" in args.device)):
                slice_feats = extractor.extract_slices(images) # (D, embed_dim)

            embeddings_dict[study_uid] = slice_feats.half().cpu()

    elapsed = time.time() - t_start
    print(f"✅ Extracted embeddings for {len(embeddings_dict)} studies in {elapsed:.1f}s ({len(embeddings_dict)/elapsed:.1f} studies/s)")

    save_payload = {
        "model_name": args.model,
        "embed_dim": extractor.embed_dim,
        "img_size": target_img_size,
        "embeddings": embeddings_dict,
    }
    torch.save(save_payload, out_file)
    print(f"💾 Saved {args.model} embeddings to {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
