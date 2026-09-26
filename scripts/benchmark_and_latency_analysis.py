#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection - Comprehensive Foundation Model Benchmark & Local Inference Profiler.

Evaluates:
1. 5-Fold Cross-Validation Full Val Macro AUC
2. Gold Benchmark Subset (N=58) Macro AUC
3. Local Inference Latency (ms/slice, ms/16-slice study, ms/48-slice triplanar study, FPS, Studies/sec)
4. Parameter counts & memory footprint
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.table import Table

# Ensure repo root in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.mil_backbone import GatedAttentionMILPool
from src.training.losses import ConfidenceWeightedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything
from src.benchmarks.foundation_extractors import get_foundation_extractor
import timm


class FrozenEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_dict: Dict[str, torch.Tensor], feat_dim: int):
        self.df = df.reset_index(drop=True)
        self.embeddings_dict = embeddings_dict
        self.feat_dim = feat_dim
        self.targets = torch.tensor(self.df[TARGET_COLUMNS].values, dtype=torch.float32)
        self.is_gold = torch.tensor(self.df["is_gold"].values if "is_gold" in self.df.columns else np.zeros(len(self.df)), dtype=torch.bool)
        uid_col = "StudyInstanceUID" if "StudyInstanceUID" in self.df.columns else "study_uid"
        self.study_uids = self.df[uid_col].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        emb = self.embeddings_dict.get(uid, torch.zeros(16, self.feat_dim, dtype=torch.float32)).float()
        return {
            "embeddings": emb,
            "targets": self.targets[idx],
            "is_gold": self.is_gold[idx],
            "study_uid": uid,
        }


def embedding_collate_fn(batch):
    depths = [item["embeddings"].shape[0] for item in batch]
    max_d = max(depths)
    b_size = len(batch)
    feat_dim = batch[0]["embeddings"].shape[1]

    padded_emb = torch.zeros(b_size, max_d, feat_dim, dtype=torch.float32)
    masks = torch.zeros(b_size, max_d, dtype=torch.bool)

    for i, item in enumerate(batch):
        d = item["embeddings"].shape[0]
        padded_emb[i, :d] = item["embeddings"]
        masks[i, :d] = True

    return {
        "embeddings": padded_emb,
        "mask": masks,
        "targets": torch.stack([item["targets"] for item in batch]),
        "is_gold": torch.stack([item["is_gold"] for item in batch]),
        "study_uids": [item["study_uid"] for item in batch],
    }


class StandaloneMILHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 12, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mil_pool = GatedAttentionMILPool(in_features=in_features, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        pooled, attn = self.mil_pool(x, mask=mask)
        logits = self.classifier(pooled)
        return logits, attn


def train_and_eval_fold(fold_idx, train_loader, val_loader, in_features, epochs=25, lr=1e-3, device="cuda"):
    model = StandaloneMILHead(in_features=in_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = ConfidenceWeightedBCEWithLogitsLoss()

    best_macro_auc = 0.0
    best_gold_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits, _ = model(x, mask=mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets, val_is_gold = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["embeddings"].to(device)
                mask = batch["mask"].to(device)
                logits, _ = model(x, mask=mask)
                probs = torch.sigmoid(logits)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(batch["targets"].cpu().numpy())
                val_is_gold.append(batch["is_gold"].cpu().numpy())

        preds_arr = np.concatenate(val_preds, axis=0)
        targets_arr = np.concatenate(val_targets, axis=0)
        is_gold_arr = np.concatenate(val_is_gold, axis=0)

        full_metrics = compute_macro_auc(targets_arr, preds_arr)
        full_auc = full_metrics["macro_auc"]

        gold_mask = is_gold_arr.astype(bool)
        if np.sum(gold_mask) >= 4:
            gold_metrics = compute_macro_auc(targets_arr[gold_mask], preds_arr[gold_mask])
            gold_auc = gold_metrics["macro_auc"]
        else:
            gold_auc = float("nan")

        if full_auc > best_macro_auc:
            best_macro_auc = full_auc
            best_gold_auc = gold_auc

    return best_macro_auc, best_gold_auc


def measure_inference_latency(model_name: str, device: str = "cuda:0", num_warmup: int = 15, num_runs: int = 50) -> Dict[str, Any]:
    """Measure precise backbone + MIL inference latency on GPU."""
    try:
        extractor = get_foundation_extractor(model_name, device=device)
        total_params = sum(p.numel() for p in extractor.parameters())
        img_size = getattr(extractor, "img_size", 224)
        embed_dim = extractor.embed_dim
        mil_head = StandaloneMILHead(in_features=embed_dim).to(device).eval()

        # Benchmark 1: Single Slice Latency
        dummy_slice = torch.randn(1, 3, img_size, img_size, device=device)
        for _ in range(num_warmup):
            _ = extractor.extract_slices(dummy_slice)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            _ = extractor.extract_slices(dummy_slice)
        end_event.record()
        torch.cuda.synchronize()
        slice_latency_ms = start_event.elapsed_time(end_event) / num_runs

        # Benchmark 2: 16-Slice Sagittal Study
        dummy_16 = torch.randn(1, 16, 3, img_size, img_size, device=device)
        for _ in range(num_warmup):
            feats = extractor.extract_slices(dummy_16)
            _ = mil_head(feats)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_runs):
            feats = extractor.extract_slices(dummy_16)
            _ = mil_head(feats)
        end_event.record()
        torch.cuda.synchronize()
        study_16_ms = start_event.elapsed_time(end_event) / num_runs

        # Benchmark 3: 48-Slice Tri-Planar Study
        dummy_48 = torch.randn(1, 48, 3, img_size, img_size, device=device)
        for _ in range(num_warmup):
            feats = extractor.extract_slices(dummy_48)
            _ = mil_head(feats)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_runs):
            feats = extractor.extract_slices(dummy_48)
            _ = mil_head(feats)
        end_event.record()
        torch.cuda.synchronize()
        study_48_ms = start_event.elapsed_time(end_event) / num_runs

        # MIL head latency only
        dummy_emb_16 = torch.randn(1, 16, embed_dim, device=device)
        start_event.record()
        for _ in range(num_runs):
            _ = mil_head(dummy_emb_16)
        end_event.record()
        torch.cuda.synchronize()
        mil_only_ms = start_event.elapsed_time(end_event) / num_runs

        del extractor, mil_head
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "param_count_m": round(total_params / 1e6, 1),
            "embed_dim": embed_dim,
            "img_size": img_size,
            "slice_latency_ms": round(slice_latency_ms, 2),
            "study_16_ms": round(study_16_ms, 2),
            "study_48_ms": round(study_48_ms, 2),
            "mil_only_ms": round(mil_only_ms, 3),
            "fps_slices": round(1000.0 / slice_latency_ms, 1) if slice_latency_ms > 0 else 0,
            "throughput_16_studies_sec": round(1000.0 / study_16_ms, 1) if study_16_ms > 0 else 0,
            "throughput_48_studies_sec": round(1000.0 / study_48_ms, 1) if study_48_ms > 0 else 0,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def evaluate_foundation_model_combination(
    model_keys: List[str],
    labels_df: pd.DataFrame,
    embeddings_dir: Path,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    seed_everything(42)
    total_dim = 0
    loaded_embeddings = {}

    for m in model_keys:
        emb_path = embeddings_dir / f"{m}_embeddings.pt"
        if not emb_path.exists():
            return {"error": f"Missing {emb_path}"}
        payload = torch.load(emb_path, map_location="cpu")
        m_embs = payload["embeddings"]
        dim = payload.get("embed_dim", 512)
        total_dim += dim
        for uid, emb in m_embs.items():
            if uid not in loaded_embeddings:
                loaded_embeddings[uid] = []
            loaded_embeddings[uid].append(emb.float())

    unified = {}
    for uid, emb_list in loaded_embeddings.items():
        unified[uid] = emb_list[0] if len(emb_list) == 1 else torch.cat(emb_list, dim=-1)

    fold_results = []
    t_start = time.time()
    for fold in range(5):
        train_df = labels_df[labels_df["fold"] != fold]
        val_df = labels_df[labels_df["fold"] == fold]

        train_ds = FrozenEmbeddingDataset(train_df, unified, feat_dim=total_dim)
        val_ds = FrozenEmbeddingDataset(val_df, unified, feat_dim=total_dim)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=embedding_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=embedding_collate_fn)

        f_auc, g_auc = train_and_eval_fold(fold, train_loader, val_loader, in_features=total_dim, epochs=epochs, lr=lr, device=device)
        fold_results.append((f_auc, g_auc))
        print(f"    Fold {fold}: Full Val AUC = {f_auc:.4f} | Gold AUC = {g_auc:.4f}")

    training_time = time.time() - t_start
    mean_val = float(np.mean([r[0] for r in fold_results]))
    std_val = float(np.std([r[0] for r in fold_results]))
    mean_gold = float(np.mean([r[1] for r in fold_results]))
    std_gold = float(np.std([r[1] for r in fold_results]))

    return {
        "model_keys": model_keys,
        "name": " + ".join(model_keys),
        "feature_dim": total_dim,
        "folds": [{"fold": i, "val_macro_auc": round(r[0], 4), "gold_auc": round(r[1], 4)} for i, r in enumerate(fold_results)],
        "val_macro_auc_mean": round(mean_val, 4),
        "val_macro_auc_std": round(std_val, 4),
        "gold_auc_mean": round(mean_gold, 4),
        "gold_auc_std": round(std_gold, 4),
        "cv_train_time_sec": round(training_time, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load master dataframe & splits
    labels_df = pd.read_parquet("data/dense_labels_master.parquet")
    splits_df = pd.read_parquet("data/splits_5fold.parquet")

    uid_col = "StudyInstanceUID" if "StudyInstanceUID" in labels_df.columns else "study_uid"
    if "fold" not in labels_df.columns and "fold" in splits_df.columns:
        splits_uid_col = "StudyInstanceUID" if "StudyInstanceUID" in splits_df.columns else "study_uid"
        labels_df = labels_df.merge(splits_df[[splits_uid_col, "fold"]].rename(columns={splits_uid_col: uid_col}).drop_duplicates(uid_col), on=uid_col, how="left")
    labels_df["fold"] = labels_df["fold"].fillna(0).astype(int)

    emb_dir = Path("artifacts/experiments/benchmark_foundation_models/embeddings")

    # Define model configurations to test
    configs = [
        ["biomedclip"],
        ["dinov2_small"],
        ["dinov2"],
        ["siglip"],
        ["dinov2", "biomedclip"],
        ["dinov2", "siglip"],
        ["biomedclip", "siglip"],
        ["dinov2", "biomedclip", "siglip"],
    ]

    all_results = {}
    print("\n" + "="*80)
    print("🚀 RUNNING 5-FOLD CV EVALUATION ACROSS ALL FOUNDATION MODELS")
    print("="*80)

    for cfg in configs:
        tag = "+".join(cfg)
        print(f"\n--- Benchmarking [{tag}] ---")
        res = evaluate_foundation_model_combination(cfg, labels_df, emb_dir, epochs=args.epochs, device=device)
        all_results[tag] = res
        if "error" not in res:
            print(f"  --> Mean Val Macro AUC: {res['val_macro_auc_mean']:.4f} ± {res['val_macro_auc_std']:.4f} | Gold AUC: {res['gold_auc_mean']:.4f} ± {res['gold_auc_std']:.4f}")

    print("\n" + "="*80)
    print("⚡ PROFILING LOCAL INFERENCE LATENCY & THROUGHPUT (NVIDIA RTX A6000)")
    print("="*80)

    latency_models = [
        "convnext_tiny",
        "convnext_small",
        "dinov2_small",
        "dinov2",
        "biomedclip",
        "siglip",
    ]

    latency_results = {}
    for m in latency_models:
        print(f"Profiling latency for: {m}...")
        lat = measure_inference_latency(m, device=device)
        latency_results[m] = lat
        print(f"  --> {m}: {lat.get('param_count_m', '?')}M params | 16-Slice: {lat.get('study_16_ms', '?')} ms ({lat.get('throughput_16_studies_sec', '?')} studies/s) | 48-Slice: {lat.get('study_48_ms', '?')} ms | Per Slice: {lat.get('slice_latency_ms', '?')} ms")

    out_dir = Path("artifacts/experiments/benchmark_foundation_models/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "foundation_benchmark_summary.json"

    final_payload = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_results": all_results,
        "latency_profiles": latency_results,
    }

    with open(summary_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n✅ All Benchmark & Latency Profiling Saved to: {summary_path}")


if __name__ == "__main__":
    main()
