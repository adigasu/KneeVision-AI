#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection - Comprehensive Foundation Model Benchmark & Local Inference Profiler.

Evaluates:
1. 5-Fold Cross-Validation Full Val Macro AUC
2. Gold Benchmark Subset (N=58) Macro AUC
3. Local Inference Latency (ms/slice, ms/16-slice study, ms/48-slice triplanar study, FPS, Studies/sec)
4. Model comparisons including DINOv3, MedSigLIP, and RadImageNet
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.mil_backbone import GatedAttentionMILPool
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.training.losses import ConfidenceWeightedBCEWithLogitsLoss
from src.benchmarks.foundation_extractors import get_foundation_extractor

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

def measure_inference_latency(model_name: str, device: str = "cuda:0", num_warmup: int = 15, num_runs: int = 50):
    try:
        extractor = get_foundation_extractor(model_name, device=device)
        total_params = sum(p.numel() for p in extractor.parameters())
        img_size = getattr(extractor, "img_size", 224)
        embed_dim = extractor.embed_dim
        mil_head = StandaloneMILHead(in_features=embed_dim).to(device).eval()

        dummy_slice = torch.randn(1, 3, img_size, img_size, device=device)
        for _ in range(num_warmup):
            _ = extractor.extract_slices(dummy_slice)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            _ = extractor.extract_slices(dummy_slice)
        end_event.record()
        torch.cuda.synchronize()
        slice_latency_ms = start_event.elapsed_time(end_event) / num_runs

        # 16-Slice Study
        dummy_16 = torch.randn(1, 16, 3, img_size, img_size, device=device)
        for _ in range(num_warmup):
            feats = extractor.extract_slices(dummy_16)
            _ = mil_head(feats)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_runs):
            feats = extractor.extract_slices(dummy_16)
            _ = mil_head(feats)
        end_event.record()
        torch.cuda.synchronize()
        study_16_ms = start_event.elapsed_time(end_event) / num_runs

        # 48-Slice Study
        dummy_48 = torch.randn(1, 48, 3, img_size, img_size, device=device)
        for _ in range(num_warmup):
            feats = extractor.extract_slices(dummy_48)
            _ = mil_head(feats)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_runs):
            feats = extractor.extract_slices(dummy_48)
            _ = mil_head(feats)
        end_event.record()
        torch.cuda.synchronize()
        study_48_ms = start_event.elapsed_time(end_event) / num_runs

        del extractor, mil_head
        torch.cuda.empty_cache()

        return {
            "param_count_m": round(total_params / 1e6, 1),
            "embed_dim": embed_dim,
            "img_size": img_size,
            "slice_latency_ms": round(slice_latency_ms, 2),
            "study_16_ms": round(study_16_ms, 2),
            "study_48_ms": round(study_48_ms, 2),
            "throughput_16_studies_sec": round(1000.0 / study_16_ms, 1) if study_16_ms > 0 else 0,
            "throughput_48_studies_sec": round(1000.0 / study_48_ms, 1) if study_48_ms > 0 else 0,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    labels_df = pd.read_parquet("data/dense_labels_master.parquet")
    splits_df = pd.read_parquet("data/splits_5fold.parquet")
    uid_col = "StudyInstanceUID" if "StudyInstanceUID" in labels_df.columns else "study_uid"
    if "fold" not in labels_df.columns and "fold" in splits_df.columns:
        splits_uid_col = "StudyInstanceUID" if "StudyInstanceUID" in splits_df.columns else "study_uid"
        labels_df = labels_df.merge(splits_df[[splits_uid_col, "fold"]].rename(columns={splits_uid_col: uid_col}).drop_duplicates(uid_col), on=uid_col, how="left")
    labels_df["fold"] = labels_df["fold"].fillna(0).astype(int)

    targets_all = torch.tensor(labels_df[TARGET_COLUMNS].values, dtype=torch.float32, device=device)
    is_gold_all = torch.tensor(labels_df["is_gold"].values if "is_gold" in labels_df.columns else np.zeros(len(labels_df)), dtype=torch.bool, device=device)
    folds_all = torch.tensor(labels_df["fold"].values, dtype=torch.long, device=device)
    uids = labels_df[uid_col].astype(str).tolist()

    emb_dir = Path("artifacts/experiments/benchmark_foundation_models/embeddings")
    available_models = ["biomedclip", "dinov2_small", "dinov2", "dinov3", "siglip", "medsiglip", "radimagenet"]
    model_embs = {}
    for m in available_models:
        payload = torch.load(emb_dir / f"{m}_embeddings.pt", map_location="cpu", weights_only=False)
        m_dict = payload["embeddings"]
        dim = payload.get("embed_dim", 512)
        
        tensor_list = []
        for uid in uids:
            if uid in m_dict:
                t = m_dict[uid].float()
                if t.shape[0] < 16:
                    pad = torch.zeros(16 - t.shape[0], dim)
                    t = torch.cat([t, pad], dim=0)
                elif t.shape[0] > 16:
                    t = t[:16]
            else:
                t = torch.zeros(16, dim)
            tensor_list.append(t)
        model_embs[m] = torch.stack(tensor_list, dim=0).to(device)
        print(f"Loaded {m}: {model_embs[m].shape} on {device}")

    configs = [
        # Individual Models
        ["radimagenet"],
        ["dinov3"],
        ["medsiglip"],
        ["dinov2"],
        ["dinov2_small"],
        ["biomedclip"],
        ["siglip"],
        # Key Dual-Foundation Combos
        ["dinov3", "medsiglip"],
        ["dinov3", "biomedclip"],
        ["radimagenet", "dinov3"],
        ["radimagenet", "dinov2"],
        ["radimagenet", "biomedclip"],
        ["radimagenet", "medsiglip"],
        ["dinov2", "biomedclip"],
        # Top Multi-Foundation Fusions
        ["dinov3", "medsiglip", "biomedclip"],
        ["radimagenet", "dinov3", "medsiglip"],
        ["radimagenet", "dinov3", "medsiglip", "biomedclip"],
    ]

    all_results = {}
    print("\n" + "="*85)
    print("🚀 5-FOLD CV BENCHMARK (RADIMAGENET + DINOV3 + MEDSIGLIP + DINOV2 + BIOMEDCLIP)")
    print("="*85)

    for cfg in configs:
        tag = "+".join(cfg)
        t_start = time.time()
        
        if len(cfg) == 1:
            X = model_embs[cfg[0]]
        else:
            X = torch.cat([model_embs[m] for m in cfg], dim=-1)
        
        in_features = X.shape[-1]
        N = X.shape[0]
        fold_metrics = []

        for fold in range(5):
            train_mask = (folds_all != fold)
            val_mask = (folds_all == fold)

            X_tr, y_tr = X[train_mask], targets_all[train_mask]
            X_va, y_va = X[val_mask], targets_all[val_mask]
            gold_va = is_gold_all[val_mask].cpu().numpy()

            torch.manual_seed(42 + fold)
            model = StandaloneMILHead(in_features=in_features).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
            criterion = ConfidenceWeightedBCEWithLogitsLoss()

            batch_size = 64
            n_train = X_tr.shape[0]

            best_val_auc = 0.0
            best_gold_auc = 0.0

            for epoch in range(1, 26):
                model.train()
                perm = torch.randperm(n_train, device=device)
                for i in range(0, n_train, batch_size):
                    idx_batch = perm[i:i+batch_size]
                    optimizer.zero_grad()
                    logits, _ = model(X_tr[idx_batch])
                    loss = criterion(logits, y_tr[idx_batch])
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                if epoch in [10, 15, 20, 25]:
                    model.eval()
                    with torch.no_grad():
                        logits_va, _ = model(X_va)
                        probs_va = torch.sigmoid(logits_va).cpu().numpy()
                        y_va_np = y_va.cpu().numpy()

                    full_m = compute_macro_auc(y_va_np, probs_va)
                    f_auc = full_m["macro_auc"]

                    if np.sum(gold_va) >= 4:
                        gold_m = compute_macro_auc(y_va_np[gold_va], probs_va[gold_va])
                        g_auc = gold_m["macro_auc"]
                    else:
                        g_auc = float("nan")

                    if f_auc > best_val_auc:
                        best_val_auc = f_auc
                        best_gold_auc = g_auc

            fold_metrics.append((best_val_auc, best_gold_auc))

        elapsed = time.time() - t_start
        mean_val = float(np.mean([r[0] for r in fold_metrics]))
        std_val = float(np.std([r[0] for r in fold_metrics]))
        mean_gold = float(np.mean([r[1] for r in fold_metrics]))
        std_gold = float(np.std([r[1] for r in fold_metrics]))

        all_results[tag] = {
            "name": tag,
            "feature_dim": in_features,
            "mean_val_auc": round(mean_val, 4),
            "std_val_auc": round(std_val, 4),
            "mean_gold_auc": round(mean_gold, 4),
            "std_gold_auc": round(std_gold, 4),
            "time_sec": round(elapsed, 2),
            "fold_details": [{"fold": i, "val_auc": round(r[0], 4), "gold_auc": round(r[1], 4)} for i, r in enumerate(fold_metrics)]
        }
        print(f"[{tag:46s}] (Dim {in_features:4d}) | Val Macro AUC: {mean_val:.4f} ± {std_val:.4f} | Gold AUC: {mean_gold:.4f} ± {std_gold:.4f} | Time: {elapsed:.2f}s")

    print("\n" + "="*85)
    print("⚡ PROFILING LOCAL INFERENCE LATENCY & THROUGHPUT (NVIDIA RTX A6000)")
    print("="*85)

    latency_models = [
        "convnext_tiny",
        "convnext_small",
        "dinov2_small",
        "dinov2",
        "dinov3",
        "biomedclip",
        "siglip",
        "medsiglip",
        "radimagenet",
    ]

    latency_results = {}
    for m in latency_models:
        lat = measure_inference_latency(m, device=device)
        latency_results[m] = lat
        print(f"[{m:16s}] Params: {lat.get('param_count_m', '?')}M | 16-Sl: {lat.get('study_16_ms', '?')} ms ({lat.get('throughput_16_studies_sec', '?')} std/s) | 48-Sl: {lat.get('study_48_ms', '?')} ms | Per Slice: {lat.get('slice_latency_ms', '?')} ms")

    out_file = Path("artifacts/experiments/benchmark_foundation_models/results/complete_foundation_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results,
            "latency": latency_results,
        }, f, indent=2)
    print(f"\n✅ All results written to {out_file}")

if __name__ == "__main__":
    main()
