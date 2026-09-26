"""
RSNA Knee Abnormality Detection - High-Speed Foundation Model MIL Benchmarking.
Evaluates 5-fold CV on frozen foundation embeddings in < 60s per model.
Supports single-model and multi-model feature ensembling (concatenated embeddings).
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.table import Table

from src.models.mil_backbone import GatedAttentionMILPool
from src.training.losses import ConfidenceWeightedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


class FrozenEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embeddings_dict: Dict[str, torch.Tensor]):
        self.df = df.reset_index(drop=True)
        self.embeddings_dict = embeddings_dict
        self.targets = torch.tensor(self.df[TARGET_COLUMNS].values, dtype=torch.float32)
        self.is_gold = torch.tensor(self.df["is_gold"].values if "is_gold" in self.df.columns else np.zeros(len(self.df)), dtype=torch.bool)
        uid_col = "StudyInstanceUID" if "StudyInstanceUID" in self.df.columns else "study_uid"
        self.study_uids = self.df[uid_col].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        emb = self.embeddings_dict.get(uid, torch.zeros(16, 512, dtype=torch.float16)).float()
        return {
            "embeddings": emb, # (D, Feat_Dim)
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark Foundation Model MIL Heads")
    parser.add_argument("--models", nargs="+", default=["biomedclip"], help="Models or list of models to evaluate/concatenate")
    parser.add_argument("--labels_path", type=str, default="data/dense_labels_master.parquet")
    parser.add_argument("--splits_path", type=str, default="data/splits_5fold.parquet")
    parser.add_argument("--embeddings_dir", type=str, default="artifacts/experiments/benchmark_foundation_models/embeddings")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(42)
    console = Console()

    # Load master dataframe & splits
    labels_df = pd.read_parquet(args.labels_path)
    splits_df = pd.read_parquet(args.splits_path)

    # Merge fold assignment if not in labels
    uid_col = "StudyInstanceUID" if "StudyInstanceUID" in labels_df.columns else "study_uid"
    if "fold" not in labels_df.columns and "fold" in splits_df.columns:
        splits_uid_col = "StudyInstanceUID" if "StudyInstanceUID" in splits_df.columns else "study_uid"
        labels_df = labels_df.merge(splits_df[[splits_uid_col, "fold"]].rename(columns={splits_uid_col: uid_col}).drop_duplicates(uid_col), on=uid_col, how="left")

    labels_df["fold"] = labels_df["fold"].fillna(0).astype(int)

    # Load embeddings
    loaded_embeddings = {}
    total_dim = 0
    model_tag = "+".join(args.models)

    print(f"\n================================================================")
    print(f"🔬 Foundation Benchmark: {model_tag}")
    print(f"================================================================")

    for m in args.models:
        emb_path = Path(args.embeddings_dir) / f"{m}_embeddings.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}. Run 19_extract_foundation_embeddings.py --model {m} first.")
        
        print(f"Loading {m} embeddings from {emb_path.name}...")
        payload = torch.load(emb_path, map_location="cpu")
        m_embs = payload["embeddings"]
        dim = payload.get("embed_dim", 512)
        total_dim += dim

        for uid, emb in m_embs.items():
            if uid not in loaded_embeddings:
                loaded_embeddings[uid] = []
            loaded_embeddings[uid].append(emb)

    # Concatenate features if multiple
    unified_embeddings = {}
    for uid, emb_list in loaded_embeddings.items():
        if len(emb_list) == 1:
            unified_embeddings[uid] = emb_list[0]
        else:
            unified_embeddings[uid] = torch.cat(emb_list, dim=-1)

    print(f"✅ Total Embedding Feature Dimension: {total_dim} across {len(unified_embeddings)} studies")

    # Run 5-Fold Cross Validation
    fold_results = []
    t_start = time.time()

    for fold in range(5):
        train_df = labels_df[labels_df["fold"] != fold]
        val_df = labels_df[labels_df["fold"] == fold]

        train_ds = FrozenEmbeddingDataset(train_df, unified_embeddings)
        val_ds = FrozenEmbeddingDataset(val_df, unified_embeddings)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=embedding_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=embedding_collate_fn)

        f_auc, g_auc = train_and_eval_fold(fold, train_loader, val_loader, in_features=total_dim, epochs=args.epochs, lr=args.lr, device=args.device)
        fold_results.append((f_auc, g_auc))
        print(f"  Fold {fold}: Full Macro AUC = {f_auc:.4f} | Gold AUC = {g_auc:.4f}")

    total_time = time.time() - t_start
    mean_full = np.mean([r[0] for r in fold_results])
    mean_gold = np.mean([r[1] for r in fold_results])

    table = Table(title=f"Foundation Model MIL Benchmark — {model_tag}")
    table.add_column("Model / Ensemble", style="cyan")
    table.add_column("Feature Dim", justify="right")
    table.add_column("Mean Full Macro AUC", justify="right", style="green")
    table.add_column("Mean Gold AUC", justify="right", style="magenta")
    table.add_column("5-Fold Time", justify="right")

    table.add_row(model_tag, str(total_dim), f"{mean_full:.4f}", f"{mean_gold:.4f}", f"{total_time:.1f}s")
    console.print(table)


if __name__ == "__main__":
    main()