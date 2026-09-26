"""
RSNA Knee Abnormality Detection - Phase 8: Tri-Planar ConvNeXt-Tiny Fold 0 Training.
Jointly ingests Sagittal, Coronal, and Axial series per study, aligns inter-plane features
with Multi-Head Cross-Plane Attention, and trains with Confidence-Weighted Soft BCE Loss.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.data.triplanar_dataset import TriPlanarKneeDataset
from src.models.cross_plane_fusion import TriPlanarKneeModel
from src.training.losses import ConfidenceWeightedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


def triplanar_collate_fn(batch):
    sag = torch.stack([item["sagittal"] for item in batch])
    cor = torch.stack([item["coronal"] for item in batch])
    ax = torch.stack([item["axial"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])
    study_uids = [item["study_uid"] for item in batch]
    is_gold = torch.tensor([item.get("is_gold", False) for item in batch], dtype=torch.bool)

    return {
        "sagittal": sag,
        "coronal": cor,
        "axial": ax,
        "targets": targets,
        "study_uids": study_uids,
        "is_gold": is_gold,
    }


def evaluate_triplanar(model, val_loader, device="cuda"):
    model.eval()
    all_preds, all_targets, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            sag = batch["sagittal"].to(device)
            cor = batch["coronal"].to(device)
            ax = batch["axial"].to(device)

            with torch.amp.autocast("cuda", enabled=("cuda" in str(device))):
                out = model(sag, cor, ax)
                probs = torch.sigmoid(out["logits"])

            all_preds.append(probs.cpu().numpy())
            all_targets.append(batch["targets"].cpu().numpy())
            all_is_gold.append(batch["is_gold"].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    # 1. Full Cohort OOF Macro AUC
    full_metrics = compute_macro_auc(targets, preds)
    full_auc = full_metrics["macro_auc"]

    # 2. Gold Benchmark Human-Annotated Subset AUC
    gold_mask = is_gold.astype(bool)
    if np.sum(gold_mask) >= 4:
        gold_metrics = compute_macro_auc(targets[gold_mask], preds[gold_mask])
        gold_auc = gold_metrics["macro_auc"]
    else:
        gold_auc = float("nan")

    metrics = {
        "macro_auc": full_auc,
        "gold_macro_auc": gold_auc,
        "per_class_auc": full_metrics["per_class_auc"],
    }
    return metrics, preds


def main():
    parser = argparse.ArgumentParser(description="Tri-Planar ConvNeXt-Tiny Training (Fold 0)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--target_slices", type=int, default=16)
    parser.add_argument("--backbone_lr", type=float, default=6e-5)
    parser.add_argument("--head_lr", type=float, default=4e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--labels_path", type=str, default="data/dense_labels_master.parquet")
    parser.add_argument("--splits_path", type=str, default="data/splits_5fold.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/preprocessed_256")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="artifacts/experiments/phase_08_triplanar_fusion")
    args = parser.parse_args()

    seed_everything(42)
    console = Console()

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    log_dir = Path(args.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print(f"🚀 Phase 8: Tri-Planar ConvNeXt-Tiny Training — Fold {args.fold}")
    print(f"   Modality: 3-Plane Fusion (Sagittal + Coronal + Axial: {args.target_slices*3} Slices/Study)")
    print(f"   Device: {args.device} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print("====================================================================")

    # 1. Load Data
    labels_df = pd.read_parquet(args.labels_path)
    splits_df = pd.read_parquet(args.splits_path)

    if "fold" not in labels_df.columns and "fold" in splits_df.columns:
        labels_df = labels_df.merge(splits_df[["StudyInstanceUID", "fold"]].drop_duplicates("StudyInstanceUID"), on="StudyInstanceUID", how="left")

    labels_df["fold"] = labels_df["fold"].fillna(0).astype(int)

    train_df = labels_df[labels_df["fold"] != args.fold].reset_index(drop=True)
    val_df = labels_df[labels_df["fold"] == args.fold].reset_index(drop=True)

    print(f"Train Studies: {len(train_df)} | Val Studies: {len(val_df)}")

    train_ds = TriPlanarKneeDataset(train_df, cache_dir=args.cache_dir, target_slices=args.target_slices, is_training=True)
    val_ds = TriPlanarKneeDataset(val_df, cache_dir=args.cache_dir, target_slices=args.target_slices, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=triplanar_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=triplanar_collate_fn)

    # 2. Build Model
    model = TriPlanarKneeModel(backbone_name="convnext_tiny", pretrained=True, num_classes=12, use_grad_checkpointing=True).to(args.device)

    # Optimizer with differential learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.sag_pool.parameters()) +
        list(model.cor_pool.parameters()) +
        list(model.ax_pool.parameters()) +
        list(model.fusion.parameters()) +
        list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr * 0.1},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = ConfidenceWeightedBCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in str(args.device)))

    best_macro_auc = 0.0
    best_gold_auc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Warmup adjustment
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            optimizer.param_groups[0]["lr"] = args.backbone_lr * warmup_factor
            optimizer.param_groups[1]["lr"] = args.head_lr * warmup_factor
        
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}"):
            sag = batch["sagittal"].to(args.device)
            cor = batch["coronal"].to(args.device)
            ax = batch["axial"].to(args.device)
            targets = batch["targets"].to(args.device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=("cuda" in str(args.device))):
                out = model(sag, cor, ax)
                loss = criterion(out["logits"], targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        if epoch > args.warmup_epochs:
            scheduler.step()

        train_loss /= len(train_loader)

        # Validation
        val_metrics, _ = evaluate_triplanar(model, val_loader, device=args.device)
        macro_auc = val_metrics.get("macro_auc", 0.0)
        gold_auc = val_metrics.get("gold_macro_auc", 0.0)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs}] - Loss: {train_loss:.4f} | Val Macro AUC: {macro_auc:.4f} | Gold AUC: {gold_auc:.4f} | Time: {elapsed:.1f}s")

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_gold_auc = gold_auc
            best_epoch = epoch
            ckpt_path = ckpt_dir / f"triplanar_convnext_tiny_fold{args.fold}_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ⭐ Saved new best Tri-Planar checkpoint: {ckpt_path.name} (Macro AUC: {macro_auc:.4f})")

    print(f"✅ Tri-Planar Fold {args.fold} Training Complete! Best Epoch {best_epoch}: Macro AUC = {best_macro_auc:.4f} | Gold AUC = {best_gold_auc:.4f}")


if __name__ == "__main__":
    main()
