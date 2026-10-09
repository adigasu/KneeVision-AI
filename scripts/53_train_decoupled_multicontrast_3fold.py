"""
RSNA Knee Abnormality Detection - Multi-Contrast 3-Fold Decoupled MIL Training.
Trains either:
- Independent T1 Decoupled model (--contrast_mode t1_only)
- Independent T2 Decoupled model (--contrast_mode t2_only)
- Joint Multi-Contrast 6-Stream model (--contrast_mode dual)
on 3-Fold Multilabel Stratified CV (Folds 0, 1, 2).
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.data.triplanar_dataset import TriPlanarKneeDataset, triplanar_collate
from src.models.plane_decoupled_mil import PlaneDecoupledTriPlanarMILModel
from src.models.contrast_decoupled_mil import ContrastDecoupledMILModel
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_soft, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, mask=mask)
                probs = torch.sigmoid(out["logits"]).cpu().numpy()

            all_preds.append(probs)
            all_soft.append(batch["soft_targets"].numpy())
            all_is_gold.append(batch["is_gold"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_soft, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    # Full macro AUC across all validation studies
    full_metrics = compute_macro_auc(targets, preds)
    full_macro = full_metrics["macro_auc"]
    full_per_label = full_metrics["per_class_auc"]

    # Gold subset macro AUC (strict human consensus annotations)
    gold_macro = 0.0
    gold_per_label = {}
    if is_gold.sum() > 3:
        gold_metrics = compute_macro_auc(targets[is_gold], preds[is_gold])
        gold_macro = gold_metrics["macro_auc"]
        gold_per_label = gold_metrics["per_class_auc"]

    return full_macro, full_per_label, gold_macro, gold_per_label, preds


def run_training(args):
    console = Console()
    console.print(f"[bold cyan]=== KneeVision Multi-Contrast Decoupled Training | Mode: {args.contrast_mode} | Fold {args.fold} (3-Fold CV) ===[/bold cyan]")
    console.print(f"Backbone: [green]{args.backbone}[/green] | Img Size: [green]{args.image_size}[/green] | Slices/Stream: [green]{args.slices_per_plane}[/green]")

    # 1. Load Data
    df_splits = pd.read_parquet(args.splits_path)
    df_tri = pd.read_parquet("data/tristate_labels_master.parquet")
    df_dense = pd.read_parquet("data/dense_labels_master.parquet")

    df_all = df_splits[["StudyInstanceUID", "fold"]].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all["is_gold"] = labeled_mask

    tri_renamed = df_tri[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"hard_{c}" for c in TARGET_COLUMNS})
    dense_renamed = df_dense[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on="StudyInstanceUID", how="left")
    df_all = df_all.merge(dense_renamed, on="StudyInstanceUID", how="left")

    train_df = df_all[df_all["fold"] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == args.fold].reset_index(drop=True)

    console.print(f"Train Studies: {len(train_df):,} (Gold: {train_df['is_gold'].sum()})")
    console.print(f"Val Studies:   {len(val_df):,} (Gold: {val_df['is_gold'].sum()})")

    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_ds = TriPlanarKneeDataset(
        train_df,
        slices_per_plane=args.slices_per_plane,
        transforms=train_transforms,
        is_training=True,
        contrast_mode=args.contrast_mode,
    )
    val_ds = TriPlanarKneeDataset(
        val_df,
        slices_per_plane=args.slices_per_plane,
        transforms=val_transforms,
        is_training=False,
        contrast_mode=args.contrast_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=triplanar_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=triplanar_collate,
        pin_memory=True,
    )

    # 2. Instantiate Model
    if args.contrast_mode == "dual":
        model = ContrastDecoupledMILModel(
            backbone_name=args.backbone,
            pretrained=True,
            num_classes=12,
            mil_hidden_dim=args.mil_hidden_dim,
            dropout=args.dropout,
            chunk_size=args.chunk_size,
            slices_per_plane=args.slices_per_plane,
            classifier_type=args.classifier_type,
        )
    else:
        model = PlaneDecoupledTriPlanarMILModel(
            backbone_name=args.backbone,
            pretrained=True,
            num_classes=12,
            mil_hidden_dim=args.mil_hidden_dim,
            dropout=args.dropout,
            chunk_size=args.chunk_size,
            slices_per_plane=args.slices_per_plane,
            classifier_type=args.classifier_type,
        )

    model.to(args.device)

    # 3. Loss Criterion
    criterion = ConsensusDenoisedBCEWithLogitsLoss(
        gold_weight=args.gold_weight,
        pos_thresh=args.pos_thresh,
        neg_thresh=args.neg_thresh,
    )

    # 4. Optimizer & Scheduler
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-2,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    # 5. Training Loop
    os.makedirs("checkpoints", exist_ok=True)
    best_gold_auc = 0.0
    best_val_macro = 0.0
    best_preds = None
    checkpoint_path = f"checkpoints/{args.run_name}_best.pth"
    val_pred_path = f"checkpoints/{args.run_name}_val_preds.npy"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(args.device)
            mask = batch["mask"].to(args.device)
            hard = batch["hard_targets"].to(args.device)
            soft = batch["soft_targets"].to(args.device)
            is_gold = batch["is_gold"].to(args.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, mask=mask)
                loss = criterion(out["logits"], hard, soft, is_gold)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scale_before <= scaler.get_scale():
                    scheduler.step()

        train_loss /= len(train_loader)
        epoch_time = time.time() - t0

        # Evaluate
        val_macro, val_per_label, gold_macro, gold_per_label, preds = evaluate(model, val_loader, args.device)

        is_best = val_macro > best_val_macro
        if is_best:
            best_val_macro = val_macro
            best_gold_auc = gold_macro
            best_preds = preds
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_auc": val_macro,
                    "gold_macro_auc": gold_macro,
                    "per_class_auc": val_per_label,
                    "gold_per_class_auc": gold_per_label,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            np.save(val_pred_path, preds)
            np.save(f"checkpoints/{args.run_name}_val_uids.npy", val_df["StudyInstanceUID"].values)

        star = " ★ NEW BEST" if is_best else ""
        oa_str = f"MedOA: {val_per_label.get('Medial OA', 0):.3f} | LatOA: {val_per_label.get('Lateral OA', 0):.3f} | PFOA: {val_per_label.get('PF OA', 0):.3f} | Frac: {val_per_label.get('Fracture', 0):.3f}"
        console.print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.0f}s] | Loss: {train_loss:.4f} | Val Macro AUC: [bold green]{val_macro:.4f}[/bold green] | Gold AUC: [bold magenta]{gold_macro:.4f}[/bold magenta]{star}\n"
            f"   -> MCL: {val_per_label.get('MCL', 0):.4f} | {oa_str}"
        )

    console.print(f"\n[bold green]✓ Training Complete for {args.run_name}![/bold green]")
    console.print(f"Best Val Macro AUC: [bold green]{best_val_macro:.4f}[/bold green] (Gold: [bold magenta]{best_gold_auc:.4f}[/bold magenta])")
    console.print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default="convnext_small")
    parser.add_argument("--contrast_mode", type=str, default="t1_only", choices=["t1_only", "t2_only", "dual"])
    parser.add_argument("--splits_path", type=str, default="data/splits_3fold.parquet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--slices_per_plane", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--gold_weight", type=float, default=2.0)
    parser.add_argument("--pos_thresh", type=float, default=0.70)
    parser.add_argument("--neg_thresh", type=float, default=0.30)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--classifier_type", type=str, default="linear")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_name", type=str, default="phase17_decoupled_t1_f0")
    args = parser.parse_args()

    run_training(args)
