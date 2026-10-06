"""
RSNA Knee Abnormality Detection - Phase 15: Cleaned Dataset v3 + Curriculum Denoised Training.
Trains TriPlanar ConvNeXt-Tiny (24 slices/plane -> 72 slices total at 288x288)
using cleaned labels (data/dense_labels_master_v3_cleaned.parquet) and
CurriculumDenoisedBCEWithLogitsLoss (late-epoch small-loss sample trimming).
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
from src.models.triplanar_mil import TriPlanarLabelSpecificMILModel
from src.training.losses import CurriculumDenoisedBCEWithLogitsLoss, ConsensusDenoisedBCEWithLogitsLoss


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_soft, all_is_gold = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            plane_ids = batch["plane_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, plane_ids=plane_ids, mask=mask)
                probs = torch.sigmoid(out["logits"]).cpu().numpy()

            all_preds.append(probs)
            all_soft.append(batch["soft_targets"].numpy())
            all_is_gold.append(batch["is_gold"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_soft, axis=0)
    is_gold = np.concatenate(all_is_gold, axis=0)

    full_metrics = compute_macro_auc(targets, preds)
    full_macro = full_metrics["macro_auc"]
    full_per_label = full_metrics["per_class_auc"]

    gold_macro = 0.0
    if is_gold.sum() > 5:
        gold_metrics = compute_macro_auc(targets[is_gold], preds[is_gold])
        gold_macro = gold_metrics["macro_auc"]

    return full_macro, full_per_label, gold_macro, preds


def main():
    parser = argparse.ArgumentParser(description="Phase 15 Cleaned Dataset v3 + Curriculum Denoised Training")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--slices_per_plane", type=int, default=24)
    parser.add_argument("--image_size", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=12, help="12 epochs reaches peak validation")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="curriculum_denoised", choices=["curriculum_denoised", "consensus_denoised"])
    parser.add_argument("--labels_path", type=str, default="data/dense_labels_master_v3_cleaned.parquet")
    parser.add_argument("--trim_ratio", type=float, default=0.10)
    parser.add_argument("--trim_start_epoch", type=int, default=5)
    parser.add_argument("--gold_weight", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    total_slices = args.slices_per_plane * 3
    if not args.run_name:
        args.run_name = f"phase15_triplanar_{args.backbone}_f{args.fold}_curriculum"

    console = Console()
    console.print(f"[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║  Phase 15: Cleaned Dataset v3 + Curriculum Denoised (Fold {args.fold})   ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")
    console.print(f"Backbone: {args.backbone} | Fold: {args.fold} | Slices: {total_slices} ({args.slices_per_plane}/plane) | Size: {args.image_size}x{args.image_size}")
    console.print(f"Device: {args.device} | Batch: {args.batch_size} (Eff: {args.batch_size * args.grad_accum_steps}) | Epochs: {args.epochs}")
    console.print(f"Loss: {args.loss_type} (Gold W: {args.gold_weight}, Trim: {args.trim_ratio} from Ep {args.trim_start_epoch})")
    console.print(f"Labels: {args.labels_path}")

    # Load and merge metadata and labels
    df_splits = pd.read_parquet("data/splits_5fold.parquet")
    df_tri = pd.read_parquet("data/tristate_labels_master.parquet")
    df_dense = pd.read_parquet(args.labels_path)

    df_all = df_splits[["StudyInstanceUID", "fold"]].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all["is_gold"] = labeled_mask

    tri_renamed = df_tri[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"hard_{c}" for c in TARGET_COLUMNS})
    dense_renamed = df_dense[["StudyInstanceUID"] + TARGET_COLUMNS].rename(columns={c: f"soft_{c}" for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on="StudyInstanceUID", how="left")
    df_all = df_all.merge(dense_renamed, on="StudyInstanceUID", how="left")

    train_df = df_all[df_all["fold"] != args.fold].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == args.fold].reset_index(drop=True)

    console.print(f"Train studies: {len(train_df)} | Val studies: {len(val_df)} (Gold in Val: {val_df['is_gold'].sum()})")

    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_ds = TriPlanarKneeDataset(
        train_df,
        slices_per_plane=args.slices_per_plane,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = TriPlanarKneeDataset(
        val_df,
        slices_per_plane=args.slices_per_plane,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=triplanar_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=triplanar_collate,
    )

    model = TriPlanarLabelSpecificMILModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=128,
        dropout=0.3,
        chunk_size=32,
    ).to(args.device)

    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    mil_params = list(model.mil_pool.parameters()) + list(model.plane_embedding.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": mil_params, "lr": args.lr},
    ], weight_decay=1e-2)

    total_steps = args.epochs * (len(train_loader) // args.grad_accum_steps)
    warmup_steps = args.warmup_epochs * (len(train_loader) // args.grad_accum_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    if args.loss_type == "curriculum_denoised":
        criterion = CurriculumDenoisedBCEWithLogitsLoss(
            gold_weight=args.gold_weight,
            trim_ratio=args.trim_ratio,
            trim_start_epoch=args.trim_start_epoch,
        )
    else:
        criterion = ConsensusDenoisedBCEWithLogitsLoss(gold_weight=args.gold_weight)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", f"{args.run_name}_best.pth")
    val_pred_path = os.path.join("checkpoints", f"{args.run_name}_val_preds.npy")

    best_macro_auc = 0.0
    best_gold_auc = 0.0
    best_preds = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(args.device)
            plane_ids = batch["plane_ids"].to(args.device)
            mask = batch["mask"].to(args.device)
            hard = batch["hard_targets"].to(args.device)
            soft = batch["soft_targets"].to(args.device)
            is_gold = batch["is_gold"].to(args.device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(imgs, plane_ids=plane_ids, mask=mask)
                if args.loss_type == "curriculum_denoised":
                    loss = criterion(out["logits"], hard, soft, is_gold=is_gold, epoch=epoch)
                else:
                    loss = criterion(out["logits"], hard, soft, is_gold=is_gold)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        train_loss /= len(train_loader)
        epoch_sec = time.time() - t0

        # Evaluate after each epoch
        val_macro, per_class, val_gold, preds = evaluate(model, val_loader, args.device)

        is_best = val_macro > best_macro_auc
        if is_best:
            best_macro_auc = val_macro
            best_gold_auc = val_gold
            best_preds = preds
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_auc": val_macro,
                "gold_auc": val_gold,
                "per_class_auc": per_class,
                "args": vars(args),
            }, ckpt_path)
            np.save(val_pred_path, best_preds)

        star = " ⭐ BEST" if is_best else ""
        console.print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_sec:.0f}s] | "
            f"Loss: {train_loss:.4f} | Val Macro AUC: {val_macro:.4f} | Gold AUC: {val_gold:.4f}{star}"
        )
        med_oa = per_class.get("Medial OA", 0.0)
        lat_oa = per_class.get("Lateral OA", 0.0)
        pf_oa = per_class.get("PF OA", 0.0)
        mcl = per_class.get("MCL", 0.0)
        fracture = per_class.get("Fracture", 0.0)
        console.print(
            f"   -> Key Pathologies: MedOA: {med_oa:.3f} | LatOA: {lat_oa:.3f} | PFOA: {pf_oa:.3f} | MCL: {mcl:.3f} | Fracture: {fracture:.3f}"
        )

    console.print(f"\n[bold green]✓ Training Complete (Fold {args.fold})![/bold green]")
    console.print(f"Peak Val Macro AUC: [bold yellow]{best_macro_auc:.4f}[/bold yellow] | Gold AUC: [bold yellow]{best_gold_auc:.4f}[/bold yellow]")
    console.print(f"Checkpoint: {ckpt_path}")
    console.print(f"Predictions: {val_pred_path}\n")


if __name__ == "__main__":
    main()
