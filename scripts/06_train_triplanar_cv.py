"""
RSNA Knee Abnormality Detection - Tri-Planar Multi-View 5-Fold Cross-Validation Training Script.
Jointly ingests Sagittal, Coronal, and Axial series per study, aligns inter-plane features
with Multi-Head Cross Attention, and trains with Tri-State Masked BCE / ASL across all 4,407 studies.
"""

import os
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.data.triplanar_dataset import TriPlanarKneeDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.cross_plane_fusion import TriPlanarKneeModel
from src.training.losses import AsymmetricLoss, MaskedBCEWithLogitsLoss
from src.training.trainer import ModelEMA
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


def triplanar_collate_fn(batch):
    """Batches 3-view tensors and tri-state targets."""
    sag = torch.stack([item["sagittal"] for item in batch])
    cor = torch.stack([item["coronal"] for item in batch])
    ax = torch.stack([item["axial"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])
    study_uids = [item["study_uid"] for item in batch]

    return {
        "sagittal": sag,
        "coronal": cor,
        "axial": ax,
        "targets": targets,
        "study_uids": study_uids,
    }


def train_single_triplanar_fold(
    fold: int,
    args: argparse.Namespace,
    df_splits: pd.DataFrame,
    df_labels: pd.DataFrame,
    console: Console,
) -> Dict[str, Any]:
    console.print(f"\n[bold blue]━━━━━━━━━━━━━━━ TRI-PLANAR FOLD {fold} / 4 ━━━━━━━━━━━━━━━[/bold blue]")

    # Merge splits with master tri-state labels
    if 'fold' in df_labels.columns:
        merged_df = df_labels.copy()
    else:
        merged_df = pd.merge(df_splits[["StudyInstanceUID", "fold"]], df_labels, on="StudyInstanceUID", how="inner")

    if args.gold_only:
        labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
        gold_uids = set(df_splits[labeled_mask]["StudyInstanceUID"])
        merged_df = merged_df[merged_df["StudyInstanceUID"].isin(gold_uids)].reset_index(drop=True)

    train_df = merged_df[merged_df["fold"] != fold].reset_index(drop=True)
    val_df = merged_df[merged_df["fold"] == fold].reset_index(drop=True)

    console.print(f"Train Studies: [green]{len(train_df):,}[/green] | Validation Studies: [yellow]{len(val_df):,}[/yellow]")

    # Create Datasets
    train_transforms = get_training_transforms(image_size=args.image_size)
    val_transforms = get_validation_transforms(image_size=args.image_size)

    train_ds = TriPlanarKneeDataset(
        df=train_df,
        cache_dir=args.cache_dir,
        series_df=args.index_path,
        target_slices=args.target_slices,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = TriPlanarKneeDataset(
        df=val_df,
        cache_dir=args.cache_dir,
        series_df=args.index_path,
        target_slices=args.target_slices,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=triplanar_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=triplanar_collate_fn,
        pin_memory=True,
    )

    # Initialize Model
    model = TriPlanarKneeModel(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=args.mil_hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_grad_checkpointing=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Differential Learning Rates (backbone 0.2x, fusion & classifier 1.0x)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.2},
        {"params": model.sag_pool.parameters(), "lr": args.lr},
        {"params": model.cor_pool.parameters(), "lr": args.lr},
        {"params": model.ax_pool.parameters(), "lr": args.lr},
        {"params": model.fusion.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    if args.loss == "bce":
        criterion = MaskedBCEWithLogitsLoss()
    else:
        criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema = ModelEMA(model, decay=0.999) if args.ema else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_macro_auc = -1.0
    best_epoch = 0

    # Training Loop
    num_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            sag = batch["sagittal"].to(device, non_blocking=True)
            cor = batch["coronal"].to(device, non_blocking=True)
            ax = batch["axial"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(sag, cor, ax)
                logits = outputs["logits"]
                loss = criterion(logits, targets)
                scaled_loss = loss / args.grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validation Evaluation
        eval_model = ema.module if ema is not None else model
        eval_model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                sag = batch["sagittal"].to(device, non_blocking=True)
                cor = batch["coronal"].to(device, non_blocking=True)
                ax = batch["axial"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = eval_model(sag, cor, ax)
                    probs = torch.sigmoid(outputs["logits"])

                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        y_p = np.vstack(all_preds)
        y_t = np.vstack(all_targets)
        auc_res = compute_macro_auc(y_t, y_p)
        macro_auc = auc_res["macro_auc"]

        avg_loss = total_loss / max(1, num_batches)
        is_best = macro_auc > best_macro_auc
        if is_best:
            best_macro_auc = macro_auc
            best_epoch = epoch
            ckpt_path = os.path.join(args.checkpoint_dir, f"triplanar_{args.backbone}_fold{fold}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": eval_model.state_dict(),
                "macro_auc": macro_auc,
                "per_class_auc": auc_res["per_class_auc"],
            }, ckpt_path)

        status_str = f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Val Macro AUC: [bold magenta]{macro_auc:.4f}[/bold magenta]"
        if is_best:
            status_str += " [bold green]⭐ (Best)[/bold green]"
        console.print(status_str)

    console.print(f"[bold green]✓ Finished Tri-Planar Fold {fold}. Best Val Macro AUC: {best_macro_auc:.4f} at Epoch {best_epoch}[/bold green]")
    return {"fold": fold, "best_macro_auc": best_macro_auc, "best_epoch": best_epoch}


def run_triplanar_cv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_path", type=str, default="data/splits_5fold.parquet")
    parser.add_argument("--labels_path", type=str, default="data/tristate_labels_master.parquet")
    parser.add_argument("--index_path", type=str, default="data/preprocessed_index.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/preprocessed_256")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--fold", type=int, default=-1, help="0-4 for single fold, or -1 for all 5 folds")
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--target_slices", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "asl"])
    parser.add_argument("--gamma_neg", type=float, default=2.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--gold_only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    console = Console()

    console.print("[bold green]=== KneeVision-AI: Tri-Planar Multi-View Cross-Validation Training ===[/bold green]")
    console.print(f"Backbone: [cyan]{args.backbone}[/cyan] | Views: [cyan]Sagittal + Coronal + Axial[/cyan] | Loss: [cyan]{args.loss.upper()}[/cyan]")

    df_splits = pd.read_parquet(args.splits_path) if args.splits_path.endswith(".parquet") else pd.read_csv(args.splits_path)
    df_labels = pd.read_parquet(args.labels_path) if args.labels_path.endswith(".parquet") else pd.read_csv(args.labels_path)

    folds_to_run = [args.fold] if args.fold >= 0 else list(range(5))
    results = []

    for f in folds_to_run:
        res = train_single_triplanar_fold(f, args, df_splits, df_labels, console)
        results.append(res)

    table = Table(title=f"Tri-Planar Cross-Validation Results ({args.backbone})")
    table.add_column("Fold", style="cyan")
    table.add_column("Best Val Macro AUC", justify="right", style="magenta")

    macro_aucs = []
    for res in results:
        table.add_row(f"Fold {res['fold']}", f"{res['best_macro_auc']:.4f}")
        macro_aucs.append(res["best_macro_auc"])

    table.add_section()
    table.add_row("[bold]Mean 5-Fold Macro AUC[/bold]", f"[bold green]{np.mean(macro_aucs):.4f}[/bold green]")
    console.print(table)


if __name__ == "__main__":
    run_triplanar_cv()
