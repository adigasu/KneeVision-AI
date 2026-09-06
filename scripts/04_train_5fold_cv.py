"""
RSNA Knee Abnormality Detection - 5-Fold Cross-Validation 2.5D MIL Training Script.
Trains and evaluates 2.5D ConvNeXt / ResNet + Gated Attention MIL models across all 4,407 studies
using Tri-State Master Labels (+1.0, 0.0, NaN) with NaN-masked loss, differential learning rates, and TTA.
"""

import os
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.data.dataset import KneeMRIDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
from src.models.mil_backbone import KneeMILModel
from src.training.losses import AsymmetricLoss, MaskedBCEWithLogitsLoss
from src.training.trainer import KneeTrainer
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS
from src.utils.common import seed_everything


def mil_padded_collate(batch):
    """
    Batches variable-slice series into a padded tensor with a boolean mask.
    """
    depths = [item["images"].shape[0] for item in batch]
    max_d = max(depths)
    b_size = len(batch)
    _, channels, h, w = batch[0]["images"].shape

    padded_images = torch.zeros(b_size, max_d, channels, h, w, dtype=torch.float32)
    masks = torch.zeros(b_size, max_d, dtype=torch.bool)
    targets_stack = torch.stack([item["targets"] for item in batch])

    for i, item in enumerate(batch):
        d = item["images"].shape[0]
        padded_images[i, :d] = item["images"]
        masks[i, :d] = True

    return {
        "images": padded_images,
        "mask": masks,
        "targets": targets_stack,
        "study_uids": [item["study_uid"] for item in batch],
    }


def evaluate_with_tta(model, val_loader, device="cuda", use_amp=True) -> Dict[str, Any]:
    """
    Runs evaluation with Test-Time Augmentation (Original + Horizontal Flip).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_study_uids = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out_orig = model(images, mask=mask)
                probs_orig = torch.sigmoid(out_orig["logits"])

                images_flip = torch.flip(images, dims=[-1])
                out_flip = model(images_flip, mask=mask)
                probs_flip = torch.sigmoid(out_flip["logits"])

                probs_tta = 0.5 * (probs_orig + probs_flip)

            all_preds.append(probs_tta.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_study_uids.extend(batch.get("study_uids", []))

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    auc_res = compute_macro_auc(y_true, y_pred)
    auc_res["y_pred"] = y_pred
    auc_res["y_true"] = y_true
    auc_res["study_uids"] = all_study_uids
    return auc_res


def train_single_fold(
    fold: int,
    args: argparse.Namespace,
    df_master: pd.DataFrame,
    df_index: pd.DataFrame,
    gold_uids: set,
    console: Console,
) -> Dict[str, Any]:
    console.print(f"\n[bold blue]━━━━━━━━━━━━━━━━━━━━ FOLD {fold} / 4 ━━━━━━━━━━━━━━━━━━━━[/bold blue]")

    # Partition train vs validation for this fold across all studies
    train_df = df_master[df_master["fold"] != fold].reset_index(drop=True)
    val_df = df_master[df_master["fold"] == fold].reset_index(drop=True)

    val_gold_count = len(set(val_df["StudyInstanceUID"]).intersection(gold_uids))
    console.print(
        f"Train Studies: [green]{len(train_df)}[/green] | "
        f"Val Studies: [yellow]{len(val_df)}[/yellow] (Gold Studies in Val: [cyan]{val_gold_count}[/cyan])"
    )

    # Automatic image size adjustment for ViTs (DINOv2 patch size 14)
    img_sz = args.image_size
    if 'dinov2' in args.backbone.lower() and img_sz % 14 != 0:
        img_sz = (img_sz // 14) * 14  # e.g., 256 -> 252

    # Create Datasets & Transforms
    train_transforms = get_training_transforms(image_size=img_sz)
    val_transforms = get_validation_transforms(image_size=img_sz)

    train_ds = KneeMRIDataset(
        df=train_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        target_slices=args.target_slices,
        preferred_plane=args.plane,
        transforms=train_transforms,
        is_training=True,
    )
    val_ds = KneeMRIDataset(
        df=val_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        target_slices=args.target_slices,
        preferred_plane=args.plane,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_padded_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mil_padded_collate,
        pin_memory=True,
    )

    # Initialize Model
    model = KneeMILModel(
        backbone_name=args.backbone,
        pretrained=(args.pretrained_weights is None),
        num_classes=12,
        mil_hidden_dim=args.mil_hidden_dim,
        dropout=args.dropout,
    )

    if args.pretrained_weights is not None and os.path.exists(args.pretrained_weights):
        console.print(f"[bold yellow]Loading Pretrained Weights from: {args.pretrained_weights}[/bold yellow]")
        ckpt = torch.load(args.pretrained_weights, map_location="cpu")
        weights = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(weights, strict=False)
        console.print(f"✓ Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Differential Learning Rates (0.2x for backbone, 1.0x for MIL pool & head)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.mil_pool.parameters()) + list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.2},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    if args.loss == "bce":
        criterion = MaskedBCEWithLogitsLoss()
    else:
        criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05)

    exp_tag = f"tristate_{args.backbone}_{args.plane.lower()}"

    # Trainer Engine
    trainer = KneeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        use_amp=args.amp,
        use_ema=args.ema,
        grad_accum_steps=args.grad_accum_steps,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=exp_tag,
    )

    fit_results = trainer.fit(num_epochs=args.epochs, fold=fold)

    # Load best checkpoint for final OOF + TTA evaluation
    ckpt_path = os.path.join(args.checkpoint_dir, f"{exp_tag}_fold{fold}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    best_model = KneeMILModel(
        backbone_name=args.backbone,
        pretrained=False,
        num_classes=12,
        mil_hidden_dim=args.mil_hidden_dim,
        dropout=args.dropout,
    )
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model = best_model.to(trainer.device)

    # Standard Eval vs TTA Eval
    standard_eval = trainer.evaluate(use_ema_for_eval=False)
    tta_eval = evaluate_with_tta(best_model, val_loader, device=trainer.device, use_amp=args.amp)

    # Isolated Gold Subset Eval
    gold_mask = [uid in gold_uids for uid in tta_eval["study_uids"]]
    if any(gold_mask):
        gold_y_true = tta_eval["y_true"][gold_mask]
        gold_y_pred = tta_eval["y_pred"][gold_mask]
        gold_auc_res = compute_macro_auc(gold_y_true, gold_y_pred)
        gold_auc = gold_auc_res["macro_auc"]
    else:
        gold_auc = np.nan

    console.print(
        f"[bold cyan]Fold {fold} Results[/bold cyan] -> "
        f"Standard Val Macro AUC: [magenta]{standard_eval['macro_auc']:.4f}[/magenta] | "
        f"TTA Val Macro AUC: [bold green]{tta_eval['macro_auc']:.4f}[/bold green] | "
        f"Gold Subset AUC: [yellow]{gold_auc:.4f}[/yellow]"
    )

    tta_eval["fold"] = fold
    tta_eval["standard_macro_auc"] = standard_eval["macro_auc"]
    tta_eval["tta_macro_auc"] = tta_eval["macro_auc"]
    tta_eval["gold_macro_auc"] = gold_auc
    return tta_eval


def run_5fold_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", type=str, default="data/tristate_labels_master.parquet")
    parser.add_argument("--splits_path", type=str, default="data/splits_5fold.parquet")
    parser.add_argument("--index_path", type=str, default="data/preprocessed_index.parquet")
    parser.add_argument("--cache_dir", type=str, default="data/preprocessed_256")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--fold", type=int, default=-1, help="0-4 for single fold, or -1 for all 5 folds")
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--plane", type=str, default="Sagittal")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--target_slices", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--loss", type=str, default="bce", choices=["asl", "bce"])
    parser.add_argument("--gamma_neg", type=float, default=2.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    console = Console()

    console.print("[bold green]=== KneeVision-AI: Full-Scale Tri-State Cross-Validation Training ===[/bold green]")
    console.print(
        f"Backbone: [cyan]{args.backbone}[/cyan] | Plane: [cyan]{args.plane}[/cyan] | "
        f"Batch Size: [cyan]{args.batch_size}[/cyan] | Epochs: [cyan]{args.epochs}[/cyan] | "
        f"Loss: [cyan]{args.loss.upper()}[/cyan]"
    )

    # Load master labels & gold UIDs
    df_master = pd.read_parquet(args.labels_path) if args.labels_path.endswith(".parquet") else pd.read_csv(args.labels_path)
    df_index = pd.read_parquet(args.index_path) if args.index_path.endswith(".parquet") else pd.read_csv(args.index_path)

    df_splits = pd.read_parquet(args.splits_path) if args.splits_path.endswith(".parquet") else pd.read_csv(args.splits_path)
    gold_uids = set(df_splits[df_splits[TARGET_COLUMNS].notnull().any(axis=1)]["StudyInstanceUID"])

    folds_to_run = [args.fold] if args.fold >= 0 else list(range(5))
    oof_results = []

    for f in folds_to_run:
        res = train_single_fold(f, args, df_master, df_index, gold_uids, console)
        oof_results.append(res)

    # Summary Table
    table = Table(title=f"Full-Scale Tri-State Cross Validation ({args.backbone} - {args.plane})")
    table.add_column("Fold", style="cyan")
    table.add_column("Standard Val AUC", justify="right", style="blue")
    table.add_column("TTA Val AUC", justify="right", style="magenta")
    table.add_column("Gold Subset AUC", justify="right", style="yellow")

    std_aucs, tta_aucs, gold_aucs = [], [], []
    for res in oof_results:
        table.add_row(
            f"Fold {res['fold']}",
            f"{res['standard_macro_auc']:.4f}",
            f"{res['tta_macro_auc']:.4f}",
            f"{res['gold_macro_auc']:.4f}" if not np.isnan(res['gold_macro_auc']) else "N/A",
        )
        std_aucs.append(res["standard_macro_auc"])
        tta_aucs.append(res["tta_macro_auc"])
        if not np.isnan(res["gold_macro_auc"]):
            gold_aucs.append(res["gold_macro_auc"])

    table.add_section()
    table.add_row(
        "[bold]Mean 5-Fold AUC[/bold]",
        f"[bold blue]{np.mean(std_aucs):.4f}[/bold blue]",
        f"[bold green]{np.mean(tta_aucs):.4f}[/bold green]",
        f"[bold yellow]{np.mean(gold_aucs):.4f}[/bold yellow]" if gold_aucs else "N/A",
    )
    console.print(table)


if __name__ == "__main__":
    run_5fold_training()
