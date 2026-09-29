"""
RSNA Knee Abnormality Detection - Phase 10: 2-GPU Tri-Planar Multi-View Fine-Tuning Pipeline.
Trains TriPlanar DINOv3 and TriPlanar ConvNeXt-Small with Cross-Plane Attention,
3-epoch Warmup, Cosine Annealing for 25 epochs across 5-Fold Cross Validation.
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from src.data.triplanar_dataset import TriPlanarKneeDataset
from src.data.transforms import get_training_transforms, get_validation_transforms
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


def evaluate_triplanar(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    use_amp: bool = True,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_preds = []
    all_targets = []
    all_uids = []

    with torch.no_grad():
        for batch in val_loader:
            sag = batch["sagittal"].to(device, non_blocking=True)
            cor = batch["coronal"].to(device, non_blocking=True)
            ax = batch["axial"].to(device, non_blocking=True)
            targets = batch["targets"].cpu().numpy()
            uids = batch["study_uids"]

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(sag, cor, ax)
                probs = torch.sigmoid(out["logits"])

                if use_tta:
                    out_flipped = model(
                        torch.flip(sag, dims=[-1]),
                        torch.flip(cor, dims=[-1]),
                        torch.flip(ax, dims=[-1]),
                    )
                    probs_flipped = torch.sigmoid(out_flipped["logits"])
                    probs = 0.5 * (probs + probs_flipped)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets)
            all_uids.extend(uids)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0), all_uids


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_single_fold(
    fold: int,
    args: argparse.Namespace,
    df_dense: pd.DataFrame,
    df_index: pd.DataFrame,
    gold_uids: set,
    console: Console,
    device: str,
    output_dir: str,
) -> Dict[str, Any]:
    console.print(f"\n[bold cyan]══════════════ Starting TriPlanar Fold {fold} ({args.backbone_name} on {device}) ══════════════[/bold cyan]")
    start_time = time.time()

    train_df = df_dense[df_dense["fold"] != fold].reset_index(drop=True)
    val_df = df_dense[df_dense["fold"] == fold].reset_index(drop=True)

    if args.dry_run:
        train_df = train_df.iloc[:24]
        val_df = val_df.iloc[:12]

    console.print(f"Train Studies: [green]{len(train_df):,}[/green] | Val Studies: [yellow]{len(val_df):,}[/yellow]")

    train_transforms = get_training_transforms((args.image_size, args.image_size))
    val_transforms = get_validation_transforms((args.image_size, args.image_size))

    train_dataset = TriPlanarKneeDataset(
        df=train_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        target_slices=args.target_slices,
        transforms=train_transforms,
        is_training=True,
    )
    val_dataset = TriPlanarKneeDataset(
        df=val_df,
        cache_dir=args.cache_dir,
        series_df=df_index,
        target_slices=args.target_slices,
        transforms=val_transforms,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=triplanar_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=triplanar_collate_fn,
    )

    model = TriPlanarKneeModel(
        backbone_name=args.backbone_name,
        pretrained=True,
        num_classes=12,
        mil_hidden_dim=args.mil_hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_grad_checkpointing=True,
    ).to(device)

    criterion = ConfidenceWeightedBCEWithLogitsLoss()

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_val_auc = 0.0
    best_gold_auc = 0.0
    best_metrics = {}
    history = []

    fold_ckpt_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            sag = batch["sagittal"].to(device, non_blocking=True)
            cor = batch["coronal"].to(device, non_blocking=True)
            ax = batch["axial"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(sag, cor, ax)
                loss = criterion(out["logits"], targets)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum_steps
            num_batches += 1

            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        train_loss /= max(num_batches, 1)

        # Validation (Correct order: y_true=targets, y_pred=preds)
        preds, targets, uids = evaluate_triplanar(model, val_loader, device=device, use_amp=args.amp, use_tta=args.tta)
        val_metrics = compute_macro_auc(y_true=targets, y_pred=preds)
        val_auc = val_metrics["macro_auc"]

        # Gold Subset Metric
        gold_indices = [i for i, u in enumerate(uids) if u in gold_uids]
        if len(gold_indices) >= 4:
            gold_metrics = compute_macro_auc(y_true=targets[gold_indices], y_pred=preds[gold_indices])
            gold_auc = gold_metrics["macro_auc"]
        else:
            gold_auc = float("nan")

        current_lr = optimizer.param_groups[-1]["lr"]
        eval_score = val_auc if np.isnan(gold_auc) else (0.4 * val_auc + 0.6 * gold_auc)
        is_best = eval_score > best_val_auc or epoch == 1
        if is_best:
            best_val_auc = eval_score
            best_gold_auc = gold_auc
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "gold_auc": gold_auc,
                    "eval_score": eval_score,
                    "metrics": val_metrics,
                    "args": vars(args),
                },
                os.path.join(fold_ckpt_dir, "best_model.pt"),
            )

        console.print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: [red]{train_loss:.4f}[/red] | "
            f"Val AUC: [bold green]{val_auc:.4f}[/bold green] | "
            f"Gold AUC: [bold magenta]{gold_auc:.4f}[/bold magenta] | LR: {current_lr:.2e} {'🌟' if is_best else ''}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": val_auc,
            "gold_auc": gold_auc,
            "eval_score": eval_score,
            "lr": current_lr,
            "is_best": is_best,
        })

    # Save last epoch checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "val_auc": val_auc,
            "gold_auc": gold_auc,
            "history": history,
        },
        os.path.join(fold_ckpt_dir, "last_model.pt"),
    )

    with open(os.path.join(fold_ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    console.print(f"[bold green]TriPlanar Fold {fold} Complete in {elapsed/60:.1f}m | Best Score: {best_val_auc:.4f} | Val AUC: {val_auc:.4f} | Gold AUC: {best_gold_auc:.4f}[/bold green]")

    return {
        "fold": fold,
        "best_val_auc": val_auc,
        "best_gold_auc": best_gold_auc,
        "best_metrics": best_metrics,
        "elapsed_sec": elapsed,
    }


def run_full_triplanar_training(args: argparse.Namespace):
    console = Console()
    seed_everything(args.seed)

    tag = f"triplanar_{args.backbone_name.replace('/', '_')}_{args.image_size}px_{args.epochs}ep"
    output_dir = os.path.join(args.output_base_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    console.print(f"[bold yellow]════════════════════════════════════════════════════════════════════[/bold yellow]")
    console.print(f"[bold yellow]  Tri-Planar Full Backbone Fine-Tuning: {args.backbone_name} on {args.device} [/bold yellow]")
    console.print(f"[bold yellow]  Epochs: {args.epochs} (Warmup: {args.warmup_epochs}) | Slices per plane: {args.target_slices}[/bold yellow]")
    console.print(f"[bold yellow]  Output Directory: {output_dir}[/bold yellow]")
    console.print(f"[bold yellow]════════════════════════════════════════════════════════════════════[/bold yellow]")

    # Load labels
    df_dense = pd.read_csv(args.labels_csv)
    df_index = pd.read_csv(args.index_csv)

    # Gold UIDs
    gold_uids = set()
    if os.path.exists(args.splits_csv):
        df_splits = pd.read_csv(args.splits_csv)
        labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
        gold_uids = set(df_splits.loc[labeled_mask, "StudyInstanceUID"].astype(str).unique())
    if not gold_uids and "is_gold" in df_dense.columns:
        gold_uids = set(df_dense[df_dense["is_gold"] == True]["StudyInstanceUID"].astype(str).unique())

    console.print(f"Loaded [magenta]{len(gold_uids)}[/magenta] Gold Benchmark Study UIDs")

    folds_to_run = [int(f) for f in args.folds.split(",")] if args.folds else list(range(5))
    results = []

    for fold in folds_to_run:
        fold_res = train_single_fold(
            fold=fold,
            args=args,
            df_dense=df_dense,
            df_index=df_index,
            gold_uids=gold_uids,
            console=console,
            device=args.device,
            output_dir=output_dir,
        )
        results.append(fold_res)

    mean_val = np.mean([r["best_val_auc"] for r in results])
    std_val = np.std([r["best_val_auc"] for r in results])
    mean_gold = np.mean([r["best_gold_auc"] for r in results])
    std_gold = np.std([r["best_gold_auc"] for r in results])

    summary = {
        "backbone": args.backbone_name,
        "image_size": args.image_size,
        "target_slices": args.target_slices,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "mean_val_auc": float(mean_val),
        "std_val_auc": float(std_val),
        "mean_gold_auc": float(mean_gold),
        "std_gold_auc": float(std_gold),
        "folds": results,
    }

    with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    console.print("\n" + "="*80)
    console.print(f"[bold green]5-Fold Tri-Planar CV Completed for {args.backbone_name}![/bold green]")
    console.print(f"Mean Val Macro AUC: [bold cyan]{mean_val:.4f} ± {std_val:.4f}[/bold cyan]")
    console.print(f"Mean Gold Set AUC:  [bold magenta]{mean_gold:.4f} ± {std_gold:.4f}[/bold magenta]")
    console.print("="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel 2-GPU Tri-Planar Full Model Fine-Tuning")
    parser.add_argument("--backbone_name", type=str, default="vit_base_patch16_dinov3", help="Timm backbone name")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device identifier")
    parser.add_argument("--epochs", type=int, default=25, help="Total training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per step (TriPlanar uses 48 slices per study)")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Effective batch size = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate for Cross-Attention head")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.15, help="Learning rate multiplier for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--mil_hidden_dim", type=int, default=128, help="MIL hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Cross-Plane attention heads")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--image_size", type=int, default=288, help="Input image dimension")
    parser.add_argument("--target_slices", type=int, default=16, help="Target slices per anatomical plane")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated folds to train")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--tta", action="store_true", default=True, help="Use Test-Time Augmentation on Val")
    parser.add_argument("--labels_csv", type=str, default="data/dense_labels_master.csv")
    parser.add_argument("--index_csv", type=str, default="data/preprocessed_index.csv")
    parser.add_argument("--splits_csv", type=str, default="data/splits_5fold.csv")
    from src.config import resolve_cache_dir
    parser.add_argument("--cache_dir", type=str, default=str(resolve_cache_dir()), help="Preprocessed series cache directory (from config)")
    parser.add_argument("--output_base_dir", type=str, default="artifacts/experiments/phase_10_triplanar_parallel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_triplanar_training(args)
