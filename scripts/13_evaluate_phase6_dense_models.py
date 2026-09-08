"""
RSNA Knee Abnormality Detection - Phase 6 Dense Model Evaluation & Benchmarking.
Evaluates out-of-fold (OOF) predictions of 288px dense-trained models across all 4,407 studies
and measures exact Macro ROC-AUC on the 58 Gold Human Ground-Truth Benchmark.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.data.dataset import KneeMRIDataset
from src.data.transforms import get_validation_transforms
from src.models.mil_backbone import KneeMILModel
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mil_padded_collate(batch):
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


def evaluate_backbone_oof(
    backbone_name: str,
    ckpt_prefix: str,
    df_dense: pd.DataFrame,
    df_index: pd.DataFrame,
    image_size: int = 288,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    val_transforms = get_validation_transforms((image_size, image_size))
    all_preds = []
    all_targets = []
    all_uids = []

    for fold in range(5):
        ckpt_path = f"checkpoints/{ckpt_prefix}_fold{fold}_best.pth"
        if not os.path.exists(ckpt_path):
            console.print(f"[yellow]Checkpoint {ckpt_path} not found. Skipping fold {fold}.[/yellow]")
            continue

        console.print(f"Loading {ckpt_path} for Fold {fold} validation...")
        val_df = df_dense[df_dense["fold"] == fold].reset_index(drop=True)
        val_dataset = KneeMRIDataset(
            df=val_df,
            cache_dir="data/preprocessed_256",
            series_df=df_index,
            target_slices=24,
            preferred_plane="Sagittal",
            transforms=val_transforms,
            is_training=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            collate_fn=mil_padded_collate,
        )

        model = KneeMILModel(
            backbone_name=backbone_name,
            pretrained=False,
            num_classes=12,
            mil_hidden_dim=128,
            in_chans=3,
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold} (TTA={use_tta})"):
                images = batch["images"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                with torch.amp.autocast("cuda"):
                    out = model(images, mask=masks)
                    probs = torch.sigmoid(out["logits"])

                    if use_tta:
                        images_flipped = torch.flip(images, dims=[-1])
                        out_flipped = model(images_flipped, mask=masks)
                        probs = 0.5 * (probs + torch.sigmoid(out_flipped["logits"]))

                all_preds.append(probs.cpu().numpy())
                all_targets.append(batch["targets"].numpy())
                all_uids.extend(batch["study_uids"])

    if len(all_preds) == 0:
        return np.array([]), np.array([]), []

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0), all_uids


def run_benchmark():
    console.print("[bold green]=== KneeVision-AI: Phase 6 Dense Model Benchmark & Evaluation ===[/bold green]")

    df_dense = pd.read_parquet("data/dense_labels_master.parquet")
    df_index = pd.read_parquet("data/preprocessed_index.parquet")
    df_train = pd.read_csv("../Datasets/rsna-knee-abnormality-detection/train.csv")

    gold_mask = df_train[TARGET_COLUMNS].notnull().all(axis=1)
    gold_uids = set(df_train[gold_mask]["StudyInstanceUID"])
    console.print(f"Master Dense Dataset: [cyan]{len(df_dense):,}[/cyan] studies | Gold Human Benchmark: [bold green]{len(gold_uids)}[/bold green] studies.")

    # Evaluate ResNet-34 288px
    preds_resnet, targets, uids = evaluate_backbone_oof(
        backbone_name="resnet34",
        ckpt_prefix="dense_288px_resnet34_sagittal",
        df_dense=df_dense,
        df_index=df_index,
        image_size=288,
        use_tta=True,
    )

    if len(preds_resnet) > 0:
        # 1. Full Cohort OOF Metric
        full_res = compute_macro_auc(targets, preds_resnet)
        console.print(f"\n[bold magenta]ResNet-34 (288px Dense) Full Cohort OOF Macro AUC: {full_res['macro_auc']:.4f}[/bold magenta]")

        # 2. Gold Benchmark Subset Metric
        gold_indices = [i for i, uid in enumerate(uids) if uid in gold_uids]
        if len(gold_indices) > 0:
            gold_preds = preds_resnet[gold_indices]
            gold_targets = targets[gold_indices]
            gold_res = compute_macro_auc(gold_targets, gold_preds)
            console.print(f"[bold yellow]⭐ ResNet-34 (288px Dense) Gold Human Benchmark AUC: {gold_res['macro_auc']:.4f}[/bold yellow]\n")

            # Per-Class Breakdown Table
            table = Table(title="Phase 6 ResNet-34 (288px) Per-Class Performance Breakdown")
            table.add_column("Abnormality Target", style="cyan")
            table.add_column("Gold Benchmark AUC", justify="right", style="yellow")
            table.add_column("Full Cohort OOF AUC", justify="right", style="magenta")

            for col in TARGET_COLUMNS:
                g_auc = gold_res["per_class_auc"].get(col, np.nan)
                f_auc = full_res["per_class_auc"].get(col, np.nan)
                table.add_row(
                    col,
                    f"{g_auc:.4f}" if not np.isnan(g_auc) else "N/A",
                    f"{f_auc:.4f}" if not np.isnan(f_auc) else "N/A",
                )

            table.add_section()
            table.add_row(
                "[bold]Macro-Averaged AUC[/bold]",
                f"[bold yellow]{gold_res['macro_auc']:.4f}[/bold yellow]",
                f"[bold magenta]{full_res['macro_auc']:.4f}[/bold magenta]",
            )
            console.print(table)


if __name__ == "__main__":
    run_benchmark()
