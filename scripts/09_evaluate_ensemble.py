"""
RSNA Knee Abnormality Detection - Out-Of-Fold (OOF) Ensembling & Blending Benchmark.
Evaluates 5-fold predictions across model families:
1. 2.5D ResNet-34 Gated MIL
2. 2.5D DINOv2 ViT-Small Gated MIL
3. Tri-Planar Multi-View Cross-Attention Model (Sagittal + Coronal + Axial)
Computes optimal ensemble blend weights and outputs comprehensive benchmark metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src.data.dataset import KneeMRIDataset
from src.data.triplanar_dataset import TriPlanarKneeDataset
from src.data.transforms import get_validation_transforms
from src.models.mil_backbone import KneeMILModel
from src.models.cross_plane_fusion import TriPlanarKneeModel
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS

console = Console()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_oof_predictions_mil(backbone_name: str, exp_prefix: str, df_master: pd.DataFrame, df_index: pd.DataFrame, image_size: int = 256):
    """Generates 5-Fold OOF predictions for a single-plane MIL backbone."""
    console.print(f"[bold blue]Generating OOF Predictions: {exp_prefix}[/bold blue]")
    all_oof_preds = {}
    all_oof_targets = {}

    val_transforms = get_validation_transforms(image_size=image_size)

    for fold in range(5):
        ckpt_path = f"checkpoints/{exp_prefix}_fold{fold}_best.pth"
        if not os.path.exists(ckpt_path):
            console.print(f"[yellow]Warning: Checkpoint {ckpt_path} not found. Skipping fold {fold}.[/yellow]")
            continue

        val_df = df_master[df_master["fold"] == fold].reset_index(drop=True)
        val_ds = KneeMRIDataset(
            df=val_df,
            cache_dir="data/preprocessed_256",
            series_df=df_index,
            target_slices=24,
            preferred_plane="Sagittal",
            transforms=val_transforms,
            is_training=False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=8,
            collate_fn=lambda b: {
                "images": torch.stack([item["images"] for item in b]),
                "targets": torch.stack([item["targets"] for item in b]),
                "study_uids": [item["study_uid"] for item in b],
            },
        )

        model = KneeMILModel(backbone_name=backbone_name, pretrained=False, num_classes=12)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(DEVICE).eval()

        fold_preds = []
        fold_targets = []
        fold_uids = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["images"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    out = model(imgs)
                    probs = torch.sigmoid(out["logits"])
                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(batch["targets"].numpy())
                fold_uids.extend(batch["study_uids"])

        y_p = np.vstack(fold_preds)
        y_t = np.vstack(fold_targets)

        for i, uid in enumerate(fold_uids):
            all_oof_preds[uid] = y_p[i]
            all_oof_targets[uid] = y_t[i]

    return all_oof_preds, all_oof_targets


def get_oof_predictions_triplanar(exp_prefix: str, df_master: pd.DataFrame, df_index: pd.DataFrame):
    """Generates 5-Fold OOF predictions for Tri-Planar Cross-Attention model."""
    console.print(f"[bold blue]Generating OOF Predictions: {exp_prefix}[/bold blue]")
    all_oof_preds = {}

    val_transforms = get_validation_transforms(image_size=256)

    for fold in range(5):
        ckpt_path = f"checkpoints/{exp_prefix}_fold{fold}_best.pth"
        if not os.path.exists(ckpt_path):
            console.print(f"[yellow]Warning: Checkpoint {ckpt_path} not found. Skipping fold {fold}.[/yellow]")
            continue

        val_df = df_master[df_master["fold"] == fold].reset_index(drop=True)
        val_ds = TriPlanarKneeDataset(
            df=val_df,
            cache_dir="data/preprocessed_256",
            series_df=df_index,
            target_slices=24,
            transforms=val_transforms,
            is_training=False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=8,
            collate_fn=lambda b: {
                "sag": torch.stack([item["sagittal"] for item in b]),
                "cor": torch.stack([item["coronal"] for item in b]),
                "ax": torch.stack([item["axial"] for item in b]),
                "study_uids": [item["study_uid"] for item in b],
            },
        )

        model = TriPlanarKneeModel(backbone_name="resnet34", pretrained=False, num_classes=12)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(DEVICE).eval()

        fold_preds = []
        fold_uids = []

        with torch.no_grad():
            for batch in val_loader:
                sag = batch["sag"].to(DEVICE, non_blocking=True)
                cor = batch["cor"].to(DEVICE, non_blocking=True)
                ax = batch["ax"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    out = model(sag, cor, ax)
                    probs = torch.sigmoid(out["logits"])
                fold_preds.append(probs.cpu().numpy())
                fold_uids.extend(batch["study_uids"])

        y_p = np.vstack(fold_preds)
        for i, uid in enumerate(fold_uids):
            all_oof_preds[uid] = y_p[i]

    return all_oof_preds


def evaluate_and_blend_ensembles():
    console.print("[bold green]=== KneeVision-AI: Out-Of-Fold Ensemble Evaluation & Blending ===[/bold green]\n")

    df_master = pd.read_parquet("data/tristate_labels_master.parquet")
    df_index = pd.read_parquet("data/preprocessed_index.parquet")

    # 1. ResNet-34 2.5D MIL
    preds_resnet, targets_master = get_oof_predictions_mil(
        backbone_name="resnet34",
        exp_prefix="tristate_resnet34_sagittal",
        df_master=df_master,
        df_index=df_index,
        image_size=256,
    )

    # 2. DINOv2 ViT-Small MIL
    preds_dinov2, _ = get_oof_predictions_mil(
        backbone_name="vit_small_patch14_dinov2.lvd142m",
        exp_prefix="tristate_vit_small_patch14_dinov2.lvd142m_sagittal",
        df_master=df_master,
        df_index=df_index,
        image_size=252,
    )

    # 3. Tri-Planar ResNet-34 Multi-View
    preds_triplanar = get_oof_predictions_triplanar(
        exp_prefix="triplanar_resnet34",
        df_master=df_master,
        df_index=df_index,
    )

    # Align common study UIDs
    common_uids = sorted(list(set(preds_resnet.keys()).intersection(set(targets_master.keys()))))
    console.print(f"\n[green]Aligned {len(common_uids):,} evaluated studies across cross-validation splits.[/green]")

    y_t = np.array([targets_master[uid] for uid in common_uids])
    y_resnet = np.array([preds_resnet[uid] for uid in common_uids])

    # Single Model AUC: ResNet-34
    res_resnet = compute_macro_auc(y_t, y_resnet)
    console.print(f" • [cyan]ResNet-34 (2.5D MIL)[/cyan] 5-Fold OOF Macro AUC: [bold magenta]{res_resnet['macro_auc']:.4f}[/bold magenta]")

    # Single Model AUC: DINOv2
    if len(preds_dinov2) > 0:
        dino_uids = sorted(list(set(common_uids).intersection(set(preds_dinov2.keys()))))
        y_t_dino = np.array([targets_master[uid] for uid in dino_uids])
        y_dino = np.array([preds_dinov2[uid] for uid in dino_uids])
        res_dino = compute_macro_auc(y_t_dino, y_dino)
        console.print(f" • [cyan]DINOv2 ViT-Small[/cyan] 5-Fold OOF Macro AUC: [bold magenta]{res_dino['macro_auc']:.4f}[/bold magenta]")

    # Single Model AUC: Tri-Planar
    if len(preds_triplanar) > 0:
        tri_uids = sorted(list(set(common_uids).intersection(set(preds_triplanar.keys()))))
        y_t_tri = np.array([targets_master[uid] for uid in tri_uids])
        y_tri = np.array([preds_triplanar[uid] for uid in tri_uids])
        res_tri = compute_macro_auc(y_t_tri, y_tri)
        console.print(f" • [cyan]Tri-Planar Cross-Attention[/cyan] 5-Fold OOF Macro AUC: [bold magenta]{res_tri['macro_auc']:.4f}[/bold magenta]")

    # Multi-Model Ensemble Blending (Grid Search over weights)
    best_blend_auc = -1.0
    best_weights = {}

    blend_uids = sorted(list(set(preds_resnet.keys()).intersection(set(preds_triplanar.keys())))) if len(preds_triplanar) > 0 else common_uids
    y_t_blend = np.array([targets_master[uid] for uid in blend_uids])
    y_res_b = np.array([preds_resnet[uid] for uid in blend_uids])
    y_tri_b = np.array([preds_triplanar[uid] for uid in blend_uids]) if len(preds_triplanar) > 0 else None
    y_dino_b = np.array([preds_dinov2.get(uid, preds_resnet[uid]) for uid in blend_uids])

    console.print("\n[bold yellow]Optimizing Ensemble Blend Weights...[/bold yellow]")
    for w_res in np.linspace(0.4, 0.9, 6):
        for w_tri in np.linspace(0.1, 0.5, 5):
            w_dino = max(0.0, 1.0 - w_res - w_tri)
            if w_dino < 0:
                continue

            if y_tri_b is not None:
                y_blend = (w_res * y_res_b) + (w_tri * y_tri_b) + (w_dino * y_dino_b)
            else:
                y_blend = (w_res * y_res_b) + ((1.0 - w_res) * y_dino_b)

            score_res = compute_macro_auc(y_t_blend, y_blend)
            score = score_res["macro_auc"]

            if score > best_blend_auc:
                best_blend_auc = score
                best_weights = {
                    "resnet34_weight": float(w_res),
                    "triplanar_weight": float(w_tri) if y_tri_b is not None else 0.0,
                    "dinov2_weight": float(w_dino),
                }

    console.print(f"\n[bold green]⭐ Optimal Ensemble Blend Weights: {best_weights}[/bold green]")
    console.print(f"[bold green]⭐ Blended Multi-Model 5-Fold Macro AUC: {best_blend_auc:.4f}[/bold green]\n")

    # Save weights
    with open("checkpoints/ensemble_weights.json", "w") as f:
        json.dump({
            "weights": best_weights,
            "blended_macro_auc": best_blend_auc,
            "models": ["resnet34_mil", "triplanar_resnet34", "dinov2_vit_small"],
        }, f, indent=2)

    console.print("[green]Saved ensemble configuration to checkpoints/ensemble_weights.json[/green]")


if __name__ == "__main__":
    evaluate_and_blend_ensembles()
