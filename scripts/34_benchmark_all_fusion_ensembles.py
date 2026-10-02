"""
RSNA Knee Abnormality Detection - Multi-Backbone Ensemble & Fusion Benchmark
Benchmarking 6 models (ConvNeXt-Tiny, ConvNeXt-Small, DINOv2-Small across Fold 0 & Fold 1)
over 8 ensemble and fusion strategies:
1. ConvNeXt-Tiny Standalone (Baseline)
2. ConvNeXt-Small Standalone (Baseline)
3. DINOv2-Small Standalone (Baseline)
4. Uniform Mean Blend (Simple Late Fusion)
5. Calibrated Rank Averaging (Percentile Monotonic Ranking)
6. SLSQP Stacking Meta-Learner (Pathology-Specific Denoised Stacking)
7. Bottleneck Feature Fusion (Middle Fusion with Residual Bypass)
8. Residual GatedFusion (Dynamic Context-Conditioned Modality Gating)
9. Pathology-Grouped Soft-MoE (Clinical Specialist Routing)
10. Dual-Axis Label-Specific ABMIL (Hierarchical Slice-Depth & Model Attention)
11. Pathology Slot Attention (Competitive Object-Centric Slot Normalization)

All artifacts, checkpoints, logs, and benchmark summaries are strictly preserved in:
artifacts/experiments/phase_11_fusion_benchmarks/
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table
import timm

try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from src.data.mri_aware_dataset import MRIAwareKneeDataset, mri_aware_collate
from src.data.transforms import get_validation_transforms
from src.models.mil_backbone import LabelSpecificGatedAttentionMILPool
from src.models.ensemble_fusion import (
    CalibratedRankAveraging,
    BottleneckFeatureFusion,
    ResidualGatedFusion,
    DualAxisLabelSpecificABMIL,
    PathologyGroupedSoftMoE,
    PathologySlotAttention,
)
from src.training.losses import ConsensusDenoisedBCEWithLogitsLoss
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS

console = Console()

class BaseMILFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', mil_hidden_dim=128):
        super().__init__()
        self.backbone_name = backbone_name
        extra_kwargs = {}
        if 'dinov2' in backbone_name.lower():
            extra_kwargs = {'img_size': 336, 'dynamic_img_size': True}
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, in_chans=3, **extra_kwargs)
        self.num_features = self.backbone.num_features
        self.mil_pool = LabelSpecificGatedAttentionMILPool(in_features=self.num_features, num_classes=12, hidden_dim=mil_hidden_dim)

    def forward(self, images):
        B, D, C, H, W = images.shape
        x_flat = images.view(B * D, C, H, W)
        feats_flat = self.backbone(x_flat)
        slice_feats = feats_flat.view(B, D, self.num_features)

        A_V = self.mil_pool.attention_V(slice_feats)
        A_U = self.mil_pool.attention_U(slice_feats)
        raw_attn = self.mil_pool.attention_weights(A_V * A_U).transpose(1, 2)
        attn = torch.softmax(raw_attn, dim=-1)
        h = torch.bmm(attn, slice_feats)
        h_norm = self.mil_pool.norm(h)  # (B, 12, F)
        logits = torch.einsum('bkf,kf->bk', h_norm, self.mil_pool.classifier_w) + self.mil_pool.classifier_b
        return logits, h_norm, slice_feats


def load_extractor(backbone_name, ckpt_path, device):
    model = BaseMILFeatureExtractor(backbone_name)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    st = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(st)
    return model.to(device).eval().half()


def extract_embeddings_for_loader(loader, m_tiny, m_small, m_dino, device, desc=""):
    console.print(f"[cyan]Extracting representations ({desc})...[/cyan]")
    h_tiny_list, h_small_list, h_dino_list = [], [], []
    sl_tiny_list, sl_small_list, sl_dino_list = [], [], []
    l_tiny_list, l_small_list, l_dino_list = [], [], []
    hard_list, soft_list, is_gold_list, uids = [], [], [], []

    with torch.inference_mode():
        for batch in loader:
            vols = batch['images'].to(device).half()
            B, D, C, H, W = vols.shape
            vols_dino = F.interpolate(vols.view(B * D, C, H, W), size=(280, 280), mode='bilinear', align_corners=False).view(B, D, C, 280, 280)

            with torch.amp.autocast('cuda'):
                l_tiny, h_tiny, sl_tiny = m_tiny(vols)
                l_small, h_small, sl_small = m_small(vols)
                l_dino, h_dino, sl_dino = m_dino(vols_dino)

            h_tiny_list.append(h_tiny.cpu().half())
            h_small_list.append(h_small.cpu().half())
            h_dino_list.append(h_dino.cpu().half())

            sl_tiny_list.append(sl_tiny.cpu().half())
            sl_small_list.append(sl_small.cpu().half())
            sl_dino_list.append(sl_dino.cpu().half())

            l_tiny_list.append(l_tiny.cpu().float())
            l_small_list.append(l_small.cpu().float())
            l_dino_list.append(l_dino.cpu().float())

            hard_list.append(batch['hard_targets'].detach().cpu().clone())
            soft_list.append(batch['soft_targets'].detach().cpu().clone())
            is_gold_list.append(batch['is_gold'].detach().cpu().clone())
            uids.extend(batch.get('study_uids', batch.get('study_uid', [])))

    return {
        'h_tiny': torch.cat(h_tiny_list, dim=0),
        'h_small': torch.cat(h_small_list, dim=0),
        'h_dino': torch.cat(h_dino_list, dim=0),
        'sl_tiny': torch.cat(sl_tiny_list, dim=0),
        'sl_small': torch.cat(sl_small_list, dim=0),
        'sl_dino': torch.cat(sl_dino_list, dim=0),
        'l_tiny': torch.cat(l_tiny_list, dim=0),
        'l_small': torch.cat(l_small_list, dim=0),
        'l_dino': torch.cat(l_dino_list, dim=0),
        'hard': torch.cat(hard_list, dim=0),
        'soft': torch.cat(soft_list, dim=0),
        'is_gold': torch.cat(is_gold_list, dim=0),
        'uids': uids,
    }


def get_or_extract_features(fold, device, base_artifacts_dir='artifacts/experiments/phase_11_fusion_benchmarks'):
    data_dir = os.path.join(base_artifacts_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f'features_fold{fold}.pt')

    if os.path.exists(cache_path):
        console.print(f"[bold green]✓ Loaded cached features for Fold {fold} from {cache_path}[/bold green]")
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    console.print(f"[bold yellow]Extracting features for Fold {fold}...[/bold yellow]")
    ckpts = {
        'tiny': f'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_tiny_mri_aware_fold{fold}_best.pth',
        'small': f'kaggle_upload/datasets/phase_11_tri_ensemble/convnext_small_mri_aware_fold{fold}_best.pth',
        'dino': f'kaggle_upload/datasets/phase_11_tri_ensemble/dinov2_small_mri_aware_fold{fold}_best.pth',
    }

    # Use dataset splits & soft-hard master labels matching 27_train_mri_aware_mil.py
    df_splits = pd.read_parquet('data/splits_5fold.parquet')
    df_tri = pd.read_parquet('data/tristate_labels_master.parquet')
    df_dense = pd.read_parquet('data/dense_labels_master.parquet')

    df_all = df_splits[['StudyInstanceUID', 'fold']].copy()
    labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
    df_all['is_gold'] = labeled_mask

    tri_renamed = df_tri[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'hard_{c}' for c in TARGET_COLUMNS})
    dense_renamed = df_dense[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'soft_{c}' for c in TARGET_COLUMNS})
    df_all = df_all.merge(tri_renamed, on='StudyInstanceUID', how='left').merge(dense_renamed, on='StudyInstanceUID', how='left')

    train_df = df_all[df_all['fold'] != fold].reset_index(drop=True)
    val_df   = df_all[df_all['fold'] == fold].reset_index(drop=True)

    val_transforms = get_validation_transforms((288, 288))
    train_ds = MRIAwareKneeDataset(train_df, target_slices=32, transforms=val_transforms, is_training=False)
    val_ds   = MRIAwareKneeDataset(val_df, target_slices=32, transforms=val_transforms, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=mri_aware_collate)

    m_tiny = load_extractor('convnext_tiny', ckpts['tiny'], device)
    m_small = load_extractor('convnext_small', ckpts['small'], device)
    m_dino = load_extractor('vit_small_patch14_dinov2.lvd142m', ckpts['dino'], device)

    train_feats = extract_embeddings_for_loader(train_loader, m_tiny, m_small, m_dino, device, desc=f"Fold {fold} Train Set")
    val_feats   = extract_embeddings_for_loader(val_loader, m_tiny, m_small, m_dino, device, desc=f"Fold {fold} Val Set")

    del m_tiny, m_small, m_dino
    torch.cuda.empty_cache()

    feature_pack = {'train': train_feats, 'val': val_feats}
    torch.save(feature_pack, cache_path)
    console.print(f"[bold green]✓ Cached features saved to: {cache_path}[/bold green]")
    return feature_pack


def run_benchmark_for_fold(fold, device_str='cuda:0', epochs=20, lr=1e-3, weight_decay=1e-3, base_artifacts_dir='artifacts/experiments/phase_11_fusion_benchmarks'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    ckpt_dir = os.path.join(base_artifacts_dir, 'checkpoints')
    results_dir = os.path.join(base_artifacts_dir, 'results')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    console.print(f"\n[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]")
    console.print(f"[bold green]║ BENCHMARKING MULTI-BACKBONE ENSEMBLE FUSIONS (FOLD {fold})        ║[/bold green]")
    console.print(f"[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]")

    features = get_or_extract_features(fold, device, base_artifacts_dir=base_artifacts_dir)
    train_data, val_data = features['train'], features['val']

    # Ground truth targets matching 27_train_mri_aware_mil.py: continuous soft targets
    val_targets = val_data['soft'].numpy()
    val_is_gold = val_data['is_gold'].numpy().astype(bool)

    p_val_tiny  = torch.sigmoid(val_data['l_tiny']).numpy()
    p_val_small = torch.sigmoid(val_data['l_small']).numpy()
    p_val_dino  = torch.sigmoid(val_data['l_dino']).numpy()

    p_tr_tiny   = torch.sigmoid(train_data['l_tiny']).numpy()
    p_tr_small  = torch.sigmoid(train_data['l_small']).numpy()
    p_tr_dino   = torch.sigmoid(train_data['l_dino']).numpy()

    tr_hard = train_data['hard'].numpy()
    tr_soft = train_data['soft'].numpy()
    tr_is_gold = train_data['is_gold'].numpy().astype(bool)

    results = {}

    def record_metrics(strategy_name, val_probs):
        res = compute_macro_auc(val_targets, val_probs)
        overall_auc = res['macro_auc']
        gold_auc = None
        if val_is_gold.sum() > 5:
            gold_auc = compute_macro_auc(val_targets[val_is_gold], val_probs[val_is_gold])['macro_auc']
        results[strategy_name] = {
            'overall_auc': float(overall_auc),
            'gold_auc': float(gold_auc) if gold_auc is not None else 0.0,
            'per_class': {k: float(v) for k, v in res.get('per_class_auc', {}).items()}
        }
        return overall_auc

    # 1. Standalone Baselines
    record_metrics("ConvNeXt-Tiny Standalone", p_val_tiny)
    record_metrics("ConvNeXt-Small Standalone", p_val_small)
    record_metrics("DINOv2-Small Standalone", p_val_dino)

    # 2. Strategy 1: Uniform Mean Blend
    p_val_mean = (p_val_tiny + p_val_small + p_val_dino) / 3.0
    record_metrics("Uniform Mean Blend (1/3 each)", p_val_mean)

    # 3. Strategy 2: Calibrated Rank Averaging
    rank_module = CalibratedRankAveraging(num_models=3, num_classes=12).to(device)
    p_tr_tensors = [torch.tensor(p, dtype=torch.float32, device=device) for p in [p_tr_tiny, p_tr_small, p_tr_dino]]
    p_val_tensors = [torch.tensor(p, dtype=torch.float32, device=device) for p in [p_val_tiny, p_val_small, p_val_dino]]
    
    with torch.no_grad():
        p_val_rank = rank_module(p_val_tensors, use_ranks=True).cpu().numpy()
    record_metrics("Calibrated Rank Averaging", p_val_rank)

    # 4. Strategy 3: SLSQP Pathology-Specific Stacking
    optimal_weights = np.zeros((12, 3))
    p_val_stacking = np.zeros_like(p_val_tiny)

    for k in range(12):
        y_k_tr = tr_soft[:, k]
        w_k_gold = np.where(tr_is_gold, 2.0, 1.0)
        p_models_tr = np.column_stack([p_tr_tiny[:, k], p_tr_small[:, k], p_tr_dino[:, k]])
        p_models_val = np.column_stack([p_val_tiny[:, k], p_val_small[:, k], p_val_dino[:, k]])

        def loss_fn(w):
            p_blend = p_models_tr @ w
            p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
            bce = -(y_k_tr * np.log(p_blend) + (1 - y_k_tr) * np.log(1 - p_blend))
            return np.mean(bce * w_k_gold)

        init_w = np.array([0.5, 0.3, 0.2])
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        opt_res = minimize(loss_fn, init_w, method='SLSQP', bounds=bounds, constraints=cons)
        w_best = opt_res.x if opt_res.success else init_w
        optimal_weights[k] = w_best
        p_val_stacking[:, k] = p_models_val @ w_best

    record_metrics("SLSQP Stacking Meta-Learner", p_val_stacking)

    # 5. Shared Neural Training Engine for Strategies 4, 5, 6, 7, 8
    criterion = ConsensusDenoisedBCEWithLogitsLoss(gold_weight=2.0)

    # Input study feats: tiny (768), small (768), dino (384) -> float32 on device
    tr_feats_list = [train_data['h_tiny'].float(), train_data['h_small'].float(), train_data['h_dino'].float()]
    val_feats_list = [val_data['h_tiny'].float().to(device), val_data['h_small'].float().to(device), val_data['h_dino'].float().to(device)]

    tr_logits_list = [train_data['l_tiny'].float(), train_data['l_small'].float(), train_data['l_dino'].float()]
    val_logits_list = [val_data['l_tiny'].float().to(device), val_data['l_small'].float().to(device), val_data['l_dino'].float().to(device)]

    # Slice feats for ABMIL: (B, D, F)
    tr_slice_dict = {
        'tiny': train_data['sl_tiny'].float(),
        'small': train_data['sl_small'].float(),
        'dino': train_data['sl_dino'].float(),
    }
    val_slice_dict = {
        'convnext_tiny': val_data['sl_tiny'].float().to(device),
        'convnext_small': val_data['sl_small'].float().to(device),
        'dinov2': val_data['sl_dino'].float().to(device),
    }

    tr_dataset = TensorDataset(
        tr_feats_list[0], tr_feats_list[1], tr_feats_list[2],
        tr_logits_list[0], tr_logits_list[1], tr_logits_list[2],
        tr_slice_dict['tiny'], tr_slice_dict['small'], tr_slice_dict['dino'],
        train_data['hard'], train_data['soft'], train_data['is_gold']
    )
    tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True)

    def train_neural_module(model, forward_fn, name, save_key):
        console.print(f"[yellow]Training {name} on Fold {fold}...[/yellow]")
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        best_auc = 0.0
        best_probs = None
        best_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            for b_t, b_s, b_d, l_t, l_s, l_d, sl_t, sl_s, sl_d, hard_b, soft_b, is_gold_b in tr_loader:
                b_t, b_s, b_d = b_t.to(device), b_s.to(device), b_d.to(device)
                l_t, l_s, l_d = l_t.to(device), l_s.to(device), l_d.to(device)
                sl_t, sl_s, sl_d = sl_t.to(device), sl_s.to(device), sl_d.to(device)
                hard_b, soft_b, is_gold_b = hard_b.to(device), soft_b.to(device), is_gold_b.to(device)

                opt.zero_grad()
                logits = forward_fn(model, [b_t, b_s, b_d], [l_t, l_s, l_d], {'convnext_tiny': sl_t, 'convnext_small': sl_s, 'dinov2': sl_d})
                loss = criterion(logits, hard_b, soft_b, is_gold_b)
                loss.backward()
                opt.step()

            sched.step()

            model.eval()
            with torch.no_grad():
                val_out_logits = forward_fn(model, val_feats_list, val_logits_list, val_slice_dict)
                val_probs = torch.sigmoid(val_out_logits).cpu().numpy()
            
            curr_auc = compute_macro_auc(val_targets, val_probs)['macro_auc']
            if curr_auc > best_auc:
                best_auc = curr_auc
                best_probs = val_probs
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        torch.save(best_state, os.path.join(ckpt_dir, f'{save_key}_fold{fold}_best.pth'))
        record_metrics(name, best_probs)
        return best_auc

    # Strategy 4: Bottleneck Feature Fusion
    bnf = BottleneckFeatureFusion(in_dims=[768, 768, 384], num_classes=12, bottleneck_dim=256, dropout=0.35)
    train_neural_module(bnf, lambda m, f, l, s: m(f, model_standalone_logits=l), "Bottleneck Feature Fusion", "bottleneck_fusion")

    # Strategy 5: Residual GatedFusion
    rgf = ResidualGatedFusion(in_dims=[768, 768, 384], num_classes=12, hidden_dim=128, dropout=0.25)
    train_neural_module(rgf, lambda m, f, l, s: m(f)[0], "Residual GatedFusion", "residual_gated")

    # Strategy 6: Pathology-Grouped Soft-MoE
    moe = PathologyGroupedSoftMoE(in_dims=[768, 768, 384], num_classes=12, expert_dim=128, dropout=0.25)
    train_neural_module(moe, lambda m, f, l, s: m(f)[0], "Pathology-Grouped Soft-MoE", "grouped_moe")

    # Strategy 7: Dual-Axis Label-Specific ABMIL
    abmil = DualAxisLabelSpecificABMIL(
        in_features_dict={'convnext_tiny': 768, 'convnext_small': 768, 'dinov2': 384},
        num_classes=12, mil_hidden=128, cross_hidden=64, dropout=0.25
    )
    train_neural_module(abmil, lambda m, f, l, s: m(s)[0], "Dual-Axis Label-Specific ABMIL", "dual_axis_abmil")

    # Strategy 8: Pathology Slot Attention
    psa = PathologySlotAttention(in_features_list=[768, 768, 384], slot_dim=128, num_slots=12, iters=3)
    train_neural_module(psa, lambda m, f, l, s: m(f)[0], "Pathology Slot Attention", "slot_attention")

    # Build and print comparison table
    base_tiny = results["ConvNeXt-Tiny Standalone"]["overall_auc"]
    base_mean = results["Uniform Mean Blend (1/3 each)"]["overall_auc"]

    table = Table(title=f"Multi-Backbone Ensemble & Fusion Benchmark - Fold {fold}")
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Val Macro AUC", style="bold green")
    table.add_column("Gold Consensus AUC", style="bold magenta")
    table.add_column("Delta vs Tiny", style="yellow")
    table.add_column("Delta vs Mean Blend", style="blue")

    for strat, data in sorted(results.items(), key=lambda x: x[1]['overall_auc'], reverse=True):
        auc = data['overall_auc']
        g_auc = data['gold_auc']
        d_tiny = auc - base_tiny
        d_mean = auc - base_mean
        table.add_row(strat, f"{auc:.4f}", f"{g_auc:.4f}", f"{d_tiny:+.4f}", f"{d_mean:+.4f}")

    console.print(table)

    # Save to json inside artifacts
    out_json = os.path.join(results_dir, f'benchmark_results_fold{fold}.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved benchmark results to: {out_json}[/green]")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--base_artifacts_dir', type=str, default='artifacts/experiments/phase_11_fusion_benchmarks')
    args = parser.parse_args()

    run_benchmark_for_fold(args.fold, args.device, args.epochs, args.lr, args.weight_decay, args.base_artifacts_dir)

if __name__ == '__main__':
    main()
