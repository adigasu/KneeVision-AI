import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from src.metrics.auc_metrics import compute_macro_auc, TARGET_COLUMNS

# 1. Load dense consensus targets
df_splits = pd.read_parquet('data/splits_5fold.parquet')
df_dense = pd.read_parquet('data/dense_labels_master.parquet')

# Merge
df_all = df_splits[['StudyInstanceUID', 'fold']].copy()
labeled_mask = df_splits[TARGET_COLUMNS].notnull().any(axis=1)
df_all['is_gold'] = labeled_mask

dense_renamed = df_dense[['StudyInstanceUID'] + TARGET_COLUMNS].rename(columns={c: f'soft_{c}' for c in TARGET_COLUMNS})
df_all = df_all.merge(dense_renamed, on='StudyInstanceUID', how='left')

val_df = df_all[df_all['fold'] == 0].sort_values('StudyInstanceUID').reset_index(drop=True)
print(f"Loaded {len(val_df)} validation cases for Fold 0 (Gold: {val_df['is_gold'].sum()}).")

oof_paths = {
    'ConvNeXt-Tiny': 'artifacts/experiments/phase_11_convnext_tiny/robust_consensus_denoised/oof_fold0.parquet',
    'ConvNeXt-Small': 'artifacts/experiments/phase_11_convnext_small/robust_consensus_denoised/oof_fold0.parquet',
    'DINOv2-Small': 'artifacts/experiments/phase_11_dinov2_small/robust_consensus_denoised/oof_fold0.parquet',
    'DINOv3-Small': 'artifacts/experiments/phase_12_dinov3_small/robust_consensus_denoised/oof_fold0.parquet'
}

dfs = {}
for name, p in oof_paths.items():
    if not os.path.exists(p):
        print(f"Error: {p} does not exist.")
        exit(1)
    df = pd.read_parquet(p)
    merged = val_df[['StudyInstanceUID']].merge(df, on='StudyInstanceUID', how='left')
    dfs[name] = merged

pred_cols = [f'pred_{c}' for c in TARGET_COLUMNS]
preds = {name: df[pred_cols].values for name, df in dfs.items()}

# Extract target matrix
targets = np.zeros((len(val_df), len(TARGET_COLUMNS)))
for j, c in enumerate(TARGET_COLUMNS):
    targets[:, j] = val_df[f'soft_{c}'].values

is_gold = val_df['is_gold'].values

def evaluate_predictions(p):
    full_m = compute_macro_auc(targets, p)
    gold_m = compute_macro_auc(targets[is_gold], p[is_gold]) if is_gold.sum() > 5 else {'macro_auc': 0.0}
    return full_m['macro_auc'], gold_m['macro_auc'], full_m['per_class_auc']

print("\n" + "="*70)
print(f"{'ARCHITECTURE / ENSEMBLE':<32} | {'FULL VAL AUC':<14} | {'GOLD AUC':<10}")
print("="*70)

per_class_all = {}
for name in oof_paths.keys():
    full_auc, gold_auc, per_class = evaluate_predictions(preds[name])
    per_class_all[name] = per_class
    print(f"{name:<32} | {full_auc:.4f}         | {gold_auc:.4f}")

# 1. 3-Model Blend (Tiny + Small + DINOv2) Equal
p_3m = (preds['ConvNeXt-Tiny'] + preds['ConvNeXt-Small'] + preds['DINOv2-Small']) / 3.0
auc_3m, gold_3m, per_class_3m = evaluate_predictions(p_3m)
print("-" * 70)
print(f"{'3-Model Blend (Equal)':<32} | {auc_3m:.4f}         | {gold_3m:.4f}")

# 2. 4-Model Blend (Tiny + Small + DINOv2 + DINOv3) Equal
p_4m_eq = (preds['ConvNeXt-Tiny'] + preds['ConvNeXt-Small'] + preds['DINOv2-Small'] + preds['DINOv3-Small']) / 4.0
auc_4m_eq, gold_4m_eq, per_class_4m_eq = evaluate_predictions(p_4m_eq)
print(f"{'4-Model Blend (Equal)':<32} | {auc_4m_eq:.4f}         | {gold_4m_eq:.4f}")

# 3. Optimized Weighted 4-Model Blend
def loss_func(w):
    w = np.maximum(w, 0)
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    p_blend = (
        w[0] * preds['ConvNeXt-Tiny'] +
        w[1] * preds['ConvNeXt-Small'] +
        w[2] * preds['DINOv2-Small'] +
        w[3] * preds['DINOv3-Small']
    )
    score = compute_macro_auc(targets, p_blend)['macro_auc']
    return -score

init_w = [0.35, 0.25, 0.25, 0.15]
res = minimize(loss_func, init_w, method='Nelder-Mead', options={'maxiter': 500})
opt_w = np.maximum(res.x, 0)
opt_w = opt_w / opt_w.sum()

p_4m_opt = (
    opt_w[0] * preds['ConvNeXt-Tiny'] +
    opt_w[1] * preds['ConvNeXt-Small'] +
    opt_w[2] * preds['DINOv2-Small'] +
    opt_w[3] * preds['DINOv3-Small']
)
auc_4m_opt, gold_4m_opt, per_class_4m_opt = evaluate_predictions(p_4m_opt)
print(f"{'4-Model Blend (Optimized)':<32} | {auc_4m_opt:.4f}         | {gold_4m_opt:.4f}")
print(f"Optimal Weights: Tiny={opt_w[0]:.3f}, Small={opt_w[1]:.3f}, DINOv2={opt_w[2]:.3f}, DINOv3={opt_w[3]:.3f}")
print("="*70)

# Per-target breakdown
print("\n" + "="*80)
print(f"{'PATHOLOGY TARGET':<25} | {'3-MODEL':<10} | {'4-MODEL OPT':<14} | {'DELTA':<10}")
print("="*80)
for col in TARGET_COLUMNS:
    a3 = per_class_3m.get(col, 0.0)
    a4 = per_class_4m_opt.get(col, 0.0)
    delta = a4 - a3
    sign = "+" if delta >= 0 else ""
    print(f"{col:<25} | {a3:.4f}     | {a4:.4f}         | {sign}{delta:.4f}")
print("="*80)
