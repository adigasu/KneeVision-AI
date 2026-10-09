"""
RSNA Knee Abnormality Detection - Evaluate Multi-Contrast 3-Fold 6-Model Ensemble.
Evaluates:
- 3 Folds of T1 Anatomical Decoupled ConvNeXt-Small 384px
- 3 Folds of T2 FS Decoupled ConvNeXt-Small 384px
Computes:
1. Per-fold T1, T2, and Blended Macro AUC
2. Full Out-Of-Fold (OOF) Macro AUC across all 4,407 studies
3. Gold consensus subset Macro AUC across all 58 gold studies
4. Solves optimal per-pathology contrast ensembling weights
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

from src.metrics.auc_metrics import TARGET_COLUMNS, compute_macro_auc


def evaluate_3fold_ensemble(splits_path='data/splits_3fold.parquet', output_json='checkpoints/phase17_3fold_ensemble_summary.json'):
    console = Console()
    console.print('[bold blue]=== RSNA Knee: Phase 17 Multi-Contrast 3-Fold 6-Model Ensemble Evaluation ===[/bold blue]')

    df_splits = pd.read_parquet(splits_path)
    df_dense = pd.read_parquet('data/dense_labels_master.parquet')
    dense_map = df_dense.set_index('StudyInstanceUID')[TARGET_COLUMNS]

    t1_pattern = 'checkpoints/phase17_decoupled_t1_convnext_small_384px_f{fold}_val_preds.npy'
    t2_pattern = 'checkpoints/phase17_decoupled_t2_convnext_small_384px_f{fold}_val_preds.npy'

    all_t1_preds = {}
    all_t2_preds = {}
    all_targets = {}
    all_is_gold = {}

    table_folds = Table(title='Fold-by-Fold Performance Summary')
    table_folds.add_column('Fold', style='cyan')
    table_folds.add_column('Studies', justify='right')
    table_folds.add_column('T1 Macro AUC', justify='right', style='magenta')
    table_folds.add_column('T2 Macro AUC', justify='right', style='green')
    table_folds.add_column('50/50 Blend', justify='right', style='yellow')
    table_folds.add_column('Optimal Blend', justify='right', style='bold green')
    table_folds.add_column('Δ vs T2', justify='right', style='bold cyan')

    completed_folds = []

    for f in [0, 1, 2]:
        p_t1_file = t1_pattern.format(fold=f)
        p_t2_file = t2_pattern.format(fold=f)

        fold_df = df_splits[df_splits['fold'] == f].reset_index(drop=True)
        y_fold = dense_map.loc[fold_df['StudyInstanceUID']].fillna(0.50).values.astype(float)
        is_gold_fold = fold_df[TARGET_COLUMNS].notnull().any(axis=1).values

        has_t1 = os.path.exists(p_t1_file)
        has_t2 = os.path.exists(p_t2_file)

        if has_t1 and has_t2:
            preds_t1 = np.load(p_t1_file)
            preds_t2 = np.load(p_t2_file)

            m_t1 = compute_macro_auc(y_fold, preds_t1)['macro_auc']
            m_t2 = compute_macro_auc(y_fold, preds_t2)['macro_auc']
            preds_uni = 0.5 * preds_t1 + 0.5 * preds_t2
            m_uni = compute_macro_auc(y_fold, preds_uni)['macro_auc']

            # Per-pathology optimal search for fold
            preds_opt = np.zeros_like(preds_t2)
            for c_idx in range(len(TARGET_COLUMNS)):
                y_c = (y_fold[:, c_idx] >= 0.5).astype(int)
                best_w = 0.5
                best_auc = 0.0
                for w in np.linspace(0.0, 1.0, 101):
                    cand = w * preds_t2[:, c_idx] + (1 - w) * preds_t1[:, c_idx]
                    try:
                        auc = roc_auc_score(y_c, cand) if len(np.unique(y_c)) > 1 else 0.5
                    except:
                        auc = 0.5
                    if auc > best_auc:
                        best_auc = auc
                        best_w = w
                preds_opt[:, c_idx] = best_w * preds_t2[:, c_idx] + (1 - best_w) * preds_t1[:, c_idx]

            m_opt = compute_macro_auc(y_fold, preds_opt)['macro_auc']
            delta = m_opt - m_t2

            table_folds.add_row(
                f'Fold {f}',
                f'{len(fold_df):,}',
                f'{m_t1:.4f}',
                f'{m_t2:.4f}',
                f'{m_uni:.4f}',
                f'{m_opt:.4f}',
                f'[bold green]+{delta:.4f}[/bold green]' if delta > 0 else f'{delta:.4f}'
            )

            all_t1_preds[f] = preds_t1
            all_t2_preds[f] = preds_t2
            all_targets[f] = y_fold
            all_is_gold[f] = is_gold_fold
            completed_folds.append(f)
        else:
            status = []
            if not has_t1: status.append('T1 pending')
            if not has_t2: status.append('T2 pending')
            table_folds.add_row(f'Fold {f}', f'{len(fold_df):,}', f'[{status[0]}]', '', '', '', '')

    console.print(table_folds)

    if len(completed_folds) == 0:
        console.print('[yellow]No completed fold pairs found yet.[/yellow]')
        return

    # If any folds completed, evaluate pooled OOF
    oof_t1 = np.concatenate([all_t1_preds[f] for f in completed_folds], axis=0)
    oof_t2 = np.concatenate([all_t2_preds[f] for f in completed_folds], axis=0)
    oof_y = np.concatenate([all_targets[f] for f in completed_folds], axis=0)
    oof_gold = np.concatenate([all_is_gold[f] for f in completed_folds], axis=0)

    oof_m_t1 = compute_macro_auc(oof_y, oof_t1)
    oof_m_t2 = compute_macro_auc(oof_y, oof_t2)
    oof_uni = 0.5 * oof_t1 + 0.5 * oof_t2
    oof_m_uni = compute_macro_auc(oof_y, oof_uni)

    # Solve global pathology weights
    global_weights = {}
    oof_opt = np.zeros_like(oof_t2)

    for c_idx, col in enumerate(TARGET_COLUMNS):
        y_c = (oof_y[:, c_idx] >= 0.5).astype(int)
        best_w = 0.5
        best_auc = 0.0
        for w in np.linspace(0.0, 1.0, 101):
            cand = w * oof_t2[:, c_idx] + (1 - w) * oof_t1[:, c_idx]
            try:
                auc = roc_auc_score(y_c, cand) if len(np.unique(y_c)) > 1 else 0.5
            except:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_w = w
        global_weights[col] = float(round(best_w, 3))
        oof_opt[:, c_idx] = best_w * oof_t2[:, c_idx] + (1 - best_w) * oof_t1[:, c_idx]

    oof_m_opt = compute_macro_auc(oof_y, oof_opt)

    gold_t1 = compute_macro_auc(oof_y[oof_gold], oof_t1[oof_gold])['macro_auc'] if oof_gold.sum() > 3 else 0.0
    gold_t2 = compute_macro_auc(oof_y[oof_gold], oof_t2[oof_gold])['macro_auc'] if oof_gold.sum() > 3 else 0.0
    gold_uni = compute_macro_auc(oof_y[oof_gold], oof_uni[oof_gold])['macro_auc'] if oof_gold.sum() > 3 else 0.0
    gold_opt = compute_macro_auc(oof_y[oof_gold], oof_opt[oof_gold])['macro_auc'] if oof_gold.sum() > 3 else 0.0

    table_oof = Table(title=f'Out-Of-Fold Pathology Breakdown (Folds: {completed_folds})')
    table_oof.add_column('Pathology', style='cyan')
    table_oof.add_column('T2 FS Alone', justify='right', style='green')
    table_oof.add_column('T1 Alone', justify='right', style='magenta')
    table_oof.add_column('50/50 Blend', justify='right', style='yellow')
    table_oof.add_column('Optimal Blend', justify='right', style='bold green')
    table_oof.add_column('Δ vs T2', justify='right', style='bold cyan')
    table_oof.add_column('Weight (T2 : T1)', justify='center')

    for col in TARGET_COLUMNS:
        auc_t2 = oof_m_t2['per_class_auc'].get(col, 0.0)
        auc_t1 = oof_m_t1['per_class_auc'].get(col, 0.0)
        auc_uni = oof_m_uni['per_class_auc'].get(col, 0.0)
        auc_opt = oof_m_opt['per_class_auc'].get(col, 0.0)
        w_t2 = global_weights[col]
        delta = auc_opt - auc_t2
        table_oof.add_row(
            col,
            f'{auc_t2:.4f}',
            f'{auc_t1:.4f}',
            f'{auc_uni:.4f}',
            f'{auc_opt:.4f}',
            f'[bold green]+{delta:.4f}[/bold green]' if delta > 0 else f'{delta:.4f}',
            f'{w_t2:.2f} : {1.0 - w_t2:.2f}'
        )

    console.print(table_oof)

    console.print()
    console.print(f"[bold]Overall Out-Of-Fold Summary ({len(oof_y):,} studies):[/bold]")
    console.print(f'  • T2 FS Alone Macro AUC:        {oof_m_t2["macro_auc"]:.4f} (Gold: {gold_t2:.4f})')
    console.print(f'  • T1 Anatomical Alone Macro AUC: {oof_m_t1["macro_auc"]:.4f} (Gold: {gold_t1:.4f})')
    console.print(f'  • 50/50 Uniform Blend Macro AUC: {oof_m_uni["macro_auc"]:.4f} (Gold: {gold_uni:.4f}) [bold green](+{oof_m_uni["macro_auc"] - oof_m_t2["macro_auc"]:+.4f})[/bold green]')
    console.print(f'  • [bold green]Pathology-Optimal Blend Macro AUC:[/bold green] [bold green]{oof_m_opt["macro_auc"]:.4f}[/bold green] (Gold: [bold magenta]{gold_opt:.4f}[/bold magenta]) [bold green](+{oof_m_opt["macro_auc"] - oof_m_t2["macro_auc"]:+.4f})[/bold green]')

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump({
            'completed_folds': completed_folds,
            'total_studies': len(oof_y),
            'gold_studies': int(oof_gold.sum()),
            't2_macro_auc': float(oof_m_t2['macro_auc']),
            't1_macro_auc': float(oof_m_t1['macro_auc']),
            'uniform_macro_auc': float(oof_m_uni['macro_auc']),
            'optimal_macro_auc': float(oof_m_opt['macro_auc']),
            'weights': global_weights,
        }, f, indent=2)
    console.print(f'Saved ensemble summary to: [green]{output_json}[/green]')


if __name__ == '__main__':
    evaluate_3fold_ensemble()
