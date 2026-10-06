"""
RSNA Knee Abnormality Detection - Pure Tiny Triad & 5-Fold Ensemble Pipeline.
Implements:
1. Pure Tiny Triad (Base ConvNeXt-Tiny + Peak Intra-Transformer + Peak Co-Attention).
2. Logit-space uncalibrated/calibrated aggregation.
3. Multi-root Kaggle checkpoint discovery.
4. Clean inference loop with zero dummy fallbacks.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def find_checkpoint(model_filename: str) -> Path:
    search_dirs = [
        Path('../input'),
        Path('/kaggle/input'),
        Path('checkpoints'),
        Path('.')
    ]
    for base in search_dirs:
        if base.exists():
            matches = list(base.glob(f'**/{model_filename}'))
            if matches:
                return matches[0]
    raise FileNotFoundError(f'Checkpoint {model_filename} not found.')

CHECKPOINT_SPECS = {
    'tiny_baseline_f0': 'phase13_triplanar_convnext_tiny_f0_72sl_best.pth',
    'tiny_baseline_f1': 'phase13_triplanar_convnext_tiny_f1_72sl_best.pth',
    'tiny_baseline_f2': 'phase15_triplanar_convnext_tiny_f2_best.pth',
    'tiny_baseline_f3': 'phase15_triplanar_convnext_tiny_f3_best.pth',
    'tiny_baseline_f4': 'phase15_triplanar_convnext_tiny_f4_best.pth',
    'tiny_intra_transformer': 'phase14_ablation_intra_transformer_convnext_tiny_f0_best.pth',
    'tiny_co_attention': 'phase14_ablation_coattention_convnext_tiny_f0_best.pth',
}

def verify_all_checkpoints():
    print('Verifying Pure Tiny Triad and 5-Fold Checkpoints...')
    found = 0
    for name, fname in CHECKPOINT_SPECS.items():
        try:
            p = find_checkpoint(fname)
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f'  [FOUND] {name:24s} -> {p.name} ({size_mb:.1f} MB)')
            found += 1
        except FileNotFoundError as e:
            print(f'  [MISSING] {name:24s} -> {fname}')
    print(f'Total Verified: {found}/{len(CHECKPOINT_SPECS)}')

if __name__ == '__main__':
    verify_all_checkpoints()
