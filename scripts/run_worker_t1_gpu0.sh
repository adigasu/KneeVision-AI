#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Phase 17: T1 Anatomical 3-Fold Stream (cuda:0)
# Sequential training of Folds 0, 1, 2 on RTX A6000 GPU 0
# ==============================================================================
set -euo pipefail
PYTHON="/home/AQ44130/miniconda3/envs/rsna-knee/bin/python"

for FOLD in 0 1 2; do
    CKPT="checkpoints/phase17_decoupled_t1_convnext_small_384px_f${FOLD}_best.pth"
    PREDS="checkpoints/phase17_decoupled_t1_convnext_small_384px_f${FOLD}_val_preds.npy"

    if [ -f "${CKPT}" ] && [ -f "${PREDS}" ]; then
        continue
    fi

    echo "============================================================"
    echo "Starting T1 Anatomical Training - Fold ${FOLD} on cuda:0"
    echo "Date: $(date)"
    echo "============================================================"
    
    ${PYTHON} -u scripts/53_train_decoupled_multicontrast_3fold.py         --device cuda:0         --backbone convnext_small         --contrast_mode t1_only         --splits_path data/splits_3fold.parquet         --fold ${FOLD}         --image_size 384         --slices_per_plane 24         --epochs 12         --warmup_epochs 2         --batch_size 2         --grad_accum_steps 8         --lr 3e-4         --backbone_lr_mult 0.1         --gold_weight 2.0         --classifier_type linear         --run_name "phase17_decoupled_t1_convnext_small_384px_f${FOLD}"
        
    echo "Completed T1 Fold ${FOLD} at $(date)"
    ${PYTHON} scripts/evaluate_phase17_6model_ensemble.py || true
done
