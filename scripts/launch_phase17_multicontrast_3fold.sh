#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Phase 17: Multi-Contrast 3-Fold Training
# Trains Decoupled models on 3-Fold CV (Folds 0, 1, 2) across GPU 0 & GPU 1.
# ==============================================================================

set -euo pipefail

PYTHON="python"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}" checkpoints

MODE="${1:-t1_only}" # choices: t1_only, t2_only, dual
BACKBONE="${2:-convnext_small}"
IMG_SIZE="${3:-384}"
EPOCHS="${4:-15}"

echo "========================================================================"
echo " Launching Phase 17 Multi-Contrast 3-Fold Decoupled Training"
echo " Contrast Mode:  ${MODE}"
echo " Backbone:       ${BACKBONE}"
echo " Image Size:     ${IMG_SIZE}px"
echo " Epochs:         ${EPOCHS}"
echo " Splits:         data/splits_3fold.parquet"
echo "========================================================================"

# Launch Fold 0 on GPU 0
LOG_F0="${LOG_DIR}/train_phase17_${MODE}_${BACKBONE}_${IMG_SIZE}px_f0_gpu0.log"
echo "Starting Fold 0 on GPU 0 -> Logging to ${LOG_F0}..."
nohup ${PYTHON} -u scripts/53_train_decoupled_multicontrast_3fold.py \
    --device cuda:0 \
    --backbone "${BACKBONE}" \
    --contrast_mode "${MODE}" \
    --fold 0 \
    --image_size "${IMG_SIZE}" \
    --slices_per_plane 24 \
    --epochs "${EPOCHS}" \
    --warmup_epochs 2 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 3e-4 \
    --backbone_lr_mult 0.1 \
    --gold_weight 2.0 \
    --classifier_type linear \
    --run_name "phase17_${MODE}_${BACKBONE}_${IMG_SIZE}px_f0" \
    > "${LOG_F0}" 2>&1 &
PID_F0=$!
echo "Fold 0 running with PID: ${PID_F0}"

# Launch Fold 1 on GPU 1
LOG_F1="${LOG_DIR}/train_phase17_${MODE}_${BACKBONE}_${IMG_SIZE}px_f1_gpu1.log"
echo "Starting Fold 1 on GPU 1 -> Logging to ${LOG_F1}..."
nohup ${PYTHON} -u scripts/53_train_decoupled_multicontrast_3fold.py \
    --device cuda:1 \
    --backbone "${BACKBONE}" \
    --contrast_mode "${MODE}" \
    --fold 1 \
    --image_size "${IMG_SIZE}" \
    --slices_per_plane 24 \
    --epochs "${EPOCHS}" \
    --warmup_epochs 2 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 3e-4 \
    --backbone_lr_mult 0.1 \
    --gold_weight 2.0 \
    --classifier_type linear \
    --run_name "phase17_${MODE}_${BACKBONE}_${IMG_SIZE}px_f1" \
    > "${LOG_F1}" 2>&1 &
PID_F1=$!
echo "Fold 1 running with PID: ${PID_F1}"

echo "Both Folds 0 & 1 launched successfully on GPU 0 & GPU 1!"
echo "Monitor with: tail -f ${LOG_F0} or tail -f ${LOG_F1}"
