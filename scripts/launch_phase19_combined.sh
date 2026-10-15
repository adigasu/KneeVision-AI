#!/usr/bin/env bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Phase 19: Combined Asymmetric Slices + EMA
# (Combines 32 Sag + 28 Cor + 20 Ax = 80 Slices with EMA beta=0.999)
#
# Usage:
#   1. Single Fold on specific GPU:
#      ./scripts/launch_phase19_combined.sh --fold 0 --gpu 0
#      ./scripts/launch_phase19_combined.sh --fold 1 --gpu 1
#      ./scripts/launch_phase19_combined.sh --fold 3 --gpu 2
#
#   2. Run All Top-3 Folds (0, 1, 3) in Parallel across 3 GPUs (0, 1, 2):
#      ./scripts/launch_phase19_combined.sh --all-top3
# ==============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"

IMAGE_SIZE=384
SLICES_SAG=32
SLICES_COR=28
SLICES_AX=20
EPOCHS=5
WARMUP_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=8
LR=5e-5
BACKBONE_LR_MULT=0.1
LABEL_SMOOTHING=0.1
GOLD_WEIGHT=2.0
EMA_DECAY=0.999
OUTPUT_DIR="checkpoints/phase19_combined"
mkdir -p "$OUTPUT_DIR" logs

if [ "${1:-}" == "--all-top3" ]; then
    echo "=================================================================="
    echo " Launching Phase 19 Combined (Asymmetric Slices + EMA) across Top-3 Folds"
    echo " Slices: 32 Sag + 28 Cor + 20 Ax = 80 Slices | EMA: $EMA_DECAY"
    echo " Warm-starting from: checkpoints/consensus_ft/ (Phase 18 weights)"
    echo " GPUs: Fold 0 -> GPU 0 | Fold 1 -> GPU 1 | Fold 3 -> GPU 2"
    echo "=================================================================="

    $PYTHON -u scripts/62_finetune_5fold_consensus.py \
      --device cuda:0 --fold 0 --init_from_consensus 1 --image_size $IMAGE_SIZE \
      --slices_sag $SLICES_SAG --slices_cor $SLICES_COR --slices_ax $SLICES_AX \
      --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS \
      --batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM \
      --lr $LR --backbone_lr_mult $BACKBONE_LR_MULT \
      --label_smoothing $LABEL_SMOOTHING --gold_weight $GOLD_WEIGHT \
      --output_dir $OUTPUT_DIR --use_ema 1 --ema_decay $EMA_DECAY \
      > logs/train_comb_f0_gpu0.log 2>&1 &
    PID_F0=$!
    echo "  ✓ Fold 0 running on GPU 0 (PID: $PID_F0) -> logs/train_comb_f0_gpu0.log"

    $PYTHON -u scripts/62_finetune_5fold_consensus.py \
      --device cuda:1 --fold 1 --init_from_consensus 1 --image_size $IMAGE_SIZE \
      --slices_sag $SLICES_SAG --slices_cor $SLICES_COR --slices_ax $SLICES_AX \
      --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS \
      --batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM \
      --lr $LR --backbone_lr_mult $BACKBONE_LR_MULT \
      --label_smoothing $LABEL_SMOOTHING --gold_weight $GOLD_WEIGHT \
      --output_dir $OUTPUT_DIR --use_ema 1 --ema_decay $EMA_DECAY \
      > logs/train_comb_f1_gpu1.log 2>&1 &
    PID_F1=$!
    echo "  ✓ Fold 1 running on GPU 1 (PID: $PID_F1) -> logs/train_comb_f1_gpu1.log"

    $PYTHON -u scripts/62_finetune_5fold_consensus.py \
      --device cuda:2 --fold 3 --init_from_consensus 1 --image_size $IMAGE_SIZE \
      --slices_sag $SLICES_SAG --slices_cor $SLICES_COR --slices_ax $SLICES_AX \
      --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS \
      --batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM \
      --lr $LR --backbone_lr_mult $BACKBONE_LR_MULT \
      --label_smoothing $LABEL_SMOOTHING --gold_weight $GOLD_WEIGHT \
      --output_dir $OUTPUT_DIR --use_ema 1 --ema_decay $EMA_DECAY \
      > logs/train_comb_f3_gpu2.log 2>&1 &
    PID_F3=$!
    echo "  ✓ Fold 3 running on GPU 2 (PID: $PID_F3) -> logs/train_comb_f3_gpu2.log"

    echo ""
    echo "All 3 Combined folds running in background! Follow progress with:"
    echo "  tail -f logs/train_comb_f0_gpu0.log"
    echo "  tail -f logs/train_comb_f1_gpu1.log"
    echo "  tail -f logs/train_comb_f3_gpu2.log"
    exit 0
fi

FOLD=0
GPU=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --fold) FOLD="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=================================================================="
echo " Launching Phase 19 Combined (Fold $FOLD on GPU $GPU)"
echo " Slices: 32 Sag + 28 Cor + 20 Ax = 80 Slices | EMA: $EMA_DECAY"
echo " Warm-starting from: checkpoints/consensus_ft/ (Phase 18 weights)"
echo "=================================================================="

$PYTHON -u scripts/62_finetune_5fold_consensus.py \
  --device "cuda:$GPU" \
  --fold "$FOLD" \
  --init_from_consensus 1 \
  --image_size $IMAGE_SIZE \
  --slices_sag $SLICES_SAG \
  --slices_cor $SLICES_COR \
  --slices_ax $SLICES_AX \
  --epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --backbone_lr_mult $BACKBONE_LR_MULT \
  --label_smoothing $LABEL_SMOOTHING \
  --gold_weight $GOLD_WEIGHT \
  --output_dir "$OUTPUT_DIR" \
  --use_ema 1 \
  --ema_decay $EMA_DECAY
