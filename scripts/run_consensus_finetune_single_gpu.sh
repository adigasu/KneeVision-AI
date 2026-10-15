#!/usr/bin/env bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Phase 18: Consensus Finetuning (Single GPU)
# Runs 5-Fold ConvNeXt-Small (384px) consensus finetuning on a designated GPU.
#
# Usage:
#   bash scripts/run_consensus_finetune_single_gpu.sh <GPU_ID> [FOLD]
#
# Examples:
#   # Run all 5 folds (0 -> 4) sequentially on GPU 1:
#   bash scripts/run_consensus_finetune_single_gpu.sh 1
#
#   # Run only Fold 0 on GPU 2:
#   bash scripts/run_consensus_finetune_single_gpu.sh 2 0
# ==============================================================================

set -euo pipefail

GPU_ID="${1:-0}"
SPECIFIC_FOLD="${2:-all}"

PYTHON="${PYTHON:-python}"

echo "=================================================================="
echo " Consensus Finetuning (ConvNeXt-Small 384px)"
echo " Designated GPU: cuda:${GPU_ID}"
echo " Fold(s): ${SPECIFIC_FOLD}"
echo " Python: ${PYTHON}"
echo " Start Time: $(date)"
echo "=================================================================="

# 1. Safety Backup of Phase 16 Checkpoints
echo ""
echo "[Step 1] Verifying / Creating Safety Backup of Phase 16 weights..."
mkdir -p checkpoints/backup_phase16
mkdir -p checkpoints/consensus_ft
mkdir -p logs

for f in 0 1 2 3 4; do
    SRC="checkpoints/phase16_decoupled_convnext_small_384px_f${f}_best.pth"
    DST="checkpoints/backup_phase16/phase16_decoupled_convnext_small_384px_f${f}_best.pth"
    if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
        cp -v "$SRC" "$DST"
    fi
done
echo "✓ Original Phase 16 checkpoints are safe!"

# 2. Hyperparameters
IMAGE_SIZE=384
SLICES_PER_PLANE=24
EPOCHS=6
WARMUP_EPOCHS=1
BATCH_SIZE=4
GRAD_ACCUM=4
LR=1e-4
BACKBONE_LR_MULT=0.1
LABEL_SMOOTHING=0.1
GOLD_WEIGHT=2.0
NUM_WORKERS=4
OUTPUT_DIR="checkpoints/consensus_ft"
CONSENSUS_PATH="../Datasets/rsna-knee-abnormality-detection/consensus_labels.csv"

# 3. Determine Folds to Run
if [ "$SPECIFIC_FOLD" = "all" ]; then
    FOLDS_TO_RUN=(0 1 2 3 4)
else
    FOLDS_TO_RUN=("$SPECIFIC_FOLD")
fi

# 4. Sequential Execution Loop on Designated GPU
for fold in "${FOLDS_TO_RUN[@]}"; do
    echo ""
    echo "------------------------------------------------------------------"
    echo " Starting Fold ${fold} on GPU cuda:${GPU_ID} [$(date)]"
    echo " Log: logs/train_consensus_ft_f${fold}_gpu${GPU_ID}.log"
    echo "------------------------------------------------------------------"

    $PYTHON -u scripts/62_finetune_5fold_consensus.py \
      --device "cuda:${GPU_ID}" \
      --fold "${fold}" \
      --image_size "${IMAGE_SIZE}" \
      --slices_per_plane "${SLICES_PER_PLANE}" \
      --epochs "${EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --grad_accum_steps "${GRAD_ACCUM}" \
      --lr "${LR}" \
      --backbone_lr_mult "${BACKBONE_LR_MULT}" \
      --label_smoothing "${LABEL_SMOOTHING}" \
      --gold_weight "${GOLD_WEIGHT}" \
      --num_workers "${NUM_WORKERS}" \
      --output_dir "${OUTPUT_DIR}" \
      --consensus_path "${CONSENSUS_PATH}" \
      2>&1 | tee "logs/train_consensus_ft_f${fold}_gpu${GPU_ID}.log"

    echo "✓ Fold ${fold} Complete!"
done

echo ""
echo "=================================================================="
echo " Training Complete for Fold(s) [${FOLDS_TO_RUN[*]}]!"
echo " Finetuned Weights Saved In: ${OUTPUT_DIR}/"
echo " Original Weights Preserved In: checkpoints/ and checkpoints/backup_phase16/"
echo " End Time: $(date)"
echo "=================================================================="
