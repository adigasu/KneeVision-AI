#!/usr/bin/env bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Phase 18: Consensus Labels 5-Fold Finetuner
# Finetunes 5-Fold ConvNeXt-Small (384px) checkpoints using new consensus labels.
#
# Hardware: Optimized for 4x NVIDIA RTX A6000 (48GB VRAM each).
# Execution:
#   Stage 1: Folds 0, 1, 2, 3 launched in parallel on GPUs 0, 1, 2, 3
#   Stage 2: Fold 4 launched on GPU 0
# ==============================================================================

set -euo pipefail

# 1. Environment
PYTHON="python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

echo "=================================================================="
echo " Starting 5-Fold ConvNeXt-Small (384px) Consensus Finetuning"
echo " Python: $PYTHON"
echo " Start Time: $(date)"
echo "=================================================================="

# 2. Safety: Create a Dedicated Backup of Existing Phase 16 Weights
echo ""
echo "[Step 1] Creating Safety Backup of existing Phase 16 weights..."
mkdir -p checkpoints/backup_phase16
mkdir -p checkpoints/consensus_ft
mkdir -p logs

for fold in 0 1 2 3 4; do
    SRC="checkpoints/phase16_decoupled_convnext_small_384px_f${fold}_best.pth"
    DST="checkpoints/backup_phase16/phase16_decoupled_convnext_small_384px_f${fold}_best.pth"
    if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
        cp -v "$SRC" "$DST"
    fi
done
echo "✓ All existing Phase 16 weights backed up safely to checkpoints/backup_phase16/!"

# 3. Finetuning Hyperparameters
IMAGE_SIZE=384
SLICES_PER_PLANE=24
EPOCHS=6
WARMUP_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=8
LR=1e-4
BACKBONE_LR_MULT=0.1
LABEL_SMOOTHING=0.1
GOLD_WEIGHT=2.0
NUM_WORKERS=4
OUTPUT_DIR="checkpoints/consensus_ft"
CONSENSUS_PATH="../Datasets/rsna-knee-abnormality-detection/consensus_labels.csv"

# 4. Stage 1: Folds 0, 1, 2, 3 in Parallel on GPUs 0, 1, 2, 3
echo ""
echo "[Step 2] Stage 1: Launching Folds 0, 1, 2, 3 simultaneously on GPUs 0, 1, 2, 3..."

$PYTHON -u scripts/62_finetune_5fold_consensus.py \
  --device cuda:0 \
  --fold 0 \
  --image_size $IMAGE_SIZE \
  --slices_per_plane $SLICES_PER_PLANE \
  --epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --backbone_lr_mult $BACKBONE_LR_MULT \
  --label_smoothing $LABEL_SMOOTHING \
  --gold_weight $GOLD_WEIGHT \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  --consensus_path $CONSENSUS_PATH \
  > logs/train_consensus_ft_f0_gpu0.log 2>&1 &
PID_F0=$!
echo "  ✓ Fold 0 running on GPU 0 (PID: $PID_F0) -> logs/train_consensus_ft_f0_gpu0.log"

$PYTHON -u scripts/62_finetune_5fold_consensus.py \
  --device cuda:1 \
  --fold 1 \
  --image_size $IMAGE_SIZE \
  --slices_per_plane $SLICES_PER_PLANE \
  --epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --backbone_lr_mult $BACKBONE_LR_MULT \
  --label_smoothing $LABEL_SMOOTHING \
  --gold_weight $GOLD_WEIGHT \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  --consensus_path $CONSENSUS_PATH \
  > logs/train_consensus_ft_f1_gpu1.log 2>&1 &
PID_F1=$!
echo "  ✓ Fold 1 running on GPU 1 (PID: $PID_F1) -> logs/train_consensus_ft_f1_gpu1.log"

$PYTHON -u scripts/62_finetune_5fold_consensus.py \
  --device cuda:2 \
  --fold 2 \
  --image_size $IMAGE_SIZE \
  --slices_per_plane $SLICES_PER_PLANE \
  --epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --backbone_lr_mult $BACKBONE_LR_MULT \
  --label_smoothing $LABEL_SMOOTHING \
  --gold_weight $GOLD_WEIGHT \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  --consensus_path $CONSENSUS_PATH \
  > logs/train_consensus_ft_f2_gpu2.log 2>&1 &
PID_F2=$!
echo "  ✓ Fold 2 running on GPU 2 (PID: $PID_F2) -> logs/train_consensus_ft_f2_gpu2.log"

$PYTHON -u scripts/62_finetune_5fold_consensus.py \
  --device cuda:3 \
  --fold 3 \
  --image_size $IMAGE_SIZE \
  --slices_per_plane $SLICES_PER_PLANE \
  --epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --backbone_lr_mult $BACKBONE_LR_MULT \
  --label_smoothing $LABEL_SMOOTHING \
  --gold_weight $GOLD_WEIGHT \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  --consensus_path $CONSENSUS_PATH \
  > logs/train_consensus_ft_f3_gpu3.log 2>&1 &
PID_F3=$!
echo "  ✓ Fold 3 running on GPU 3 (PID: $PID_F3) -> logs/train_consensus_ft_f3_gpu3.log"

echo ""
echo "[Step 2] Waiting for Folds 0, 1, 2, 3 to complete..."
wait $PID_F0
wait $PID_F1
wait $PID_F2
wait $PID_F3
echo "✓ Stage 1 Complete (Folds 0, 1, 2, 3 finished successfully!)"

# 5. Stage 2: Fold 4 on GPU 0
echo ""
echo "[Step 3] Stage 2: Launching Fold 4 on GPU 0..."

$PYTHON -u scripts/62_finetune_5fold_consensus.py \
  --device cuda:0 \
  --fold 4 \
  --image_size $IMAGE_SIZE \
  --slices_per_plane $SLICES_PER_PLANE \
  --epochs $EPOCHS \
  --warmup_epochs $WARMUP_EPOCHS \
  --batch_size $BATCH_SIZE \
  --grad_accum_steps $GRAD_ACCUM \
  --lr $LR \
  --backbone_lr_mult $BACKBONE_LR_MULT \
  --label_smoothing $LABEL_SMOOTHING \
  --gold_weight $GOLD_WEIGHT \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  --consensus_path $CONSENSUS_PATH \
  > logs/train_consensus_ft_f4_gpu0.log 2>&1 &
PID_F4=$!
echo "  ✓ Fold 4 running on GPU 0 (PID: $PID_F4) -> logs/train_consensus_ft_f4_gpu0.log"

wait $PID_F4
echo "✓ Stage 2 Complete (Fold 4 finished successfully!)"

echo ""
echo "=================================================================="
echo " All 5 Folds (0, 1, 2, 3, 4) Complete! Finished at: $(date)"
echo " Checkpoints saved in: checkpoints/consensus_ft/"
echo " Old weights safely preserved in: checkpoints/ and checkpoints/backup_phase16/"
echo "=================================================================="
