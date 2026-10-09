#!/bin/bash
set -e

PYTHON="/home/AQ44130/miniconda3/envs/rsna-knee/bin/python"
WORKDIR="/data/users/sukesh/KneeVision-AI"
cd $WORKDIR

mkdir -p logs checkpoints

echo "=================================================================="
echo " Starting Multi-Fold Decoupled ConvNeXt-Small 384px Training (20 Epochs)"
echo " Start Time: $(date)"
echo "=================================================================="

# ── STAGE 1: Fold 1 (GPU 0) and Fold 2 (GPU 1) in Parallel ─────────────────────
echo ""
echo "[STAGE 1] Launching Fold 1 on GPU 0 and Fold 2 on GPU 1..."

$PYTHON -u scripts/46_train_decoupled_triplanar_mil.py \
  --device cuda:0 \
  --backbone convnext_small \
  --fold 1 \
  --image_size 384 \
  --epochs 20 \
  --warmup_epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lr 3e-4 \
  --backbone_lr_mult 0.1 \
  --gold_weight 2.0 \
  --pos_thresh 0.70 \
  --neg_thresh 0.30 \
  --num_workers 4 \
  --use_bottleneck_weights 0 \
  --classifier_type linear \
  --run_name phase16_decoupled_convnext_small_384px_f1 \
  > logs/train_phase16_decoupled_convnext_small_384px_f1_gpu0.log 2>&1 &
PID_F1=$!
echo "  ✓ Fold 1 launched on GPU 0 (PID: $PID_F1) -> logs/train_phase16_decoupled_convnext_small_384px_f1_gpu0.log"

$PYTHON -u scripts/46_train_decoupled_triplanar_mil.py \
  --device cuda:1 \
  --backbone convnext_small \
  --fold 2 \
  --image_size 384 \
  --epochs 20 \
  --warmup_epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lr 3e-4 \
  --backbone_lr_mult 0.1 \
  --gold_weight 2.0 \
  --pos_thresh 0.70 \
  --neg_thresh 0.30 \
  --num_workers 4 \
  --use_bottleneck_weights 0 \
  --classifier_type linear \
  --run_name phase16_decoupled_convnext_small_384px_f2 \
  > logs/train_phase16_decoupled_convnext_small_384px_f2_gpu1.log 2>&1 &
PID_F2=$!
echo "  ✓ Fold 2 launched on GPU 1 (PID: $PID_F2) -> logs/train_phase16_decoupled_convnext_small_384px_f2_gpu1.log"

echo ""
echo "[STAGE 1] Waiting for Fold 1 ($PID_F1) and Fold 2 ($PID_F2) to complete..."
wait $PID_F1
STATUS_F1=$?
wait $PID_F2
STATUS_F2=$?

echo ""
echo "Stage 1 Finished: Fold 1 exited with code $STATUS_F1, Fold 2 exited with code $STATUS_F2"
echo "Timestamp: $(date)"

# ── STAGE 2: Fold 3 (GPU 0) and Fold 4 (GPU 1) in Parallel ─────────────────────
echo ""
echo "[STAGE 2] Launching Fold 3 on GPU 0 and Fold 4 on GPU 1..."

$PYTHON -u scripts/46_train_decoupled_triplanar_mil.py \
  --device cuda:0 \
  --backbone convnext_small \
  --fold 3 \
  --image_size 384 \
  --epochs 20 \
  --warmup_epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lr 3e-4 \
  --backbone_lr_mult 0.1 \
  --gold_weight 2.0 \
  --pos_thresh 0.70 \
  --neg_thresh 0.30 \
  --num_workers 4 \
  --use_bottleneck_weights 0 \
  --classifier_type linear \
  --run_name phase16_decoupled_convnext_small_384px_f3 \
  > logs/train_phase16_decoupled_convnext_small_384px_f3_gpu0.log 2>&1 &
PID_F3=$!
echo "  ✓ Fold 3 launched on GPU 0 (PID: $PID_F3) -> logs/train_phase16_decoupled_convnext_small_384px_f3_gpu0.log"

$PYTHON -u scripts/46_train_decoupled_triplanar_mil.py \
  --device cuda:1 \
  --backbone convnext_small \
  --fold 4 \
  --image_size 384 \
  --epochs 20 \
  --warmup_epochs 2 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lr 3e-4 \
  --backbone_lr_mult 0.1 \
  --gold_weight 2.0 \
  --pos_thresh 0.70 \
  --neg_thresh 0.30 \
  --num_workers 4 \
  --use_bottleneck_weights 0 \
  --classifier_type linear \
  --run_name phase16_decoupled_convnext_small_384px_f4 \
  > logs/train_phase16_decoupled_convnext_small_384px_f4_gpu1.log 2>&1 &
PID_F4=$!
echo "  ✓ Fold 4 launched on GPU 1 (PID: $PID_F4) -> logs/train_phase16_decoupled_convnext_small_384px_f4_gpu1.log"

echo ""
echo "[STAGE 2] Waiting for Fold 3 ($PID_F3) and Fold 4 ($PID_F4) to complete..."
wait $PID_F3
STATUS_F3=$?
wait $PID_F4
STATUS_F4=$?

echo ""
echo "Stage 2 Finished: Fold 3 exited with code $STATUS_F3, Fold 4 exited with code $STATUS_F4"
echo "=================================================================="
echo " All 4 Folds (1, 2, 3, 4) Complete! Finished at: $(date)"
echo "=================================================================="
