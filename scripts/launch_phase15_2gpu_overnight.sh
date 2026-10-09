#!/usr/bin/env bash
# ==============================================================================
# RSNA KneeVision-AI: Phase 15 Cleaned Dataset v3 + Curriculum Loss (2-GPU Overnight)
# ==============================================================================
# GPU 0: Fold 2 -> Fold 4
# GPU 1: Fold 3 -> Fold 0 (Head-to-head baseline comparison)
# ==============================================================================

set -e
mkdir -p logs checkpoints

PY="python"

echo "=================================================================="
echo " Launching Phase 15 2-GPU Overnight Training (Dataset v3 + Loss)  "
echo "=================================================================="

# ------------------------------------------------------------------------------
# GPU 0 Runner: Fold 2 then Fold 4
# ------------------------------------------------------------------------------
nohup bash -c '
    echo "[GPU 0] Starting Fold 2 ConvNeXt-Tiny on Dataset v3 + Curriculum Loss..."
    '"$PY"' -u scripts/45_train_phase15_denoised_curriculum.py \
        --fold 2 \
        --device cuda:0 \
        --epochs 12 \
        --batch_size 2 \
        --grad_accum_steps 8 \
        --labels_path data/dense_labels_master_v3_cleaned.parquet \
        --loss_type curriculum_denoised \
        --run_name phase15_triplanar_convnext_tiny_f2 \
        > logs/train_phase15_convnext_tiny_f2_gpu0.log 2>&1

    echo "[GPU 0] Fold 2 complete! Starting Fold 4 ConvNeXt-Tiny..."
    '"$PY"' -u scripts/45_train_phase15_denoised_curriculum.py \
        --fold 4 \
        --device cuda:0 \
        --epochs 12 \
        --batch_size 2 \
        --grad_accum_steps 8 \
        --labels_path data/dense_labels_master_v3_cleaned.parquet \
        --loss_type curriculum_denoised \
        --run_name phase15_triplanar_convnext_tiny_f4 \
        > logs/train_phase15_convnext_tiny_f4_gpu0.log 2>&1

    echo "[GPU 0] All GPU 0 Phase 15 jobs completed!"
' > /dev/null 2>&1 &

PID_GPU0=$!
echo "✓ Dispatched GPU 0 queue (PID: $PID_GPU0): Fold 2 -> Fold 4"

# ------------------------------------------------------------------------------
# GPU 1 Runner: Fold 3 then Fold 0
# ------------------------------------------------------------------------------
nohup bash -c '
    echo "[GPU 1] Starting Fold 3 ConvNeXt-Tiny on Dataset v3 + Curriculum Loss..."
    '"$PY"' -u scripts/45_train_phase15_denoised_curriculum.py \
        --fold 3 \
        --device cuda:1 \
        --epochs 12 \
        --batch_size 2 \
        --grad_accum_steps 8 \
        --labels_path data/dense_labels_master_v3_cleaned.parquet \
        --loss_type curriculum_denoised \
        --run_name phase15_triplanar_convnext_tiny_f3 \
        > logs/train_phase15_convnext_tiny_f3_gpu1.log 2>&1

    echo "[GPU 1] Fold 3 complete! Starting Fold 0 ConvNeXt-Tiny (Cleaned Head-to-Head)..."
    '"$PY"' -u scripts/45_train_phase15_denoised_curriculum.py \
        --fold 0 \
        --device cuda:1 \
        --epochs 12 \
        --batch_size 2 \
        --grad_accum_steps 8 \
        --labels_path data/dense_labels_master_v3_cleaned.parquet \
        --loss_type curriculum_denoised \
        --run_name phase15_triplanar_convnext_tiny_f0 \
        > logs/train_phase15_convnext_tiny_f0_gpu1.log 2>&1

    echo "[GPU 1] All GPU 1 Phase 15 jobs completed!"
' > /dev/null 2>&1 &

PID_GPU1=$!
echo "✓ Dispatched GPU 1 queue (PID: $PID_GPU1): Fold 3 -> Fold 0"

echo "=================================================================="
echo " Both GPU pipelines are running in background!"
echo " Monitor GPU 0 with: tail -f logs/train_phase15_convnext_tiny_f2_gpu0.log"
echo " Monitor GPU 1 with: tail -f logs/train_phase15_convnext_tiny_f3_gpu1.log"
echo "=================================================================="
