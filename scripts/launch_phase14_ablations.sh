#!/usr/bin/env bash
# RSNA Knee Abnormality Detection - Phase 14 Independent Ablation Launcher (20 Epochs Benchmark)
set -e

PY="/home/AQ44130/miniconda3/envs/rsna-knee/bin/python"

mkdir -p logs checkpoints

echo "======================================================================"
echo "  Dispatching Phase 14 Independent Ablations on 2x RTX A6000 GPUs"
echo "  Benchmark Setting: Exactly 20 Epochs, Warmup 3 Epochs (matching Phase 13)"
echo "  Baseline (Phase 13 ConvNeXt-Tiny F0): Val AUC 0.8684 | Gold AUC 0.9117"
echo "======================================================================"

# GPU 0: Run Arm 1 (AFCL Loss alone) then Arm 3 (3D Intra-Plane Sequence Transformer alone)
nohup bash -c "
echo '[GPU 0] Starting Arm 1: Asymmetric Focal Consensus Loss (AFCL alone, 20 Epochs)...'
$PY -u scripts/39_train_ablation.py \
    --ablation_mode afcl \
    --backbone convnext_tiny \
    --fold 0 \
    --epochs 20 \
    --warmup_epochs 3 \
    --device cuda:0 \
    > logs/train_phase14_ablation_afcl_convnext_tiny_f0_gpu0.log 2>&1

echo '[GPU 0] Arm 1 Complete! Starting Arm 3: 3D Intra-Plane Sequence Transformer alone (20 Epochs)...'
$PY -u scripts/39_train_ablation.py \
    --ablation_mode intra_transformer \
    --backbone convnext_tiny \
    --fold 0 \
    --epochs 20 \
    --warmup_epochs 3 \
    --device cuda:0 \
    > logs/train_phase14_ablation_intra_convnext_tiny_f0_gpu0.log 2>&1

echo '[GPU 0] All GPU 0 ablation jobs complete!'
" > logs/gpu0_ablation_worker.log 2>&1 &
PID_GPU0=$!
echo "Dispatched GPU 0 Worker (Arm 1 AFCL -> Arm 3 Intra-Transformer) [PID: $PID_GPU0]"

# GPU 1: Run Arm 2 (High-Res 384px 96sl alone) then Arm 4 (Tri-Planar Co-Attention alone)
nohup bash -c "
echo '[GPU 1] Starting Arm 2: High-Resolution Multi-Planar Pipeline (384px, 96 slices alone, 20 Epochs)...'
$PY -u scripts/39_train_ablation.py \
    --ablation_mode highres \
    --backbone convnext_tiny \
    --fold 0 \
    --epochs 20 \
    --warmup_epochs 3 \
    --device cuda:1 \
    > logs/train_phase14_ablation_highres_convnext_tiny_f0_gpu1.log 2>&1

echo '[GPU 1] Arm 2 Complete! Starting Arm 4: Tri-Planar Co-Attention alone (20 Epochs)...'
$PY -u scripts/39_train_ablation.py \
    --ablation_mode coattention \
    --backbone convnext_tiny \
    --fold 0 \
    --epochs 20 \
    --warmup_epochs 3 \
    --device cuda:1 \
    > logs/train_phase14_ablation_coattention_convnext_tiny_f0_gpu1.log 2>&1

echo '[GPU 1] All GPU 1 ablation jobs complete!'
" > logs/gpu1_ablation_worker.log 2>&1 &
PID_GPU1=$!
echo "Dispatched GPU 1 Worker (Arm 2 High-Res -> Arm 4 Co-Attention) [PID: $PID_GPU1]"

echo "======================================================================"
echo "  Both workers active in background! Monitor progress via:"
echo "  tail -f logs/train_phase14_ablation_afcl_convnext_tiny_f0_gpu0.log"
echo "  tail -f logs/train_phase14_ablation_highres_convnext_tiny_f0_gpu1.log"
echo "======================================================================"
