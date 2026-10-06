#!/usr/bin/env bash
set -e

PY="/home/AQ44130/miniconda3/envs/rsna-knee/bin/python"

echo "=== Launching Phase 13 Fold 1 Tri-Planar Models ==="

# 1. Fold 1 ConvNeXt-Tiny on GPU 0
nohup $PY -u scripts/36_train_triplanar_mil.py \
    --backbone convnext_tiny \
    --fold 1 \
    --slices_per_plane 24 \
    --image_size 288 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --epochs 20 \
    --lr 3e-4 \
    --backbone_lr_mult 0.1 \
    --loss_type consensus_denoised \
    --gold_weight 2.0 \
    --device cuda:0 \
    --run_name phase13_triplanar_convnext_tiny_f1_72sl \
    > logs/train_phase13_triplanar_convnext_tiny_f1_gpu0.log 2>&1 &
echo "Launched Fold 1 ConvNeXt-Tiny on GPU 0 (PID: $!)"

# 2. Fold 1 ConvNeXt-Small on GPU 1
nohup $PY -u scripts/36_train_triplanar_mil.py \
    --backbone convnext_small \
    --fold 1 \
    --slices_per_plane 24 \
    --image_size 288 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --epochs 20 \
    --lr 3e-4 \
    --backbone_lr_mult 0.1 \
    --loss_type consensus_denoised \
    --gold_weight 2.0 \
    --device cuda:1 \
    --run_name phase13_triplanar_convnext_small_f1_72sl \
    > logs/train_phase13_triplanar_convnext_small_f1_gpu1.log 2>&1 &
echo "Launched Fold 1 ConvNeXt-Small on GPU 1 (PID: $!)"

# 3. Fold 1 DINOv2-Small on GPU 0
nohup $PY -u scripts/36_train_triplanar_mil.py \
    --backbone vit_small_patch14_dinov2.lvd142m \
    --fold 1 \
    --slices_per_plane 24 \
    --image_size 280 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --epochs 20 \
    --lr 2e-4 \
    --backbone_lr_mult 0.1 \
    --loss_type consensus_denoised \
    --gold_weight 2.0 \
    --device cuda:0 \
    --run_name phase13_triplanar_dinov2_small_f1_72sl \
    > logs/train_phase13_triplanar_dinov2_small_f1_gpu0.log 2>&1 &
echo "Launched Fold 1 DINOv2-Small on GPU 0 (PID: $!)"

echo "=== All 3 jobs dispatched successfully! ==="
