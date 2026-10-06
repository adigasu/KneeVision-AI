#!/bin/bash
PYTHON=/home/AQ44130/miniconda3/envs/rsna-knee/bin/python

echo "Launching 336px Tiny on GPU 0..."
nohup $PYTHON -u scripts/46_train_decoupled_triplanar_mil.py \
  --device cuda:0 \
  --backbone convnext_tiny \
  --image_size 336 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --use_bottleneck_weights 0 \
  --run_name phase16_decoupled_convnext_tiny_336px_f0 \
  > logs/train_phase16_decoupled_convnext_tiny_336px_f0_gpu0.log 2>&1 &

echo "Launching 384px Small on GPU 1..."
nohup $PYTHON -u scripts/46_train_decoupled_triplanar_mil.py \
  --device cuda:1 \
  --backbone convnext_small \
  --image_size 384 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --use_bottleneck_weights 0 \
  --run_name phase16_decoupled_convnext_small_384px_f0 \
  > logs/train_phase16_decoupled_convnext_small_384px_f0_gpu1.log 2>&1 &

echo "Both processes launched in background."
