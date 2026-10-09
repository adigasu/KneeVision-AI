#!/usr/bin/env bash
# RSNA KneeVision-AI: Phase 10 Parallel 2-GPU Tri-Planar Fine-Tuning Launcher
# GPU 0: TriPlanar Model 1 (DINOv3 ViT-B/16, 25 epochs, 3 warmup epochs, 5-Fold CV)
# GPU 1: TriPlanar Model 2 (ConvNeXt-Small, 25 epochs, 3 warmup epochs, 5-Fold CV)

set -e
mkdir -p artifacts/experiments/phase_10_triplanar_parallel/logs

MODEL1="${1:-vit_base_patch16_dinov3}"
MODEL2="${2:-convnext_small.fb_in22k_ft_in1k_384}"
EPOCHS="${3:-25}"
WARMUP="${4:-3}"
IMG_SIZE="${5:-384}"

echo "======================================================================================"
echo "🚀 LAUNCHING TRI-PLANAR PARALLEL 2-GPU FINE-TUNING PIPELINE (PHASE 10)"
echo "======================================================================================"
echo "GPU 1 -> TriPlanar Model 1: ${MODEL1} (${EPOCHS} epochs, ${WARMUP} warmup, 5-Fold CV)"
echo "GPU 0 -> TriPlanar Model 2: ${MODEL2} (${EPOCHS} epochs, ${WARMUP} warmup, 5-Fold CV)"
echo "Logs Directory: artifacts/experiments/phase_10_triplanar_parallel/logs/"
echo "======================================================================================"

# Launch TriPlanar Model 1 on GPU 1
nohup python -u scripts/25_train_parallel_triplanar_2gpu.py \
    --backbone_name "${MODEL1}" \
    --device "cuda:1" \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP} \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-4 \
    --backbone_lr_mult 0.15 \
    --image_size ${IMG_SIZE} \
    --target_slices 16 \
    --folds "0,1,2,3,4" \
    --num_workers 4 \
    --output_base_dir "artifacts/experiments/phase_10_triplanar_parallel" \
    > "artifacts/experiments/phase_10_triplanar_parallel/logs/gpu0_triplanar_${MODEL1//\//_}.log" 2>&1 &

PID1=$!
echo "✅ GPU 1 Process Launched [PID: ${PID1}] -> Log: artifacts/experiments/phase_10_triplanar_parallel/logs/gpu0_triplanar_${MODEL1//\//_}.log"

# Launch TriPlanar Model 2 on GPU 0
nohup python -u scripts/25_train_parallel_triplanar_2gpu.py \
    --backbone_name "${MODEL2}" \
    --device "cuda:0" \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP} \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 2e-4 \
    --backbone_lr_mult 0.15 \
    --image_size ${IMG_SIZE} \
    --target_slices 16 \
    --folds "0,1,2,3,4" \
    --num_workers 4 \
    --output_base_dir "artifacts/experiments/phase_10_triplanar_parallel" \
    > "artifacts/experiments/phase_10_triplanar_parallel/logs/gpu1_triplanar_${MODEL2//\//_}.log" 2>&1 &

PID2=$!
echo "✅ GPU 0 Process Launched [PID: ${PID2}] -> Log: artifacts/experiments/phase_10_triplanar_parallel/logs/gpu1_triplanar_${MODEL2//\//_}.log"

echo ""
echo "======================================================================================"
echo "📊 MONITORING INSTRUCTIONS:"
echo "  • Monitor GPU 1 Log: tail -f artifacts/experiments/phase_10_triplanar_parallel/logs/gpu0_triplanar_${MODEL1//\//_}.log"
echo "  • Monitor GPU 0 Log: tail -f artifacts/experiments/phase_10_triplanar_parallel/logs/gpu1_triplanar_${MODEL2//\//_}.log"
echo "  • Run Monitor UI:    python scripts/monitor_parallel_training.py"
echo "  • Watch GPU compute: watch -n 2 nvidia-smi"
echo "======================================================================================"
