#!/usr/bin/env bash
# ==============================================================================
# RSNA KneeVision-AI: Unified MRI-Aware MIL Training Pipeline (Phases 11 & 12)
# ==============================================================================
# Consolidates and replaces individual run scripts:
#   - scripts/train_phase11_convnext_small_gpu1.sh
#   - scripts/train_phase11_dinov2_small_fold1_gpu0.sh
#   - scripts/train_phase11_dinov2_small_gpu0.sh
#   - scripts/train_phase12_dinov3_small_fold1_gpu1.sh
#   - scripts/train_phase12_dinov3_small_gpu1.sh
#   - scripts/train_phase12_efficientnet_b0_frozen_bn_gpu0.sh
#   - scripts/train_phase12_efficientnet_b0_gpu0.sh
#
# Target Python Script:
#   scripts/27_train_mri_aware_mil.py
#
# USAGE EXAMPLES:
#   # 1. ConvNeXt-Small (Fold 0 on GPU 1):
#   ./scripts/train_mri_aware_mil.sh --backbone convnext_small --fold 0 --gpu 1
#
#   # 2. DINOv2-Small (Fold 0 on GPU 0, 20 epochs):
#   ./scripts/train_mri_aware_mil.sh --backbone dinov2_small --fold 0 --gpu 0 --epochs 20
#
#   # 3. DINOv2-Small (Fold 1 on GPU 0, 15 epochs):
#   ./scripts/train_mri_aware_mil.sh --backbone dinov2_small --fold 1 --gpu 0 --epochs 15
#
#   # 4. DINOv3-Small (Fold 0 on GPU 1, Phase 12):
#   ./scripts/train_mri_aware_mil.sh --backbone dinov3_small --fold 0 --gpu 1
#
#   # 5. DINOv3-Small (Fold 1 on GPU 1, Phase 12):
#   ./scripts/train_mri_aware_mil.sh --backbone dinov3_small --fold 1 --gpu 1
#
#   # 6. EfficientNet-B0 (Fold 0 on GPU 0):
#   ./scripts/train_mri_aware_mil.sh --backbone efficientnet_b0 --fold 0 --gpu 0
#
#   # 7. EfficientNet-B0 with Frozen BatchNorm (Fold 0 on GPU 0):
#   ./scripts/train_mri_aware_mil.sh --backbone efficientnet_b0 --fold 0 --gpu 0 --freeze_bn
# ==============================================================================

set -euo pipefail

# Locate Python binary (conda env preferred, fallback to system python3)
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(which python3)"
fi

# Default hyperparameters
BACKBONE="dinov3_small"
FOLD=0
GPU_ID=0
EPOCHS=""
BATCH_SIZE=4
GRAD_ACCUM=8
LR=""
BACKBONE_LR_MULT=""
LOSS_TYPE="consensus_denoised"
GOLD_WEIGHT="2.0"
IMG_SIZE=288
TARGET_SLICES=32
FREEZE_BN=0
OUTPUT_DIR=""
EXP_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum_steps|--grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --backbone_lr_mult)
            BACKBONE_LR_MULT="$2"
            shift 2
            ;;
        --freeze_bn|--freeze-bn)
            FREEZE_BN=1
            shift 1
            ;;
        --output_dir|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --exp_name|--exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --loss_type|--loss-type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --gold_weight|--gold-weight)
            GOLD_WEIGHT="$2"
            shift 2
            ;;
        --image_size|--image-size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --target_slices|--target-slices)
            TARGET_SLICES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --backbone MODEL         Backbone architecture (default: dinov3_small)"
            echo "                           Options: dinov3_small, convnext_small, dinov2_small, efficientnet_b0, etc."
            echo "  --fold N                 Fold index (default: 0)"
            echo "  --gpu N                  GPU device ID (default: 0)"
            echo "  --epochs N               Training epochs (default: 20, or 15 for dinov2 fold 1)"
            echo "  --batch_size N           Batch size (default: 4)"
            echo "  --grad_accum N           Gradient accumulation steps (default: 8)"
            echo "  --lr RATE                Learning rate (auto-selected if unset)"
            echo "  --backbone_lr_mult MULT  Backbone LR multiplier (auto-selected if unset)"
            echo "  --freeze_bn              Freeze BatchNorm running stats"
            echo "  --output_dir PATH        Directory to save checkpoints and metrics"
            echo "  --exp_name NAME          Experiment name identifier"
            echo "  --loss_type TYPE         Loss function (default: consensus_denoised)"
            echo "  --gold_weight W          Gold label loss weight (default: 2.0)"
            echo "  --image_size N           Input slice resolution (default: 288)"
            echo "  --target_slices N        Target number of slices per study (default: 32)"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use -h or --help for usage details."
            exit 1
            ;;
    esac
done

# Architecture-specific defaults if not explicitly passed
if [[ -z "${LR}" ]]; then
    if [[ "${BACKBONE}" == *"efficientnet"* ]]; then
        LR="2.0e-4"
    else
        LR="1.5e-4"
    fi
fi

if [[ -z "${BACKBONE_LR_MULT}" ]]; then
    if [[ "${BACKBONE}" == *"efficientnet"* ]]; then
        BACKBONE_LR_MULT="0.20"
    else
        BACKBONE_LR_MULT="0.15"
    fi
fi

if [[ -z "${EPOCHS}" ]]; then
    if [[ "${BACKBONE}" == "dinov2_small" && "${FOLD}" == "1" ]]; then
        EPOCHS=15
    else
        EPOCHS=20
    fi
fi

# Set default output_dir and exp_name if not provided
if [[ -z "${OUTPUT_DIR}" && -z "${EXP_NAME}" ]]; then
    if [[ "${FREEZE_BN}" -eq 1 ]]; then
        OUTPUT_DIR="artifacts/experiments/phase_12_${BACKBONE}/robust_consensus_denoised_frozen_bn"
    else
        OUTPUT_DIR="artifacts/experiments/phase_12_${BACKBONE}/robust_consensus_denoised"
    fi
fi

mkdir -p logs

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=============================================================================="
echo "🎯 RSNA KneeVision-AI: MRI-Aware MIL Training"
echo "=============================================================================="
echo "Backbone         : ${BACKBONE}"
echo "Fold             : ${FOLD}"
echo "GPU ID           : ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Epochs           : ${EPOCHS}"
echo "Batch Size       : ${BATCH_SIZE} (Grad Accum: ${GRAD_ACCUM}, Effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "Learning Rate    : ${LR} (Backbone Mult: ${BACKBONE_LR_MULT})"
echo "Loss Type        : ${LOSS_TYPE} (Gold Weight: ${GOLD_WEIGHT})"
echo "Input Dimensions : ${IMG_SIZE}x${IMG_SIZE} (${TARGET_SLICES} slices)"
echo "Freeze BN        : $( [[ ${FREEZE_BN} -eq 1 ]] && echo 'Yes' || echo 'No' )"
[[ -n "${OUTPUT_DIR}" ]] && echo "Output Dir       : ${OUTPUT_DIR}"
[[ -n "${EXP_NAME}" ]] && echo "Exp Name         : ${EXP_NAME}"
echo "Python Binary    : ${PYTHON_BIN}"
echo "=============================================================================="

CMD=(
    "${PYTHON_BIN}" scripts/27_train_mri_aware_mil.py
    --backbone "${BACKBONE}"
    --fold "${FOLD}"
    --loss_type "${LOSS_TYPE}"
    --gold_weight "${GOLD_WEIGHT}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum_steps "${GRAD_ACCUM}"
    --lr "${LR}"
    --backbone_lr_mult "${BACKBONE_LR_MULT}"
    --image_size "${IMG_SIZE}"
    --target_slices "${TARGET_SLICES}"
    --device "cuda:0"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi

if [[ -n "${EXP_NAME}" ]]; then
    CMD+=(--exp_name "${EXP_NAME}")
fi

if [[ "${FREEZE_BN}" -eq 1 ]]; then
    CMD+=(--freeze_bn)
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"
