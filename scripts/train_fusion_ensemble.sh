#!/usr/bin/env bash
# ==============================================================================
# RSNA KneeVision-AI: Unified Multi-Backbone Fusion & Ensemble Trainer
# ==============================================================================
# Consolidates and replaces individual phase 11 fusion training scripts:
#   - scripts/train_joint_fusion_gpu1.sh           (30_train_joint_fusion_mil.py)
#   - scripts/train_pre_classifier_fusion_gpu1.sh  (31_train_pre_classifier_fusion.py)
#   - scripts/train_stacking_ensemble_gpu1.sh      (29_train_stacking_ensemble.py)
#
# USAGE EXAMPLES:
#   # 1. Train Joint Multi-Backbone Feature Fusion (1920d MIL):
#   ./scripts/train_fusion_ensemble.sh --mode joint --fold 0 --gpu 1
#
#   # 2. Train Pre-Classifier Embedding Fusion (1920d Head):
#   ./scripts/train_fusion_ensemble.sh --mode pre_classifier --fold 0 --gpu 1
#
#   # 3. Train Stacking Meta-Learner:
#   ./scripts/train_fusion_ensemble.sh --mode stacking --fold 0 --gpu 1
#
#   # 4. Train All Three Fusion Methods sequentially:
#   ./scripts/train_fusion_ensemble.sh --mode all --fold 0 --gpu 1
# ==============================================================================

set -euo pipefail
ulimit -n 65535 2>/dev/null || true

# Locate Python binary
PYTHON_BIN="${PYTHON_BIN:-/home/AQ44130/miniconda3/envs/rsna-knee/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(which python3)"
fi

# Defaults
MODE="joint"
FOLD=0
GPU_ID=1

# Mode-specific tunable defaults
EPOCHS=""
BATCH_SIZE=2
GRAD_ACCUM=16
LR_HEAD="1.5e-4"
LR_BACKBONE="1.5e-5"
LR="1e-3"
WEIGHT_DECAY="1e-3"
DROPOUT="0.25"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode|-m)
            MODE="$2"
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
        --lr_head)
            LR_HEAD="$2"
            shift 2
            ;;
        --lr_backbone)
            LR_BACKBONE="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --mode MODE              Training target: joint, pre_classifier, stacking, all (default: joint)"
            echo "  --fold N                 Validation fold (default: 0)"
            echo "  --gpu N                  GPU device ID (default: 1)"
            echo "  --epochs N               Epochs (default: 10 for joint, 30 for pre_classifier)"
            echo "  --batch_size N           Batch size for joint training (default: 2)"
            echo "  --grad_accum N           Gradient accumulation steps (default: 16)"
            echo "  --lr RATE                Learning rate for pre_classifier (default: 1e-3)"
            echo "  --lr_head RATE           Head LR for joint training (default: 1.5e-4)"
            echo "  --lr_backbone RATE       Backbone LR for joint training (default: 1.5e-5)"
            echo "  --dropout RATE           Dropout for pre_classifier (default: 0.25)"
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

mkdir -p logs
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

run_joint() {
    local ep="${EPOCHS:-10}"
    echo "=============================================================================="
    echo "🔗 Training Joint Multi-Backbone Feature Fusion (30_train_joint_fusion_mil.py)"
    echo "Fold: ${FOLD}, Epochs: ${ep}, Batch: ${BATCH_SIZE}, GradAccum: ${GRAD_ACCUM}, GPU: ${GPU_ID}"
    echo "=============================================================================="
    "${PYTHON_BIN}" scripts/30_train_joint_fusion_mil.py         --fold "${FOLD}"         --epochs "${ep}"         --batch_size "${BATCH_SIZE}"         --grad_accum_steps "${GRAD_ACCUM}"         --lr_head "${LR_HEAD}"         --lr_backbone "${LR_BACKBONE}"         --device "cuda:0"
}

run_pre_classifier() {
    local ep="${EPOCHS:-30}"
    echo "=============================================================================="
    echo "�� Training Pre-Classifier Embedding Fusion (31_train_pre_classifier_fusion.py)"
    echo "Fold: ${FOLD}, Epochs: ${ep}, LR: ${LR}, Weight Decay: ${WEIGHT_DECAY}, Dropout: ${DROPOUT}, GPU: ${GPU_ID}"
    echo "=============================================================================="
    "${PYTHON_BIN}" scripts/31_train_pre_classifier_fusion.py         --fold "${FOLD}"         --epochs "${ep}"         --lr "${LR}"         --weight_decay "${WEIGHT_DECAY}"         --dropout "${DROPOUT}"         --device "cuda:0"
}

run_stacking() {
    echo "=============================================================================="
    echo "📚 Training Stacking Meta-Learner (29_train_stacking_ensemble.py)"
    echo "Fold: ${FOLD}, GPU: ${GPU_ID}"
    echo "=============================================================================="
    "${PYTHON_BIN}" scripts/29_train_stacking_ensemble.py         --fold "${FOLD}"         --device "cuda:0"
}

case "${MODE}" in
    joint)
        run_joint
        ;;
    pre_classifier|pre-classifier)
        run_pre_classifier
        ;;
    stacking)
        run_stacking
        ;;
    all)
        echo "🚀 Running all 3 fusion training stages in sequence..."
        run_joint
        run_pre_classifier
        run_stacking
        ;;
    *)
        echo "Error: Unknown mode "${MODE}". Choose from: joint, pre_classifier, stacking, all."
        exit 1
        ;;
esac
