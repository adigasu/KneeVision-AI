#!/usr/bin/env bash
# ==============================================================================
# RSNA KneeVision-AI: Foundation Model Fast Benchmark Runner
# ==============================================================================
# Consolidates and replaces individual benchmark scripts:
#   - scripts/run_foundation_benchmark_gpu0.sh (set1 models)
#   - scripts/run_second_foundation_benchmark_gpu0.sh (set2 models)
#
# Target Python Script:
#   scripts/33_fast_foundation_benchmark.py
#
# USAGE EXAMPLES:
#   # Run Set 1 (radimagenet, dinov3_small, dinov2_reg4, convnextv2_tiny, dinov2_small, convnext_tiny):
#   ./scripts/run_foundation_benchmark.sh --preset set1 --gpu 0
#
#   # Run Set 2 (dinov3_small, efficientnet_b0, efficientnet_b2, coatnet_0/1, vit_small/base):
#   ./scripts/run_foundation_benchmark.sh --preset set2 --gpu 0
#
#   # Run All Foundation Models:
#   ./scripts/run_foundation_benchmark.sh --preset all --gpu 0
#
#   # Run Specific Custom Models:
#   ./scripts/run_foundation_benchmark.sh --models dinov3_small convnext_tiny --fold 1 --gpu 1
# ==============================================================================

set -euo pipefail

# Locate Python binary
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(which python3)"
fi

# Model Presets
SET1_MODELS="radimagenet dinov3_small dinov2_reg4 convnextv2_tiny dinov2_small convnext_tiny"
SET2_MODELS="dinov3_small efficientnet_b0 efficientnet_b2 coatnet_0_rw_224 coatnet_1_rw_224 vit_small_patch16_224 vit_base_patch16_224.mae"
ALL_MODELS="radimagenet dinov3_small dinov2_reg4 convnextv2_tiny dinov2_small convnext_tiny efficientnet_b0 efficientnet_b2 coatnet_0_rw_224 coatnet_1_rw_224 vit_small_patch16_224 vit_base_patch16_224.mae"

# Defaults
PRESET="set1"
MODELS=""
FOLD=0
EPOCHS=15
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --models)
            # Collect all following tokens until next flag
            shift 1
            MODELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS="${MODELS} $1"
                shift 1
            done
            MODELS="$(echo "${MODELS}" | xargs)"
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --preset NAME     Benchmark suite preset: set1, set2, all (default: set1)"
            echo "  --models M1 M2... Custom space-separated list of models to evaluate"
            echo "  --fold N          Validation fold index (default: 0)"
            echo "  --epochs N        Number of probe training epochs (default: 15)"
            echo "  --gpu N           GPU device ID (default: 0)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Presets:"
            echo "  set1: ${SET1_MODELS}"
            echo "  set2: ${SET2_MODELS}"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use -h or --help for usage details."
            exit 1
            ;;
    esac
done

if [[ -z "${MODELS}" ]]; then
    case "${PRESET}" in
        set1) MODELS="${SET1_MODELS}" ;;
        set2) MODELS="${SET2_MODELS}" ;;
        all)  MODELS="${ALL_MODELS}" ;;
        *)
            echo "Error: Unknown preset "${PRESET}". Use set1, set2, all, or provide --models explicitly."
            exit 1
            ;;
    esac
fi

mkdir -p logs

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=============================================================================="
echo "⚡ RSNA KneeVision-AI: Fast Foundation Benchmark"
echo "=============================================================================="
echo "Preset        : ${PRESET}"
echo "Models        : ${MODELS}"
echo "Fold          : ${FOLD}"
echo "Epochs        : ${EPOCHS}"
echo "GPU ID        : ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Python Binary : ${PYTHON_BIN}"
echo "=============================================================================="

# shellcheck disable=SC2086
"${PYTHON_BIN}" scripts/33_fast_foundation_benchmark.py     --models ${MODELS}     --fold "${FOLD}"     --epochs "${EPOCHS}"     --device "cuda:0"
