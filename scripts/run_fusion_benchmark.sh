#!/usr/bin/env bash
# ==============================================================================
# RSNA KneeVision-AI: Fusion Ensembles Benchmark Runner
# ==============================================================================
# Consolidates and replaces individual benchmark scripts:
#   - scripts/run_fusion_benchmark_fold0.sh
#   - scripts/run_fusion_benchmark_fold1.sh
#
# Target Python Script:
#   scripts/34_benchmark_all_fusion_ensembles.py
#
# USAGE EXAMPLES:
#   # Run Fold 0 on GPU 0 (default):
#   ./scripts/run_fusion_benchmark.sh --fold 0 --gpu 0
#
#   # Run Fold 1 on GPU 1:
#   ./scripts/run_fusion_benchmark.sh --fold 1 --gpu 1
#
#   # Run Fold 0 with custom epochs and live stdout:
#   ./scripts/run_fusion_benchmark.sh --fold 0 --gpu 0 --epochs 25 --foreground
# ==============================================================================

set -euo pipefail

# Locate Python binary
PYTHON_BIN="${PYTHON_BIN:-/home/AQ44130/miniconda3/envs/rsna-knee/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(which python3)"
fi

FOLD=0
GPU_ID=""
EPOCHS=20
LR="1e-3"
WEIGHT_DECAY="1e-3"
BASE_ARTIFACTS_DIR="artifacts/experiments/phase_11_fusion_benchmarks"
FOREGROUND=0

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --lr)
            LR="$2"
            shift 2
            ;;
        --weight_decay|--weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --base_artifacts_dir|--base-artifacts-dir)
            BASE_ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --foreground|-f)
            FOREGROUND=1
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --fold N                 Validation fold (default: 0)"
            echo "  --gpu N                  GPU device ID (default: matches fold or 0)"
            echo "  --epochs N               Number of training epochs (default: 20)"
            echo "  --lr RATE                Learning rate (default: 1e-3)"
            echo "  --weight_decay RATE      Weight decay (default: 1e-3)"
            echo "  --base_artifacts_dir DIR Base output artifacts directory"
            echo "  --foreground, -f         Print output directly to terminal instead of log file"
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

if [[ -z "${GPU_ID}" ]]; then
    GPU_ID="${FOLD}"
fi

LOG_DIR="${BASE_ARTIFACTS_DIR}/logs"
LOG_FILE="${LOG_DIR}/benchmark_fold${FOLD}.log"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=============================================================================="
echo "📊 RSNA KneeVision-AI: Fusion Ensembles Benchmark"
echo "=============================================================================="
echo "Fold             : ${FOLD}"
echo "GPU ID           : ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Epochs           : ${EPOCHS}"
echo "Learning Rate    : ${LR} (Weight Decay: ${WEIGHT_DECAY})"
echo "Base Output Dir  : ${BASE_ARTIFACTS_DIR}"
echo "Log File         : ${LOG_FILE}"
echo "Python Binary    : ${PYTHON_BIN}"
echo "=============================================================================="

CMD=(
    "${PYTHON_BIN}" scripts/34_benchmark_all_fusion_ensembles.py
    --fold "${FOLD}"
    --device "cuda:0"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --base_artifacts_dir "${BASE_ARTIFACTS_DIR}"
)

if [[ ${FOREGROUND} -eq 1 ]]; then
    "${CMD[@]}"
else
    echo "Logging output to ${LOG_FILE}..."
    "${CMD[@]}" > "${LOG_FILE}" 2>&1
    echo "Completed benchmark for fold ${FOLD}. See logs at ${LOG_FILE}"
fi
