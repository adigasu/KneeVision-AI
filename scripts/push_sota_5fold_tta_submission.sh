#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Push SOTA 5-Fold ConvNeXt-Small (384px) + TTA
# Target Kernel: sukeshadigav/kneevision-ai-rsna-submission
# ==============================================================================
set -euo pipefail

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_74784511185887b4fad7d60e85b0feb5}"
PYTHON="${PYTHON:-python}"
KAGGLE="${KAGGLE:-kaggle}"

echo "========================================================================"
echo " 1. Regenerating 5-Fold SOTA Notebook (Parity Preprocessing + TTA)"
echo "========================================================================"
${PYTHON} scripts/55_export_sota_5fold_tta_submission.py

echo ""
echo "========================================================================"
echo " 2. Pushing Upgraded Submission Notebook to Kaggle"
echo "    Kernel: sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
${KAGGLE} kernels push -p kaggle_upload/kernel

echo ""
echo "========================================================================"
echo " Push Complete! Track your submission kernel status with:"
echo "   ${KAGGLE} kernels status sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
