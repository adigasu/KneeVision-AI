#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Push 6-Model T1+T2 Kaggle Submission
# ==============================================================================
set -euo pipefail

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_74784511185887b4fad7d60e85b0feb5}"
PYTHON="${PYTHON:-python}"
KAGGLE="${KAGGLE:-kaggle}"

echo "========================================================================"
echo " 1. Regenerating 6-Model Notebook & Verifying Checkpoints"
echo "========================================================================"
${PYTHON} scripts/54_export_6model_t1_t2_ensemble.py

echo ""
echo "========================================================================"
echo " 2. Uploading New Weights Version to Kaggle"
echo "    Dataset: sukeshadigav/rsna-knee-kneevision-weights"
echo "========================================================================"
${KAGGLE} datasets version -p kaggle_upload/weights -m "Phase 17 6-Model (3-Fold T1+T2) Decoupled Checkpoints & Optimal Blend Weights"

echo ""
echo "========================================================================"
echo " 3. Pushing 6-Model Multi-Contrast Submission Notebook to Kaggle"
echo "    Kernel: sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
${KAGGLE} kernels push -p kaggle_upload/kernel

echo ""
echo "========================================================================"
echo " Push Complete! Track your submission kernel status with:"
echo "   ${KAGGLE} kernels status sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
