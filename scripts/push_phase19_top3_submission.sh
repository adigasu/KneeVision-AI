#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Push Phase 19 Top-3 Consensus Submission
# ==============================================================================
set -euo pipefail

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_74784511185887b4fad7d60e85b0feb5}"
PYTHON="${PYTHON:-python}"
KAGGLE="${KAGGLE:-kaggle}"

echo "========================================================================"
echo " 1. Regenerating Phase 19 Top-3 Consensus Notebook & Syncing Weights"
echo "========================================================================"
${PYTHON} scripts/64_export_phase18_top3_ensemble.py

echo ""
echo "========================================================================"
echo " 2. Pushing Phase 19 Top-3 Consensus Submission Notebook to Kaggle"
echo "    Kernel: sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
${KAGGLE} kernels push -p kaggle_upload/kernel

echo ""
echo "========================================================================"
echo " Push Complete! Track your submission kernel status with:"
echo "   ${KAGGLE} kernels status sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
