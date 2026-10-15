#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Push Phase 18 Consensus 5-Fold Submission
# ==============================================================================
set -euo pipefail

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_74784511185887b4fad7d60e85b0feb5}"
PYTHON="${PYTHON:-python}"
KAGGLE="${KAGGLE:-kaggle}"

echo "========================================================================"
echo " 1. Regenerating Phase 18 Consensus 5-Fold Notebook & Syncing Weights"
echo "========================================================================"
${PYTHON} scripts/63_export_phase18_consensus_5fold_submission.py

echo ""
echo "========================================================================"
echo " 2. Uploading New Weights Version to Kaggle"
echo "    Dataset: sukeshadigav/rsna-knee-kneevision-weights"
echo "========================================================================"
${KAGGLE} datasets version -p kaggle_upload/weights -m "Phase 18 Consensus 5-Fold ConvNeXt-Small 384px Checkpoints"

echo ""
echo "========================================================================"
echo " 3. Pushing Phase 18 Consensus 5-Fold Submission Notebook to Kaggle"
echo "    Kernel: sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
${KAGGLE} kernels push -p kaggle_upload/kernel

echo ""
echo "========================================================================"
echo " Push Complete! Track your submission kernel status with:"
echo "   ${KAGGLE} kernels status sukeshadigav/kneevision-ai-rsna-submission"
echo "========================================================================"
