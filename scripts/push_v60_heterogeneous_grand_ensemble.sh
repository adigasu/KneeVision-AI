#!/usr/bin/env bash
set -e

export KAGGLE_API_TOKEN="KGAT_74784511185887b4fad7d60e85b0feb5"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=== KneeVision-AI: Pushing Version 60 Heterogeneous Grand Ensemble ==="
echo "Generating latest notebook..."
$PYTHON_BIN scripts/60_export_heterogeneous_grand_ensemble.py

echo "Pushing kernel to Kaggle..."
$KAGGLE_BIN kernels push -p kaggle_upload/kernel

echo "Checking kernel status..."
$KAGGLE_BIN kernels status sukeshadigav/kneevision-ai-rsna-submission
