#!/bin/bash
# ==============================================================================
# RSNA Knee Abnormality Detection - Launch 6-Model 3-Fold Decoupled Pipeline
# Spawns parallel worker processes on GPU 0 (T1 stream) and GPU 1 (T2 stream).
# ==============================================================================
set -euo pipefail
mkdir -p logs checkpoints

LOG_T1="logs/pipeline_phase17_t1_3fold_gpu0.log"
LOG_T2="logs/pipeline_phase17_t2_3fold_gpu1.log"

echo "======================================================================"
echo " Launching Phase 17: 6-Model 3-Fold Parallel Training Pipeline"
echo "   • GPU 0: T1 Anatomical (Folds 0, 1, 2) -> Logging to ${LOG_T1}"
echo "   • GPU 1: T2 FS         (Folds 0, 1, 2) -> Logging to ${LOG_T2}"
echo "======================================================================"

nohup bash scripts/run_worker_t1_gpu0.sh > "${LOG_T1}" 2>&1 &
PID_T1=$!
echo "T1 Worker (GPU 0) launched with PID: ${PID_T1}"

nohup bash scripts/run_worker_t2_gpu1.sh > "${LOG_T2}" 2>&1 &
PID_T2=$!
echo "T2 Worker (GPU 1) launched with PID: ${PID_T2}"

echo ""
echo "Both workers active in background!"
echo "Monitor T1 on GPU 0 with: tail -f ${LOG_T1}"
echo "Monitor T2 on GPU 1 with: tail -f ${LOG_T2}"
