"""
RSNA Knee Abnormality Detection - Master Foundation Model Benchmarking Runner.
Sequentially extracts frozen slice features and evaluates 5-fold CV MIL heads.
Generates master benchmark leaderboard.
"""

import os
import subprocess
import sys
import time

MODELS = [
    "biomedclip",
    "dinov2_small",
    "dinov2",
    "siglip",
]

PYTHON = sys.executable

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"⚠️ Command failed with exit code {ret.returncode}")

def main():
    print("======================================================================")
    print("🔬 STARTING FOUNDATION MODEL BENCHMARKING SUITE")
    print("======================================================================")

    # 1. Feature Extraction
    for m in MODELS:
        print(f"\n[1/2] Extracting embeddings for: {m}")
        run_cmd(f"{PYTHON} scripts/19_extract_foundation_embeddings.py --model {m}")

    # 2. Individual Model Benchmarks
    for m in MODELS:
        print(f"\n[2/2] Benchmarking MIL Head for: {m}")
        run_cmd(f"{PYTHON} scripts/20_benchmark_foundation_mil.py --models {m}")

    # 3. Multi-Model Embedding Concat / Fusion Benchmarks
    print(f"\n[Bonus] Benchmarking Joint Embedding Fusion: BioMedCLIP + DINOv2 + SigLIP")
    run_cmd(f"{PYTHON} scripts/20_benchmark_foundation_mil.py --models biomedclip dinov2 siglip")

    print("\n🎉 Master Foundation Benchmarking Completed!")

if __name__ == "__main__":
    main()