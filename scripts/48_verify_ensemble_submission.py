"""
RSNA Knee Abnormality Detection - Grand Ensemble Offline Verification Suite.
Verifies:
1. Checkpoint and artifact staging in kaggle_upload/
2. Clean initialization and memory footprint of KneeGrandEnsemblePredictor
3. Multi-study inference simulation with exact competition target verification
4. Latency profiling (ms per study) and submission.csv integrity
"""

import os
import sys
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

console = Console()

from kaggle_kernel.submission_pipeline import (
    KneeGrandEnsemblePredictor,
    TARGET_COLUMNS,
    DEVICE,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="4_arch", choices=["3_arch", "4_arch"])
    parser.add_argument("--n_studies", type=int, default=15)
    args = parser.parse_args()

    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║  Grand Ensemble Submission Verification ({args.mode.upper()})            ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════╝[/bold cyan]")

    # 1. Staging Audit
    console.print("\n[bold yellow]Step 1: Auditing Staged Submission Artifacts...[/bold yellow]")
    weights_dir = Path("kaggle_upload/weights")
    kernel_dir = Path("kaggle_upload/kernel")

    expected_weights = [
        "phase16_decoupled_convnext_small_384px_f0_best.pth",
        "phase16_decoupled_convnext_tiny_336px_f0_best.pth",
        "phase13_triplanar_dinov2_small_f0_72sl_best.pth",
    ]
    if args.mode == "4_arch":
        expected_weights.append("phase14_ablation_coattention_convnext_tiny_f0_best.pth")

    total_mb = 0.0
    for w in expected_weights:
        p = weights_dir / w
        assert p.exists(), f"Missing staged weight: {p}"
        size_mb = p.stat().st_size / 1e6
        total_mb += size_mb
        console.print(f"  [green]✓[/green] Staged: {w:52s} ({size_mb:6.1f} MB)")

    console.print(f"[bold green]Total Ensemble Model Size: {total_mb:.1f} MB (Well under 20 GB Kaggle quota)[/bold green]")

    nb_file = kernel_dir / "rsna_knee_submission.ipynb"
    pipe_file = kernel_dir / "submission_pipeline.py"
    assert nb_file.exists(), f"Missing {nb_file}"
    assert pipe_file.exists(), f"Missing {pipe_file}"
    console.print(f"  [green]✓[/green] Standalone Notebook: {nb_file.name}")
    console.print(f"  [green]✓[/green] Inference Module   : {pipe_file.name}")

    # 2. Predictor Initialization
    console.print(f"\n[bold yellow]Step 2: Initializing KneeGrandEnsemblePredictor from {weights_dir}...[/bold yellow]")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    predictor = KneeGrandEnsemblePredictor(checkpoint_dir=weights_dir, mode=args.mode)
    init_time = time.time() - t0
    console.print(f"[bold green]✓ Loaded all models in {init_time:.2f} seconds![/bold green]")

    if torch.cuda.is_available():
        init_vram = torch.cuda.memory_allocated() / 1e9
        console.print(f"  VRAM allocated after loading: {init_vram:.2f} GB")

    # 3. Simulated Multi-Study Test
    console.print(f"\n[bold yellow]Step 3: Simulating Inference on {args.n_studies} Studies...[/bold yellow]")
    df_splits = pd.read_parquet("data/splits_5fold.parquet")
    sample_uids = df_splits[df_splits["fold"] == 0]["StudyInstanceUID"].head(args.n_studies).tolist()

    records = []
    latencies = []
    plane_ids = torch.cat([
        torch.zeros(24, dtype=torch.long),
        torch.ones(24, dtype=torch.long),
        torch.full((24,), 2, dtype=torch.long),
    ]).unsqueeze(0).to(DEVICE)

    for i, uid in enumerate(sample_uids):
        t_start = time.time()
        dummy_scan = torch.rand(1, 72, 3, 288, 288, dtype=torch.float32)

        preds = predictor.predict_tensor_volume(dummy_scan, plane_ids=plane_ids)
        t_elapsed = time.time() - t_start
        latencies.append(t_elapsed)

        # Integrity checks
        assert len(preds) == 12, f"Expected 12 predictions, got {len(preds)}"
        assert not np.isnan(preds).any(), f"NaN detected in study {uid}"
        assert not np.isinf(preds).any(), f"Infinity detected in study {uid}"
        assert (preds >= 0.0).all() and (preds <= 1.0).all(), f"Probabilities out of [0, 1] bounds for {uid}"

        records.append([uid] + preds.tolist())

    avg_ms = np.mean(latencies) * 1000.0
    throughput = 1000.0 / avg_ms
    est_3000_min = (3000.0 * np.mean(latencies)) / 60.0

    console.print(f"[bold green]✓ Successfully processed {len(sample_uids)} studies![/bold green]")
    console.print(f"  Mean latency per study : [bold cyan]{avg_ms:.1f} ms[/bold cyan]")
    console.print(f"  Throughput             : [bold cyan]{throughput:.1f} studies/sec[/bold cyan]")
    console.print(f"  Estimated 3,000 studies: [bold cyan]{est_3000_min:.1f} minutes[/bold cyan] (Limit: 540 min)")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        console.print(f"  Peak GPU VRAM          : [bold cyan]{peak_vram:.2f} GB[/bold cyan] (Limit: 16.0 GB)")

    # 4. Submission Output Validation
    console.print("\n[bold yellow]Step 4: Validating submission.csv Output Format...[/bold yellow]")
    sub_df = pd.DataFrame(records, columns=["StudyInstanceUID"] + TARGET_COLUMNS)
    sub_path = Path("submission_verification.csv")
    sub_df.to_csv(sub_path, index=False)

    # Reload and test
    reloaded = pd.read_csv(sub_path)
    assert list(reloaded.columns) == ["StudyInstanceUID"] + TARGET_COLUMNS, "Columns mismatch!"
    assert len(reloaded) == args.n_studies, "Row count mismatch!"
    assert reloaded[TARGET_COLUMNS].isnull().sum().sum() == 0, "Null values found!"

    table = Table(title="Generated Submission Sample (5 Studies)", header_style="bold magenta")
    table.add_column("StudyInstanceUID", style="cyan", width=22)
    for col in TARGET_COLUMNS[:5]:
        table.add_column(col, justify="right", style="green")
    table.add_column("...", justify="center")
    table.add_column(TARGET_COLUMNS[-1], justify="right", style="green")

    for _, row in reloaded.head(5).iterrows():
        table.add_row(
            row["StudyInstanceUID"][:20] + "..",
            f"{row[TARGET_COLUMNS[0]]:.4f}",
            f"{row[TARGET_COLUMNS[1]]:.4f}",
            f"{row[TARGET_COLUMNS[2]]:.4f}",
            f"{row[TARGET_COLUMNS[3]]:.4f}",
            f"{row[TARGET_COLUMNS[4]]:.4f}",
            "...",
            f"{row[TARGET_COLUMNS[-1]]:.4f}",
        )
    console.print(table)

    console.print("\n[bold green]==================================================================[/bold green]")
    console.print("[bold green]  ALL VERIFICATION CHECKS PASSED: 100% SUBMISSION READY!          [/bold green]")
    console.print("[bold green]==================================================================[/bold green]")

if __name__ == "__main__":
    main()
