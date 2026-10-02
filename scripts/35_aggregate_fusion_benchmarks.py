import os
import json
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    base_dir = 'artifacts/experiments/phase_11_fusion_benchmarks'
    res0_p = os.path.join(base_dir, 'results', 'benchmark_results_fold0.json')
    res1_p = os.path.join(base_dir, 'results', 'benchmark_results_fold1.json')

    if not (os.path.exists(res0_p) and os.path.exists(res1_p)):
        console.print("[red]Missing one of the fold results![/red]")
        return

    with open(res0_p) as f:
        r0 = json.load(f)
    with open(res1_p) as f:
        r1 = json.load(f)

    strategies = list(r0.keys())
    summary = {}

    for strat in strategies:
        auc0 = r0[strat]['overall_auc']
        auc1 = r1[strat]['overall_auc']
        mean_auc = (auc0 + auc1) / 2.0

        g_auc0 = r0[strat]['gold_auc']
        g_auc1 = r1[strat]['gold_auc']
        mean_gold = (g_auc0 + g_auc1) / 2.0

        summary[strat] = {
            'fold0_auc': auc0,
            'fold1_auc': auc1,
            'mean_auc': mean_auc,
            'fold0_gold': g_auc0,
            'fold1_gold': g_auc1,
            'mean_gold': mean_gold,
        }

    base_tiny = summary["ConvNeXt-Tiny Standalone"]["mean_auc"]
    base_mean = summary["Uniform Mean Blend (1/3 each)"]["mean_auc"]

    for strat in summary:
        summary[strat]['delta_vs_tiny'] = summary[strat]['mean_auc'] - base_tiny
        summary[strat]['delta_vs_mean_blend'] = summary[strat]['mean_auc'] - base_mean

    # Rich table
    table = Table(title="RSNA Knee Abnormality Detection: 2-Fold Multi-Backbone Ensemble & Fusion Benchmark")
    table.add_column("Fusion Strategy", style="bold cyan")
    table.add_column("Fold 0 AUC", style="white")
    table.add_column("Fold 1 AUC", style="white")
    table.add_column("2-Fold Mean AUC", style="bold green")
    table.add_column("2-Fold Gold AUC", style="bold magenta")
    table.add_column("Δ vs Tiny", style="yellow")
    table.add_column("Δ vs Mean Blend", style="blue")

    md_rows = []
    md_rows.append("| Strategy | Fold 0 AUC | Fold 1 AUC | 2-Fold Mean AUC | 2-Fold Gold AUC | Δ vs Tiny | Δ vs Mean Blend |")
    md_rows.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    sorted_summary = sorted(summary.items(), key=lambda x: x[1]['mean_auc'], reverse=True)
    for strat, d in sorted_summary:
        f0 = d['fold0_auc']
        f1 = d['fold1_auc']
        m_auc = d['mean_auc']
        m_g = d['mean_gold']
        d_tiny = d['delta_vs_tiny']
        d_blend = d['delta_vs_mean_blend']

        table.add_row(
            strat,
            f"{f0:.4f}",
            f"{f1:.4f}",
            f"{m_auc:.4f}",
            f"{m_g:.4f}",
            f"{d_tiny:+.4f}",
            f"{d_blend:+.4f}",
        )
        md_rows.append(f"| **{strat}** | {f0:.4f} | {f1:.4f} | **{m_auc:.4f}** | {m_g:.4f} | {d_tiny:+.4f} | {d_blend:+.4f} |")

    console.print(table)

    # Save summary json and md
    out_json = os.path.join(base_dir, 'results', 'benchmark_summary_2fold.json')
    out_md = os.path.join(base_dir, 'results', 'benchmark_summary_2fold.md')

    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    with open(out_md, 'w') as f:
        f.write("# Multi-Backbone Ensemble & Fusion Benchmark (2-Fold CV)\n\n")
        f.write("\n".join(md_rows) + "\n")

    console.print(f"[bold green]Saved 2-Fold benchmark summary to: {out_json} and {out_md}[/bold green]")

if __name__ == '__main__':
    main()
