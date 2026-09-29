import json

with open("artifacts/experiments/benchmark_foundation_models/results/complete_foundation_benchmark.json") as f:
    d = json.load(f)

print("="*105)
print(f"{'Model / Foundation Backbone':<40} | {'Dim':<6} | {'5-Fold Val Macro AUC':<22} | {'Gold Set AUC (N=58)':<22}")
print("="*105)
for k, v in d.get("results", {}).items():
    name = v.get("name")
    dim = v.get("feature_dim")
    val_m = v.get("mean_val_auc")
    val_s = v.get("std_val_auc")
    gold_m = v.get("mean_gold_auc")
    gold_s = v.get("std_gold_auc")
    print(f"{name:<40} | {dim:<6} | {val_m:.4f} ± {val_s:.4f}        | {gold_m:.4f} ± {gold_s:.4f}")

print("\n" + "="*105)
print(f"{'Model Architecture':<22} | {'Params':<8} | {'Slice Latency':<15} | {'16-Slice Study':<18} | {'48-Slice Study':<18}")
print("="*105)
for k, v in d.get("latency", {}).items():
    if "error" in v:
        continue
    params = f"{v.get('param_count_m', 0):.1f}M"
    sl = f"{v.get('slice_latency_ms', 0):.2f} ms"
    st16 = f"{v.get('study_16_ms', 0):.2f} ms ({v.get('throughput_16_studies_sec', 0):.1f}/s)"
    st48 = f"{v.get('study_48_ms', 0):.2f} ms ({v.get('throughput_48_studies_sec', 0):.1f}/s)"
    print(f"{k:<22} | {params:<8} | {sl:<15} | {st16:<18} | {st48:<18}")
print("="*105)
