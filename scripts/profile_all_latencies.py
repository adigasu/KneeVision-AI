import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import time
import json
import torch
from src.benchmarks.foundation_extractors import get_foundation_extractor
from scripts.fast_vectorized_benchmark import StandaloneMILHead

device = "cuda:0"

def benchmark(name):
    extractor = get_foundation_extractor(name, device=device)
    params = sum(p.numel() for p in extractor.parameters()) / 1e6
    img_size = getattr(extractor, "img_size", 224)
    dim = extractor.embed_dim
    mil = StandaloneMILHead(in_features=dim).to(device).eval()

    dummy_1 = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_16 = torch.randn(1, 16, 3, img_size, img_size, device=device)
    dummy_48 = torch.randn(1, 48, 3, img_size, img_size, device=device)

    # Warmup
    for _ in range(10):
        _ = extractor.extract_slices(dummy_1)
        feats = extractor.extract_slices(dummy_16)
        _ = mil(feats)
    torch.cuda.synchronize()

    # Benchmark 16-slice
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(30):
        feats = extractor.extract_slices(dummy_16)
        _ = mil(feats)
    e.record()
    torch.cuda.synchronize()
    t16 = s.elapsed_time(e) / 30

    # Benchmark 48-slice
    s.record()
    for _ in range(30):
        feats = extractor.extract_slices(dummy_48)
        _ = mil(feats)
    e.record()
    torch.cuda.synchronize()
    t48 = s.elapsed_time(e) / 30

    # Benchmark 1 slice
    s.record()
    for _ in range(30):
        _ = extractor.extract_slices(dummy_1)
    e.record()
    torch.cuda.synchronize()
    t1 = s.elapsed_time(e) / 30

    del extractor, mil
    torch.cuda.empty_cache()

    return {
        "params_m": round(params, 1),
        "embed_dim": dim,
        "img_size": img_size,
        "slice_ms": round(t1, 2),
        "study_16_ms": round(t16, 2),
        "study_48_ms": round(t48, 2),
        "throughput_16_sec": round(1000.0 / t16, 1) if t16 > 0 else 0,
    }

models = [
    "convnext_tiny",
    "convnext_small",
    "dinov2_small",
    "dinov2",
    "dinov3",
    "biomedclip",
    "siglip",
    "medsiglip",
    "radimagenet",
]

results = {}
for m in models:
    try:
        res = benchmark(m)
        results[m] = res
        print(f"[{m:16s}] Params: {res['params_m']:5.1f}M | Dim: {res['embed_dim']:4d} | 16-Sl: {res['study_16_ms']:6.2f} ms ({res['throughput_16_sec']:4.1f} std/s) | 48-Sl: {res['study_48_ms']:6.2f} ms | Slice: {res['slice_ms']:5.2f} ms")
    except Exception as e:
        print(f"[{m:16s}] Error: {e}")

with open("artifacts/experiments/benchmark_foundation_models/results/all_latency_profiles.json", "w") as f:
    json.dump(results, f, indent=2)
