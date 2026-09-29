"""
RSNA Knee Abnormality Detection - Multi-Process DICOM Preprocessing & Caching.
Processes all DICOM series into spatially-sorted, intensity-normalized, 2.5D slice tensors
and saves them to high-speed disk storage with a Parquet and CSV index manifest.
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.preprocessing import process_single_series
from src.metrics.auc_metrics import TARGET_COLUMNS


def process_and_save_series(args_tuple):
    """
    Worker function to process a single series and save as uint8 .npy tensor.
    """
    study_uid, series_uid, series_dir, out_dir, target_size = args_tuple
    save_dir = os.path.join(out_dir, study_uid)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{series_uid}.npy")

    # Skip if already processed and valid
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        try:
            arr = np.load(save_path, mmap_mode="r")
            return {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "num_slices": arr.shape[0],
                "cache_path": save_path,
                "status": "cached",
            }
        except Exception:
            pass

    try:
        res = process_single_series(
            series_dir=series_dir,
            target_size=(target_size, target_size),
            normalize="minmax",
            stack_25d=True,
        )
        volume = res["volume"]  # (D, 3, H, W) in [0, 1]
        
        # Save as uint8 [0, 255] for 4x memory savings & ultra-fast disk reads
        volume_uint8 = (volume * 255.0).clip(0, 255).astype(np.uint8)
        np.save(save_path, volume_uint8)

        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "num_slices": res["num_slices"],
            "orig_shape": str(res["meta"]["orig_shape"]),
            "cache_path": save_path,
            "status": "success",
        }
    except Exception as e:
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "error": str(e),
            "status": "failed",
        }


def run_preprocessing(
    data_dir: str = "../Datasets/rsna-knee-abnormality-detection",
    output_dir: str = "./data/preprocessed_256",
    target_size: int = 256,
    labeled_only: bool = False,
    max_studies: int = None,
    num_workers: int = 8,
):
    print("=== RSNA Knee: Multi-Process Volume Preprocessor & Caching ===")
    print(f"Source Data Directory: {data_dir}")
    print(f"Output Cache Directory: {output_dir}")
    print(f"Target In-Plane Resolution: {target_size}x{target_size}")
    print(f"Workers: {num_workers}")

    # Support Parquet or CSV for train and train_series metadata
    train_parquet = os.path.join(data_dir, "train.parquet")
    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_parquet(train_parquet) if os.path.exists(train_parquet) else pd.read_csv(train_csv)

    train_series_parquet = os.path.join(data_dir, "train_series.parquet")
    train_series_csv = os.path.join(data_dir, "train_series.csv")
    train_series_df = pd.read_parquet(train_series_parquet) if os.path.exists(train_series_parquet) else pd.read_csv(train_series_csv)

    if labeled_only:
        labeled_mask = train_df[TARGET_COLUMNS].notnull().any(axis=1)
        train_df = train_df[labeled_mask].copy()
        print(f"Filtering to Ground-Truth Labeled Studies Only (N={len(train_df)})")

    if max_studies is not None:
        train_df = train_df.iloc[:max_studies].copy()
        print(f"Limiting to first {len(train_df)} studies.")

    # Filter series to matching target studies
    target_studies = set(train_df["StudyInstanceUID"])
    series_subset = train_series_df[train_series_df["StudyInstanceUID"].isin(target_studies)].copy()
    print(f"Total Studies to Process: {len(target_studies):,}")
    print(f"Total Series to Process:  {len(series_subset):,}")

    tasks = []
    raw_series_base = os.path.join(data_dir, "train_series")
    for _, row in series_subset.iterrows():
        s_uid = row["StudyInstanceUID"]
        ser_uid = row["SeriesInstanceUID"]
        ser_dir = os.path.join(raw_series_base, s_uid, ser_uid)
        tasks.append((s_uid, ser_uid, ser_dir, output_dir, target_size))

    # Run multi-process extraction
    records = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_and_save_series, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Preprocessing Volumes"):
            res = future.result()
            if res["status"] == "failed":
                failed_count += 1
            else:
                records.append(res)

    print(f"\n✓ Completed: {len(records):,} series successfully preprocessed.")
    if failed_count > 0:
        print(f"⚠️ Failed: {failed_count} series.")

    # Build manifest dataframe
    df_manifest = pd.DataFrame(records)
    merged_index = pd.merge(series_subset, df_manifest, on=["StudyInstanceUID", "SeriesInstanceUID"], how="inner")
    
    # Merge label targets
    merged_index = pd.merge(merged_index, train_df, on="StudyInstanceUID", how="left")

    manifest_parquet = os.path.join("data", "preprocessed_index.parquet")
    manifest_csv = os.path.join("data", "preprocessed_index.csv")
    os.makedirs(os.path.dirname(manifest_parquet), exist_ok=True)

    # Save both Parquet (primary fast format) and CSV (inspection)
    merged_index.to_parquet(manifest_parquet, engine="pyarrow", compression="snappy", index=False)
    merged_index.to_csv(manifest_csv, index=False)
    print(f"✓ Saved preprocessed index manifest to: {manifest_parquet} and {manifest_csv}")


if __name__ == "__main__":
    from src.config import resolve_data_dir, resolve_cache_dir

    default_data_dir = str(resolve_data_dir()) if hasattr(resolve_data_dir, "__call__") else "../Datasets/rsna-knee-abnormality-detection"
    default_cache_dir = str(resolve_cache_dir()) if hasattr(resolve_cache_dir, "__call__") else "./data/cached_series_384"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="RSNA Knee dataset directory (default: from config)")
    parser.add_argument("--output_dir", type=str, default=default_cache_dir, help="Output cache directory (default: from config)")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--labeled_only", action="store_true", help="Process only the 58 labeled studies")
    parser.add_argument("--max_studies", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    run_preprocessing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        labeled_only=args.labeled_only,
        max_studies=args.max_studies,
        num_workers=args.num_workers,
    )
