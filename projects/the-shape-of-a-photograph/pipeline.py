#!/usr/bin/env python3
"""
The Shape of a Photograph: Persistent Homology of Real vs. AI-Generated Images
Pipeline for CIFAKE Dataset:
1. Ingestion: Download / load CIFAKE benchmark images (REAL vs. AI-generated / FAKE)
2. Grayscale Conversion: Standard luma transform to [0.0, 1.0] intensity matrix
3. Cubical Persistent Homology: Compute H0 and H1 persistence via gudhi.CubicalComplex
4. Topological Summary Statistics: Feature counts, total persistence, max persistence, persistence entropy, moments
5. Labeled Dataset Export: Output structured CSV and Parquet files with image labels
"""

import argparse
import io
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import gudhi as gd
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


DEFAULT_DATA_URL = (
    "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/"
    "resolve/main/data/test-00000-of-00001.parquet"
)


def download_dataset(url: str, output_path: str) -> None:
    """Download dataset file if not already present."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f"Dataset already cached at {output_path} ({os.path.getsize(output_path):,} bytes).")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.tmp"
    print(f"Downloading dataset from:\n  {url}\nto:\n  {output_path} ...")
    start = time.time()

    cmd = ["curl", "-L", "-f", "--progress-bar", "-o", temp_path, url]
    res = subprocess.run(cmd)
    if res.returncode != 0 or not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download dataset from {url}")

    os.rename(temp_path, output_path)
    elapsed = time.time() - start
    print(f"Downloaded {os.path.getsize(output_path):,} bytes in {elapsed:.2f}s.")


def extract_image_array(raw_image: Any) -> np.ndarray:
    """
    Extract grayscale normalized float32 image array from raw cell.
    Returns 2D np.ndarray of shape (H, W) with values in [0.0, 1.0].
    """
    if isinstance(raw_image, dict):
        if "bytes" in raw_image and raw_image["bytes"] is not None:
            img = Image.open(io.BytesIO(raw_image["bytes"]))
        elif "path" in raw_image and raw_image["path"]:
            img = Image.open(raw_image["path"])
        else:
            raise ValueError(f"Unknown dict image structure: {raw_image.keys()}")
    elif isinstance(raw_image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(raw_image))
    elif isinstance(raw_image, Image.Image):
        img = raw_image
    elif isinstance(raw_image, np.ndarray):
        if raw_image.ndim == 3:
            img = Image.fromarray(raw_image.astype(np.uint8))
        else:
            return raw_image.astype(np.float32) / (255.0 if raw_image.max() > 1.0 else 1.0)
    else:
        raise TypeError(f"Unsupported image type: {type(raw_image)}")

    # Step 2: Convert to grayscale (ITU-R 601-2 luma)
    img_gray = img.convert("L")
    arr = np.asarray(img_gray, dtype=np.float32) / 255.0
    return arr


def compute_persistence_entropy(lifetimes: np.ndarray) -> float:
    """
    Compute Shannon persistence entropy in bits:
    E = -sum(p_i * log2(p_i)) where p_i = l_i / sum(l)
    """
    if len(lifetimes) == 0:
        return 0.0
    total_l = np.sum(lifetimes)
    if total_l <= 1e-12:
        return 0.0
    probs = lifetimes / total_l
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_topological_summary(
    image_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Step 3 & 4: Compute Cubical Persistent Homology on pixel intensities
    and extract summary statistics.
    """
    # 1. Pixel baseline metrics
    pixel_mean = float(np.mean(image_matrix))
    pixel_std = float(np.std(image_matrix))
    pixel_min = float(np.min(image_matrix))
    pixel_max = float(np.max(image_matrix))

    # 2. Cubical complex sublevel set filtration
    cubical = gd.CubicalComplex(top_dimensional_cells=image_matrix)
    pers = cubical.persistence()

    # Extract finite lifetimes for H0 and H1
    h0_lifetimes: List[float] = []
    h1_lifetimes: List[float] = []

    for dim, (birth, death) in pers:
        # Note: death can be inf for the essential component in H0
        if math.isinf(death):
            continue
        lifetime = death - birth
        if lifetime > 0:
            if dim == 0:
                h0_lifetimes.append(lifetime)
            elif dim == 1:
                h1_lifetimes.append(lifetime)

    h0_arr = np.array(h0_lifetimes, dtype=np.float64)
    h1_arr = np.array(h1_lifetimes, dtype=np.float64)
    all_arr = np.concatenate([h0_arr, h1_arr]) if (len(h0_arr) + len(h1_arr)) > 0 else np.array([], dtype=np.float64)

    # Feature counts
    h0_count = int(len(h0_arr))
    h1_count = int(len(h1_arr))
    total_feature_count = h0_count + h1_count

    # Total persistence
    h0_total_persistence = float(np.sum(h0_arr)) if h0_count > 0 else 0.0
    h1_total_persistence = float(np.sum(h1_arr)) if h1_count > 0 else 0.0
    total_persistence = h0_total_persistence + h1_total_persistence

    # Max persistence
    h0_max_persistence = float(np.max(h0_arr)) if h0_count > 0 else 0.0
    h1_max_persistence = float(np.max(h1_arr)) if h1_count > 0 else 0.0
    max_persistence = max(h0_max_persistence, h1_max_persistence)

    # Persistence entropy
    h0_entropy = compute_persistence_entropy(h0_arr)
    h1_entropy = compute_persistence_entropy(h1_arr)
    total_entropy = compute_persistence_entropy(all_arr)

    # Distribution moments
    h0_mean_persistence = float(np.mean(h0_arr)) if h0_count > 0 else 0.0
    h1_mean_persistence = float(np.mean(h1_arr)) if h1_count > 0 else 0.0
    h0_std_persistence = float(np.std(h0_arr)) if h0_count > 0 else 0.0
    h1_std_persistence = float(np.std(h1_arr)) if h1_count > 0 else 0.0

    return {
        "h0_count": h0_count,
        "h1_count": h1_count,
        "total_feature_count": total_feature_count,
        "h0_total_persistence": h0_total_persistence,
        "h1_total_persistence": h1_total_persistence,
        "total_persistence": total_persistence,
        "h0_max_persistence": h0_max_persistence,
        "h1_max_persistence": h1_max_persistence,
        "max_persistence": max_persistence,
        "h0_entropy": h0_entropy,
        "h1_entropy": h1_entropy,
        "total_entropy": total_entropy,
        "h0_mean_persistence": h0_mean_persistence,
        "h1_mean_persistence": h1_mean_persistence,
        "h0_std_persistence": h0_std_persistence,
        "h1_std_persistence": h1_std_persistence,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
    }


def _worker_process_row(args: Tuple[int, Any, int]) -> Dict[str, Any]:
    """Worker task processing a single image row."""
    idx, raw_image, label_int = args
    label_name = "real" if label_int == 1 else "ai"
    try:
        arr = extract_image_array(raw_image)
        stats = compute_topological_summary(arr)
        res = {
            "image_id": f"img_{idx:06d}",
            "label": label_int,
            "label_name": label_name,
            **stats,
        }
        return res
    except Exception as e:
        print(f"Error processing image {idx}: {e}", file=sys.stderr)
        raise


def run_pipeline(
    data_url: str = DEFAULT_DATA_URL,
    raw_parquet: str = "data/cifake_test.parquet",
    output_dir: str = "data",
    sample_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """Execute complete CIFAKE Persistent Homology pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download / locate CIFAKE parquet
    download_dataset(data_url, raw_parquet)

    print(f"Reading CIFAKE data from {raw_parquet}...")
    t0 = time.time()
    df_raw = pd.read_parquet(raw_parquet)
    print(f"Loaded {len(df_raw):,} raw images with columns: {list(df_raw.columns)} in {time.time() - t0:.2f}s")

    # Sample if requested
    if sample_size is not None and sample_size < len(df_raw):
        # Stratified sampling across labels (label: 0=FAKE, 1=REAL)
        print(f"Stratified subsampling to {sample_size:,} images...")
        per_class = sample_size // 2
        sub_dfs = [
            df_raw[df_raw["label"] == lbl].head(per_class)
            for lbl in sorted(df_raw["label"].unique())
        ]
        df_raw = pd.concat(sub_dfs, ignore_index=True)
        print(f"Subsampled dataset: {len(df_raw):,} images ({df_raw['label'].value_counts().to_dict()})")

    # Prepare inputs for parallel processing
    total_imgs = len(df_raw)
    tasks = [
        (i, row["image"], int(row["label"]))
        for i, (_, row) in enumerate(df_raw.iterrows())
    ]

    workers = num_workers or max(1, os.cpu_count() or 1)
    print(f"Processing {total_imgs:,} images via {workers} parallel workers...")

    results: List[Dict[str, Any]] = []
    start_compute = time.time()

    if workers > 1:
        # Use chunksize for high throughput
        chunksize = max(1, min(100, total_imgs // (workers * 4)))
        with mp.Pool(processes=workers) as pool:
            for res in tqdm(
                pool.imap(_worker_process_row, tasks, chunksize=chunksize),
                total=total_imgs,
                desc="Computing persistent homology",
                unit="img",
            ):
                results.append(res)
    else:
        for item in tqdm(tasks, desc="Computing persistent homology", unit="img"):
            results.append(_worker_process_row(item))

    compute_duration = time.time() - start_compute
    throughput = total_imgs / compute_duration if compute_duration > 0 else 0
    print(f"Processed {total_imgs:,} images in {compute_duration:.2f}s ({throughput:.1f} images/sec).")

    # Step 6: Create labeled dataset
    df_out = pd.DataFrame(results)

    # Order columns logically
    first_cols = ["image_id", "label", "label_name"]
    stat_cols = [c for c in df_out.columns if c not in first_cols]
    df_out = df_out[first_cols + stat_cols]

    csv_path = os.path.join(output_dir, "cifake_topological_dataset.csv")
    parquet_path = os.path.join(output_dir, "cifake_topological_dataset.parquet")

    print(f"Saving dataset to CSV: {csv_path} ...")
    df_out.to_csv(csv_path, index=False)

    print(f"Saving dataset to Parquet: {parquet_path} ...")
    df_out.to_parquet(parquet_path, index=False)

    print(f"Pipeline successfully completed!")
    print(f"Dataset shape: {df_out.shape[0]:,} rows x {df_out.shape[1]} columns")
    print(f"Label distribution:\n{df_out['label_name'].value_counts().to_string()}")

    return df_out, csv_path, parquet_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cubical Persistent Homology on CIFAKE Real vs. AI Images"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        default=DEFAULT_DATA_URL,
        help="URL to CIFAKE parquet file",
    )
    parser.add_argument(
        "--raw-parquet",
        type=str,
        default="data/cifake_test.parquet",
        help="Local path to save/load raw CIFAKE parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save output dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of images to process (stratified by class)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: all CPU cores)",
    )

    args = parser.parse_args()
    run_pipeline(
        data_url=args.data_url,
        raw_parquet=args.raw_parquet,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
