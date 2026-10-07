#!/usr/bin/env python3
"""
Generate Persistence Diagrams and Persistence Images for CIFAKE (Real vs. AI Images)
- Ingests raw images from CIFAKE benchmark (data/cifake_test.parquet).
- Computes 2D Cubical Persistent Homology (H0 and H1) via GUDHI.
- Generates 2-channel 32x32 Persistence Images (Adams et al., 2017) using lifetime weighting.
- Exports complete labeled dataset:
    1. data/cifake_persistence_dataset.parquet (structured table with diagrams & image arrays)
    2. data/cifake_persistence_tensors.npz (packed tensors ready for CNN PyTorch DataLoader)
- Generates diagnostic visualization: figures/persistence_images_samples.png
"""

import argparse
import io
import math
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

# Global grid definitions for 32x32 persistence images
RESOLUTION = 32
BANDWIDTH = 0.05
IM_RANGE = (0.0, 1.0, 0.0, 1.0)  # [birth_min, birth_max, pers_min, pers_max]

# Precompute spatial coordinate grid for fast vectorized kernel evaluation
X_COORDS = np.linspace(IM_RANGE[0], IM_RANGE[1], RESOLUTION, dtype=np.float32)
Y_COORDS = np.linspace(IM_RANGE[2], IM_RANGE[3], RESOLUTION, dtype=np.float32)
GRID_X, GRID_Y = np.meshgrid(X_COORDS, Y_COORDS)  # shape: (32, 32)


def extract_image_array(raw_image: Any) -> np.ndarray:
    """Extract normalized grayscale [0.0, 1.0] float32 image array (32x32)."""
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

    img_gray = img.convert("L")
    arr = np.asarray(img_gray, dtype=np.float32) / 255.0
    return arr


def compute_persistence_image(
    diagram_pts: np.ndarray,
    grid_x: np.ndarray = GRID_X,
    grid_y: np.ndarray = GRID_Y,
    bandwidth: float = BANDWIDTH,
) -> np.ndarray:
    """
    Compute 2D persistence image from (N, 2) persistence diagram points [birth, death].
    Uses lifetime weighting w(b, l) = l = (death - birth) as in Adams et al. (2017).
    Returns (RESOLUTION, RESOLUTION) float32 matrix.
    """
    if diagram_pts is None or len(diagram_pts) == 0:
        return np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)

    births = diagram_pts[:, 0].astype(np.float32)
    deaths = diagram_pts[:, 1].astype(np.float32)
    lifetimes = deaths - births

    # Filter strictly positive lifetimes and valid finite values
    valid_mask = (lifetimes > 1e-6) & np.isfinite(births) & np.isfinite(deaths)
    if not np.any(valid_mask):
        return np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)

    b = births[valid_mask]
    l = lifetimes[valid_mask]
    w = l  # linear lifetime weighting

    # Vectorized 2D Gaussian density summation over the grid
    # grid_x: (H, W), b: (N,) -> diff_x: (H, W, N)
    diff_x = grid_x[:, :, np.newaxis] - b[np.newaxis, np.newaxis, :]
    diff_y = grid_y[:, :, np.newaxis] - l[np.newaxis, np.newaxis, :]
    dist_sq = diff_x**2 + diff_y**2

    # Gaussian kernel evaluation
    inv_two_sig_sq = 1.0 / (2.0 * bandwidth**2)
    norm_const = 1.0 / (2.0 * math.pi * bandwidth**2)
    gaussians = norm_const * np.exp(-dist_sq * inv_two_sig_sq)

    # Weighted sum across all persistence points
    pi = np.tensordot(gaussians, w, axes=([2], [0])).astype(np.float32)
    return pi


def process_single_image(args: Tuple[int, Any, int]) -> Dict[str, Any]:
    """Process a single image: compute cubical homology, diagrams, and persistence images."""
    idx, raw_image, label_int = args
    label_name = "real" if label_int == 1 else "ai"

    arr = extract_image_array(raw_image)

    # Cubical complex persistent homology
    cubical = gd.CubicalComplex(top_dimensional_cells=arr)
    pers = cubical.persistence()

    h0_list: List[Tuple[float, float]] = []
    h1_list: List[Tuple[float, float]] = []

    for dim, (b, d) in pers:
        if math.isinf(d):
            continue
        lifetime = d - b
        if lifetime > 1e-6:
            if dim == 0:
                h0_list.append((float(b), float(d)))
            elif dim == 1:
                h1_list.append((float(b), float(d)))

    h0_pts = np.array(h0_list, dtype=np.float32) if h0_list else np.empty((0, 2), dtype=np.float32)
    h1_pts = np.array(h1_list, dtype=np.float32) if h1_list else np.empty((0, 2), dtype=np.float32)

    # Generate persistence images
    pi_h0 = compute_persistence_image(h0_pts)
    pi_h1 = compute_persistence_image(h1_pts)

    return {
        "idx": idx,
        "image_id": f"img_{idx:06d}",
        "label": label_int,
        "label_name": label_name,
        "h0_pts": h0_pts,
        "h1_pts": h1_pts,
        "pi_h0": pi_h0,
        "pi_h1": pi_h1,
    }


def generate_dataset(
    raw_parquet_path: str = "data/cifake_test.parquet",
    output_parquet_path: str = "data/cifake_persistence_dataset.parquet",
    output_tensors_path: str = "data/cifake_persistence_tensors.npz",
    figures_dir: str = "figures",
    sample_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"[*] Reading raw CIFAKE images from: {raw_parquet_path}")
    t0 = time.time()
    df_raw = pd.read_parquet(raw_parquet_path)
    print(f"[+] Loaded {len(df_raw):,} raw images in {time.time() - t0:.2f}s.")

    if sample_size is not None and sample_size < len(df_raw):
        print(f"[*] Subsampling to {sample_size:,} images (stratified)...")
        per_class = sample_size // 2
        sub_dfs = [
            df_raw[df_raw["label"] == lbl].head(per_class)
            for lbl in sorted(df_raw["label"].unique())
        ]
        df_raw = pd.concat(sub_dfs, ignore_index=True)

    total_imgs = len(df_raw)
    tasks = [
        (i, row["image"], int(row["label"]))
        for i, (_, row) in enumerate(df_raw.iterrows())
    ]

    workers = num_workers or max(1, os.cpu_count() or 1)
    print(f"[*] Extracting persistence diagrams and images via {workers} parallel workers...")

    results = []
    t_start = time.time()
    chunksize = max(1, min(100, total_imgs // (workers * 4)))

    with mp.Pool(processes=workers) as pool:
        for res in tqdm(
            pool.imap(process_single_image, tasks, chunksize=chunksize),
            total=total_imgs,
            desc="Generating Persistence Images",
            unit="img",
        ):
            results.append(res)

    elapsed = time.time() - t_start
    print(f"[+] Processed {total_imgs:,} images in {elapsed:.2f}s ({total_imgs / elapsed:.1f} img/s).")

    # Sort results by index to preserve exact original row order
    results.sort(key=lambda r: r["idx"])

    # Prepare packed tensor arrays for PyTorch: shape (N, 2, 32, 32)
    print("[*] Packing tensor arrays for CNN training...")
    tensor_images = np.zeros((total_imgs, 2, RESOLUTION, RESOLUTION), dtype=np.float32)
    tensor_labels = np.zeros(total_imgs, dtype=np.int64)
    tensor_ids = []

    for i, res in enumerate(results):
        tensor_images[i, 0] = res["pi_h0"]
        tensor_images[i, 1] = res["pi_h1"]
        tensor_labels[i] = res["label"]
        tensor_ids.append(res["image_id"])

    print(f"[+] Packed tensor array shape: {tensor_images.shape}, memory: {tensor_images.nbytes / (1024**2):.1f} MB")
    print(f"[*] Saving packed tensors to: {output_tensors_path}")
    np.savez_compressed(
        output_tensors_path,
        images=tensor_images,
        labels=tensor_labels,
        image_ids=np.array(tensor_ids),
    )
    print(f"[+] Saved compressed NPZ ({os.path.getsize(output_tensors_path) / (1024**2):.1f} MB).")

    # Build Parquet table
    print(f"[*] Constructing Parquet labeled dataset table...")
    parquet_records = {
        "image_id": [r["image_id"] for r in results],
        "label": [r["label"] for r in results],
        "label_name": [r["label_name"] for r in results],
        "h0_diagram": [r["h0_pts"].tolist() for r in results],
        "h1_diagram": [r["h1_pts"].tolist() for r in results],
        "h0_persistence_image": [r["pi_h0"].flatten().tolist() for r in results],
        "h1_persistence_image": [r["pi_h1"].flatten().tolist() for r in results],
    }

    schema = pa.schema([
        ("image_id", pa.string()),
        ("label", pa.int64()),
        ("label_name", pa.string()),
        ("h0_diagram", pa.list_(pa.list_(pa.float32()))),
        ("h1_diagram", pa.list_(pa.list_(pa.float32()))),
        ("h0_persistence_image", pa.list_(pa.float32())),
        ("h1_persistence_image", pa.list_(pa.float32())),
    ])

    table = pa.Table.from_pydict(parquet_records, schema=schema)
    print(f"[*] Saving Parquet dataset to: {output_parquet_path}")
    pq.write_table(table, output_parquet_path, compression="zstd")
    print(f"[+] Saved Parquet dataset ({os.path.getsize(output_parquet_path) / (1024**2):.1f} MB).")

    # Generate visual diagnostics: Real vs. AI Persistence Image comparison
    print("[*] Rendering diagnostic visualizations...")
    plot_diagnostic_persistence_images(results, figures_dir)


def plot_diagnostic_persistence_images(results: List[Dict[str, Any]], figures_dir: str) -> None:
    real_indices = [i for i, r in enumerate(results) if r["label"] == 1]
    ai_indices = [i for i, r in enumerate(results) if r["label"] == 0]

    # Mean persistence images across all samples
    real_h0_mean = np.mean([results[i]["pi_h0"] for i in real_indices], axis=0)
    real_h1_mean = np.mean([results[i]["pi_h1"] for i in real_indices], axis=0)
    ai_h0_mean = np.mean([results[i]["pi_h0"] for i in ai_indices], axis=0)
    ai_h1_mean = np.mean([results[i]["pi_h1"] for i in ai_indices], axis=0)

    diff_h0 = ai_h0_mean - real_h0_mean
    diff_h1 = ai_h1_mean - real_h1_mean

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)

    # Extent: [birth_min, birth_max, lifetime_min, lifetime_max]
    extent = [0.0, 1.0, 0.0, 1.0]

    # Row 0: H0 (Connected Components)
    im00 = axes[0, 0].imshow(real_h0_mean, origin="lower", extent=extent, cmap="viridis")
    axes[0, 0].set_title("Real: H0 Mean Persistence Image", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Birth")
    axes[0, 0].set_ylabel("Persistence (Lifetime)")
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].imshow(ai_h0_mean, origin="lower", extent=extent, cmap="viridis")
    axes[0, 1].set_title("AI: H0 Mean Persistence Image", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Birth")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    vlim_h0 = max(abs(diff_h0.min()), abs(diff_h0.max()))
    im02 = axes[0, 2].imshow(diff_h0, origin="lower", extent=extent, cmap="coolwarm", vmin=-vlim_h0, vmax=vlim_h0)
    axes[0, 2].set_title("Difference H0 (AI - Real)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Birth")
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Sample H0 diagram overlay
    sample_real_idx = real_indices[0]
    sample_ai_idx = ai_indices[0]
    axes[0, 3].scatter(
        results[sample_real_idx]["h0_pts"][:, 0],
        results[sample_real_idx]["h0_pts"][:, 1] - results[sample_real_idx]["h0_pts"][:, 0],
        color="#2980b9", alpha=0.7, s=25, label="Real Sample"
    )
    axes[0, 3].scatter(
        results[sample_ai_idx]["h0_pts"][:, 0],
        results[sample_ai_idx]["h0_pts"][:, 1] - results[sample_ai_idx]["h0_pts"][:, 0],
        color="#e74c3c", alpha=0.7, s=25, marker="^", label="AI Sample"
    )
    axes[0, 3].set_xlim([0.0, 1.0])
    axes[0, 3].set_ylim([0.0, 1.0])
    axes[0, 3].set_title("Sample H0 Diagrams (Birth vs. Pers)", fontsize=11, fontweight="bold")
    axes[0, 3].set_xlabel("Birth")
    axes[0, 3].legend(loc="upper right")
    axes[0, 3].grid(True, linestyle=":", alpha=0.5)

    # Row 1: H1 (Topological Loops)
    im10 = axes[1, 0].imshow(real_h1_mean, origin="lower", extent=extent, cmap="plasma")
    axes[1, 0].set_title("Real: H1 Mean Persistence Image", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Birth")
    axes[1, 0].set_ylabel("Persistence (Lifetime)")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im11 = axes[1, 1].imshow(ai_h1_mean, origin="lower", extent=extent, cmap="plasma")
    axes[1, 1].set_title("AI: H1 Mean Persistence Image", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Birth")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    vlim_h1 = max(abs(diff_h1.min()), abs(diff_h1.max()))
    im12 = axes[1, 2].imshow(diff_h1, origin="lower", extent=extent, cmap="coolwarm", vmin=-vlim_h1, vmax=vlim_h1)
    axes[1, 2].set_title("Difference H1 (AI - Real)", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Birth")
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Sample H1 diagram overlay
    axes[1, 3].scatter(
        results[sample_real_idx]["h1_pts"][:, 0],
        results[sample_real_idx]["h1_pts"][:, 1] - results[sample_real_idx]["h1_pts"][:, 0],
        color="#2980b9", alpha=0.7, s=25, label="Real Sample"
    )
    axes[1, 3].scatter(
        results[sample_ai_idx]["h1_pts"][:, 0],
        results[sample_ai_idx]["h1_pts"][:, 1] - results[sample_ai_idx]["h1_pts"][:, 0],
        color="#e74c3c", alpha=0.7, s=25, marker="^", label="AI Sample"
    )
    axes[1, 3].set_xlim([0.0, 1.0])
    axes[1, 3].set_ylim([0.0, 1.0])
    axes[1, 3].set_title("Sample H1 Diagrams (Birth vs. Pers)", fontsize=11, fontweight="bold")
    axes[1, 3].set_xlabel("Birth")
    axes[1, 3].legend(loc="upper right")
    axes[1, 3].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle(
        "Persistence Image Analysis: Real Photographs vs. AI-Generated Synthetic Images\n"
        "(20,000 CIFAKE Benchmark Images, H0 Components and H1 Loops, 32x32 Resolution)",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    out_fig = os.path.join(figures_dir, "persistence_images_samples.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[+] Saved diagnostic plot to: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Compute Persistence Diagrams and Persistence Images for CIFAKE")
    parser.add_argument("--raw-parquet", type=str, default="data/cifake_test.parquet")
    parser.add_argument("--output-parquet", type=str, default="data/cifake_persistence_dataset.parquet")
    parser.add_argument("--output-tensors", type=str, default="data/cifake_persistence_tensors.npz")
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()
    generate_dataset(
        raw_parquet_path=args.raw_parquet,
        output_parquet_path=args.output_parquet,
        output_tensors_path=args.output_tensors,
        figures_dir=args.figures_dir,
        sample_size=args.sample_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
