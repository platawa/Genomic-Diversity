#!/usr/bin/env python3
"""
normalize_sae_features.py

Apply genome-wide normalization to a single chromosome's SAE max-pooled vectors
using global statistics from aggregate_genome_sae_stats.py.

Normalization methods:
  zscore:  z_j = (x_j - mean_j) / std_j
  minmax:  x_norm = (x - min) / (max - min)   → scales to [0, 1]
  robust:  x_norm = (x - median) / IQR         → robust to outliers

Usage:
    python tools/normalize_sae_features.py \\
        --chrom chr22 \\
        --results_dir results/ \\
        --auto

    python tools/normalize_sae_features.py \\
        --chrom chr22 \\
        --results_dir results/ \\
        --stats_file results/_genome_sae_stats/<run>/data/global_feature_stats.npz \\
        --method robust
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import find_latest_completed, find_latest_completed_global

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def normalize_zscore(vectors, stats):
    """Z-score normalization: (x - mean) / std.

    Uses per-nucleotide genome-wide stats (nuc_mean/nuc_std) if available,
    falling back to region-max-pool stats (mean/std) otherwise.
    """
    if "nuc_mean" in stats and "nuc_std" in stats:
        mean = stats["nuc_mean"]
        std = stats["nuc_std"]
        valid = std > 0
        logger.info("Using per-nucleotide genome-wide stats for z-score")
    else:
        mean = stats["mean"]
        std = stats["std"]
        valid = stats["valid_mask"]
        logger.info("Using region-max-pool stats for z-score (nuc stats not available)")

    normalized = np.zeros_like(vectors)
    normalized[:, valid] = (vectors[:, valid] - mean[valid]) / std[valid]
    return normalized


def normalize_minmax(vectors, stats):
    """Min-max normalization: (x - min) / (max - min) → [0, 1]."""
    feat_min = stats["min"]
    feat_max = stats["max"]
    span = feat_max - feat_min
    valid = span > 0

    normalized = np.zeros_like(vectors)
    normalized[:, valid] = (vectors[:, valid] - feat_min[valid]) / span[valid]
    return normalized


def normalize_robust(vectors, stats):
    """Robust normalization: (x - median) / IQR."""
    median = stats["median"]
    iqr = stats["q75"] - stats["q25"]
    valid = iqr > 0

    normalized = np.zeros_like(vectors)
    normalized[:, valid] = (vectors[:, valid] - median[valid]) / iqr[valid]
    return normalized


METHODS = {
    "zscore": normalize_zscore,
    "minmax": normalize_minmax,
    "robust": normalize_robust,
}


def main():
    parser = argparse.ArgumentParser(
        description="Apply genome-wide normalization to per-chromosome SAE features",
    )
    parser.add_argument("--chrom", required=True,
                        help="Chromosome name (e.g. chr22)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--stats_file", type=str, default=None,
                        help="Path to global_feature_stats.npz (overrides --auto)")
    parser.add_argument("--method", type=str, default="zscore",
                        choices=list(METHODS.keys()),
                        help="Normalization method (default: zscore)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover latest genome stats and SAE run")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    global logger
    logger = setup_logging(args.log_level)

    results_dir = os.path.abspath(args.results_dir)

    logger.info("=" * 70)
    logger.info("SAE Feature Normalization")
    logger.info("=" * 70)

    # ── Find global stats ────────────────────────────────────────────────────
    if args.stats_file:
        stats_path = os.path.abspath(args.stats_file)
    elif args.auto:
        genome_run = find_latest_completed_global(results_dir, "_genome_sae_stats")
        if genome_run is None:
            logger.error("No completed genome stats run found. Run aggregate_genome_sae_stats.py first.")
            sys.exit(1)
        stats_path = os.path.join(genome_run, "data", "global_feature_stats.npz")
    else:
        parser.error("Specify --stats_file or --auto")

    if not os.path.isfile(stats_path):
        # Also try genome_wide_sae_stats_corrected.npz in same dir
        alt_path = os.path.join(os.path.dirname(stats_path), "genome_wide_sae_stats_corrected.npz")
        if os.path.isfile(alt_path):
            stats_path = alt_path
        else:
            logger.error(f"Stats file not found: {stats_path}")
            sys.exit(1)

    logger.info(f"Global stats: {stats_path}")
    stats = dict(np.load(stats_path))

    # ── Find chromosome SAE run ──────────────────────────────────────────────
    sae_run_dir = find_latest_completed(results_dir, args.chrom, "sae")
    if sae_run_dir is None:
        logger.error(f"No completed SAE run found for {args.chrom}")
        sys.exit(1)

    logger.info(f"SAE run: {sae_run_dir}")

    # Load max-pooled vectors
    maxpool_path = os.path.join(sae_run_dir, "latent_analysis", "data", "maxpooled_vectors.npy")
    if not os.path.isfile(maxpool_path):
        logger.error(f"maxpooled_vectors.npy not found at {maxpool_path}")
        sys.exit(1)

    vectors = np.load(maxpool_path)
    logger.info(f"Loaded vectors: {vectors.shape}")

    # ── Normalize ────────────────────────────────────────────────────────────
    logger.info(f"Method: {args.method}")
    normalize_fn = METHODS[args.method]
    normalized = normalize_fn(vectors, stats)

    logger.info(f"Normalized shape: {normalized.shape}")
    logger.info(f"  Original range: [{vectors.min():.4f}, {vectors.max():.4f}]")
    logger.info(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_dir = os.path.join(sae_run_dir, "latent_analysis", "data")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "normalized_maxpooled_vectors.npy")
    np.save(out_path, normalized)
    logger.info(f"Saved: {out_path}")

    # Save metadata
    meta = {
        "method": args.method,
        "stats_file": stats_path,
        "chrom": args.chrom,
        "sae_run_dir": sae_run_dir,
        "original_shape": list(vectors.shape),
        "normalized_shape": list(normalized.shape),
        "n_valid_features": int(np.sum(stats.get("valid_mask", stats["std"] > 0))),
    }
    meta_path = os.path.join(output_dir, "normalization_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    logger.info(f"Saved: {meta_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
