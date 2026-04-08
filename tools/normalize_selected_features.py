#!/usr/bin/env python3
"""
normalize_selected_features.py — Apply genome-wide z-score normalization
to SAE feature matrices in merged SAE runs.

For every feature j (32768 features), uses genome-wide nuc_mean_j and nuc_std_j
(computed across ALL nucleotide positions in the entire genome) to z-score
normalize feature activations in detected regions:

    z_{i,j} = (x_{i,j} - nuc_mean_j) / (nuc_std_j + 1e-6)

Input:
    - Genome-wide stats: results/_genome_sae_stats/<latest>/data/genome_wide_sae_stats_corrected.npz
      Must contain nuc_mean and nuc_std keys.
    - Per-chromosome merged SAE: results/<chrom>/sae/<merged_run>/data/feature_matrices.npz

Output (written alongside the original feature_matrices.npz):
    - feature_matrices_normalized.npz  — z-scored feature matrices (same structure)
    - feature_norm_stats_genome.npz    — copy of the genome-wide nuc_mean/nuc_std used

Usage:
    # Single chromosome
    python tools/normalize_selected_features.py \\
        --chrom chr22 --results_dir results/

    # All human chromosomes
    python tools/normalize_selected_features.py \\
        --all_human --results_dir results/

    # With explicit global stats path
    python tools/normalize_selected_features.py \\
        --all_human --results_dir results/ \\
        --global_stats results/_genome_sae_stats/<run>/data/genome_wide_sae_stats_corrected.npz
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import find_latest_completed, load_global_stats

ALL_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]

logger = logging.getLogger(__name__)


def find_global_stats(results_dir):
    """Find the latest genome-wide stats file with nuc_mean/nuc_std."""
    genome_stats_dir = os.path.join(results_dir, "_genome_sae_stats")
    if not os.path.isdir(genome_stats_dir):
        return None

    # Look for corrected aggregation runs (sorted by timestamp)
    runs = sorted([d for d in os.listdir(genome_stats_dir)
                   if os.path.isdir(os.path.join(genome_stats_dir, d))
                   and not d.startswith("_")])

    # Prefer the latest corrected run
    for run_name in reversed(runs):
        for fname in ["genome_wide_sae_stats_corrected.npz",
                      "genome_wide_sae_stats.npz",
                      "global_feature_stats.npz"]:
            path = os.path.join(genome_stats_dir, run_name, "data", fname)
            if os.path.isfile(path):
                # Check it has nuc_mean
                try:
                    data = np.load(path)
                    if "nuc_mean" in data:
                        return path
                except Exception:
                    continue

    return None


def normalize_chromosome(chrom, results_dir, gstats):
    """Z-score normalize feature matrices for one chromosome using genome-wide stats."""
    # Find the latest merged SAE run
    run_dir = find_latest_completed(results_dir, chrom, "sae")
    if run_dir is None:
        logger.warning(f"  {chrom}: no completed SAE run found, skipping")
        return False

    data_dir = os.path.join(run_dir, "data")
    matrices_path = os.path.join(data_dir, "feature_matrices.npz")
    if not os.path.isfile(matrices_path):
        logger.warning(f"  {chrom}: no feature_matrices.npz in {run_dir}, skipping")
        return False

    logger.info(f"  {chrom}: normalizing {run_dir}")

    nuc_mean = gstats["mean"]  # shape (32768,)
    nuc_std = gstats["std"]    # shape (32768,) — already has 1e-6 floor
    valid = gstats["valid_mask"]

    # Load and normalize each region matrix
    mat_data = np.load(matrices_path)
    region_keys = sorted([k for k in mat_data.keys() if k.startswith("region_")],
                         key=lambda k: int(k.split("_")[1]))

    normalized = {}
    n_positions_total = 0
    for key in region_keys:
        mat = mat_data[key]  # shape (n_positions, 32768)
        z = np.zeros_like(mat)
        z[:, valid] = (mat[:, valid] - nuc_mean[valid]) / nuc_std[valid]
        normalized[key] = z.astype(np.float32)
        n_positions_total += mat.shape[0]

    # Save normalized matrices
    out_path = os.path.join(data_dir, "feature_matrices_normalized.npz")
    np.savez_compressed(out_path, **normalized)

    # Save a copy of the genome-wide stats used (for provenance)
    stats_out = os.path.join(data_dir, "feature_norm_stats_genome.npz")
    np.savez_compressed(stats_out,
                        nuc_mean=nuc_mean, nuc_std=nuc_std,
                        valid_mask=valid)

    logger.info(f"    {len(region_keys)} regions, {n_positions_total:,} positions normalized")
    logger.info(f"    -> {out_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply genome-wide z-score normalization to SAE feature matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chrom", type=str, default=None,
                        help="Single chromosome to normalize")
    parser.add_argument("--chroms", nargs="+", type=str, default=None,
                        help="List of chromosomes to normalize")
    parser.add_argument("--all_human", action="store_true",
                        help="Normalize all 24 human chromosomes")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base results directory")
    parser.add_argument("--global_stats", type=str, default=None,
                        help="Path to genome-wide stats .npz with nuc_mean/nuc_std. "
                             "Auto-detected from results/_genome_sae_stats/ if not specified.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results_dir = os.path.abspath(args.results_dir)

    # Determine chromosomes
    if args.all_human:
        chroms = ALL_HUMAN_CHROMS
    elif args.chroms:
        chroms = args.chroms
    elif args.chrom:
        chroms = [args.chrom]
    else:
        parser.error("Specify --chrom, --chroms, or --all_human")

    # Load genome-wide stats
    stats_path = args.global_stats
    if stats_path is None:
        stats_path = find_global_stats(results_dir)
        if stats_path is None:
            logger.error("No genome-wide stats with nuc_mean/nuc_std found. "
                         "Run scan_sae_global_stats.py --aggregate_corrected first.")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("Genome-Wide Z-Score Normalization of SAE Feature Matrices")
    logger.info("=" * 70)
    logger.info(f"Global stats: {stats_path}")
    logger.info(f"Chromosomes: {len(chroms)}")

    gstats = load_global_stats(stats_path, prefer_nuc=True)
    if not gstats.get("nuc_stats", False):
        logger.error("Global stats file does not contain nuc_mean/nuc_std. "
                     "Re-run scans with updated scan_sae_global_stats.py, then re-aggregate.")
        sys.exit(1)

    n_valid = gstats["valid_mask"].sum()
    logger.info(f"Features with valid std: {n_valid}/32768")
    logger.info(f"nuc_mean range: [{gstats['mean'].min():.6f}, {gstats['mean'].max():.6f}]")
    logger.info(f"nuc_std range:  [{gstats['std'].min():.6f}, {gstats['std'].max():.6f}]")
    logger.info("")

    # Normalize each chromosome
    t0 = time.time()
    success = 0
    for chrom in chroms:
        if normalize_chromosome(chrom, results_dir, gstats):
            success += 1

    elapsed = time.time() - t0
    logger.info("")
    logger.info(f"Done: {success}/{len(chroms)} chromosomes normalized in {elapsed:.1f}s")
    if success < len(chroms):
        logger.warning(f"{len(chroms) - success} chromosomes skipped (no merged SAE data)")


if __name__ == "__main__":
    main()
