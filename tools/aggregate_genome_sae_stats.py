#!/usr/bin/env python3
"""
aggregate_genome_sae_stats.py

Collect max-pooled SAE feature vectors from all chromosomes and compute
genome-wide per-feature statistics for cross-chromosome normalization.

Two-pass approach:
  Pass 1 (per-chromosome): run_sae_on_chromosome_drops.py + analyze_sae_regions.py
  Pass 2 (this script):    aggregate across chromosomes → global stats

Usage:
    python tools/aggregate_genome_sae_stats.py \\
        --results_dir results/ \\
        --all_human

    python tools/aggregate_genome_sae_stats.py \\
        --results_dir results/ \\
        --chroms chr21 chr22 \\
        --force
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import (
    build_run_dir,
    find_latest_completed,
    write_completed,
    write_source,
)

ALL_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_maxpooled_vectors(sae_run_dir):
    """Load max-pooled vectors from a completed SAE run.

    Tries latent_analysis/data/maxpooled_vectors.npy first, then falls back
    to computing from data/feature_matrices.npz.

    Returns:
        np.ndarray of shape (N_regions, 32768), or None if loading fails.
    """
    # Primary path: pre-computed max-pooled vectors
    maxpool_path = os.path.join(sae_run_dir, "latent_analysis", "data", "maxpooled_vectors.npy")
    if os.path.isfile(maxpool_path):
        vectors = np.load(maxpool_path)
        logger.info(f"  Loaded maxpooled_vectors.npy: shape {vectors.shape}")
        return vectors

    # Fallback: compute from feature_matrices.npz
    npz_path = os.path.join(sae_run_dir, "data", "feature_matrices.npz")
    if not os.path.isfile(npz_path):
        logger.warning(f"  No maxpooled_vectors.npy or feature_matrices.npz found")
        return None

    logger.info(f"  Computing max-pool from feature_matrices.npz...")
    npz = np.load(npz_path)
    region_keys = sorted([k for k in npz.files if k.startswith("region_")],
                         key=lambda k: int(k.split("_")[1]))
    if not region_keys:
        logger.warning(f"  No region keys found in feature_matrices.npz")
        return None

    feature_matrices = [npz[k] for k in region_keys]
    n_regions = len(feature_matrices)
    n_features = feature_matrices[0].shape[1]
    pooled = np.zeros((n_regions, n_features), dtype=np.float32)
    for i, fm in enumerate(feature_matrices):
        pooled[i] = np.max(fm, axis=0).astype(np.float32)

    logger.info(f"  Max-pooled {n_regions} regions to shape {pooled.shape}")
    return pooled


def load_region_metadata(sae_run_dir):
    """Load region metadata from sae_results.tsv if available."""
    tsv_path = os.path.join(sae_run_dir, "data", "sae_results.tsv")
    if not os.path.isfile(tsv_path):
        return None

    rows = []
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            rows.append(dict(zip(header, fields)))
    return rows


def compute_global_stats(all_vectors):
    """Compute per-feature statistics across all regions.

    Args:
        all_vectors: np.ndarray of shape (N_total, 32768)

    Returns:
        dict of stat_name -> np.ndarray of shape (32768,)
    """
    logger.info(f"Computing global stats over {all_vectors.shape[0]} regions, "
                f"{all_vectors.shape[1]} features...")

    stats = {
        "mean": np.mean(all_vectors, axis=0).astype(np.float32),
        "std": np.std(all_vectors, axis=0).astype(np.float32),
        "min": np.min(all_vectors, axis=0).astype(np.float32),
        "max": np.max(all_vectors, axis=0).astype(np.float32),
        "median": np.median(all_vectors, axis=0).astype(np.float32),
        "q25": np.percentile(all_vectors, 25, axis=0).astype(np.float32),
        "q75": np.percentile(all_vectors, 75, axis=0).astype(np.float32),
        "q95": np.percentile(all_vectors, 95, axis=0).astype(np.float32),
        "q99": np.percentile(all_vectors, 99, axis=0).astype(np.float32),
        "n_nonzero": np.count_nonzero(all_vectors, axis=0).astype(np.int32),
    }

    # Valid mask: features with non-zero std (safe for z-scoring)
    stats["valid_mask"] = (stats["std"] > 0)

    n_features = all_vectors.shape[1]
    n_valid = np.sum(stats["valid_mask"])
    n_ever_active = np.sum(stats["n_nonzero"] > 0)
    mean_nnz_per_region = np.mean(np.count_nonzero(all_vectors, axis=1))

    logger.info(f"  Features with nonzero std (valid for z-score): {n_valid}/{n_features}")
    logger.info(f"  Features ever active (any region): {n_ever_active}/{n_features}")
    logger.info(f"  Mean nonzero features per region: {mean_nnz_per_region:.0f}/{n_features}")

    # Sanity check
    assert np.all(stats["min"] <= stats["mean"]), "min > mean detected"
    assert np.all(stats["mean"] <= stats["max"]), "mean > max detected"

    # Log top 20 features by global mean
    top_idx = np.argsort(stats["mean"])[::-1][:20]
    logger.info("  Top 20 features by global mean activation:")
    for rank, idx in enumerate(top_idx):
        logger.info(f"    #{rank+1}: feature {idx}, mean={stats['mean'][idx]:.4f}, "
                     f"std={stats['std'][idx]:.4f}, n_nonzero={stats['n_nonzero'][idx]}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate genome-wide SAE feature statistics for cross-chromosome normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--chroms", nargs="+", type=str, default=None,
                        help="Specific chromosomes to include")
    parser.add_argument("--all_human", action="store_true",
                        help="Use all 24 human chromosomes (chr1-22, chrX, chrY)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/_genome_sae_stats/)")
    parser.add_argument("--min_chromosomes", type=int, default=20,
                        help="Minimum chromosomes required (default: 20)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even with fewer than --min_chromosomes")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    global logger
    logger = setup_logging(args.log_level)

    t0 = time.time()

    # Determine chromosome list
    if args.all_human:
        chroms = ALL_HUMAN_CHROMS
    elif args.chroms:
        chroms = args.chroms
    else:
        parser.error("Specify --chroms or --all_human")

    results_dir = os.path.abspath(args.results_dir)

    logger.info("=" * 70)
    logger.info("Genome-Wide SAE Feature Aggregation")
    logger.info("=" * 70)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Chromosomes requested: {len(chroms)}")

    # ── Discover completed SAE runs ──────────────────────────────────────────
    chrom_data = {}  # chrom -> {sae_run_dir, vectors, metadata}
    missing = []

    for chrom in chroms:
        sae_run_dir = find_latest_completed(results_dir, chrom, "sae")
        if sae_run_dir is None:
            logger.warning(f"  {chrom}: no completed SAE run found")
            missing.append(chrom)
            continue

        logger.info(f"  {chrom}: {os.path.basename(sae_run_dir)}")
        vectors = load_maxpooled_vectors(sae_run_dir)
        if vectors is None:
            logger.warning(f"  {chrom}: could not load feature vectors")
            missing.append(chrom)
            continue

        metadata = load_region_metadata(sae_run_dir)
        chrom_data[chrom] = {
            "sae_run_dir": sae_run_dir,
            "vectors": vectors,
            "n_regions": vectors.shape[0],
            "metadata": metadata,
        }

    n_found = len(chrom_data)
    logger.info(f"\nFound {n_found}/{len(chroms)} chromosomes with completed SAE data")
    if missing:
        logger.info(f"Missing: {', '.join(missing)}")

    if n_found < args.min_chromosomes and not args.force:
        logger.error(
            f"Only {n_found} chromosomes found, need {args.min_chromosomes}. "
            f"Use --force to proceed anyway."
        )
        sys.exit(1)

    if n_found == 0:
        logger.error("No chromosomes with SAE data found. Exiting.")
        sys.exit(1)

    # ── Stack all vectors ────────────────────────────────────────────────────
    all_vectors_list = []
    manifest_rows = []
    chrom_summary_rows = []

    for chrom in chroms:
        if chrom not in chrom_data:
            continue
        info = chrom_data[chrom]
        n_regions = info["n_regions"]
        offset = len(manifest_rows)

        chrom_summary_rows.append({
            "chrom": chrom,
            "n_regions": n_regions,
            "sae_run_dir": info["sae_run_dir"],
        })

        for i in range(n_regions):
            row = {"global_index": offset + i, "chrom": chrom, "region_index": i}
            if info["metadata"] and i < len(info["metadata"]):
                meta = info["metadata"][i]
                row["start"] = meta.get("start", "")
                row["end"] = meta.get("end", "")
                row["detection_method"] = meta.get("detection_method", meta.get("method", ""))
            manifest_rows.append(row)

        all_vectors_list.append(info["vectors"])

    all_vectors = np.concatenate(all_vectors_list, axis=0)
    logger.info(f"\nConcatenated matrix: {all_vectors.shape} "
                f"({all_vectors.nbytes / 1e6:.1f} MB)")

    # ── Compute global stats ─────────────────────────────────────────────────
    stats = compute_global_stats(all_vectors)

    # ── Create output directory ──────────────────────────────────────────────
    output_base = args.output_dir or os.path.join(results_dir, "_genome_sae_stats")
    flags = f"genome_stats_{n_found}chroms"
    run_dir = build_run_dir(output_base, "", "", flags) if args.output_dir else None

    if run_dir is None:
        # build_run_dir expects chrom/stage, so do it manually for global output
        from datetime import datetime
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_base, f"{ts_str}_{flags}")
        os.makedirs(run_dir, exist_ok=True)

    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"\nOutput dir: {run_dir}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    # 1. Global feature stats
    stats_path = os.path.join(data_dir, "global_feature_stats.npz")
    np.savez_compressed(stats_path, **stats)
    logger.info(f"Saved global_feature_stats.npz ({len(stats)} arrays, each shape "
                f"{stats['mean'].shape})")

    # 2. All max-pooled vectors
    vectors_path = os.path.join(data_dir, "all_maxpooled_vectors.npy")
    np.save(vectors_path, all_vectors)
    logger.info(f"Saved all_maxpooled_vectors.npy: {all_vectors.shape}")

    # 3. Region manifest
    manifest_path = os.path.join(data_dir, "region_manifest.tsv")
    if manifest_rows:
        cols = list(manifest_rows[0].keys())
        with open(manifest_path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for row in manifest_rows:
                f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
        logger.info(f"Saved region_manifest.tsv: {len(manifest_rows)} regions")

    # 4. Chromosome summary
    summary_path = os.path.join(data_dir, "chromosome_summary.tsv")
    with open(summary_path, "w") as f:
        f.write("chrom\tn_regions\tsae_run_dir\n")
        for row in chrom_summary_rows:
            f.write(f"{row['chrom']}\t{row['n_regions']}\t{row['sae_run_dir']}\n")
    logger.info(f"Saved chromosome_summary.tsv: {len(chrom_summary_rows)} chromosomes")

    # 5. Aggregation metadata
    meta = {
        "chroms_requested": chroms,
        "chroms_found": list(chrom_data.keys()),
        "chroms_missing": missing,
        "n_total_regions": int(all_vectors.shape[0]),
        "n_features": int(all_vectors.shape[1]),
        "n_valid_features": int(np.sum(stats["valid_mask"])),
        "min_chromosomes": args.min_chromosomes,
        "force": args.force,
    }
    meta_path = os.path.join(data_dir, "aggregation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # 6. Source JSON
    source_inputs = {}
    for chrom, info in chrom_data.items():
        source_inputs[f"sae_{chrom}"] = info["sae_run_dir"]
    write_source(run_dir, **source_inputs)

    # 7. COMPLETED sentinel
    wall_time = time.time() - t0
    write_completed(run_dir, os.path.basename(__file__), wall_time)

    logger.info(f"\nDone in {wall_time:.1f}s")
    logger.info(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
