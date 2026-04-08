#!/usr/bin/env python3
"""
compute_sae_latent.py

Compute-only stage of SAE latent analysis. No matplotlib, no plotting.
Produces all data files needed for downstream plotting.

Stages:
  pool    — Load SAE features from shards/NPZ, max-pool → maxpooled_vectors.npy
  cluster — Compute cosine similarity, t-SNE/UMAP, Leiden clustering
  both    — Run pool then cluster (default)

When --global_stats is set, loads raw pooled vectors from latent_analysis/
and applies z-score normalization before clustering. Output goes to
latent_analysis_normalized/.

Usage:
    # Full pipeline from shards (raw)
    python tools/compute_sae_latent.py --from_shards --chrom chr22 --results_dir results/

    # Normalized (reuses raw pooled vectors, no re-pooling)
    python tools/compute_sae_latent.py --from_shards --chrom chr22 --results_dir results/ \\
        --global_stats results/_genome_sae_stats/.../genome_wide_sae_stats.npz

    # Just pooling
    python tools/compute_sae_latent.py --from_shards --chrom chr22 --results_dir results/ --stage pool

    # Just clustering (pooled vectors must exist)
    python tools/compute_sae_latent.py --from_shards --chrom chr22 --results_dir results/ --stage cluster
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_sae_regions import (
    setup_logging,
    load_and_pool_feature_matrices,
    load_and_pool_from_shards,
    load_region_metadata,
    compute_cosine_similarity,
    compute_embedding_and_clusters,
    summarize_clusters,
    N_SAE_FEATURES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute SAE latent analysis (no plotting).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    parser.add_argument("--from_shards", action="store_true",
                        help="Read directly from shard directories.")
    parser.add_argument("--input_dir", default=None,
                        help="Path to merged SAE run directory.")
    parser.add_argument("--chrom", type=str, default=None,
                        help="Chromosome name (required with --from_shards).")
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="Root results directory.")
    parser.add_argument("--n_shards", type=int, default=36)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory.")

    # What to compute
    parser.add_argument("--stage", type=str, default="both",
                        choices=["pool", "cluster", "both"],
                        help="Which stages to run (default: both).")

    # Parameters
    parser.add_argument("--pool_method", type=str, default="max",
                        choices=["max", "mean"])
    parser.add_argument("--embedding", type=str, default="both",
                        choices=["tsne", "umap", "both"])
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--random_state", type=int, default=42)

    # Normalization
    parser.add_argument("--global_stats", type=str, default=None,
                        help="Path to genome-wide stats NPZ for z-score normalization.")

    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # Determine output directory
    _suffix = "latent_analysis_normalized" if args.global_stats else "latent_analysis"

    if args.from_shards:
        if not args.chrom:
            parser.error("--from_shards requires --chrom")
        results_dir = os.path.abspath(args.results_dir)
        output_dir = args.output_dir or os.path.join(results_dir, args.chrom, "sae", _suffix)
        # For normalized runs, look for raw cached vectors in the matching raw output dir
        # e.g. latent_analysis_conf0_normalized -> latent_analysis_conf0
        if args.global_stats and args.output_dir:
            raw_data_dir = os.path.join(args.output_dir.replace("_normalized", ""), "data")
        else:
            raw_data_dir = os.path.join(results_dir, args.chrom, "sae", "latent_analysis", "data")
    elif args.input_dir:
        input_dir = os.path.abspath(args.input_dir)
        output_dir = args.output_dir or os.path.join(input_dir, _suffix)
        raw_data_dir = os.path.join(input_dir, "latent_analysis", "data")
    else:
        parser.error("--from_shards or --input_dir required")

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SAE Latent Analysis — COMPUTE ONLY")
    logger.info("=" * 70)
    logger.info(f"Stage:      {args.stage}")
    logger.info(f"Output dir: {output_dir}")
    if args.global_stats:
        logger.info(f"Normalization: {args.global_stats}")

    # =========================================================================
    # STAGE: POOL
    # =========================================================================
    pooled_path = os.path.join(data_dir, "maxpooled_vectors.npy")
    sim_path = os.path.join(data_dir, "cosine_similarity.npy")
    tsv_paths_from_shards = None

    if args.stage in ("pool", "both"):
        # Check for cached raw vectors (for normalized runs)
        if args.global_stats and os.path.isfile(os.path.join(raw_data_dir, "maxpooled_vectors.npy")):
            logger.info("Loading raw pooled vectors (normalization will be applied)")
            pooled_vectors = np.load(os.path.join(raw_data_dir, "maxpooled_vectors.npy"))
            n_regions = pooled_vectors.shape[0]
            logger.info(f"  Loaded {n_regions} vectors from {raw_data_dir}/")
        elif os.path.isfile(pooled_path) and not args.global_stats:
            logger.info(f"RESUME: Cached pooled vectors found at {pooled_path}")
            pooled_vectors = np.load(pooled_path)
            n_regions = pooled_vectors.shape[0]
        elif args.from_shards:
            logger.info("Pooling from shard directories...")
            pooled_vectors, n_regions, tsv_paths_from_shards = load_and_pool_from_shards(
                results_dir, args.chrom,
                pool_method=args.pool_method,
                n_shards=args.n_shards,
                logger=logger,
            )
        else:
            npz_path = os.path.join(args.input_dir, "data", "feature_matrices.npz")
            pooled_vectors, n_regions = load_and_pool_feature_matrices(
                npz_path, pool_method=args.pool_method, logger=logger
            )

        # Apply normalization if requested
        if args.global_stats:
            logger.info(f"Applying genome-wide z-score normalization")
            gstats = dict(np.load(args.global_stats))
            mean = gstats.get("mean", gstats.get("cross_chrom_mean"))
            std = gstats.get("std", gstats.get("cross_chrom_std"))
            valid = std > 0
            normalized = np.zeros_like(pooled_vectors)
            normalized[:, valid] = (pooled_vectors[:, valid] - mean[valid]) / std[valid]
            pooled_vectors = normalized
            logger.info(f"  Range: [{pooled_vectors.min():.4f}, {pooled_vectors.max():.4f}]")

        # Save pooled vectors
        np.save(pooled_path, pooled_vectors)
        logger.info(f"Saved: {pooled_path} ({pooled_vectors.shape})")

        # Compute and save cosine similarity
        if not os.path.isfile(sim_path):
            logger.info("Computing cosine similarity...")
            similarity_matrix = compute_cosine_similarity(pooled_vectors, logger=logger)
            np.save(sim_path, similarity_matrix)
            logger.info(f"Saved: {sim_path}")
        else:
            logger.info(f"RESUME: Similarity matrix exists at {sim_path}")

        if args.stage == "pool":
            logger.info("DONE (pool stage only)")
            return

    # =========================================================================
    # STAGE: CLUSTER
    # =========================================================================
    if args.stage in ("cluster", "both"):
        # Load pooled vectors if not already in memory
        if args.stage == "cluster":
            if not os.path.isfile(pooled_path):
                logger.error(f"Pooled vectors not found: {pooled_path}")
                logger.error("Run --stage pool first")
                sys.exit(1)
            pooled_vectors = np.load(pooled_path)
            n_regions = pooled_vectors.shape[0]
            logger.info(f"Loaded pooled vectors: {pooled_vectors.shape}")

        # Load metadata
        logger.info("Loading region metadata...")
        if tsv_paths_from_shards:
            region_metadata = []
            for tsv in tsv_paths_from_shards:
                region_metadata.extend(load_region_metadata(tsv))
        elif args.from_shards:
            import re, glob
            sae_root = os.path.join(results_dir, args.chrom, "sae")
            shard_pattern = re.compile(r"shard(\d+)of(\d+)")
            seen = {}
            for entry in sorted(os.listdir(sae_root)):
                if "merged" in entry:
                    continue
                m = shard_pattern.search(entry)
                if not m or int(m.group(2)) != args.n_shards:
                    continue
                full = os.path.join(sae_root, entry)
                if os.path.isfile(os.path.join(full, "COMPLETED")):
                    tsv = os.path.join(full, "data", "sae_results.tsv")
                    if os.path.isfile(tsv):
                        seen[int(m.group(1))] = tsv
            region_metadata = []
            for k in sorted(seen.keys()):
                region_metadata.extend(load_region_metadata(seen[k]))
        else:
            tsv_path = os.path.join(args.input_dir, "data", "sae_results.tsv")
            region_metadata = load_region_metadata(tsv_path, logger=logger)

        logger.info(f"  {len(region_metadata)} regions")

        # Align metadata with vectors
        if len(region_metadata) != n_regions:
            logger.warning(f"Metadata ({len(region_metadata)}) != vectors ({n_regions}), using min")
            n = min(len(region_metadata), n_regions)
            region_metadata = region_metadata[:n]
            pooled_vectors = pooled_vectors[:n]
            n_regions = n

        # Remove zero vectors
        zero_mask = np.all(pooled_vectors == 0, axis=1)
        if np.any(zero_mask):
            n_zero = np.sum(zero_mask)
            logger.warning(f"Removing {n_zero} all-zero regions")
            keep = ~zero_mask
            pooled_vectors = pooled_vectors[keep]
            region_metadata = [m for m, k in zip(region_metadata, keep) if k]
            n_regions = len(region_metadata)

        # Compute embedding and clustering
        logger.info("Computing embedding and Leiden clustering...")
        embedding_results = compute_embedding_and_clusters(
            pooled_vectors, region_metadata,
            method=args.embedding,
            leiden_resolution=args.leiden_resolution,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
            logger=logger,
        )

        clusters = embedding_results["cluster_assignments"]

        # Compute cluster summaries
        similarity_matrix = np.load(sim_path) if os.path.isfile(sim_path) else None
        if similarity_matrix is not None and embedding_results["n_clusters"] > 1:
            cluster_summaries = summarize_clusters(
                clusters, region_metadata, pooled_vectors, similarity_matrix,
                logger=logger,
            )
        else:
            cluster_summaries = []

        # Save all results
        logger.info("Saving results...")

        # Cluster assignments TSV
        import pandas as pd
        cluster_df = pd.DataFrame([
            {**m, "cluster": int(clusters[i])} for i, m in enumerate(region_metadata)
        ])
        if embedding_results.get("embedding_tsne") is not None:
            cluster_df["tsne_1"] = embedding_results["embedding_tsne"][:, 0]
            cluster_df["tsne_2"] = embedding_results["embedding_tsne"][:, 1]
        if embedding_results.get("embedding_umap") is not None:
            cluster_df["umap_1"] = embedding_results["embedding_umap"][:, 0]
            cluster_df["umap_2"] = embedding_results["embedding_umap"][:, 1]

        tsv_out = os.path.join(data_dir, "cluster_assignments.tsv")
        with open(tsv_out, "w") as f:
            f.write(f"# Latent analysis: cluster assignments and embedding coordinates\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# N regions: {n_regions}\n")
            f.write(f"# N clusters: {embedding_results['n_clusters']}\n")
            f.write(f"#\n")
        cluster_df.to_csv(tsv_out, sep="\t", index=False, mode="a")
        logger.info(f"  {tsv_out}")

        # Embeddings
        if embedding_results.get("embedding_tsne") is not None:
            np.save(os.path.join(data_dir, "embedding_tsne.npy"), embedding_results["embedding_tsne"])
        if embedding_results.get("embedding_umap") is not None:
            np.save(os.path.join(data_dir, "embedding_umap.npy"), embedding_results["embedding_umap"])

        # Cluster summaries
        if cluster_summaries:
            sum_path = os.path.join(data_dir, "cluster_summaries.tsv")
            with open(sum_path, "w") as f:
                if cluster_summaries:
                    header = "\t".join(cluster_summaries[0].keys())
                    f.write(header + "\n")
                    for s in cluster_summaries:
                        f.write("\t".join(str(v) for v in s.values()) + "\n")

        # Metadata
        meta = {
            "script": "compute_sae_latent.py",
            "timestamp": datetime.now().isoformat(),
            "chrom": args.chrom,
            "n_regions": n_regions,
            "n_clusters": embedding_results["n_clusters"],
            "embedding": args.embedding,
            "leiden_resolution": args.leiden_resolution,
            "normalized": args.global_stats is not None,
            "pool_method": args.pool_method,
        }
        with open(os.path.join(data_dir, "analysis_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("=" * 70)
        logger.info("DONE")
        logger.info(f"  Regions: {n_regions}")
        logger.info(f"  Clusters: {embedding_results['n_clusters']}")
        logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
