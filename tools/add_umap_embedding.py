#!/usr/bin/env python3
"""
add_umap_embedding.py

Add UMAP embedding to existing latent analysis directories that only have t-SNE.
Reads maxpooled_vectors.npy, computes UMAP, saves embedding_umap.npy and updates
cluster_assignments.tsv with umap_1/umap_2 columns.

Does NOT recompute t-SNE or Leiden clustering — preserves existing results.

Usage:
    # Single chromosome
    python tools/add_umap_embedding.py --chrom chr22 --results_dir results/

    # All human chromosomes missing UMAP
    python tools/add_umap_embedding.py --all_human --results_dir results/

    # E. coli (specify SAE run)
    python tools/add_umap_embedding.py --latent_dir results/NC_000913.3/sae/20260309_134205_max1000_conf3.0/latent_analysis
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_umap_only(pooled_vectors, n_neighbors=15, random_state=42):
    """Compute UMAP embedding from pooled vectors."""
    try:
        import scanpy as sc
    except ImportError:
        logger.error("scanpy required for UMAP. Install: pip install scanpy")
        sys.exit(1)

    n = pooled_vectors.shape[0]
    if n < 5:
        logger.warning(f"Only {n} regions, too few for UMAP")
        return None

    n_neighbors = min(n_neighbors, n - 1)
    logger.info(f"Computing UMAP for {n} regions ({pooled_vectors.shape[1]} features)...")

    adata = sc.AnnData(X=pooled_vectors.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    umap_coords = adata.obsm["X_umap"].copy()
    logger.info(f"UMAP done: {umap_coords.shape}")
    return umap_coords


def add_umap_to_latent_dir(latent_dir, n_neighbors=15, random_state=42):
    """Add UMAP to a single latent analysis directory."""
    data_dir = os.path.join(latent_dir, "data")
    mp_path = os.path.join(data_dir, "maxpooled_vectors.npy")
    umap_path = os.path.join(data_dir, "embedding_umap.npy")
    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")

    if not os.path.isfile(mp_path):
        logger.error(f"No maxpooled_vectors.npy: {mp_path}")
        return False

    if os.path.isfile(umap_path):
        logger.info(f"UMAP already exists: {umap_path}, skipping")
        return True

    # Load vectors
    pooled = np.load(mp_path)
    logger.info(f"Loaded {pooled.shape} from {mp_path}")

    # Compute UMAP
    umap_coords = compute_umap_only(pooled, n_neighbors=n_neighbors, random_state=random_state)
    if umap_coords is None:
        return False

    # Save embedding
    np.save(umap_path, umap_coords)
    logger.info(f"Saved: {umap_path}")

    # Update cluster_assignments.tsv with umap columns
    if os.path.isfile(ca_path):
        ca = pd.read_csv(ca_path, sep="\t", comment="#")
        if "umap_1" not in ca.columns and len(ca) == len(umap_coords):
            ca["umap_1"] = umap_coords[:, 0]
            ca["umap_2"] = umap_coords[:, 1]

            # Preserve header comments
            header_lines = []
            with open(ca_path) as f:
                for line in f:
                    if line.startswith("#"):
                        header_lines.append(line)
                    else:
                        break

            with open(ca_path, "w") as f:
                for line in header_lines:
                    f.write(line)
            ca.to_csv(ca_path, sep="\t", index=False, mode="a")
            logger.info(f"Updated {ca_path} with umap_1/umap_2 columns")
        elif "umap_1" in ca.columns:
            logger.info("cluster_assignments.tsv already has umap columns")

    return True


def find_latent_dir(results_dir, chrom):
    """Find latent analysis directory for a chromosome."""
    # Try direct symlink path first
    direct = os.path.join(results_dir, chrom, "sae", "latent_analysis")
    if os.path.isdir(os.path.join(direct, "data")):
        return direct

    # Try under completed SAE runs
    sae_dir = os.path.join(results_dir, chrom, "sae")
    if not os.path.isdir(sae_dir):
        return None

    for entry in sorted(os.listdir(sae_dir), reverse=True):
        latent = os.path.join(sae_dir, entry, "latent_analysis")
        if os.path.isdir(os.path.join(latent, "data")):
            return latent
    return None


def main():
    parser = argparse.ArgumentParser(description="Add UMAP to existing latent analysis")
    parser.add_argument("--chrom", type=str, help="Single chromosome")
    parser.add_argument("--all_human", action="store_true", help="All human chroms missing UMAP")
    parser.add_argument("--latent_dir", type=str, help="Direct path to latent_analysis/ directory")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.latent_dir:
        add_umap_to_latent_dir(args.latent_dir, args.n_neighbors, args.random_state)
        return

    if args.all_human:
        chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    elif args.chrom:
        chroms = [args.chrom]
    else:
        parser.error("Specify --chrom, --all_human, or --latent_dir")
        return

    for chrom in chroms:
        latent = find_latent_dir(args.results_dir, chrom)
        if latent is None:
            logger.warning(f"No latent analysis found for {chrom}, skipping")
            continue

        umap_path = os.path.join(latent, "data", "embedding_umap.npy")
        if os.path.isfile(umap_path):
            logger.info(f"{chrom}: UMAP already exists, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {chrom}: {latent}")
        logger.info(f"{'='*60}")
        add_umap_to_latent_dir(latent, args.n_neighbors, args.random_state)


if __name__ == "__main__":
    main()
