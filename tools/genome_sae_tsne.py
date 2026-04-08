#!/usr/bin/env python3
"""
genome_sae_tsne.py

Aggregate SAE region fingerprints across all chromosomes, compute genome-wide
t-SNE/UMAP embeddings with Leiden clustering, and generate multi-color plots:
  - Genomic annotation (CDS / UTR-exon / Intron / Intergenic)
  - Individual annotation types (separate plots for each)
  - Chromosome ID
  - Detection method (zscore / MAD)
  - Leiden cluster assignment
  - Continuous properties (confidence, region length)

Saves embedding checkpoints for fast plot regeneration without re-computing.

Reuses existing functions from:
  - analyze_sae_regions.py: compute_embedding_and_clusters(), load_region_metadata()
  - tools/plot_tsne_by_annotation.py: load_gtf_features(), classify_region(), CHROM_MAP
  - tools/aggregate_genome_sae_stats.py: load_maxpooled_vectors(), ALL_HUMAN_CHROMS
  - results_utils.py: find_latest_completed(), write_completed(), write_source()

Generated plots (per embedding method):
  - tsne/umap_by_annotation.png — all annotation types
  - tsne/umap_annotation_{cds,utr_exon,intron,intergenic}.png — individual types
  - tsne/umap_annotation_and_confidence.png — side-by-side
  - tsne/umap_continuous.png — confidence + region length
  - tsne/umap_by_chromosome.png — chromosome ID
  - tsne/umap_by_method.png — detection method
  - tsne/umap_by_cluster.png — Leiden clusters

Checkpoints saved in data/:
  - embedding_tsne.npy, embedding_umap.npy — for fast plot regeneration
  - cluster_assignments_array.npy — cluster IDs
  - cluster_assignments.tsv — full table with coordinates

Usage:
    # All human chromosomes, both t-SNE and UMAP
    python tools/genome_sae_tsne.py \\
        --all_human \\
        --gtf /path/to/genomic.gtf \\
        --results_dir results/ \\
        --embedding both

    # Specific chromosomes, z-score normalized
    python tools/genome_sae_tsne.py \\
        --chroms chr21 chr22 \\
        --gtf /path/to/genomic.gtf \\
        --embedding tsne \\
        --global_stats results/_global_stats/global_feature_stats.npz

    # t-SNE only, custom Leiden resolution
    python tools/genome_sae_tsne.py \\
        --all_human \\
        --gtf /path/to/genomic.gtf \\
        --embedding tsne \\
        --leiden_resolution 0.5
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import find_latest_completed, write_completed, write_source

from analyze_sae_regions import (
    compute_embedding_and_clusters,
    load_region_metadata,
)
from tools.aggregate_genome_sae_stats import (
    ALL_HUMAN_CHROMS,
    load_maxpooled_vectors,
)
from tools.plot_tsne_by_annotation import (
    CHROM_MAP,
    classify_region,
    load_gtf_features,
)

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genome-wide SAE t-SNE with GTF annotation coloring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chroms", nargs="+", type=str, default=None,
                        help="Specific chromosomes to include (e.g., chr21 chr22)")
    parser.add_argument("--all_human", action="store_true",
                        help="Use all 24 human chromosomes (chr1-22, chrX, chrY)")
    parser.add_argument("--gtf", required=True,
                        help="Path to GTF annotation file")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory (default: results/)")
    parser.add_argument("--embedding", type=str, default="tsne",
                        choices=["tsne", "umap", "both"],
                        help="Embedding method (default: tsne)")
    parser.add_argument("--leiden_resolution", type=float, default=1.0,
                        help="Leiden clustering resolution (default: 1.0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/_genome_wide/sae_tsne/)")
    parser.add_argument("--max_regions_per_chrom", type=int, default=0,
                        help="Max regions per chromosome, 0=all (default: 0)")
    parser.add_argument("--global_stats", type=str, default=None,
                        help="Path to global_feature_stats.npz for z-score normalization "
                             "of pooled vectors before embedding")
    parser.add_argument("--latent_subdir", type=str, default="latent_analysis",
                        help="Subdirectory name under results/chrN/sae/ to read pooled vectors from "
                             "(default: latent_analysis). Use latent_analysis_conf0 for conf0.0.")
    parser.add_argument("--n_pca", type=int, default=0,
                        help="Number of PCA components for dimensionality reduction before "
                             "neighbor graph (0=disabled, 50 recommended for large datasets)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def plot_embedding(coords, labels, colors, order, title, xlabel, ylabel,
                   out_path, point_size=15, alpha=0.7):
    """Plot a 2D scatter colored by categorical labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in order:
        mask = np.array([l == label for l in labels])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[label], label=f"{label} ({mask.sum()})",
                       s=point_size, alpha=alpha, edgecolors="none")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_annotation_and_confidence(coords, annotations, confidences,
                                   annotation_colors, annotation_order,
                                   title, xlabel, ylabel, out_path):
    """Side-by-side: annotation colors + confidence colormap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: annotation
    ax = axes[0]
    for label in annotation_order:
        mask = np.array([a == label for a in annotations])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=annotation_colors[label],
                       label=f"{label} ({mask.sum()})",
                       s=15, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title("Genomic Annotation")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Right: confidence
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=confidences, cmap="viridis",
                    s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Confidence")
    ax.set_title("Detection Confidence")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_by_chromosome(coords, chrom_labels, title, xlabel, ylabel, out_path):
    """Color points by chromosome using a discrete colormap."""
    unique_chroms = sorted(set(chrom_labels),
                           key=lambda c: ALL_HUMAN_CHROMS.index(c)
                           if c in ALL_HUMAN_CHROMS else 999)
    cmap = plt.cm.get_cmap("tab20", len(unique_chroms))
    chrom_to_color = {c: cmap(i) for i, c in enumerate(unique_chroms)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for chrom in unique_chroms:
        mask = np.array([c == chrom for c in chrom_labels])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[chrom_to_color[chrom]], label=f"{chrom} ({mask.sum()})",
                   s=15, alpha=0.6, edgecolors="none")
    ax.legend(fontsize=8, markerscale=2, ncol=2, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_single_annotation(coords, annotations, annotation_type, annotation_colors,
                           title, xlabel, ylabel, out_path, point_size=15):
    """Plot single annotation type only."""
    mask = np.array([a == annotation_type for a in annotations])
    if not mask.any():
        logger.warning(f"No regions with annotation '{annotation_type}'")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=annotation_colors.get(annotation_type, "#999999"),
               s=point_size, alpha=0.7, edgecolors="none")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_by_method(coords, methods, title, xlabel, ylabel, out_path):
    """Color points by detection method (zscore/MAD)."""
    method_colors = {
        "zscore": "#e74c3c",
        "MAD": "#3498db",
    }
    unique_methods = sorted(set(methods))

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in unique_methods:
        mask = np.array([m == method for m in methods])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=method_colors.get(method, "#999999"),
                   label=f"{method} ({mask.sum()})",
                   s=15, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_by_cluster(coords, clusters, title, xlabel, ylabel, out_path):
    """Color points by Leiden cluster."""
    unique_clusters = sorted(np.unique(clusters))
    cmap = plt.cm.get_cmap("tab20" if len(unique_clusters) <= 10 else "tab20b")
    cluster_colors = {c: cmap(i % 20) for i, c in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in unique_clusters:
        mask = clusters == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cluster_colors[c]], label=f"Cluster {c} ({mask.sum()})",
                   s=15, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=8, markerscale=2, ncol=2, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_continuous_colormaps(coords, confidences, region_lengths, title,
                               xlabel, ylabel, out_path):
    """Side-by-side: confidence and region length colormaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confidence
    ax = axes[0]
    sc1 = ax.scatter(coords[:, 0], coords[:, 1],
                     c=confidences, cmap="viridis",
                     s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(sc1, ax=ax, label="Confidence")
    ax.set_title("Colored by Confidence")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Region length
    ax = axes[1]
    sc2 = ax.scatter(coords[:, 0], coords[:, 1],
                     c=region_lengths, cmap="plasma",
                     s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(sc2, ax=ax, label="Region Length (bp)")
    ax.set_title("Colored by Region Length")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def main():
    args = parse_args()
    global logger
    logger = setup_logging(args.log_level)

    t0 = time.time()

    # Determine chromosome list
    if args.all_human:
        chroms = ALL_HUMAN_CHROMS
    elif args.chroms:
        chroms = args.chroms
    else:
        logger.error("Specify --chroms or --all_human")
        sys.exit(1)

    results_dir = os.path.abspath(args.results_dir)

    logger.info("=" * 70)
    logger.info("Genome-Wide SAE t-SNE with Annotation Coloring")
    logger.info("=" * 70)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"GTF: {args.gtf}")
    logger.info(f"Chromosomes requested: {len(chroms)}")
    logger.info(f"Embedding: {args.embedding}")

    # ── Check cache first before loading chromosomes ───────────────────────
    cache_base = args.output_dir or os.path.join(results_dir, "_genome_wide", "sae_tsne")
    os.makedirs(os.path.join(cache_base, "_cache"), exist_ok=True)
    cache_suffix = "_normalized" if args.global_stats else "_raw"
    combined_cache = os.path.join(cache_base, "_cache", f'combined_maxpooled{cache_suffix}.npy')
    metadata_cache = os.path.join(cache_base, "_cache", f'combined_metadata{cache_suffix}.json')

    skip_loading = False
    if os.path.isfile(combined_cache) and os.path.isfile(metadata_cache):
        logger.info(f"RESUME: Found cached combined vectors at {combined_cache}")
        skip_loading = True

    # ── 1. Collect data across chromosomes ───────────────────────────────────
    all_vectors = []
    all_metadata = []
    source_inputs = {}
    chrom_region_counts = {}

    latent_subdir = args.latent_subdir
    latent_subdir_norm = latent_subdir + "_normalized" if not latent_subdir.endswith("_normalized") else latent_subdir

    if not skip_loading:
      for chrom in chroms:
        # Try from_shards output first, then fall back to latest completed SAE run
        from_shards_vectors = os.path.join(results_dir, chrom, "sae", latent_subdir, "data", "maxpooled_vectors.npy")
        from_shards_norm = os.path.join(results_dir, chrom, "sae", latent_subdir_norm, "data", "maxpooled_vectors.npy")

        sae_run = find_latest_completed(results_dir, chrom, "sae")
        vectors = None
        results_tsv = None

        # Helper: find deduplicated shard TSVs (newest COMPLETED per index)
        def _find_deduped_shard_tsvs(sae_root):
            import re as _re
            shard_pat = _re.compile(r"shard(\d+)of(\d+)")
            seen = {}
            for entry in sorted(os.listdir(sae_root)):
                if "merged" in entry:
                    continue
                m = shard_pat.search(entry)
                if not m:
                    continue
                full = os.path.join(sae_root, entry)
                if os.path.isfile(os.path.join(full, "COMPLETED")):
                    tsv = os.path.join(full, "data", "sae_results.tsv")
                    if os.path.isfile(tsv):
                        idx = int(m.group(1))
                        seen[idx] = tsv  # newest wins (sorted order)
            return [seen[k] for k in sorted(seen.keys())]

        sae_root = os.path.join(results_dir, chrom, "sae")

        # Priority: from_shards normalized > from_shards raw > sae_run
        if args.global_stats and os.path.isfile(from_shards_norm):
            vectors = np.load(from_shards_norm)
            logger.info(f"  {chrom}: loaded from_shards normalized ({vectors.shape})")
            # Prefer cluster_assignments.tsv (already aligned with vectors)
            ca_tsv = os.path.join(results_dir, chrom, "sae", latent_subdir_norm, "data", "cluster_assignments.tsv")
            if os.path.isfile(ca_tsv):
                results_tsv = ca_tsv
            else:
                shard_tsvs = _find_deduped_shard_tsvs(sae_root)
                if shard_tsvs:
                    results_tsv = shard_tsvs
        elif os.path.isfile(from_shards_vectors):
            vectors = np.load(from_shards_vectors)
            logger.info(f"  {chrom}: loaded from_shards raw ({vectors.shape})")
            # Prefer cluster_assignments.tsv (already aligned with vectors)
            ca_tsv = os.path.join(results_dir, chrom, "sae", latent_subdir, "data", "cluster_assignments.tsv")
            if os.path.isfile(ca_tsv):
                results_tsv = ca_tsv
            else:
                shard_tsvs = _find_deduped_shard_tsvs(sae_root)
                if shard_tsvs:
                    results_tsv = shard_tsvs
        elif sae_run is not None:
            logger.info(f"  {chrom}: {os.path.basename(sae_run)}")
            vectors = load_maxpooled_vectors(sae_run)
            results_tsv = os.path.join(sae_run, "data", "sae_results.tsv")

        if vectors is None:
            logger.warning(f"  {chrom}: could not load feature vectors — skipping")
            continue

        # Load metadata
        if isinstance(results_tsv, list):
            # Multiple shard TSVs
            metadata = []
            for tsv in results_tsv:
                metadata.extend(load_region_metadata(tsv))
            logger.info(f"  {chrom}: loaded metadata from {len(results_tsv)} shard TSVs ({len(metadata)} regions)")
        elif results_tsv and os.path.isfile(results_tsv):
            metadata = load_region_metadata(results_tsv, logger=logger)
        else:
            logger.warning(f"  {chrom}: no sae_results.tsv — skipping")
            continue

        metadata = load_region_metadata(results_tsv, logger=logger) if not isinstance(results_tsv, list) else metadata
        if len(metadata) != vectors.shape[0]:
            logger.warning(
                f"  {chrom}: metadata ({len(metadata)}) != vectors ({vectors.shape[0]}) "
                f"— skipping"
            )
            continue

        # Optional cap on regions per chromosome
        if args.max_regions_per_chrom > 0 and len(metadata) > args.max_regions_per_chrom:
            # Random subsample
            rng = np.random.default_rng(42)
            idx = rng.choice(len(metadata), args.max_regions_per_chrom, replace=False)
            idx.sort()
            vectors = vectors[idx]
            metadata = [metadata[i] for i in idx]
            logger.info(f"    Subsampled to {len(metadata)} regions")

        # Tag metadata with chromosome
        for m in metadata:
            m["chrom"] = chrom

        all_vectors.append(vectors)
        all_metadata.extend(metadata)
        source_inputs[f"sae_{chrom}"] = sae_run
        chrom_region_counts[chrom] = len(metadata)

    if not skip_loading:
        n_chroms = len(chrom_region_counts)
        if n_chroms < 2:
            logger.error(f"Only {n_chroms} chromosome(s) with SAE data — need at least 2.")
            sys.exit(1)

    if skip_loading:
        # Copy to local /tmp if on a cluster node (NFS → local SSD is 10-50x faster)
        import shutil
        local_tmp = os.path.join("/tmp", f"sae_cache_{os.getpid()}")
        os.makedirs(local_tmp, exist_ok=True)
        local_combined = os.path.join(local_tmp, os.path.basename(combined_cache))
        if not os.path.isfile(local_combined):
            file_size_gb = os.path.getsize(combined_cache) / (1024**3)
            logger.info(f"RESUME: Copying {file_size_gb:.1f}GB to local /tmp for fast I/O...")
            import time as _time
            t0 = _time.time()
            shutil.copy2(combined_cache, local_combined)
            logger.info(f"  Copied in {_time.time()-t0:.0f}s")
        else:
            logger.info(f"RESUME: Using existing local copy at {local_combined}")
        logger.info(f"RESUME: Loading combined vectors from local copy...")
        combined = np.load(local_combined)
        import json as _json
        with open(metadata_cache) as f:
            all_metadata = _json.load(f)
        chrom_labels = [m.get('chrom', '?') for m in all_metadata]
        n_total = combined.shape[0]
        n_chroms = len(set(chrom_labels))
        logger.info(f"  Loaded {n_total} vectors from {n_chroms} chromosomes")
    else:
        combined = np.vstack(all_vectors)
        n_total = combined.shape[0]
        logger.info(f"\nCombined matrix: {combined.shape} from {n_chroms} chromosomes")
        for chrom, cnt in chrom_region_counts.items():
            logger.info(f"  {chrom}: {cnt} regions")

        # Optional: apply global z-score normalization
        if args.global_stats:
            logger.info(f"Applying genome-wide z-score normalization from {args.global_stats}")
            gstats = dict(np.load(args.global_stats))
            mean = gstats.get("mean", gstats.get("cross_chrom_mean"))
            std = gstats.get("std", gstats.get("cross_chrom_std"))
            valid = gstats.get("valid_mask", std > 0)
            normalized = np.zeros_like(combined)
            normalized[:, valid] = (combined[:, valid] - mean[valid]) / std[valid]
            combined = normalized
            logger.info(f"  Normalized range: [{combined.min():.4f}, {combined.max():.4f}]")

        # Save checkpoint
        logger.info(f"Saving combined vectors checkpoint to {combined_cache}")
        np.save(combined_cache, combined)
        import json as _json
        with open(metadata_cache, 'w') as f:
            _json.dump(all_metadata, f)
        logger.info(f"  Saved {combined.shape} vectors + {len(all_metadata)} metadata entries")

    # ── 2. Compute embedding + clustering ────────────────────────────────────
    # Check for PCA / neighbor graph cache (must be defined before use below)
    pca_suffix = f"_pca{args.n_pca}" if args.n_pca > 0 else ""

    # Check for cached embeddings
    cached_clusters = os.path.join(cache_base, "_cache", f"cluster_assignments{cache_suffix}{pca_suffix}.npy")
    cached_umap = os.path.join(cache_base, "_cache", f"embedding_umap{cache_suffix}.npy")
    cached_tsne = os.path.join(cache_base, "_cache", f"embedding_tsne{cache_suffix}.npy")

    # Check which embeddings are needed vs cached
    need_umap = args.embedding in ("umap", "both")
    need_tsne = args.embedding in ("tsne", "both")
    have_umap = os.path.isfile(cached_umap)
    have_tsne = os.path.isfile(cached_tsne)
    have_clusters = os.path.isfile(cached_clusters)
    cached_pca = os.path.join(cache_base, "_cache", f"pca_vectors{cache_suffix}{pca_suffix}.npy")
    cached_neighbors = os.path.join(cache_base, "_cache", f"neighbors{cache_suffix}{pca_suffix}.h5ad")

    all_cached = have_clusters and (not need_umap or have_umap) and (not need_tsne or have_tsne)

    if all_cached:
        logger.info("RESUME: Loading cached embeddings + clusters (all requested embeddings found)")
        result = {
            'cluster_assignments': np.load(cached_clusters),
            'n_clusters': len(np.unique(np.load(cached_clusters))),
            'embedding_umap': np.load(cached_umap) if have_umap else None,
            'embedding_tsne': np.load(cached_tsne) if have_tsne else None,
        }
        logger.info(f"  Clusters: {result['n_clusters']}, UMAP: {result['embedding_umap'] is not None}, t-SNE: {result['embedding_tsne'] is not None}")
    else:
        # Check if we can resume from cached PCA/neighbors
        if os.path.isfile(cached_pca) and os.path.isfile(cached_neighbors):
            logger.info(f"RESUME: Loading cached PCA vectors + neighbor graph")
            import scanpy as sc
            import anndata
            pca_vectors = np.load(cached_pca)
            adata = sc.read_h5ad(cached_neighbors)
            logger.info(f"  PCA: {pca_vectors.shape}, neighbors loaded")

            # Only compute missing embeddings
            result = {}
            if have_clusters:
                result['cluster_assignments'] = np.load(cached_clusters)
            else:
                sc.tl.leiden(adata, resolution=args.leiden_resolution, random_state=42)
                result['cluster_assignments'] = adata.obs['leiden'].astype(int).values

            result['n_clusters'] = len(np.unique(result['cluster_assignments']))

            if need_umap and have_umap:
                result['embedding_umap'] = np.load(cached_umap)
                logger.info("  Loaded cached UMAP")
            elif need_umap:
                logger.info("  Computing UMAP from cached neighbors...")
                sc.tl.umap(adata, random_state=42)
                result['embedding_umap'] = adata.obsm['X_umap'].copy()
                logger.info("  UMAP done")
            else:
                result['embedding_umap'] = None

            if need_tsne and have_tsne:
                result['embedding_tsne'] = np.load(cached_tsne)
                logger.info("  Loaded cached t-SNE")
            elif need_tsne:
                logger.info("  Computing t-SNE from cached PCA vectors...")
                sc.tl.tsne(adata, random_state=42, use_rep='X_pca' if 'X_pca' in adata.obsm else 'X', n_jobs=-1)
                result['embedding_tsne'] = adata.obsm['X_tsne'].copy()
                logger.info("  t-SNE done")
            else:
                result['embedding_tsne'] = None
        else:
            logger.info(f"\nComputing {args.embedding} embedding + Leiden clustering...")
            cache_data_dir = os.path.join(cache_base, "_cache")
            result = compute_embedding_and_clusters(
                combined, all_metadata,
                method=args.embedding,
                leiden_resolution=args.leiden_resolution,
                logger=logger,
                checkpoint_dir=cache_data_dir,
                n_pca_components=args.n_pca,
            )
            # Rename PCA/neighbor checkpoints to include suffix
            for src_name, dst_name in [
                ("pca_vectors.npy", f"pca_vectors{cache_suffix}{pca_suffix}.npy"),
                ("neighbors.h5ad", f"neighbors{cache_suffix}{pca_suffix}.h5ad"),
            ]:
                src = os.path.join(cache_data_dir, src_name)
                dst = os.path.join(cache_data_dir, dst_name)
                if os.path.isfile(src) and src != dst:
                    os.rename(src, dst)
                    logger.info(f"  Renamed {src_name} → {dst_name}")
    logger.info(f"Clustering found {result['n_clusters']} clusters")

    # Save embeddings + clusters immediately (before GTF parsing and plotting)
    cache_data_dir = os.path.join(cache_base, "_cache")
    logger.info(f"Saving embedding checkpoints to {cache_data_dir}/")
    np.save(os.path.join(cache_data_dir, f"cluster_assignments{cache_suffix}{pca_suffix}.npy"),
            result["cluster_assignments"])
    if result.get("embedding_umap") is not None:
        np.save(os.path.join(cache_data_dir, f"embedding_umap{cache_suffix}.npy"),
                result["embedding_umap"])
        logger.info(f"  Saved UMAP embedding ({result['embedding_umap'].shape})")
    if result.get("embedding_tsne") is not None:
        np.save(os.path.join(cache_data_dir, f"embedding_tsne{cache_suffix}.npy"),
                result["embedding_tsne"])
        logger.info(f"  Saved t-SNE embedding ({result['embedding_tsne'].shape})")
    logger.info(f"  Saved cluster assignments ({result['n_clusters']} clusters)")

    # ── 3. GTF annotation ────────────────────────────────────────────────────
    logger.info(f"\nClassifying regions by GTF annotation...")
    unique_chroms = sorted(set(m["chrom"] for m in all_metadata))
    gtf_intervals = {}
    for chrom in unique_chroms:
        chrom_id = CHROM_MAP.get(chrom, chrom)
        gtf_intervals[chrom] = load_gtf_features(args.gtf, chrom_id)

    annotations = []
    for m in all_metadata:
        intervals = gtf_intervals[m["chrom"]]
        annotations.append(
            classify_region(m["genomic_start"], m["genomic_end"], intervals)
        )

    # Print annotation summary
    from collections import Counter
    ann_counts = Counter(annotations)
    logger.info("Annotation counts:")
    for label in ["CDS", "UTR/exon", "Intron", "Intergenic"]:
        logger.info(f"  {label}: {ann_counts.get(label, 0)}")

    # ── 4. Create output directory ───────────────────────────────────────────
    output_base = args.output_dir or os.path.join(results_dir, "_genome_wide", "sae_tsne")
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = f"{n_chroms}chroms_{n_total}regions"
    run_dir = os.path.join(output_base, f"{ts_str}_{flags}")
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(f"\nOutput dir: {run_dir}")

    # ── 5. Prepare metadata for plotting ──────────────────────────────────────
    annotation_colors = {
        "CDS": "#e41a1c",
        "UTR/exon": "#ff7f00",
        "Intron": "#377eb8",
        "Intergenic": "#999999",
    }
    annotation_order = ["CDS", "UTR/exon", "Intron", "Intergenic"]
    chrom_labels = [m["chrom"] for m in all_metadata]
    confidences = np.array([m.get("confidence", 0.0) for m in all_metadata])
    region_lengths = np.array([m.get("region_length", 0) for m in all_metadata])
    methods = [m.get("method", "unknown") for m in all_metadata]

    # ── 5a. Save embeddings as checkpoints ─────────────────────────────────
    logger.info(f"\nSaving embedding checkpoints...")
    if result.get("embedding_tsne") is not None:
        np.save(os.path.join(data_dir, "embedding_tsne.npy"), result["embedding_tsne"])
        logger.info(f"Saved embedding_tsne.npy: {result['embedding_tsne'].shape}")
    if result.get("embedding_umap") is not None:
        np.save(os.path.join(data_dir, "embedding_umap.npy"), result["embedding_umap"])
        logger.info(f"Saved embedding_umap.npy: {result['embedding_umap'].shape}")
    np.save(os.path.join(data_dir, "cluster_assignments_array.npy"),
            result["cluster_assignments"])
    logger.info(f"Saved cluster_assignments_array.npy: {result['cluster_assignments'].shape}")

    # ── 5b. Generate plots ─────────────────────────────────────────────────
    logger.info(f"\nGenerating plots...")
    for emb_name, emb_key in [("tsne", "embedding_tsne"), ("umap", "embedding_umap")]:
        coords = result.get(emb_key)
        if coords is None:
            continue

        prefix = emb_name.upper()
        xlabel = f"{prefix} 1"
        ylabel = f"{prefix} 2"

        # --- Annotation plots ---
        plot_embedding(
            coords, annotations, annotation_colors, annotation_order,
            title=f"{prefix} of SAE Region Fingerprints — {n_chroms} Chromosomes "
                  f"(N={n_total})\nColored by Genomic Annotation",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_annotation.png"),
        )

        # Individual annotation type plots
        for ann_type in annotation_order:
            plot_single_annotation(
                coords, annotations, ann_type, annotation_colors,
                title=f"{prefix} of SAE Regions — {ann_type} Only (N={sum(a == ann_type for a in annotations)})",
                xlabel=xlabel, ylabel=ylabel,
                out_path=os.path.join(plots_dir, f"{emb_name}_annotation_{ann_type.lower().replace('/', '_')}.png"),
            )

        # Annotation + confidence side by side
        plot_annotation_and_confidence(
            coords, annotations, confidences,
            annotation_colors, annotation_order,
            title=f"SAE Region Fingerprints — {n_chroms} Chromosomes (N={n_total})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_annotation_and_confidence.png"),
        )

        # Continuous colormaps (confidence + region length)
        plot_continuous_colormaps(
            coords, confidences, region_lengths,
            title=f"{prefix} of SAE Regions — Continuous Properties",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_continuous.png"),
        )

        # By chromosome
        plot_by_chromosome(
            coords, chrom_labels,
            title=f"{prefix} of SAE Region Fingerprints — Colored by Chromosome "
                  f"(N={n_total})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_chromosome.png"),
        )

        # By detection method
        plot_by_method(
            coords, methods,
            title=f"{prefix} of SAE Regions — Colored by Detection Method",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_method.png"),
        )

        # By cluster
        plot_by_cluster(
            coords, result["cluster_assignments"],
            title=f"{prefix} of SAE Regions — Leiden Clusters (N={result['n_clusters']})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_cluster.png"),
        )

    # ── 6. Save data outputs ─────────────────────────────────────────────────
    # Combined max-pooled vectors
    np.save(os.path.join(data_dir, "combined_maxpooled.npy"), combined)
    logger.info(f"Saved combined_maxpooled.npy: {combined.shape}")

    # Cluster assignments TSV
    tsv_path = os.path.join(data_dir, "cluster_assignments.tsv")
    with open(tsv_path, "w") as f:
        cols = ["chrom", "genomic_start", "genomic_end", "method", "confidence",
                "annotation", "cluster"]
        if result.get("embedding_tsne") is not None:
            cols += ["tsne_1", "tsne_2"]
        if result.get("embedding_umap") is not None:
            cols += ["umap_1", "umap_2"]
        f.write("\t".join(cols) + "\n")

        for i, m in enumerate(all_metadata):
            row = [
                m["chrom"],
                str(m["genomic_start"]),
                str(m["genomic_end"]),
                m.get("method", ""),
                f"{m.get('confidence', 0.0):.4f}",
                annotations[i],
                str(result["cluster_assignments"][i]),
            ]
            if result.get("embedding_tsne") is not None:
                row += [f"{result['embedding_tsne'][i, 0]:.4f}",
                        f"{result['embedding_tsne'][i, 1]:.4f}"]
            if result.get("embedding_umap") is not None:
                row += [f"{result['embedding_umap'][i, 0]:.4f}",
                        f"{result['embedding_umap'][i, 1]:.4f}"]
            f.write("\t".join(row) + "\n")
    logger.info(f"Saved cluster_assignments.tsv: {n_total} regions")

    # Source and COMPLETED
    write_source(run_dir, **source_inputs)
    wall_time = time.time() - t0
    write_completed(run_dir, os.path.basename(__file__), wall_time)

    logger.info(f"\nDone in {wall_time:.1f}s")
    logger.info(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
