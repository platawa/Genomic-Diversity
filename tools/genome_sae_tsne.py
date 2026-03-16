#!/usr/bin/env python3
"""
genome_sae_tsne.py

Aggregate SAE region fingerprints across multiple chromosomes, compute t-SNE
(and optionally UMAP) on the combined matrix, and color by GTF annotation
(CDS / UTR-exon / Intron / Intergenic).

Reuses existing functions from:
  - analyze_sae_regions.py: compute_embedding_and_clusters(), load_region_metadata()
  - tools/plot_tsne_by_annotation.py: load_gtf_features(), classify_region(), CHROM_MAP
  - tools/aggregate_genome_sae_stats.py: load_maxpooled_vectors(), ALL_HUMAN_CHROMS
  - results_utils.py: find_latest_completed(), write_completed(), write_source()

Usage:
    python tools/genome_sae_tsne.py \\
        --all_human \\
        --gtf /path/to/genomic.gtf \\
        --results_dir results/ \\
        --embedding both

    python tools/genome_sae_tsne.py \\
        --chroms chr21 chr22 \\
        --gtf /path/to/genomic.gtf \\
        --embedding tsne
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
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def plot_embedding(coords, labels, colors, order, title, xlabel, ylabel,
                   out_path, point_size=15, alpha=0.7):
    """Plot a 2D scatter colored by categorical labels."""
    fig, ax = plt.subplots(figsize=(10, 8))
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
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

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

    fig, ax = plt.subplots(figsize=(12, 8))
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

    # ── 1. Collect data across chromosomes ───────────────────────────────────
    all_vectors = []
    all_metadata = []
    source_inputs = {}
    chrom_region_counts = {}

    for chrom in chroms:
        sae_run = find_latest_completed(results_dir, chrom, "sae")
        if sae_run is None:
            logger.warning(f"  {chrom}: no completed SAE run — skipping")
            continue

        logger.info(f"  {chrom}: {os.path.basename(sae_run)}")

        vectors = load_maxpooled_vectors(sae_run)
        if vectors is None:
            logger.warning(f"  {chrom}: could not load feature vectors — skipping")
            continue

        results_tsv = os.path.join(sae_run, "data", "sae_results.tsv")
        if not os.path.isfile(results_tsv):
            logger.warning(f"  {chrom}: no sae_results.tsv — skipping")
            continue

        metadata = load_region_metadata(results_tsv, logger=logger)
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

    n_chroms = len(chrom_region_counts)
    if n_chroms < 2:
        logger.error(f"Only {n_chroms} chromosome(s) with SAE data — need at least 2.")
        sys.exit(1)

    combined = np.vstack(all_vectors)
    n_total = combined.shape[0]
    logger.info(f"\nCombined matrix: {combined.shape} from {n_chroms} chromosomes")
    for chrom, cnt in chrom_region_counts.items():
        logger.info(f"  {chrom}: {cnt} regions")

    # ── 2. Compute embedding + clustering ────────────────────────────────────
    logger.info(f"\nComputing {args.embedding} embedding + Leiden clustering...")
    result = compute_embedding_and_clusters(
        combined, all_metadata,
        method=args.embedding,
        leiden_resolution=args.leiden_resolution,
        logger=logger,
    )
    logger.info(f"Clustering found {result['n_clusters']} clusters")

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

    # ── 5. Generate plots ────────────────────────────────────────────────────
    annotation_colors = {
        "CDS": "#e41a1c",
        "UTR/exon": "#ff7f00",
        "Intron": "#377eb8",
        "Intergenic": "#999999",
    }
    annotation_order = ["CDS", "UTR/exon", "Intron", "Intergenic"]
    chrom_labels = [m["chrom"] for m in all_metadata]
    confidences = np.array([m.get("confidence", 0.0) for m in all_metadata])

    for emb_name, emb_key in [("tsne", "embedding_tsne"), ("umap", "embedding_umap")]:
        coords = result.get(emb_key)
        if coords is None:
            continue

        prefix = emb_name.upper()
        xlabel = f"{prefix} 1"
        ylabel = f"{prefix} 2"

        # Annotation plot
        plot_embedding(
            coords, annotations, annotation_colors, annotation_order,
            title=f"{prefix} of SAE Region Fingerprints — {n_chroms} Chromosomes "
                  f"(N={n_total})\nColored by Genomic Annotation",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_annotation.png"),
        )

        # Annotation + confidence side by side
        plot_annotation_and_confidence(
            coords, annotations, confidences,
            annotation_colors, annotation_order,
            title=f"SAE Region Fingerprints — {n_chroms} Chromosomes (N={n_total})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_annotation_and_confidence.png"),
        )

        # By chromosome
        plot_by_chromosome(
            coords, chrom_labels,
            title=f"{prefix} of SAE Region Fingerprints — Colored by Chromosome "
                  f"(N={n_total})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_chromosome.png"),
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
