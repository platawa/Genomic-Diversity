#!/usr/bin/env python3
"""
plot_sae_latent.py

Plot-only stage of SAE latent analysis. Reads pre-computed data files
from compute_sae_latent.py and generates t-SNE/UMAP plots with various
colorings including genomic annotation from GTF.

No heavy computation — just loads saved embeddings and plots.

Usage:
    # Raw t-SNE + annotation
    python tools/plot_sae_latent.py --chrom chr22 --results_dir results/ \\
        --gtf /path/to/genomic.gtf

    # Normalized version
    python tools/plot_sae_latent.py --chrom chr22 --results_dir results/ \\
        --normalized --gtf /path/to/genomic.gtf

    # Just t-SNE, no annotation
    python tools/plot_sae_latent.py --chrom chr22 --results_dir results/ \\
        --plots tsne
"""

import argparse
import os
import sys
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_sae_regions import (
    setup_logging,
    plot_embedding,
    N_SAE_FEATURES,
)

from plot_tsne_by_annotation import (
    load_gtf_features,
    classify_region,
    CHROM_MAP,
)

# Annotation colors
ANNOTATION_COLORS = {
    "CDS": "#B80000",
    "UTR/exon": "#CC7A00",
    "Intron": "#1A6FAA",
    "Intergenic": "#888888",
}


def plot_annotation_tsne(
    coords: np.ndarray,
    annotations: list,
    output_path: str,
    chrom: str,
    n_regions: int,
    embedding_name: str = "t-SNE",
    logger=None,
):
    """Plot embedding colored by genomic annotation."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Count per category
    from collections import Counter
    counts = Counter(annotations)

    # Plot each category
    for cat in ["Intergenic", "Intron", "UTR/exon", "CDS"]:
        mask = [a == cat for a in annotations]
        if not any(mask):
            continue
        idx = np.where(mask)[0]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=ANNOTATION_COLORS[cat],
            s=5, alpha=0.8, edgecolors='none', rasterized=True,
            label=f"{cat} ({counts[cat]:,})",
        )

    ax.set_xlabel(f"{embedding_name} 1", fontsize=14)
    ax.set_ylabel(f"{embedding_name} 2", fontsize=14)
    ax.set_title(f"{embedding_name} of SAE Region Fingerprints — {chrom} (N={n_regions:,})\nColored by Genomic Annotation",
                 fontsize=16, fontweight='bold')
    ax.legend(markerscale=4, fontsize=12, framealpha=0.9)
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAE latent analysis from pre-computed data.",
    )

    parser.add_argument("--chrom", required=True, help="Chromosome name")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--normalized", action="store_true",
                        help="Read from latent_analysis_normalized/ instead of latent_analysis/")
    parser.add_argument("--gtf", default=None,
                        help="Path to GTF file for annotation coloring")
    parser.add_argument("--plots", default="all", choices=["tsne", "umap", "all"],
                        help="Which embedding plots to generate (default: all)")
    parser.add_argument("--input_dir", default=None,
                        help="Override: direct path to latent analysis directory")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # Determine data directory
    if args.input_dir:
        latent_dir = os.path.abspath(args.input_dir)
    else:
        suffix = "latent_analysis_normalized" if args.normalized else "latent_analysis"
        latent_dir = os.path.join(os.path.abspath(args.results_dir), args.chrom, "sae", suffix)

    data_dir = os.path.join(latent_dir, "data")
    plots_dir = os.path.join(latent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SAE Latent Analysis — PLOT ONLY")
    logger.info("=" * 70)
    logger.info(f"Data dir:  {data_dir}")
    logger.info(f"Plots dir: {plots_dir}")
    norm_label = " (normalized)" if args.normalized else " (raw)"
    logger.info(f"Mode:      {norm_label}")

    # Load cluster assignments (has t-SNE coordinates + metadata)
    assignments_path = os.path.join(data_dir, "cluster_assignments.tsv")
    if not os.path.isfile(assignments_path):
        logger.error(f"Not found: {assignments_path}")
        logger.error("Run compute_sae_latent.py first")
        sys.exit(1)

    regions = pd.read_csv(assignments_path, sep="\t", comment="#")
    n_regions = len(regions)
    logger.info(f"Loaded {n_regions} regions")

    # Extract data
    clusters = regions["cluster"].values if "cluster" in regions.columns else np.zeros(n_regions, dtype=int)
    if "cluster_id" in regions.columns:
        clusters = regions["cluster_id"].values

    region_metadata = []
    for _, row in regions.iterrows():
        m = {}
        for col in regions.columns:
            m[col] = row[col]
        region_metadata.append(m)

    # ---- t-SNE plots ----
    if args.plots in ("tsne", "all") and "tsne_1" in regions.columns:
        tsne_coords = regions[["tsne_1", "tsne_2"]].values
        logger.info(f"Generating t-SNE 4-panel plot...")

        plot_embedding(
            tsne_coords, region_metadata, clusters,
            os.path.join(plots_dir, "tsne_4panel.png"),
            embedding_name="t-SNE",
            logger=logger,
        )

        # Annotation-colored t-SNE
        if args.gtf:
            logger.info("Computing genomic annotations from GTF...")
            chrom_id = CHROM_MAP.get(args.chrom, args.chrom)

            # Check which ID format the GTF uses
            with open(args.gtf) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    gtf_chrom = line.split("\t")[0]
                    if gtf_chrom == args.chrom:
                        chrom_id = args.chrom
                        break
                    if gtf_chrom == chrom_id:
                        break

            intervals = load_gtf_features(args.gtf, chrom_id)
            logger.info(f"  GTF features: CDS={len(intervals['CDS'])}, "
                       f"exon={len(intervals['exon'])}, gene={len(intervals['gene'])}")

            annotations = []
            for _, row in regions.iterrows():
                start = int(row.get("genomic_start", 0))
                end = int(row.get("genomic_end", 0))
                ann = classify_region(start, end, intervals)
                annotations.append(ann)

            # Count
            from collections import Counter
            ann_counts = Counter(annotations)
            for cat, count in sorted(ann_counts.items()):
                logger.info(f"  {cat}: {count}")

            plot_annotation_tsne(
                tsne_coords, annotations,
                os.path.join(plots_dir, "tsne_by_annotation.png"),
                chrom=args.chrom, n_regions=n_regions,
                embedding_name="t-SNE", logger=logger,
            )

            # Save annotated TSV
            regions_ann = regions.copy()
            regions_ann["annotation"] = annotations
            ann_tsv = os.path.join(data_dir, "cluster_assignments_annotated.tsv")
            regions_ann.to_csv(ann_tsv, sep="\t", index=False)
            logger.info(f"Saved: {ann_tsv}")

    # ---- UMAP plots ----
    if args.plots in ("umap", "all") and "umap_1" in regions.columns:
        umap_coords = regions[["umap_1", "umap_2"]].values
        logger.info(f"Generating UMAP 4-panel plot...")

        plot_embedding(
            umap_coords, region_metadata, clusters,
            os.path.join(plots_dir, "umap_4panel.png"),
            embedding_name="UMAP",
            logger=logger,
        )

        if args.gtf and "annotations" in dir():
            plot_annotation_tsne(
                umap_coords, annotations,
                os.path.join(plots_dir, "umap_by_annotation.png"),
                chrom=args.chrom, n_regions=n_regions,
                embedding_name="UMAP", logger=logger,
            )

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info(f"  Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
