#!/usr/bin/env python3
"""
genome_replot_cached.py

Lightweight replotting of genome-wide UMAP/t-SNE from cached embeddings.
Loads only the cached coordinates (~6MB), clusters (~6MB), and metadata (~108MB)
instead of the full 91GB combined vectors. Runs in ~2-3 minutes with ~1GB RAM.

Usage:
    python tools/genome_replot_cached.py \
        --cache_dir results/_genome_wide/sae_tsne/_cache \
        --gtf /path/to/genomic.gtf \
        --output_dir results/_genome_wide/sae_tsne
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import write_completed

from tools.plot_tsne_by_annotation import (
    CHROM_MAP,
    classify_region,
    load_gtf_features,
)

# Import plotting functions from genome_sae_tsne
from tools.genome_sae_tsne import (
    plot_embedding,
    plot_single_annotation,
    plot_annotation_and_confidence,
    plot_continuous_colormaps,
    plot_by_chromosome,
    plot_by_method,
    plot_by_cluster,
)

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fast genome-wide replot from cached embeddings (no 91G load)",
    )
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Path to _cache/ directory with embeddings and metadata")
    parser.add_argument("--gtf", type=str, required=True,
                        help="Path to GTF annotation file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output dir (default: parent of cache_dir)")
    parser.add_argument("--embedding", type=str, default="umap",
                        choices=["tsne", "umap", "both"])
    parser.add_argument("--normalized", action="store_true",
                        help="Use normalized cache files instead of raw")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    global logger
    logger = setup_logging(args.log_level)
    t0 = time.time()

    cache_dir = args.cache_dir
    suffix = "_normalized" if args.normalized else "_raw"

    logger.info("=" * 70)
    logger.info("Genome-Wide Replot from Cached Embeddings (lightweight)")
    logger.info("=" * 70)
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Mode: {'normalized' if args.normalized else 'raw'}")

    # ── 1. Load cached metadata (108MB, ~2s) ────────────────────────────────
    metadata_path = os.path.join(cache_dir, f"combined_metadata{suffix}.json")
    if not os.path.isfile(metadata_path):
        logger.error(f"Metadata cache not found: {metadata_path}")
        sys.exit(1)

    logger.info(f"Loading cached metadata from {metadata_path}")
    with open(metadata_path) as f:
        all_metadata = json.load(f)
    n_total = len(all_metadata)
    chrom_labels = [m.get("chrom", "?") for m in all_metadata]
    n_chroms = len(set(chrom_labels))
    logger.info(f"  {n_total} regions from {n_chroms} chromosomes")

    # ── 2. Load cached embeddings (~6MB each, instant) ──────────────────────
    result = {}

    clusters_path = os.path.join(cache_dir, f"cluster_assignments{suffix}.npy")
    if os.path.isfile(clusters_path):
        result["cluster_assignments"] = np.load(clusters_path)
        result["n_clusters"] = len(np.unique(result["cluster_assignments"]))
        logger.info(f"  Clusters: {result['n_clusters']}")
    else:
        logger.error(f"Cluster cache not found: {clusters_path}")
        sys.exit(1)

    for emb_name in ["umap", "tsne"]:
        if args.embedding not in (emb_name, "both"):
            continue
        emb_path = os.path.join(cache_dir, f"embedding_{emb_name}{suffix}.npy")
        if os.path.isfile(emb_path):
            result[f"embedding_{emb_name}"] = np.load(emb_path)
            logger.info(f"  {emb_name.upper()}: {result[f'embedding_{emb_name}'].shape}")
        else:
            logger.warning(f"  {emb_name.upper()} cache not found: {emb_path}")

    load_time = time.time() - t0
    logger.info(f"Cache loading done in {load_time:.1f}s")

    # ── 3. GTF annotation ───────────────────────────────────────────────────
    logger.info(f"Classifying regions by GTF annotation...")
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

    ann_counts = Counter(annotations)
    logger.info("Annotation counts:")
    for label in ["CDS", "UTR/exon", "Intron", "Intergenic"]:
        logger.info(f"  {label}: {ann_counts.get(label, 0)}")

    # ── 4. Create output directory ──────────────────────────────────────────
    output_base = args.output_dir or os.path.dirname(cache_dir)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = f"{n_chroms}chroms_{n_total}regions_replot"
    run_dir = os.path.join(output_base, f"{ts_str}_{flags}")
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output dir: {run_dir}")

    # ── 5. Prepare metadata arrays ──────────────────────────────────────────
    annotation_colors = {
        "CDS": "#e41a1c",
        "UTR/exon": "#ff7f00",
        "Intron": "#377eb8",
        "Intergenic": "#999999",
    }
    annotation_order = ["CDS", "UTR/exon", "Intron", "Intergenic"]
    confidences = np.array([m.get("confidence", 0.0) for m in all_metadata])
    region_lengths = np.array([m.get("region_length", 0) for m in all_metadata])
    methods = [m.get("method", "unknown") for m in all_metadata]

    # ── 6. Generate plots ───────────────────────────────────────────────────
    logger.info("Generating plots...")
    for emb_name, emb_key in [("tsne", "embedding_tsne"), ("umap", "embedding_umap")]:
        coords = result.get(emb_key)
        if coords is None:
            continue

        prefix = emb_name.upper()
        xlabel = f"{prefix} 1"
        ylabel = f"{prefix} 2"

        plot_embedding(
            coords, annotations, annotation_colors, annotation_order,
            title=f"{prefix} of SAE Region Fingerprints — {n_chroms} Chromosomes "
                  f"(N={n_total})\nColored by Genomic Annotation",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_annotation.png"),
        )

        for ann_type in annotation_order:
            plot_single_annotation(
                coords, annotations, ann_type, annotation_colors,
                title=f"{prefix} of SAE Regions — {ann_type} Only (N={sum(a == ann_type for a in annotations)})",
                xlabel=xlabel, ylabel=ylabel,
                out_path=os.path.join(plots_dir, f"{emb_name}_annotation_{ann_type.lower().replace('/', '_')}.png"),
            )

        plot_annotation_and_confidence(
            coords, annotations, confidences,
            annotation_colors, annotation_order,
            title=f"SAE Region Fingerprints — {n_chroms} Chromosomes (N={n_total})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_annotation_and_confidence.png"),
        )

        plot_continuous_colormaps(
            coords, confidences, region_lengths,
            title=f"{prefix} of SAE Regions — Continuous Properties",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_continuous.png"),
        )

        plot_by_chromosome(
            coords, chrom_labels,
            title=f"{prefix} of SAE Region Fingerprints — Colored by Chromosome "
                  f"(N={n_total})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_chromosome.png"),
        )

        plot_by_method(
            coords, methods,
            title=f"{prefix} of SAE Regions — Colored by Detection Method",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_method.png"),
        )

        plot_by_cluster(
            coords, result["cluster_assignments"],
            title=f"{prefix} of SAE Regions — Leiden Clusters (N={result['n_clusters']})",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(plots_dir, f"{emb_name}_by_cluster.png"),
        )

    # ── 7. Save cluster assignments TSV ─────────────────────────────────────
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

    wall_time = time.time() - t0
    write_completed(run_dir, os.path.basename(__file__), wall_time)

    logger.info(f"\nDone in {wall_time:.1f}s")
    logger.info(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
