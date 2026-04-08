#!/usr/bin/env python3
"""
plot_interactive_latent.py

Generate interactive HTML plots from saved latent analysis data using Plotly.
Reads cluster_assignments.tsv (with t-SNE/UMAP coordinates) and optionally
adds genomic annotation from GTF.

Produces:
  - Interactive 2D scatter (t-SNE or UMAP) with hover info
  - Interactive 3D t-SNE (requires maxpooled_vectors.npy for recomputation)
  - Color by: cluster, annotation, method, confidence, chromosome

Usage:
    # 2D interactive t-SNE for one chromosome (conf8.0 raw)
    python tools/plot_interactive_latent.py --chrom chr22 --results_dir results/ \\
        --gtf /path/to/genomic.gtf

    # 2D interactive for normalized
    python tools/plot_interactive_latent.py --chrom chr22 --results_dir results/ \\
        --normalized --gtf /path/to/genomic.gtf

    # 3D t-SNE (recomputes from pooled vectors)
    python tools/plot_interactive_latent.py --chrom chr22 --results_dir results/ \\
        --mode 3d --gtf /path/to/genomic.gtf

    # Genome-wide (from genome_sae_tsne output)
    python tools/plot_interactive_latent.py --genome_dir results/_genome_wide/sae_tsne/<run_dir>/ \\
        --gtf /path/to/genomic.gtf
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level="INFO"):
    logger = logging.getLogger("interactive_plot")
    logger.setLevel(getattr(logging, level))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def add_annotations(regions, gtf_path, chrom):
    """Add genomic annotation column from GTF."""
    from plot_tsne_by_annotation import load_gtf_features, classify_region, CHROM_MAP

    chrom_id = CHROM_MAP.get(chrom, chrom)
    # Check which ID format the GTF uses
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            gtf_chrom = line.split("\t")[0]
            if gtf_chrom == chrom:
                chrom_id = chrom
                break
            if gtf_chrom == chrom_id:
                break

    intervals = load_gtf_features(gtf_path, chrom_id)
    annotations = []
    for _, row in regions.iterrows():
        start = int(row.get("genomic_start", 0))
        end = int(row.get("genomic_end", 0))
        annotations.append(classify_region(start, end, intervals))
    regions["annotation"] = annotations
    return regions


def plot_2d_interactive(regions, coord_cols, color_col, title, output_path, logger=None):
    """Create interactive 2D scatter plot with Plotly."""
    import plotly.express as px

    hover_cols = ["genomic_start", "genomic_end", "method", "confidence", "region_length"]
    if "cluster" in regions.columns:
        hover_cols.append("cluster")
    if "cluster_id" in regions.columns:
        hover_cols.append("cluster_id")
    if "annotation" in regions.columns:
        hover_cols.append("annotation")
    if "chrom" in regions.columns:
        hover_cols.append("chrom")

    # Filter to existing columns
    hover_cols = [c for c in hover_cols if c in regions.columns]

    # Color setup
    color_map = None
    color_continuous_scale = None
    if color_col == "annotation":
        color_map = {
            "CDS": "#E74C3C",
            "UTR/exon": "#F39C12",
            "Intron": "#5DADE2",
            "Intergenic": "#BDC3C7",
        }
    elif color_col == "genomic_position_mb":
        color_continuous_scale = "Viridis"

    fig = px.scatter(
        regions,
        x=coord_cols[0],
        y=coord_cols[1],
        color=color_col,
        color_discrete_map=color_map,
        color_continuous_scale=color_continuous_scale,
        hover_data=hover_cols,
        title=title,
        opacity=0.5,
        width=1200,
        height=900,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        xaxis_title=coord_cols[0],
        yaxis_title=coord_cols[1],
    )

    fig.write_html(output_path)
    # Also save PNG and SVG
    png_path = output_path.replace(".html", ".png")
    svg_path = output_path.replace(".html", ".svg")
    fig.write_image(png_path, scale=2)
    fig.write_image(svg_path, format="svg")
    if logger:
        logger.info(f"Saved: {output_path} + .png + .svg")


def plot_3d_interactive(vectors, regions, color_col, title, output_path, logger=None):
    """Compute 3D t-SNE and create interactive 3D scatter."""
    from sklearn.manifold import TSNE
    import plotly.express as px

    if logger:
        logger.info(f"Computing 3D t-SNE on {vectors.shape[0]} vectors...")

    tsne = TSNE(
        n_components=3,
        metric='cosine',
        random_state=42,
        perplexity=min(30, vectors.shape[0] - 1),
        verbose=2,
    )
    coords_3d = tsne.fit_transform(vectors)

    regions = regions.copy()
    regions["tsne3d_1"] = coords_3d[:, 0]
    regions["tsne3d_2"] = coords_3d[:, 1]
    regions["tsne3d_3"] = coords_3d[:, 2]

    hover_cols = [c for c in ["genomic_start", "genomic_end", "method", "confidence", "annotation", "chrom"] if c in regions.columns]

    color_map = None
    if color_col == "annotation":
        color_map = {
            "CDS": "#E74C3C",
            "UTR/exon": "#F39C12",
            "Intron": "#5DADE2",
            "Intergenic": "#BDC3C7",
        }

    fig = px.scatter_3d(
        regions,
        x="tsne3d_1",
        y="tsne3d_2",
        z="tsne3d_3",
        color=color_col,
        color_discrete_map=color_map,
        hover_data=hover_cols,
        title=title,
        opacity=0.5,
        width=1200,
        height=900,
    )
    fig.update_traces(marker=dict(size=2))

    fig.write_html(output_path)
    png_path = output_path.replace(".html", ".png")
    svg_path = output_path.replace(".html", ".svg")
    fig.write_image(png_path, scale=2)
    fig.write_image(svg_path, format="svg")

    # Save 3D coordinates for reuse
    coords_path = output_path.replace(".html", "_coords.npy")
    np.save(coords_path, coords_3d)

    if logger:
        logger.info(f"Saved: {output_path} + .png + .svg + _coords.npy")


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML plots from latent analysis data.",
    )

    parser.add_argument("--chrom", default=None, help="Chromosome name (per-chromosome mode)")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--normalized", action="store_true",
                        help="Read from latent_analysis_normalized/")
    parser.add_argument("--conf0", action="store_true",
                        help="Read from latent_analysis_conf0/ or latent_analysis_conf0_normalized/")
    parser.add_argument("--genome_dir", default=None,
                        help="Path to genome-wide run directory (overrides --chrom)")
    parser.add_argument("--gtf", default=None, help="GTF file for annotation coloring")
    parser.add_argument("--mode", default="2d", choices=["2d", "3d", "both"],
                        help="2D scatter, 3D t-SNE, or both (default: 2d)")
    parser.add_argument("--embedding", default="tsne", choices=["tsne", "umap"],
                        help="Which 2D embedding to use (default: tsne)")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # Determine data directory
    if args.genome_dir:
        data_dir = os.path.join(args.genome_dir, "data")
        plots_dir = os.path.join(args.genome_dir, "plots")
        chrom_label = "genome"
    elif args.chrom:
        if args.conf0:
            suffix = "latent_analysis_conf0_normalized" if args.normalized else "latent_analysis_conf0"
        else:
            suffix = "latent_analysis_normalized" if args.normalized else "latent_analysis"
        base = os.path.join(os.path.abspath(args.results_dir), args.chrom, "sae", suffix)
        data_dir = os.path.join(base, "data")
        plots_dir = os.path.join(base, "plots")
        chrom_label = args.chrom
    else:
        parser.error("--chrom or --genome_dir required")

    os.makedirs(plots_dir, exist_ok=True)

    # Load cluster assignments
    tsv_path = os.path.join(data_dir, "cluster_assignments.tsv")
    if not os.path.isfile(tsv_path):
        logger.error(f"Not found: {tsv_path}")
        sys.exit(1)

    regions = pd.read_csv(tsv_path, sep="\t", comment="#")
    n_regions = len(regions)
    logger.info(f"Loaded {n_regions} regions from {tsv_path}")

    # Determine coordinate columns
    if args.embedding == "tsne" and "tsne_1" in regions.columns:
        coord_cols = ["tsne_1", "tsne_2"]
    elif args.embedding == "umap" and "umap_1" in regions.columns:
        coord_cols = ["umap_1", "umap_2"]
    elif "tsne_1" in regions.columns:
        coord_cols = ["tsne_1", "tsne_2"]
    elif "umap_1" in regions.columns:
        coord_cols = ["umap_1", "umap_2"]
    else:
        logger.error("No embedding coordinates found in TSV")
        sys.exit(1)

    emb_name = "t-SNE" if "tsne" in coord_cols[0] else "UMAP"
    logger.info(f"Using {emb_name} coordinates: {coord_cols}")

    # Add annotation if GTF provided
    if args.gtf and args.chrom:
        logger.info("Adding genomic annotations from GTF...")
        regions = add_annotations(regions, args.gtf, args.chrom)

    # Determine cluster column
    cluster_col = "cluster" if "cluster" in regions.columns else "cluster_id" if "cluster_id" in regions.columns else None

    # Add genomic position as continuous color (Mb from start of chromosome)
    if "genomic_start" in regions.columns:
        regions["genomic_position_mb"] = regions["genomic_start"] / 1e6

    # Determine color columns to plot
    norm_label = " (normalized)" if args.normalized else ""
    color_variants = []
    if cluster_col:
        regions[cluster_col] = regions[cluster_col].astype(str)
        color_variants.append((cluster_col, f"{emb_name} — {chrom_label}{norm_label} — Leiden Clusters"))
    if "annotation" in regions.columns:
        color_variants.append(("annotation", f"{emb_name} — {chrom_label}{norm_label} — Genomic Annotation"))
    if "method" in regions.columns:
        color_variants.append(("method", f"{emb_name} — {chrom_label}{norm_label} — Detection Method"))
    if "chrom" in regions.columns:
        color_variants.append(("chrom", f"{emb_name} — {chrom_label}{norm_label} — Chromosome"))
    if "genomic_position_mb" in regions.columns:
        color_variants.append(("genomic_position_mb", f"{emb_name} — {chrom_label}{norm_label} — Genomic Position (Mb)"))

    # 2D plots
    if args.mode in ("2d", "both"):
        for color_col, title in color_variants:
            safe_name = color_col.replace("/", "_")
            out_path = os.path.join(plots_dir, f"interactive_{emb_name.lower().replace('-','')}_{safe_name}.html")
            plot_2d_interactive(regions, coord_cols, color_col, title, out_path, logger)

    # 3D plot
    if args.mode in ("3d", "both"):
        vectors_path = os.path.join(data_dir, "maxpooled_vectors.npy")
        if not os.path.isfile(vectors_path):
            logger.error(f"Need maxpooled_vectors.npy for 3D t-SNE: {vectors_path}")
            sys.exit(1)

        vectors = np.load(vectors_path)
        # Align with regions (in case some zero vectors were removed)
        if vectors.shape[0] != n_regions:
            logger.warning(f"Vectors ({vectors.shape[0]}) != regions ({n_regions}), using min")
            n = min(vectors.shape[0], n_regions)
            vectors = vectors[:n]
            regions = regions.iloc[:n]

        color_col = "annotation" if "annotation" in regions.columns else (cluster_col or "method")
        title = f"3D t-SNE — {chrom_label}{norm_label}"
        out_path = os.path.join(plots_dir, "interactive_tsne_3d.html")
        plot_3d_interactive(vectors, regions, color_col, title, out_path, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
