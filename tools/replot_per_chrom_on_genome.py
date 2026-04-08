#!/usr/bin/env python3
"""
replot_per_chrom_on_genome.py

For each chromosome, plot genome-wide t-SNE/UMAP with all other points in gray
and the current chromosome highlighted. Generate coloring variants:
  - By chromosome color (highlight only)
  - By annotation (CDS/UTR/Intron/Intergenic)
  - By confidence
  - By region length
  - By n_features_fired (if available)

Uses genome-wide cluster_assignments.tsv for coordinates and per-chrom
latent_analysis data for coloring properties.

Usage:
    python tools/replot_per_chrom_on_genome.py \\
        --genome_tsv results/_genome_wide/sae_tsne/.../data/cluster_assignments.tsv \\
        --results_dir results/ \\
        --output_dir results/_genome_wide/per_chrom_highlights/
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANNOTATION_COLORS = {
    "CDS": "#e41a1c",
    "UTR/exon": "#ff7f00",
    "Intron": "#377eb8",
    "Intergenic": "#999999",
}
ANNOTATION_ORDER = ["CDS", "UTR/exon", "Intron", "Intergenic"]


def get_chrom_color(chrom, chrom_list):
    """Get the tab20 color for a chromosome matching genome-wide plots."""
    cmap = plt.cm.get_cmap("tab20", len(chrom_list))
    idx = chrom_list.index(chrom) if chrom in chrom_list else 0
    return cmap(idx)


def plot_highlighted(coords_bg, coords_fg, colors_fg, title, xlabel, ylabel,
                     out_path, s_bg=0.3, s_fg=1.0, alpha_bg=0.05, alpha_fg=0.5,
                     cmap=None, colorbar_label=None, legend_items=None, dpi=300):
    """Plot with gray background and colored foreground."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: all points in gray
    ax.scatter(coords_bg[:, 0], coords_bg[:, 1],
               c="#e0e0e0", s=s_bg, alpha=alpha_bg, rasterized=True)

    # Foreground: chromosome points colored
    if cmap is not None:
        sc = ax.scatter(coords_fg[:, 0], coords_fg[:, 1],
                        c=colors_fg, cmap=cmap, s=s_fg, alpha=alpha_fg, rasterized=True)
        plt.colorbar(sc, ax=ax, label=colorbar_label or "")
    elif legend_items is not None:
        # Categorical coloring
        for label, color, mask in legend_items:
            if mask.sum() == 0:
                continue
            ax.scatter(coords_fg[mask, 0], coords_fg[mask, 1],
                       c=color, s=s_fg, alpha=alpha_fg,
                       label=f"{label} ({mask.sum():,})", rasterized=True)
        ax.legend(markerscale=max(1, 6/s_fg), fontsize=9, loc="upper left")
    else:
        ax.scatter(coords_fg[:, 0], coords_fg[:, 1],
                   c=colors_fg, s=s_fg, alpha=alpha_fg, rasterized=True)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome_tsv", required=True, help="Genome-wide cluster_assignments.tsv")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--latent_subdir", default="latent_analysis",
                        help="Per-chrom latent subdir for firing stats etc.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dot_size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--chroms", nargs="+", default=None)
    args = parser.parse_args()

    # Load genome-wide data
    logger.info(f"Loading genome-wide TSV: {args.genome_tsv}")
    gw = pd.read_csv(args.genome_tsv, sep="\t", comment="#")
    logger.info(f"  {len(gw):,} regions, {gw['chrom'].nunique()} chromosomes")

    # Detect embedding columns
    embeddings = []
    if "tsne_1" in gw.columns:
        embeddings.append(("tsne", "tsne_1", "tsne_2"))
    if "umap_1" in gw.columns:
        embeddings.append(("umap", "umap_1", "umap_2"))

    if not embeddings:
        logger.error("No embedding columns found in TSV")
        sys.exit(1)

    # Chromosome ordering (for consistent colors)
    chrom_list = sorted(gw["chrom"].unique(),
                        key=lambda c: (int(c.replace("chr", "").replace("X", "23").replace("Y", "24"))
                                       if c.startswith("chr") else 99))

    chroms_to_plot = args.chroms or chrom_list
    os.makedirs(args.output_dir, exist_ok=True)

    s_fg = args.dot_size
    s_bg = max(0.3, s_fg * 0.3)

    for emb_name, col1, col2 in embeddings:
        coords_all = gw[[col1, col2]].values
        prefix = emb_name.upper()
        xlabel = f"{prefix} 1"
        ylabel = f"{prefix} 2"

        for chrom in chroms_to_plot:
            if chrom not in gw["chrom"].values:
                logger.warning(f"  {chrom}: not in genome-wide data, skipping")
                continue

            mask_chrom = gw["chrom"] == chrom
            n_chrom = mask_chrom.sum()
            coords_fg = coords_all[mask_chrom]
            chrom_color = get_chrom_color(chrom, chrom_list)
            gw_chrom = gw[mask_chrom]

            chrom_dir = os.path.join(args.output_dir, chrom)
            os.makedirs(chrom_dir, exist_ok=True)

            logger.info(f"  {chrom} ({n_chrom:,} regions) — {emb_name}")

            # 1. Highlight only (chromosome color)
            plot_highlighted(
                coords_all, coords_fg, [chrom_color],
                f"{prefix} — {chrom} Highlighted ({n_chrom:,} regions)",
                xlabel, ylabel,
                os.path.join(chrom_dir, f"{emb_name}_highlight.png"),
                s_bg=s_bg, s_fg=s_fg, alpha_bg=0.05, alpha_fg=min(args.alpha + 0.1, 0.8),
                dpi=args.dpi,
            )

            # 2. By annotation
            if "annotation" in gw.columns:
                ann = gw_chrom["annotation"].values
                legend_items = []
                for ann_type in ANNOTATION_ORDER:
                    ann_mask = np.array([a == ann_type for a in ann])
                    legend_items.append((ann_type, ANNOTATION_COLORS.get(ann_type, "#ccc"), ann_mask))
                plot_highlighted(
                    coords_all, coords_fg, None,
                    f"{prefix} — {chrom} by Annotation ({n_chrom:,} regions)",
                    xlabel, ylabel,
                    os.path.join(chrom_dir, f"{emb_name}_annotation.png"),
                    s_bg=s_bg, s_fg=s_fg, alpha_bg=0.05, alpha_fg=args.alpha,
                    legend_items=legend_items, dpi=args.dpi,
                )

            # 3. By confidence
            if "confidence" in gw.columns:
                plot_highlighted(
                    coords_all, coords_fg, gw_chrom["confidence"].values,
                    f"{prefix} — {chrom} by Confidence ({n_chrom:,} regions)",
                    xlabel, ylabel,
                    os.path.join(chrom_dir, f"{emb_name}_confidence.png"),
                    s_bg=s_bg, s_fg=s_fg, alpha_bg=0.05, alpha_fg=args.alpha,
                    cmap="viridis", colorbar_label="Confidence", dpi=args.dpi,
                )

            # 4. By region length
            lengths = (gw_chrom["genomic_end"] - gw_chrom["genomic_start"]).values
            p99 = np.percentile(lengths, 99) if len(lengths) > 0 else 1000
            plot_highlighted(
                coords_all, coords_fg, np.clip(lengths, 0, p99),
                f"{prefix} — {chrom} by Region Length (clipped P99={p99:.0f}bp)",
                xlabel, ylabel,
                os.path.join(chrom_dir, f"{emb_name}_region_length.png"),
                s_bg=s_bg, s_fg=s_fg, alpha_bg=0.05, alpha_fg=args.alpha,
                cmap="plasma", colorbar_label="Region Length (bp)", dpi=args.dpi,
            )

            # 5. By n_features_fired (from per-chrom data)
            firing_path = os.path.join(args.results_dir, chrom, "sae",
                                       args.latent_subdir, "data", "firing_stats.tsv")
            if os.path.exists(firing_path):
                firing = pd.read_csv(firing_path, sep="\t")
                # Match by genomic_start
                merged = gw_chrom.merge(firing[["genomic_start", "genomic_end", "n_features_fired"]],
                                        on=["genomic_start", "genomic_end"], how="left")
                n_fired = merged["n_features_fired"].values
                valid = ~np.isnan(n_fired)
                if valid.sum() > 0:
                    plot_highlighted(
                        coords_all, coords_fg[valid], n_fired[valid],
                        f"{prefix} — {chrom} by # Features Fired ({valid.sum():,} regions)",
                        xlabel, ylabel,
                        os.path.join(chrom_dir, f"{emb_name}_n_features_fired.png"),
                        s_bg=s_bg, s_fg=s_fg, alpha_bg=0.05, alpha_fg=args.alpha,
                        cmap="viridis", colorbar_label="# Features Fired", dpi=args.dpi,
                    )

        logger.info(f"  Done {emb_name} for {len(chroms_to_plot)} chromosomes")

    n_files = sum(len(os.listdir(os.path.join(args.output_dir, d)))
                  for d in os.listdir(args.output_dir)
                  if os.path.isdir(os.path.join(args.output_dir, d)))
    logger.info(f"\nDone. {n_files} plots in {args.output_dir}")


if __name__ == "__main__":
    main()
