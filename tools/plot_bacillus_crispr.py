#!/usr/bin/env python3
"""
plot_bacillus_crispr.py

Highlight CRISPR-associated regions on Bacillus subtilis t-SNE/UMAP plots.
Joins genomic coordinates from scoring boundaries with embedding coordinates
from latent analysis, then highlights regions overlapping known CRISPR loci.

Known B. subtilis 168 CRISPR/mobile element loci:
  - CRISPR locus (~2,747,000-2,749,000): cas genes (csn/cas9-like)
  - ICEBs1 mobile element (~466,000-549,000): integrative conjugative element

Usage:
    python tools/plot_bacillus_crispr.py \
        --boundaries results/NC_000964.3/scoring/.../data/drop_boundaries.tsv \
        --latent results/NC_000964.3/sae/.../latent_analysis/data/cluster_assignments_annotated.tsv \
        --output_dir results/NC_000964.3/sae/.../latent_analysis/plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Known loci of interest in B. subtilis 168 (NC_000964.3)
KNOWN_LOCI = {
    "CRISPR (csn/cas)": (2747000, 2750000),
    "ICEBs1 element": (466000, 550000),
    "SPbeta prophage": (2088000, 2220000),
    "PBSX prophage": (1470000, 1530000),
    "skin element": (988000, 1040000),
}


def main():
    parser = argparse.ArgumentParser(description="Highlight CRISPR regions on Bacillus t-SNE/UMAP")
    parser.add_argument("--boundaries", required=True, help="Path to drop_boundaries.tsv from scoring")
    parser.add_argument("--latent", required=True, help="Path to cluster_assignments_annotated.tsv from latent analysis")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--loci", nargs="+", default=None,
                        help="Custom loci as name:start:end (e.g., 'CRISPR:2747000:2750000')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load boundaries (genomic coordinates)
    bounds = pd.read_csv(args.boundaries, sep="\t", comment="#")
    print(f"Loaded {len(bounds)} boundaries")

    # Load latent analysis (embedding coordinates)
    latent = pd.read_csv(args.latent, sep="\t")
    print(f"Loaded {len(latent)} latent regions")

    n = min(len(bounds), len(latent))
    bounds = bounds.iloc[:n]
    latent = latent.iloc[:n]

    # Join genomic coordinates onto latent data
    latent["real_genomic_start"] = bounds["genomic_start"].values
    latent["real_genomic_end"] = bounds["genomic_end"].values
    latent["real_method"] = bounds["method"].values
    latent["real_confidence"] = bounds.get("start_confidence", bounds.get("confidence", pd.Series([0.0]*n))).values

    # Parse custom loci if provided
    loci = dict(KNOWN_LOCI)
    if args.loci:
        for loc in args.loci:
            parts = loc.split(":")
            loci[parts[0]] = (int(parts[1]), int(parts[2]))

    # Colors for each locus
    locus_colors = ["#FF0000", "#0000FF", "#00AA00", "#FF8C00", "#8B008B", "#00CED1", "#DC143C"]

    for emb_name, col1, col2 in [("tsne", "tsne_1", "tsne_2"), ("umap", "umap_1", "umap_2")]:
        if col1 not in latent.columns:
            continue

        coords = latent[[col1, col2]].values
        prefix = emb_name.upper()

        # --- Plot 1: All known loci highlighted ---
        fig, ax = plt.subplots(figsize=(14, 11))
        ax.scatter(coords[:, 0], coords[:, 1], c="#d0d0d0", s=3, alpha=0.3, rasterized=True, label="Other")

        for i, (name, (start, end)) in enumerate(loci.items()):
            mask = (latent["real_genomic_start"] >= start) & (latent["real_genomic_end"] <= end)
            n_hit = mask.sum()
            if n_hit == 0:
                # Try overlap instead of containment
                mask = (latent["real_genomic_end"] >= start) & (latent["real_genomic_start"] <= end)
                n_hit = mask.sum()
            if n_hit > 0:
                color = locus_colors[i % len(locus_colors)]
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=color, s=15, alpha=0.9, edgecolors='black', linewidths=0.3,
                           label=f"{name} ({n_hit} regions)", zorder=5)
                print(f"  {name}: {n_hit} regions ({start:,}-{end:,})")
            else:
                print(f"  {name}: 0 regions in embedding")

        ax.set_title(f"{prefix} of Bacillus subtilis SAE Regions (N={n:,})\nCRISPR & Mobile Elements Highlighted",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel(f"{prefix} 1", fontsize=14)
        ax.set_ylabel(f"{prefix} 2", fontsize=14)
        ax.legend(fontsize=11, loc="upper left", markerscale=2)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_crispr_loci.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {out}")

        # --- Plot 2: Individual locus plots ---
        for i, (name, (start, end)) in enumerate(loci.items()):
            mask = (latent["real_genomic_end"] >= start) & (latent["real_genomic_start"] <= end)
            n_hit = mask.sum()
            if n_hit == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 10))
            ax.scatter(coords[:, 0], coords[:, 1], c="#e0e0e0", s=2, alpha=0.1, rasterized=True)
            color = locus_colors[i % len(locus_colors)]
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=12, alpha=0.9, edgecolors='black', linewidths=0.3,
                       label=f"{name} ({n_hit} regions)", zorder=5)
            ax.set_title(f"{prefix} — {name} Highlighted ({n_hit} regions, {start:,}-{end:,} bp)",
                         fontsize=14, fontweight='bold')
            ax.set_xlabel(f"{prefix} 1", fontsize=14)
            ax.set_ylabel(f"{prefix} 2", fontsize=14)
            ax.legend(fontsize=12, markerscale=2)
            plt.tight_layout()
            safe_name = name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            out = os.path.join(args.output_dir, f"{emb_name}_{safe_name}.png")
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            print(f"Saved {out}")

        # --- Plot 3: Genomic position heatmap (to verify coordinates work) ---
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=latent["real_genomic_start"].values / 1e6,
                        cmap="viridis", s=5, alpha=0.7, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Genomic Position (Mbp)", fontsize=13)
        ax.set_title(f"{prefix} — Colored by Genomic Position", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{prefix} 1", fontsize=14)
        ax.set_ylabel(f"{prefix} 2", fontsize=14)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_genomic_position.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
