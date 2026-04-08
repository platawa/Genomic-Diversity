#!/usr/bin/env python3
"""
genome_replot_from_tsv.py

Ultra-fast replot from cluster_assignments.tsv — no GTF parsing, no cache loading.
Just reads the TSV (with UMAP coords + annotations already computed) and renders plots.
Takes ~30 seconds.

Usage:
    python tools/genome_replot_from_tsv.py \
        --tsv results/_genome_wide/sae_tsne/20260331_211349_23chroms_740913regions/data/cluster_assignments.tsv \
        --output_dir results/_genome_wide/sae_tsne/replot_small_dots \
        --dot_size 1 --alpha 0.3
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scatter(coords, labels, colors, order, title, xlabel, ylabel, out_path, s=1, alpha=0.3, dpi=300):
    fig, ax = plt.subplots(figsize=(12, 10))
    for label in order:
        mask = np.array([l == label for l in labels])
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[label], s=s, alpha=alpha, label=f"{label} ({mask.sum():,})", rasterized=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=max(1, 6/s), fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast genome UMAP replot from TSV")
    parser.add_argument("--tsv", required=True, help="Path to cluster_assignments.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots")
    parser.add_argument("--dot_size", type=float, default=1.0, help="Scatter dot size (s parameter)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Scatter alpha transparency")
    parser.add_argument("--dpi", type=int, default=300, help="Plot resolution in DPI (default: 300)")
    parser.add_argument("--per_chrom", action="store_true",
                        help="Generate individual chromosome highlight plots")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading {args.tsv}...")
    df = pd.read_csv(args.tsv, sep="\t")
    n_total = len(df)
    n_chroms = df["chrom"].nunique()
    print(f"  {n_total:,} regions from {n_chroms} chromosomes")

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect embedding columns
    has_umap = "umap_1" in df.columns and "umap_2" in df.columns
    has_tsne = "tsne_1" in df.columns and "tsne_2" in df.columns

    annotation_colors = {
        "CDS": "#e41a1c",
        "UTR/exon": "#ff7f00",
        "Intron": "#377eb8",
        "Intergenic": "#999999",
    }
    annotation_order = ["CDS", "UTR/exon", "Intron", "Intergenic"]

    s = args.dot_size
    alpha = args.alpha

    for emb_name, col1, col2 in [("umap", "umap_1", "umap_2"), ("tsne", "tsne_1", "tsne_2")]:
        if col1 not in df.columns:
            continue

        coords = df[[col1, col2]].values
        prefix = emb_name.upper()
        xlabel = f"{prefix} 1"
        ylabel = f"{prefix} 2"

        # By annotation
        plot_scatter(
            coords, df["annotation"].tolist(), annotation_colors, annotation_order,
            title=f"{prefix} of SAE Region Fingerprints — {n_chroms} Chromosomes (N={n_total:,})\nColored by Genomic Annotation",
            xlabel=xlabel, ylabel=ylabel,
            out_path=os.path.join(args.output_dir, f"{emb_name}_by_annotation.png"),
            s=s, alpha=alpha, dpi=args.dpi,
        )

        # Individual annotation types
        for ann_type in annotation_order:
            mask = df["annotation"] == ann_type
            n_ann = mask.sum()
            fig, ax = plt.subplots(figsize=(12, 10))
            # Background: all other points in light gray
            other = ~mask
            ax.scatter(coords[other, 0], coords[other, 1], c="#e0e0e0", s=s*0.5, alpha=0.1, rasterized=True)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=annotation_colors[ann_type], s=s, alpha=alpha,
                       label=f"{ann_type} ({n_ann:,})", rasterized=True)
            ax.set_title(f"{prefix} of SAE Regions — {ann_type} Only (N={n_ann:,})", fontsize=14)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(markerscale=max(1, 6/s), fontsize=9)
            plt.tight_layout()
            out = os.path.join(args.output_dir, f"{emb_name}_annotation_{ann_type.lower().replace('/', '_')}.png")
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            print(f"  Saved {out}")

        # By chromosome
        chrom_list = sorted(df["chrom"].unique(), key=lambda c: (int(c.replace("chr", "").replace("X", "23").replace("Y", "24")) if c.startswith("chr") else 99))
        cmap = plt.cm.get_cmap("tab20", len(chrom_list))
        fig, ax = plt.subplots(figsize=(14, 10))
        for i, chrom in enumerate(chrom_list):
            mask = df["chrom"] == chrom
            n_c = mask.sum()
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(i)], s=s, alpha=alpha, label=f"{chrom} ({n_c:,})", rasterized=True)
        ax.set_title(f"{prefix} of SAE Region Fingerprints — Colored by Chromosome (N={n_total:,})", fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(markerscale=max(1, 6/s), fontsize=8, loc="upper left", ncol=2)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_by_chromosome.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {out}")

        # Per-chromosome highlight plots
        if args.per_chrom:
            chrom_dir = os.path.join(args.output_dir, f"{emb_name}_per_chromosome")
            os.makedirs(chrom_dir, exist_ok=True)
            highlight_color = "#800000"  # maroon for all chromosomes
            for ci, chrom in enumerate(chrom_list):
                mask = df["chrom"] == chrom
                n_c = mask.sum()
                other = ~mask
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.scatter(coords[other, 0], coords[other, 1],
                           c="#e0e0e0", s=s*0.5, alpha=0.05, rasterized=True)
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=highlight_color, s=s, alpha=min(alpha + 0.1, 0.8),
                           label=f"{chrom} ({n_c:,})", rasterized=True)
                ax.set_title(f"{prefix} — {chrom} Highlighted ({n_c:,} regions)", fontsize=14)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend(markerscale=max(1, 8/s), fontsize=11, loc="upper left")
                plt.tight_layout()
                out = os.path.join(chrom_dir, f"{emb_name}_{chrom}.png")
                fig.savefig(out, dpi=args.dpi)
                plt.close(fig)
            print(f"  Saved {len(chrom_list)} per-chromosome plots to {chrom_dir}/")

        # By cluster
        clusters = df["cluster"].values
        n_clusters = len(np.unique(clusters))
        cmap_c = plt.cm.get_cmap("tab20", min(n_clusters, 20))
        fig, ax = plt.subplots(figsize=(12, 10))
        for ci in range(n_clusters):
            mask = clusters == ci
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap_c(ci % 20)], s=s, alpha=alpha, rasterized=True)
        ax.set_title(f"{prefix} of SAE Regions — Leiden Clusters (N={n_clusters})", fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_by_cluster.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {out}")

        # By method
        methods = df["method"].unique()
        method_colors = {"zscore": "#1f77b4", "MAD": "#ff7f0e", "mad": "#ff7f0e", "": "#999999", "unknown": "#999999"}
        fig, ax = plt.subplots(figsize=(12, 10))
        for method in sorted(methods):
            mask = df["method"] == method
            c = method_colors.get(method, "#999999")
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=c, s=s, alpha=alpha, label=f"{method} ({mask.sum():,})", rasterized=True)
        ax.set_title(f"{prefix} of SAE Regions — Colored by Detection Method", fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(markerscale=max(1, 6/s), fontsize=9)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_by_method.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {out}")

        # Confidence + region length continuous (log10)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        conf_vals = df["confidence"].values
        sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=conf_vals,
                              cmap="viridis", s=s, alpha=alpha, rasterized=True)
        axes[0].set_title("Confidence")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        plt.colorbar(sc1, ax=axes[0])
        # Add confidence stats
        conf_stats = (f"median={np.median(conf_vals):.1f}  mean={np.mean(conf_vals):.1f}  "
                      f"std={np.std(conf_vals):.1f}\n"
                      f"P5={np.percentile(conf_vals,5):.1f}  P95={np.percentile(conf_vals,95):.1f}  "
                      f"range=[{conf_vals.min():.1f}, {conf_vals.max():.1f}]")
        axes[0].text(0.02, 0.02, conf_stats, transform=axes[0].transAxes, fontsize=8,
                     va="bottom", ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        lengths = (df["genomic_end"] - df["genomic_start"]).values
        sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=np.log10(lengths + 1),
                              cmap="plasma", s=s, alpha=alpha, rasterized=True)
        axes[1].set_title("Region Length (log10 bp)")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        plt.colorbar(sc2, ax=axes[1])

        fig.suptitle(f"{prefix} of SAE Regions — Continuous Properties", fontsize=14)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_continuous.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {out}")

        # Region length LINEAR (clipped at P99 for visibility)
        p99 = np.percentile(lengths, 99)
        lengths_clipped = np.clip(lengths, 0, p99)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sc1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=lengths_clipped,
                              cmap="plasma", s=s, alpha=alpha, rasterized=True)
        axes[0].set_title(f"Region Length — linear, clipped at P99 ({p99:.0f} bp)")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        cb1 = plt.colorbar(sc1, ax=axes[0])
        cb1.set_label("Region Length (bp)")

        # Same but clipped at P95
        p95 = np.percentile(lengths, 95)
        lengths_p95 = np.clip(lengths, 0, p95)
        sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=lengths_p95,
                              cmap="plasma", s=s, alpha=alpha, rasterized=True)
        axes[1].set_title(f"Region Length — linear, clipped at P95 ({p95:.0f} bp)")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        cb2 = plt.colorbar(sc2, ax=axes[1])
        cb2.set_label("Region Length (bp)")

        len_stats = (f"median={np.median(lengths):.0f}  mean={np.mean(lengths):.0f}  "
                     f"P95={p95:.0f}  P99={p99:.0f}  max={lengths.max():.0f} bp\n"
                     f"{100*np.mean(lengths<500):.1f}% < 500bp  "
                     f"{100*np.mean(lengths<1000):.1f}% < 1000bp")
        for ax in axes:
            ax.text(0.02, 0.02, len_stats, transform=ax.transAxes, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        fig.suptitle(f"{prefix} of SAE Regions — Region Length (linear scale)", fontsize=14)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{emb_name}_region_length_linear.png")
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {out}")

        # By genomic start position (normalized per-chromosome)
        if "genomic_start" in df.columns and "chrom" in df.columns:
            starts = df["genomic_start"].values.astype(float).copy()
            for chrom in df["chrom"].unique():
                mask = df["chrom"] == chrom
                chrom_starts = starts[mask]
                rng = chrom_starts.max() - chrom_starts.min()
                if rng > 0:
                    starts[mask] = (chrom_starts - chrom_starts.min()) / rng
            fig, ax = plt.subplots(figsize=(12, 10))
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=starts, cmap="hsv",
                            s=max(s, 1.5), alpha=min(alpha + 0.2, 0.8), rasterized=True)
            plt.colorbar(sc, ax=ax, label="Genomic Position (normalized per-chromosome)")
            ax.set_title(f"{prefix} of SAE Regions — Colored by Genomic Start Position (N={n_total:,})", fontsize=13)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            out = os.path.join(args.output_dir, f"{emb_name}_by_start_position.png")
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            print(f"  Saved {out}")

    wall_time = time.time() - t0
    print(f"\nDone in {wall_time:.1f}s — {len(os.listdir(args.output_dir))} plots in {args.output_dir}")


if __name__ == "__main__":
    main()
