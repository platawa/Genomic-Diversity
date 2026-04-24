#!/usr/bin/env python3
"""
plot_box_investigation.py

Generate summary plots from a directory produced by investigate_tsne_region.py.

Reads:
  regions_in_box.tsv
  summary.tsv
  top_genes_by_class.tsv          (if present)
  intergenic_flanking_pairs.tsv   (if present)

Emits:
  top_intron_genes.png            horizontal bar of top-N genes with intron regions
  top_intergenic_pairs.png        horizontal bar of top-N flanking-gene pairs
  distance_to_gene_histogram.png  log-scale histogram of up/down distances
  chromosome_position.png         region density along the chromosome
  annotation_enrichment.png       log2 enrichment bar chart

Usage:
  python tools/plot_box_investigation.py \
      --inv-dir results/chr1/sae/latent_analysis/investigations/red_island \
      --chrom chr1
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_top_genes(regions, out_path, top_n=25):
    sub = regions[(regions.annotation == "Intron") & regions.upstream_gene.notna()]
    counts = sub.upstream_gene.value_counts().head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(counts))))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#5DADE2")
    ax.set_xlabel("Number of SAE regions")
    ax.set_title(f"Top {top_n} genes with intronic SAE regions in box\n"
                 f"(n={len(sub)} intronic regions total)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top_intergenic_pairs(pairs_path, out_path, top_n=20):
    if not os.path.isfile(pairs_path):
        return
    df = pd.read_csv(pairs_path, sep="\t").head(top_n)
    if df.empty:
        return
    labels = [f"{up or '∅'} ↔ {dn or '∅'}"
              for up, dn in zip(df.upstream_gene, df.downstream_gene)]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
    ax.barh(labels[::-1], df.n_regions.values[::-1], color="#BDC3C7")
    ax.set_xlabel("Number of SAE regions")
    ax.set_title(f"Top {top_n} flanking-gene pairs for intergenic regions in box")
    # Annotate each bar with median distances
    for i, row in enumerate(df[::-1].itertuples()):
        ax.text(row.n_regions, i,
                f"  up ~{int(row.median_upstream_dist/1000)}kb, "
                f"dn ~{int(row.median_downstream_dist/1000)}kb",
                va="center", fontsize=8, color="#555")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_distance_histogram(regions, out_path):
    ig = regions[regions.annotation == "Intergenic"]
    if ig.empty:
        return
    up = ig.upstream_dist.dropna()
    dn = ig.downstream_dist.dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(2, 7, 40)
    ax.hist(up[up > 0], bins=bins, alpha=0.55, label=f"upstream (n={len(up)})",
            color="#E74C3C")
    ax.hist(dn[dn > 0], bins=bins, alpha=0.55, label=f"downstream (n={len(dn)})",
            color="#5DADE2")
    ax.set_xscale("log")
    ax.axvline(up.median(), ls="--", color="#E74C3C", lw=1,
               label=f"up median {int(up.median()):,} bp")
    ax.axvline(dn.median(), ls="--", color="#5DADE2", lw=1,
               label=f"dn median {int(dn.median()):,} bp")
    ax.set_xlabel("Distance to nearest flanking gene (bp, log)")
    ax.set_ylabel("Intergenic regions in box")
    ax.set_title("Distance-to-gene distribution for intergenic regions in box")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_chromosome_position(regions, out_path, chrom, n_bins=200):
    if regions.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    classes = ["Intron", "Intergenic", "CDS"]
    colors = {"CDS": "#E74C3C", "Intron": "#5DADE2", "Intergenic": "#BDC3C7"}
    for ax, cls in zip(axes, classes):
        sub = regions[regions.annotation == cls]
        if sub.empty:
            ax.set_title(f"{cls}: n=0")
            continue
        ax.hist(sub.genomic_start, bins=n_bins, color=colors[cls])
        ax.set_ylabel(cls)
        ax.set_title(f"{cls}: n={len(sub)}")
    axes[-1].set_xlabel(f"{chrom} genomic position (bp)")
    fig.suptitle(f"Box region density along {chrom}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_enrichment(summary_path, out_path):
    if not os.path.isfile(summary_path):
        return
    df = pd.read_csv(summary_path, sep="\t")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in df.enrichment_log2]
    ax.barh(df["class"], df.enrichment_log2, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("log2( in-box % / rest-of-chrom % )")
    ax.set_title("Annotation enrichment in box (vs rest of chromosome)")
    for i, (v, n) in enumerate(zip(df.enrichment_log2, df.in_box)):
        ax.text(v, i, f"  {2**v:.2f}×  (n={n})", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inv-dir", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--top-n", type=int, default=25)
    args = p.parse_args()

    regions = pd.read_csv(os.path.join(args.inv_dir, "regions_in_box.tsv"),
                          sep="\t")
    plot_top_genes(regions,
                   os.path.join(args.inv_dir, "top_intron_genes.png"),
                   top_n=args.top_n)
    plot_top_intergenic_pairs(
        os.path.join(args.inv_dir, "intergenic_flanking_pairs.tsv"),
        os.path.join(args.inv_dir, "top_intergenic_pairs.png"),
        top_n=args.top_n)
    plot_distance_histogram(regions,
                            os.path.join(args.inv_dir, "distance_to_gene_histogram.png"))
    plot_chromosome_position(regions,
                             os.path.join(args.inv_dir, "chromosome_position.png"),
                             args.chrom)
    plot_enrichment(os.path.join(args.inv_dir, "summary.tsv"),
                    os.path.join(args.inv_dir, "annotation_enrichment.png"))


if __name__ == "__main__":
    main()
