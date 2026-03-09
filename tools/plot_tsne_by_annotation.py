#!/usr/bin/env python3
"""
plot_tsne_by_annotation.py

Re-plot saved t-SNE embeddings from SAE latent analysis, colored by
genomic annotation (CDS, UTR/exon, intron, intergenic) from a GTF file.

No GPU needed — reads pre-computed data from the SAE run.

Usage:
    python tools/plot_tsne_by_annotation.py \
        --sae_run results/chr22/sae/20260309_131134_max1000_conf8.0 \
        --gtf /path/to/genomic.gtf \
        --chrom chr22

    # Auto-discover latest SAE run:
    python tools/plot_tsne_by_annotation.py \
        --auto --chrom chr22 \
        --gtf /path/to/genomic.gtf
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chromosome name → RefSeq accession (for GTF matching)
CHROM_MAP = {
    "chr1": "NC_000001.11", "chr2": "NC_000002.12", "chr3": "NC_000003.12",
    "chr4": "NC_000004.12", "chr5": "NC_000005.10", "chr6": "NC_000006.12",
    "chr7": "NC_000007.14", "chr8": "NC_000008.11", "chr9": "NC_000009.12",
    "chr10": "NC_000010.11", "chr11": "NC_000011.10", "chr12": "NC_000012.12",
    "chr13": "NC_000013.11", "chr14": "NC_000014.9", "chr15": "NC_000015.10",
    "chr16": "NC_000016.10", "chr17": "NC_000017.11", "chr18": "NC_000018.10",
    "chr19": "NC_000019.10", "chr20": "NC_000020.11", "chr21": "NC_000021.9",
    "chr22": "NC_000022.11", "chrX": "NC_000023.11", "chrY": "NC_000024.10",
}


def load_gtf_features(gtf_path, chrom_id):
    """Load CDS, exon, and gene intervals from a GTF for one chromosome."""
    intervals = defaultdict(list)
    n_total = 0
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            if parts[0] != chrom_id:
                continue
            ftype = parts[2]
            if ftype in ("CDS", "exon", "gene"):
                intervals[ftype].append((int(parts[3]), int(parts[4])))
                n_total += 1
    print(f"Loaded {n_total} GTF features for {chrom_id} "
          f"(CDS={len(intervals['CDS'])}, exon={len(intervals['exon'])}, "
          f"gene={len(intervals['gene'])})")
    return intervals


def classify_region(start, end, intervals):
    """Classify a region by its midpoint overlap with GTF features.

    Priority: CDS > UTR/exon > Intron > Intergenic
    """
    mid = (start + end) // 2
    for s, e in intervals["CDS"]:
        if s <= mid <= e:
            return "CDS"
    for s, e in intervals["exon"]:
        if s <= mid <= e:
            return "UTR/exon"
    for s, e in intervals["gene"]:
        if s <= mid <= e:
            return "Intron"
    return "Intergenic"


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot SAE t-SNE colored by genomic annotation")
    parser.add_argument("--sae_run", default=None,
                        help="Path to SAE run directory")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover latest COMPLETED SAE run")
    parser.add_argument("--chrom", required=True,
                        help="Chromosome name (e.g., chr22, NC_000913.3)")
    parser.add_argument("--gtf", required=True,
                        help="Path to GTF annotation file")
    parser.add_argument("--results_dir", default="./results",
                        help="Root results directory (default: ./results)")
    args = parser.parse_args()

    # Find SAE run directory
    if args.auto:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from results_utils import find_latest_completed
        sae_run = find_latest_completed(args.results_dir, args.chrom, "sae")
        if sae_run is None:
            print(f"ERROR: No COMPLETED SAE run for {args.chrom} "
                  f"in {args.results_dir}/{args.chrom}/sae/")
            sys.exit(1)
        print(f"Using SAE run: {sae_run}")
    elif args.sae_run:
        sae_run = args.sae_run
    else:
        parser.error("Provide --sae_run or use --auto")

    # Load cluster assignments (has t-SNE coordinates + genomic coords)
    assignments_path = os.path.join(sae_run, "latent_analysis", "data",
                                    "cluster_assignments.tsv")
    if not os.path.exists(assignments_path):
        print(f"ERROR: {assignments_path} not found. "
              f"Run SAE with --run_latent_analysis first.")
        sys.exit(1)

    regions = pd.read_csv(assignments_path, sep="\t", comment="#")
    print(f"Loaded {len(regions)} regions with t-SNE coordinates")

    # Determine GTF chromosome ID
    # Try the chrom name directly first, then map to RefSeq
    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
    # Peek at GTF to see if the chrom_id exists, otherwise try the raw name
    with open(args.gtf) as f:
        for line in f:
            if line.startswith("#"):
                continue
            gtf_chrom = line.split("\t")[0]
            if gtf_chrom == chrom_id:
                break
            if gtf_chrom == args.chrom:
                chrom_id = args.chrom
                break

    # Load GTF features
    intervals = load_gtf_features(args.gtf, chrom_id)
    if not any(intervals.values()):
        print(f"WARNING: No features found for {chrom_id} in GTF. "
              f"Check chromosome naming.")

    # Classify each region
    regions["annotation"] = regions.apply(
        lambda r: classify_region(r.genomic_start, r.genomic_end, intervals),
        axis=1)

    print("\nRegion annotation counts:")
    print(regions.annotation.value_counts().to_string())

    # ── Plot: t-SNE colored by annotation ──
    annotation_colors = {
        "CDS": "#e41a1c",
        "UTR/exon": "#ff7f00",
        "Intron": "#377eb8",
        "Intergenic": "#999999",
    }
    # Order for legend
    annotation_order = ["CDS", "UTR/exon", "Intron", "Intergenic"]

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in annotation_order:
        mask = regions.annotation == label
        if mask.any():
            ax.scatter(regions.loc[mask, "tsne_1"],
                       regions.loc[mask, "tsne_2"],
                       c=annotation_colors[label],
                       label=f"{label} ({mask.sum()})",
                       s=20, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=11, markerscale=2)
    ax.set_title(f"t-SNE of SAE Region Fingerprints — {args.chrom} "
                 f"(N={len(regions)})\nColored by Genomic Annotation",
                 fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    out_dir = os.path.join(sae_run, "latent_analysis", "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tsne_by_annotation.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # ── Plot: t-SNE colored by annotation + confidence side by side ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: annotation
    ax = axes[0]
    for label in annotation_order:
        mask = regions.annotation == label
        if mask.any():
            ax.scatter(regions.loc[mask, "tsne_1"],
                       regions.loc[mask, "tsne_2"],
                       c=annotation_colors[label],
                       label=f"{label} ({mask.sum()})",
                       s=20, alpha=0.7, edgecolors="none")
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title("Genomic Annotation")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Right: confidence
    ax = axes[1]
    sc = ax.scatter(regions.tsne_1, regions.tsne_2,
                    c=regions.confidence, cmap="viridis",
                    s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Start Confidence")
    ax.set_title("Start Confidence")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.suptitle(f"SAE Region Fingerprints — {args.chrom} (N={len(regions)})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "tsne_annotation_and_confidence.png")
    fig.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.close()

    # ── Save annotated table ──
    out_tsv = os.path.join(sae_run, "latent_analysis", "data",
                           "cluster_assignments_annotated.tsv")
    regions.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
