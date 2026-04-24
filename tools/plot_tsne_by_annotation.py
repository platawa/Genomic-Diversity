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
# Known CRISPR and prophage loci in E. coli K-12 MG1655 (NC_000913.3)
# Coordinates are 0-based, half-open
ECOLI_K12_SPECIAL_LOCI = {
    "CP4-6":    (262246, 282260),
    "DLP12":    (556698, 582543),
    "e14":      (1195432, 1211059),
    "Rac":      (1408685, 1433217),
    "Qin":      (1629855, 1651856),
    "CP4-44":   (2064327, 2078613),
    "CPS-53":   (2161314, 2175866),
    "CPZ-55":   (2556942, 2563568),
    "CP4-57":   (2747020, 2773709),
    "KpLE2":    (3449036, 3467424),
    "CRISPR-I":  (2875441, 2876516),
    "CRISPR-II": (2877618, 2878569),
}

SPECIAL_LOCUS_COLORS = {
    "CRISPR-I":  "#00CC00",  # bright green
    "CRISPR-II": "#CC00CC",  # bright magenta
    "CP4-6":     "#1f77b4",
    "DLP12":     "#ff7f0e",
    "e14":       "#2ca02c",
    "Rac":       "#d62728",
    "Qin":       "#9467bd",
    "CP4-44":    "#8c564b",
    "CPS-53":    "#e377c2",
    "CPZ-55":    "#7f7f7f",
    "CP4-57":    "#bcbd22",
    "KpLE2":     "#17becf",
}


def check_special_locus(start, end, loci):
    """Return list of overlapping special locus names."""
    hits = []
    for name, (ls, le) in loci.items():
        if start < le and end > ls:
            hits.append(name)
    return hits


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

    # Determine GTF chromosome ID
    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
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

    # Load GTF features (once, shared across both variants)
    intervals = load_gtf_features(args.gtf, chrom_id)
    if not any(intervals.values()):
        print(f"WARNING: No features found for {chrom_id} in GTF. "
              f"Check chromosome naming.")

    # Process both latent_analysis and latent_analysis_normalized
    for variant in ["latent_analysis", "latent_analysis_normalized"]:
        assignments_path = os.path.join(sae_run, variant, "data",
                                        "cluster_assignments.tsv")
        if not os.path.exists(assignments_path):
            print(f"Skipping {variant}: {assignments_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {variant}...")
        print(f"{'='*60}")

        regions = pd.read_csv(assignments_path, sep="\t", comment="#")
        print(f"Loaded {len(regions)} regions with t-SNE coordinates")

        if "tsne_1" not in regions.columns:
            print(f"  Skipping {variant}: no t-SNE coordinates")
            continue

        # Classify each region
        regions["annotation"] = regions.apply(
            lambda r: classify_region(r.genomic_start, r.genomic_end, intervals),
            axis=1)

        print("\nRegion annotation counts:")
        print(regions.annotation.value_counts().to_string())

        # ── Shared plot settings ──
        annotation_colors = {
            "CDS": "#E74C3C",
            "UTR/exon": "#F39C12",
            "Intron": "#5DADE2",
            "Intergenic": "#BDC3C7",
        }
        annotation_order = ["CDS", "UTR/exon", "Intron", "Intergenic"]

        out_dir = os.path.join(sae_run, variant, "plots")
        os.makedirs(out_dir, exist_ok=True)

        # Fixed dot size across all plots for visual comparability.
        n_pts = len(regions)
        dot_size = 6
        dot_alpha = 0.7 if n_pts < 2000 else (0.6 if n_pts < 10000 else 0.5)

        # ── Plot: t-SNE colored by annotation ──
        fig, ax = plt.subplots(figsize=(14, 12))
        for label in annotation_order:
            mask = regions.annotation == label
            if mask.any():
                ax.scatter(regions.loc[mask, "tsne_1"],
                           regions.loc[mask, "tsne_2"],
                           c=annotation_colors[label],
                           label=f"{label} ({mask.sum()})",
                           s=dot_size, alpha=dot_alpha,
                           edgecolors="none", rasterized=True)
        ax.legend(fontsize=11, markerscale=4)
        suffix = " (normalized)" if "normalized" in variant else ""
        ax.set_title(f"t-SNE of SAE Region Fingerprints — {args.chrom} "
                     f"(N={len(regions)}){suffix}\nColored by Genomic Annotation",
                     fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()

        out_path = os.path.join(out_dir, "tsne_by_annotation.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {out_path}")
        plt.close()

        # ── Plot: t-SNE colored by region length (human chromosomes only) ──
        if args.chrom.startswith("chr") and "region_length" in regions.columns:
            from matplotlib.colors import LinearSegmentedColormap
            blue_red_cmap = LinearSegmentedColormap.from_list('blue_red',
                ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b'])

            fig, ax = plt.subplots(figsize=(14, 12))
            sc = ax.scatter(regions.tsne_1, regions.tsne_2,
                            c=regions.region_length,
                            cmap=blue_red_cmap,
                            s=dot_size, alpha=dot_alpha,
                            edgecolors="none", rasterized=True)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Region Length (bp)", fontsize=13)
            cbar.ax.tick_params(labelsize=11)
            ax.set_title(f"t-SNE of SAE Region Fingerprints — {args.chrom} "
                         f"(N={len(regions)}){suffix}\nColored by Region Length",
                         fontsize=13)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            plt.tight_layout()

            out_len = os.path.join(out_dir, "tsne_by_region_length.png")
            fig.savefig(out_len, dpi=200, bbox_inches="tight")
            print(f"Saved: {out_len}")
            plt.close()

        # ── Plot: t-SNE colored by annotation + confidence side by side ──
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        ax = axes[0]
        for label in annotation_order:
            mask = regions.annotation == label
            if mask.any():
                ax.scatter(regions.loc[mask, "tsne_1"],
                           regions.loc[mask, "tsne_2"],
                           c=annotation_colors[label],
                           label=f"{label} ({mask.sum()})",
                           s=dot_size, alpha=dot_alpha,
                           edgecolors="none", rasterized=True)
        ax.legend(fontsize=10, markerscale=4)
        ax.set_title("Genomic Annotation")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        ax = axes[1]
        sc = ax.scatter(regions.tsne_1, regions.tsne_2,
                        c=regions.confidence, cmap="viridis",
                        s=dot_size, alpha=dot_alpha,
                        edgecolors="none", rasterized=True)
        plt.colorbar(sc, ax=ax, label="Start Confidence")
        ax.set_title("Start Confidence")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        fig.suptitle(f"SAE Region Fingerprints — {args.chrom} "
                     f"(N={len(regions)}){suffix}",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        out_path2 = os.path.join(out_dir, "tsne_annotation_and_confidence.png")
        fig.savefig(out_path2, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path2}")
        plt.close()

        # ── Plot: t-SNE with CRISPR/prophage loci highlighted (E. coli only) ──
        if args.chrom == "NC_000913.3":
            regions["special_locus"] = regions.apply(
                lambda r: check_special_locus(r.genomic_start, r.genomic_end,
                                              ECOLI_K12_SPECIAL_LOCI),
                axis=1)

            fig, ax = plt.subplots(figsize=(14, 12))

            # Background: all points in light gray
            bg_size = max(dot_size * 0.8, 2)
            ax.scatter(regions.tsne_1, regions.tsne_2,
                       c="#E0E0E0", s=bg_size, alpha=0.3,
                       edgecolors="none", rasterized=True, label=None)

            # Overlay each locus type
            # Plot prophages first, then CRISPR on top
            locus_order = [k for k in ECOLI_K12_SPECIAL_LOCI
                           if not k.startswith("CRISPR")] + \
                          [k for k in ECOLI_K12_SPECIAL_LOCI
                           if k.startswith("CRISPR")]

            for locus_name in locus_order:
                mask = regions.special_locus.apply(lambda x: locus_name in x)
                if not mask.any():
                    continue
                is_crispr = locus_name.startswith("CRISPR")
                ax.scatter(regions.loc[mask, "tsne_1"],
                           regions.loc[mask, "tsne_2"],
                           c=SPECIAL_LOCUS_COLORS[locus_name],
                           label=f"{locus_name} ({mask.sum()})",
                           s=25 if is_crispr else 15,
                           alpha=0.9 if is_crispr else 0.7,
                           edgecolors="black" if is_crispr else "none",
                           linewidths=0.5 if is_crispr else 0,
                           zorder=10 if is_crispr else 5,
                           rasterized=True)

            ax.legend(fontsize=10, markerscale=2, loc="upper right",
                      title="Special Loci", title_fontsize=11)
            ax.set_title(f"t-SNE of SAE Region Fingerprints — {args.chrom} "
                         f"(N={len(regions)}){suffix}\n"
                         f"CRISPR & Prophage Loci Highlighted",
                         fontsize=13)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            plt.tight_layout()

            out_locus = os.path.join(out_dir, "tsne_by_special_locus.png")
            fig.savefig(out_locus, dpi=200, bbox_inches="tight")
            print(f"Saved: {out_locus}")
            plt.close()

        # ── Save annotated table ──
        out_tsv = os.path.join(sae_run, variant, "data",
                               "cluster_assignments_annotated.tsv")
        regions.to_csv(out_tsv, sep="\t", index=False)
        print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
