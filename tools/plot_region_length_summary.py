#!/usr/bin/env python3
"""Cross-chromosome region length uniformity analysis.

Aggregates region_length_stats.json across all chromosomes and produces
a summary plot showing whether entropy drop regions are uniform in length.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latent_dir(results_dir, chrom):
    """Find latent analysis directory."""
    direct = os.path.join(results_dir, chrom, "sae", "latent_analysis")
    if os.path.isdir(os.path.join(direct, "data")):
        return direct
    sae_dir = os.path.join(results_dir, chrom, "sae")
    if os.path.isdir(sae_dir):
        for entry in sorted(os.listdir(sae_dir), reverse=True):
            candidate = os.path.join(sae_dir, entry, "latent_analysis")
            if os.path.isdir(os.path.join(candidate, "data")):
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--all_human", action="store_true")
    parser.add_argument("--include_bacteria", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=[], help="Chromosomes to exclude (e.g. chr19)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    chroms = []
    if args.all_human:
        chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    if args.include_bacteria:
        chroms += ["NC_000913.3", "NC_000964.3"]
    if args.exclude:
        chroms = [c for c in chroms if c not in args.exclude]

    # Collect stats from each chromosome
    all_stats = []
    all_lengths = {}

    for chrom in chroms:
        latent_dir = find_latent_dir(args.results_dir, chrom)
        if latent_dir is None:
            print(f"  {chrom}: no latent_analysis, skipping")
            continue

        # Try loading stats JSON
        stats_path = os.path.join(latent_dir, "data", "region_length_stats.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            stats["chrom"] = chrom
            all_stats.append(stats)

        # Also load actual lengths from cluster_assignments.tsv
        ca_path = os.path.join(latent_dir, "data", "cluster_assignments.tsv")
        if os.path.isfile(ca_path):
            try:
                ca = pd.read_csv(ca_path, sep="\t", comment="#")
                if len(ca) == 0:
                    print(f"  {chrom}: empty TSV, skipping")
                    continue
                if "region_length" in ca.columns:
                    all_lengths[chrom] = ca["region_length"].values
                elif "genomic_start" in ca.columns and "genomic_end" in ca.columns:
                    all_lengths[chrom] = (ca["genomic_end"] - ca["genomic_start"]).values
            except Exception as e:
                print(f"  {chrom}: error reading TSV: {e}, skipping")

    if not all_lengths:
        print("No region length data found.")
        return

    # Summary statistics table
    summary_rows = []
    for chrom, lengths in all_lengths.items():
        summary_rows.append({
            "chrom": chrom,
            "n_regions": len(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths),
            "min": np.min(lengths),
            "max": np.max(lengths),
            "q25": np.percentile(lengths, 25),
            "q75": np.percentile(lengths, 75),
            "cv": np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "region_length_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False, float_format="%.1f")
    print(f"Saved summary: {summary_path}")

    # Plot 1: Box plot of region lengths per chromosome
    fig, ax = plt.subplots(figsize=(16, 6))
    chrom_order = [c for c in chroms if c in all_lengths]
    data_for_box = [all_lengths[c] for c in chrom_order]
    bp = ax.boxplot(data_for_box, labels=chrom_order, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4a90d9")
        patch.set_alpha(0.7)
    ax.set_ylabel("Region Length (bp)", fontsize=12)
    ax.set_title("Entropy Drop Region Length Distribution by Chromosome", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "region_length_boxplot_by_chrom.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Plot 2: Histogram of all region lengths combined
    all_combined = np.concatenate(list(all_lengths.values()))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(all_combined, bins=100, color="#4a90d9", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(all_combined), color="red", linestyle="--",
               label=f"Median: {np.median(all_combined):.0f} bp")
    ax.axvline(np.mean(all_combined), color="orange", linestyle="--",
               label=f"Mean: {np.mean(all_combined):.0f} bp")
    ax.set_xlabel("Region Length (bp)")
    ax.set_ylabel("Count")
    ax.set_title(f"All Regions Combined (N={len(all_combined):,})")
    ax.legend()

    ax = axes[1]
    ax.hist(all_combined, bins=100, color="#4a90d9", alpha=0.7, edgecolor="white", log=True)
    ax.axvline(np.median(all_combined), color="red", linestyle="--")
    ax.axvline(np.mean(all_combined), color="orange", linestyle="--")
    ax.set_xlabel("Region Length (bp)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Log Scale")

    fig.suptitle(f"Region Length Distribution — {len(all_lengths)} Chromosomes\n"
                 f"CV={np.std(all_combined)/np.mean(all_combined):.3f}, "
                 f"Range: [{np.min(all_combined)}, {np.max(all_combined)}] bp",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "region_length_histogram_combined.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Plot 3: Mean ± std per chromosome
    fig, ax = plt.subplots(figsize=(16, 6))
    x = range(len(summary_df))
    ax.bar(x, summary_df["mean"], yerr=summary_df["std"], capsize=3,
           color="#4a90d9", alpha=0.7, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["chrom"], rotation=45)
    ax.set_ylabel("Region Length (bp)")
    ax.set_title("Mean Region Length ± Std by Chromosome", fontsize=14)
    ax.axhline(np.mean(all_combined), color="red", linestyle="--", alpha=0.5,
               label=f"Global mean: {np.mean(all_combined):.0f} bp")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(args.output_dir, "region_length_mean_by_chrom.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary
    print(f"\n=== Region Length Uniformity Summary ===")
    print(f"Total regions: {len(all_combined):,}")
    print(f"Global mean: {np.mean(all_combined):.1f} bp")
    print(f"Global median: {np.median(all_combined):.1f} bp")
    print(f"Global std: {np.std(all_combined):.1f} bp")
    print(f"CV (coefficient of variation): {np.std(all_combined)/np.mean(all_combined):.3f}")
    print(f"Range: [{np.min(all_combined)}, {np.max(all_combined)}] bp")
    uniform = "YES" if np.std(all_combined)/np.mean(all_combined) < 0.5 else "NO"
    print(f"Approximately uniform: {uniform}")


if __name__ == "__main__":
    main()
