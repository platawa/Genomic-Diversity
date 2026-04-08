#!/usr/bin/env python3
"""
top_features_genome_spread.py

Analyze how top SAE features are distributed across the genome using
existing max-pooled vectors (no GPU needed).

For each top feature:
  - How many regions activate it across chromosomes?
  - What is the distribution of activations?
  - Which annotation types (CDS/Intron/Intergenic/UTR) activate it most?
  - Per-chromosome activation counts and means

Uses:
  genome_wide_sae_stats_corrected.npz (to identify top features)
  per-chromosome latent_analysis[_normalized]/data/maxpooled_vectors.npy
  per-chromosome latent_analysis[_normalized]/data/cluster_assignments.tsv (for annotations)

Usage:
    python tools/top_features_genome_spread.py \\
        --results_dir results/ \\
        --global_stats results/_genome_sae_stats/.../genome_wide_sae_stats_corrected.npz \\
        --top_n 20 \\
        --output_dir results/_genome_wide/top_features_spread/
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_HUMAN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def load_chrom_data(results_dir, chrom, latent_subdir):
    """Load maxpooled vectors and cluster assignments for a chromosome."""
    latent_dir = os.path.join(results_dir, chrom, "sae", latent_subdir, "data")
    mp_path = os.path.join(latent_dir, "maxpooled_vectors.npy")
    ca_path = os.path.join(latent_dir, "cluster_assignments.tsv")

    if not os.path.exists(mp_path):
        return None, None
    vectors = np.load(mp_path)

    ca_df = None
    # Prefer annotated TSV (has annotation column)
    ca_annotated = os.path.join(latent_dir, "cluster_assignments_annotated.tsv")
    if os.path.exists(ca_annotated):
        ca_df = pd.read_csv(ca_annotated, sep="\t", comment="#")
    elif os.path.exists(ca_path):
        ca_df = pd.read_csv(ca_path, sep="\t", comment="#")
    return vectors, ca_df


def main():
    parser = argparse.ArgumentParser(description="Analyze top SAE feature spread across genome")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--global_stats", required=True, help="Path to genome_wide_sae_stats_corrected.npz")
    parser.add_argument("--latent_subdir", default="latent_analysis_normalized")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top features to analyze")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chroms", nargs="+", default=None, help="Chromosomes (default: all human)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    chroms = args.chroms or ALL_HUMAN_CHROMS

    # Load global stats to identify top features
    logger.info(f"Loading global stats from {args.global_stats}")
    stats = np.load(args.global_stats, allow_pickle=True)
    nuc_mean = stats["nuc_mean"]
    top_features = np.argsort(nuc_mean)[-args.top_n:][::-1]
    logger.info(f"Top {args.top_n} features by genome-wide mean activation:")
    for fid in top_features:
        logger.info(f"  f{fid}: mean={nuc_mean[fid]:.4f}")

    # Collect per-chromosome data
    all_rows = []  # per-feature, per-chrom stats
    feature_activation_distributions = {int(fid): [] for fid in top_features}
    feature_annotation_counts = {int(fid): {} for fid in top_features}

    for chrom in chroms:
        vectors, ca_df = load_chrom_data(args.results_dir, chrom, args.latent_subdir)
        if vectors is None:
            logger.warning(f"  {chrom}: no data, skipping")
            continue

        n_regions = vectors.shape[0]
        logger.info(f"  {chrom}: {n_regions} regions")

        # Get annotation column if available
        ann_col = None
        if ca_df is not None:
            for col in ["annotation", "ann"]:
                if col in ca_df.columns:
                    ann_col = col
                    break

        for fid in top_features:
            fid = int(fid)
            activations = vectors[:, fid]
            active_mask = activations > 0
            n_active = int(active_mask.sum())
            mean_when_active = float(activations[active_mask].mean()) if n_active > 0 else 0.0
            max_val = float(activations.max())

            all_rows.append({
                "feature_id": fid,
                "chrom": chrom,
                "n_regions": n_regions,
                "n_active": n_active,
                "pct_active": 100.0 * n_active / n_regions if n_regions > 0 else 0,
                "mean_when_active": mean_when_active,
                "max_activation": max_val,
            })

            # Collect for distribution
            feature_activation_distributions[fid].extend(activations[active_mask].tolist())

            # Annotation breakdown
            if ann_col and ca_df is not None and n_active > 0:
                for ann_type in ca_df[ann_col].unique():
                    ann_mask = (ca_df[ann_col] == ann_type).values & active_mask
                    if ann_mask.sum() > 0:
                        key = str(ann_type)
                        feature_annotation_counts[fid][key] = (
                            feature_annotation_counts[fid].get(key, 0) + int(ann_mask.sum())
                        )

    # Save per-chrom stats
    df = pd.DataFrame(all_rows)
    tsv_path = os.path.join(args.output_dir, "top_features_per_chrom.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"Saved {tsv_path}")

    # === PLOTS ===

    # 1. Heatmap: feature × chromosome (% active)
    pivot = df.pivot_table(index="feature_id", columns="chrom", values="pct_active", fill_value=0)
    # Sort chroms
    chrom_order = [c for c in ALL_HUMAN_CHROMS if c in pivot.columns]
    pivot = pivot[chrom_order]

    fig, ax = plt.subplots(figsize=(16, max(6, len(top_features) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(chrom_order)))
    ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"f{fid}" for fid in pivot.index], fontsize=8)
    ax.set_title(f"Top {args.top_n} Features — % Regions Active per Chromosome", fontsize=13)
    plt.colorbar(im, ax=ax, label="% regions active")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "heatmap_pct_active_by_chrom.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved heatmap_pct_active_by_chrom.png")

    # 2. Heatmap: feature × chromosome (mean activation when active)
    pivot_mean = df.pivot_table(index="feature_id", columns="chrom", values="mean_when_active", fill_value=0)
    pivot_mean = pivot_mean[[c for c in chrom_order if c in pivot_mean.columns]]

    fig, ax = plt.subplots(figsize=(16, max(6, len(top_features) * 0.4)))
    im = ax.imshow(pivot_mean.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_yticklabels([f"f{fid}" for fid in pivot_mean.index], fontsize=8)
    ax.set_title(f"Top {args.top_n} Features — Mean Activation (when active) per Chromosome", fontsize=13)
    plt.colorbar(im, ax=ax, label="Mean activation")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "heatmap_mean_activation_by_chrom.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved heatmap_mean_activation_by_chrom.png")

    # 3. Bar chart: annotation breakdown per top feature
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    ann_colors = {"CDS": "#e41a1c", "UTR/exon": "#ff7f00", "Intron": "#377eb8", "Intergenic": "#999999"}
    for idx, fid in enumerate(top_features[:20]):
        fid = int(fid)
        ax = axes[idx]
        counts = feature_annotation_counts.get(fid, {})
        if not counts:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            ax.set_title(f"f{fid}", fontsize=9)
            continue
        labels = list(counts.keys())
        values = [counts[l] for l in labels]
        colors = [ann_colors.get(l, "#cccccc") for l in labels]
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"f{fid} (mean={nuc_mean[fid]:.2f})", fontsize=9)
        ax.set_ylabel("# regions", fontsize=7)
    fig.suptitle(f"Top {min(args.top_n, 20)} Features — Annotation Breakdown", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "annotation_breakdown_per_feature.png"), dpi=200)
    plt.close(fig)
    logger.info("Saved annotation_breakdown_per_feature.png")

    # 4. Summary stats
    summary = []
    for fid in top_features:
        fid = int(fid)
        fdf = df[df["feature_id"] == fid]
        total_active = int(fdf["n_active"].sum())
        total_regions = int(fdf["n_regions"].sum())
        summary.append({
            "feature_id": fid,
            "genome_mean": float(nuc_mean[fid]),
            "total_regions_active": total_active,
            "total_regions": total_regions,
            "pct_active_genome": 100.0 * total_active / total_regions if total_regions > 0 else 0,
            "n_chroms_active": int((fdf["n_active"] > 0).sum()),
            "annotation_breakdown": json.dumps(feature_annotation_counts.get(fid, {})),
        })
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, "top_features_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Saved {summary_path}")

    logger.info(f"\nDone. {len(os.listdir(args.output_dir))} files in {args.output_dir}")


if __name__ == "__main__":
    main()
