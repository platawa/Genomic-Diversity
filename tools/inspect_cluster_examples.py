#!/usr/bin/env python3
"""
inspect_cluster_examples.py

For each Leiden cluster, pick representative regions and plot per-nucleotide
SAE feature activation traces.  Helps answer:
  - Do clusters share the same top features or different ones?
  - Are regions clustering by similar activation patterns or by motifs?

Reads:
  latent_analysis[_normalized]/data/cluster_assignments.tsv
  latent_analysis[_normalized]/data/maxpooled_vectors.npy
  <merged_run>/data/feature_matrices.npz   (per-nucleotide activations)

Outputs per cluster:
  cluster_<id>_examples.png   — top-K feature traces for each example region
Cross-cluster comparison:
  cross_cluster_comparison.png — same features plotted across clusters

Usage:
    python tools/inspect_cluster_examples.py \\
        --chrom chr12 --results_dir results/ --latent_subdir latent_analysis_normalized \\
        --n_examples 3 --top_k 5
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def find_latest_merged_run(sae_dir):
    """Find the latest merged SAE run directory that has feature_matrices.npz."""
    candidates = []
    for d in sorted(Path(sae_dir).iterdir(), reverse=True):
        if not d.is_dir():
            continue
        fm = d / "data" / "feature_matrices.npz"
        if fm.exists() and "merged" in d.name:
            candidates.append(d)
    if candidates:
        return candidates[0]
    # Fall back to any dir with feature_matrices.npz
    for d in sorted(Path(sae_dir).iterdir(), reverse=True):
        if not d.is_dir():
            continue
        fm = d / "data" / "feature_matrices.npz"
        if fm.exists():
            return d
    return None


def pick_representative_regions(cluster_regions_df, maxpooled, n_examples):
    """Pick regions closest to the cluster centroid (by cosine similarity)."""
    indices = cluster_regions_df.index.values
    if len(indices) <= n_examples:
        return indices

    vectors = maxpooled[indices]
    centroid = vectors.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    sims = (vectors / norms) @ centroid_norm
    top_idx = np.argsort(sims)[-n_examples:][::-1]
    return indices[top_idx]


def get_top_features(maxpooled_vector, top_k):
    """Get top-K feature indices by activation in a max-pooled vector."""
    nonzero = np.nonzero(maxpooled_vector)[0]
    if len(nonzero) == 0:
        return []
    k = min(top_k, len(nonzero))
    top_idx = nonzero[np.argsort(maxpooled_vector[nonzero])[-k:][::-1]]
    return top_idx.tolist()


def plot_feature_traces(feature_matrix, feature_ids, region_info, ax, title=None):
    """Plot per-nucleotide activation traces for selected features.

    feature_matrix: shape (seq_len, n_features)
    feature_ids: list of feature indices to plot
    """
    seq_len = feature_matrix.shape[0]
    x = np.arange(seq_len)

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(feature_ids), 1)))
    for i, fid in enumerate(feature_ids):
        trace = feature_matrix[:, fid]
        ax.plot(x, trace, color=colors[i], alpha=0.8, linewidth=0.8,
                label=f"f{fid} (max={trace.max():.1f})")

    ax.set_xlabel("Position in region (nucleotides)")
    ax.set_ylabel("SAE feature activation")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlim(0, seq_len)

    # Add genomic coordinate info
    start = region_info.get("genomic_start", "?")
    end = region_info.get("genomic_end", "?")
    length = region_info.get("region_length", seq_len)
    ax.text(0.02, 0.98, f"{start:,}–{end:,} ({length:,} bp)",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))


def main():
    parser = argparse.ArgumentParser(description="Inspect cluster examples with per-nucleotide feature traces")
    parser.add_argument("--chrom", required=True, help="Chromosome name (e.g. chr12)")
    parser.add_argument("--results_dir", default="results/", help="Base results directory")
    parser.add_argument("--latent_subdir", default="latent_analysis_normalized",
                        help="Latent analysis subdirectory (latent_analysis or latent_analysis_normalized)")
    parser.add_argument("--n_examples", type=int, default=3, help="Number of example regions per cluster")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top features to plot per region")
    parser.add_argument("--max_clusters", type=int, default=20,
                        help="Maximum number of clusters to plot (by size, descending)")
    parser.add_argument("--output_dir", default=None, help="Output directory (auto-detected if not set)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Find paths
    sae_dir = os.path.join(args.results_dir, args.chrom, "sae")
    latent_dir = os.path.join(sae_dir, args.latent_subdir)
    data_dir = os.path.join(latent_dir, "data")

    if not os.path.isdir(data_dir):
        logger.error(f"Latent data directory not found: {data_dir}")
        sys.exit(1)

    # Load cluster assignments
    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")
    logger.info(f"Loading cluster assignments from {ca_path}")
    df = pd.read_csv(ca_path, sep="\t", comment="#")
    # Handle both column names: "cluster_id" and "cluster"
    cluster_col = "cluster_id" if "cluster_id" in df.columns else "cluster"
    n_clusters = df[cluster_col].nunique()
    logger.info(f"  {len(df)} regions, {n_clusters} clusters (column: {cluster_col})")

    # Load max-pooled vectors
    mp_path = os.path.join(data_dir, "maxpooled_vectors.npy")
    logger.info(f"Loading max-pooled vectors from {mp_path}")
    maxpooled = np.load(mp_path)
    logger.info(f"  Shape: {maxpooled.shape}")

    # Find feature_matrices.npz
    merged_run = find_latest_merged_run(sae_dir)
    if merged_run is None:
        logger.error(f"No feature_matrices.npz found in {sae_dir}/*/data/")
        sys.exit(1)
    fm_path = merged_run / "data" / "feature_matrices.npz"
    logger.info(f"Loading feature matrices from {fm_path}")

    # Output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(latent_dir, "cluster_investigation")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Select top clusters by size
    cluster_sizes = df[cluster_col].value_counts().sort_values(ascending=False)
    selected_clusters = cluster_sizes.index[:args.max_clusters].tolist()
    logger.info(f"Investigating {len(selected_clusters)} clusters (of {n_clusters} total)")

    # Load feature matrices (lazy — open file handle, load per-region on demand)
    fm = np.load(str(fm_path), allow_pickle=True)
    fm_keys = set(fm.keys())

    # Track top features per cluster for cross-cluster comparison
    cluster_top_features = {}  # cluster_id -> list of feature_ids
    cluster_examples = {}      # cluster_id -> list of (region_idx, region_info_dict)

    for cluster_id in selected_clusters:
        cluster_df = df[df[cluster_col] == cluster_id].copy()
        cluster_df = cluster_df.reset_index(drop=True)
        # Re-index to match maxpooled array — use original DataFrame index
        cluster_orig_idx = df[df[cluster_col] == cluster_id].index.values

        n_in_cluster = len(cluster_df)
        logger.info(f"\nCluster {cluster_id}: {n_in_cluster} regions")

        # Pick representative examples
        rep_indices = pick_representative_regions(
            df[df[cluster_col] == cluster_id], maxpooled, args.n_examples
        )

        # Get top features for this cluster (from first example's max-pooled vector)
        cluster_features = get_top_features(maxpooled[rep_indices[0]], args.top_k)
        cluster_top_features[cluster_id] = cluster_features
        logger.info(f"  Top-{args.top_k} features: {cluster_features}")

        # Plot per-example traces
        n_ex = len(rep_indices)
        fig, axes = plt.subplots(n_ex, 1, figsize=(14, 4 * n_ex), squeeze=False)
        fig.suptitle(f"Cluster {cluster_id} — {n_in_cluster} regions — Top-{args.top_k} Feature Traces",
                     fontsize=13, y=1.02)

        examples_for_cluster = []
        for ei, ridx in enumerate(rep_indices):
            region_key = f"region_{ridx}"
            row = df.iloc[ridx]
            region_info = {
                "genomic_start": int(row["genomic_start"]),
                "genomic_end": int(row["genomic_end"]),
                "region_length": int(row.get("region_length", row["genomic_end"] - row["genomic_start"])),
            }
            examples_for_cluster.append((ridx, region_info))

            if region_key not in fm_keys:
                logger.warning(f"  region_key {region_key} not found in feature_matrices.npz, skipping")
                axes[ei, 0].text(0.5, 0.5, f"region_{ridx} not in feature_matrices",
                                 transform=axes[ei, 0].transAxes, ha="center")
                continue

            feature_matrix = fm[region_key]
            top_feats = get_top_features(maxpooled[ridx], args.top_k)

            plot_feature_traces(
                feature_matrix, top_feats, region_info, axes[ei, 0],
                title=f"Example {ei+1} (region {ridx}) — cluster {cluster_id}"
            )

        cluster_examples[cluster_id] = examples_for_cluster
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"cluster_{cluster_id}_examples.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {out_path}")

    # === Cross-cluster comparison ===
    # Take top 4 clusters, plot: features from cluster A on examples from cluster B
    top4 = selected_clusters[:min(4, len(selected_clusters))]
    if len(top4) >= 2:
        logger.info(f"\nCross-cluster comparison for clusters: {top4}")
        n_top = len(top4)
        fig = plt.figure(figsize=(5 * n_top, 4 * n_top))
        gs = GridSpec(n_top, n_top, figure=fig, hspace=0.4, wspace=0.3)

        for row_i, src_cluster in enumerate(top4):
            src_features = cluster_top_features.get(src_cluster, [])
            if not src_features:
                continue

            for col_j, tgt_cluster in enumerate(top4):
                ax = fig.add_subplot(gs[row_i, col_j])
                examples = cluster_examples.get(tgt_cluster, [])
                if not examples:
                    ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                    continue

                # Use first example from target cluster
                ridx, region_info = examples[0]
                region_key = f"region_{ridx}"
                if region_key not in fm_keys:
                    ax.text(0.5, 0.5, "Missing", ha="center", transform=ax.transAxes)
                    continue

                feature_matrix = fm[region_key]
                plot_feature_traces(
                    feature_matrix, src_features, region_info, ax,
                    title=f"C{src_cluster} features → C{tgt_cluster} example"
                )

                if row_i == 0:
                    ax.set_title(f"Target: Cluster {tgt_cluster}\n{ax.get_title()}", fontsize=9)
                if col_j == 0:
                    ax.set_ylabel(f"Source: C{src_cluster}\n{ax.get_ylabel()}", fontsize=9)

        fig.suptitle(f"Cross-Cluster Feature Comparison — {args.chrom} ({args.latent_subdir})",
                     fontsize=14, y=1.02)
        out_path = os.path.join(out_dir, "cross_cluster_comparison.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")

    # === Summary table ===
    summary_rows = []
    for cid in selected_clusters:
        feats = cluster_top_features.get(cid, [])
        n_regions = len(df[df[cluster_col] == cid])
        summary_rows.append({
            "cluster_id": cid,
            "n_regions": n_regions,
            "top_features": ",".join(str(f) for f in feats),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "cluster_top_features_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"Saved summary: {summary_path}")

    # Check feature overlap between clusters
    logger.info("\n=== Feature overlap between top clusters ===")
    for i, c1 in enumerate(top4):
        for c2 in top4[i+1:]:
            f1 = set(cluster_top_features.get(c1, []))
            f2 = set(cluster_top_features.get(c2, []))
            overlap = f1 & f2
            logger.info(f"  C{c1} vs C{c2}: {len(overlap)}/{args.top_k} shared features "
                        f"({overlap if overlap else 'none'})")

    fm.close()
    logger.info(f"\nDone. {len(os.listdir(out_dir))} files in {out_dir}")


if __name__ == "__main__":
    main()
