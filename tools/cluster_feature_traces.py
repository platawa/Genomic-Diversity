#!/usr/bin/env python3
"""
Cluster validation via per-nucleotide feature activation traces.

For each specified cluster:
  - Pick N example regions
  - Load per-nucleotide SAE activations from chunk NPZ files
  - Identify top-K activated features
  - Plot activation traces (x=nucleotide, y=activation) for those features

Generates:
  - Per-cluster panels: top features for each example
  - Cross-cluster comparison: same features plotted across clusters
  - Summary: are clusters driven by the same features or different ones?

Usage:
    python tools/cluster_feature_traces.py \\
        --chrom chr12 \\
        --results_dir results/ \\
        --clusters 0,5,12,19,26,42 \\
        --n_examples 3 \\
        --n_features 5 \\
        --latent_subdir latent_analysis_postnorm
"""

import argparse
import glob
import json
import logging
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_chunk_for_region(shard_dirs, region_idx):
    """Find and load the per-nucleotide feature matrix for a given region_idx.

    Returns numpy array of shape (seq_len, n_features) or None.
    """
    for shard_dir in shard_dirs:
        data_dir = os.path.join(shard_dir, "data")
        chunk_files = sorted(glob.glob(os.path.join(data_dir, "_chunk_*.npz")))

        for chunk_file in chunk_files:
            # Parse chunk range from filename: _chunk_0000000_0000200.npz
            basename = os.path.basename(chunk_file)
            match = re.match(r"_chunk_(\d+)_(\d+)\.npz", basename)
            if not match:
                continue
            chunk_start = int(match.group(1))
            chunk_end = int(match.group(2))

            if chunk_start <= region_idx < chunk_end:
                data = np.load(chunk_file, allow_pickle=False)
                key = f"region_{region_idx}"
                if key in data:
                    mat = data[key]
                    data.close()
                    return mat
                data.close()

    return None


def get_top_features(feature_matrix, n_features=5):
    """Get top-N features by max activation across nucleotides."""
    max_per_feature = feature_matrix.max(axis=0)
    top_indices = np.argsort(max_per_feature)[-n_features:][::-1]
    return top_indices


def get_differential_features(mp, ca, cluster_id, n_features=5):
    """Get features that are most differentially activated in this cluster vs others.

    Uses the ratio: mean_activation_in_cluster / mean_activation_elsewhere.
    This finds features that are SPECIFIC to the cluster, not just globally strong.
    """
    mask = (ca['cluster'] == cluster_id).values[:len(mp)]
    in_cluster = mp[mask]
    out_cluster = mp[~mask]

    mean_in = in_cluster.mean(axis=0)
    mean_out = out_cluster.mean(axis=0)

    # Avoid division by zero: add small epsilon
    eps = 1e-8
    fold_change = mean_in / (mean_out + eps)

    # Only consider features that are actually active in this cluster
    active_mask = mean_in > eps
    fold_change[~active_mask] = 0

    top_indices = np.argsort(fold_change)[-n_features:][::-1]
    return top_indices, fold_change[top_indices]


def plot_cluster_traces(cluster_id, examples, n_features, output_dir,
                        override_features=None):
    """Plot per-nucleotide activation traces for examples from one cluster.

    Style matches sparse_autoencoder.ipynb:
    - Each feature gets its own subplot row
    - Wide figure, thin lines, shared x-axis
    - Drop region highlighted with colored span
    - One figure per example region

    If override_features is provided, uses those (differential features)
    instead of per-region top features.
    """
    n_examples = len(examples)
    if n_examples == 0:
        return

    all_top_features = []

    for ex_idx, (region_idx, meta, feat_mat) in enumerate(examples):
        if override_features is not None:
            top_feat = override_features
        else:
            top_feat = get_top_features(feat_mat, n_features)
        all_top_features.append(set(top_feat.tolist() if hasattr(top_feat, 'tolist') else top_feat))

        n_rows = len(top_feat)
        fig, axes = plt.subplots(n_rows, 1, figsize=(30, 1.2 * n_rows),
                                 sharex=True, squeeze=False)

        # Estimate drop region boundaries within padded sequence
        seq_len = feat_mat.shape[0]
        region_len = int(meta.get('region_length', 0))
        padding_est = max(0, (seq_len - region_len) // 2)
        drop_start_local = padding_est
        drop_end_local = padding_est + region_len

        for row, feat_idx in enumerate(top_feat):
            ax = axes[row, 0]
            trace = feat_mat[:, feat_idx]
            ax.plot(trace, lw=0.5, label=f"feature {feat_idx}", alpha=0.9,
                    color=f"C{row}")
            ax.set_xlim(0, seq_len)
            y_max = max(trace.max() * 1.1, 1.0)
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, round(y_max / 2, 1)])

            # Highlight drop region
            if padding_est > 0:
                ax.axvspan(drop_start_local, drop_end_local,
                           color='lightyellow', alpha=0.4)
                ax.axvline(drop_start_local, color='red', ls='--', lw=0.5, alpha=0.5)
                ax.axvline(drop_end_local, color='darkred', ls='--', lw=0.5, alpha=0.5)

            ax.legend(fontsize=8, loc="upper right")

        axes[0, 0].set_title(
            f"Cluster {cluster_id} — Example {ex_idx+1}: Region {region_idx} "
            f"({meta['genomic_start']:,}–{meta['genomic_end']:,}, "
            f"{region_len}bp, pad={padding_est}bp, "
            f"conf={meta.get('confidence', 0):.1f})",
            fontsize=11)
        axes[-1, 0].set_xlabel("Nucleotide position")

        plt.tight_layout()
        out_path = os.path.join(output_dir,
                                f"cluster_{cluster_id}_ex{ex_idx}_traces.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {out_path}")

    # Also make a combined summary: all examples side-by-side for this cluster
    # Each column = one example, each row = one feature (union of top features)
    all_feat_union = set()
    for s in all_top_features:
        all_feat_union |= s
    all_feat_sorted = sorted(all_feat_union)
    n_feat_total = len(all_feat_sorted)

    fig, axes = plt.subplots(n_feat_total, n_examples,
                             figsize=(10 * n_examples, 1.2 * n_feat_total),
                             sharex='col', squeeze=False)
    fig.suptitle(f"Cluster {cluster_id} — All Examples × All Top Features",
                 fontsize=13, y=1.01)

    for col, (region_idx, meta, feat_mat) in enumerate(examples):
        seq_len = feat_mat.shape[0]
        region_len = int(meta.get('region_length', 0))
        padding_est = max(0, (seq_len - region_len) // 2)
        drop_start_local = padding_est
        drop_end_local = padding_est + region_len

        axes[0, col].set_title(f"Region {region_idx} ({region_len}bp)", fontsize=9)
        for row, feat_idx in enumerate(all_feat_sorted):
            ax = axes[row, col]
            trace = feat_mat[:, feat_idx]
            is_top = feat_idx in all_top_features[col]
            ax.plot(trace, lw=0.5, alpha=0.9 if is_top else 0.4,
                    color=f"C{row}" if is_top else "gray")
            ax.set_xlim(0, seq_len)
            y_max = max(trace.max() * 1.1, 1.0)
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, round(y_max / 2, 1)])
            if padding_est > 0:
                ax.axvspan(drop_start_local, drop_end_local,
                           color='lightyellow', alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"F{feat_idx}", fontsize=8, rotation=0, labelpad=30)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"cluster_{cluster_id}_combined.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")

    # Feature overlap stats
    if len(all_top_features) > 1:
        intersection = all_top_features[0]
        for s in all_top_features[1:]:
            intersection = intersection & s
        union = set()
        for s in all_top_features:
            union |= s
        overlap_pct = len(intersection) / len(union) * 100 if union else 0
        logger.info(f"  Cluster {cluster_id} feature overlap: "
                    f"{len(intersection)}/{len(union)} shared ({overlap_pct:.0f}% Jaccard)")

    return all_top_features


def plot_cross_cluster_comparison(cluster_examples, n_features, output_dir,
                                  override_features=None):
    """Compare differential features across clusters.

    For each cluster, take its differential features and plot them
    on examples from other clusters to see if features are cluster-specific.
    """
    cluster_ids = sorted(cluster_examples.keys())
    n_clusters = len(cluster_ids)
    if n_clusters < 2:
        return

    # Use override (differential) features if provided, else per-region top
    cluster_top_features = {}
    for cid in cluster_ids:
        if override_features and cid in override_features:
            cluster_top_features[cid] = override_features[cid]
        else:
            examples = cluster_examples[cid]
            if examples:
                region_idx, meta, feat_mat = examples[0]
                cluster_top_features[cid] = get_top_features(feat_mat, n_features)

    # Plot: rows = source cluster features, cols = target cluster examples
    fig, axes = plt.subplots(n_clusters, n_clusters, figsize=(5 * n_clusters, 4 * n_clusters),
                             squeeze=False)
    fig.suptitle(f"Cross-Cluster Feature Comparison (Top {n_features} features)",
                 fontsize=14, y=1.01)

    for row_idx, src_cid in enumerate(cluster_ids):
        src_features = cluster_top_features.get(src_cid, [])
        for col_idx, tgt_cid in enumerate(cluster_ids):
            ax = axes[row_idx, col_idx]
            tgt_examples = cluster_examples[tgt_cid]
            if not tgt_examples or len(src_features) == 0:
                ax.set_visible(False)
                continue

            # Plot source cluster's features on target cluster's first example
            region_idx, meta, feat_mat = tgt_examples[0]
            for feat_idx in src_features:
                trace = feat_mat[:, feat_idx]
                ax.plot(trace, label=f"F{feat_idx}", alpha=0.7, linewidth=0.8)

            if row_idx == 0:
                ax.set_title(f"Cluster {tgt_cid}\n(region {region_idx})", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"C{src_cid} features", fontsize=9)

            # Only show legend on diagonal
            if row_idx == col_idx:
                ax.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cross_cluster_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_feature_overlap_matrix(cluster_top_features, output_dir):
    """Heatmap of Jaccard similarity of top features between clusters."""
    cluster_ids = sorted(cluster_top_features.keys())
    n = len(cluster_ids)
    jaccard = np.zeros((n, n))

    for i, ci in enumerate(cluster_ids):
        for j, cj in enumerate(cluster_ids):
            si = set(cluster_top_features[ci])
            sj = set(cluster_top_features[cj])
            union = si | sj
            jaccard[i, j] = len(si & sj) / len(union) if union else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=9)
    ax.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=9)
    ax.set_title("Top Feature Overlap (Jaccard Similarity) Between Clusters")
    plt.colorbar(im, ax=ax, label="Jaccard Index")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if jaccard[i, j] > 0.5 else "black")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_overlap_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster validation via feature activation traces")
    parser.add_argument("--chrom", required=True, help="Chromosome name (e.g., chr12)")
    parser.add_argument("--results_dir", default="results/", help="Root results directory")
    parser.add_argument("--clusters", required=True,
                        help="Comma-separated cluster IDs to analyze (e.g., 0,5,12,19)")
    parser.add_argument("--n_examples", type=int, default=3,
                        help="Number of example regions per cluster (default: 3)")
    parser.add_argument("--n_features", type=int, default=5,
                        help="Number of top features to plot (default: 5)")
    parser.add_argument("--latent_subdir", default="latent_analysis",
                        help="Latent analysis subdirectory (default: latent_analysis)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: {latent_dir}/cluster_traces/)")
    parser.add_argument("--conf_filter", default=None,
                        help="Filter shard dirs by this pattern (e.g., 'conf8.0', 'conf0.0')")
    args = parser.parse_args()

    cluster_ids = [int(c) for c in args.clusters.split(",")]
    logger.info(f"Analyzing clusters {cluster_ids} for {args.chrom}")

    # Load cluster assignments
    latent_dir = os.path.join(args.results_dir, args.chrom, "sae", args.latent_subdir)
    ca_path = os.path.join(latent_dir, "data", "cluster_assignments.tsv")
    if not os.path.isfile(ca_path):
        logger.error(f"Not found: {ca_path}")
        sys.exit(1)

    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    logger.info(f"Loaded {len(ca)} regions, {ca['cluster'].nunique()} clusters")

    # Find shard directories with chunk files
    sae_root = os.path.join(args.results_dir, args.chrom, "sae")
    shard_dirs = []
    for entry in sorted(os.listdir(sae_root)):
        full = os.path.join(sae_root, entry)
        if not os.path.isdir(full):
            continue
        if args.conf_filter and args.conf_filter not in entry:
            continue
        if glob.glob(os.path.join(full, "data", "_chunk_*.npz")):
            shard_dirs.append(full)

    logger.info(f"Found {len(shard_dirs)} shard dirs with chunk files")
    if not shard_dirs:
        logger.error("No shard directories with chunk files found")
        sys.exit(1)

    # Load maxpooled vectors for differential feature analysis
    mp_path = os.path.join(latent_dir, "data", "maxpooled_vectors.npy")
    mp = None
    if os.path.isfile(mp_path):
        mp = np.load(mp_path)
        if len(mp) > len(ca):
            mp = mp[:len(ca)]
        logger.info(f"Loaded maxpooled vectors: {mp.shape}")
    else:
        logger.warning(f"No maxpooled vectors at {mp_path}, using per-region top features only")

    # Output directory
    output_dir = args.output_dir or os.path.join(latent_dir, "cluster_traces")
    os.makedirs(output_dir, exist_ok=True)

    # Compute DIFFERENTIAL features per cluster (what makes each cluster unique)
    all_cluster_diff_features = {}
    if mp is not None:
        logger.info("Computing differential features per cluster...")
        for cid in cluster_ids:
            diff_feat, fold_changes = get_differential_features(mp, ca, cid, args.n_features)
            all_cluster_diff_features[cid] = diff_feat
            logger.info(f"  Cluster {cid}: differential features = {diff_feat.tolist()}, "
                        f"fold changes = {[f'{fc:.1f}x' for fc in fold_changes]}")

    # For each cluster, pick examples and load their feature matrices
    cluster_examples = {}
    all_cluster_top_features = {}

    for cid in cluster_ids:
        cluster_rows = ca[ca["cluster"] == cid]
        if len(cluster_rows) == 0:
            logger.warning(f"Cluster {cid}: no regions found")
            continue

        # Pick examples: prefer high-confidence, diverse lengths
        sample = cluster_rows.sort_values("confidence", ascending=False).head(args.n_examples * 3)
        if len(sample) > args.n_examples:
            sample = sample.iloc[::max(1, len(sample) // args.n_examples)][:args.n_examples]

        examples = []
        for _, row in sample.iterrows():
            region_idx = int(row["region_idx"])
            meta = row.to_dict()

            feat_mat = find_chunk_for_region(shard_dirs, region_idx)
            if feat_mat is None:
                logger.warning(f"  Region {region_idx}: chunk not found, skipping")
                continue

            examples.append((region_idx, meta, feat_mat))
            logger.info(f"  Cluster {cid}, region {region_idx}: loaded {feat_mat.shape}")

            if len(examples) >= args.n_examples:
                break

        cluster_examples[cid] = examples

        # Use DIFFERENTIAL features if available, else fall back to per-region top
        if cid in all_cluster_diff_features:
            features_to_plot = all_cluster_diff_features[cid]
        elif examples:
            features_to_plot = get_top_features(examples[0][2], args.n_features)
        else:
            features_to_plot = np.array([])

        all_cluster_top_features[cid] = features_to_plot

        # Plot per-cluster traces using differential features
        if examples and len(features_to_plot) > 0:
            plot_cluster_traces(cid, examples, args.n_features, output_dir,
                                override_features=features_to_plot)

    # Cross-cluster comparison
    if len(cluster_examples) >= 2:
        plot_cross_cluster_comparison(cluster_examples, args.n_features, output_dir,
                                      override_features=all_cluster_top_features)
        plot_feature_overlap_matrix(all_cluster_top_features, output_dir)

    # Summary statistics
    summary = {}
    for cid, top_feats in all_cluster_top_features.items():
        feat_list = top_feats.tolist() if hasattr(top_feats, 'tolist') else list(top_feats)
        summary[f"cluster_{cid}"] = {
            "differential_features": [int(f) for f in feat_list],
            "n_regions": int(len(ca[ca["cluster"] == cid])),
            "n_examples_loaded": len(cluster_examples.get(cid, [])),
            "feature_type": "differential" if cid in all_cluster_diff_features else "top_per_region",
        }

    summary_path = os.path.join(output_dir, "cluster_trace_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")

    logger.info(f"\nDone! Output: {output_dir}")
    logger.info(f"Clusters analyzed: {list(cluster_examples.keys())}")


if __name__ == "__main__":
    main()
