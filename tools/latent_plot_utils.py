#!/usr/bin/env python3
"""
latent_plot_utils.py

Shared utility functions for enhanced SAE latent analysis plots:
  - Distance to nearest gene (upstream/downstream) from GTF
  - Entropy drop statistics (average/minimum)
  - Region length computation and distribution plots
  - Feature firing statistics (top-K, non-zero counts)
  - Firing threshold analysis (1%, 5%, 10% of neurons)
  - Generic continuous-color scatter plotting
"""

import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Distance to nearest gene
# ═══════════════════════════════════════════════════════════════════════════════

def load_gene_boundaries(gtf_path, chrom_id):
    """Load gene start/end positions from GTF for a single chromosome.

    Tries the given chrom_id first; if no genes found, does a quick scan
    to auto-detect the GTF's chromosome naming convention.

    Returns sorted arrays of gene starts and gene ends.
    """
    def _scan(fpath, target_id):
        starts, ends = [], []
        with open(fpath) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 9:
                    continue
                if parts[0] != target_id:
                    continue
                if parts[2] != "gene":
                    continue
                starts.append(int(parts[3]))
                ends.append(int(parts[4]))
        return starts, ends

    starts, ends = _scan(gtf_path, chrom_id)

    # If nothing found, try auto-detecting GTF chromosome naming
    if not starts:
        # Try without version suffix (NC_000913.3 → NC_000913)
        alt_id = chrom_id.rsplit(".", 1)[0] if "." in chrom_id else None
        if alt_id:
            starts, ends = _scan(gtf_path, alt_id)
            if starts:
                logger.info(f"GTF autodetect: used {alt_id} instead of {chrom_id}")

    if not starts:
        logger.warning(f"No genes found for {chrom_id} in GTF: {gtf_path}")

    starts = np.array(starts, dtype=np.int64)
    ends = np.array(ends, dtype=np.int64)
    order = np.argsort(starts)
    return starts[order], ends[order]


def compute_distance_to_nearest_gene(region_starts, region_ends, gtf_path, chrom_id):
    """Compute distance to nearest upstream and downstream gene for each region.

    For intergenic regions:
      - upstream_dist: distance from region midpoint to the end of nearest upstream gene
      - downstream_dist: distance from region midpoint to the start of nearest downstream gene
    For genic regions (overlapping a gene): distance = 0.

    Parameters
    ----------
    region_starts, region_ends : array-like
        Genomic start/end for each region.
    gtf_path : str
        Path to GTF annotation file.
    chrom_id : str
        Chromosome ID as it appears in the GTF (e.g., NC_000001.11).

    Returns
    -------
    upstream_dist, downstream_dist : np.ndarray
        Distance arrays. 0 means the region overlaps a gene.
    """
    gene_starts, gene_ends = load_gene_boundaries(gtf_path, chrom_id)
    n_genes = len(gene_starts)
    n_regions = len(region_starts)

    if n_genes == 0:
        logger.warning(f"No genes found for {chrom_id} in GTF")
        return np.full(n_regions, np.nan), np.full(n_regions, np.nan)

    midpoints = (np.asarray(region_starts) + np.asarray(region_ends)) // 2
    upstream_dist = np.full(n_regions, np.nan)
    downstream_dist = np.full(n_regions, np.nan)

    # For each midpoint, check overlap with genes, then find nearest
    # Use searchsorted on gene_starts for efficient lookup
    for i, mid in enumerate(midpoints):
        # Check if midpoint is inside any gene
        # A gene covers [gene_starts[j], gene_ends[j]]
        idx = np.searchsorted(gene_starts, mid, side="right") - 1
        if 0 <= idx < n_genes and gene_starts[idx] <= mid <= gene_ends[idx]:
            upstream_dist[i] = 0
            downstream_dist[i] = 0
            continue

        # Upstream: nearest gene whose end < mid
        # gene_ends sorted by gene_starts, so scan backward from idx
        best_up = np.inf
        for j in range(min(idx + 1, n_genes) - 1, -1, -1):
            if gene_ends[j] < mid:
                best_up = mid - gene_ends[j]
                break
        upstream_dist[i] = best_up if best_up != np.inf else np.nan

        # Downstream: nearest gene whose start > mid
        search_idx = np.searchsorted(gene_starts, mid, side="right")
        if search_idx < n_genes:
            downstream_dist[i] = gene_starts[search_idx] - mid
        # else: no downstream gene → stays NaN

    logger.info(f"Distance to gene: {n_regions} regions, {n_genes} genes, "
                f"overlap={np.sum(upstream_dist == 0)}")
    return upstream_dist, downstream_dist


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Entropy drop statistics
# ═══════════════════════════════════════════════════════════════════════════════

def load_entropy_drop_stats(scoring_boundaries_path, region_starts, region_ends):
    """Load mean/min entropy from scoring drop_boundaries.tsv, matched by coordinates.

    Parameters
    ----------
    scoring_boundaries_path : str
        Path to drop_boundaries.tsv from score_chromosome.py.
    region_starts, region_ends : array-like
        Genomic coordinates of SAE regions (from sae_results.tsv / cluster_assignments.tsv).

    Returns
    -------
    avg_entropy, min_entropy : np.ndarray
        Per-region entropy values. NaN for unmatched regions.
    """
    # Parse drop_boundaries.tsv (skip comment lines)
    bounds = pd.read_csv(scoring_boundaries_path, sep="\t", comment="#")

    n_regions = len(region_starts)
    avg_entropy = np.full(n_regions, np.nan)
    min_entropy = np.full(n_regions, np.nan)

    # Build a lookup: (genomic_start, genomic_end) → (mean_entropy, min_entropy)
    lookup = {}
    for _, row in bounds.iterrows():
        key = (int(row["genomic_start"]), int(row["genomic_end"]))
        lookup[key] = (row["mean_entropy"], row["min_entropy"])

    matched = 0
    for i in range(n_regions):
        key = (int(region_starts[i]), int(region_ends[i]))
        if key in lookup:
            avg_entropy[i], min_entropy[i] = lookup[key]
            matched += 1

    logger.info(f"Entropy stats: matched {matched}/{n_regions} regions")
    if matched < n_regions * 0.5:
        logger.warning("Less than 50% of regions matched — check coordinate alignment")

    return avg_entropy, min_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Region lengths
# ═══════════════════════════════════════════════════════════════════════════════

def compute_region_lengths(region_starts, region_ends):
    """Compute region lengths."""
    return np.asarray(region_ends) - np.asarray(region_starts)


def compute_length_stats(lengths):
    """Compute summary statistics for region lengths."""
    return {
        "count": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "q25": float(np.percentile(lengths, 25)),
        "q75": float(np.percentile(lengths, 75)),
        "q95": float(np.percentile(lengths, 95)),
        "uniform": bool(np.std(lengths) < 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feature firing statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_firing_stats(maxpooled_vectors, top_k=5):
    """Compute per-region feature firing counts and top-K features.

    Parameters
    ----------
    maxpooled_vectors : np.ndarray, shape (N, D)
        Max-pooled SAE feature vectors.
    top_k : int
        Number of top features to extract per region.

    Returns
    -------
    n_fired : np.ndarray, shape (N,)
        Number of non-zero features per region.
    top_k_features : list of list of (feature_idx, activation)
        Top-K features for each region, sorted by activation descending.
    """
    n_regions, n_features = maxpooled_vectors.shape
    n_fired = np.count_nonzero(maxpooled_vectors, axis=1)

    top_k_features = []
    for i in range(n_regions):
        row = maxpooled_vectors[i]
        nonzero_mask = row != 0
        if nonzero_mask.sum() == 0:
            top_k_features.append([])
            continue

        # Get top-K by magnitude
        k = min(top_k, int(nonzero_mask.sum()))
        if k == 0:
            top_k_features.append([])
            continue
        top_indices = np.argpartition(row, -k)[-k:]
        top_indices = top_indices[np.argsort(row[top_indices])[::-1]]

        features = [(int(idx), float(row[idx])) for idx in top_indices if row[idx] != 0]
        top_k_features.append(features)

    logger.info(f"Feature firing: {n_regions} regions, "
                f"mean={np.mean(n_fired):.1f}, median={np.median(n_fired):.0f} "
                f"non-zero features per region")
    return n_fired, top_k_features


def compute_firing_threshold_counts(maxpooled_vectors, thresholds=(0.01, 0.05, 0.10)):
    """Count features that fire in >= threshold fraction of all regions.

    For each threshold T:
      1. Find features active in >= T fraction of regions ("common features")
      2. For each region, count how many of these common features are active

    Parameters
    ----------
    maxpooled_vectors : np.ndarray, shape (N, D)
    thresholds : tuple of float

    Returns
    -------
    dict of threshold → {
        'per_region_counts': np.ndarray (N,),
        'n_common_features': int,
        'common_feature_indices': np.ndarray
    }
    """
    n_regions, n_features = maxpooled_vectors.shape
    # Fraction of regions each feature fires in
    feature_freq = np.count_nonzero(maxpooled_vectors, axis=0) / n_regions

    results = {}
    for thresh in thresholds:
        common_mask = feature_freq >= thresh
        n_common = common_mask.sum()

        # For each region, count active common features
        active_common = maxpooled_vectors[:, common_mask] != 0
        per_region = active_common.sum(axis=1)

        results[thresh] = {
            "per_region_counts": per_region,
            "n_common_features": int(n_common),
            "common_feature_indices": np.where(common_mask)[0],
        }
        pct = thresh * 100
        logger.info(f"  Threshold {pct:.0f}%: {n_common} common features, "
                    f"mean {np.mean(per_region):.1f} active per region")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plotting utilities
# ═══════════════════════════════════════════════════════════════════════════════

def plot_continuous_scatter(coords, values, cmap, colorbar_label, title, out_path,
                           emb_name="tsne", log_scale=False, vmin=None, vmax=None,
                           point_size=None, alpha=None, figsize=(10, 8)):
    """Generic scatter plot colored by a continuous variable.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
    values : np.ndarray, shape (N,)
    cmap : str or Colormap
    colorbar_label : str
    title : str
    out_path : str
    emb_name : str
        'tsne' or 'umap' — used for axis labels.
    log_scale : bool
        Use LogNorm for colorbar.
    """
    n = len(coords)
    if point_size is None:
        point_size = 30 if n < 2000 else (12 if n < 10000 else 5)
    if alpha is None:
        alpha = 0.9 if n < 2000 else (0.8 if n < 10000 else 0.7)

    # Filter NaN values
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        logger.warning(f"All values are NaN, skipping plot: {out_path}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Plot NaN points as light gray background
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c="#e0e0e0", s=point_size * 0.5, alpha=0.3,
                   edgecolors="none", rasterized=True, label="N/A")

    norm = None
    if log_scale:
        pos_vals = values[valid & (values > 0)]
        if len(pos_vals) > 0:
            norm = LogNorm(vmin=pos_vals.min(), vmax=pos_vals.max())
        else:
            log_scale = False

    if not log_scale:
        v = values[valid]
        norm = Normalize(vmin=vmin if vmin is not None else np.nanmin(v),
                         vmax=vmax if vmax is not None else np.nanmax(v))

    sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                    c=values[valid], cmap=cmap, norm=norm,
                    s=point_size, alpha=alpha,
                    edgecolors="none", rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=12)

    prefix = emb_name.upper()
    ax.set_xlabel(f"{prefix} 1", fontsize=11)
    ax.set_ylabel(f"{prefix} 2", fontsize=11)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def plot_length_distribution(lengths, out_path, title_prefix=""):
    """Plot region length distribution: histogram + box plot + stats.

    Parameters
    ----------
    lengths : np.ndarray
    out_path : str
    title_prefix : str
        E.g., "chr22" or "E. coli"
    """
    stats = compute_length_stats(lengths)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(lengths, bins=min(50, max(10, len(lengths) // 10)),
            color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(stats["mean"], color="red", ls="--", label=f"Mean={stats['mean']:.0f}")
    ax.axvline(stats["median"], color="green", ls="--", label=f"Median={stats['median']:.0f}")
    ax.set_xlabel("Region Length (bp)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} Region Length Distribution (N={stats['count']})")
    ax.legend()

    # Stats text
    stats_text = (
        f"Min: {stats['min']}\n"
        f"Q25: {stats['q25']:.0f}\n"
        f"Median: {stats['median']:.0f}\n"
        f"Q75: {stats['q75']:.0f}\n"
        f"Q95: {stats['q95']:.0f}\n"
        f"Max: {stats['max']}\n"
        f"Std: {stats['std']:.1f}\n"
        f"Uniform: {'Yes' if stats['uniform'] else 'No'}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(lengths, vert=True, widths=0.6, patch_artist=True)
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][0].set_alpha(0.6)
    ax.set_ylabel("Region Length (bp)")
    ax.set_title("Box Plot")
    ax.set_xticklabels(["Regions"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")

    return stats


def save_top_features_tsv(top_k_features, region_starts, region_ends, out_path):
    """Save top-K features per region to TSV.

    Parameters
    ----------
    top_k_features : list of list of (feature_idx, activation)
    region_starts, region_ends : array-like
    out_path : str
    """
    with open(out_path, "w") as f:
        f.write("region_idx\tgenomic_start\tgenomic_end\t"
                "top_feature_ids\ttop_feature_activations\tn_nonzero_in_top\n")
        for i, feats in enumerate(top_k_features):
            ids = ",".join(str(idx) for idx, _ in feats)
            acts = ",".join(f"{act:.4f}" for _, act in feats)
            n_nz = sum(1 for _, act in feats if act != 0)
            f.write(f"{i}\t{int(region_starts[i])}\t{int(region_ends[i])}\t"
                    f"{ids}\t{acts}\t{n_nz}\n")
    logger.info(f"Saved top features: {out_path} ({len(top_k_features)} regions)")


def save_firing_stats_tsv(n_fired, region_starts, region_ends, out_path):
    """Save per-region firing counts to TSV."""
    with open(out_path, "w") as f:
        f.write("region_idx\tgenomic_start\tgenomic_end\tn_features_fired\n")
        for i in range(len(n_fired)):
            f.write(f"{i}\t{int(region_starts[i])}\t{int(region_ends[i])}\t"
                    f"{int(n_fired[i])}\n")
    logger.info(f"Saved firing stats: {out_path}")
