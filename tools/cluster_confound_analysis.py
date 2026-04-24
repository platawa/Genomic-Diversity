#!/usr/bin/env python3
"""
Cluster confound analysis — Phase 1 of scientific cluster validation.

Tests whether Leiden clusters reflect real biological motifs or artifacts
of confounds (region length, detection method, confidence, genomic position).

Generates diagnostic plots and statistical tests.

Usage:
    python tools/cluster_confound_analysis.py \\
        --chrom chr12 \\
        --results_dir results/ \\
        --latent_subdir latent_analysis_postnorm
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data(latent_dir):
    """Load cluster assignments and maxpooled vectors."""
    ca_path = os.path.join(latent_dir, "data", "cluster_assignments.tsv")
    mp_path = os.path.join(latent_dir, "data", "maxpooled_vectors.npy")

    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    logger.info(f"Loaded {len(ca)} regions, {ca['cluster'].nunique()} clusters")

    mp = None
    if os.path.isfile(mp_path):
        mp = np.load(mp_path)
        # Truncate to match ca if needed
        if len(mp) > len(ca):
            mp = mp[:len(ca)]
        logger.info(f"Loaded maxpooled vectors: {mp.shape}")

    return ca, mp


def analyze_length_confound(ca, output_dir):
    """Test 1a: Does region length predict cluster membership?"""
    logger.info("=== 1a. Region Length vs Cluster ===")

    # Get top clusters by size (limit to manageable number)
    top_clusters = ca['cluster'].value_counts().head(20).index.tolist()
    ca_top = ca[ca['cluster'].isin(top_clusters)]

    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: boxplot of length by cluster
    cluster_lengths = [ca_top[ca_top['cluster'] == c]['region_length'].values
                       for c in sorted(top_clusters)]
    bp = axes[0].boxplot(cluster_lengths, labels=[str(c) for c in sorted(top_clusters)],
                         patch_artist=True, showfliers=False)
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Region Length (bp)")
    axes[0].set_title("Region Length Distribution by Cluster")
    axes[0].tick_params(axis='x', rotation=45)

    # Right: scatter of mean length vs cluster size
    cluster_stats = ca.groupby('cluster').agg(
        n_regions=('region_length', 'count'),
        mean_length=('region_length', 'mean'),
        std_length=('region_length', 'std'),
    ).reset_index()
    axes[1].scatter(cluster_stats['n_regions'], cluster_stats['mean_length'],
                    s=30, alpha=0.7)
    axes[1].set_xlabel("Cluster Size (# regions)")
    axes[1].set_ylabel("Mean Region Length (bp)")
    axes[1].set_title("Cluster Size vs Mean Length")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confound_1a_length.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Kruskal-Wallis test
    groups = [ca[ca['cluster'] == c]['region_length'].values for c in ca['cluster'].unique()]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) >= 2:
        stat, p_value = stats.kruskal(*groups)
        logger.info(f"  Kruskal-Wallis: H={stat:.1f}, p={p_value:.2e}")
        # Effect size: eta-squared
        n_total = sum(len(g) for g in groups)
        eta_sq = (stat - len(groups) + 1) / (n_total - len(groups))
        logger.info(f"  Effect size (eta²): {eta_sq:.4f}")
        return {"kruskal_wallis_H": float(stat), "p_value": float(p_value),
                "eta_squared": float(eta_sq)}
    return {}


def analyze_method_confound(ca, output_dir):
    """Test 1b: Does detection method predict cluster membership?"""
    logger.info("=== 1b. Detection Method vs Cluster ===")

    if 'method' not in ca.columns:
        logger.warning("No 'method' column, skipping")
        return {}

    top_clusters = ca['cluster'].value_counts().head(20).index.tolist()
    ca_top = ca[ca['cluster'].isin(top_clusters)]

    # Stacked bar chart
    ct = pd.crosstab(ca_top['cluster'], ca_top['method'], normalize='index')
    fig, ax = plt.subplots(figsize=(12, 5))
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Proportion")
    ax.set_title("Detection Method Composition by Cluster")
    ax.legend(title="Method")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confound_1b_method.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Chi-squared test
    ct_counts = pd.crosstab(ca['cluster'], ca['method'])
    chi2, p_value, dof, expected = stats.chi2_contingency(ct_counts)
    cramers_v = np.sqrt(chi2 / (len(ca) * (min(ct_counts.shape) - 1)))
    logger.info(f"  Chi-squared: χ²={chi2:.1f}, p={p_value:.2e}, Cramer's V={cramers_v:.4f}")
    return {"chi2": float(chi2), "p_value": float(p_value), "cramers_v": float(cramers_v)}


def analyze_confidence_confound(ca, output_dir):
    """Test 1c: Does confidence predict cluster membership?"""
    logger.info("=== 1c. Confidence vs Cluster ===")

    top_clusters = ca['cluster'].value_counts().head(20).index.tolist()
    ca_top = ca[ca['cluster'].isin(top_clusters)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: boxplot
    cluster_confs = [ca_top[ca_top['cluster'] == c]['confidence'].values
                     for c in sorted(top_clusters)]
    axes[0].boxplot(cluster_confs, labels=[str(c) for c in sorted(top_clusters)],
                    patch_artist=True, showfliers=False)
    axes[0].set_xlabel("Cluster ID")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("Confidence Distribution by Cluster")
    axes[0].tick_params(axis='x', rotation=45)

    # Right: mean confidence vs cluster size
    cluster_stats = ca.groupby('cluster').agg(
        n_regions=('confidence', 'count'),
        mean_confidence=('confidence', 'mean'),
    ).reset_index()
    axes[1].scatter(cluster_stats['n_regions'], cluster_stats['mean_confidence'],
                    s=30, alpha=0.7)
    axes[1].set_xlabel("Cluster Size")
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_title("Cluster Size vs Mean Confidence")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confound_1c_confidence.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Spearman correlation between cluster size and mean confidence
    rho, p_val = stats.spearmanr(cluster_stats['n_regions'], cluster_stats['mean_confidence'])
    logger.info(f"  Spearman (size vs conf): rho={rho:.3f}, p={p_val:.2e}")
    return {"spearman_rho": float(rho), "p_value": float(p_val)}


def analyze_position_confound(ca, output_dir):
    """Test 1d: Does genomic position predict cluster membership?"""
    logger.info("=== 1d. Genomic Position vs Cluster ===")

    top_clusters = ca['cluster'].value_counts().head(10).index.tolist()
    ca_top = ca[ca['cluster'].isin(top_clusters)]

    fig, ax = plt.subplots(figsize=(14, 6))
    for cid in sorted(top_clusters):
        subset = ca_top[ca_top['cluster'] == cid]
        ax.scatter(subset['genomic_start'], [cid] * len(subset),
                   s=2, alpha=0.3, label=f"C{cid} (n={len(subset)})")
    ax.set_xlabel("Genomic Position")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Genomic Position Distribution by Cluster")
    ax.legend(fontsize=7, markerscale=5, ncol=2, loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confound_1d_position.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {}


def analyze_feature_overlap(ca, mp, output_dir, n_top=20):
    """Test 1e: Do clusters have distinct top features?"""
    logger.info("=== 1e. Feature Overlap Between Clusters ===")

    if mp is None:
        logger.warning("No maxpooled vectors, skipping feature overlap")
        return {}

    top_clusters = ca['cluster'].value_counts().head(15).index.tolist()

    # Get top features per cluster
    cluster_top = {}
    for cid in top_clusters:
        mask = ca['cluster'] == cid
        if mask.sum() == 0:
            continue
        cluster_vecs = mp[mask.values[:len(mp)]] if len(mp) >= len(ca) else mp[mask.values[:len(mp)]]
        mean_activation = cluster_vecs.mean(axis=0)
        top_idx = np.argsort(mean_activation)[-n_top:][::-1]
        cluster_top[cid] = set(top_idx.tolist())

    # Jaccard matrix
    cids = sorted(cluster_top.keys())
    n = len(cids)
    jaccard = np.zeros((n, n))
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            si, sj = cluster_top[ci], cluster_top[cj]
            union = si | sj
            jaccard[i, j] = len(si & sj) / len(union) if union else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cids], fontsize=8)
    ax.set_yticklabels([f"C{c}" for c in cids], fontsize=8)
    ax.set_title(f"Top-{n_top} Feature Overlap (Jaccard) Between Clusters")
    plt.colorbar(im, ax=ax, label="Jaccard Index")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if jaccard[i, j] > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confound_1e_feature_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats
    off_diag = jaccard[np.triu_indices(n, k=1)]
    logger.info(f"  Mean off-diagonal Jaccard: {off_diag.mean():.3f} ± {off_diag.std():.3f}")
    logger.info(f"  Max off-diagonal Jaccard: {off_diag.max():.3f}")
    return {"mean_jaccard": float(off_diag.mean()), "std_jaccard": float(off_diag.std()),
            "max_jaccard": float(off_diag.max()), "n_top_features": n_top}


def analyze_intra_inter_similarity(ca, mp, output_dir):
    """Test 1f: Intra-cluster vs inter-cluster cosine similarity."""
    logger.info("=== 1f. Intra vs Inter-Cluster Similarity ===")

    if mp is None:
        logger.warning("No maxpooled vectors, skipping similarity analysis")
        return {}

    top_clusters = ca['cluster'].value_counts().head(10).index.tolist()

    intra_sims = []
    inter_sims = []
    cluster_mean_vecs = {}

    for cid in top_clusters:
        mask = (ca['cluster'] == cid).values[:len(mp)]
        vecs = mp[mask]
        if len(vecs) < 2:
            continue

        # Sample to avoid huge similarity matrices
        if len(vecs) > 200:
            idx = np.random.choice(len(vecs), 200, replace=False)
            vecs_sample = vecs[idx]
        else:
            vecs_sample = vecs

        sim = cosine_similarity(vecs_sample)
        np.fill_diagonal(sim, np.nan)
        intra_sims.append(np.nanmean(sim))
        cluster_mean_vecs[cid] = vecs.mean(axis=0)

    # Inter-cluster: cosine between cluster centroids
    cids = sorted(cluster_mean_vecs.keys())
    centroid_matrix = np.array([cluster_mean_vecs[c] for c in cids])
    inter_sim = cosine_similarity(centroid_matrix)
    np.fill_diagonal(inter_sim, np.nan)
    inter_mean = np.nanmean(inter_sim)

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    ratio = intra_mean / inter_mean if inter_mean > 0 else float('inf')

    logger.info(f"  Mean intra-cluster similarity: {intra_mean:.4f}")
    logger.info(f"  Mean inter-cluster similarity (centroids): {inter_mean:.4f}")
    logger.info(f"  Ratio (intra/inter): {ratio:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(intra_sims)), intra_sims, color='steelblue', alpha=0.7)
    axes[0].axhline(inter_mean, color='red', ls='--', label=f"Inter-cluster mean={inter_mean:.3f}")
    axes[0].set_xlabel("Cluster (top 10 by size)")
    axes[0].set_ylabel("Mean Cosine Similarity")
    axes[0].set_title("Intra-Cluster Similarity vs Inter-Cluster Baseline")
    axes[0].legend()

    im = axes[1].imshow(inter_sim, cmap="coolwarm", vmin=0.5, vmax=1)
    axes[1].set_xticks(range(len(cids)))
    axes[1].set_yticks(range(len(cids)))
    axes[1].set_xticklabels([f"C{c}" for c in cids], fontsize=8)
    axes[1].set_yticklabels([f"C{c}" for c in cids], fontsize=8)
    axes[1].set_title("Inter-Cluster Centroid Similarity")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confound_1f_similarity.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {"intra_mean": float(intra_mean), "inter_mean": float(inter_mean),
            "ratio": float(ratio)}


def main():
    parser = argparse.ArgumentParser(description="Cluster confound analysis")
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--latent_subdir", default="latent_analysis_postnorm")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    latent_dir = os.path.join(args.results_dir, args.chrom, "sae", args.latent_subdir)
    output_dir = args.output_dir or os.path.join(latent_dir, "cluster_validation")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Cluster confound analysis: {args.chrom}")
    logger.info(f"Latent dir: {latent_dir}")
    logger.info(f"Output: {output_dir}")

    ca, mp = load_data(latent_dir)

    results = {}
    results["1a_length"] = analyze_length_confound(ca, output_dir)
    results["1b_method"] = analyze_method_confound(ca, output_dir)
    results["1c_confidence"] = analyze_confidence_confound(ca, output_dir)
    results["1d_position"] = analyze_position_confound(ca, output_dir)
    results["1e_feature_overlap"] = analyze_feature_overlap(ca, mp, output_dir)
    results["1f_similarity"] = analyze_intra_inter_similarity(ca, mp, output_dir)

    # Save results
    results["meta"] = {
        "chrom": args.chrom,
        "latent_subdir": args.latent_subdir,
        "n_regions": len(ca),
        "n_clusters": int(ca['cluster'].nunique()),
    }

    with open(os.path.join(output_dir, "confound_stats.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDone! Output: {output_dir}")

    # Print verdict
    logger.info("\n=== VERDICT ===")
    if results.get("1e_feature_overlap", {}).get("mean_jaccard", 1) < 0.3:
        logger.info("✓ Feature overlap is LOW — clusters have distinct feature signatures")
    else:
        logger.info("⚠ Feature overlap is HIGH — clusters may share too many features")

    ratio = results.get("1f_similarity", {}).get("ratio", 0)
    if ratio > 1.5:
        logger.info(f"✓ Intra/inter ratio = {ratio:.2f} — clusters are well-separated")
    elif ratio > 1.0:
        logger.info(f"~ Intra/inter ratio = {ratio:.2f} — moderate separation")
    else:
        logger.info(f"✗ Intra/inter ratio = {ratio:.2f} — clusters may not be meaningful")


if __name__ == "__main__":
    main()
