#!/usr/bin/env python3
"""
investigate_bacteria_cds_clusters.py

Investigate why CDS regions split into 2 (or more) clusters in bacteria
SAE latent analysis. Filters to CDS-only regions, applies k=2 clustering,
and generates:
  1. t-SNE/UMAP with CDS clusters highlighted
  2. Differential feature activation between CDS cluster 1 vs 2
  3. Top discriminative features heatmap
  4. Example regions from each cluster with per-nucleotide feature traces
  5. Genomic position distribution per cluster

Uses existing patterns from inspect_cluster_examples.py and
cross_organism_features.py.

Usage:
    python tools/investigate_bacteria_cds_clusters.py \
        --chrom NC_000913.3 --organism ecoli \
        --gtf /path/to/ecoli_genomic.gtf \
        --results_dir results/ \
        --n_clusters 2 --n_examples 3 --top_k 5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_utils import find_latest_completed
from tools.plot_tsne_by_annotation import load_gtf_features, classify_region
from tools.inspect_cluster_examples import (
    find_latest_merged_run,
    get_top_features,
    pick_representative_regions,
    plot_feature_traces,
)

logger = logging.getLogger(__name__)

ORGANISM_GTF = {
    "ecoli": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf",
    "bacillus": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf",
}


def load_latent_data(results_dir, chrom, latent_subdir="latent_analysis"):
    """Load cluster assignments, embeddings, and maxpooled vectors."""
    sae_dir = os.path.join(results_dir, chrom, "sae")

    def _has_valid_data(d):
        ca = os.path.join(d, "data", "cluster_assignments.tsv")
        return os.path.isfile(ca) and os.path.getsize(ca) > 500

    # Try under latest completed SAE run first (most reliable)
    latent_dir = None
    sae_run = find_latest_completed(results_dir, chrom, "sae")
    if sae_run:
        candidate = os.path.join(sae_run, latent_subdir)
        if _has_valid_data(candidate):
            latent_dir = candidate

    # Fall back to top-level sae/latent_analysis/
    if latent_dir is None:
        candidate = os.path.join(sae_dir, latent_subdir)
        if _has_valid_data(candidate):
            latent_dir = candidate

    if latent_dir is None:
        logger.error(f"No valid latent data found under {sae_dir}")
        return None, None, None
    data_dir = os.path.join(latent_dir, "data")

    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")
    mp_path = os.path.join(data_dir, "maxpooled_vectors.npy")

    if not os.path.isfile(ca_path):
        logger.error(f"Missing: {ca_path}")
        return None, None

    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    mp = np.load(mp_path) if os.path.isfile(mp_path) else None

    # Load embeddings
    for name, fname in [("tsne", "embedding_tsne.npy"), ("umap", "embedding_umap.npy")]:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            emb = np.load(path)
            ca[f"{name}_1"] = emb[:, 0]
            ca[f"{name}_2"] = emb[:, 1]
        elif f"{name}_1" not in ca.columns:
            pass  # no embedding available

    logger.info(f"Loaded {len(ca)} regions from {ca_path}")
    return ca, mp, latent_dir


def classify_all_regions(ca, gtf_path, chrom_id):
    """Add annotation labels to all regions."""
    intervals = load_gtf_features(gtf_path, chrom_id)
    labels = []
    for _, row in ca.iterrows():
        labels.append(classify_region(int(row["genomic_start"]),
                                      int(row["genomic_end"]), intervals))
    ca["annotation"] = labels
    counts = ca["annotation"].value_counts()
    logger.info(f"Annotation counts:\n{counts}")
    return ca


def cluster_cds_regions(ca, mp, n_clusters=2):
    """Apply k-means clustering on CDS-only regions."""
    from sklearn.cluster import KMeans

    cds_mask = ca["annotation"] == "CDS"
    cds_indices = np.where(cds_mask)[0]
    n_cds = len(cds_indices)
    logger.info(f"CDS regions: {n_cds} / {len(ca)} total")

    if n_cds < n_clusters * 2:
        logger.error(f"Too few CDS regions ({n_cds}) for {n_clusters} clusters")
        return None, None, None

    cds_vectors = mp[cds_indices]

    # K-means on max-pooled vectors
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cds_labels = km.fit_predict(cds_vectors)

    # Log cluster sizes
    for k in range(n_clusters):
        logger.info(f"  CDS cluster {k}: {(cds_labels == k).sum()} regions")

    return cds_indices, cds_labels, km


def compute_differential_features(mp, cds_indices, cds_labels, n_clusters=2):
    """Find features that discriminate between CDS clusters."""
    results = {}
    cds_vectors = mp[cds_indices]

    for k in range(n_clusters):
        mask_k = cds_labels == k
        mask_other = ~mask_k

        mean_k = cds_vectors[mask_k].mean(axis=0)
        mean_other = cds_vectors[mask_other].mean(axis=0)

        # Difference in mean activation
        diff = mean_k - mean_other

        # Fraction of regions where feature is active
        frac_k = (cds_vectors[mask_k] > 0).mean(axis=0)
        frac_other = (cds_vectors[mask_other] > 0).mean(axis=0)

        # Top features enriched in cluster k
        top_enriched = np.argsort(diff)[::-1][:30]
        # Top features depleted in cluster k
        top_depleted = np.argsort(diff)[:30]

        results[k] = {
            "mean_activation": mean_k,
            "mean_other": mean_other,
            "diff": diff,
            "frac_active": frac_k,
            "frac_active_other": frac_other,
            "top_enriched": top_enriched,
            "top_depleted": top_depleted,
        }

    return results


def plot_cds_clusters_embedding(ca, cds_indices, cds_labels, n_clusters, out_dir):
    """Plot t-SNE/UMAP with CDS clusters highlighted, non-CDS as gray."""
    for emb_name in ["tsne", "umap"]:
        col1, col2 = f"{emb_name}_1", f"{emb_name}_2"
        if col1 not in ca.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Left: all regions colored by annotation, CDS split by cluster
        ax = axes[0]
        annot_colors = {
            "UTR/exon": "#4CAF50", "Intron": "#FF9800", "Intergenic": "#F44336",
        }
        for annot, color in annot_colors.items():
            mask = ca["annotation"] == annot
            if mask.sum() == 0:
                continue
            ax.scatter(ca.loc[mask, col1], ca.loc[mask, col2],
                       s=8, alpha=0.3, c=color, label=annot, edgecolors="none")

        # CDS colored by cluster
        cluster_colors = plt.cm.Set1(np.linspace(0, 0.5, n_clusters))
        for k in range(n_clusters):
            mask = cds_labels == k
            idx = cds_indices[mask]
            n_k = mask.sum()
            ax.scatter(ca.iloc[idx][col1], ca.iloc[idx][col2],
                       s=12, alpha=0.6, c=[cluster_colors[k]], edgecolors="none",
                       label=f"CDS cluster {k} (n={n_k})")

        ax.legend(markerscale=3, fontsize=9)
        ax.set_xlabel(f"{emb_name.upper()} 1")
        ax.set_ylabel(f"{emb_name.upper()} 2")
        ax.set_title(f"All Regions — CDS Split by K-means Cluster")

        # Right: CDS only, colored by cluster
        ax = axes[1]
        for k in range(n_clusters):
            mask = cds_labels == k
            idx = cds_indices[mask]
            ax.scatter(ca.iloc[idx][col1], ca.iloc[idx][col2],
                       s=15, alpha=0.6, c=[cluster_colors[k]], edgecolors="none",
                       label=f"Cluster {k} (n={mask.sum()})")
        ax.legend(markerscale=3, fontsize=10)
        ax.set_xlabel(f"{emb_name.upper()} 1")
        ax.set_ylabel(f"{emb_name.upper()} 2")
        ax.set_title(f"CDS Regions Only — K-means Clusters")

        fig.suptitle(f"Bacteria CDS Cluster Investigation ({emb_name.upper()})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{emb_name}_cds_clusters.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {emb_name}_cds_clusters.png")


def plot_differential_heatmap(diff_results, n_clusters, out_dir):
    """Plot heatmap of top discriminative features between CDS clusters."""
    n_top = 20
    fig, axes = plt.subplots(1, n_clusters, figsize=(8 * n_clusters, 8))
    if n_clusters == 1:
        axes = [axes]

    for k in range(n_clusters):
        ax = axes[k]
        res = diff_results[k]
        top_feats = res["top_enriched"][:n_top]

        # Build matrix: rows = features, cols = [cluster_k_mean, other_mean, diff, frac_k, frac_other]
        data = np.column_stack([
            res["mean_activation"][top_feats],
            res["mean_other"][top_feats],
            res["frac_active"][top_feats],
            res["frac_active_other"][top_feats],
        ])

        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(n_top))
        ax.set_yticklabels([f"f{i}" for i in top_feats], fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Mean (this)", "Mean (other)",
                            "% active (this)", "% active (other)"],
                           rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Cluster {k} — Top {n_top} Enriched Features")
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Differential Feature Activation Between CDS Clusters", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "differential_features_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved differential_features_heatmap.png")


def plot_genomic_position_by_cluster(ca, cds_indices, cds_labels, n_clusters, out_dir):
    """Show genomic position distribution for each CDS cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of genomic positions by cluster
    ax = axes[0]
    cluster_colors = plt.cm.Set1(np.linspace(0, 0.5, n_clusters))
    for k in range(n_clusters):
        mask = cds_labels == k
        positions = ca.iloc[cds_indices[mask]]["genomic_start"].values / 1e6
        ax.hist(positions, bins=50, alpha=0.5, color=cluster_colors[k],
                label=f"Cluster {k} (n={mask.sum()})")
    ax.set_xlabel("Genomic Position (Mbp)")
    ax.set_ylabel("Count")
    ax.set_title("CDS Cluster Distribution Along Genome")
    ax.legend()

    # Right: region length by cluster
    ax = axes[1]
    for k in range(n_clusters):
        mask = cds_labels == k
        idx = cds_indices[mask]
        lengths = (ca.iloc[idx]["genomic_end"] - ca.iloc[idx]["genomic_start"]).values
        ax.hist(lengths, bins=30, alpha=0.5, color=cluster_colors[k],
                label=f"Cluster {k} (med={np.median(lengths):.0f})")
    ax.set_xlabel("Region Length (bp)")
    ax.set_ylabel("Count")
    ax.set_title("CDS Region Lengths by Cluster")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cds_cluster_genomic_distribution.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cds_cluster_genomic_distribution.png")


def plot_cluster_feature_traces(ca, mp, cds_indices, cds_labels, n_clusters,
                                 sae_dir, n_examples, top_k, out_dir):
    """Plot per-nucleotide feature traces for examples from each CDS cluster."""
    merged_run = find_latest_merged_run(sae_dir)
    if merged_run is None:
        logger.warning("No feature_matrices.npz found, skipping feature traces")
        return

    fm_path = merged_run / "data" / "feature_matrices.npz"
    fm = np.load(str(fm_path), allow_pickle=True)
    fm_keys = set(fm.keys())

    cluster_top_features = {}

    for k in range(n_clusters):
        mask = cds_labels == k
        k_indices = cds_indices[mask]
        n_k = len(k_indices)

        # Pick examples closest to centroid
        cds_df = ca.iloc[k_indices].copy()
        cds_df.index = np.arange(len(cds_df))
        mp_k = mp[k_indices]

        if n_k <= n_examples:
            rep_local = np.arange(n_k)
        else:
            centroid = mp_k.mean(axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            norms = np.linalg.norm(mp_k, axis=1, keepdims=True) + 1e-8
            sims = (mp_k / norms) @ centroid_norm
            rep_local = np.argsort(sims)[-n_examples:][::-1]

        rep_global = k_indices[rep_local]

        # Get cluster-level top features from centroid
        centroid = mp_k.mean(axis=0)
        cluster_feats = get_top_features(centroid, top_k)
        cluster_top_features[k] = cluster_feats
        logger.info(f"CDS cluster {k}: top features = {cluster_feats}")

        # Plot
        n_ex = len(rep_global)
        fig, axes = plt.subplots(n_ex, 1, figsize=(14, 4 * n_ex), squeeze=False)
        fig.suptitle(f"CDS Cluster {k} — {n_k} regions — Top-{top_k} Feature Traces",
                     fontsize=13, y=1.02)

        for ei, ridx in enumerate(rep_global):
            region_key = f"region_{ridx}"
            row = ca.iloc[ridx]
            region_info = {
                "genomic_start": int(row["genomic_start"]),
                "genomic_end": int(row["genomic_end"]),
                "region_length": int(row.get("region_length",
                                             row["genomic_end"] - row["genomic_start"])),
            }

            if region_key not in fm_keys:
                axes[ei, 0].text(0.5, 0.5, f"{region_key} not in feature_matrices",
                                 transform=axes[ei, 0].transAxes, ha="center")
                continue

            feature_matrix = fm[region_key]
            top_feats = get_top_features(mp[ridx], top_k)

            plot_feature_traces(
                feature_matrix, top_feats, region_info, axes[ei, 0],
                title=f"Example {ei+1} (region {ridx}) — CDS cluster {k}"
            )

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"cds_cluster_{k}_examples.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved cds_cluster_{k}_examples.png")

    # Cross-cluster comparison
    if n_clusters >= 2 and cluster_top_features:
        logger.info("Generating cross-cluster CDS feature comparison")
        fig = plt.figure(figsize=(5 * n_clusters, 4 * n_clusters))
        gs = GridSpec(n_clusters, n_clusters, figure=fig, hspace=0.4, wspace=0.3)

        for row_i in range(n_clusters):
            src_features = cluster_top_features.get(row_i, [])
            if not src_features:
                continue
            src_indices = cds_indices[cds_labels == row_i]

            for col_j in range(n_clusters):
                ax = fig.add_subplot(gs[row_i, col_j])
                tgt_indices = cds_indices[cds_labels == col_j]
                if len(tgt_indices) == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                    continue

                # Use centroid-nearest example from target cluster
                mp_tgt = mp[tgt_indices]
                centroid = mp_tgt.mean(axis=0)
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                norms = np.linalg.norm(mp_tgt, axis=1, keepdims=True) + 1e-8
                best = tgt_indices[np.argmax((mp_tgt / norms) @ centroid_norm)]

                region_key = f"region_{best}"
                row = ca.iloc[best]
                region_info = {
                    "genomic_start": int(row["genomic_start"]),
                    "genomic_end": int(row["genomic_end"]),
                    "region_length": int(row.get("region_length",
                                                 row["genomic_end"] - row["genomic_start"])),
                }

                if region_key in fm_keys:
                    feature_matrix = fm[region_key]
                    plot_feature_traces(
                        feature_matrix, src_features, region_info, ax,
                        title=f"C{row_i} features → C{col_j} example"
                    )
                else:
                    ax.text(0.5, 0.5, "Missing", ha="center", transform=ax.transAxes)

                if row_i == 0:
                    ax.set_title(f"Target: CDS Cluster {col_j}\n{ax.get_title()}", fontsize=9)
                if col_j == 0:
                    ax.set_ylabel(f"Source: C{row_i}\n{ax.get_ylabel()}", fontsize=9)

        fig.suptitle("Cross-CDS-Cluster Feature Comparison", fontsize=14, y=1.02)
        fig.savefig(os.path.join(out_dir, "cds_cross_cluster_comparison.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved cds_cross_cluster_comparison.png")

    fm.close()

    # Save summary
    summary_rows = []
    for k in range(n_clusters):
        feats = cluster_top_features.get(k, [])
        n_k = (cds_labels == k).sum()
        summary_rows.append({
            "cluster": k,
            "n_regions": n_k,
            "top_features": ",".join(str(f) for f in feats),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "cds_cluster_summary.tsv"),
                      sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Investigate bacteria CDS cluster split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chrom", required=True, help="Chromosome (e.g. NC_000913.3)")
    parser.add_argument("--organism", default="ecoli", choices=["ecoli", "bacillus"],
                        help="Organism name")
    parser.add_argument("--gtf", default=None,
                        help="GTF path (auto-detected from organism if omitted)")
    parser.add_argument("--results_dir", default="results/", help="Base results directory")
    parser.add_argument("--latent_subdir", default="latent_analysis",
                        help="Latent analysis subdirectory")
    parser.add_argument("--n_clusters", type=int, default=2,
                        help="Number of CDS clusters to find (default: 2)")
    parser.add_argument("--n_examples", type=int, default=3,
                        help="Number of example regions per cluster")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top features to plot per region")
    parser.add_argument("--output_dir", default=None, help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve GTF
    gtf_path = args.gtf or ORGANISM_GTF.get(args.organism)
    if not gtf_path:
        logger.error(f"No GTF path for organism {args.organism}. Use --gtf.")
        sys.exit(1)

    # Load data
    logger.info(f"Loading latent data for {args.chrom}...")
    result = load_latent_data(args.results_dir, args.chrom, args.latent_subdir)
    if result[0] is None:
        sys.exit(1)
    ca, mp, latent_dir = result

    if mp is None:
        logger.error("No maxpooled vectors available")
        sys.exit(1)

    # Classify regions
    logger.info("Classifying regions by annotation...")
    ca = classify_all_regions(ca, gtf_path, args.chrom)

    # Cluster CDS regions
    logger.info(f"Clustering CDS regions into {args.n_clusters} groups...")
    cds_indices, cds_labels, km = cluster_cds_regions(ca, mp, args.n_clusters)
    if cds_indices is None:
        sys.exit(1)

    # Output directory
    out_dir = args.output_dir or os.path.join(
        latent_dir, "cds_cluster_investigation"
    )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    # Differential features
    logger.info("Computing differential features...")
    diff_results = compute_differential_features(mp, cds_indices, cds_labels, args.n_clusters)

    # Generate all plots
    logger.info("Generating embedding plots...")
    plot_cds_clusters_embedding(ca, cds_indices, cds_labels, args.n_clusters, out_dir)

    logger.info("Generating differential heatmap...")
    plot_differential_heatmap(diff_results, args.n_clusters, out_dir)

    logger.info("Generating genomic distribution plots...")
    plot_genomic_position_by_cluster(ca, cds_indices, cds_labels, args.n_clusters, out_dir)

    logger.info("Generating per-nucleotide feature traces...")
    sae_dir = os.path.join(args.results_dir, args.chrom, "sae")
    plot_cluster_feature_traces(
        ca, mp, cds_indices, cds_labels, args.n_clusters,
        sae_dir, args.n_examples, args.top_k, out_dir,
    )

    # Save cluster assignments
    ca_cds = ca.iloc[cds_indices].copy()
    ca_cds["cds_cluster"] = cds_labels
    ca_cds.to_csv(os.path.join(out_dir, "cds_cluster_assignments.tsv"),
                  sep="\t", index=False)

    # Save differential features as JSON
    diff_json = {}
    for k, res in diff_results.items():
        diff_json[str(k)] = {
            "top_enriched": res["top_enriched"].tolist(),
            "top_depleted": res["top_depleted"].tolist(),
            "top_enriched_activations": res["mean_activation"][res["top_enriched"]].tolist(),
            "top_enriched_frac_active": res["frac_active"][res["top_enriched"]].tolist(),
        }
    with open(os.path.join(out_dir, "differential_features.json"), "w") as f:
        json.dump(diff_json, f, indent=2)

    logger.info(f"\nDone. Results in {out_dir}")
    logger.info(f"  Files: {sorted(os.listdir(out_dir))}")


if __name__ == "__main__":
    main()
