#!/usr/bin/env python3
"""Plotting for intergenic feature specificity analysis.

Can be run standalone to re-generate plots from saved data, or imported
by intergenic_feature_analysis.py.

Usage (standalone, to tweak plots without re-running analysis):
    python tools/plot_intergenic_features.py --run_dir results/NC_000913.3/intergenic_analysis/YYYYMMDD_HHMMSS_*/
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def plot_volcano(df, fdr_threshold, min_fold_change, out_path):
    """Volcano plot: log2(fold_change) vs -log10(FDR q-value)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    x = df["log2_fold_change"].values
    y = -np.log10(df["fdr_qvalue"].values + 1e-300)

    sig = (df["fdr_qvalue"] < fdr_threshold) & (df["fold_change"] > min_fold_change)
    sig_genic = (df["fdr_qvalue"] < fdr_threshold) & (df["fold_change"] < 1.0 / min_fold_change)

    ax.scatter(x[~sig & ~sig_genic], y[~sig & ~sig_genic], c="grey", alpha=0.3, s=4, label="Not significant")
    ax.scatter(x[sig_genic], y[sig_genic], c="blue", alpha=0.5, s=8, label="Genic-specific")
    ax.scatter(x[sig], y[sig], c="red", alpha=0.5, s=8, label=f"Intergenic-specific (n={sig.sum()})")

    ax.axhline(-np.log10(fdr_threshold), ls="--", c="grey", alpha=0.5)
    ax.axvline(np.log2(min_fold_change), ls="--", c="grey", alpha=0.5)
    ax.axvline(-np.log2(min_fold_change), ls="--", c="grey", alpha=0.5)

    ax.set_xlabel("log2(Fold Change: Intergenic / Genic)")
    ax.set_ylabel("-log10(FDR q-value)")
    ax.set_title("Feature Specificity: Intergenic vs Genic Regions")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved volcano plot: {out_path}")


ANNOTATION_COLORS = {"CDS": "blue", "UTR/exon": "cyan", "Intron": "green", "Intergenic": "red"}


def plot_top_features_heatmap(vectors, annotations, top_features, out_path):
    """Heatmap: rows=top features, cols=regions sorted by annotation."""
    annotations = np.array(annotations)
    sort_idx = np.argsort(annotations)
    sorted_annot = annotations[sort_idx]

    mat = vectors[sort_idx][:, top_features].T

    fig, ax = plt.subplots(figsize=(14, max(4, len(top_features) * 0.3 + 1)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")

    unique_labels = sorted(set(annotations))
    for label in unique_labels:
        idxs = np.where(sorted_annot == label)[0]
        if len(idxs):
            ax.axvline(idxs[0] - 0.5, c="white", lw=0.5, alpha=0.7)

    ax.set_ylabel("Feature index")
    ax.set_xlabel("Regions (sorted by annotation)")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=7)
    ax.set_title("Top Intergenic-Specific Features × Regions")
    plt.colorbar(im, ax=ax, label="Activation")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap: {out_path}")


def plot_activation_distributions(vectors, annotations, top_features, out_path):
    """Violin plots comparing intergenic vs genic for top features."""
    annotations = np.array(annotations)
    is_intergenic = annotations == "Intergenic"
    n_feats = min(len(top_features), 12)
    top_features = top_features[:n_feats]

    nrows = (n_feats + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 3.5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, feat_idx in enumerate(top_features):
        ax = axes[i]
        vals_ig = vectors[is_intergenic, feat_idx]
        vals_g = vectors[~is_intergenic, feat_idx]
        parts = ax.violinplot([vals_ig, vals_g], positions=[0, 1], showmedians=True)
        for pc, color in zip(parts["bodies"], ["red", "blue"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Intergenic", "Genic"])
        ax.set_title(f"Feature {feat_idx}", fontsize=9)

    for i in range(n_feats, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Activation Distributions: Top Intergenic-Specific Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved distributions: {out_path}")


def plot_tsne_highlight(vectors, annotations, top_features, out_path):
    """t-SNE colored by annotation, with top feature activation overlay."""
    from sklearn.manifold import TSNE

    n = vectors.shape[0]
    perplexity = min(30, n - 1)
    if n < 5:
        logger.warning("Too few regions for t-SNE, skipping")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, metric="cosine")
    coords = tsne.fit_transform(vectors)

    annotations = np.array(annotations)
    unique_labels = sorted(set(annotations))

    n_panels = 1 + min(len(top_features), 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    for label in unique_labels:
        mask = annotations == label
        ax.scatter(coords[mask, 0], coords[mask, 1], c=ANNOTATION_COLORS.get(label, "grey"),
                   s=10, alpha=0.6, label=label)
    ax.legend(fontsize=7)
    ax.set_title("By Annotation")

    for i, feat_idx in enumerate(top_features[:n_panels - 1]):
        ax = axes[i + 1]
        vals = vectors[:, feat_idx]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="hot", s=10, alpha=0.7)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Feature {feat_idx}")

    fig.suptitle("t-SNE of SAE Regions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved t-SNE: {out_path}")


def plot_annotation_counts(annotations, out_path):
    """Bar chart of region annotation counts."""
    labels, counts = np.unique(annotations, return_counts=True)
    colors = [ANNOTATION_COLORS.get(l, "grey") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color=colors)
    for i, (l, c) in enumerate(zip(labels, counts)):
        ax.text(i, c + 1, f"{c}\n({100*c/sum(counts):.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Number of regions")
    ax.set_title("Region Annotation Distribution")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved annotation counts: {out_path}")


def generate_all_plots(run_dir, vectors=None, annotations=None, fdr_threshold=0.05, min_fold_change=2.0, top_n=20):
    """Generate all plots from a completed analysis run directory.

    If vectors/annotations are None, loads them from the run_dir data files.
    """
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load specificity table
    df = pd.read_csv(os.path.join(data_dir, "feature_specificity.tsv"), sep="\t")

    # Load annotations if not provided
    if annotations is None:
        annot_df = pd.read_csv(os.path.join(data_dir, "region_annotations.tsv"), sep="\t")
        annotations = annot_df["annotation"].tolist()

    # Load vectors if not provided (need for heatmap/violin/tsne)
    if vectors is None:
        summary = json.load(open(os.path.join(data_dir, "summary.json")))
        sae_run = summary.get("sae_run")
        if sae_run:
            # Resolve relative path from source.json
            source_path = os.path.join(run_dir, "source.json")
            if os.path.exists(source_path):
                source = json.load(open(source_path))
                sae_run_dir = os.path.normpath(os.path.join(run_dir, source["sae_run"]))
            else:
                sae_run_dir = sae_run
            from tools.aggregate_genome_sae_stats import load_maxpooled_vectors
            vectors = load_maxpooled_vectors(sae_run_dir)
            if vectors is not None:
                vectors = vectors[:len(annotations)]

    # Get top features
    sig = df[(df["fdr_qvalue"] < fdr_threshold) & (df["fold_change"] > min_fold_change)]
    sig = sig.sort_values("specificity_index", ascending=False)
    if len(sig) > 0:
        top_features = sig["feature_idx"].values[:top_n].astype(int)
    else:
        top_features = df.sort_values("specificity_index", ascending=False)["feature_idx"].values[:top_n].astype(int)

    # Generate plots
    plot_volcano(df, fdr_threshold, min_fold_change, os.path.join(plots_dir, "volcano_plot.png"))
    plot_annotation_counts(annotations, os.path.join(plots_dir, "annotation_counts.png"))

    if vectors is not None and len(top_features) > 0:
        plot_top_features_heatmap(vectors, annotations, top_features, os.path.join(plots_dir, "top_features_heatmap.png"))
        plot_activation_distributions(vectors, annotations, top_features, os.path.join(plots_dir, "activation_distributions.png"))
        plot_tsne_highlight(vectors, annotations, top_features, os.path.join(plots_dir, "tsne_top_features.png"))
    elif vectors is None:
        logger.warning("Could not load vectors — skipping heatmap/violin/tsne plots")

    logger.info(f"All plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Re-generate intergenic analysis plots from saved data")
    parser.add_argument("--run_dir", required=True, help="Path to completed intergenic_analysis run directory")
    parser.add_argument("--fdr_threshold", type=float, default=0.05)
    parser.add_argument("--min_fold_change", type=float, default=2.0)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.isdir(args.run_dir):
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)

    generate_all_plots(args.run_dir, fdr_threshold=args.fdr_threshold,
                       min_fold_change=args.min_fold_change, top_n=args.top_n)


if __name__ == "__main__":
    main()
