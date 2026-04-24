#!/usr/bin/env python3
"""
normalization_distribution_analysis.py

Quantifies how raw, prenorm (Option A: per-nucleotide z-score before max-pool),
and postnorm (Option B: max-pool then z-score with chunk-max stats) reshape the
max-pooled SAE feature distribution.

Per chromosome (and aggregated genome-wide), emits:
  - histogram_comparison.png       — raw/prenorm/postnorm histograms side by side
  - per_feature_stats.png          — per-feature μ and σ bar chart + skewness
  - qq_vs_normal.png               — QQ plot versus a normal reference (sampled features)
  - summary.tsv                    — stats per feature (n regions, nonzero, mean, std,
                                     skew, kurtosis) per normalization mode

Also answers: if we normalize per-chromosome with chromosome stats vs with
genome-wide stats, how different are the resulting distributions?
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scistats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed

logger = logging.getLogger(__name__)

MODES = ["raw", "prenorm", "postnorm"]
SUBDIR = {
    "raw": "latent_analysis",
    "prenorm": "latent_analysis_prenorm",
    "postnorm": "latent_analysis_postnorm",
}


def load_vectors(results_dir, chrom, mode):
    path = os.path.join(results_dir, chrom, "sae", SUBDIR[mode], "data", "maxpooled_vectors.npy")
    if not os.path.isfile(path):
        logger.warning(f"Missing {mode} vectors for {chrom}: {path}")
        return None
    return np.load(path)


def summarize(vectors):
    """Per-feature summary: mean, std, skew, kurtosis, fraction nonzero."""
    n_regions, n_feat = vectors.shape
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    nonzero = (vectors != 0).mean(axis=0)
    # Per-feature skew + kurtosis (slow on 32K features; subsample if needed)
    skews = scistats.skew(vectors, axis=0)
    kurt = scistats.kurtosis(vectors, axis=0)
    return pd.DataFrame({
        "feature_idx": np.arange(n_feat),
        "mean": means,
        "std": stds,
        "frac_nonzero": nonzero,
        "skew": skews,
        "kurtosis": kurt,
    })


def plot_histograms(mode_vectors, out_path, title_prefix=""):
    """3-panel histogram: flatten each mode's full matrix and overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (mode, v) in zip(axes, mode_vectors.items()):
        if v is None:
            ax.text(0.5, 0.5, f"{mode}\nunavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mode)
            continue
        flat = v.ravel()
        # Subsample to avoid memory blow-up for 30K * 40K = 1.2B values
        if flat.size > 2_000_000:
            flat = np.random.default_rng(0).choice(flat, 2_000_000, replace=False)
        ax.hist(flat, bins=200, log=True, color="steelblue", alpha=0.85)
        ax.set_title(f"{mode} (n={v.shape[0]} regions × {v.shape[1]} feats)")
        ax.set_xlabel("activation")
    axes[0].set_ylabel("count (log)")
    fig.suptitle(f"{title_prefix} activation distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_per_feature_mu_sigma(summaries, out_path, title_prefix=""):
    """Compare per-feature μ and σ across modes (first 500 features for readability)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    colors = {"raw": "tab:gray", "prenorm": "tab:blue", "postnorm": "tab:orange"}
    n_show = 500
    x = np.arange(n_show)
    for mode, df in summaries.items():
        if df is None:
            continue
        ax1.plot(x, df["mean"].values[:n_show], color=colors[mode], label=mode, lw=0.8)
        ax2.plot(x, df["std"].values[:n_show], color=colors[mode], label=mode, lw=0.8)
    ax1.set_ylabel("per-feature mean")
    ax2.set_ylabel("per-feature std")
    ax2.set_xlabel("feature index (first 500)")
    ax1.legend(loc="upper right")
    ax1.set_title(f"{title_prefix} per-feature μ (top) and σ (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_qq(mode_vectors, out_path, n_features_sampled=6, title_prefix=""):
    """QQ vs normal for a handful of active features across modes."""
    available = [m for m, v in mode_vectors.items() if v is not None]
    if not available:
        return
    # Pick features with nonzero variance across all modes
    ref = mode_vectors[available[0]]
    var = ref.std(axis=0)
    top_features = np.argsort(-var)[:n_features_sampled]
    rows = len(top_features)
    cols = len(available)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for r, feat in enumerate(top_features):
        for c, mode in enumerate(available):
            v = mode_vectors[mode][:, feat]
            scistats.probplot(v, dist="norm", plot=axes[r, c])
            axes[r, c].set_title(f"{mode} feat={feat}")
    fig.suptitle(f"{title_prefix} QQ vs normal (top-variance features)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def compare_chrom_vs_genome_norm(chrom, results_dir, global_stats_path, out_path):
    """For a chromosome's raw vectors, apply (a) chromosome-local z-score and
    (b) genome-wide z-score using nuc_mean/nuc_std, and compare distributions.
    """
    raw = load_vectors(results_dir, chrom, "raw")
    if raw is None:
        return
    gstats = np.load(global_stats_path)
    mu_g = gstats["nuc_mean"] if "nuc_mean" in gstats.files else gstats["global_mean"]
    sd_g = gstats["nuc_std"] if "nuc_std" in gstats.files else gstats["global_std"]
    sd_g = np.where(sd_g > 0, sd_g, 1.0)
    genome_norm = (raw - mu_g) / sd_g
    mu_c = raw.mean(axis=0)
    sd_c = raw.std(axis=0)
    sd_c = np.where(sd_c > 0, sd_c, 1.0)
    chrom_norm = (raw - mu_c) / sd_c

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (label, v) in zip(axes, [
        ("chrom-local z-score", chrom_norm),
        ("genome-wide z-score", genome_norm),
    ]):
        flat = v.ravel()
        if flat.size > 2_000_000:
            flat = rng.choice(flat, 2_000_000, replace=False)
        ax.hist(flat, bins=200, log=True, color="tab:purple", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("z-score")
    axes[0].set_ylabel("count (log)")
    fig.suptitle(f"{chrom}: chrom-local vs genome-wide z-score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def process_chrom(chrom, args, out_dir):
    chrom_dir = os.path.join(out_dir, chrom)
    os.makedirs(chrom_dir, exist_ok=True)
    mode_vectors = {m: load_vectors(args.results_dir, chrom, m) for m in MODES}
    if all(v is None for v in mode_vectors.values()):
        logger.warning(f"{chrom}: nothing to process")
        return
    plot_histograms(mode_vectors, os.path.join(chrom_dir, "histogram_comparison.png"),
                    title_prefix=chrom)
    summaries = {m: (summarize(v) if v is not None else None) for m, v in mode_vectors.items()}
    plot_per_feature_mu_sigma(summaries, os.path.join(chrom_dir, "per_feature_stats.png"),
                              title_prefix=chrom)
    plot_qq(mode_vectors, os.path.join(chrom_dir, "qq_vs_normal.png"), title_prefix=chrom)
    if args.global_stats:
        compare_chrom_vs_genome_norm(chrom, args.results_dir, args.global_stats,
                                     os.path.join(chrom_dir, "chrom_vs_genome_zscore.png"))
    for m, df in summaries.items():
        if df is None:
            continue
        df.to_csv(os.path.join(chrom_dir, f"summary_{m}.tsv"), sep="\t", index=False)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--chroms", nargs="+",
                   default=["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
                            "chr8", "chr9", "chr10", "chr11", "chr12", "chr13",
                            "chr14", "chr15", "chr16", "chr17", "chr18", "chr20",
                            "chr21", "chr22", "chrX", "chrY"])
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--global_stats", default=None,
                   help="Path to genome_wide_sae_stats_corrected.npz for chrom-vs-genome comparison")
    p.add_argument("--output_dir", default=None,
                   help="Default: results/_genome_wide/normalization_distribution/<timestamp>/")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.output_dir is None:
        out_dir = build_run_dir(
            args.results_dir, "_genome_wide", "normalization_distribution", "all_chroms")
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    for chrom in args.chroms:
        logger.info(f"=== {chrom} ===")
        process_chrom(chrom, args, out_dir)

    write_completed(out_dir, "normalization_distribution_analysis.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
