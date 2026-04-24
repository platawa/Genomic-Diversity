#!/usr/bin/env python3
"""
activation_distribution.py

For a given scope (chrom or genome_wide):
  - Compute per-feature activation distribution statistics (mean, std, skew,
    kurtosis, Shapiro-Wilk p-value on a random 5000-region sample)
  - Emit histograms for the K most-variable features
  - Emit QQ plots vs normal for the same K features
  - Emit scatter of skew vs kurtosis, color-coded by fraction-nonzero
  - Per-cluster breakdown if cluster_assignments.tsv has a `cluster_id` column

The output answers: "are activation distributions Gaussian or something else?"
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


def load_scope(scope_dir):
    paths = {
        "maxpool": os.path.join(scope_dir, "data", "maxpooled_vectors.npy"),
        "ca": os.path.join(scope_dir, "data", "cluster_assignments.tsv"),
    }
    mp = np.load(paths["maxpool"]) if os.path.isfile(paths["maxpool"]) else None
    ca = pd.read_csv(paths["ca"], sep="\t", comment="#") if os.path.isfile(paths["ca"]) else None
    return mp, ca


def summarize(vectors, shapiro_sample_size=5000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n_regions, n_feat = vectors.shape
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    skews = scistats.skew(vectors, axis=0)
    kurt = scistats.kurtosis(vectors, axis=0)
    frac_nonzero = (vectors != 0).mean(axis=0)
    # Shapiro on subsample (only works for n<=5000)
    if n_regions > shapiro_sample_size:
        sample_idx = rng.choice(n_regions, shapiro_sample_size, replace=False)
        sample = vectors[sample_idx]
    else:
        sample = vectors
    shapiro_p = np.full(n_feat, np.nan)
    for f in range(n_feat):
        col = sample[:, f]
        if col.std() == 0:
            continue
        try:
            _, p = scistats.shapiro(col)
            shapiro_p[f] = p
        except Exception:
            pass
    return pd.DataFrame({
        "feature_idx": np.arange(n_feat),
        "mean": means,
        "std": stds,
        "skew": skews,
        "kurtosis": kurt,
        "frac_nonzero": frac_nonzero,
        "shapiro_p": shapiro_p,
    })


def plot_top_feature_histograms(vectors, top_feats, out_path):
    n = len(top_feats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, f in enumerate(top_feats):
        ax = axes[i // cols][i % cols]
        v = vectors[:, f]
        ax.hist(v, bins=80, color="steelblue", alpha=0.85)
        ax.set_title(f"feat {f} (σ={v.std():.3f})")
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_qq(vectors, top_feats, out_path):
    n = len(top_feats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for i, f in enumerate(top_feats):
        ax = axes[i // cols][i % cols]
        scistats.probplot(vectors[:, f], dist="norm", plot=ax)
        ax.set_title(f"feat {f}")
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_skew_kurt_scatter(summary, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(summary["skew"], summary["kurtosis"],
                    c=summary["frac_nonzero"], s=6, cmap="viridis", alpha=0.7)
    ax.axvline(0, color="gray", lw=0.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("skewness")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("Per-feature skew vs kurtosis (color = fraction nonzero)")
    plt.colorbar(sc, ax=ax, label="fraction nonzero")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scope_dir", required=True)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mp, ca = load_scope(args.scope_dir)
    if mp is None:
        logger.error("maxpooled_vectors.npy missing")
        return 2

    out_dir = args.output_dir or os.path.join(args.scope_dir, "activation_distribution")
    os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    summary = summarize(mp)
    summary.to_csv(os.path.join(out_dir, "feature_distribution_stats.tsv"),
                   sep="\t", index=False)

    top_feats = np.argsort(-summary["std"].values)[:args.top_k]
    plot_top_feature_histograms(mp, top_feats,
                                os.path.join(out_dir, "top_feature_histograms.png"))
    plot_qq(mp, top_feats, os.path.join(out_dir, "top_feature_qq.png"))
    plot_skew_kurt_scatter(summary, os.path.join(out_dir, "skew_kurtosis_scatter.png"))

    # Fraction of features whose Shapiro-Wilk rejects normality at p<0.05
    non_normal = (summary["shapiro_p"] < 0.05).sum()
    logger.info(f"Shapiro-Wilk: {non_normal}/{len(summary)} features reject normality (p<0.05)")

    write_completed(out_dir, "activation_distribution.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
