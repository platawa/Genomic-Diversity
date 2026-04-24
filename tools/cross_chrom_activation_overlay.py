#!/usr/bin/env python3
"""
cross_chrom_activation_overlay.py

Cross-chromosome activation-distribution overlay. For one normalization mode
(raw, prenorm, or postnorm), stacks all chromosomes' max-pooled activation
histograms on a single figure to spot outlier chromosomes.

Also emits:
  - per-chrom summary of (mean, std, skew, kurtosis, frac_nonzero) stacked
    into a single TSV so cross-chrom comparisons are one grep away.
  - a boxplot of mean activation and std activation per chromosome.

Outputs:
  <output_dir>/
    cross_chrom_histograms_<mode>.png     overlayed histograms (log-y)
    cross_chrom_mean_std_box_<mode>.png   side-by-side box: mean + std per chrom
    cross_chrom_stats_<mode>.tsv          chrom × {mean, std, skew, kurtosis, frac_nonzero}
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

SUBDIR = {
    "raw": "latent_analysis",
    "prenorm": "latent_analysis_prenorm",
    "postnorm": "latent_analysis_postnorm",
}

DEFAULT_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def load_vectors(results_dir, chrom, mode, sample_n=1_000_000):
    path = os.path.join(results_dir, chrom, "sae", SUBDIR[mode],
                        "data", "maxpooled_vectors.npy")
    if not os.path.isfile(path):
        return None
    v = np.load(path, mmap_mode="r")
    flat = v.ravel()
    if flat.size > sample_n:
        rng = np.random.default_rng(abs(hash(chrom)) % (2**32))
        idx = rng.choice(flat.size, sample_n, replace=False)
        flat = np.asarray(flat[idx])
    else:
        flat = np.asarray(flat)
    return flat, v


def summarize(v):
    # Per-feature stats on full matrix (not the flattened sample)
    means = v.mean(axis=0)
    stds = v.std(axis=0)
    return {
        "mean": float(means.mean()),
        "std": float(stds.mean()),
        "skew": float(np.nanmean(scistats.skew(v, axis=0))),
        "kurtosis": float(np.nanmean(scistats.kurtosis(v, axis=0))),
        "frac_nonzero": float((v != 0).mean()),
    }


def plot_histograms_overlay(samples, out_path, mode):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(samples))]
    for (chrom, flat), color in zip(samples.items(), colors):
        ax.hist(flat, bins=120, histtype="step", density=True, color=color,
                alpha=0.8, label=chrom, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("activation value")
    ax.set_ylabel("density (log)")
    ax.set_title(f"Cross-chromosome SAE activation distributions ({mode})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_mean_std_box(summary_df, out_path, mode):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    chroms = summary_df["chrom"].tolist()
    x = np.arange(len(chroms))
    ax1.bar(x, summary_df["mean"], color="tab:blue")
    ax1.set_xticks(x); ax1.set_xticklabels(chroms, rotation=45, fontsize=8)
    ax1.set_ylabel("mean activation (per-feature avg)")
    ax1.set_title(f"Per-chromosome mean activation ({mode})")
    ax2.bar(x, summary_df["std"], color="tab:orange")
    ax2.set_xticks(x); ax2.set_xticklabels(chroms, rotation=45, fontsize=8)
    ax2.set_ylabel("std of activation")
    ax2.set_title(f"Per-chromosome std activation ({mode})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--mode", choices=list(SUBDIR.keys()), default="prenorm")
    p.add_argument("--chroms", nargs="+", default=DEFAULT_CHROMS)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sample_per_chrom", type=int, default=1_000_000)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.output_dir is None:
        out_dir = build_run_dir(args.results_dir, "_genome_wide",
                                "cross_chrom_distribution", args.mode)
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    samples = {}
    summaries = []
    for c in args.chroms:
        result = load_vectors(args.results_dir, c, args.mode, args.sample_per_chrom)
        if result is None:
            logger.warning(f"{c}: no {args.mode} vectors — skipping")
            continue
        flat, full = result
        samples[c] = flat
        s = summarize(full)
        s["chrom"] = c
        s["n_regions"] = int(full.shape[0])
        summaries.append(s)
        logger.info(f"{c}: regions={full.shape[0]} mean={s['mean']:.4f} "
                    f"std={s['std']:.4f} frac_nonzero={s['frac_nonzero']:.3f}")

    if not samples:
        logger.error("No data loaded")
        return 2

    summary_df = pd.DataFrame(summaries)[
        ["chrom", "n_regions", "mean", "std", "skew", "kurtosis", "frac_nonzero"]
    ]
    summary_df.to_csv(os.path.join(out_dir, f"cross_chrom_stats_{args.mode}.tsv"),
                      sep="\t", index=False)

    plot_histograms_overlay(samples,
        os.path.join(out_dir, f"cross_chrom_histograms_{args.mode}.png"),
        args.mode)
    plot_mean_std_box(summary_df,
        os.path.join(out_dir, f"cross_chrom_mean_std_box_{args.mode}.png"),
        args.mode)

    write_completed(out_dir, "cross_chrom_activation_overlay.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
