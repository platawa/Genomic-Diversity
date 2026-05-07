#!/usr/bin/env python3
"""firing_percent_histograms.py

Generates two distribution histograms for an existing firing_percent run:
  1. histogram_per_region_pct_fired.png — multi-panel grid (one per tau) showing
     the distribution of % SAE features firing per region.
  2. histogram_per_feature_firing_rate.png — distribution of per-feature firing
     rates across the SAE feature dimension (linear-x | log-x panels).

Reads the .npz / .json files written by tools/firing_percent_plots.py.
No recomputation; pure plotting.

Inputs:
  <input_dir>/<firing_percent_subdir>/data/per_point_pct_fired.npz
  <input_dir>/<firing_percent_subdir>/data/feature_firing_stats.npz
  <input_dir>/<firing_percent_subdir>/data/summary.json

Outputs:
  <input_dir>/<firing_percent_subdir>/plots/histogram_per_region_pct_fired.png
  <input_dir>/<firing_percent_subdir>/plots/histogram_per_feature_firing_rate.png
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

BUCKET_EDGES = [1.0, 2.0, 5.0, 10.0]
BUCKET_COLORS = ["#fdae61", "#f46d43", "#d73027", "#7f0000"]
BUCKET_LABELS = ["1%", "2%", "5%", "10%"]


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_dir", required=True,
                   help="Parent dir containing the firing_percent subdir")
    p.add_argument("--firing_percent_subdir", default="firing_percent",
                   help='Subdir name (e.g. "firing_percent" or "firing_percent_fast")')
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing histogram PNGs")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def parse_tau_from_key(key):
    prefix = "pct_fired_tau_"
    if not key.startswith(prefix):
        return None
    return float(key[len(prefix):])


def plot_per_region_grid(per_point_npz, summary, out_path):
    keys = [k for k in per_point_npz.files if k.startswith("pct_fired_tau_")]
    keys.sort(key=parse_tau_from_key)
    taus = [parse_tau_from_key(k) for k in keys]
    recommended_tau = float(summary["recommended_tau"]) if summary and "recommended_tau" in summary else None

    n = len(keys)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0),
                             sharey=False)
    axes = np.atleast_1d(axes).flatten()
    n_regions = len(per_point_npz[keys[0]])

    for i, (k, t) in enumerate(zip(keys, taus)):
        ax = axes[i]
        pct = per_point_npz[k]
        ax.hist(pct, bins=80, range=(0.0, 100.0),
                color="steelblue", alpha=0.85)
        ax.set_yscale("log")
        for be, bc, bl in zip(BUCKET_EDGES, BUCKET_COLORS, BUCKET_LABELS):
            ax.axvline(be, color=bc, linestyle="--", linewidth=1, alpha=0.7,
                       label=f">={bl}")
        title = f"tau={t:g}  median={np.median(pct):.2f}%, mean={pct.mean():.2f}%"
        if recommended_tau is not None and abs(t - recommended_tau) < 1e-9:
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2.0)
            title = f"tau={t:g} (recommended)  median={np.median(pct):.2f}%, mean={pct.mean():.2f}%"
            ax.set_title(title, fontsize=10, fontweight="bold")
        else:
            ax.set_title(title, fontsize=10)
        ax.set_xlabel("% features firing per region")
        ax.set_ylabel("# regions (log)")
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Per-region firing-density distribution by tau "
                 f"(N regions = {n_regions:,})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_feature(stats_npz, summary, out_path):
    firing_rate = stats_npz["firing_rate"]
    if "avg_firing_rate" in stats_npz.files:
        avg = float(stats_npz["avg_firing_rate"])
    else:
        avg = float(firing_rate.mean())
    median = float(np.median(firing_rate))
    n_features = int(firing_rate.shape[0])
    n_dead = int((firing_rate == 0).sum())
    n_alive = n_features - n_dead

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(firing_rate, bins=80, color="darkorange", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].axvline(avg, color="black", linestyle="--",
                    label=f"mean = {avg:.4f}")
    axes[0].axvline(median, color="grey", linestyle=":",
                    label=f"median = {median:.4f}")
    axes[0].set_xlabel("Per-feature firing rate (fraction of regions where feature > 0)")
    axes[0].set_ylabel("# features (log)")
    axes[0].set_title("Linear x")
    axes[0].legend(frameon=False)

    nz = firing_rate[firing_rate > 0]
    if nz.size > 0:
        lo = max(float(nz.min()), 1e-7)
        hi = float(nz.max())
        bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 60)
        axes[1].hist(nz, bins=bins, color="darkgreen", alpha=0.85)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].axvline(avg, color="black", linestyle="--",
                        label=f"mean = {avg:.4f}")
        axes[1].set_xlabel("Per-feature firing rate (log x, nonzero only)")
        axes[1].set_ylabel("# features (log)")
        axes[1].set_title("Log x — tail visibility")
        axes[1].legend(frameon=False)
    else:
        axes[1].text(0.5, 0.5, "No features with firing_rate > 0",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Log x — empty")

    n_points = summary.get("n_points") if summary else None
    suptitle = (f"Per-feature firing-rate distribution "
                f"(D={n_features:,}, alive={n_alive:,}, dead={n_dead:,}")
    if n_points:
        suptitle += f", N regions={n_points:,}"
    suptitle += ")"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_logging(args.log_level)

    fp_dir = os.path.join(os.path.abspath(args.input_dir), args.firing_percent_subdir)
    data_dir = os.path.join(fp_dir, "data")
    plots_dir = os.path.join(fp_dir, "plots")

    if not os.path.isdir(data_dir):
        logger.error(f"Missing {data_dir} (run firing_percent_plots.py first)")
        sys.exit(2)
    os.makedirs(plots_dir, exist_ok=True)

    per_point_path = os.path.join(data_dir, "per_point_pct_fired.npz")
    feat_stats_path = os.path.join(data_dir, "feature_firing_stats.npz")
    summary_path = os.path.join(data_dir, "summary.json")

    if not os.path.isfile(per_point_path):
        logger.error(f"Missing {per_point_path}")
        sys.exit(2)
    if not os.path.isfile(feat_stats_path):
        logger.error(f"Missing {feat_stats_path}")
        sys.exit(2)

    summary = None
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    out_region = os.path.join(plots_dir, "histogram_per_region_pct_fired.png")
    out_feature = os.path.join(plots_dir, "histogram_per_feature_firing_rate.png")

    if not args.force and os.path.isfile(out_region) and os.path.isfile(out_feature):
        logger.info(f"SKIP {fp_dir}: histograms already present (use --force to overwrite)")
        return

    logger.info(f"Reading {per_point_path}")
    pp_npz = np.load(per_point_path)
    logger.info(f"Reading {feat_stats_path}")
    fs_npz = np.load(feat_stats_path)

    plot_per_region_grid(pp_npz, summary, out_region)
    logger.info(f"Wrote: {out_region}")
    plot_per_feature(fs_npz, summary, out_feature)
    logger.info(f"Wrote: {out_feature}")


if __name__ == "__main__":
    main()
