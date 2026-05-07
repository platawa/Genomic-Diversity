#!/usr/bin/env python3
"""
genome_feature_distribution_plots.py

Reads stats.npz produced by genome_feature_distribution.py and emits:

  plots/
    heatmap_mean.png               per-feature mean across entities x modes
    heatmap_q99.png                per-feature 99th percentile (log color)
    heatmap_sparsity.png           per-feature fraction nonzero
    violin_per_feature_mean.png    distribution of per-feature means per entity
    violin_per_feature_q99.png     distribution of per-feature 99th pctile per entity
    violin_per_feature_sparsity.png  distribution of per-feature frac_nonzero

The heatmap rows are all 32 768 features sorted by the human_genome x raw mean
(features with no signal cluster at the bottom).
"""

import argparse
import logging
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

logger = logging.getLogger(__name__)


def _load(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    return {
        "stat_names": [s for s in z["stat_names"]],
        "entities": [e for e in z["entities"]],
        "modes": [m for m in z["modes"]],
        "stats": z["stats"],           # (E, M, F, S)
        "n_regions": z["n_regions"],   # (E, M)
    }


def _sort_features(stats, entities, modes, stat_names, reference_entity="human_genome",
                   reference_mode="raw"):
    """Return a permutation of feature indices sorted by the reference entity x mode x mean.

    If the reference cell is missing, fall back to mean across all non-NaN cells.
    """
    mean_idx = stat_names.index("mean")
    try:
        ei = entities.index(reference_entity)
        mi = modes.index(reference_mode)
        ref = stats[ei, mi, :, mean_idx]
    except ValueError:
        ref = None
    if ref is None or np.all(np.isnan(ref)):
        # Fall back: mean across all (entity, mode) cells for the mean stat
        ref = np.nanmean(stats[:, :, :, mean_idx].reshape(-1, stats.shape[2]), axis=0)
    order = np.argsort(ref)[::-1]  # high means on top
    return order


def _column_labels_and_groups(entities, modes):
    """Return flat list of (entity, mode) columns grouped by mode."""
    cols = []
    for m in modes:
        for e in entities:
            cols.append((e, m))
    return cols


def plot_heatmap(stats, entities, modes, n_regions, stat_names, stat, order,
                 out_path, log_color=False, title=None):
    """One heatmap with columns grouped by mode (mode blocks side by side)."""
    si = stat_names.index(stat)
    cols = _column_labels_and_groups(entities, modes)
    n_cols = len(cols)
    f_ = stats.shape[2]

    M = np.full((f_, n_cols), np.nan, dtype=np.float32)
    for k, (e, m) in enumerate(cols):
        ei = entities.index(e)
        mi = modes.index(m)
        if n_regions[ei, mi] == 0:
            continue
        col = stats[ei, mi, :, si]
        M[:, k] = col
    M = M[order]

    fig_h = max(6, min(18, f_ / 2000.0 * 6))
    fig_w = max(10, n_cols * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if log_color:
        # Clip for log scale: lowest positive value -> floor
        pos = M[np.isfinite(M) & (M > 0)]
        vmin = float(np.nanquantile(pos, 0.01)) if pos.size else 1e-6
        vmax = float(np.nanmax(M))
        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 10))
    else:
        vmax = float(np.nanquantile(np.abs(M), 0.99))
        norm = Normalize(vmin=-vmax, vmax=vmax) if stat in ("skew",) else \
               Normalize(vmin=0.0, vmax=vmax)

    cmap = "RdBu_r" if stat in ("skew",) else "viridis"
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{e}\n{m}" for (e, m) in cols], rotation=90, fontsize=6)
    ax.set_ylabel(f"feature (sorted by human_genome x raw mean)  — {f_} features")

    # Vertical separators between mode groups
    boundary = len(entities)
    for k in range(1, len(modes)):
        ax.axvline(k * boundary - 0.5, color="white", linewidth=1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(stat + (" (log)" if log_color else ""))

    ax.set_title(title or f"Per-feature {stat} across entities x modes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def plot_violin(stats, entities, modes, n_regions, stat_names, stat, out_path,
                log_y=False, title=None):
    """One figure per stat with 3 panels (one per mode). Each panel: violin per entity."""
    si = stat_names.index(stat)
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 0.28 * len(entities) + 3),
                             sharey=False)
    if len(modes) == 1:
        axes = [axes]

    for mi, m in enumerate(modes):
        ax = axes[mi]
        data = []
        labels = []
        for ei, e in enumerate(entities):
            if n_regions[ei, mi] == 0:
                continue
            col = stats[ei, mi, :, si]
            col = col[np.isfinite(col)]
            if log_y:
                col = col[col > 0]
            data.append(col)
            labels.append(e)
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(m)
            continue
        parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=True,
                              widths=0.85)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
            pc.set_facecolor("steelblue")
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=7)
        if log_y:
            ax.set_xscale("log")
        ax.set_xlabel(f"per-feature {stat}")
        ax.set_title(m)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(title or f"Distribution of per-feature {stat} across entities")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--stats_npz", required=True,
                   help="Path to stats.npz produced by genome_feature_distribution.py")
    p.add_argument("--output_dir", default=None,
                   help="Default: <dirname(stats.npz)>/plots/")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data = _load(args.stats_npz)
    stats = data["stats"]
    entities = data["entities"]
    modes = data["modes"]
    n_regions = data["n_regions"]
    stat_names = data["stat_names"]
    logger.info(f"Loaded stats.npz: entities={len(entities)}, modes={len(modes)}, "
                f"features={stats.shape[2]}, stats={stats.shape[3]}")

    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.stats_npz)),
                                              "plots")
    os.makedirs(out_dir, exist_ok=True)

    order = _sort_features(stats, entities, modes, stat_names)

    # Heatmaps
    plot_heatmap(stats, entities, modes, n_regions, stat_names, "mean",
                 order, os.path.join(out_dir, "heatmap_mean.png"), log_color=False)
    plot_heatmap(stats, entities, modes, n_regions, stat_names, "q99",
                 order, os.path.join(out_dir, "heatmap_q99.png"), log_color=True)
    plot_heatmap(stats, entities, modes, n_regions, stat_names, "frac_nonzero",
                 order, os.path.join(out_dir, "heatmap_sparsity.png"), log_color=False,
                 title="Per-feature fraction nonzero (sparsity) across entities x modes")

    # Violin distributions of per-feature stats
    plot_violin(stats, entities, modes, n_regions, stat_names, "mean",
                os.path.join(out_dir, "violin_per_feature_mean.png"), log_y=False)
    plot_violin(stats, entities, modes, n_regions, stat_names, "q99",
                os.path.join(out_dir, "violin_per_feature_q99.png"), log_y=True)
    plot_violin(stats, entities, modes, n_regions, stat_names, "frac_nonzero",
                os.path.join(out_dir, "violin_per_feature_sparsity.png"), log_y=False)

    logger.info(f"Done. Plots in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
