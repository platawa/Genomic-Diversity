#!/usr/bin/env python3
"""firing_percent_plots.py

Plot genome-wide (or per-chromosome) UMAP / tSNE points colored by the
**percentage of SAE features that fire at each position**, with optional
activation thresholds tau > 0 and discrete firing-percentage buckets
(<1%, 1-2%, 2-5%, 5-10%, >=10%).

This is distinct from tools/latent_plot_utils.py::compute_firing_threshold_counts,
which uses "commonness" thresholds (a feature is common if it fires in >= T% of
regions). Here we measure the dual: at each position, what fraction of features
is active.

Inputs (from an existing genome-wide tSNE run directory, or a per-chrom
latent_analysis dir):
  <input_dir>/data/combined_maxpooled.npy   (or maxpooled_vectors.npy)
  <input_dir>/data/embedding_tsne.npy
  <input_dir>/data/embedding_umap.npy
  <input_dir>/data/cluster_assignments_array.npy  (or derive from cluster_assignments.tsv)

Outputs under <input_dir>/<output_subdir>/:
  data/
    feature_firing_stats.npz
    firing_sweep.tsv
    per_point_pct_fired.npz
    summary.json
  plots/
    {emb}_pct_fired_tau{tau}_continuous.png
    {emb}_pct_fired_tau{tau}_buckets.png
    {emb}_leiden_vs_buckets_tau{tau}.png
    sweep_overview.png
  COMPLETED
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_utils import write_completed, write_source  # noqa: E402
from tools.latent_plot_utils import plot_continuous_scatter  # noqa: E402

logger = logging.getLogger(__name__)

BUCKET_EDGES = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 100.0001])  # percent
BUCKET_LABELS = ["<1%", "1-2%", "2-5%", "5-10%", ">=10%"]
BUCKET_COLORS = ["#d9d9d9", "#fdae61", "#f46d43", "#d73027", "#7f0000"]


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--input_dir", required=True,
                   help="Directory containing data/{combined_maxpooled,maxpooled_vectors}.npy "
                        "and data/embedding_{tsne,umap}.npy")
    p.add_argument("--mode", required=True, choices=["prenorm", "raw", "postnorm", "custom"],
                   help="Label used in plot titles")
    p.add_argument("--output_subdir", default="firing_percent",
                   help="Subdirectory under input_dir to write outputs (default: firing_percent)")
    p.add_argument("--thresholds", default="0,0.01,0.05,0.10,0.25,0.5,1.0,2.0,3.0",
                   help="Comma-separated activation thresholds tau to sweep")
    p.add_argument("--chunk_rows", type=int, default=10000,
                   help="Rows per streaming chunk (default 10000 -> 1.3GB per chunk for D=32768)")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _find_matrix(input_data_dir):
    for fname in ("combined_maxpooled.npy", "maxpooled_vectors.npy"):
        path = os.path.join(input_data_dir, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Expected combined_maxpooled.npy or maxpooled_vectors.npy in {input_data_dir}")


def _load_clusters(input_data_dir):
    arr_path = os.path.join(input_data_dir, "cluster_assignments_array.npy")
    if os.path.isfile(arr_path):
        return np.load(arr_path)
    tsv_path = os.path.join(input_data_dir, "cluster_assignments.tsv")
    if os.path.isfile(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", comment="#")
        for col in ("cluster", "cluster_id", "leiden"):
            if col in df.columns:
                return df[col].values
    return None


def stream_reduce(matrix_path, thresholds, chunk_rows):
    """One streaming pass over the matrix.

    Returns
    -------
    pct_fired_by_tau : dict tau -> np.ndarray(N,) float32 (percent of features > tau per row)
    feat_fire_count  : np.ndarray(D,) int64  (count of rows where feature j > 0)
    feat_fire_sum    : np.ndarray(D,) float64 (sum of x[j] over rows where x[j] > 0)
    feat_fire_sum_sq : np.ndarray(D,) float64 (sum of x[j]^2 over rows where x[j] > 0)
    """
    X = np.load(matrix_path, mmap_mode="r")
    N, D = X.shape
    logger.info(f"Matrix: N={N}, D={D}, dtype={X.dtype}, path={matrix_path}")

    pct_fired_by_tau = {t: np.empty(N, dtype=np.float32) for t in thresholds}
    feat_fire_count = np.zeros(D, dtype=np.int64)
    feat_fire_sum = np.zeros(D, dtype=np.float64)
    feat_fire_sum_sq = np.zeros(D, dtype=np.float64)

    n_chunks = (N + chunk_rows - 1) // chunk_rows
    t0 = time.time()
    for ci in range(n_chunks):
        lo = ci * chunk_rows
        hi = min(N, lo + chunk_rows)
        # Materialize the chunk once as float32
        chunk = np.asarray(X[lo:hi], dtype=np.float32)

        # Per-row firing percentages for each tau
        for t in thresholds:
            # (chunk > t) is a bool; sum along axis=1 gives counts
            if t == 0:
                n = (chunk > 0).sum(axis=1)
            else:
                n = (chunk > t).sum(axis=1)
            pct_fired_by_tau[t][lo:hi] = n.astype(np.float32) * (100.0 / D)

        # Per-feature accumulators based on tau=0 ("nonzero")
        pos_mask = chunk > 0
        feat_fire_count += pos_mask.sum(axis=0)
        # Use np.where to avoid summing zeros across the dense matrix twice
        masked = np.where(pos_mask, chunk, 0.0)
        feat_fire_sum += masked.sum(axis=0)
        feat_fire_sum_sq += (masked * masked).sum(axis=0)

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            dt = time.time() - t0
            rate = (hi) / max(dt, 1e-6)
            eta = (N - hi) / max(rate, 1e-6)
            logger.info(f"  chunk {ci + 1}/{n_chunks}  rows {hi}/{N}  "
                        f"elapsed {dt:.0f}s  eta {eta:.0f}s")

    return pct_fired_by_tau, feat_fire_count, feat_fire_sum, feat_fire_sum_sq


def compute_feature_stats(feat_fire_count, feat_fire_sum, feat_fire_sum_sq, n_total):
    firing_rate = feat_fire_count / max(n_total, 1)  # per-feature
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_when_fired = np.where(
            feat_fire_count > 0, feat_fire_sum / np.maximum(feat_fire_count, 1), 0.0)
        var_when_fired = np.where(
            feat_fire_count > 1,
            (feat_fire_sum_sq - feat_fire_sum * feat_fire_sum / np.maximum(feat_fire_count, 1))
            / np.maximum(feat_fire_count - 1, 1),
            0.0,
        )
    std_when_fired = np.sqrt(np.maximum(var_when_fired, 0.0))
    return firing_rate, mean_when_fired, std_when_fired


def build_sweep_table(pct_fired_by_tau, thresholds):
    rows = []
    for t in thresholds:
        pct = pct_fired_by_tau[t]
        rows.append({
            "threshold": t,
            "mean_pct_firing": float(pct.mean()),
            "median_pct_firing": float(np.median(pct)),
            "pct_points_ge1pct": float((pct >= 1.0).mean() * 100),
            "pct_points_ge2pct": float((pct >= 2.0).mean() * 100),
            "pct_points_ge5pct": float((pct >= 5.0).mean() * 100),
            "pct_points_ge10pct": float((pct >= 10.0).mean() * 100),
        })
    return pd.DataFrame(rows)


def pick_recommended_tau(sweep_df):
    """Pick the tau whose >=1% point-fraction is nearest to 50% while keeping
    >=10% point-fraction below 95% (not saturated) and above 0.1% (not dead)."""
    cand = sweep_df[(sweep_df.pct_points_ge10pct < 95.0)
                    & (sweep_df.pct_points_ge10pct > 0.1)]
    if cand.empty:
        cand = sweep_df
    # minimize |pct_points_ge1pct - 50|
    idx = (cand.pct_points_ge1pct - 50.0).abs().idxmin()
    return float(sweep_df.loc[idx, "threshold"])


def bucketize(pct):
    """Return integer bucket index per point: 0..4."""
    idx = np.digitize(pct, BUCKET_EDGES[1:-1], right=False)
    return np.clip(idx, 0, len(BUCKET_LABELS) - 1).astype(np.int32)


def plot_buckets(coords, bucket_idx, title, out_path, emb_name="tsne"):
    cmap = ListedColormap(BUCKET_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(BUCKET_LABELS) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=bucket_idx, cmap=cmap, norm=norm,
               s=1, alpha=0.3, linewidths=0, rasterized=True)
    xlab = f"{emb_name.upper()}-1"
    ylab = f"{emb_name.upper()}-2"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(BUCKET_COLORS, BUCKET_LABELS)]
    ax.legend(handles=handles, loc="upper right", title="% features firing",
              frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_leiden_vs_buckets(coords, clusters, bucket_idx, title, out_path, emb_name="tsne"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # Left: Leiden clusters (tab20 cycled)
    if clusters is not None:
        uniq = np.unique(clusters)
        cmap_left = plt.get_cmap("tab20", max(len(uniq), 1))
        color_map = {c: cmap_left(i % cmap_left.N) for i, c in enumerate(uniq)}
        colors_left = np.array([color_map[c] for c in clusters])
        axes[0].scatter(coords[:, 0], coords[:, 1], c=colors_left,
                        s=1, alpha=0.3, linewidths=0, rasterized=True)
        axes[0].set_title(f"Leiden clusters ({len(uniq)})")
    else:
        axes[0].text(0.5, 0.5, "No cluster labels", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].set_title("Leiden clusters (unavailable)")
    axes[0].set_xlabel(f"{emb_name.upper()}-1")
    axes[0].set_ylabel(f"{emb_name.upper()}-2")

    # Right: firing buckets
    cmap_right = ListedColormap(BUCKET_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(BUCKET_LABELS) + 0.5), cmap_right.N)
    axes[1].scatter(coords[:, 0], coords[:, 1], c=bucket_idx, cmap=cmap_right,
                    norm=norm, s=1, alpha=0.3, linewidths=0, rasterized=True)
    axes[1].set_xlabel(f"{emb_name.upper()}-1")
    axes[1].set_ylabel(f"{emb_name.upper()}-2")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(BUCKET_COLORS, BUCKET_LABELS)]
    axes[1].legend(handles=handles, loc="upper right",
                   title="% features firing", frameon=False)
    axes[1].set_title("Firing buckets")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_overview(sweep_df, recommended_tau, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, lbl in [("pct_points_ge1pct", ">=1% firing"),
                     ("pct_points_ge2pct", ">=2% firing"),
                     ("pct_points_ge5pct", ">=5% firing"),
                     ("pct_points_ge10pct", ">=10% firing")]:
        ax.plot(sweep_df.threshold, sweep_df[col], marker="o", label=lbl)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Activation threshold tau")
    ax.set_ylabel("% of points in bucket")
    ax.axvline(recommended_tau, color="black", linestyle="--",
               label=f"recommended tau={recommended_tau:g}")
    ax.set_title("Firing-bucket coverage vs activation threshold")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_logging(args.log_level)

    thresholds = [float(x) for x in args.thresholds.split(",")]

    input_dir = os.path.abspath(args.input_dir)
    data_in = os.path.join(input_dir, "data")
    if not os.path.isdir(data_in):
        logger.error(f"Expected {data_in} to exist")
        sys.exit(2)

    matrix_path = _find_matrix(data_in)
    tsne_path = os.path.join(data_in, "embedding_tsne.npy")
    umap_path = os.path.join(data_in, "embedding_umap.npy")
    tsne = np.load(tsne_path) if os.path.isfile(tsne_path) else None
    umap = np.load(umap_path) if os.path.isfile(umap_path) else None
    if tsne is None and umap is None:
        logger.error(f"No embedding_tsne.npy or embedding_umap.npy in {data_in}")
        sys.exit(2)

    clusters = _load_clusters(data_in)

    out_dir = os.path.join(input_dir, args.output_subdir)
    data_out = os.path.join(out_dir, "data")
    plots_out = os.path.join(out_dir, "plots")
    os.makedirs(data_out, exist_ok=True)
    os.makedirs(plots_out, exist_ok=True)

    write_source(out_dir,
                 matrix=matrix_path,
                 embedding_tsne=tsne_path if tsne is not None else None,
                 embedding_umap=umap_path if umap is not None else None)

    t_start = time.time()
    pct_fired_by_tau, feat_count, feat_sum, feat_sum_sq = stream_reduce(
        matrix_path, thresholds, args.chunk_rows)

    # Per-feature stats
    X_shape = np.load(matrix_path, mmap_mode="r").shape
    N, D = X_shape
    firing_rate, mean_when_fired, std_when_fired = compute_feature_stats(
        feat_count, feat_sum, feat_sum_sq, N)

    avg_firing_rate = float(firing_rate.mean())
    above_avg_mask = firing_rate > avg_firing_rate
    above_avg_indices = np.where(above_avg_mask)[0]
    if above_avg_indices.size > 0:
        avg_value_for_above_avg = float(mean_when_fired[above_avg_mask].mean())
        med_value_for_above_avg = float(np.median(mean_when_fired[above_avg_mask]))
    else:
        avg_value_for_above_avg = float("nan")
        med_value_for_above_avg = float("nan")

    # Sweep table
    sweep_df = build_sweep_table(pct_fired_by_tau, thresholds)
    sweep_df.to_csv(os.path.join(data_out, "firing_sweep.tsv"),
                    sep="\t", index=False, float_format="%.6g")
    logger.info(f"\n{sweep_df.to_string(index=False)}")

    recommended_tau = pick_recommended_tau(sweep_df)
    logger.info(f"Recommended tau: {recommended_tau}")

    # Save per-feature stats
    np.savez(os.path.join(data_out, "feature_firing_stats.npz"),
             firing_rate=firing_rate.astype(np.float32),
             mean_when_fired=mean_when_fired.astype(np.float32),
             std_when_fired=std_when_fired.astype(np.float32),
             avg_firing_rate=np.float32(avg_firing_rate),
             above_avg_indices=above_avg_indices.astype(np.int32))

    # Save per-point percent-fired arrays
    np.savez(os.path.join(data_out, "per_point_pct_fired.npz"),
             **{f"pct_fired_tau_{t:g}": pct_fired_by_tau[t] for t in thresholds})

    # Summary
    summary = {
        "mode": args.mode,
        "input_dir": input_dir,
        "matrix": matrix_path,
        "n_points": int(N),
        "n_features": int(D),
        "avg_firing_rate": avg_firing_rate,
        "median_firing_rate": float(np.median(firing_rate)),
        "fraction_features_above_avg": float(above_avg_mask.mean()),
        "avg_value_for_above_avg_features_when_fired": avg_value_for_above_avg,
        "median_value_for_above_avg_features_when_fired": med_value_for_above_avg,
        "recommended_tau": recommended_tau,
        "thresholds": thresholds,
        "wall_time_s_reduce": time.time() - t_start,
    }
    with open(os.path.join(data_out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

    # Plots
    embeddings = {"tsne": tsne, "umap": umap}
    for t in thresholds:
        pct = pct_fired_by_tau[t]
        buckets = bucketize(pct)
        tau_tag = f"tau{t:g}".replace(".", "p")
        for emb_name, coords in embeddings.items():
            if coords is None:
                continue
            # Continuous
            title_c = (f"{emb_name.upper()} — % SAE features firing (tau={t:g})\n"
                       f"mode={args.mode}, N={N}, D={D}")
            out_c = os.path.join(plots_out, f"{emb_name}_pct_fired_{tau_tag}_continuous.png")
            plot_continuous_scatter(
                coords, pct.astype(float), cmap="inferno",
                colorbar_label="% features firing",
                title=title_c, out_path=out_c, emb_name=emb_name,
                point_size=1, alpha=0.3,
            )
            # Buckets
            title_b = (f"{emb_name.upper()} — firing buckets (tau={t:g})\n"
                       f"mode={args.mode}, N={N}, D={D}")
            out_b = os.path.join(plots_out, f"{emb_name}_pct_fired_{tau_tag}_buckets.png")
            plot_buckets(coords, buckets, title_b, out_b, emb_name=emb_name)
            # Leiden vs buckets
            out_lv = os.path.join(plots_out, f"{emb_name}_leiden_vs_buckets_{tau_tag}.png")
            plot_leiden_vs_buckets(
                coords, clusters, buckets,
                title=f"{emb_name.upper()} — Leiden vs firing buckets "
                      f"(tau={t:g}, mode={args.mode})",
                out_path=out_lv, emb_name=emb_name,
            )

    plot_sweep_overview(sweep_df, recommended_tau,
                        os.path.join(plots_out, "sweep_overview.png"))

    wall = time.time() - t_start
    write_completed(out_dir, "firing_percent_plots.py", wall)
    logger.info(f"Done in {wall:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()
