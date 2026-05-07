#!/usr/bin/env python3
"""
region_distribution_investigator.py

Answer the question: "Are these activations at a given region significant, or
just basal noise?" — by overlaying the region's per-feature activation values
on the global distribution from the same (entity, mode) pair.

Inputs:
  --entity       e.g. chr22, NC_000913.3
  --mode         raw | prenorm | postnorm
  --region_idx   the 0-based index into maxpooled_vectors for the region
                 (or use --chrom_pos to find it by genomic coordinates)
  --features     comma-separated feature indices, OR 'auto' to pick the top-K
                 features by the region's own activation value (default)
  --top_k        when --features=auto, how many features to inspect (default 12)
  --threshold    percentile flag threshold, default 99

Outputs (under a fresh timestamped subdirectory):
  percentiles.tsv       per-feature rank within the global distribution
  region_hist.png       grid of per-feature histograms of the global distribution,
                        each with a red line at the region's activation + percentile

The tool writes INTO a fresh subdirectory of the form:
  {stats_run_dir}/region_investigations/{YYYYMMDD_HHMMSS}_{entity}_{region_label}/

so no existing plot is overwritten.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import write_completed, write_source

logger = logging.getLogger(__name__)

SUBDIR = {
    "raw": "latent_analysis",
    "prenorm": "latent_analysis_prenorm",
    "postnorm": "latent_analysis_postnorm",
}


def vector_path(results_dir, entity, mode):
    return os.path.join(results_dir, entity, "sae", SUBDIR[mode], "data",
                        "maxpooled_vectors.npy")


def _resolve_region_idx(results_dir, entity, mode, region_idx, chrom_pos):
    if region_idx is not None:
        return int(region_idx), f"idx{region_idx}"
    if chrom_pos is None:
        raise ValueError("Provide --region_idx OR --chrom_pos chrom:start-end")
    # Look for per-region coordinate file alongside the vectors
    sae_dir = os.path.join(results_dir, entity, "sae", SUBDIR[mode])
    candidates = [
        os.path.join(sae_dir, "data", "region_coords.tsv"),
        os.path.join(sae_dir, "data", "regions.tsv"),
        os.path.join(results_dir, entity, "scoring"),  # fallback: scoring run
    ]
    chrom, span = chrom_pos.split(":")
    start, end = (int(x) for x in span.split("-"))
    for c in candidates:
        if os.path.isfile(c):
            df = pd.read_csv(c, sep="\t")
            hit = df[(df.get("chrom", chrom) == chrom) &
                     (df["start"] <= end) & (df["end"] >= start)]
            if not hit.empty:
                i = int(hit.index[0])
                return i, f"{chrom}_{start}_{end}"
    raise FileNotFoundError(
        "Could not resolve region by coordinates. Provide --region_idx explicitly "
        "or ensure region_coords.tsv exists alongside maxpooled_vectors.")


def _pick_features(region_vec, features_arg, top_k):
    if features_arg is None or features_arg == "auto":
        # Top-K by region activation magnitude
        mag = np.abs(region_vec)
        idx = np.argsort(-mag)[:top_k]
        return np.asarray(idx, dtype=np.int64)
    return np.asarray([int(s) for s in features_arg.split(",")], dtype=np.int64)


def _load_global_columns(vectors_mmap, feature_idx, max_rows=500_000, seed=0):
    """Load full columns for the selected features. Subsamples rows if too many."""
    n = vectors_mmap.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        rows = rng.choice(n, size=max_rows, replace=False)
        rows.sort()
        return np.asarray(vectors_mmap[rows][:, feature_idx]), n, max_rows
    return np.asarray(vectors_mmap[:, feature_idx]), n, n


def _percentile_rank(global_col, value):
    """Percentile rank (0-100) of value inside global_col."""
    g = global_col[np.isfinite(global_col)]
    if g.size == 0:
        return np.nan
    return float((g <= value).mean() * 100.0)


def plot_region_vs_global(region_vec, feature_idx, global_cols, percentiles,
                          entity, mode, region_label, out_path, threshold):
    n = len(feature_idx)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.0 * rows), squeeze=False)
    for k, fi in enumerate(feature_idx):
        ax = axes[k // cols, k % cols]
        g = global_cols[:, k]
        g_fin = g[np.isfinite(g)]
        if g_fin.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"feat {fi}")
            continue
        ax.hist(g_fin, bins=80, color="steelblue", alpha=0.75, log=True)
        rv = float(region_vec[fi])
        pct = percentiles[k]
        color = "crimson" if (np.isfinite(pct) and pct >= threshold) else "darkorange"
        ax.axvline(rv, color=color, linewidth=2.0)
        ax.set_title(f"feat {fi}  v={rv:.3g}  p={pct:.2f}%",
                     fontsize=9,
                     color=("crimson" if color == "crimson" else "black"))
        ax.tick_params(labelsize=7)
    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_visible(False)
    fig.suptitle(f"{entity} / {mode} — region {region_label}  "
                 f"(red = ≥{threshold:.1f}th pctile vs global)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--entity", required=True)
    p.add_argument("--mode", choices=list(SUBDIR.keys()), required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--region_idx", type=int, default=None)
    g.add_argument("--chrom_pos", default=None,
                   help="chrom:start-end (requires region_coords.tsv alongside vectors)")
    p.add_argument("--features", default="auto",
                   help="Comma-separated feature indices, or 'auto' (top --top_k).")
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--threshold", type=float, default=99.0,
                   help="Percentile threshold for significance flag.")
    p.add_argument("--max_global_rows", type=int, default=500_000,
                   help="Subsample cap for the global column (per feature).")
    p.add_argument("--parent_dir", default=None,
                   help="Parent dir for the investigation output. Default: "
                        "results/_genome_wide/feature_distribution/region_investigations/")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    vec_path = vector_path(args.results_dir, args.entity, args.mode)
    if not os.path.isfile(vec_path):
        logger.error(f"Vectors not found: {vec_path}")
        return 1

    region_idx, region_label = _resolve_region_idx(
        args.results_dir, args.entity, args.mode, args.region_idx, args.chrom_pos)

    # Load region's feature vector (1 row) + set up mmap for global
    vectors = np.load(vec_path, mmap_mode="r")
    if region_idx < 0 or region_idx >= vectors.shape[0]:
        logger.error(f"region_idx {region_idx} out of range [0, {vectors.shape[0]})")
        return 2
    region_vec = np.asarray(vectors[region_idx]).astype(np.float32)

    feature_idx = _pick_features(region_vec, args.features, args.top_k)
    logger.info(f"Investigating region {region_label} ({args.entity} x {args.mode}), "
                f"features {feature_idx.tolist()}")

    global_cols, n_total, n_sampled = _load_global_columns(
        vectors, feature_idx, max_rows=args.max_global_rows)
    logger.info(f"Global columns: n_total={n_total}, n_sampled={n_sampled}")

    percentiles = np.array([
        _percentile_rank(global_cols[:, k], float(region_vec[feature_idx[k]]))
        for k in range(len(feature_idx))
    ], dtype=np.float64)

    # Build fresh timestamped output dir so nothing is overwritten
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_parent = (args.parent_dir or
                   os.path.join(args.results_dir, "_genome_wide",
                                "feature_distribution", "region_investigations"))
    out_dir = os.path.join(base_parent, f"{ts}_{args.entity}_{region_label}")
    os.makedirs(out_dir, exist_ok=False)
    logger.info(f"Output dir: {out_dir}")

    # Write percentiles TSV
    tsv_rows = []
    for k, fi in enumerate(feature_idx):
        rv = float(region_vec[int(fi)])
        pct = float(percentiles[k])
        g = global_cols[:, k]
        g_fin = g[np.isfinite(g)]
        tsv_rows.append({
            "feature_idx": int(fi),
            "region_value": rv,
            "percentile": pct,
            "is_significant": bool(np.isfinite(pct) and pct >= args.threshold),
            "global_q50": float(np.nanmedian(g_fin)) if g_fin.size else np.nan,
            "global_q95": float(np.nanquantile(g_fin, 0.95)) if g_fin.size else np.nan,
            "global_q99": float(np.nanquantile(g_fin, 0.99)) if g_fin.size else np.nan,
            "global_q999": float(np.nanquantile(g_fin, 0.999)) if g_fin.size else np.nan,
            "global_max": float(np.nanmax(g_fin)) if g_fin.size else np.nan,
        })
    tsv_df = pd.DataFrame(tsv_rows)
    tsv_path = os.path.join(out_dir, "percentiles.tsv")
    tsv_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6g")
    logger.info(f"Wrote {tsv_path}")

    # Plot
    png_path = os.path.join(out_dir, "region_hist.png")
    plot_region_vs_global(region_vec, feature_idx, global_cols, percentiles,
                          args.entity, args.mode, region_label, png_path,
                          args.threshold)

    # Record inputs for traceability
    write_source(out_dir, vectors=vec_path)

    t0 = time.time()
    write_completed(out_dir, "region_distribution_investigator.py", time.time() - t0)
    logger.info(f"Done. Output: {out_dir}")

    # Brief console summary
    sig = int(tsv_df["is_significant"].sum())
    logger.info(f"{sig}/{len(tsv_df)} features flagged significant "
                f"(>= {args.threshold:.1f}th pctile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
