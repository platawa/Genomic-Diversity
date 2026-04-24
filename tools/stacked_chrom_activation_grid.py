#!/usr/bin/env python3
"""
stacked_chrom_activation_grid.py

Produces a single 23 x 3 grid figure (rows = chromosomes, cols =
raw/prenorm/postnorm) of the per-chromosome max-pooled SAE activation
distributions. Streams one chromosome x mode at a time, precomputes
histogram counts, then tiles them into one figure — so memory stays
bounded even though the combined vectors total ~500 GB.

X-range within each column is shared (computed as the global min/max
across chromosomes for that mode) so rows are comparable.

Output: stacked_activation_grid.png (+ .tsv with the bin counts).
"""

import argparse
import logging
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import write_completed

logger = logging.getLogger(__name__)

MODES = ["raw", "prenorm", "postnorm"]
SUBDIR = {
    "raw": "latent_analysis",
    "prenorm": "latent_analysis_prenorm",
    "postnorm": "latent_analysis_postnorm",
}


def vector_path(results_dir, chrom, mode):
    return os.path.join(
        results_dir, chrom, "sae", SUBDIR[mode], "data", "maxpooled_vectors.npy"
    )


def mode_range(results_dir, chroms, mode, quantiles=(0.001, 0.999), subsample=500_000):
    """Scan chroms for a robust (low, high) clip range per mode."""
    lows, highs = [], []
    rng = np.random.default_rng(0)
    for c in chroms:
        p = vector_path(results_dir, c, mode)
        if not os.path.isfile(p):
            continue
        v = np.load(p, mmap_mode="r")
        flat = np.asarray(v).ravel()
        if flat.size > subsample:
            flat = rng.choice(flat, subsample, replace=False)
        lo, hi = np.quantile(flat, quantiles)
        lows.append(lo)
        highs.append(hi)
        del v, flat
    if not lows:
        return None
    return float(min(lows)), float(max(highs))


def chrom_histogram(results_dir, chrom, mode, bin_edges, subsample=1_000_000):
    p = vector_path(results_dir, chrom, mode)
    if not os.path.isfile(p):
        return None, None
    v = np.load(p)
    flat = v.ravel()
    if flat.size > subsample:
        rng = np.random.default_rng(abs(hash((chrom, mode))) % (2**32))
        flat = rng.choice(flat, subsample, replace=False)
    counts, _ = np.histogram(flat, bins=bin_edges)
    n_regions = v.shape[0]
    del v, flat
    return counts, n_regions


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--chroms", nargs="+",
                   default=["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
                            "chr8", "chr9", "chr10", "chr11", "chr12", "chr13",
                            "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                            "chr20", "chr21", "chr22", "chrX", "chrY"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_bins", type=int, default=120)
    p.add_argument("--row_height", type=float, default=0.55, help="inches per chrom row")
    p.add_argument("--col_width", type=float, default=5.0, help="inches per mode column")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = __import__("time").time()

    # Pass 1: per-mode x-ranges
    logger.info("Pass 1: scanning ranges...")
    ranges = {m: mode_range(args.results_dir, args.chroms, m) for m in MODES}
    for m, r in ranges.items():
        logger.info(f"  {m}: {r}")

    edges = {}
    for m in MODES:
        if ranges[m] is None:
            edges[m] = None
        else:
            lo, hi = ranges[m]
            edges[m] = np.linspace(lo, hi, args.n_bins + 1)

    # Pass 2: per-(chrom, mode) histogram
    logger.info("Pass 2: computing histograms...")
    data = {}
    region_counts = {}
    for chrom in args.chroms:
        for m in MODES:
            if edges[m] is None:
                continue
            counts, nreg = chrom_histogram(args.results_dir, chrom, m, edges[m])
            data[(chrom, m)] = counts
            if nreg is not None:
                region_counts[(chrom, m)] = nreg
            logger.info(f"  {chrom} {m}: "
                        f"{'n/a' if counts is None else counts.sum()}")

    # Save counts to TSV for reuse
    tsv_path = os.path.join(args.output_dir, "bin_counts.tsv")
    with open(tsv_path, "w") as fh:
        fh.write("chrom\tmode\tbin_lo\tbin_hi\tcount\n")
        for (chrom, m), counts in data.items():
            if counts is None:
                continue
            e = edges[m]
            for i, c in enumerate(counts):
                fh.write(f"{chrom}\t{m}\t{e[i]:.6g}\t{e[i+1]:.6g}\t{int(c)}\n")
    logger.info(f"Wrote {tsv_path}")

    # Plot 23 x 3 grid
    n_rows = len(args.chroms)
    n_cols = len(MODES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(args.col_width * n_cols, args.row_height * n_rows),
        sharex="col", squeeze=False,
    )
    for c, m in enumerate(MODES):
        axes[0, c].set_title(m, fontsize=11)
    for r, chrom in enumerate(args.chroms):
        for c, m in enumerate(MODES):
            ax = axes[r, c]
            counts = data.get((chrom, m))
            if counts is None or edges[m] is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_yticks([])
                ax.set_xticks([])
                if c == 0:
                    ax.set_ylabel(chrom, fontsize=8, rotation=0, ha="right", va="center")
                continue
            widths = np.diff(edges[m])
            ax.bar(edges[m][:-1], counts, width=widths, align="edge",
                   color="steelblue", alpha=0.85, log=True, linewidth=0)
            ax.tick_params(axis="both", labelsize=6)
            ax.set_yticks([])
            if c == 0:
                nreg = region_counts.get((chrom, m), "?")
                ax.set_ylabel(f"{chrom}\n(n={nreg})", fontsize=7,
                              rotation=0, ha="right", va="center")
            if r == n_rows - 1:
                ax.set_xlabel(f"activation ({m})", fontsize=8)
    fig.suptitle("Per-chromosome SAE activation distributions (log count)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "stacked_activation_grid.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_png}")

    write_completed(args.output_dir, "stacked_chrom_activation_grid.py",
                    __import__("time").time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
