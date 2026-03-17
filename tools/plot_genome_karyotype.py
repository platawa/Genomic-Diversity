#!/usr/bin/env python3
"""
plot_genome_karyotype.py

Karyotype-style genome visualization: all 24 human chromosomes as horizontal
bars with entropy (or drop density) displayed as a heatmap track.

Usage:
    python tools/plot_genome_karyotype.py --results_dir results/ --all_human
    python tools/plot_genome_karyotype.py --results_dir results/ --chroms chr1 chr22 chrX
    python tools/plot_genome_karyotype.py --results_dir results/ --all_human --bin_size 50000
    python tools/plot_genome_karyotype.py --results_dir results/ --all_human --gtf /path/to/genomic.gtf
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from results_utils import (
    build_run_dir, write_completed, write_source,
    find_latest_completed, find_all_completed,
)

logger = logging.getLogger(__name__)

ALL_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
    "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
    "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
    "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]

CHROM_SIZES_GRCH38 = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(level)


def load_entropy(run_dir):
    npz_path = os.path.join(run_dir, "data", "entropy.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No entropy.npz in {run_dir}/data")
    npz = np.load(npz_path, allow_pickle=True)
    entropy = npz["entropy"]
    start = int(npz["start"])
    end = int(npz["end"])
    chrom = str(npz["chrom"])
    return entropy, start, end, chrom


def bin_entropy(entropy, start, end, chrom_size, bin_size):
    n_bins = int(np.ceil(chrom_size / bin_size))
    bin_sums = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    positions = np.arange(start, start + len(entropy))
    bin_indices = positions // bin_size

    valid = (bin_indices >= 0) & (bin_indices < n_bins) & np.isfinite(entropy)
    bi = bin_indices[valid]
    ev = entropy[valid]

    np.add.at(bin_sums, bi, ev)
    np.add.at(bin_counts, bi, 1)

    binned = np.full(n_bins, np.nan)
    mask = bin_counts > 0
    binned[mask] = bin_sums[mask] / bin_counts[mask]
    return binned


def load_drop_boundaries(run_dir):
    tsv_path = os.path.join(run_dir, "data", "drop_boundaries.tsv")
    if not os.path.exists(tsv_path):
        return []

    drops = []
    with open(tsv_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("chrom\t"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue
            drops.append((int(fields[3]), int(fields[4])))
    return drops


def bin_drop_density(drops, chrom_size, bin_size):
    n_bins = int(np.ceil(chrom_size / bin_size))
    counts = np.zeros(n_bins, dtype=np.int64)
    for gstart, gend in drops:
        b0 = max(0, gstart // bin_size)
        b1 = min(n_bins - 1, gend // bin_size)
        counts[b0:b1 + 1] += 1
    return counts


def parse_centromeres_from_gtf(gtf_path):
    centromeres = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            if "centromere" in fields[2].lower() or "centromere" in fields[8].lower():
                centromeres[fields[0]] = (int(fields[3]), int(fields[4]))
    return centromeres


def plot_entropy_karyotype(chrom_data, chrom_order, bin_size, output_path,
                           centromeres=None, title="Genome-wide Entropy Profile",
                           vmin=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    chroms_to_plot = [c for c in chrom_order if c in chrom_data]
    if not chroms_to_plot:
        return

    n_chroms = len(chroms_to_plot)
    max_size = max(CHROM_SIZES_GRCH38.get(c, 0) for c in chroms_to_plot)

    if vmin is None or vmax is None:
        all_vals = np.concatenate([chrom_data[c] for c in chroms_to_plot])
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            if vmin is None:
                vmin = float(np.percentile(finite, 1))
            if vmax is None:
                vmax = float(np.percentile(finite, 99))

    fig_height = max(4.0, 0.32 * n_chroms + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bar_height = 0.7
    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad(color="#d0d0d0")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i, chrom in enumerate(chroms_to_plot):
        y = n_chroms - 1 - i
        binned = chrom_data[chrom]
        n_bins_chrom = len(binned)

        row = binned.reshape(1, -1)
        extent = [0, n_bins_chrom * bin_size / 1e6, y - bar_height / 2, y + bar_height / 2]
        ax.imshow(row, aspect="auto", cmap=cmap, norm=norm,
                  extent=extent, interpolation="none", origin="lower")

        chrom_len_mb = CHROM_SIZES_GRCH38.get(chrom, n_bins_chrom * bin_size) / 1e6
        rect = Rectangle((0, y - bar_height / 2), chrom_len_mb, bar_height,
                          linewidth=0.5, edgecolor="black", facecolor="none", zorder=3)
        ax.add_patch(rect)

        if centromeres and chrom in centromeres:
            cen_start, cen_end = centromeres[chrom]
            ax.plot((cen_start + cen_end) / 2e6, y, marker="o",
                    color="black", markersize=3, zorder=4)

    ax.set_yticks(range(n_chroms))
    ax.set_yticklabels(list(reversed(chroms_to_plot)), fontsize=9)
    ax.set_xlabel("Genomic Position (Mb)", fontsize=11)
    ax.set_xlim(0, max_size / 1e6 * 1.02)
    ax.set_ylim(-0.5, n_chroms - 0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Mean Entropy (nats)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if bin_size >= 1_000_000:
        bin_label = f"{bin_size / 1e6:.1f} Mb bins"
    elif bin_size >= 1000:
        bin_label = f"{bin_size / 1e3:.0f} kb bins"
    else:
        bin_label = f"{bin_size} bp bins"
    ax.annotate(bin_label, xy=(0.99, 0.01), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved entropy karyotype: {output_path}")


def plot_drop_density_karyotype(chrom_density, chrom_order, bin_size, output_path,
                                centromeres=None, title="Genome-wide Drop Density"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    chroms_to_plot = [c for c in chrom_order if c in chrom_density]
    if not chroms_to_plot:
        return

    n_chroms = len(chroms_to_plot)
    max_size = max(CHROM_SIZES_GRCH38.get(c, 0) for c in chroms_to_plot)

    all_counts = np.concatenate([chrom_density[c] for c in chroms_to_plot])
    max_count = max(int(np.max(all_counts)), 1)

    fig_height = max(4.0, 0.32 * n_chroms + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bar_height = 0.7
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(color="white")
    norm = mcolors.Normalize(vmin=0, vmax=max_count)

    for i, chrom in enumerate(chroms_to_plot):
        y = n_chroms - 1 - i
        counts = chrom_density[chrom].astype(np.float64)
        n_bins_chrom = len(counts)

        row = counts.reshape(1, -1)
        extent = [0, n_bins_chrom * bin_size / 1e6, y - bar_height / 2, y + bar_height / 2]
        ax.imshow(row, aspect="auto", cmap=cmap, norm=norm,
                  extent=extent, interpolation="none", origin="lower")

        chrom_len_mb = CHROM_SIZES_GRCH38.get(chrom, n_bins_chrom * bin_size) / 1e6
        rect = Rectangle((0, y - bar_height / 2), chrom_len_mb, bar_height,
                          linewidth=0.5, edgecolor="black", facecolor="none", zorder=3)
        ax.add_patch(rect)

        if centromeres and chrom in centromeres:
            cen_start, cen_end = centromeres[chrom]
            ax.plot((cen_start + cen_end) / 2e6, y, marker="o",
                    color="black", markersize=3, zorder=4)

    ax.set_yticks(range(n_chroms))
    ax.set_yticklabels(list(reversed(chroms_to_plot)), fontsize=9)
    ax.set_xlabel("Genomic Position (Mb)", fontsize=11)
    ax.set_xlim(0, max_size / 1e6 * 1.02)
    ax.set_ylim(-0.5, n_chroms - 0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Drops per Bin", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if bin_size >= 1_000_000:
        bin_label = f"{bin_size / 1e6:.1f} Mb bins"
    elif bin_size >= 1000:
        bin_label = f"{bin_size / 1e3:.0f} kb bins"
    else:
        bin_label = f"{bin_size} bp bins"
    ax.annotate(bin_label, xy=(0.99, 0.01), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved drop density karyotype: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Karyotype-style genome-wide entropy / drop-density visualization.",
    )
    parser.add_argument("--results_dir", default="results/",
                        help="Root results directory (default: results/).")
    parser.add_argument("--chroms", nargs="+", default=None,
                        help="Chromosome names to include.")
    parser.add_argument("--all_human", action="store_true",
                        help="Include all 24 human chromosomes.")
    parser.add_argument("--output", default=None,
                        help="Output directory or file path.")
    parser.add_argument("--bin_size", type=int, default=100_000,
                        help="Genomic bin size in bp (default: 100000 = 100 kb).")
    parser.add_argument("--gtf", default=None,
                        help="Optional GTF file for centromere annotation.")

    args = parser.parse_args()
    setup_logging()
    t_start = time.time()

    if args.all_human:
        chrom_order = list(ALL_HUMAN_CHROMS)
    elif args.chroms:
        chrom_order = args.chroms
    else:
        parser.error("Specify --chroms or --all_human.")

    completed = find_all_completed(args.results_dir, chrom_order, "scoring")
    if not completed:
        logger.error("No completed scoring runs found.")
        sys.exit(1)

    logger.info(f"Found {len(completed)}/{len(chrom_order)} completed scoring runs")

    chrom_entropy_binned = {}
    chrom_drop_density = {}
    source_runs = {}

    for chrom in chrom_order:
        if chrom not in completed:
            continue
        run_dir = completed[chrom]
        source_runs[chrom] = run_dir

        try:
            entropy, start, end, _ = load_entropy(run_dir)
            chrom_size = CHROM_SIZES_GRCH38.get(chrom, end)
            binned = bin_entropy(entropy, start, end, chrom_size, args.bin_size)
            chrom_entropy_binned[chrom] = binned
            logger.info(f"  {chrom}: {len(entropy):,} positions -> {len(binned):,} bins")
        except Exception as e:
            logger.warning(f"  {chrom}: failed to load entropy: {e}")
            continue

        try:
            drops = load_drop_boundaries(run_dir)
            density = bin_drop_density(drops, chrom_size, args.bin_size)
            chrom_drop_density[chrom] = density
        except Exception as e:
            logger.warning(f"  {chrom}: failed to load drops: {e}")

    if not chrom_entropy_binned:
        logger.error("No entropy data could be loaded.")
        sys.exit(1)

    centromeres = None
    if args.gtf:
        try:
            centromeres = parse_centromeres_from_gtf(args.gtf)
            logger.info(f"Loaded centromere positions for {len(centromeres)} sequences")
        except Exception as e:
            logger.warning(f"Could not parse centromeres from GTF: {e}")

    # Output directory
    if args.output and os.path.splitext(args.output)[1] in (".png", ".pdf", ".svg"):
        output_dir = os.path.dirname(args.output) or "."
        entropy_path = args.output
        base, ext = os.path.splitext(args.output)
        density_path = f"{base}_drop_density{ext}"
        run_dir_out = output_dir
    else:
        flags = f"bin{args.bin_size // 1000}kb_{len(chrom_entropy_binned)}chroms"
        run_dir_out = build_run_dir(args.results_dir, "_genome_wide", "karyotype", flags)
        output_dir = run_dir_out
        entropy_path = os.path.join(output_dir, "entropy_karyotype.png")
        density_path = os.path.join(output_dir, "drop_density_karyotype.png")

    os.makedirs(output_dir, exist_ok=True)

    plot_entropy_karyotype(
        chrom_entropy_binned, chrom_order, args.bin_size, entropy_path,
        centromeres=centromeres,
    )

    if chrom_drop_density:
        plot_drop_density_karyotype(
            chrom_drop_density, chrom_order, args.bin_size, density_path,
            centromeres=centromeres,
        )

    write_source(run_dir_out, **{c: d for c, d in source_runs.items()})
    wall_time = time.time() - t_start
    write_completed(run_dir_out, "plot_genome_karyotype.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {output_dir}")


if __name__ == "__main__":
    main()
