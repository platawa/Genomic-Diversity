#!/usr/bin/env python3
"""
chromosome_analysis.py

Six chromosome-level analyses of Evo2 entropy scoring results.

Analyses:
  1. Gene density vs. drop density (correlation per chromosome)
  2. Entropy distribution per chromosome (violin plot)
  3. Drop size distribution per chromosome (box plot)
  4. Mean chromosome entropy vs. gene count (scatter, 1 point per chrom)
  5. Centromere-flanking drop density (drop density vs. distance from centromere)
  6. SAE feature diversity per chromosome (unique features, top features)

Usage:
    python tools/chromosome_analysis.py \
        --results_dir results/ \
        --gtf /path/to/genomic.gtf \
        [--output_dir results/_chromosome_analysis/] \
        [--analyses 1 2 3 4 5 6]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from results_utils import find_latest_completed, build_run_dir, write_completed

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

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
    "chr16": 90338345,  "chr17": 83257441,  "chr18": 80373285,
    "chr19": 58617616,  "chr20": 64444167,  "chr21": 46709983,
    "chr22": 50818468,  "chrX": 156040895,  "chrY": 57227415,
}

# GRCh38 centromere midpoints (bp) — from UCSC cytoBand track
CENTROMERE_MID_GRCH38 = {
    "chr1": 123400000, "chr2": 93900000,  "chr3": 90900000,
    "chr4": 50400000,  "chr5": 48400000,  "chr6": 61000000,
    "chr7": 59900000,  "chr8": 45200000,  "chr9": 43000000,
    "chr10": 39800000, "chr11": 53400000, "chr12": 35500000,
    "chr13": 17700000, "chr14": 17200000, "chr15": 19000000,
    "chr16": 36800000, "chr17": 25100000, "chr18": 18500000,
    "chr19": 26500000, "chr20": 28500000, "chr21": 12000000,
    "chr22": 15000000, "chrX": 61000000,  "chrY": 10400000,
}

# NCBI accession -> chr name (GRCh38)
ACCESSION_TO_CHROM = {
    "NC_000001.11": "chr1",  "NC_000002.12": "chr2",  "NC_000003.12": "chr3",
    "NC_000004.12": "chr4",  "NC_000005.10": "chr5",  "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",  "NC_000008.11": "chr8",  "NC_000009.12": "chr9",
    "NC_000010.11": "chr10", "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9":  "chr14", "NC_000015.10": "chr15",
    "NC_000016.10": "chr16", "NC_000017.11": "chr17", "NC_000018.10": "chr18",
    "NC_000019.10": "chr19", "NC_000020.11": "chr20", "NC_000021.9":  "chr21",
    "NC_000022.11": "chr22", "NC_000023.11": "chrX",  "NC_000024.10": "chrY",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_entropy(run_dir: str) -> Tuple[np.ndarray, int, int]:
    npz = np.load(os.path.join(run_dir, "data", "entropy.npz"), allow_pickle=True)
    return npz["entropy"], int(npz["start"]), int(npz["end"])


def load_drop_boundaries(run_dir: str) -> List[Dict]:
    path = os.path.join(run_dir, "data", "drop_boundaries.tsv")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                continue
            if len(fields) < len(header):
                continue
            rows.append(dict(zip(header, fields)))
    return rows


def parse_gtf_genes(gtf_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Return {chr_name: [(start, end), ...]} for 'gene' features."""
    genes: Dict[str, List] = defaultdict(list)
    logger.info(f"Parsing GTF: {gtf_path}")
    n = 0
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "gene":
                continue
            acc = parts[0]
            chrom = ACCESSION_TO_CHROM.get(acc, acc)
            start, end = int(parts[3]) - 1, int(parts[4])  # convert to 0-based
            genes[chrom].append((start, end))
            n += 1
    logger.info(f"Parsed {n:,} gene records across {len(genes)} sequences")
    return dict(genes)


def bin_gene_density(gene_intervals: List[Tuple[int, int]],
                     chrom_size: int, bin_size: int) -> np.ndarray:
    n_bins = int(np.ceil(chrom_size / bin_size))
    counts = np.zeros(n_bins, dtype=np.int32)
    for start, end in gene_intervals:
        b0 = max(0, start // bin_size)
        b1 = min(n_bins - 1, end // bin_size)
        counts[b0:b1 + 1] += 1
    return counts


def bin_drop_density(drops: List[Dict], chrom_size: int, bin_size: int) -> np.ndarray:
    n_bins = int(np.ceil(chrom_size / bin_size))
    counts = np.zeros(n_bins, dtype=np.int32)
    for d in drops:
        try:
            gs, ge = int(d["genomic_start"]), int(d["genomic_end"])
        except (KeyError, ValueError):
            continue
        b0 = max(0, gs // bin_size)
        b1 = min(n_bins - 1, ge // bin_size)
        counts[b0:b1 + 1] += 1
    return counts


def load_sae_results(sae_run_dir: str) -> Optional[List[Dict]]:
    path = os.path.join(sae_run_dir, "data", "sae_results.tsv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                continue
            if len(fields) >= len(header):
                rows.append(dict(zip(header, fields)))
    return rows


# ── Analysis 1: Gene density vs drop density ─────────────────────────────────

def analysis_gene_density_vs_drop_density(chrom_data, gene_data, output_dir, bin_size=100_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from scipy import stats

    logger.info("Analysis 1: Gene density vs. drop density")
    chroms = sorted(chrom_data.keys())

    correlations = {}
    fig, axes = plt.subplots(6, 4, figsize=(36, 48))
    axes = axes.flatten()

    for ax_idx, chrom in enumerate(chroms):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        drops = chrom_data[chrom]["drops"]
        chrom_size = CHROM_SIZES_GRCH38.get(chrom, 0)
        if chrom_size == 0:
            ax.set_visible(False)
            continue

        drop_dens = bin_drop_density(drops, chrom_size, bin_size)
        gene_dens = gene_data.get(chrom, np.array([]))
        if len(gene_dens) == 0:
            gene_dens = np.zeros(len(drop_dens), dtype=np.int32)
        else:
            gene_dens = bin_gene_density(gene_dens, chrom_size, bin_size)

        n = min(len(drop_dens), len(gene_dens))
        gd, dd = gene_dens[:n].astype(float), drop_dens[:n].astype(float)

        # Exclude fully empty bins (assembly gaps / centromeres)
        mask = ~((gd == 0) & (dd == 0))
        if mask.sum() < 5:
            ax.set_visible(False)
            continue

        r, p = stats.pearsonr(gd[mask], dd[mask])
        correlations[chrom] = {"r": float(r), "p": float(p), "n_bins": int(mask.sum())}

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.scatter(gd[mask], dd[mask], s=6, alpha=0.35, color="steelblue",
                   label=f"{mask.sum()} bins (100 kb each)")
        m, b, *_ = stats.linregress(gd[mask], dd[mask])
        x_line = np.linspace(gd[mask].min(), gd[mask].max(), 100)
        ax.plot(x_line, m * x_line + b, color="crimson", lw=2.0,
                label=f"Linear fit (slope={m:.1f})")
        ax.set_title(f"{chrom}", fontsize=16, fontweight="bold", pad=6)
        ax.set_xlabel("Genes overlapping bin", fontsize=13)
        ax.set_ylabel("Entropy drops in bin", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.text(0.97, 0.05, f"r = {r:.2f}  {sig}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=13, color="crimson",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.8))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in range(len(chroms), len(axes)):
        axes[i].set_visible(False)

    # Shared legend in an empty panel if available, else figure-level
    dot_handle  = mlines.Line2D([], [], marker="o", color="steelblue", linestyle="None",
                                markersize=10, alpha=0.7, label="100 kb genomic bin\n(x = genes, y = drops)")
    line_handle = mlines.Line2D([], [], color="crimson", lw=2,
                                label="Linear regression fit")
    fig.legend(handles=[dot_handle, line_handle], fontsize=16, loc="lower right",
               frameon=True, framealpha=0.9, edgecolor="gray",
               title="Legend", title_fontsize=15,
               bbox_to_anchor=(0.98, 0.01))

    subtitle = ("Each panel = one chromosome. Each dot = one 100 kb genomic bin.\n"
                "X-axis: number of annotated genes overlapping that bin. "
                "Y-axis: number of Evo2 entropy drops detected in that bin.\n"
                "Bins in assembly gaps (both axes = 0) are excluded. "
                "r = Pearson correlation. * p<0.05, ** p<0.01, *** p<0.001.")
    fig.suptitle("Gene Density vs. Drop Density  (100 kb bins, GRCh38)",
                 fontsize=26, fontweight="bold", y=0.995)
    fig.text(0.5, 0.988, subtitle, ha="center", va="top", fontsize=14,
             color="#444444", wrap=True)

    fig.tight_layout(rect=[0, 0, 1, 0.983])
    out = os.path.join(output_dir, "1_gene_density_vs_drop_density.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    # Save correlations JSON
    with open(os.path.join(output_dir, "1_gene_drop_correlations.json"), "w") as f:
        json.dump(correlations, f, indent=2)

    return correlations


# ── Analysis 2: Entropy distribution per chromosome ───────────────────────────

def analysis_entropy_distributions(chrom_data, output_dir, n_sample=100_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    logger.info("Analysis 2: Entropy distributions per chromosome")

    chrom_samples = {}
    for chrom in sorted(chrom_data.keys()):
        run_dir = chrom_data[chrom]["run_dir"]
        try:
            entropy, _, _ = load_entropy(run_dir)
        except Exception as e:
            logger.warning(f"  {chrom}: {e}")
            continue
        valid = entropy[np.isfinite(entropy)]
        if len(valid) == 0:
            continue
        if len(valid) > n_sample:
            idx = np.random.choice(len(valid), n_sample, replace=False)
            valid = valid[idx]
        chrom_samples[chrom] = valid

    chroms = sorted(chrom_samples.keys(),
                    key=lambda c: [int(c[3:]) if c[3:].isdigit() else 99])

    fig, ax = plt.subplots(figsize=(28, 11))
    positions = list(range(len(chroms)))

    parts = ax.violinplot([chrom_samples[c] for c in chroms],
                          positions=positions,
                          showmedians=True, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.55)
        pc.set_edgecolor("#2a5f8f")
        pc.set_linewidth(0.8)
    parts["cmedians"].set_color("crimson")
    parts["cmedians"].set_linewidth(3.0)

    # IQR box + mean dot per chromosome
    for i, chrom in enumerate(chroms):
        v = chrom_samples[chrom]
        q25, q75 = np.percentile(v, 25), np.percentile(v, 75)
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.08, q25), 0.16, q75 - q25,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor="#1a3a5c", facecolor="white", alpha=0.7, zorder=4))
        ax.scatter(i, np.mean(v), color="navy", s=60, zorder=6, marker="D")

    ax.set_xticks(positions)
    ax.set_xticklabels(chroms, rotation=40, ha="right", fontsize=14)
    ax.set_ylabel("Entropy (nats)", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)

    # Clip y-axis to the informative range (1st–99th percentile across all chroms)
    all_vals = np.concatenate(list(chrom_samples.values()))
    ylo = max(0.0, float(np.percentile(all_vals, 1)) - 0.05)
    yhi = float(np.percentile(all_vals, 99)) + 0.05
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.7, len(chroms) - 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference line at genome-wide median
    genome_median = float(np.median(all_vals))
    ax.axhline(genome_median, color="gray", lw=1.5, linestyle="--", zorder=1,
               label=f"Genome-wide median ({genome_median:.3f} nats)")

    # Legend
    violin_patch = mpatches.Patch(facecolor="steelblue", alpha=0.6,
                                  edgecolor="#2a5f8f", label="Entropy distribution (violin)")
    median_line  = mlines.Line2D([], [], color="crimson", lw=3,
                                 label="Median entropy per chromosome")
    iqr_patch    = mpatches.Patch(facecolor="white", edgecolor="#1a3a5c", lw=1.2,
                                  label="Interquartile range (25th–75th %ile)")
    mean_dot     = mlines.Line2D([], [], marker="D", color="navy", linestyle="None",
                                 markersize=9, label="Mean entropy per chromosome")
    genome_line  = mlines.Line2D([], [], color="gray", lw=1.5, linestyle="--",
                                 label=f"Genome-wide median ({genome_median:.3f} nats)")
    ax.legend(handles=[violin_patch, median_line, iqr_patch, mean_dot, genome_line],
              fontsize=13, loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="gray", ncol=2)

    subtitle = ("Each violin shows the full distribution of per-position entropy scores "
                "sampled from that chromosome (100,000 positions randomly selected).\n"
                "Entropy (nats) measures Evo2's uncertainty: LOW entropy = model is confident "
                "= sequence is highly constrained/functional. HIGH entropy = model is uncertain "
                "= less structured sequence.\n"
                "Y-axis clipped to 1st–99th percentile range to focus on the bulk distribution "
                "(outliers near 0 exist due to assembly gap edges).")
    fig.suptitle("Entropy Distribution per Chromosome  (Evo2 log-probability scores)",
                 fontsize=22, fontweight="bold", y=1.01)
    fig.text(0.5, 0.995, subtitle, ha="center", va="top", fontsize=12.5,
             color="#444", wrap=True)

    fig.tight_layout()
    out = os.path.join(output_dir, "2_entropy_distributions.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    stats_out = {}
    for chrom in chroms:
        v = chrom_samples[chrom]
        stats_out[chrom] = {
            "mean": float(np.mean(v)), "median": float(np.median(v)),
            "std": float(np.std(v)),
            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
        }
    with open(os.path.join(output_dir, "2_entropy_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    return stats_out


# ── Analysis 3: Drop size distribution ───────────────────────────────────────

def analysis_drop_sizes(chrom_data, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    logger.info("Analysis 3: Drop size distributions")

    chrom_sizes = {}
    for chrom in sorted(chrom_data.keys()):
        drops = chrom_data[chrom]["drops"]
        sizes = []
        for d in drops:
            try:
                sizes.append(int(d["genomic_end"]) - int(d["genomic_start"]))
            except (KeyError, ValueError):
                continue
        if sizes:
            chrom_sizes[chrom] = np.array(sizes)

    if not chrom_sizes:
        logger.warning("  No drop size data found")
        return {}

    chroms = sorted(chrom_sizes.keys(),
                    key=lambda c: [int(c[3:]) if c[3:].isdigit() else 99])

    all_sizes = np.concatenate(list(chrom_sizes.values())) / 1000  # → kb
    clip_kb   = 5.0   # focus on the vast majority of drops
    pct_shown = float(np.mean(all_sizes <= clip_kb)) * 100

    fig, axes = plt.subplots(1, 2, figsize=(32, 12))

    # ── Left: violin per chromosome (clipped to clip_kb) ──────────────────
    ax = axes[0]
    data_clipped = [np.clip(chrom_sizes[c] / 1000, 0, clip_kb) for c in chroms]

    vp = ax.violinplot(data_clipped, positions=range(len(chroms)),
                       showmedians=True, showextrema=False)
    for pc in vp["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.55)
        pc.set_edgecolor("#2a5f8f")
        pc.set_linewidth(0.8)
    vp["cmedians"].set_color("crimson")
    vp["cmedians"].set_linewidth(2.5)

    for i, chrom in enumerate(chroms):
        v = chrom_sizes[chrom] / 1000
        q25, q75 = np.percentile(v, 25), np.percentile(v, 75)
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.07, q25), 0.14, max(q75 - q25, 0.01),
            boxstyle="square,pad=0", linewidth=1.0,
            edgecolor="#1a3a5c", facecolor="white", alpha=0.75, zorder=4))
        ax.scatter(i, np.mean(v), color="navy", s=40, zorder=6, marker="D")

    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels(chroms, rotation=40, ha="right", fontsize=13)
    ax.set_ylabel("Drop length (kb)", fontsize=15)
    ax.set_ylim(0, clip_kb)
    ax.set_xlim(-0.6, len(chroms) - 0.4)
    ax.set_title(f"Drop Length per Chromosome\n(y-axis clipped to {clip_kb} kb — "
                 f"shows {pct_shown:.1f}% of all drops)",
                 fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    violin_patch = mpatches.Patch(facecolor="steelblue", alpha=0.6,
                                  edgecolor="#2a5f8f",
                                  label="Drop length distribution per chromosome\n(violin shape = density of drop sizes)")
    med_line  = mlines.Line2D([], [], color="crimson", lw=2.5,
                              label="Median drop length")
    iqr_patch = mpatches.Patch(facecolor="white", edgecolor="#1a3a5c", lw=1.0,
                               label="IQR (25th–75th percentile)")
    mean_dot  = mlines.Line2D([], [], marker="D", color="navy", linestyle="None",
                              markersize=8, label="Mean drop length")
    ax.legend(handles=[violin_patch, med_line, iqr_patch, mean_dot],
              fontsize=12, loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="gray")

    # ── Right: histogram focused on <5 kb ─────────────────────────────────
    ax2 = axes[1]
    small = all_sizes[all_sizes <= clip_kb]
    med_s  = float(np.median(small))
    mean_s = float(np.mean(small))

    ax2.hist(small, bins=100, color="steelblue", alpha=0.75, edgecolor="none",
             label=f"Drop lengths ≤ {clip_kb} kb  ({len(small):,} drops, {pct_shown:.1f}% of total)")
    ax2.axvline(med_s,  color="crimson",    lw=2.5,
                label=f"Median: {med_s:.2f} kb")
    ax2.axvline(mean_s, color="darkorange", lw=2.5, linestyle="--",
                label=f"Mean: {mean_s:.2f} kb")

    # Annotate the peak bin
    counts, edges = np.histogram(small, bins=100)
    peak_idx  = np.argmax(counts)
    peak_x    = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    peak_count = counts[peak_idx]
    ax2.annotate(f"Peak: {peak_x:.2f} kb\n({peak_count:,} drops)",
                 xy=(peak_x, peak_count),
                 xytext=(peak_x + clip_kb * 0.15, peak_count * 0.85),
                 fontsize=12, color="#1a3a5c",
                 arrowprops=dict(arrowstyle="->", color="#1a3a5c", lw=1.2))

    ax2.set_xlabel("Drop length (kb)", fontsize=15)
    ax2.set_ylabel("Number of drops", fontsize=15)
    ax2.set_xlim(0, clip_kb)
    ax2.tick_params(labelsize=13)
    n_total = len(all_sizes)
    pct_large = 100 - pct_shown
    ax2.set_title(f"Genome-wide Drop Length Histogram  (≤ {clip_kb} kb shown)\n"
                  f"Total drops: {n_total:,}  ·  {pct_large:.1f}% are longer than {clip_kb} kb (not shown)",
                  fontsize=16, fontweight="bold")
    ax2.legend(fontsize=13, frameon=True, framealpha=0.9, edgecolor="gray")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    subtitle = ("An entropy 'drop' is a contiguous genomic region where Evo2's per-base entropy "
                "falls significantly below the local baseline — indicating the model finds the "
                "sequence highly predictable/constrained. Drop length = genomic span of that region.\n"
                "Most drops are sub-kilobase (single functional elements); rare large drops span "
                "repetitive or highly conserved multi-gene regions.")
    fig.suptitle("Entropy Drop Length Distributions  (all human chromosomes)",
                 fontsize=22, fontweight="bold", y=1.01)
    fig.text(0.5, 0.995, subtitle, ha="center", va="top", fontsize=13,
             color="#444", wrap=True)

    fig.tight_layout()
    out = os.path.join(output_dir, "3_drop_size_distributions.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    stats_out = {}
    for chrom in chroms:
        s = chrom_sizes[chrom]
        stats_out[chrom] = {
            "n_drops": len(s),
            "mean_kb": float(np.mean(s) / 1000),
            "median_kb": float(np.median(s) / 1000),
            "p95_kb": float(np.percentile(s, 95) / 1000),
            "max_kb": float(np.max(s) / 1000),
        }
    with open(os.path.join(output_dir, "3_drop_size_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    return stats_out


# ── Analysis 4: Mean entropy vs. gene count (scatter) ─────────────────────────

def analysis_entropy_vs_gene_count(chrom_data, gene_data, entropy_stats, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    logger.info("Analysis 4: Mean entropy vs. gene count scatter")

    chroms, mean_entropies, gene_counts, n_drops_per_mbp = [], [], [], []

    for chrom in sorted(chrom_data.keys()):
        if chrom not in entropy_stats:
            continue
        genes = gene_data.get(chrom, [])
        drops = chrom_data[chrom]["drops"]
        chrom_size = CHROM_SIZES_GRCH38.get(chrom, 0)
        if chrom_size == 0:
            continue

        chroms.append(chrom)
        mean_entropies.append(entropy_stats[chrom]["mean"])
        gene_counts.append(len(genes))
        n_drops_per_mbp.append(len(drops) / (chrom_size / 1e6))

    if len(chroms) < 3:
        logger.warning("  Not enough chroms for scatter")
        return

    me = np.array(mean_entropies)
    gc = np.array(gene_counts)
    nd = np.array(n_drops_per_mbp)

    import matplotlib.lines as mlines

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    for ax, y, ylabel, color, interp in [
        (axes[0], me, "Mean Entropy (nats)", "steelblue",
         "Lower entropy = model more confident = more constrained/functional sequence"),
        (axes[1], nd, "Entropy Drops per Mbp", "darkorange",
         "More drops per Mbp = higher density of functional elements detected"),
    ]:
        r, p = scipy_stats.pearsonr(gc, y)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

        ax.scatter(gc, y, s=120, color=color, alpha=0.85, zorder=3,
                   label="One chromosome (24 total)")
        for i, chrom in enumerate(chroms):
            ax.annotate(chrom.replace("chr", ""), (gc[i], y[i]),
                        fontsize=11, ha="center", va="bottom",
                        fontweight="bold", color="#333",
                        xytext=(0, 6), textcoords="offset points")
        m, b, *_ = scipy_stats.linregress(gc, y)
        x_line = np.linspace(gc.min(), gc.max(), 100)
        ax.plot(x_line, m * x_line + b, color="crimson", lw=2.0, linestyle="--",
                label=f"Linear fit  (r = {r:.2f}, {sig})")
        ax.set_xlabel("Number of annotated genes on chromosome (GRCh38)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(f"{ylabel}\nvs. Gene Count per Chromosome",
                     fontsize=15, fontweight="bold")
        ax.text(0.97, 0.05, f"r = {r:.3f}  {sig}\np = {p:.2e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
                color="crimson",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="crimson", alpha=0.85))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        chrom_dot = mlines.Line2D([], [], marker="o", color=color, linestyle="None",
                                  markersize=11, alpha=0.85,
                                  label="One chromosome (24 total)\n(labels = chr number)")
        fit_line  = mlines.Line2D([], [], color="crimson", lw=2, linestyle="--",
                                  label=f"Linear regression  r={r:.2f} {sig}")
        ax.legend(handles=[chrom_dot, fit_line], fontsize=12, loc="upper left",
                  frameon=True, framealpha=0.9, edgecolor="gray")
        ax.set_xlabel("Number of annotated genes on chromosome (GRCh38)", fontsize=14)

    subtitle = ("Each point = one human chromosome. Gene counts from GRCh38 GTF annotation.\n"
                "Left: does having more genes correlate with lower (more confident) entropy? "
                "Right: does higher gene count mean more functional elements detected per Mbp?\n"
                "Chromosome labels show chr number (e.g. '19' = chr19, 'X' = chrX, 'Y' = chrY).")
    fig.suptitle("Chromosome-level: Gene Count vs. Evo2 Entropy Metrics",
                 fontsize=22, fontweight="bold", y=1.02)
    fig.text(0.5, 1.005, subtitle, ha="center", va="top", fontsize=13, color="#444")

    fig.tight_layout()
    out = os.path.join(output_dir, "4_entropy_vs_gene_count.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Analysis 5: Centromere-flanking drop density ──────────────────────────────

def analysis_centromere_flanking(chrom_data, output_dir,
                                 flank_bp=30_000_000, bin_size=500_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Analysis 5: Centromere-flanking drop density")

    n_bins_per_side = flank_bp // bin_size
    total_bins = 2 * n_bins_per_side
    bin_centers_mb = (np.arange(total_bins) - n_bins_per_side + 0.5) * bin_size / 1e6

    chrom_profiles = {}

    for chrom in sorted(chrom_data.keys()):
        cen_mid = CENTROMERE_MID_GRCH38.get(chrom)
        if cen_mid is None:
            continue

        drops = chrom_data[chrom]["drops"]
        chrom_size = CHROM_SIZES_GRCH38.get(chrom, 0)
        if chrom_size == 0 or not drops:
            continue

        # Build profile: count drops in bins relative to centromere midpoint
        profile = np.zeros(total_bins, dtype=np.float64)
        for d in drops:
            try:
                mid = (int(d["genomic_start"]) + int(d["genomic_end"])) // 2
            except (KeyError, ValueError):
                continue
            offset = mid - cen_mid
            bin_idx = int(offset // bin_size) + n_bins_per_side
            if 0 <= bin_idx < total_bins:
                profile[bin_idx] += 1

        # Normalize to drops per Mbp
        profile = profile / (bin_size / 1e6)
        chrom_profiles[chrom] = profile

    if not chrom_profiles:
        logger.warning("  No centromere flanking data")
        return

    import matplotlib.lines as mlines

    fig, axes = plt.subplots(1, 2, figsize=(26, 10))

    # ── Left: individual chromosome profiles ──────────────────────────────
    ax = axes[0]
    cmap_c = plt.cm.tab20
    chrom_list = sorted(chrom_profiles.keys(),
                        key=lambda c: int(c[3:]) if c[3:].isdigit() else 99)
    for i, chrom in enumerate(chrom_list):
        profile = chrom_profiles[chrom]
        ax.plot(bin_centers_mb, profile, alpha=0.5, lw=1.5,
                color=cmap_c(i / max(len(chrom_profiles), 1)),
                label=chrom)
    ax.axvline(0, color="black", lw=2.5, linestyle="--", zorder=5)
    ax.text(0.3, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            "← centromere →", fontsize=11, color="black", va="top")
    ax.set_xlabel("Distance from centromere midpoint (Mb)\nNegative = p-arm  ·  Positive = q-arm",
                  fontsize=14)
    ax.set_ylabel("Entropy drops per Mbp", fontsize=14)
    ax.set_title("Drop Density Profile Around Centromere\n(each line = one chromosome)",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper right", frameon=True,
              framealpha=0.85, edgecolor="gray",
              title="Chromosome", title_fontsize=10)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Right: mean ± SEM across chromosomes ──────────────────────────────
    ax2 = axes[1]
    stack     = np.stack([chrom_profiles[c] for c in chrom_list])
    mean_prof = np.mean(stack, axis=0)
    sem_prof  = np.std(stack, axis=0) / np.sqrt(len(stack))

    ax2.fill_between(bin_centers_mb, mean_prof - sem_prof, mean_prof + sem_prof,
                     alpha=0.25, color="steelblue", label="Mean ± SEM across chromosomes")
    ax2.plot(bin_centers_mb, mean_prof, color="steelblue", lw=3,
             label=f"Mean drop density (n={len(chrom_list)} chroms)")
    ax2.axvline(0, color="crimson", lw=2.5, linestyle="--",
                label="Centromere midpoint")

    # Shade the centromere zone (±2 Mb)
    ax2.axvspan(-2, 2, alpha=0.08, color="crimson", label="±2 Mb centromere zone")

    ax2.set_xlabel("Distance from centromere midpoint (Mb)\nNegative = p-arm  ·  Positive = q-arm",
                   fontsize=14)
    ax2.set_ylabel("Entropy drops per Mbp  (mean ± SEM)", fontsize=14)
    ax2.set_title(f"Mean Drop Density Profile Around Centromeres\n"
                  f"(averaged across {len(chrom_list)} chromosomes, 500 kb bins)",
                  fontsize=15, fontweight="bold")
    ax2.legend(fontsize=12, loc="upper right", frameon=True,
               framealpha=0.9, edgecolor="gray")
    ax2.tick_params(labelsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    subtitle = ("Centromere positions from GRCh38 UCSC cytoBand coordinates. "
                "Each chromosome's drop density profile is computed in 500 kb bins "
                "relative to its centromere midpoint, then averaged.\n"
                "If Evo2 detects fewer functional elements near centromeres, "
                "drop density should dip toward zero at x=0. "
                "A rise would suggest pericentric heterochromatin has unusual sequence patterns.")
    fig.suptitle("Entropy Drop Density Around Centromeres  (±30 Mb)",
                 fontsize=22, fontweight="bold", y=1.02)
    fig.text(0.5, 1.005, subtitle, ha="center", va="top", fontsize=13, color="#444")

    fig.tight_layout()
    out = os.path.join(output_dir, "5_centromere_flanking.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Analysis 6: SAE feature diversity per chromosome ─────────────────────────

def analysis_sae_feature_diversity(chrom_order, results_dir, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Analysis 6: SAE feature diversity per chromosome")

    chrom_feature_stats = {}

    for chrom in chrom_order:
        sae_run = find_latest_completed(results_dir, chrom, "sae")
        if sae_run is None:
            # Try without COMPLETED sentinel (runs that have data but no sentinel)
            sae_base = os.path.join(results_dir, chrom, "sae")
            if not os.path.isdir(sae_base):
                continue
            runs = sorted([d for d in os.listdir(sae_base)
                           if os.path.isdir(os.path.join(sae_base, d))])
            if not runs:
                continue
            sae_run = os.path.join(sae_base, runs[-1])
            tsv_path = os.path.join(sae_run, "data", "sae_results.tsv")
            if not os.path.exists(tsv_path):
                continue

        rows = load_sae_results(sae_run)
        if not rows:
            continue

        feature_counts: Dict[int, int] = defaultdict(int)
        for row in rows:
            top = row.get("top_features", "")
            for feat_str in top.split(","):
                feat_str = feat_str.strip().split(":")[0]
                if feat_str.isdigit():
                    feature_counts[int(feat_str)] += 1

        n_regions = len(rows)
        n_unique = len(feature_counts)
        top10 = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top1_frac = top10[0][1] / n_regions if top10 and n_regions > 0 else 0.0

        chrom_feature_stats[chrom] = {
            "n_regions": n_regions,
            "n_unique_features": n_unique,
            "top10_features": [(int(f), int(c)) for f, c in top10],
            "top1_fraction": float(top1_frac),
            "sae_run": sae_run,
        }
        logger.info(f"  {chrom}: {n_regions} regions, {n_unique} unique features, "
                    f"top feature appears in {top1_frac:.1%} of regions")

    # Need a meaningful number of chromosomes before this plot is useful
    MIN_CHROMS_FOR_PLOT = 10
    if len(chrom_feature_stats) < MIN_CHROMS_FOR_PLOT:
        n_have = len(chrom_feature_stats)
        logger.warning(f"  SAE data available for only {n_have}/{len(ALL_HUMAN_CHROMS)} "
                       f"chromosomes (need {MIN_CHROMS_FOR_PLOT}+). "
                       f"Skipping plot — re-run once SAE jobs complete.")
        with open(os.path.join(output_dir, "6_sae_not_ready.txt"), "w") as f:
            f.write(f"SAE data available for {n_have} chromosomes. "
                    f"Need {MIN_CHROMS_FOR_PLOT}+ for a meaningful comparison.\n"
                    f"Chromosomes with data: {sorted(chrom_feature_stats.keys())}\n"
                    f"Re-run with: sbatch tools/run_all_plots.sh --analyses 6\n")
        return chrom_feature_stats

    chroms_with_data = sorted(chrom_feature_stats.keys(),
                               key=lambda c: [int(c[3:]) if c[3:].isdigit() else 99])

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Unique feature count per chromosome
    ax = axes[0]
    n_unique = [chrom_feature_stats[c]["n_unique_features"] for c in chroms_with_data]
    ax.bar(range(len(chroms_with_data)), n_unique, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(chroms_with_data)))
    ax.set_xticklabels(chroms_with_data, rotation=40, ha="right", fontsize=13)
    ax.set_ylabel("Unique SAE features activated", fontsize=14)
    ax.set_title("SAE Feature Diversity per Chromosome", fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Top-feature dominance
    ax2 = axes[1]
    top1_fracs = [chrom_feature_stats[c]["top1_fraction"] for c in chroms_with_data]
    ax2.bar(range(len(chroms_with_data)), top1_fracs, color="darkorange", alpha=0.8)
    ax2.set_xticks(range(len(chroms_with_data)))
    ax2.set_xticklabels(chroms_with_data, rotation=40, ha="right", fontsize=13)
    ax2.set_ylabel("Fraction of regions with top feature", fontsize=14)
    ax2.set_title("Top Feature Dominance per Chromosome", fontsize=15, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("SAE Feature Statistics per Chromosome\n"
                 f"(data available for {len(chroms_with_data)}/{len(ALL_HUMAN_CHROMS)} chroms)",
                 fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(output_dir, "6_sae_feature_diversity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {out}")

    with open(os.path.join(output_dir, "6_sae_feature_stats.json"), "w") as f:
        json.dump(chrom_feature_stats, f, indent=2)

    return chrom_feature_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Six chromosome-level analyses of Evo2 scoring results.")
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--gtf", default=None, help="Path to genomic GTF (required for analyses 1, 4)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--analyses", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                    help="Which analyses to run (default: all)")
    ap.add_argument("--bin_size", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if args.output_dir is None:
        analyses_str = "".join(str(a) for a in sorted(args.analyses))
        flags = f"analyses{analyses_str}"
        args.output_dir = build_run_dir(args.results_dir, "_chromosome_analysis", "plots", flags)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output dir: {args.output_dir}")

    t0 = time.time()

    # ── Load all scoring runs ──────────────────────────────────────────────
    logger.info("Loading scoring runs...")
    chrom_data = {}
    for chrom in ALL_HUMAN_CHROMS:
        run_dir = find_latest_completed(args.results_dir, chrom, "scoring")
        if run_dir is None:
            continue
        drops = load_drop_boundaries(run_dir)
        chrom_data[chrom] = {"run_dir": run_dir, "drops": drops}
        logger.info(f"  {chrom}: {len(drops)} drops")

    logger.info(f"Loaded {len(chrom_data)} chromosomes")

    if not chrom_data:
        logger.error("No completed scoring runs found.")
        sys.exit(1)

    # ── Parse GTF if needed ────────────────────────────────────────────────
    gene_data = {}
    if args.gtf and any(a in args.analyses for a in [1, 4]):
        gene_data = parse_gtf_genes(args.gtf)

    # ── Run analyses ────────────────────────────────────────────────────────
    entropy_stats = {}

    if 1 in args.analyses:
        if not gene_data:
            logger.warning("Analysis 1 requires --gtf; skipping")
        else:
            analysis_gene_density_vs_drop_density(chrom_data, gene_data, args.output_dir, args.bin_size)

    if 2 in args.analyses:
        entropy_stats = analysis_entropy_distributions(chrom_data, args.output_dir)

    if 3 in args.analyses:
        analysis_drop_sizes(chrom_data, args.output_dir)

    if 4 in args.analyses:
        if not gene_data:
            logger.warning("Analysis 4 requires --gtf; skipping")
        elif not entropy_stats:
            # Run entropy stats if not already done
            entropy_stats = analysis_entropy_distributions(chrom_data, args.output_dir)
        analysis_entropy_vs_gene_count(chrom_data, gene_data, entropy_stats, args.output_dir)

    if 5 in args.analyses:
        analysis_centromere_flanking(chrom_data, args.output_dir)

    if 6 in args.analyses:
        analysis_sae_feature_diversity(ALL_HUMAN_CHROMS, args.results_dir, args.output_dir)

    wall = time.time() - t0
    write_completed(args.output_dir, "chromosome_analysis.py", wall)
    logger.info(f"All analyses done in {wall:.1f}s. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
