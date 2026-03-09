#!/usr/bin/env python3
"""
analyze_scoring_results.py

Analyze output from score_chromosome.py: coverage, region statistics,
and publication-quality entropy visualization plots.

Usage:
    # Basic text report (no GPU / matplotlib needed)
    python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_4gpu

    # With basic entropy profile plot
    python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_4gpu --plot

    # Full plot suite (transitions, zoom, interactive)
    python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_4gpu --all_plots

    # Restrict plot to a genomic window
    python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_4gpu \
        --plot --plot_start 20000000 --plot_end 25000000

    # Chromosome-wide dashboard (binned overview)
    python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full --dashboard

    # Dashboard with custom bin size (100kb)
    python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full \
        --dashboard --dashboard_bin_size 100000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import Counter

from urllib.parse import unquote as _url_unquote

import numpy as np

# Add project root to path for results_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source, find_latest_completed

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure module logger to write timestamped messages to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(level)


def _out(prefix, suffix):
    """Build output path. Uses clean name in _out.directory when set."""
    if _out.directory:
        return os.path.join(_out.directory, suffix)
    return f"{prefix}.{suffix}"

_out.directory = None  # set by --outdir in main()


class _TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self._file = open(filepath, "w")
        self._stdout = sys.stdout

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _rolling_mean(arr, w):
    """Rolling mean with edge handling."""
    if len(arr) < w:
        return arr.copy()
    kernel = np.ones(w) / w
    pad = w // 2
    filled = np.where(np.isnan(arr), np.nanmean(arr), arr)
    padded = np.pad(filled, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _bin_entropy(entropy, regions, bin_size=50_000):
    """Aggregate per-position entropy into fixed-width genomic bins.

    Returns dict with bin_centers, mean/min/max entropy, coverage fraction,
    and region counts per bin (zscore and MAD separately).
    """
    L = len(entropy)
    n_bins = int(np.ceil(L / bin_size))
    padded_len = n_bins * bin_size

    # Pad entropy to exact multiple of bin_size
    padded = np.full(padded_len, np.nan)
    padded[:L] = entropy
    reshaped = padded.reshape(n_bins, bin_size)

    with np.errstate(all='ignore'):
        mean_entropy = np.nanmean(reshaped, axis=1)
        min_entropy = np.nanmin(reshaped, axis=1)
        max_entropy = np.nanmax(reshaped, axis=1)

    # Coverage fraction: ratio of non-NaN values per bin
    coverage_frac = np.sum(~np.isnan(reshaped), axis=1) / bin_size

    # Bin centers (array-index based)
    bin_centers = np.arange(n_bins) * bin_size + bin_size // 2

    # Count regions per bin by method
    n_regions_zscore = np.zeros(n_bins, dtype=int)
    n_regions_mad = np.zeros(n_bins, dtype=int)
    for r in regions:
        rs = r["drop_start"]
        re = r["drop_end"]
        bin_lo = rs // bin_size
        bin_hi = min(re // bin_size, n_bins - 1)
        target = n_regions_zscore if r["method"] == "zscore" else n_regions_mad
        for b in range(bin_lo, bin_hi + 1):
            target[b] += 1

    return {
        "bin_centers": bin_centers,
        "mean_entropy": mean_entropy,
        "min_entropy": min_entropy,
        "max_entropy": max_entropy,
        "coverage_frac": coverage_frac,
        "n_regions_zscore": n_regions_zscore,
        "n_regions_mad": n_regions_mad,
        "bin_size": bin_size,
        "n_bins": n_bins,
    }


# ── GTF / GFF3 annotation parsing ─────────────────────────────────────────

def parse_gtf_attributes(attr_str):
    """Parse GTF attribute string ('key "value"; ...') into dict."""
    out = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(" ", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        val = parts[1].strip().strip('"')
        out[key] = val
    return out


def parse_gff3_attributes(attr_str):
    """Parse GFF3 attribute string ('key=value;...') into dict."""
    out = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, val = item.split("=", 1)
            out[key.strip()] = _url_unquote(val.strip())
        else:
            out[item] = ""
    return out


def load_annotation_features(path, chrom, start, end, fmt="gtf"):
    """Load gene/exon/CDS features overlapping [start, end) from a GTF or GFF3.

    Args:
        path: Path to GTF or GFF3 file
        chrom: Chromosome/seqid to filter (e.g. 'NC_000022.11')
        start: Region start (1-based inclusive)
        end: Region end (1-based exclusive)
        fmt: 'gtf' or 'gff3'

    Returns:
        Sorted list of feature dicts with keys: seqid, source, feature_type,
        start, end, end_exclusive, score, strand, frame, attributes, name.
    """
    parse_attrs = parse_gtf_attributes if fmt == "gtf" else parse_gff3_attributes
    keep_types = {"gene", "exon", "CDS", "mRNA", "transcript",
                  "five_prime_UTR", "three_prime_UTR",
                  "start_codon", "stop_codon"}

    features = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#") or not line.strip():
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqid, source, feature_type, f_start, f_end, score, strand, frame, attrs = fields

            if seqid != chrom:
                continue
            if feature_type not in keep_types:
                continue

            f_start_i = int(f_start)       # 1-based inclusive
            f_end_i = int(f_end)           # 1-based inclusive
            f_end_excl = f_end_i + 1       # 1-based exclusive

            # Check overlap with [start, end)
            if f_start_i >= end or f_end_excl <= start:
                continue

            attrs_d = parse_attrs(attrs)

            # Extract a readable name
            name = (attrs_d.get("gene_name")
                    or attrs_d.get("Name")
                    or attrs_d.get("gene")
                    or attrs_d.get("gene_id", ""))

            features.append({
                "seqid": seqid,
                "source": source,
                "feature_type": feature_type,
                "start": f_start_i,
                "end": f_end_i,
                "end_exclusive": f_end_excl,
                "score": score,
                "strand": strand,
                "frame": frame,
                "attributes": attrs_d,
                "name": name,
            })

    features.sort(key=lambda f: (f["start"], f["end_exclusive"]))
    return features


# ── Gene annotation drawing functions ─────────────────────────────────────

_GFF_FEATURE_COLORS = {
    "CDS":              "#3498db",  # Blue
    "gene":             "#2ecc71",  # Green
    "mRNA":             "#1abc9c",  # Teal
    "exon":             "#a8e6cf",  # Light green
    "transcript":       "#1abc9c",  # Teal
    "five_prime_UTR":   "#e67e22",  # Orange
    "three_prime_UTR":  "#e74c3c",  # Red
    "start_codon":      "#9b59b6",  # Purple
    "stop_codon":       "#8e44ad",  # Dark purple
}
_GFF_DEFAULT_COLOR = "#95a5a6"  # Gray


def draw_gene_track(ax, features, view_start, view_end):
    """Draw a gene/exon annotation track on a dedicated axes.

    Features are drawn as colored horizontal bars grouped by feature type,
    using genomic coordinates on the x-axis and normalized [0, 1] y-space.
    This axes should be a separate subplot below the entropy plot.

    Args:
        ax: matplotlib Axes (dedicated annotation panel)
        features: list of feature dicts from load_annotation_features()
        view_start: left edge of the current view (genomic coord)
        view_end: right edge of the current view (genomic coord)
    """
    from matplotlib.patches import Rectangle, Patch

    if not features:
        return

    # Filter to features overlapping [view_start, view_end)
    vis = [f for f in features
           if f["start"] < view_end and f["end_exclusive"] > view_start]
    if not vis:
        return

    x_range = view_end - view_start
    if x_range <= 0:
        return

    # Group by feature type and assign rows
    types_present = sorted(set(f["feature_type"] for f in vis))
    n_types = len(types_present)
    type_to_row = {ft: i for i, ft in enumerate(types_present)}

    # Draw in normalized y-space [0, 1]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    sub_height = 1.0 / max(n_types, 1)

    # Track background
    ax.add_patch(Rectangle(
        (view_start, 0), x_range, 1.0,
        facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=0.5, zorder=0
    ))

    _type_short = {
        "five_prime_UTR": "5' UTR", "three_prime_UTR": "3' UTR",
        "start_codon": "start", "stop_codon": "stop",
    }

    # Row labels
    for ftype, row in type_to_row.items():
        row_mid_y = (row + 0.5) * sub_height
        display = _type_short.get(ftype, ftype)
        color = _GFF_FEATURE_COLORS.get(ftype, _GFF_DEFAULT_COLOR)
        ax.text(
            view_start + x_range * 0.005, row_mid_y, display,
            ha="left", va="center", fontsize=6, fontweight="bold",
            color=color, zorder=13, clip_on=True,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8),
        )

    # Feature bars
    for feat in vis:
        row = type_to_row[feat["feature_type"]]
        color = _GFF_FEATURE_COLORS.get(feat["feature_type"], _GFF_DEFAULT_COLOR)
        s = max(feat["start"], view_start)
        e = min(feat["end_exclusive"], view_end)
        feat_bottom = row * sub_height

        ax.add_patch(Rectangle(
            (s, feat_bottom + sub_height * 0.075), e - s, sub_height * 0.85,
            facecolor=color, edgecolor="none", alpha=0.85, zorder=11
        ))

        # Label wide bars
        bar_frac = (e - s) / x_range
        mid_x = (s + e) / 2
        mid_y = feat_bottom + sub_height / 2
        if bar_frac > 0.06 and feat["name"]:
            label = feat["name"][:25] + "..." if len(feat["name"]) > 25 else feat["name"]
            ax.text(mid_x, mid_y, label, ha="center", va="center",
                    fontsize=5.5, color="white", fontweight="bold",
                    zorder=12, clip_on=True)
        elif bar_frac > 0.015:
            short = _type_short.get(feat["feature_type"], feat["feature_type"])
            ax.text(mid_x, mid_y, short, ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold",
                    zorder=12, clip_on=True)

    # Build fresh legend for annotation types
    handles = []
    labels = []
    for ftype in types_present:
        color = _GFF_FEATURE_COLORS.get(ftype, _GFF_DEFAULT_COLOR)
        display = _type_short.get(ftype, ftype)
        handles.append(Patch(facecolor=color, edgecolor="none", label=display))
        labels.append(display)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
             fontsize=7, ncol=1, framealpha=0.9)
    ax.set_ylabel("Annotations", fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def shade_genes(ax, features, alpha=0.08, color="#2ecc71"):
    """Add light vertical shading behind gene regions.

    Args:
        ax: matplotlib Axes
        features: list of feature dicts from load_annotation_features()
        alpha: shading transparency
        color: shading color
    """
    for f in features:
        if f["feature_type"] == "gene":
            ax.axvspan(f["start"], f["end_exclusive"], alpha=alpha,
                       facecolor=color, edgecolor="none")


def _resolve_path(prefix, suffix):
    """Find a data file, trying multiple conventions:
    1. {prefix}.{suffix}          (flat: chromosome_scores/chr22_full.entropy.npz)
    2. {prefix}/{suffix}          (prefix is a directory: .../data/entropy.npz)
    3. {prefix_dir}/{suffix}      (clean name in parent dir)
    """
    # 1. Traditional flat prefix
    prefixed = f"{prefix}.{suffix}"
    if os.path.exists(prefixed):
        return prefixed
    # 2. Prefix is a directory (user passed --prefix path/to/data/)
    if os.path.isdir(prefix):
        inside = os.path.join(prefix, suffix)
        if os.path.exists(inside):
            return inside
    # 3. Clean name in the same directory as prefix
    clean = os.path.join(os.path.dirname(prefix), suffix)
    if os.path.exists(clean):
        return clean
    # Default to prefixed path (will raise FileNotFoundError naturally)
    return prefixed


def _load_entropy(prefix: str):
    """Load entropy.npz and return (entropy, chrom, start, end)."""
    path = _resolve_path(prefix, "entropy.npz")
    data = np.load(path, allow_pickle=True)
    entropy = data["entropy"]
    chrom = str(data["chrom"])
    start = int(data["start"])
    end = int(data["end"])
    return entropy, chrom, start, end


def _load_boundaries(prefix: str):
    """Load drop_boundaries.tsv into a list of dicts."""
    path = _resolve_path(prefix, "drop_boundaries.tsv")
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            row = dict(zip(header, cols))
            # cast numeric fields
            for k in ("drop_start", "drop_end", "genomic_start", "genomic_end",
                       "region_length"):
                if k in row:
                    row[k] = int(row[k])
            for k in ("start_confidence", "end_confidence",
                       "mean_entropy", "min_entropy"):
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def _load_drops_rises(prefix: str):
    """Load drops.tsv and rises.tsv into scored dicts by method.

    Returns:
        scored_drops: {"zscore": [(pos, score), ...], "mad": [...]}
        scored_rises: {"zscore": [(pos, score), ...], "mad": [...]}
    """
    scored_drops = {}
    scored_rises = {}

    for kind, out_dict in [("drops", scored_drops), ("rises", scored_rises)]:
        path = _resolve_path(prefix, f"{kind}.tsv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            header = None
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if header is None:
                    header = cols
                    continue
                row = dict(zip(header, cols))
                method = row.get("method", "unknown")
                pos = int(row["position"])
                conf = float(row["confidence"])
                out_dict.setdefault(method, []).append((pos, conf))

    return scored_drops, scored_rises


def _load_summary(prefix: str):
    path = _resolve_path(prefix, "summary.json")
    with open(path) as f:
        return json.load(f)


# ── coverage report ──────────────────────────────────────────────────────────

def coverage_report(entropy, chrom, start, end):
    """Print coverage / NaN statistics."""
    L = len(entropy)
    n_scored = int(np.count_nonzero(~np.isnan(entropy)))
    n_nan = L - n_scored
    frac_scored = n_scored / L if L else 0

    print("=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)
    print(f"  Chromosome      : {chrom}")
    print(f"  Genomic range   : {start:,} - {end:,}  ({L:,} bp)")
    print(f"  Scored positions: {n_scored:,}  ({frac_scored:.2%})")
    print(f"  NaN positions   : {n_nan:,}  ({1 - frac_scored:.2%})")

    # Identify contiguous NaN runs (likely N-blocks or OOM gaps)
    is_nan = np.isnan(entropy)
    nan_runs = []
    i = 0
    while i < L:
        if is_nan[i]:
            j = i
            while j < L and is_nan[j]:
                j += 1
            nan_runs.append((i, j, j - i))
            i = j
        else:
            i += 1

    if nan_runs:
        lengths = [r[2] for r in nan_runs]
        print(f"\n  NaN gap summary ({len(nan_runs)} gaps):")
        print(f"    Shortest gap  : {min(lengths):,} bp")
        print(f"    Longest gap   : {max(lengths):,} bp")
        print(f"    Median gap    : {int(np.median(lengths)):,} bp")
        print(f"    Total NaN bp  : {sum(lengths):,}")

        # Bucket by size
        buckets = {"<100": 0, "100-1K": 0, "1K-10K": 0, "10K-100K": 0, ">100K": 0}
        for ln in lengths:
            if ln < 100:
                buckets["<100"] += 1
            elif ln < 1_000:
                buckets["100-1K"] += 1
            elif ln < 10_000:
                buckets["1K-10K"] += 1
            elif ln < 100_000:
                buckets["10K-100K"] += 1
            else:
                buckets[">100K"] += 1
        print("    Size distribution of NaN gaps:")
        for label, count in buckets.items():
            if count:
                print(f"      {label:>8s} : {count}")

        # Show top-5 largest gaps with genomic coords
        top5 = sorted(nan_runs, key=lambda x: -x[2])[:5]
        print("    Top-5 largest NaN gaps (genomic coords):")
        for s, e, ln in top5:
            print(f"      {start + s + 1:>12,} - {start + e:>12,}  ({ln:,} bp)")

    print()


# ── entropy statistics ───────────────────────────────────────────────────────

def entropy_stats(entropy):
    """Print distribution statistics on scored (non-NaN) positions."""
    valid = entropy[~np.isnan(entropy)]
    if len(valid) == 0:
        print("No scored positions to summarise.\n")
        return

    print("=" * 70)
    print("ENTROPY DISTRIBUTION (scored positions only)")
    print("=" * 70)
    print(f"  Count : {len(valid):,}")
    print(f"  Mean  : {np.mean(valid):.6f} nats")
    print(f"  Std   : {np.std(valid):.6f}")
    print(f"  Min   : {np.min(valid):.6f}")
    print(f"  5th % : {np.percentile(valid, 5):.6f}")
    print(f"  25th %: {np.percentile(valid, 25):.6f}")
    print(f"  Median: {np.median(valid):.6f}")
    print(f"  75th %: {np.percentile(valid, 75):.6f}")
    print(f"  95th %: {np.percentile(valid, 95):.6f}")
    print(f"  Max   : {np.max(valid):.6f}")
    print()


# ── region analysis ──────────────────────────────────────────────────────────

def region_report(regions, chrom, start):
    """Summarise detected drop regions."""
    print("=" * 70)
    print("DETECTED DROP REGIONS")
    print("=" * 70)

    if not regions:
        print("  No regions detected.\n")
        return

    methods = Counter(r["method"] for r in regions)
    print(f"  Total regions : {len(regions)}")
    for m, c in sorted(methods.items()):
        print(f"    {m:>8s} : {c}")

    lengths = [r["region_length"] for r in regions]
    confidences = [r["start_confidence"] for r in regions]
    mean_ents = [r["mean_entropy"] for r in regions]

    print(f"\n  Region length (bp):")
    print(f"    Min    : {min(lengths):,}")
    print(f"    Median : {int(np.median(lengths)):,}")
    print(f"    Mean   : {np.mean(lengths):,.0f}")
    print(f"    Max    : {max(lengths):,}")

    print(f"\n  Start confidence (|score|):")
    print(f"    Min    : {min(confidences):.3f}")
    print(f"    Median : {np.median(confidences):.3f}")
    print(f"    Max    : {max(confidences):.3f}")

    print(f"\n  Mean entropy in regions:")
    print(f"    Min    : {min(mean_ents):.6f}")
    print(f"    Median : {np.median(mean_ents):.6f}")
    print(f"    Max    : {max(mean_ents):.6f}")

    # Check method agreement: regions found by BOTH zscore and mad
    zscore_intervals = set()
    mad_intervals = set()
    for r in regions:
        iv = (r["genomic_start"], r["genomic_end"])
        if r["method"] == "zscore":
            zscore_intervals.add(iv)
        else:
            mad_intervals.add(iv)

    # Overlap check (any positional overlap counts)
    def _overlaps(a, b):
        return a[0] < b[1] and b[0] < a[1]

    n_both = 0
    for ziv in zscore_intervals:
        for miv in mad_intervals:
            if _overlaps(ziv, miv):
                n_both += 1
                break

    if zscore_intervals and mad_intervals:
        print(f"\n  Method agreement:")
        print(f"    zscore-only regions : {len(zscore_intervals)}")
        print(f"    MAD-only regions    : {len(mad_intervals)}")
        print(f"    zscore regions overlapping a MAD region: {n_both}")

    # Top-10 most confident regions
    top_n = sorted(regions, key=lambda r: -r["start_confidence"])[:10]
    print(f"\n  Top-10 regions by confidence:")
    print(f"    {'genomic_start':>14s}  {'genomic_end':>12s}  {'length':>7s}  "
          f"{'method':>7s}  {'confidence':>10s}  {'mean_H':>8s}")
    for r in top_n:
        print(f"    {r['genomic_start']:>14,}  {r['genomic_end']:>12,}  "
              f"{r['region_length']:>7,}  {r['method']:>7s}  "
              f"{r['start_confidence']:>10.3f}  {r['mean_entropy']:>8.5f}")

    print()


# ── BED export (for genome browser) ─────────────────────────────────────────

def export_bed(regions, prefix):
    """Write a minimal BED file of detected regions."""
    bed_path = _out(prefix, "regions.bed")
    with open(bed_path, "w") as f:
        for r in sorted(regions, key=lambda x: x["genomic_start"]):
            # BED is 0-based half-open; genomic coords in TSV are 1-based
            bed_start = r["genomic_start"] - 1
            bed_end = r["genomic_end"] - 1
            name = f'{r["method"]}_conf{r["start_confidence"]:.2f}'
            score = int(min(r["start_confidence"] * 100, 1000))
            f.write(f'{r["chrom"]}\t{bed_start}\t{bed_end}\t{name}\t{score}\n')
    print(f"  BED file written: {bed_path}")
    print(f"  ({len(regions)} regions, for loading in IGV / UCSC browser)\n")


# ── plotting ─────────────────────────────────────────────────────────────────

def _get_mpl():
    """Import and configure matplotlib for headless rendering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches


def plot_entropy_profile(entropy, regions, chrom, start, end, prefix,
                         plot_start=None, plot_end=None, gene_features=None):
    """Plot entropy along the chromosome with detected regions highlighted."""
    logger.info("Starting entropy profile plot...")
    try:
        plt, mpatches = _get_mpl()
    except ImportError:
        logger.warning("matplotlib not available -- skipping plots.")
        return

    # Determine window
    if plot_start is not None or plot_end is not None:
        gs = plot_start if plot_start is not None else start
        ge = plot_end if plot_end is not None else end
        # Convert genomic coords to array indices
        i0 = max(gs - start, 0)
        i1 = min(ge - start, len(entropy))
    else:
        gs, ge = start, end
        i0, i1 = 0, len(entropy)

    ent_slice = entropy[i0:i1]
    positions = np.arange(i0, i1) + start  # genomic positions (0-based)

    n_panels = 4 if gene_features else 3
    ratios = [3, 1, 1, 0.8] if gene_features else [3, 1, 1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 10 + (2 if gene_features else 0)),
                             sharex=True,
                             gridspec_kw={"height_ratios": ratios})

    # --- Panel 1: entropy trace ---
    ax = axes[0]
    # Downsample for plotting if very large
    step = max(1, len(ent_slice) // 50_000)
    xp = positions[::step]
    yp = ent_slice[::step]
    ax.plot(xp, yp, linewidth=0.3, color="steelblue", alpha=0.7)

    # Overlay smoothed
    w = min(501, len(ent_slice) // 10)
    if w > 1 and np.count_nonzero(~np.isnan(ent_slice)) > w:
        sm = _rolling_mean(ent_slice, w)
        ax.plot(positions[::step], sm[::step], linewidth=0.8, color="black",
                label=f"smoothed (w={w})")

    # Shade detected regions — flat alpha, batch rendering via ax.broken_barh
    colors = {"zscore": "red", "mad": "#2ecc71"}
    flat_alpha = 0.18

    # Group regions by method for batch rendering
    region_groups = {}
    for r in regions:
        rg_s = r["genomic_start"] - 1  # to 0-based
        rg_e = r["genomic_end"] - 1
        if rg_e < gs or rg_s > ge:
            continue
        method = r["method"]
        if method not in region_groups:
            region_groups[method] = []
        region_groups[method].append((rg_s, rg_e - rg_s))  # (xstart, xwidth)

    y_lo, y_hi = ax.get_ylim() if ax.get_ylim()[1] > ax.get_ylim()[0] else (0, 1.5)
    for method, bars in region_groups.items():
        c = colors.get(method, "gray")
        ax.broken_barh(bars, (y_lo, y_hi - y_lo),
                       facecolors=c, alpha=flat_alpha, edgecolors="none")
    ax.set_ylim(y_lo, y_hi)

    ax.set_ylabel("Entropy (nats)")
    ax.set_title(f"{chrom}  {gs:,}-{ge:,}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- Panel 2: coverage track ---
    ax = axes[1]
    scored = (~np.isnan(ent_slice)).astype(np.float32)
    # bin into 1-kb windows for visibility
    bin_w = max(1, len(scored) // 2000)
    n_bins = len(scored) // bin_w
    if n_bins > 0:
        binned = scored[:n_bins * bin_w].reshape(n_bins, bin_w).mean(axis=1)
        bin_pos = np.linspace(gs, ge, n_bins)
        ax.fill_between(bin_pos, 0, binned, step="mid", color="green", alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")
    ax.grid(True, alpha=0.2)

    # --- Panel 3: region density ---
    ax = axes[2]
    region_density = np.zeros(i1 - i0, dtype=np.float32)
    for r in regions:
        rg_s = max(r["genomic_start"] - 1 - start, i0) - i0
        rg_e = min(r["genomic_end"] - 1 - start, i1) - i0
        if rg_e > rg_s:
            region_density[rg_s:rg_e] += 1
    if n_bins > 0:
        rd_binned = region_density[:n_bins * bin_w].reshape(n_bins, bin_w).max(axis=1)
        ax.fill_between(bin_pos, 0, rd_binned, step="mid", color="purple", alpha=0.5)
    ax.set_ylabel("Regions")
    if not gene_features:
        ax.set_xlabel(f"Genomic position ({chrom})")
    ax.grid(True, alpha=0.2)

    # --- Panel 4: gene density track (optional) ---
    if gene_features:
        ax = axes[3]
        genes_only = [f for f in gene_features if f["feature_type"] == "gene"]
        gene_density = np.zeros(i1 - i0, dtype=np.float32)
        for g in genes_only:
            gs_idx = max(g["start"] - start, i0) - i0
            ge_idx = min(g["end_exclusive"] - start, i1) - i0
            if ge_idx > gs_idx:
                gene_density[gs_idx:ge_idx] += 1
        if n_bins > 0:
            gd_binned = gene_density[:n_bins * bin_w].reshape(n_bins, bin_w).max(axis=1)
            ax.fill_between(bin_pos, 0, gd_binned, step="mid",
                            color="#2ecc71", alpha=0.6)
        ax.set_ylabel("Genes")
        ax.set_xlabel(f"Genomic position ({chrom})")
        ax.grid(True, alpha=0.2)

    # Legend — placed outside plot to avoid overlapping content
    from matplotlib.lines import Line2D
    patches = [
        mpatches.Patch(color="red", alpha=0.3, label="Z-score region"),
        mpatches.Patch(color="#2ecc71", alpha=0.3, label="MAD region"),
        Line2D([], [], linestyle=":", color="red", linewidth=1.2, label="Z-score boundary"),
        Line2D([], [], linestyle="--", color="#2ecc71", linewidth=1.2, label="MAD boundary"),
    ]
    axes[0].legend(handles=patches, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out_png = _out(prefix, "analysis.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved: %s", out_png)


def plot_transitions(entropy, scored_drops, scored_rises, chrom, start,
                     prefix, smooth_w=51, annotate_top_n=5, max_markers=20,
                     gene_features=None):
    """Plot drop/rise transitions with confidence-based marker sizing.

    Ported from genome_scoring_jan26_drops.py plot_suite() (Plot 4).
    RED downward triangles for drops, BLUE upward triangles for rises.
    Marker size and color intensity encode confidence score.
    """
    logger.info("Starting transition plots...")
    try:
        plt, _ = _get_mpl()
    except ImportError:
        logger.warning("matplotlib not available -- skipping transition plots.")
        return

    sm = _rolling_mean(entropy, smooth_w)
    # Heavier smoothing just for the background plot line
    plot_w = max(501, smooth_w * 10)
    sm_plot = _rolling_mean(entropy, plot_w)
    L = len(entropy)

    all_methods = set(scored_drops.keys()) | set(scored_rises.keys())
    if not all_methods:
        logger.warning("No drops/rises data for transition plots.")
        return

    for method in sorted(all_methods):
        fig, ax = plt.subplots(figsize=(16, 4.5))

        # Downsample smoothed trace for plotting
        step = max(1, L // 10_000)
        x_ds = np.arange(0, L, step)
        ax.plot(x_ds, sm_plot[::step], linewidth=1.0, label="Smoothed entropy",
                color='gray', alpha=0.7)

        # Gene shading (if provided)
        if gene_features:
            shade_genes(ax, gene_features)

        has_drops = False
        has_rises = False

        # --- Drops (RED gradient) ---
        if method in scored_drops and scored_drops[method]:
            drops = scored_drops[method]
            # Keep only top max_markers by absolute score
            drops = sorted(drops, key=lambda x: abs(x[1]), reverse=True)[:max_markers]
            has_drops = True
            positions = [p for p, _ in drops]
            scores = [abs(s) for _, s in drops]
            ys = sm[positions]

            # Normalize for sizing
            if len(scores) > 1:
                smin, smax = min(scores), max(scores)
                srange = smax - smin if smax > smin else 1.0
                norm = [(s - smin) / srange for s in scores]
            else:
                norm = [1.0]
            sizes = [40 + 160 * n for n in norm]

            scatter_d = ax.scatter(
                positions, ys, s=sizes, c=scores, cmap='Reds',
                alpha=0.8, edgecolors='black', linewidths=0.5,
                label=f"drops:{method}", marker='v',
                vmin=min(scores), vmax=max(scores),
            )

            # Annotate top N
            if annotate_top_n > 0:
                top = sorted(drops, key=lambda x: abs(x[1]), reverse=True)
                for rank, (pos, score) in enumerate(top[:annotate_top_n], 1):
                    gpos = start + pos + 1
                    ax.annotate(
                        f'D{rank}: {gpos:,}',
                        xy=(pos, sm[pos]), xytext=(0, -25),
                        textcoords='offset points', fontsize=7,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#ffcccc',
                                  alpha=0.9, edgecolor='#cc0000', linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', lw=0.8, color='#cc0000'),
                    )

        # --- Rises (BLUE gradient) ---
        if method in scored_rises and scored_rises[method]:
            rises = scored_rises[method]
            # Keep only top max_markers by absolute score
            rises = sorted(rises, key=lambda x: abs(x[1]), reverse=True)[:max_markers]
            has_rises = True
            positions = [p for p, _ in rises]
            scores = [abs(s) for _, s in rises]
            ys = sm[positions]

            if len(scores) > 1:
                smin, smax = min(scores), max(scores)
                srange = smax - smin if smax > smin else 1.0
                norm = [(s - smin) / srange for s in scores]
            else:
                norm = [1.0]
            sizes = [40 + 160 * n for n in norm]

            scatter_r = ax.scatter(
                positions, ys, s=sizes, c=scores, cmap='Blues',
                alpha=0.8, edgecolors='black', linewidths=0.5,
                label=f"rises:{method}", marker='^',
                vmin=min(scores), vmax=max(scores),
            )

            if annotate_top_n > 0:
                top = sorted(rises, key=lambda x: abs(x[1]), reverse=True)
                for rank, (pos, score) in enumerate(top[:annotate_top_n], 1):
                    gpos = start + pos + 1
                    ax.annotate(
                        f'R{rank}: {gpos:,}',
                        xy=(pos, sm[pos]), xytext=(0, 25),
                        textcoords='offset points', fontsize=7,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#cce5ff',
                                  alpha=0.9, edgecolor='#0066cc', linewidth=0.5),
                        arrowprops=dict(arrowstyle='->', lw=0.8, color='#0066cc'),
                    )

        # Colorbars
        if has_drops:
            cbar = plt.colorbar(scatter_d, ax=ax, pad=0.02)
            cbar.set_label('Drop Score', rotation=270, labelpad=15, fontsize=9)
        if has_rises:
            cbar = plt.colorbar(scatter_r, ax=ax,
                                pad=0.08 if has_drops else 0.02)
            cbar.set_label('Rise Score', rotation=270, labelpad=15, fontsize=9)

        # Styling
        if has_drops and has_rises:
            title = f"{chrom} | Drop & Rise Detection: {method}"
        elif has_rises:
            title = f"{chrom} | Rise Detection: {method}"
        else:
            title = f"{chrom} | Drop Detection: {method}"

        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
        ax.set_xlabel("Position (bp)", fontsize=12, color='#2c3e50')
        ax.set_ylabel("Entropy (nats)", fontsize=12, color='#2c3e50')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='-', alpha=0.2)
        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        out_png = _out(prefix, f"transitions_{method}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Transition plot saved: %s", out_png)



def _plot_window(ax, sm, all_regions, lo, hi, start, chrom,
                 title, mpatches, highlight_region=None,
                 gene_features=None, ax_annot=None):
    """Draw a zoomed window showing smoothed entropy and ALL detected regions.

    Parameters:
        ax: matplotlib axes
        sm: smoothed entropy array (full chromosome)
        all_regions: list of ALL region dicts
        lo, hi: array indices defining the window
        start: genomic offset (array index 0 = genomic position `start`)
        chrom: chromosome name
        title: plot title string
        highlight_region: optional region dict to highlight as the "trigger"
                          region (drawn with gold border + darker fill)
        gene_features: optional list of annotation feature dicts
        ax_annot: optional separate axes for the annotation track
    """
    xx = np.arange(lo, hi)
    ax.plot(xx, sm[lo:hi], linewidth=1.3, color='#3498db',
            label="Smoothed entropy")
    ax.fill_between(xx, sm[lo:hi], alpha=0.15, color='#3498db')

    # Find and highlight ALL regions overlapping [lo, hi]
    # Flat color per method + text label with actual confidence value
    colors = {"zscore": "#E74C3C", "mad": "#2ecc71"}
    linestyles = {"zscore": ":", "mad": "--"}
    flat_alpha = 0.18
    n_overlapping = 0
    seen_methods = set()
    y_lo_text, y_hi_text = ax.get_ylim() if ax.get_ylim()[1] > ax.get_ylim()[0] else (0, 1.5)
    label_y_offset = 0  # alternates 0/1 to stagger labels vertically
    for r in all_regions:
        rs, re = r["drop_start"], r["drop_end"]
        if re < lo or rs > hi:
            continue
        # Skip the highlight region here — we draw it separately below
        if (highlight_region is not None
                and rs == highlight_region["drop_start"]
                and re == highlight_region["drop_end"]
                and r["method"] == highlight_region["method"]):
            n_overlapping += 1
            seen_methods.add(r["method"])
            continue
        n_overlapping += 1
        method = r["method"]
        seen_methods.add(method)
        c = colors.get(method, "gray")
        ls = linestyles.get(method, "-")
        conf = r.get("start_confidence", 0)
        ax.axvspan(max(rs, lo), min(re, hi), alpha=flat_alpha, color=c)
        # Boundary lines with method-specific line styles
        if lo <= rs <= hi:
            ax.axvline(rs, linestyle=ls, linewidth=0.9, alpha=0.7, color=c)
        if lo <= re <= hi:
            ax.axvline(re, linestyle=ls, linewidth=0.9, alpha=0.7, color=c)
        # Confidence value text label at top of region
        mid_x = (max(rs, lo) + min(re, hi)) / 2
        # Stagger labels to reduce overlap
        label_y_frac = 0.95 if label_y_offset % 2 == 0 else 0.85
        label_y_offset += 1
        ax.text(mid_x, label_y_frac, f"{conf:.1f}",
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=5.5, fontweight='bold',
                color=c, alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=c, alpha=0.7, linewidth=0.5))

    # Draw the highlight (trigger) region on top with distinct style
    if highlight_region is not None:
        hrs = highlight_region["drop_start"]
        hre = highlight_region["drop_end"]
        h_conf = highlight_region.get("start_confidence", 0)
        ax.axvspan(max(hrs, lo), min(hre, hi), alpha=0.30,
                   color='#FFD700', edgecolor='#B8860B', linewidth=2.5,
                   label="Selected region")
        if lo <= hrs <= hi:
            ax.axvline(hrs, linestyle='--', linewidth=2.0, alpha=0.9,
                       color='#B8860B')
        if lo <= hre <= hi:
            ax.axvline(hre, linestyle=':', linewidth=2.0, alpha=0.9,
                       color='#B8860B')
        # Confidence label for highlight region
        h_mid = (max(hrs, lo) + min(hre, hi)) / 2
        ax.text(h_mid, 0.95, f"{h_conf:.1f}",
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, fontweight='bold',
                color='#B8860B',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFD700',
                          edgecolor='#B8860B', alpha=0.85, linewidth=1.0))

    # Legend — simplified, no opacity tiers
    from matplotlib.lines import Line2D
    patches = []
    if highlight_region is not None:
        patches.append(mpatches.Patch(
            color='#FFD700', alpha=0.5, edgecolor='#B8860B', linewidth=1.5,
            label="Selected region"))
    for m in sorted(seen_methods):
        c = colors.get(m, "gray")
        ls = linestyles.get(m, "-")
        label = "Z-score" if m == "zscore" else "MAD"
        patches.append(mpatches.Patch(color=c, alpha=0.3, label=f"{label} region"))
        patches.append(Line2D([], [], linestyle=ls, color=c, linewidth=1.2,
                               label=f"{label} boundary"))
    patches.append(Line2D([], [], marker='', linestyle='', label=''))
    patches.append(Line2D([], [], marker='', linestyle='',
                          label='Numbers = confidence score'))
    if patches:
        ax.legend(handles=patches, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.9)

    glo = start + lo + 1
    ghi = start + hi
    ax.set_title(
        f"{title}  |  {chrom} {glo:,}-{ghi:,}  |  "
        f"{n_overlapping} regions in window",
        fontsize=11, fontweight='bold', color='#2c3e50', pad=10,
    )
    ax.set_ylabel("Entropy (nats)", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.2)

    # Gene annotation track (if provided)
    if gene_features and ax_annot is not None:
        view_start = start + lo
        view_end = start + hi
        draw_gene_track(ax_annot, gene_features, view_start, view_end)
        ax_annot.set_xlabel("Position (bp)", fontsize=11)
    else:
        ax.set_xlabel("Position (bp)", fontsize=11)


def plot_zoom_regions(entropy, regions, chrom, start, prefix,
                      zoom_bp=5000, max_zoom_plots=20, smooth_w=51,
                      gene_features=None):
    """Generate zoomed plots around top-confidence detected drop regions.

    Shows ALL detected regions within each window, not just the trigger region.
    """
    logger.info("Starting zoom region plots...")
    try:
        plt, mpatches = _get_mpl()
    except ImportError:
        logger.warning("matplotlib not available -- skipping zoom plots.")
        return

    if not regions:
        logger.warning("No regions for zoom plots.")
        return

    sm = _rolling_mean(entropy, smooth_w)
    L = len(entropy)

    sorted_regions = sorted(regions, key=lambda r: -r["start_confidence"])
    plot_regions = sorted_regions[:max_zoom_plots]

    zoom_dir = _out(prefix, "zoom_plots") if _out.directory else f"{prefix}_zoom_plots"
    os.makedirs(zoom_dir, exist_ok=True)

    for idx, r in enumerate(plot_regions):
        center = (r["drop_start"] + r["drop_end"]) // 2
        lo = max(0, center - zoom_bp)
        hi = min(L, center + zoom_bp)

        ax_annot = None
        if gene_features:
            fig, (ax, ax_annot) = plt.subplots(
                2, 1, figsize=(14, 5.5), sharex=True,
                gridspec_kw={"height_ratios": [4, 1]})
        else:
            fig, ax = plt.subplots(figsize=(14, 4.5))
        _plot_window(ax, sm, regions, lo, hi, start, chrom,
                     f"Top #{idx+1} (conf={r['start_confidence']:.2f})",
                     mpatches, highlight_region=r,
                     gene_features=gene_features, ax_annot=ax_annot)
        plt.tight_layout()
        out_png = os.path.join(
            zoom_dir,
            f"zoom_{idx+1:03d}_{r['method']}_{r['drop_start']}.png"
        )
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Zoom plots saved: %s/ (%d plots)", zoom_dir, len(plot_regions))


def plot_random_regions(entropy, regions, chrom, start, prefix,
                        zoom_bp=5000, n_random=20, smooth_w=51, seed=42,
                        gene_features=None):
    """Generate zoomed plots at randomly sampled positions along the chromosome.

    Each plot shows a window of +/-zoom_bp around a random scored position,
    with ALL detected regions within that window highlighted.
    This gives an unbiased view of what the entropy landscape looks like.
    """
    logger.info("Starting random region plots...")
    try:
        plt, mpatches = _get_mpl()
    except ImportError:
        logger.warning("matplotlib not available -- skipping random region plots.")
        return

    sm = _rolling_mean(entropy, smooth_w)
    L = len(entropy)

    # Sample from scored (non-NaN) positions, spaced apart to avoid overlap
    scored_mask = ~np.isnan(entropy)
    scored_indices = np.where(scored_mask)[0]
    if len(scored_indices) == 0:
        logger.warning("No scored positions for random sampling.")
        return

    rng = np.random.RandomState(seed)
    # Space samples at least 2*zoom_bp apart so windows don't overlap
    min_spacing = 2 * zoom_bp
    candidates = []
    for idx in rng.permutation(scored_indices):
        if all(abs(int(idx) - c) >= min_spacing for c in candidates):
            candidates.append(int(idx))
        if len(candidates) >= n_random:
            break
    candidates.sort()

    rand_dir = _out(prefix, "random_plots") if _out.directory else f"{prefix}_random_plots"
    os.makedirs(rand_dir, exist_ok=True)

    for idx, center in enumerate(candidates):
        lo = max(0, center - zoom_bp)
        hi = min(L, center + zoom_bp)

        ax_annot = None
        if gene_features:
            fig, (ax, ax_annot) = plt.subplots(
                2, 1, figsize=(14, 5.5), sharex=True,
                gridspec_kw={"height_ratios": [4, 1]})
        else:
            fig, ax = plt.subplots(figsize=(14, 4.5))
        gpos = start + center + 1
        _plot_window(ax, sm, regions, lo, hi, start, chrom,
                     f"Random #{idx+1} (pos {gpos:,})",
                     mpatches, gene_features=gene_features,
                     ax_annot=ax_annot)
        plt.tight_layout()
        out_png = os.path.join(
            rand_dir,
            f"random_{idx+1:03d}_pos{center}.png"
        )
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Random region plots saved: %s/ (%d plots)", rand_dir, len(candidates))


def plot_interactive_html(entropy, regions, chrom, start, prefix, smooth_w=51,
                          gene_features=None):
    """Generate interactive Plotly HTML plot of entropy with detected regions.

    Requires plotly. Falls back gracefully if not installed.
    """
    logger.info("Starting interactive HTML plot...")
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not available -- skipping interactive HTML plot. "
                       "Install with: pip install plotly")
        return

    sm = _rolling_mean(entropy, smooth_w)
    L = len(entropy)

    # Downsample for interactivity (plotly handles ~500K points well)
    step = max(1, L // 500_000)
    x = np.arange(0, L, step) + start  # genomic coords
    y_raw = entropy[::step]
    y_sm = sm[::step]

    fig = go.Figure()

    # Raw entropy (thin, transparent)
    fig.add_trace(go.Scattergl(
        x=x, y=y_raw, mode='lines',
        line=dict(width=0.5, color='steelblue'),
        opacity=0.4, name='Raw entropy',
    ))

    # Smoothed entropy
    fig.add_trace(go.Scattergl(
        x=x, y=y_sm, mode='lines',
        line=dict(width=1.5, color='black'),
        name=f'Smoothed (w={smooth_w})',
    ))

    # Highlight detected regions as shapes
    for r in regions:
        gstart = r["genomic_start"] - 1
        gend = r["genomic_end"] - 1
        color = 'rgba(255,0,0,0.1)' if r["method"] == "zscore" else 'rgba(46,204,113,0.1)'
        fig.add_vrect(
            x0=gstart, x1=gend,
            fillcolor=color, line_width=0,
            annotation_text=r["method"],
            annotation_position="top left",
            annotation_font_size=8,
        )

    # Highlight gene regions as translucent green vrects
    if gene_features:
        genes_only = [f for f in gene_features if f["feature_type"] == "gene"]
        for g in genes_only:
            label = g["name"] or "gene"
            fig.add_vrect(
                x0=g["start"], x1=g["end_exclusive"],
                fillcolor='rgba(46,204,113,0.08)', line_width=0,
                annotation_text=label,
                annotation_position="bottom left",
                annotation_font_size=7,
                annotation_font_color='#27ae60',
            )

    fig.update_layout(
        title=f"{chrom} | Entropy Profile (interactive)",
        xaxis_title="Genomic Position",
        yaxis_title="Entropy (nats)",
        template="plotly_white",
        height=500,
        showlegend=True,
    )

    out_html = _out(prefix, "interactive.html")
    fig.write_html(out_html)
    logger.info("Interactive HTML plot saved: %s", out_html)


def plot_chromosome_dashboard(entropy, regions, chrom, start, end, prefix,
                               bin_size=50_000):
    """Generate a chromosome-wide dashboard with binned entropy overview.

    Produces an 8-panel figure aggregating data into fixed-width genomic bins
    for a readable chromosome-scale view.

    Output: {prefix}.dashboard.png at 200 DPI.
    """
    logger.info("Starting chromosome dashboard...")
    try:
        plt, mpatches = _get_mpl()
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import ListedColormap
    except ImportError:
        logger.warning("matplotlib not available -- skipping dashboard.")
        return

    # ── bin the data ──
    bd = _bin_entropy(entropy, regions, bin_size=bin_size)
    bin_centers_genomic = bd["bin_centers"] + start  # convert to genomic coords
    n_bins = bd["n_bins"]

    # Styling constants
    TITLE_COLOR = '#2c3e50'
    ZSCORE_COLOR = 'red'
    MAD_COLOR = 'blue'

    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Entropy heatmap (row 0, full width) ──
    ax_a = fig.add_subplot(gs[0, :])
    heatmap_data = bd["mean_entropy"].reshape(1, -1).copy()
    # Use a masked array so NaN bins render as gray
    masked = np.ma.masked_invalid(heatmap_data)
    cmap_heat = plt.cm.YlOrRd.copy()
    cmap_heat.set_bad(color='#cccccc')
    im = ax_a.imshow(masked, aspect='auto', cmap=cmap_heat,
                     extent=[start, start + n_bins * bin_size, 0, 1],
                     interpolation='nearest')
    ax_a.set_yticks([])
    ax_a.set_xlabel(f"Genomic position ({chrom})", fontsize=11)
    ax_a.set_title("A. Binned Mean Entropy Heatmap", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    cbar = fig.colorbar(im, ax=ax_a, orientation='horizontal',
                        fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_label("Mean entropy (nats)", fontsize=10)

    # ── Panel B: Entropy profile with min/max band (row 1, full width) ──
    ax_b = fig.add_subplot(gs[1, :])
    ax_b.fill_between(bin_centers_genomic, bd["min_entropy"],
                      bd["max_entropy"], alpha=0.2, color='steelblue',
                      label='Min-max range')
    ax_b.plot(bin_centers_genomic, bd["mean_entropy"], linewidth=0.8,
              color='steelblue', label='Mean entropy')

    # Tick marks at bins containing detected regions
    region_bins_z = np.where(bd["n_regions_zscore"] > 0)[0]
    region_bins_m = np.where(bd["n_regions_mad"] > 0)[0]
    y_bottom = np.nanmin(bd["min_entropy"][bd["coverage_frac"] > 0]) if \
        np.any(bd["coverage_frac"] > 0) else 0
    tick_y = y_bottom - 0.02 * (np.nanmax(bd["mean_entropy"]) - y_bottom + 1e-9)
    if len(region_bins_z) > 0:
        ax_b.scatter(bin_centers_genomic[region_bins_z],
                     np.full(len(region_bins_z), tick_y),
                     marker='|', color=ZSCORE_COLOR, s=30, linewidths=0.8,
                     label='zscore regions', zorder=5)
    if len(region_bins_m) > 0:
        ax_b.scatter(bin_centers_genomic[region_bins_m],
                     np.full(len(region_bins_m), tick_y),
                     marker='|', color=MAD_COLOR, s=30, linewidths=0.8,
                     label='MAD regions', zorder=5)

    ax_b.set_ylabel("Entropy (nats)", fontsize=11)
    ax_b.set_xlabel(f"Genomic position ({chrom})", fontsize=11)
    ax_b.set_title("B. Binned Entropy Profile", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_b.legend(loc='upper right', fontsize=8)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.2)

    # ── Panel C: Entropy histogram (row 2, col 0) ──
    ax_c = fig.add_subplot(gs[2, 0])
    valid = entropy[~np.isnan(entropy)]
    if len(valid) > 0:
        ax_c.hist(valid, bins=100, color='steelblue', alpha=0.7,
                  edgecolor='white', linewidth=0.3)
    ax_c.set_xlabel("Entropy (nats)", fontsize=11)
    ax_c.set_ylabel("Count", fontsize=11)
    ax_c.set_title("C. Entropy Distribution", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, alpha=0.2)

    # ── Panel D: Region length distribution (row 2, col 1) ──
    ax_d = fig.add_subplot(gs[2, 1])
    zscore_lengths = [r["region_length"] for r in regions
                      if r["method"] == "zscore"]
    mad_lengths = [r["region_length"] for r in regions
                   if r["method"] == "mad"]
    if zscore_lengths or mad_lengths:
        all_lengths = zscore_lengths + mad_lengths
        bins_d = np.linspace(0, max(all_lengths) * 1.05, 40)
        if zscore_lengths:
            ax_d.hist(zscore_lengths, bins=bins_d, color=ZSCORE_COLOR,
                      alpha=0.5, label='zscore', edgecolor='white',
                      linewidth=0.3)
        if mad_lengths:
            ax_d.hist(mad_lengths, bins=bins_d, color=MAD_COLOR,
                      alpha=0.5, label='MAD', edgecolor='white',
                      linewidth=0.3)
        ax_d.legend(fontsize=9)
    ax_d.set_xlabel("Region length (bp)", fontsize=11)
    ax_d.set_ylabel("Count", fontsize=11)
    ax_d.set_title("D. Region Length Distribution", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, alpha=0.2)

    # ── Panel E: Confidence distribution (row 2, col 2) ──
    ax_e = fig.add_subplot(gs[2, 2])
    zscore_conf = [r["start_confidence"] for r in regions
                   if r["method"] == "zscore"]
    mad_conf = [r["start_confidence"] for r in regions
                if r["method"] == "mad"]
    if zscore_conf or mad_conf:
        all_conf = zscore_conf + mad_conf
        bins_e = np.linspace(0, max(all_conf) * 1.05, 40)
        if zscore_conf:
            ax_e.hist(zscore_conf, bins=bins_e, color=ZSCORE_COLOR,
                      alpha=0.5, label='zscore', edgecolor='white',
                      linewidth=0.3)
        if mad_conf:
            ax_e.hist(mad_conf, bins=bins_e, color=MAD_COLOR,
                      alpha=0.5, label='MAD', edgecolor='white',
                      linewidth=0.3)
        ax_e.legend(fontsize=9)
    ax_e.set_xlabel("Start confidence (|score|)", fontsize=11)
    ax_e.set_ylabel("Count", fontsize=11)
    ax_e.set_title("E. Confidence Distribution", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.grid(True, alpha=0.2)

    # ── Panel F: Coverage track (row 3, col 0) ──
    ax_f = fig.add_subplot(gs[3, 0])
    bar_colors = ['#27ae60' if c > 0.5 else '#e74c3c'
                  for c in bd["coverage_frac"]]
    ax_f.bar(np.arange(n_bins), bd["coverage_frac"], width=1.0,
             color=bar_colors, edgecolor='none')
    ax_f.set_ylim(0, 1.05)
    ax_f.set_xlabel(f"Bin index ({bin_size // 1000}kb bins)", fontsize=11)
    ax_f.set_ylabel("Coverage fraction", fontsize=11)
    ax_f.set_title("F. Binned Coverage", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(True, alpha=0.2)
    # Legend
    cov_patches = [mpatches.Patch(color='#27ae60', label='Scored (>50%)'),
                   mpatches.Patch(color='#e74c3c', label='Gap (<50%)')]
    ax_f.legend(handles=cov_patches, fontsize=8, loc='lower right')

    # ── Panel G: Region density stacked bar (row 3, col 1) ──
    ax_g = fig.add_subplot(gs[3, 1])
    bin_idx = np.arange(n_bins)
    ax_g.bar(bin_idx, bd["n_regions_zscore"], width=1.0,
             color=ZSCORE_COLOR, alpha=0.6, label='zscore')
    ax_g.bar(bin_idx, bd["n_regions_mad"], width=1.0,
             bottom=bd["n_regions_zscore"], color=MAD_COLOR,
             alpha=0.6, label='MAD')
    ax_g.set_xlabel(f"Bin index ({bin_size // 1000}kb bins)", fontsize=11)
    ax_g.set_ylabel("Region count", fontsize=11)
    ax_g.set_title("G. Region Density per Bin", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)
    ax_g.legend(fontsize=9)
    ax_g.spines['top'].set_visible(False)
    ax_g.spines['right'].set_visible(False)
    ax_g.grid(True, alpha=0.2)

    # ── Panel H: Summary stats text box (row 3, col 2) ──
    ax_h = fig.add_subplot(gs[3, 2])
    ax_h.axis('off')

    L = len(entropy)
    n_scored = int(np.count_nonzero(~np.isnan(entropy)))
    frac_scored = n_scored / L if L else 0
    n_zscore = sum(1 for r in regions if r["method"] == "zscore")
    n_mad = sum(1 for r in regions if r["method"] == "mad")
    mean_ent = float(np.nanmean(entropy)) if n_scored > 0 else 0.0

    stats_text = (
        f"Chromosome: {chrom}\n"
        f"Genomic range: {start:,} - {end:,}\n"
        f"Total length: {L:,} bp\n"
        f"Scored positions: {n_scored:,} ({frac_scored:.1%})\n"
        f"Bin size: {bin_size:,} bp ({n_bins} bins)\n"
        f"\n"
        f"Mean entropy: {mean_ent:.6f} nats\n"
        f"Median entropy: {float(np.nanmedian(entropy)) if n_scored > 0 else 0:.6f}\n"
        f"\n"
        f"Total regions: {len(regions)}\n"
        f"  zscore: {n_zscore}\n"
        f"  MAD: {n_mad}\n"
    )
    ax_h.text(0.05, 0.95, stats_text, transform=ax_h.transAxes,
              fontsize=12, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0',
                        edgecolor='#cccccc', alpha=0.9))
    ax_h.set_title("H. Summary Statistics", fontsize=14,
                   fontweight='bold', color=TITLE_COLOR, pad=10)

    # ── save ──
    out_png = _out(prefix, "dashboard.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Dashboard saved: %s", out_png)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyze score_chromosome.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text report
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full

  # With entropy profile plot
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full --plot

  # Full plot suite (transitions + zoom + interactive)
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full --all_plots

  # Transition plots only with custom annotation count
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full \\
      --transitions --annotate_top_n 10

  # Chromosome-wide dashboard (binned overview)
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full --dashboard

  # Dashboard with custom bin size (100kb)
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full \\
      --dashboard --dashboard_bin_size 100000

  # With GTF gene annotation overlays
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full \\
      --gtf /path/to/genomic.gtf --zoom --transitions

  # Full suite with annotations
  python tools/analyze_scoring_results.py --prefix chromosome_scores/chr22_full \\
      --gtf /path/to/genomic.gtf --all_plots
        """,
    )
    ap.add_argument("--prefix", default=None,
                    help="Output prefix used for scoring "
                         "(e.g. chromosome_scores/chr22_4gpu). "
                         "Not required when using --auto.")
    ap.add_argument("--plot", action="store_true",
                    help="Generate entropy profile plot (needs matplotlib)")
    ap.add_argument("--plot_start", type=int, default=None,
                    help="Genomic start for plot window (0-based)")
    ap.add_argument("--plot_end", type=int, default=None,
                    help="Genomic end for plot window (0-based)")
    ap.add_argument("--bed", action="store_true",
                    help="Export detected regions as a BED file")

    # Gene annotation overlays
    ap.add_argument("--gtf", default=None, metavar="PATH",
                    help="Path to GTF annotation file for gene overlays on plots")
    ap.add_argument("--gff", default=None, metavar="PATH",
                    help="Path to GFF3 annotation file (alternative to --gtf)")

    # Enhanced plot suite (ported from genome_scoring_jan26_drops.py)
    ap.add_argument("--transitions", action="store_true",
                    help="Generate transition plots with confidence-based "
                         "drop/rise markers (RED=drops, BLUE=rises)")
    ap.add_argument("--zoom", action="store_true",
                    help="Generate zoomed plots around top detected regions")
    ap.add_argument("--zoom_bp", type=int, default=5000,
                    help="Window radius for zoom plots (default: 5000)")
    ap.add_argument("--max_zoom_plots", type=int, default=20,
                    help="Maximum number of zoom plots (default: 20)")
    ap.add_argument("--random", action="store_true",
                    help="Generate zoomed plots at randomly sampled positions "
                         "showing all detected regions in each window")
    ap.add_argument("--n_random", type=int, default=20,
                    help="Number of random region plots (default: 20)")
    ap.add_argument("--interactive", action="store_true",
                    help="Generate interactive Plotly HTML plot (requires plotly)")
    ap.add_argument("--dashboard", action="store_true",
                    help="Generate chromosome-wide binned dashboard "
                         "(8-panel overview at configurable bin size)")
    ap.add_argument("--dashboard_bin_size", type=int, default=50000,
                    help="Bin size in bp for dashboard aggregation "
                         "(default: 50000)")
    ap.add_argument("--all_plots", action="store_true",
                    help="Generate all available plots")
    ap.add_argument("--smooth_w", type=int, default=51,
                    help="Smoothing window for plots (default: 51)")
    ap.add_argument("--annotate_top_n", type=int, default=5,
                    help="Number of top drops/rises to annotate (default: 5)")
    ap.add_argument("--max_markers", type=int, default=20,
                    help="Max drops/rises markers per method to plot (default: 20)")
    ap.add_argument("--outdir", default=None, metavar="DIR",
                    help="Output directory for all generated files "
                         "(default: same location as --prefix)")
    ap.add_argument("--results_dir", default="./results", metavar="DIR",
                    help="Root results directory (default: ./results)")
    ap.add_argument("--auto", action="store_true",
                    help="Auto-discover latest COMPLETED scoring run for --chrom")
    ap.add_argument("--chrom", default=None,
                    help="Chromosome name (required with --auto)")
    args = ap.parse_args()
    import time as _time
    _viz_wall_start = _time.time()
    setup_logging()

    # Validate: need either --prefix or --auto
    if not args.auto and not args.prefix:
        ap.error("--prefix is required (or use --auto --chrom <name>)")

    # Auto-discover scoring run
    _viz_run_dir = None
    if args.auto:
        if not args.chrom:
            ap.error("--chrom is required when using --auto")
        scoring_run = find_latest_completed(args.results_dir, args.chrom, "scoring")
        if scoring_run is None:
            logger.error(f"--auto: no COMPLETED scoring run for {args.chrom} "
                         f"in {args.results_dir}/{args.chrom}/scoring/")
            sys.exit(1)
        logger.info(f"--auto: using scoring run {scoring_run}")
        # Build prefix from scoring run data dir
        args.prefix = os.path.join(scoring_run, "data", "scoring")
        # Auto-construct output in visualization stage
        _viz_run_dir = build_run_dir(args.results_dir, args.chrom,
                                     "visualization", "analyze_scoring")
        args.outdir = _viz_run_dir
        write_source(_viz_run_dir,
                     scoring_run=os.path.abspath(scoring_run))

    prefix = args.prefix

    # Compute output prefix: if --outdir is set, use clean filenames
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        _out.directory = args.outdir
        out_prefix = os.path.join(args.outdir, os.path.basename(prefix))
        logger.info("Output directory: %s", args.outdir)
    else:
        _out.directory = None
        out_prefix = prefix

    # --all_plots enables everything
    if args.all_plots:
        args.plot = True
        args.transitions = True
        args.zoom = True
        args.random = True
        args.interactive = True
        args.dashboard = True
        args.bed = True

    # ── tee stdout to a report file when --all_plots is used ──
    tee = None
    if args.all_plots:
        report_path = _out(out_prefix, "analysis_report.txt")
        tee = _TeeWriter(report_path)
        sys.stdout = tee

    # ── load data ──
    print()
    try:
        entropy, chrom, start, end = _load_entropy(prefix)
    except FileNotFoundError:
        print(f"ERROR: {prefix}.entropy.npz not found. "
              f"Did score_chromosome.py finish?")
        return 1

    try:
        regions = _load_boundaries(prefix)
    except FileNotFoundError:
        regions = []
        print(f"WARNING: {prefix}.drop_boundaries.tsv not found, "
              f"skipping region analysis.")

    try:
        summary = _load_summary(prefix)
    except FileNotFoundError:
        summary = None

    # ── load gene annotations (optional) ──
    gene_features = None
    annot_path = args.gtf or args.gff
    if annot_path:
        fmt = "gtf" if args.gtf else "gff3"
        print(f"  Loading {fmt.upper()} annotations from: {annot_path}")
        print(f"  Filtering for chromosome: {chrom}, region {start:,}-{end:,}")
        gene_features = load_annotation_features(annot_path, chrom, start, end,
                                                  fmt=fmt)
        n_genes = sum(1 for f in gene_features if f["feature_type"] == "gene")
        print(f"  Loaded {len(gene_features)} features ({n_genes} genes) "
              f"in region.\n")

    # ── print run info from summary.json ──
    if summary:
        params = summary.get("parameters", {})
        timing = summary.get("timing", {})
        print("=" * 70)
        print("RUN INFO")
        print("=" * 70)
        print(f"  Timestamp      : {summary.get('timestamp', '?')}")
        print(f"  Chromosome     : {summary.get('resolved_chrom', params.get('chrom', '?'))}")
        print(f"  Sequence length: {summary.get('sequence_length', '?'):,} bp")
        print(f"  n_gpus         : {params.get('n_gpus', '?')}")
        print(f"  max_chunk_len  : {params.get('max_chunk_len', '?'):,}")
        print(f"  chunk_overlap  : {params.get('chunk_overlap', '?')}")
        print(f"  auto_chunk_size: {params.get('auto_chunk_size', '?')}")
        print(f"  rc_average     : {params.get('rc_average', '?')}")
        print(f"  compute_logprobs: {params.get('compute_logprobs', '?')}")
        if timing:
            print(f"\n  Timing:")
            print(f"    Total wall time: {timing.get('total_wall_s', '?')}s")
            print(f"    Scoring:         {timing.get('step3_scoring_s', '?')}s")
            throughput = timing.get('step3_throughput_bp_per_s')
            if throughput:
                print(f"    Throughput:      {throughput:,.0f} bp/s")
        print()

    # ── reports ──
    coverage_report(entropy, chrom, start, end)
    entropy_stats(entropy)
    region_report(regions, chrom, start)

    # ── BED export ──
    if args.bed and regions:
        export_bed(regions, out_prefix)

    # ── plots ──
    if args.plot:
        logger.info("=== Generating entropy profile plot ===")
        plot_entropy_profile(entropy, regions, chrom, start, end, out_prefix,
                             plot_start=args.plot_start,
                             plot_end=args.plot_end,
                             gene_features=gene_features)

    if args.transitions:
        logger.info("=== Generating transition plots ===")
        scored_drops, scored_rises = _load_drops_rises(prefix)
        if scored_drops or scored_rises:
            plot_transitions(entropy, scored_drops, scored_rises,
                             chrom, start, out_prefix,
                             smooth_w=args.smooth_w,
                             annotate_top_n=args.annotate_top_n,
                             max_markers=args.max_markers,
                             gene_features=gene_features)
        else:
            logger.warning("No drops/rises TSV files found -- skipping transition plots.")

    if args.zoom:
        logger.info("=== Generating zoom plots ===")
        plot_zoom_regions(entropy, regions, chrom, start, out_prefix,
                          zoom_bp=args.zoom_bp,
                          max_zoom_plots=args.max_zoom_plots,
                          smooth_w=args.smooth_w,
                          gene_features=gene_features)

    if args.random:
        logger.info("=== Generating random region plots ===")
        plot_random_regions(entropy, regions, chrom, start, out_prefix,
                            zoom_bp=args.zoom_bp,
                            n_random=args.n_random,
                            smooth_w=args.smooth_w,
                            gene_features=gene_features)

    if args.interactive:
        logger.info("=== Generating interactive HTML plot ===")
        plot_interactive_html(entropy, regions, chrom, start, out_prefix,
                              smooth_w=args.smooth_w,
                              gene_features=gene_features)

    if args.dashboard:
        logger.info("=== Generating chromosome dashboard ===")
        plot_chromosome_dashboard(entropy, regions, chrom, start, end, out_prefix,
                                  bin_size=args.dashboard_bin_size)

    # ── close tee and print summary of saved files ──
    if tee is not None:
        print("=" * 70)
        print("SAVED FILES")
        print("=" * 70)
        print(f"  Text report : {report_path}")
        print(f"  Profile plot: {_out(out_prefix, 'analysis.png')}")
        print(f"  Transitions : {_out(out_prefix, 'transitions_*.png')}")
        zd = _out(out_prefix, "zoom_plots") if _out.directory else f"{out_prefix}_zoom_plots"
        rd = _out(out_prefix, "random_plots") if _out.directory else f"{out_prefix}_random_plots"
        print(f"  Zoom plots  : {zd}/")
        print(f"  Random plots: {rd}/")
        print(f"  Interactive : {_out(out_prefix, 'interactive.html')}")
        print(f"  Dashboard   : {_out(out_prefix, 'dashboard.png')}")
        if args.bed:
            print(f"  BED file    : {_out(out_prefix, 'regions.bed')}")
        print()
        sys.stdout = tee._stdout
        tee.close()
        print(f"  Report saved: {report_path}")

    # Write COMPLETED sentinel if using organized output
    if _viz_run_dir:
        _viz_wall_time = _time.time() - _viz_wall_start
        write_completed(_viz_run_dir, "analyze_scoring_results.py", _viz_wall_time)
        logger.info(f"COMPLETED sentinel written to {_viz_run_dir}/COMPLETED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
