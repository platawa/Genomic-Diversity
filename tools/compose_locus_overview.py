#!/usr/bin/env python3
"""
compose_locus_overview.py

Render figure-ready stacked overviews for one or more loci, faithfully
preserving the original genome_scoring_jan26_drops.plot_suite() styling
(EvoDesigner blue/tan fill, green exon track at top, yellow exon highlight
bands, red 'Reds'-cmap drop scatter sized by score, top-N annotations with
red-arrow callouts, drop-score colorbar, genomic anchors at corners).

Per locus, the figure has:
  Row 0 (top):    raw entropy (left)  |  smoothed entropy (right)
  Rows 1-3:       3x2 grid of the six drop-detection methods
                  zscore, mad, derivative, cusum, window_mean_shift, local_baseline

Outputs (under --out_dir):
  HBB_overview.png
  NPS_overview.png
  HBB_NPS_combined.png
  manifest.tsv
  COMPLETED
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7,
    "figure.titlesize": 11,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _HERE)

from results_utils import write_completed
from compose_chapter4_figures import (
    load_gtf_features,
    load_locus_data,
)
from detection_methods import run_method

logger = logging.getLogger(__name__)


# Original deck order: local, derivative, cusum, zscore, win_shift, mad
METHOD_ORDER = ["local", "derivative", "cusum",
                "zscore", "win_shift", "mad"]
# Map our internal canonical names to detection_methods method-name
METHOD_INTERNAL = {
    "zscore":    "zscore",
    "mad":       "mad",
    "derivative": "derivative",
    "cusum":     "cusum",
    "win_shift": "window_mean_shift",
    "local":     "local_baseline",
}
METHOD_DISPLAY = {
    "local":      "Local baseline",
    "derivative": "Derivative",
    "cusum":      "CUSUM",
    "zscore":     "Z-score",
    "win_shift":  "Window mean shift",
    "mad":        "MAD",
}
# Safe column names for manifest TSV (lowercase, underscores)
METHOD_SLUG = {k: v.lower().replace(" ", "_").replace("-", "_")
               for k, v in METHOD_DISPLAY.items()}

# Title color / spine color, exact original values
ORIG_TITLE_COLOR = "#2c3e50"
ORIG_SPINE_COLOR = "#bdc3c7"
ORIG_GRID_COLOR  = "#bdc3c7"
ORIG_BLUE_FILL   = "#3498db"
ORIG_TAN_FILL    = "#f39c12"
ORIG_EXON_GREEN  = "#2ecc71"
ORIG_EXON_EDGE   = "#27ae60"
ORIG_INTRON_BG   = "#ecf0f1"
ORIG_INTRON_LINE = "#7f8c8d"
ORIG_INTRON_EDGE = "#bdc3c7"
ORIG_EXON_HILITE = "#ffeaa7"


# ------------------------------ data loaders --------------------------------

def compute_drops_all_methods_scored(data):
    """Run all 6 detection methods on the entropy slice, returning per-method
    [(local_pos, score), ...] tuples. Local positions are 0-based offsets
    within the plotted window."""
    raw = data["raw"]
    out = {}
    for canon, internal in METHOD_INTERNAL.items():
        try:
            calls = run_method(internal, raw)  # list of (local_pos, score)
        except Exception as exc:
            logger.warning(f"method {internal} failed: {exc}")
            out[canon] = []
            continue
        out[canon] = [(int(p), float(s)) for (p, s) in (calls or [])]
    return out


def features_to_exon_intervals(features, locus_start, plot_origin):
    """Convert load_gtf_features() output to (s, e, exon_id) in plotted-window
    coordinates (bp from plot_origin = pos[0]). Only CDS blocks are used as
    exons; UTRs ignored for the exon track to match the original style."""
    intervals = []
    for (s, e, ftype, strand, gname) in features:
        if ftype != "CDS":
            continue
        s_loc = max(0, int(s) - plot_origin)
        e_loc = max(0, int(e) - plot_origin)
        if e_loc <= s_loc:
            continue
        intervals.append((s_loc, e_loc, 0))  # exon_id placeholder
    intervals.sort(key=lambda t: t[0])
    # Number them
    return [(s, e, i + 1) for i, (s, e, _) in enumerate(intervals)]


# ------------------------------ styling primitives --------------------------

def evodesigner_fill(ax, x, y, low_quantile=0.10):
    """Blue fill everywhere + tan/orange fill where y <= low_quantile."""
    ax.fill_between(x, y, 0, alpha=0.35, color=ORIG_BLUE_FILL, lw=0)
    if np.any(~np.isnan(y)):
        thr = np.nanquantile(y, low_quantile)
        mask = y <= thr
        ax.fill_between(x, y, 0, where=mask, alpha=0.5,
                        color=ORIG_TAN_FILL, lw=0)


def draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True):
    """Replicates genome_scoring_jan26_drops.draw_exon_track."""
    if not exon_intervals:
        return
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    track_bottom = ymax - (track_height * y_range)
    track_top = ymax
    bar_h = track_top - track_bottom
    xmin, xmax = ax.get_xlim()
    # Intron background
    ax.add_patch(Rectangle(
        (xmin, track_bottom), xmax - xmin, bar_h,
        facecolor=ORIG_INTRON_BG, edgecolor=ORIG_INTRON_EDGE,
        linewidth=0.5, zorder=10))
    # Thin intron midline
    intron_y = track_bottom + bar_h / 2
    ax.plot([xmin, xmax], [intron_y, intron_y], color=ORIG_INTRON_LINE,
            linewidth=2, solid_capstyle='round', zorder=9)
    # Exon blocks
    for (s, e, exon_id) in exon_intervals:
        ax.add_patch(Rectangle(
            (s, track_bottom + bar_h * 0.1), e - s, bar_h * 0.8,
            facecolor=ORIG_EXON_GREEN, edgecolor=ORIG_EXON_EDGE,
            linewidth=0.8, zorder=11, alpha=0.9))
        if label_exons and exon_id > 0:
            mid_x = (s + e) / 2
            mid_y = track_bottom + bar_h / 2
            if (e - s) > (xmax - xmin) * 0.025:
                ax.text(mid_x, mid_y, f'E{exon_id}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white', zorder=12)
    # Yellow exon-highlight bands in plot body
    for (s, e, _) in exon_intervals:
        ax.axvspan(s, e, ymin=0, ymax=(track_bottom - ymin) / y_range,
                   facecolor=ORIG_EXON_HILITE, alpha=0.15, zorder=1)


def apply_common_styling(ax, x, *,
                         genomic_start=None,
                         ylabel="Entropy (bits)",
                         show_xlabel=True,
                         show_xticklabels=True,
                         show_anchors=True,
                         label_fontsize=9,
                         tick_fontsize=9,
                         anchor_fontsize=9):
    """Original genome_scoring_jan26_drops styling — spines, grid, ylim
    padding for the exon track, and (optional) chromosome coordinate anchors
    at the bottom corners. Title is set by the caller (we omit it here so
    composers can put titles only at column tops)."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ORIG_SPINE_COLOR)
    ax.spines["bottom"].set_color(ORIG_SPINE_COLOR)
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color=ORIG_GRID_COLOR)
    ax.xaxis.grid(False)
    if show_xlabel:
        ax.set_xlabel("Chr. pos. (bp)", fontsize=label_fontsize,
                      color=ORIG_TITLE_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize, color=ORIG_TITLE_COLOR)
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors=ORIG_TITLE_COLOR)
    ax.set_xlim(x[0], x[-1])
    if not show_xticklabels:
        ax.set_xticklabels([])
    # Pad y-axis up by 15% so the exon track has room
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin, ymax + 0.15 * y_range)
    if show_anchors and genomic_start is not None:
        genomic_end = genomic_start + int(x[-1] - x[0])
        ax.annotate(f"{genomic_start:,}",
                    xy=(0, -0.13), xycoords="axes fraction",
                    fontsize=anchor_fontsize, color=ORIG_TITLE_COLOR,
                    fontweight="bold", ha="left", va="top")
        ax.annotate(f"{genomic_end:,}",
                    xy=(1, -0.13), xycoords="axes fraction",
                    fontsize=anchor_fontsize, color=ORIG_TITLE_COLOR,
                    fontweight="bold", ha="right", va="top")


# ------------------------------ panel painters ------------------------------

def draw_raw_entropy_panel(ax, x_local, raw, exon_intervals, genomic_start, *,
                           show_xlabel=True, show_xticklabels=True,
                           show_anchors=True, ylabel="Entropy (bits)",
                           show_legend=True):
    ax.plot(x_local, raw, linewidth=0.8, label="Entropy(main)",
            color=ORIG_BLUE_FILL)
    evodesigner_fill(ax, x_local, raw, low_quantile=0.10)
    apply_common_styling(ax, x_local, genomic_start=genomic_start,
                         ylabel=ylabel,
                         show_xlabel=show_xlabel,
                         show_xticklabels=show_xticklabels,
                         show_anchors=show_anchors)
    draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=ORIG_EXON_GREEN, edgecolor=ORIG_EXON_EDGE,
                             label="Exon regions"))
        labels.append("Exon regions")
        ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.85)


def draw_smoothed_entropy_panel(ax, x_local, sm, exon_intervals, genomic_start,
                                smooth_w, *,
                                show_xlabel=True, show_xticklabels=True,
                                show_anchors=True, ylabel="Entropy (bits)",
                                show_legend=True):
    ax.plot(x_local, sm, linewidth=1.2,
            label=f"Entropy(main) rolling_mean(w={smooth_w})",
            color=ORIG_BLUE_FILL)
    evodesigner_fill(ax, x_local, sm, low_quantile=0.10)
    apply_common_styling(ax, x_local, genomic_start=genomic_start,
                         ylabel=ylabel,
                         show_xlabel=show_xlabel,
                         show_xticklabels=show_xticklabels,
                         show_anchors=show_anchors)
    draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=ORIG_EXON_GREEN, edgecolor=ORIG_EXON_EDGE,
                             label="Exon regions"))
        labels.append("Exon regions")
        ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.85)


def draw_method_panel(ax, x_local, sm, scored_drops, method_canon,
                      exon_intervals, genomic_start, *,
                      annotate_top_n=3,
                      show_xlabel=True, show_xticklabels=True,
                      show_anchors=True, ylabel="Entropy (bits)",
                      show_legend=True):
    """Replicates the per-method drop panel from plot_suite()."""
    ax.plot(x_local, sm, linewidth=1.2, label="Smoothed entropy",
            color=ORIG_BLUE_FILL, alpha=0.9)
    ax.fill_between(x_local, sm, 0, alpha=0.35, color=ORIG_BLUE_FILL, lw=0)

    # Yellow exon-highlight bands in the plot body
    if exon_intervals:
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if (ymax > ymin) else 1.0
        # Draw highlights spanning most of the y-range (track is at top)
        for (s, e, _) in exon_intervals:
            ax.axvspan(s, e, ymin=0, ymax=0.94,
                       facecolor=ORIG_EXON_HILITE, alpha=0.15, zorder=1)

    has_drops = False
    scatter_drops = None
    drops = scored_drops or []
    if drops:
        has_drops = True
        positions = np.array([p for (p, _) in drops], dtype=int)
        scores = np.array([abs(s) for (_, s) in drops], dtype=float)
        # Clip positions to the plotted range
        mask = (positions >= int(x_local[0])) & (positions <= int(x_local[-1]))
        positions = positions[mask]
        scores = scores[mask]
        if len(positions):
            ys = np.interp(positions, x_local, sm)
            if len(scores) > 1:
                smin, smax = scores.min(), scores.max()
                rng = smax - smin if smax > smin else 1.0
                norm_scores = (scores - smin) / rng
            else:
                norm_scores = np.array([1.0])
            sizes = 6 + 36 * norm_scores
            scatter_drops = ax.scatter(
                positions, ys, s=sizes, c=scores, cmap="Reds",
                alpha=0.75, edgecolors="black", linewidths=0.3,
                vmin=scores.min(), vmax=scores.max(),
                marker="o",
                label=f"drops:{METHOD_DISPLAY[method_canon]}",
                zorder=5,
            )
            # Top-N annotations
            if annotate_top_n > 0:
                order = np.argsort(scores)[::-1][:annotate_top_n]
                for rank_i, ix in enumerate(order, 1):
                    p = int(positions[ix])
                    s_ = float(scores[ix])
                    ax.annotate(
                        f"#{rank_i}: pos {p}, score {s_:.1f}",
                        xy=(p, np.interp(p, x_local, sm)),
                        xytext=(0, -22),
                        textcoords="offset points",
                        fontsize=5, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#ffcccc",
                                  alpha=0.92, edgecolor="#cc0000",
                                  linewidth=0.4),
                        arrowprops=dict(arrowstyle="->", lw=0.6,
                                        color="#cc0000"),
                        zorder=6,
                    )

    apply_common_styling(ax, x_local, genomic_start=genomic_start,
                         ylabel=ylabel,
                         show_xlabel=show_xlabel,
                         show_xticklabels=show_xticklabels,
                         show_anchors=show_anchors)
    draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=ORIG_EXON_GREEN,
                             edgecolor=ORIG_EXON_EDGE,
                             label="Exon regions"))
        labels.append("Exon regions")
        ax.legend(handles, labels, loc="lower left", fontsize=7,
                  framealpha=0.85)

    n_in_window = 0
    if drops:
        all_pos = np.array([p for (p, _) in drops])
        n_in_window = int(np.sum((all_pos >= int(x_local[0])) &
                                 (all_pos <= int(x_local[-1]))))
    return scatter_drops, n_in_window


# ------------------------------ figure composers ----------------------------

def _locus_title_prefix(row):
    """Column header is just the gene name (e.g. 'HBB')."""
    return str(row["name"])


def _prep_locus(data, locus_strand):
    """Extract per-locus arrays + metadata used by both composers."""
    pos = data["pos"]
    raw = data["raw"]
    smoothed = data["smoothed"]
    features = data["features"]
    row = data["row"].copy()
    row["strand"] = locus_strand
    locus_start_bp = int(pos[0])
    x_local = pos - locus_start_bp
    exon_intervals = features_to_exon_intervals(
        features, int(row["start"]), plot_origin=locus_start_bp,
    )
    return {
        "x_local": x_local,
        "raw": raw,
        "smoothed": smoothed,
        "exon_intervals": exon_intervals,
        "locus_start_bp": locus_start_bp,
        "title_prefix": _locus_title_prefix(row),
        "scored_drops": data.get("scored_drops", {}),
    }


def _row_label(ax, text, fontsize=12):
    """Big bold label rotated 90deg in the left margin of an axes."""
    ax.text(-0.18, 0.5, text, transform=ax.transAxes,
            rotation=90, fontsize=fontsize, fontweight="bold",
            va="center", ha="center", color=ORIG_TITLE_COLOR)


def compose_entropy_grid(loci_blocks, out_path, smooth_w=51):
    """Figure A — 2 rows (Raw, Smoothed) x N cols (one per locus).
    Tight column spacing; y-tick labels only on leftmost column;
    a single figure-level legend placed on HBB Smoothed in the empty
    upper-left region of the panel (entropy is low at the gene's 5'
    flank so there is no data overlap)."""
    from matplotlib.lines import Line2D

    n = len(loci_blocks)
    fig = plt.figure(figsize=(8.5 * n, 7.0), facecolor="white")
    gs = gridspec.GridSpec(
        2, n, figure=fig,
        left=0.045, right=0.995, top=0.93, bottom=0.10,
        hspace=0.32, wspace=0.05,
    )
    ax_for_legend = None
    for col, b in enumerate(loci_blocks):
        is_left = (col == 0)
        ylabel_raw = "Raw entropy (bits)" if is_left else None
        ylabel_sm = "Smoothed entropy (bits)" if is_left else None

        ax_raw = fig.add_subplot(gs[0, col])
        draw_raw_entropy_panel(
            ax_raw, b["x_local"], b["raw"], b["exon_intervals"],
            b["locus_start_bp"],
            show_xlabel=False, show_xticklabels=True, show_anchors=False,
            ylabel=ylabel_raw, show_legend=False,
        )
        if not is_left:
            ax_raw.tick_params(axis="y", labelleft=False)
        ax_raw.set_title(b["title_prefix"], fontsize=11, fontweight="bold",
                         color=ORIG_TITLE_COLOR, pad=4)

        ax_sm = fig.add_subplot(gs[1, col])
        draw_smoothed_entropy_panel(
            ax_sm, b["x_local"], b["smoothed"], b["exon_intervals"],
            b["locus_start_bp"], smooth_w=smooth_w,
            show_xlabel=True, show_xticklabels=True, show_anchors=True,
            ylabel=ylabel_sm, show_legend=False,
        )
        if not is_left:
            ax_sm.tick_params(axis="y", labelleft=False)
        if is_left:
            ax_for_legend = ax_sm

    # Single legend, placed inside HBB Smoothed at upper-left where the
    # entropy trace is low (the 5' flank crashes to ~0.2 bits in HBB).
    if ax_for_legend is not None:
        legend_handles = [
            Line2D([0], [0], color=ORIG_BLUE_FILL, lw=1.5, label="Entropy"),
            Patch(facecolor=ORIG_EXON_GREEN, edgecolor=ORIG_EXON_EDGE,
                  label="Exon regions"),
        ]
        ax_for_legend.legend(
            handles=legend_handles,
            loc="upper left", bbox_to_anchor=(0.015, 0.92),
            fontsize=6.5, framealpha=0.92, frameon=True,
            handlelength=1.4, handletextpad=0.4, borderpad=0.3,
        )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def compose_drops_grid(loci_blocks, out_path, annotate_top_n=3):
    """Figure B — 6 rows (one per method) x N cols (one per locus).
    Gene name is a figure-level column header above row 0 (HBB / NPS / EGFR).
    Method name is a horizontal title at the top of every panel.
    All panels show x-tick labels; only the bottom row carries the
    "Chr. pos. (bp)" label and genomic anchors."""
    n = len(loci_blocks)
    fig = plt.figure(figsize=(8.0 * n, 16.0), facecolor="white")
    gs = gridspec.GridSpec(
        6, n, figure=fig,
        left=0.05, right=0.97, top=0.945, bottom=0.045,
        hspace=0.55, wspace=0.20,
    )
    # Figure-level column headers (gene names) — placed above row 0
    n_drops_by_locus = {}
    axes_top_row = []
    for col, b in enumerate(loci_blocks):
        is_left = (col == 0)
        ylabel = "Entropy (bits)" if is_left else None
        per_method = {}
        for r, m in enumerate(METHOD_ORDER):
            is_top = (r == 0)
            is_bottom = (r == len(METHOD_ORDER) - 1)
            ax_m = fig.add_subplot(gs[r, col])
            if is_top:
                axes_top_row.append(ax_m)
            scatter, n_drops = draw_method_panel(
                ax_m, b["x_local"], b["smoothed"],
                b["scored_drops"].get(m, []),
                m, b["exon_intervals"], b["locus_start_bp"],
                annotate_top_n=annotate_top_n,
                show_xlabel=is_bottom,
                show_xticklabels=True,
                show_anchors=is_bottom,
                ylabel=ylabel,
                show_legend=is_left and is_top,
            )
            if scatter is not None:
                cbar = fig.colorbar(scatter, ax=ax_m, pad=0.010,
                                    fraction=0.020)
                cbar.set_label("Drop score (local z-score)",
                               rotation=270, labelpad=10, fontsize=7)
                cbar.ax.tick_params(labelsize=7)
            ax_m.set_title(METHOD_DISPLAY[m], fontsize=9, fontweight="bold",
                           color=ORIG_TITLE_COLOR, pad=3)
            per_method[m] = n_drops
        n_drops_by_locus[col] = per_method

    # Gene-name column headers (figure-level), centered above each column's top axis
    fig.canvas.draw()  # ensure positions are computed
    for col, ax_top in enumerate(axes_top_row):
        bbox = ax_top.get_position()
        cx = (bbox.x0 + bbox.x1) / 2
        fig.text(cx, 0.975, loci_blocks[col]["title_prefix"],
                 ha="center", va="top", fontsize=12, fontweight="bold",
                 color=ORIG_TITLE_COLOR)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")
    return n_drops_by_locus


# ------------------------------ utilities -----------------------------------

def infer_strand_from_gtf(gtf_path, chrom, start, end):
    """Best-effort strand lookup: scan GTF for a CDS overlapping [start,end]."""
    if not gtf_path or not os.path.isfile(gtf_path):
        return "+"
    from compose_chapter4_figures import load_gtf_features
    features = load_gtf_features(gtf_path, chrom, start, end, pad=0)
    for (s, e, ftype, strand, gname) in features:
        if ftype == "CDS" and strand in ("+", "-"):
            return strand
    return "+"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loci_tsv", required=True)
    p.add_argument("--loci", nargs="+", required=True)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--human_gtf", default=None)
    p.add_argument("--ecoli_gtf", default=None)
    p.add_argument("--bacillus_gtf", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pad", type=int, default=5000)
    p.add_argument("--smooth_window", type=int, default=51)
    p.add_argument("--annotate_top_n", type=int, default=0)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    t0 = time.monotonic()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = pd.read_csv(args.loci_tsv, sep="\t")
    gtf_map = {"human": args.human_gtf, "ecoli": args.ecoli_gtf,
               "bacillus": args.bacillus_gtf, "custom": None}

    blocks = []
    rows_for_manifest = []
    for name in args.loci:
        sub = manifest[manifest["name"] == name]
        if len(sub) == 0:
            logger.warning(f"{name} not in {args.loci_tsv}; skipping")
            continue
        row = sub.iloc[0]
        gtf = gtf_map.get(row["organism"])
        data = load_locus_data(row, args.results_dir, gtf, args.pad,
                               args.smooth_window)
        if data is None:
            logger.warning(f"{name}: no data; skipping")
            continue
        data["scored_drops"] = compute_drops_all_methods_scored(data)
        strand = infer_strand_from_gtf(gtf, row["chrom"],
                                       int(row["start"]), int(row["end"]))
        block = _prep_locus(data, strand)
        block["name"] = name
        block["row"] = row
        block["strand"] = strand
        blocks.append(block)

    if not blocks:
        logger.error("No loci loaded — nothing to render.")
        return

    entropy_path = os.path.join(args.out_dir, "entropy_overview.png")
    compose_entropy_grid(blocks, entropy_path, smooth_w=args.smooth_window)

    drops_path = os.path.join(args.out_dir, "drops_overview.png")
    n_drops_by_col = compose_drops_grid(
        blocks, drops_path, annotate_top_n=args.annotate_top_n,
    )

    for col, b in enumerate(blocks):
        per_method = n_drops_by_col.get(col, {})
        rows_for_manifest.append({
            "name": b["name"],
            "chrom": b["row"]["chrom"],
            "start": int(b["row"]["start"]),
            "end": int(b["row"]["end"]),
            "strand": b["strand"],
            "scoring_run_used": b["row"].get("scoring_run", "auto"),
            **{f"n_drops_{METHOD_SLUG[m]}": per_method.get(m, 0)
               for m in METHOD_ORDER},
        })

    pd.DataFrame(rows_for_manifest).to_csv(
        os.path.join(args.out_dir, "manifest.tsv"), sep="\t", index=False)
    write_completed(args.out_dir, "compose_locus_overview.py",
                    time.monotonic() - t0)
    logger.info(f"Done. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
