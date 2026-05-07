#!/usr/bin/env python3
"""
compose_chapter4_figures.py

Compose thesis-ready Ch 4 figures from per-locus outputs produced by
make_ncRNA_figures.py. Emits three files under <per_locus_dir>/composed/:

  fig_4_1_HBB.png               single-panel hero: entropy + NCBI RefSeq annotation
  fig_4_2_contrasting_loci.png  4-panel: EGFR, NPS, CRISPR Type I-E, TIGR-Tas (or stub)
  fig_4_3_six_methods_NPS.png   3x2 grid, one drop-detection method per panel

Layout convention:
  - tight 2-axis per locus: top 80% entropy (smoothed bold + raw faint),
    bottom 20% annotation ribbon (CDS green, 5'UTR blue, 3'UTR red),
    detected boundaries as vertical dashed lines
  - shared x-axis within each panel; xlabels only on bottom ribbon
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT)   # for results_utils
sys.path.insert(0, _HERE)      # for sibling tools/ modules
from results_utils import find_latest_completed
from make_ncRNA_figures import (
    resolve_scoring_run,
    load_entropy_region,
    load_drops,
    smooth,
    chrom_matches,
)

logger = logging.getLogger(__name__)


# ---------- GTF parsing with feature types (CDS, UTR, exon, intron) ----------

def load_gtf_features(gtf_path, chrom, start, end, pad):
    """Return list of (feat_start, feat_end, feat_type, strand, gene_name)
    for everything overlapping the locus window."""
    if not gtf_path or not os.path.isfile(gtf_path):
        return []
    rows = []
    lo, hi = start - pad, end + pad
    want = {"CDS", "exon", "five_prime_utr", "three_prime_utr",
            "5UTR", "3UTR", "five_prime_UTR", "three_prime_UTR"}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or not chrom_matches(parts[0], chrom):
                continue
            if parts[2] not in want:
                continue
            s, e = int(parts[3]), int(parts[4])
            if e < lo or s > hi:
                continue
            gname = None
            for field in parts[8].split(";"):
                f_ = field.strip()
                if f_.startswith("gene_name") or f_.startswith("gene") or f_.startswith("gene_id"):
                    if '"' in f_:
                        gname = f_.split('"')[1]
                    break
            # Normalize UTR type names
            ftype = parts[2]
            if ftype in ("5UTR", "five_prime_UTR"):
                ftype = "five_prime_utr"
            elif ftype in ("3UTR", "three_prime_UTR"):
                ftype = "three_prime_utr"
            rows.append((s, e, ftype, parts[6], gname))
    return rows


# ---------- drawing primitives ----------

FEATURE_COLORS = {
    "CDS": "#2ca02c",            # green
    "exon": "#888888",           # gray fallback
    "five_prime_utr": "#1f77b4", # blue
    "three_prime_utr": "#d62728",# red
}


def draw_annotation_ribbon(ax, features, start, end, pad):
    """Draw exon/CDS/UTR blocks on the given axis."""
    ax.set_xlim(start - pad, end + pad)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    # Draw in z-order: exon (background), UTR, CDS on top
    order = {"exon": 0, "five_prime_utr": 1, "three_prime_utr": 1, "CDS": 2}
    features_sorted = sorted(features, key=lambda r: order.get(r[2], 3))
    for (s, e, ftype, strand, gname) in features_sorted:
        color = FEATURE_COLORS.get(ftype, "#bbbbbb")
        height = 0.5 if ftype == "CDS" else 0.3
        ybase = 0.25 if ftype == "CDS" else 0.35
        if ftype == "exon":
            ybase, height = 0.4, 0.2
        ax.add_patch(plt.Rectangle((s, ybase), e - s, height,
                                    color=color, alpha=0.85, linewidth=0))
    # Label gene names at unique positions
    seen_names = set()
    for (s, e, ftype, strand, gname) in features_sorted:
        if gname and gname not in seen_names and ftype == "CDS":
            ax.text((s + e) / 2, 0.05, gname, ha="center", va="bottom",
                    fontsize=7, alpha=0.9, style="italic")
            seen_names.add(gname)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("genomic position (bp)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw_entropy_panel(ax, pos, raw, smoothed, drops_by_method, color="tab:blue",
                       title=None, show_xlabels=False):
    ax.plot(pos, raw, color="#cccccc", lw=0.5, alpha=0.6, zorder=1, label="raw")
    ax.plot(pos, smoothed, color=color, lw=1.2, zorder=2, label="smoothed")
    for method, df, mcolor in drops_by_method:
        if df is None or len(df) == 0:
            continue
        pos_col = "genomic_pos" if "genomic_pos" in df.columns else (
                  "position" if "position" in df.columns else None)
        if not pos_col:
            continue
        for p in df[pos_col]:
            ax.axvline(p, color=mcolor, lw=0.6, alpha=0.4, linestyle="--", zorder=3)
    ax.set_ylabel("entropy", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if not show_xlabels:
        ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


# ---------- per-locus data loader ----------

def load_locus_data(row, results_dir, gtf_path, pad, smooth_window):
    run_dir = resolve_scoring_run(results_dir, row["chrom"], row.get("scoring_run", "auto"))
    if run_dir is None:
        logger.warning(f"{row['name']}: no scoring run for {row['chrom']}")
        return None
    ed = load_entropy_region(run_dir, row["start"], row["end"], pad)
    if ed is None:
        return None
    pos, raw = ed
    smoothed = smooth(raw, smooth_window)
    drops = {
        "zscore": load_drops(run_dir, "zscore", row["start"], row["end"], pad),
        "mad":    load_drops(run_dir, "mad",    row["start"], row["end"], pad),
    }
    features = load_gtf_features(gtf_path, row["chrom"], row["start"], row["end"], pad)
    return dict(pos=pos, raw=raw, smoothed=smoothed, drops=drops,
                features=features, row=row)


# ---------- figure composers ----------

def compose_fig_4_1(data, out_path):
    """Single-panel HBB hero: tight entropy + annotation ribbon."""
    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax_ent = fig.add_subplot(gs[0])
    ax_ann = fig.add_subplot(gs[1], sharex=ax_ent)

    draw_entropy_panel(ax_ent, data["pos"], data["raw"], data["smoothed"],
                       [("zscore", data["drops"]["zscore"], "#d62728"),
                        ("mad",    data["drops"]["mad"],    "#9467bd")],
                       title=f"{data['row']['name']} — {data['row']['chrom']}:"
                             f"{data['row']['start']:,}-{data['row']['end']:,} "
                             f"(Evo 2 per-nucleotide entropy)")
    draw_annotation_ribbon(ax_ann, data["features"],
                           data["row"]["start"], data["row"]["end"], pad=0)
    # Use same xlim as entropy
    ax_ann.set_xlim(ax_ent.get_xlim())

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def compose_fig_4_2(data_panels, out_path):
    """4-panel contrasting loci: EGFR / NPS / CRISPR / TIGR-Tas (or panel D stub)."""
    n = len(data_panels)
    fig = plt.figure(figsize=(11, 3.0 * n), constrained_layout=True)
    gs = gridspec.GridSpec(n * 2, 1, height_ratios=[4, 1] * n,
                           hspace=0.08, figure=fig)
    for i, (label, data) in enumerate(data_panels):
        ax_ent = fig.add_subplot(gs[2 * i])
        ax_ann = fig.add_subplot(gs[2 * i + 1], sharex=ax_ent)
        if data is None:
            ax_ent.text(0.5, 0.5, f"({label}) data not available",
                        ha="center", va="center", transform=ax_ent.transAxes,
                        fontsize=11, color="gray")
            ax_ent.set_xticks([]); ax_ent.set_yticks([])
            ax_ann.set_xticks([]); ax_ann.set_yticks([])
            for s in ax_ent.spines.values(): s.set_visible(False)
            for s in ax_ann.spines.values(): s.set_visible(False)
            continue
        title = f"({label}) {data['row']['name']} — {data['row']['chrom']}:" \
                f"{data['row']['start']:,}-{data['row']['end']:,}"
        draw_entropy_panel(ax_ent, data["pos"], data["raw"], data["smoothed"],
                           [("zscore", data["drops"]["zscore"], "#d62728"),
                            ("mad",    data["drops"]["mad"],    "#9467bd")],
                           title=title)
        draw_annotation_ribbon(ax_ann, data["features"],
                               data["row"]["start"], data["row"]["end"], pad=0)
        ax_ann.set_xlim(ax_ent.get_xlim())
        # Turn off scientific-notation offset on x-axis ticks
        ax_ent.ticklabel_format(axis="x", useOffset=False, style="plain")
        ax_ann.ticklabel_format(axis="x", useOffset=False, style="plain")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def compose_fig_4_3(data, methods, out_path):
    """6-method comparison grid on reference locus (NPS)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True, sharey=True)
    method_colors = {
        "zscore":            "#d62728",
        "mad":               "#9467bd",
        "derivative":        "#2ca02c",
        "cusum":             "#ff7f0e",
        "window_mean_shift": "#1f77b4",
        "local_baseline":    "#8c564b",
    }
    for ax, method in zip(axes.flat, methods):
        ax.plot(data["pos"], data["raw"], color="#cccccc", lw=0.4, alpha=0.6)
        ax.plot(data["pos"], data["smoothed"], color="tab:gray", lw=0.8)
        df = data["drops"].get(method)
        if df is not None and len(df):
            pos_col = "genomic_pos" if "genomic_pos" in df.columns else (
                      "position" if "position" in df.columns else None)
            if pos_col:
                color = method_colors.get(method, "k")
                ax.scatter(df[pos_col],
                           np.interp(df[pos_col], data["pos"], data["smoothed"]),
                           color=color, s=18, zorder=3, label=f"n={len(df)}")
                ax.legend(loc="upper right", fontsize=8)
        ax.set_title(method, fontsize=10, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(f"{data['row']['chrom']} position (bp)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("entropy", fontsize=9)
    fig.suptitle(f"Figure 4.3 — Six-method drop detection on {data['row']['name']} "
                 f"({data['row']['chrom']}:{data['row']['start']:,}-"
                 f"{data['row']['end']:,})", fontsize=11, y=1.00)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


# ---------- compute drops for all 6 methods on-the-fly (fig 4.3 needs these) ----------

def compute_drops_all_methods(data, pad):
    """Run all 6 detection methods on the locus entropy slice.
    Returns {method_name: DataFrame({genomic_pos, confidence})} for each.

    Uses detection_methods.run_method() on data['pos']+data['raw']; positions
    from run_method are local to the slice, translated to genomic coords.
    """
    # Import lazily so non-Fig-4.3 paths don't need detection_methods available
    from detection_methods import run_method

    methods = ["zscore", "mad", "derivative", "cusum",
               "window_mean_shift", "local_baseline"]
    drops = {}
    pos_arr = data["pos"]
    raw = data["raw"]
    origin = int(pos_arr[0])  # first genomic position in the slice
    for method in methods:
        try:
            calls = run_method(method, raw)  # list of (local_pos, score)
        except Exception as exc:
            logger.warning(f"method {method} failed: {exc}")
            drops[method] = pd.DataFrame(columns=["genomic_pos", "confidence"])
            continue
        if not calls:
            drops[method] = pd.DataFrame(columns=["genomic_pos", "confidence"])
            continue
        df = pd.DataFrame(calls, columns=["local_pos", "confidence"])
        df["genomic_pos"] = df["local_pos"].astype(int) + origin
        drops[method] = df[["genomic_pos", "confidence"]]
    return drops


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_locus_dir", required=True,
                   help="output directory from make_ncRNA_figures.py (contains manifest.tsv)")
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--human_gtf", default=None)
    p.add_argument("--ecoli_gtf", default=None)
    p.add_argument("--bacillus_gtf", default=None)
    p.add_argument("--reference_locus", default="NPS",
                   help="locus name to use for Fig 4.3 6-method comparison")
    p.add_argument("--pad", type=int, default=5000)
    p.add_argument("--smooth_window", type=int, default=51)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    manifest_path = os.path.join(args.per_locus_dir, "manifest.tsv")
    manifest = pd.read_csv(manifest_path, sep="\t")
    out_dir = os.path.join(args.per_locus_dir, "composed")
    os.makedirs(out_dir, exist_ok=True)

    gtf_map = {"human": args.human_gtf, "ecoli": args.ecoli_gtf,
               "bacillus": args.bacillus_gtf, "custom": None}

    # --- Fig 4.1: HBB hero ---
    hbb_row = manifest[manifest["name"] == "HBB"]
    if len(hbb_row):
        row = hbb_row.iloc[0]
        data = load_locus_data(row, args.results_dir, gtf_map.get(row["organism"]),
                               args.pad, args.smooth_window)
        if data:
            compose_fig_4_1(data, os.path.join(out_dir, "fig_4_1_HBB.png"))
    else:
        logger.warning("No HBB row in manifest — skipping Fig 4.1")

    # --- Fig 4.2: 4-panel contrasting loci ---
    panels_spec = [("A", "EGFR"), ("B", "NPS"),
                   ("C", "CRISPR_TypeIE"), ("D", "TIGR_Tas")]
    data_panels = []
    for label, name in panels_spec:
        sub = manifest[manifest["name"] == name]
        if len(sub) == 0:
            data_panels.append((label, None))
            continue
        row = sub.iloc[0]
        data = load_locus_data(row, args.results_dir, gtf_map.get(row["organism"]),
                               args.pad, args.smooth_window)
        data_panels.append((label, data))
    compose_fig_4_2(data_panels, os.path.join(out_dir, "fig_4_2_contrasting_loci.png"))

    # --- Fig 4.3: 6-method grid on reference locus ---
    ref_row = manifest[manifest["name"] == args.reference_locus]
    if len(ref_row):
        row = ref_row.iloc[0]
        data = load_locus_data(row, args.results_dir, gtf_map.get(row["organism"]),
                               args.pad, args.smooth_window)
        if data is not None:
            # Compute all 6 methods on-the-fly from the entropy slice
            # (drops.tsv only contains zscore+mad; others aren't dispatched at scoring time)
            data["drops"] = compute_drops_all_methods(data, args.pad)
            methods = ["zscore", "mad", "derivative",
                       "cusum", "window_mean_shift", "local_baseline"]
            compose_fig_4_3(data, methods,
                            os.path.join(out_dir, f"fig_4_3_six_methods_{args.reference_locus}.png"))
    else:
        logger.warning(f"No {args.reference_locus} row in manifest — skipping Fig 4.3")

    logger.info(f"Composed figures: {out_dir}")


if __name__ == "__main__":
    main()
