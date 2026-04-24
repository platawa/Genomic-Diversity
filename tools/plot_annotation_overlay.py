#!/usr/bin/env python3
"""
plot_annotation_overlay.py

Regenerate per-annotation t-SNE/UMAP plots with an all-points gray background
so the reader can see the full embedding shape and where the highlighted
annotation sits within it.

Reads an existing `sae_tsne_prenorm` / `sae_tsne_postnorm` run directory and
writes new PNGs into its `plots/` subdir with the suffix `_overlay.png` so
existing plots are not overwritten.

Usage:
    python tools/plot_annotation_overlay.py \\
        --run_dir results/_genome_wide/sae_tsne_prenorm/20260423_192324_23chroms_897129regions \\
        --annotation CDS
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANNOT_COLORS = {
    "CDS": "#e41a1c",        # red
    "UTR/exon": "#ff7f00",   # orange
    "Intron": "#377eb8",     # blue
    "Intergenic": "#4daf4a", # green (intentionally different from the
                             # '#999999' gray used for the background)
}

# Match standardized styling in genome_sae_tsne.py
FIGSIZE_SINGLE = (8, 7)
DPI = 200
TITLE_FONTSIZE = 13
AXIS_FONTSIZE = 11
LEGEND_FONTSIZE = 9


def read_tsv_annotations_and_coords(tsv_path: Path):
    """Read per-region annotation + tsne/umap coords from cluster_assignments.tsv.

    Returns (annotations: list[str], tsne: Nx2 array or None,
             umap: Nx2 array or None).
    """
    # Skip comment lines starting with '#'; first non-comment is header.
    with open(tsv_path) as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#") and ln.strip()]
    header = lines[0].split("\t")
    idx_ann = header.index("annotation")
    idx_ts1 = header.index("tsne_1") if "tsne_1" in header else None
    idx_ts2 = header.index("tsne_2") if "tsne_2" in header else None
    idx_um1 = header.index("umap_1") if "umap_1" in header else None
    idx_um2 = header.index("umap_2") if "umap_2" in header else None

    anns = []
    ts = [] if idx_ts1 is not None else None
    um = [] if idx_um1 is not None else None
    for line in lines[1:]:
        parts = line.split("\t")
        anns.append(parts[idx_ann])
        if ts is not None:
            ts.append((float(parts[idx_ts1]), float(parts[idx_ts2])))
        if um is not None:
            um.append((float(parts[idx_um1]), float(parts[idx_um2])))
    ts_arr = np.array(ts) if ts is not None else None
    um_arr = np.array(um) if um is not None else None
    return anns, ts_arr, um_arr


def plot_overlay(coords: np.ndarray, annotations, target: str, color: str,
                 title_prefix: str, xlabel: str, ylabel: str,
                 out_path: Path):
    """Plot ALL points as gray background + target-annotation points in color."""
    mask = np.array([a == target for a in annotations])
    n_target = int(mask.sum())
    n_total = len(annotations)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Background: all points in gray, small + low alpha
    ax.scatter(coords[~mask, 0], coords[~mask, 1],
               c="#cccccc", s=0.4, alpha=0.15, edgecolors="none",
               rasterized=True, label=f"Other ({n_total - n_target:,})")

    # Foreground: target annotation in color, slightly bigger + higher alpha
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, s=1.5, alpha=0.55, edgecolors="none",
               rasterized=True, label=f"{target} ({n_target:,})")

    leg = ax.legend(fontsize=LEGEND_FONTSIZE, markerscale=6, loc="best",
                    framealpha=0.9)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)

    ax.set_title(f"{title_prefix} — {target} overlay on full embedding "
                 f"(N={n_total:,})", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Existing run dir, e.g. "
                         "results/_genome_wide/sae_tsne_prenorm/20260423_192324_23chroms_897129regions")
    ap.add_argument("--annotation", default="CDS",
                    help="Annotation to highlight (CDS | UTR/exon | Intron | "
                         "Intergenic | all). Default: CDS.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    tsv = run_dir / "data" / "cluster_assignments.tsv"
    plots_dir = run_dir / "plots"
    if not tsv.is_file():
        raise SystemExit(f"missing {tsv}")
    plots_dir.mkdir(exist_ok=True)

    print(f"Reading {tsv}...")
    annotations, ts, um = read_tsv_annotations_and_coords(tsv)
    print(f"  {len(annotations):,} regions")

    if args.annotation.lower() == "all":
        targets = ["CDS", "UTR/exon", "Intron", "Intergenic"]
    else:
        if args.annotation not in ANNOT_COLORS:
            raise SystemExit(
                f"unknown annotation '{args.annotation}'. "
                f"Options: {list(ANNOT_COLORS.keys())} or 'all'."
            )
        targets = [args.annotation]

    for target in targets:
        color = ANNOT_COLORS[target]
        slug = target.lower().replace("/", "_")
        if ts is not None:
            out = plots_dir / f"tsne_annotation_{slug}_overlay.png"
            plot_overlay(ts, annotations, target, color,
                         "TSNE of SAE Regions", "TSNE 1", "TSNE 2", out)
        if um is not None:
            out = plots_dir / f"umap_annotation_{slug}_overlay.png"
            plot_overlay(um, annotations, target, color,
                         "UMAP of SAE Regions", "UMAP 1", "UMAP 2", out)


if __name__ == "__main__":
    main()
