#!/usr/bin/env python3
"""
plot_exon_position_overlay.py

Overlay per-region exon-position class (first / middle / last) onto
pre-computed t-SNE and UMAP embeddings. Produces three separate overlay
PNGs per embedding (one per class), in the gray-background style of
plot_annotation_overlay.py.

Works on:
  - Per-chromosome latent runs:
      results/<chrom>/sae/latent_analysis{,_prenorm,_normalized}/
  - Genome-wide combined runs:
      results/_genome_wide/sae_tsne_{prenorm,postnorm,raw}/<timestamp>_.../

Caches classification in `<run_dir>/data/exon_position.tsv` and reuses
it on subsequent calls.

Example:
    python tools/plot_exon_position_overlay.py \\
        --run_dir results/chr22/sae/latent_analysis_prenorm \\
        --gtf /path/to/genomic.gtf
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exon_position_classifier import classify_all_chroms_once

logger = logging.getLogger(__name__)

EXON_COLORS = {
    "first_exon":  "#f5b041",
    "middle_exon": "#1abc9c",
    "last_exon":   "#c0392b",
}
EXON_LABELS = {
    "first_exon":  "First exon",
    "middle_exon": "Middle exon",
    "last_exon":   "Last exon",
}
CLASS_ALIASES = {
    "first": "first_exon", "first_exon": "first_exon",
    "middle": "middle_exon", "middle_exon": "middle_exon",
    "last": "last_exon", "last_exon": "last_exon",
}

FIGSIZE_SINGLE = (8, 7)
DPI = 200
TITLE_FONTSIZE = 13
AXIS_FONTSIZE = 11
LEGEND_FONTSIZE = 9


def load_regions(tsv_path: Path, run_dir: Path) -> pd.DataFrame:
    ca = pd.read_csv(tsv_path, sep="\t", comment="#")
    needed = {"genomic_start", "genomic_end"}
    missing = needed - set(ca.columns)
    if missing:
        raise SystemExit(f"{tsv_path} is missing required columns: {missing}")
    # Per-chromosome TSVs omit `chrom` (it's implicit in the path). Derive
    # it from the results/<chrom>/sae/<subdir> convention.
    if "chrom" not in ca.columns:
        chrom = None
        for part in reversed(run_dir.resolve().parts):
            if part.startswith("chr") or part in {"ecoli_K12", "bacillus_subtilis"}:
                chrom = part
                break
        if chrom is None:
            raise SystemExit(
                f"{tsv_path} has no `chrom` column and one could not be "
                f"derived from {run_dir}")
        logger.info(f"No `chrom` column; derived '{chrom}' from run_dir path")
        ca.insert(0, "chrom", chrom)
    return ca


def load_or_build_classes(run_dir: Path, ca: pd.DataFrame, gtf_path: str,
                          force: bool) -> np.ndarray:
    cache_path = run_dir / "data" / "exon_position.tsv"
    if cache_path.is_file() and not force:
        cached = pd.read_csv(cache_path, sep="\t", comment="#")
        if len(cached) == len(ca) and "class" in cached.columns:
            logger.info(f"Using cached classification at {cache_path} "
                        f"({len(cached)} rows)")
            return cached["class"].values
        logger.info(f"Cache {cache_path} length or schema mismatch; rebuilding")

    logger.info(f"Classifying {len(ca)} regions against {gtf_path} ...")
    records = classify_all_chroms_once(
        gtf_path,
        ca["chrom"].values,
        ca["genomic_start"].values,
        ca["genomic_end"].values,
    )
    df = pd.DataFrame(records)
    df.insert(0, "region_idx", np.arange(len(df)))
    df.insert(1, "chrom", ca["chrom"].values)
    df.insert(2, "genomic_start", ca["genomic_start"].values)
    df.insert(3, "genomic_end", ca["genomic_end"].values)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, sep="\t", index=False)
    counts = df["class"].value_counts().to_dict()
    logger.info(f"Wrote {cache_path}  class_counts={counts}")
    return df["class"].values


def plot_overlay(coords: np.ndarray, classes: np.ndarray, target: str,
                 color: str, title_prefix: str, xlabel: str, ylabel: str,
                 out_path: Path) -> None:
    mask = classes == target
    n_target = int(mask.sum())
    n_total = len(classes)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.scatter(coords[~mask, 0], coords[~mask, 1],
               c="#cccccc", s=0.4, alpha=0.15, edgecolors="none",
               rasterized=True,
               label=f"Other ({n_total - n_target:,})")
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, s=1.5, alpha=0.55, edgecolors="none",
               rasterized=True,
               label=f"{EXON_LABELS[target]} ({n_target:,})")
    leg = ax.legend(fontsize=LEGEND_FONTSIZE, markerscale=6, loc="best",
                    framealpha=0.9)
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    for handle in handles:
        handle.set_alpha(1.0)
    ax.set_title(f"{title_prefix} — {EXON_LABELS[target]} overlay "
                 f"(N={n_total:,})", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def resolve_classes(arg: str):
    targets = []
    for item in arg.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in CLASS_ALIASES:
            logger.warning(f"Skipping unknown class '{item}'")
            continue
        canonical = CLASS_ALIASES[item]
        if canonical not in targets:
            targets.append(canonical)
    return targets


def infer_title_prefix(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    tail = parts[-3:] if len(parts) >= 3 else parts
    return "/".join(tail)


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--run_dir", required=True,
                    help="Directory containing data/cluster_assignments.tsv")
    ap.add_argument("--gtf", required=True, help="Path to genomic.gtf")
    ap.add_argument("--classes", default="first,middle,last",
                    help="Comma-list of classes to plot (first, middle, last)")
    ap.add_argument("--force_reclassify", action="store_true",
                    help="Ignore an existing exon_position.tsv cache")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_dir = Path(args.run_dir).resolve()
    tsv = run_dir / "data" / "cluster_assignments.tsv"
    if not tsv.is_file():
        logger.error(f"Missing {tsv}")
        return 2
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ca = load_regions(tsv, run_dir)
    classes = load_or_build_classes(run_dir, ca, args.gtf, args.force_reclassify)
    if len(classes) != len(ca):
        logger.error(f"Classes length {len(classes)} != regions length {len(ca)}")
        return 3

    targets = resolve_classes(args.classes)
    if not targets:
        logger.error("No valid classes selected")
        return 4

    tsne = ca[["tsne_1", "tsne_2"]].values \
        if {"tsne_1", "tsne_2"}.issubset(ca.columns) else None
    umap = ca[["umap_1", "umap_2"]].values \
        if {"umap_1", "umap_2"}.issubset(ca.columns) else None

    if tsne is None and umap is None:
        logger.error(f"{tsv} has neither tsne_1/tsne_2 nor umap_1/umap_2")
        return 5

    title_prefix = infer_title_prefix(run_dir)

    for target in targets:
        color = EXON_COLORS[target]
        slug = target.replace("_exon", "")
        if tsne is not None:
            plot_overlay(tsne, classes, target, color,
                         f"t-SNE — {title_prefix}", "t-SNE 1", "t-SNE 2",
                         plots_dir / f"tsne_exon_{slug}_overlay.png")
        if umap is not None:
            plot_overlay(umap, classes, target, color,
                         f"UMAP — {title_prefix}", "UMAP 1", "UMAP 2",
                         plots_dir / f"umap_exon_{slug}_overlay.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
