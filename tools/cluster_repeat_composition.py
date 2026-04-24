#!/usr/bin/env python3
"""
cluster_repeat_composition.py

Intersects per-cluster regions with RepeatMasker BED tracks and reports the
proportion of each repeat class (LINE, SINE, LTR, DNA, simple, tRNA, …) per
Leiden cluster.

Outputs:
  - cluster_repeat_counts.tsv          — rows=cluster_id, cols=repeat_class
  - cluster_repeat_fractions.tsv       — same but normalized per cluster
  - stacked_bar.png                    — stacked bar: fraction per class × cluster
  - top_class_per_cluster.tsv          — majority class (≥50%) assignment per cluster

Works for one chrom at a time (pass --chrom) OR for genome-wide via
--scope_dir pointing at results/_genome_wide/sae_tsne_*/<run>/.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed

logger = logging.getLogger(__name__)


def load_repeatmasker_bed(bed_path, chrom_filter=None):
    """Load (chrom, start, end, repeat_class) from a RepeatMasker BED.

    Expects 4+ column BED with the class in col 4 (name) or col 7 (sometimes).
    """
    rows = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            if chrom_filter is not None and chrom not in chrom_filter:
                continue
            try:
                s, e = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            name = parts[3]
            # RepeatMasker class is often in the form "LINE/L1" — keep the top class
            klass = name.split("/")[0] if "/" in name else name
            rows.append((chrom, s, e, klass))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "klass"])


def load_regions(scope_dir):
    """cluster_assignments.tsv must have columns: genomic_start, genomic_end,
    cluster_id. chrom may be a column or inferred from scope."""
    ca_path = os.path.join(scope_dir, "data", "cluster_assignments.tsv")
    if not os.path.isfile(ca_path):
        raise FileNotFoundError(ca_path)
    df = pd.read_csv(ca_path, sep="\t", comment="#")
    if "cluster_id" not in df.columns:
        # Try alternates
        for alt in ["cluster", "leiden", "leiden_cluster"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "cluster_id"})
                break
    return df


def intersect_regions_with_repeats(regions, repeats):
    """For each region, list the repeat classes overlapping it.

    Returns a list of (region_idx, klass) tuples.
    """
    results = []
    # Group repeats by chrom for fast filtering
    by_chrom = {c: g.sort_values("start") for c, g in repeats.groupby("chrom")}
    for i, r in regions.iterrows():
        chrom = r.get("chrom")
        if chrom is None or chrom not in by_chrom:
            continue
        g = by_chrom[chrom]
        mask = (g["start"] < r["genomic_end"]) & (g["end"] > r["genomic_start"])
        for klass in g.loc[mask, "klass"].unique():
            results.append((i, klass))
    return pd.DataFrame(results, columns=["region_idx", "klass"])


def cluster_class_matrix(regions, region_classes):
    """cluster_id × repeat_class count matrix."""
    merged = region_classes.merge(regions.reset_index().rename(columns={"index": "region_idx"}),
                                  on="region_idx")
    counts = merged.groupby(["cluster_id", "klass"]).size().unstack(fill_value=0)
    return counts


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scope_dir", required=True,
                   help="Dir containing data/cluster_assignments.tsv")
    p.add_argument("--repeatmasker_bed", required=True)
    p.add_argument("--chrom", default=None,
                   help="Limit repeats and regions to this chromosome (needed for per-chrom scope)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    regions = load_regions(args.scope_dir)
    if "chrom" not in regions.columns and args.chrom:
        regions["chrom"] = args.chrom

    chrom_filter = {args.chrom} if args.chrom else None
    repeats = load_repeatmasker_bed(args.repeatmasker_bed, chrom_filter=chrom_filter)
    logger.info(f"Loaded {len(repeats):,} repeat intervals and {len(regions):,} regions")

    out_dir = args.output_dir or os.path.join(args.scope_dir, "cluster_repeats")
    os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    region_classes = intersect_regions_with_repeats(regions, repeats)
    counts = cluster_class_matrix(regions, region_classes)
    fractions = counts.div(counts.sum(axis=1), axis=0)

    counts.to_csv(os.path.join(out_dir, "cluster_repeat_counts.tsv"), sep="\t")
    fractions.to_csv(os.path.join(out_dir, "cluster_repeat_fractions.tsv"), sep="\t")

    # Stacked bar
    fig, ax = plt.subplots(figsize=(max(10, len(fractions) * 0.4), 6))
    fractions.plot(kind="bar", stacked=True, ax=ax, cmap="tab20")
    ax.set_ylabel("fraction")
    ax.set_xlabel("cluster")
    ax.set_title("Repeat class composition per cluster")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "stacked_bar.png"), dpi=130)
    plt.close(fig)

    # Majority class
    top = fractions.idxmax(axis=1)
    top_frac = fractions.max(axis=1)
    pd.DataFrame({
        "cluster_id": top.index,
        "majority_class": top.values,
        "majority_fraction": top_frac.values,
    }).to_csv(os.path.join(out_dir, "top_class_per_cluster.tsv"),
              sep="\t", index=False)

    write_completed(out_dir, "cluster_repeat_composition.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
