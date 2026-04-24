#!/usr/bin/env python3
"""
runtime_report.py

Walks the results/ tree for every COMPLETED JSON sentinel, aggregates
wall-time data, and emits:
  - runtime_matrix.tsv     — rows=chrom, cols=pipeline stage, value=total wall_time_s
  - runtime_detail.tsv     — every COMPLETED run with path, stage, script, wall_time_s
  - runtime_stacked.png    — stacked bar per chrom, stages as color
  - runtime_per_stage.png  — box plot of wall time per stage
"""

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    "scoring", "sae", "sae_global_stats",
    "latent_analysis", "latent_analysis_prenorm", "latent_analysis_postnorm",
    "visualization", "sae_tsne",
]


def scan_completed(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "**", "COMPLETED"), recursive=True):
        try:
            with open(path) as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"unreadable {path}: {e}")
            continue
        rel = os.path.relpath(path, results_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 3:
            chrom, stage = parts[0], parts[1]
        else:
            chrom, stage = "_unknown_", "_unknown_"
        rows.append({
            "chrom": chrom,
            "stage": stage,
            "run": os.path.dirname(rel),
            "script": meta.get("script"),
            "wall_time_s": float(meta.get("wall_time_s", 0) or 0),
            "completed_at": meta.get("completed_at"),
        })
    return pd.DataFrame(rows)


def build_matrix(detail):
    pivot = (detail.groupby(["chrom", "stage"])["wall_time_s"].sum()
             .unstack(fill_value=0.0))
    cols = [c for c in STAGE_ORDER if c in pivot.columns]
    cols += [c for c in pivot.columns if c not in cols]
    return pivot[cols]


def plot_stacked(matrix, out_path):
    fig, ax = plt.subplots(figsize=(max(10, len(matrix) * 0.4), 6))
    matrix_hr = matrix / 3600.0
    matrix_hr.plot(kind="bar", stacked=True, ax=ax, cmap="tab20")
    ax.set_ylabel("wall time (hours)")
    ax.set_xlabel("chromosome / organism")
    ax.set_title("Total wall time per stage per chromosome")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_per_stage_box(detail, out_path):
    stages = [s for s in STAGE_ORDER if (detail["stage"] == s).any()]
    data = [detail.loc[detail["stage"] == s, "wall_time_s"].values / 60.0 for s in stages]
    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.2), 5))
    ax.boxplot(data, labels=stages)
    ax.set_ylabel("wall time (minutes)")
    ax.set_title("Wall time per stage (box across runs)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    detail = scan_completed(args.results_dir)
    if detail.empty:
        logger.error(f"No COMPLETED files found in {args.results_dir}")
        return 2

    if args.output_dir is None:
        out_dir = build_run_dir(args.results_dir, "_genome_wide", "runtime", "report")
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    detail.to_csv(os.path.join(out_dir, "runtime_detail.tsv"), sep="\t", index=False)
    matrix = build_matrix(detail)
    matrix.to_csv(os.path.join(out_dir, "runtime_matrix.tsv"), sep="\t")
    plot_stacked(matrix, os.path.join(out_dir, "runtime_stacked.png"))
    plot_per_stage_box(detail, os.path.join(out_dir, "runtime_per_stage.png"))

    total_hours = detail["wall_time_s"].sum() / 3600.0
    logger.info(f"Aggregate compute: {total_hours:.1f} h across {len(detail)} runs")
    write_completed(out_dir, "runtime_report.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
