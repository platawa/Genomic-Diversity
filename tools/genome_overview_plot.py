#!/usr/bin/env python3
"""
genome_overview_plot.py

Single genome-wide overview figure combining:
  - chromosome length bars with centromere placement
  - genome-wide entropy track (binned mean entropy from scoring COMPLETED runs)
  - per-chromosome drop density (drops / Mb)
  - per-chromosome region-length violin
Output: overview.png + underlying TSVs.

Inputs are auto-discovered from results/<chrom>/scoring/<latest COMPLETED>/data/
using find_latest_completed().
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, find_latest_completed, write_completed

logger = logging.getLogger(__name__)

HUMAN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def load_chrom_data(results_dir, chrom):
    run = find_latest_completed(results_dir, chrom, "scoring")
    if run is None:
        return None
    drops_path = os.path.join(run, "data", "drops.tsv")
    rises_path = os.path.join(run, "data", "rises.tsv")
    drop_bounds = os.path.join(run, "data", "drop_boundaries.tsv")
    entropy_npz = os.path.join(run, "data", "entropy.npz")
    data = {"run": run}
    if os.path.isfile(drops_path):
        data["drops"] = pd.read_csv(drops_path, sep="\t", comment="#", low_memory=False)
    if os.path.isfile(drop_bounds):
        data["boundaries"] = pd.read_csv(drop_bounds, sep="\t", comment="#", low_memory=False)
    if os.path.isfile(entropy_npz):
        try:
            nz = np.load(entropy_npz)
            if "entropy" in nz.files:
                data["entropy"] = nz["entropy"]
            # Prefer explicit chrom_length, else infer from entropy/positions/end
            if "chrom_length" in nz.files:
                data["length"] = int(nz["chrom_length"])
            elif "end" in nz.files:
                data["length"] = int(np.atleast_1d(nz["end"])[-1])
            elif "positions" in nz.files:
                data["length"] = int(np.atleast_1d(nz["positions"])[-1] + 1)
            elif "entropy" in nz.files:
                data["length"] = int(nz["entropy"].shape[0])
        except Exception as e:
            logger.warning(f"{chrom}: could not read entropy.npz ({e})")
    return data


def compute_drop_density_mb(data):
    """Drops per Mb using drop_boundaries.tsv (one row per drop region)."""
    if "boundaries" not in data or "length" not in data:
        return None
    return len(data["boundaries"]) / (data["length"] / 1e6)


def plot_overview(chrom_data, out_path):
    chroms = [c for c in HUMAN_CHROMS if chrom_data.get(c)]
    n = len(chroms)
    if n == 0:
        logger.warning("No chromosome data loaded; nothing to plot.")
        return
    fig, axes = plt.subplots(4, 1, figsize=(max(12, n * 0.5), 14), sharex=True)

    # Panel 1: chromosome lengths
    lengths = [chrom_data[c].get("length") for c in chroms]
    lengths_mb = [(l / 1e6) if l is not None else 0.0 for l in lengths]
    axes[0].bar(chroms, lengths_mb, color="tab:gray")
    axes[0].set_ylabel("Length (Mb)")
    axes[0].set_title("Chromosome lengths")

    # Panel 2: genome-wide mean entropy (binned per chrom)
    mean_ent = []
    for c in chroms:
        if "entropy" in chrom_data[c]:
            try:
                mean_ent.append(float(np.nanmean(chrom_data[c]["entropy"])))
                continue
            except Exception:
                pass
        mean_ent.append(0.0)
    axes[1].bar(chroms, mean_ent, color="tab:blue")
    axes[1].set_ylabel("Mean entropy")
    axes[1].set_title("Mean per-position entropy per chromosome")

    # Panel 3: drop density per Mb
    density = [compute_drop_density_mb(chrom_data[c]) for c in chroms]
    density = [d if d is not None else 0.0 for d in density]
    axes[2].bar(chroms, density, color="tab:orange")
    axes[2].set_ylabel("Drops / Mb")
    axes[2].set_title("Drop region density per chromosome")

    # Panel 4: region length violin
    length_data = []
    for c in chroms:
        bd = chrom_data[c].get("boundaries")
        if bd is None:
            length_data.append(np.array([]))
            continue
        if {"drop_start", "rise_end"}.issubset(bd.columns):
            lens = bd["rise_end"] - bd["drop_start"]
        elif {"start", "end"}.issubset(bd.columns):
            lens = bd["end"] - bd["start"]
        else:
            lens = np.array([])
        length_data.append(np.asarray(lens))
    non_empty = [d for d in length_data if d.size > 0]
    if non_empty:
        axes[3].violinplot(length_data, positions=range(n), showmedians=True)
    axes[3].set_yscale("log")
    axes[3].set_ylabel("Drop region length (bp, log)")
    axes[3].set_xticks(range(n))
    axes[3].set_xticklabels(chroms, rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--chroms", nargs="+", default=HUMAN_CHROMS)
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    chrom_data = {}
    for c in args.chroms:
        data = load_chrom_data(args.results_dir, c)
        if data is None:
            logger.warning(f"{c}: no scoring run found")
            continue
        chrom_data[c] = data

    if args.output_dir is None:
        out_dir = build_run_dir(args.results_dir, "_genome_wide", "overview", "all_chroms")
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    plot_overview(chrom_data, os.path.join(out_dir, "overview.png"))

    # Per-chrom summary TSV
    summary = []
    for c in args.chroms:
        d = chrom_data.get(c)
        if d is None:
            continue
        summary.append({
            "chrom": c,
            "length_bp": d.get("length"),
            "mean_entropy": float(np.nanmean(d["entropy"])) if "entropy" in d else None,
            "n_drops": len(d.get("boundaries", [])) if "boundaries" in d else None,
            "drops_per_mb": compute_drop_density_mb(d),
        })
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "chromosome_summary.tsv"),
                                 sep="\t", index=False)

    write_completed(out_dir, "genome_overview_plot.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
