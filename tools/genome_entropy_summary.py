#!/usr/bin/env python3
"""
genome_entropy_summary.py

Aggregate per-chromosome scoring results into a genome-wide entropy summary.
Produces a TSV (one row per chromosome), a JSON with aggregated stats, and
prints a formatted summary table to stdout.

Usage:
    # Summarize all human chromosomes
    python tools/genome_entropy_summary.py --all_human

    # Summarize specific chromosomes
    python tools/genome_entropy_summary.py --chroms chr1 chr22 chrX

    # Include median and percentile computation (loads entropy.npz per chrom)
    python tools/genome_entropy_summary.py --all_human --compute_percentiles

    # Custom results directory
    python tools/genome_entropy_summary.py --results_dir /path/to/results --all_human
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from results_utils import find_latest_completed, find_all_completed, write_completed


# All human chromosomes (GRCh38)
ALL_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15",
    "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22",
    "chrX", "chrY",
]


def setup_logging(log_file=None):
    logger = logging.getLogger("genome_entropy_summary")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_summary_json(run_dir):
    path = os.path.join(run_dir, "data", "summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_chrom_stats(summary, run_dir, compute_percentiles=False, logger=None):
    entropy_stats = summary.get("entropy_stats", {})
    results = summary.get("results", {})
    timing = summary.get("timing", {})

    row = {
        "seq_length": summary.get("sequence_length", 0),
        "mean_entropy": entropy_stats.get("mean", float("nan")),
        "std_entropy": entropy_stats.get("std", float("nan")),
        "median_entropy": float("nan"),
        "p5_entropy": float("nan"),
        "p95_entropy": float("nan"),
        "nan_fraction": entropy_stats.get("nan_fraction", float("nan")),
        "n_drops_zscore": results.get("zscore", {}).get("n_drops", 0),
        "n_drops_mad": results.get("mad", {}).get("n_drops", 0),
        "n_regions_zscore": results.get("zscore", {}).get("n_regions", 0),
        "n_regions_mad": results.get("mad", {}).get("n_regions", 0),
        "scoring_time_s": timing.get("total_wall_s", 0.0),
        "scoring_run_dir": run_dir,
    }

    if compute_percentiles:
        npz_path = os.path.join(run_dir, "data", "entropy.npz")
        if os.path.isfile(npz_path):
            if logger:
                logger.info(f"  Loading {npz_path} for percentiles...")
            try:
                data = np.load(npz_path)
                entropy = data["entropy"]
                valid = entropy[~np.isnan(entropy)]
                if len(valid) > 0:
                    row["median_entropy"] = float(np.median(valid))
                    row["p5_entropy"] = float(np.percentile(valid, 5))
                    row["p95_entropy"] = float(np.percentile(valid, 95))
            except Exception as e:
                if logger:
                    logger.warning(f"  Failed to load entropy.npz: {e}")

    return row


def write_genome_summary_tsv(chrom_stats, output_path, compute_percentiles=False):
    columns = ["chrom", "seq_length", "mean_entropy", "std_entropy"]
    if compute_percentiles:
        columns += ["median_entropy", "p5_entropy", "p95_entropy"]
    columns += [
        "nan_fraction", "n_drops_zscore", "n_drops_mad",
        "n_regions_zscore", "n_regions_mad", "scoring_time_s", "scoring_run_dir",
    ]

    with open(output_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for chrom in sorted(chrom_stats.keys(), key=_chrom_sort_key):
            row = chrom_stats[chrom]
            values = [chrom]
            for col in columns[1:]:
                val = row.get(col, "")
                if isinstance(val, float):
                    if col == "nan_fraction":
                        values.append(f"{val:.6f}")
                    elif "entropy" in col:
                        values.append(f"{val:.6f}")
                    else:
                        values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            f.write("\t".join(values) + "\n")


def build_genome_summary_json(chrom_stats, all_requested_chroms):
    completed_chroms = sorted(chrom_stats.keys(), key=_chrom_sort_key)
    missing_chroms = sorted(
        [c for c in all_requested_chroms if c not in chrom_stats],
        key=_chrom_sort_key,
    )

    total_bp = sum(row["seq_length"] for row in chrom_stats.values())
    total_regions_zscore = sum(row["n_regions_zscore"] for row in chrom_stats.values())
    total_regions_mad = sum(row["n_regions_mad"] for row in chrom_stats.values())
    total_drops_zscore = sum(row["n_drops_zscore"] for row in chrom_stats.values())
    total_drops_mad = sum(row["n_drops_mad"] for row in chrom_stats.values())
    total_scoring_time = sum(row["scoring_time_s"] for row in chrom_stats.values())

    # Weighted mean entropy (by chromosome length)
    valid_rows = [
        row for row in chrom_stats.values()
        if not np.isnan(row["mean_entropy"])
    ]
    if valid_rows and total_bp > 0:
        valid_bp = sum(row["seq_length"] for row in valid_rows)
        genome_mean_entropy = sum(
            row["mean_entropy"] * row["seq_length"] for row in valid_rows
        ) / valid_bp if valid_bp > 0 else float("nan")
    else:
        genome_mean_entropy = float("nan")

    # Pooled standard deviation weighted by length
    valid_std_rows = [
        row for row in chrom_stats.values()
        if not np.isnan(row["std_entropy"])
    ]
    if valid_std_rows:
        valid_bp_std = sum(row["seq_length"] for row in valid_std_rows)
        weighted_var_sum = sum(
            row["std_entropy"] ** 2 * row["seq_length"]
            for row in valid_std_rows
        )
        genome_std_entropy = float(
            np.sqrt(weighted_var_sum / valid_bp_std)
        ) if valid_bp_std > 0 else float("nan")
    else:
        genome_std_entropy = float("nan")

    return {
        "generated_at": datetime.now().isoformat(),
        "script": "genome_entropy_summary.py",
        "n_chromosomes_completed": len(completed_chroms),
        "n_chromosomes_missing": len(missing_chroms),
        "completed_chroms": completed_chroms,
        "missing_chroms": missing_chroms,
        "genome_stats": {
            "total_bp": total_bp,
            "genome_mean_entropy": round(genome_mean_entropy, 6),
            "genome_std_entropy": round(genome_std_entropy, 6),
            "total_regions_zscore": total_regions_zscore,
            "total_regions_mad": total_regions_mad,
            "total_drops_zscore": total_drops_zscore,
            "total_drops_mad": total_drops_mad,
            "total_scoring_time_s": round(total_scoring_time, 2),
            "total_scoring_time_hr": round(total_scoring_time / 3600.0, 2),
        },
    }


def print_summary_table(chrom_stats, genome_json, compute_percentiles=False):
    print()
    print("=" * 110)
    print("GENOME-WIDE ENTROPY SUMMARY")
    print("=" * 110)

    if compute_percentiles:
        header = (
            f"{'Chrom':<8} {'Length':>12} {'Mean H':>10} {'Std H':>10} "
            f"{'Median H':>10} {'P5':>10} {'P95':>10} "
            f"{'NaN%':>8} {'Drops_Z':>8} {'Drops_M':>8} "
            f"{'Reg_Z':>8} {'Reg_M':>8} {'Time(s)':>10}"
        )
    else:
        header = (
            f"{'Chrom':<8} {'Length':>12} {'Mean H':>10} {'Std H':>10} "
            f"{'NaN%':>8} {'Drops_Z':>8} {'Drops_M':>8} "
            f"{'Reg_Z':>8} {'Reg_M':>8} {'Time(s)':>10}"
        )
    print(header)
    print("-" * len(header))

    for chrom in sorted(chrom_stats.keys(), key=_chrom_sort_key):
        row = chrom_stats[chrom]
        length_str = f"{row['seq_length']:,}"
        mean_h = f"{row['mean_entropy']:.4f}" if not np.isnan(row['mean_entropy']) else "N/A"
        std_h = f"{row['std_entropy']:.4f}" if not np.isnan(row['std_entropy']) else "N/A"
        nan_pct = f"{row['nan_fraction'] * 100:.2f}%" if not np.isnan(row['nan_fraction']) else "N/A"
        time_s = f"{row['scoring_time_s']:.1f}"

        if compute_percentiles:
            med_h = f"{row['median_entropy']:.4f}" if not np.isnan(row['median_entropy']) else "N/A"
            p5 = f"{row['p5_entropy']:.4f}" if not np.isnan(row['p5_entropy']) else "N/A"
            p95 = f"{row['p95_entropy']:.4f}" if not np.isnan(row['p95_entropy']) else "N/A"
            print(
                f"{chrom:<8} {length_str:>12} {mean_h:>10} {std_h:>10} "
                f"{med_h:>10} {p5:>10} {p95:>10} "
                f"{nan_pct:>8} {row['n_drops_zscore']:>8} {row['n_drops_mad']:>8} "
                f"{row['n_regions_zscore']:>8} {row['n_regions_mad']:>8} {time_s:>10}"
            )
        else:
            print(
                f"{chrom:<8} {length_str:>12} {mean_h:>10} {std_h:>10} "
                f"{nan_pct:>8} {row['n_drops_zscore']:>8} {row['n_drops_mad']:>8} "
                f"{row['n_regions_zscore']:>8} {row['n_regions_mad']:>8} {time_s:>10}"
            )

    gs = genome_json["genome_stats"]
    print("-" * len(header))
    total_length_str = f"{gs['total_bp']:,}"
    mean_h = f"{gs['genome_mean_entropy']:.4f}"
    std_h = f"{gs['genome_std_entropy']:.4f}"
    time_s = f"{gs['total_scoring_time_s']:.1f}"

    if compute_percentiles:
        print(
            f"{'TOTAL':<8} {total_length_str:>12} {mean_h:>10} {std_h:>10} "
            f"{'':>10} {'':>10} {'':>10} "
            f"{'':>8} {gs['total_drops_zscore']:>8} {gs['total_drops_mad']:>8} "
            f"{gs['total_regions_zscore']:>8} {gs['total_regions_mad']:>8} {time_s:>10}"
        )
    else:
        print(
            f"{'TOTAL':<8} {total_length_str:>12} {mean_h:>10} {std_h:>10} "
            f"{'':>8} {gs['total_drops_zscore']:>8} {gs['total_drops_mad']:>8} "
            f"{gs['total_regions_zscore']:>8} {gs['total_regions_mad']:>8} {time_s:>10}"
        )

    print()
    print(f"Completed: {genome_json['n_chromosomes_completed']} chromosomes")
    if genome_json["missing_chroms"]:
        print(f"Missing:   {', '.join(genome_json['missing_chroms'])}")
    print(f"Total scoring time: {gs['total_scoring_time_hr']:.2f} hours")
    print()


def _chrom_sort_key(chrom):
    name = chrom.replace("chr", "")
    if name.isdigit():
        return (0, int(name), "")
    elif name == "X":
        return (1, 0, "X")
    elif name == "Y":
        return (1, 1, "Y")
    elif name in ("M", "MT"):
        return (1, 2, "M")
    else:
        return (2, 0, chrom)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-chromosome scoring results into a genome-wide summary."
    )
    ap.add_argument("--results_dir", default="results/",
                    help="Root results directory (default: results/)")
    ap.add_argument("--chroms", nargs="+", default=None,
                    help="Specific chromosomes to include")
    ap.add_argument("--all_human", action="store_true",
                    help="Include all 24 human chromosomes")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: auto-generated under results/)")
    ap.add_argument("--compute_percentiles", action="store_true",
                    help="Load entropy.npz per chrom for median/percentiles (slow)")

    args = ap.parse_args()

    if args.all_human and args.chroms:
        ap.error("--all_human and --chroms are mutually exclusive")
    if args.all_human:
        requested_chroms = list(ALL_HUMAN_CHROMS)
    elif args.chroms:
        requested_chroms = args.chroms
    else:
        ap.error("Must specify --chroms or --all_human")

    wall_start = time.time()

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = f"{len(requested_chroms)}chroms"
    if args.compute_percentiles:
        flags += "_pctiles"

    if args.output_dir:
        run_dir = args.output_dir
    else:
        run_dir = os.path.join(
            args.results_dir, "_genome_wide", "entropy_summary",
            f"{ts_str}_{flags}",
        )

    data_dir = os.path.join(run_dir, "data")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "genome_entropy_summary.log")
    logger = setup_logging(log_file)

    logger.info("=" * 70)
    logger.info("GENOME-WIDE ENTROPY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Results dir: {os.path.abspath(args.results_dir)}")
    logger.info(f"Output dir:  {os.path.abspath(run_dir)}")
    logger.info(f"Chromosomes requested: {len(requested_chroms)}")

    completed_runs = find_all_completed(args.results_dir, requested_chroms, "scoring")
    logger.info(f"Found {len(completed_runs)} completed scoring runs")

    if not completed_runs:
        logger.error("No completed scoring runs found.")
        sys.exit(1)

    chrom_stats = {}
    for chrom in sorted(completed_runs.keys(), key=_chrom_sort_key):
        run_dir_chrom = completed_runs[chrom]
        logger.info(f"  {chrom}: {run_dir_chrom}")
        summary = load_summary_json(run_dir_chrom)
        if summary is None:
            logger.warning(f"  {chrom}: summary.json not found, skipping")
            continue
        row = extract_chrom_stats(
            summary, run_dir_chrom,
            compute_percentiles=args.compute_percentiles, logger=logger,
        )
        chrom_stats[chrom] = row

    if not chrom_stats:
        logger.error("No valid summary.json files found.")
        sys.exit(1)

    tsv_path = os.path.join(data_dir, "genome_summary.tsv")
    json_path = os.path.join(data_dir, "genome_summary.json")

    write_genome_summary_tsv(chrom_stats, tsv_path, args.compute_percentiles)
    genome_json = build_genome_summary_json(chrom_stats, requested_chroms)
    with open(json_path, "w") as f:
        json.dump(genome_json, f, indent=2)
        f.write("\n")

    print_summary_table(chrom_stats, genome_json, args.compute_percentiles)

    wall_time = time.time() - wall_start
    write_completed(run_dir, "genome_entropy_summary.py", wall_time)
    logger.info(f"Done in {wall_time:.2f}s — Output: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
