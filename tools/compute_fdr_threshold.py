#!/usr/bin/env python3
"""
compute_fdr_threshold.py

Compute FDR-corrected significance thresholds across all chromosomes.
Loads drop_boundaries.tsv from each chromosome's scoring run,
converts confidence scores to p-values, applies Benjamini-Hochberg
FDR correction, and reports how many regions pass at various FDR levels.

Also compares against fixed confidence thresholds (4.0, 5.0, 6.0, 8.0).

Usage:
    python tools/compute_fdr_threshold.py --results_dir results/ --all_human
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from results_utils import find_latest_completed


def load_all_regions(results_dir, chroms):
    """Load all drop regions from all chromosomes."""
    all_regions = []
    per_chrom = {}

    for chrom in chroms:
        scoring_run = find_latest_completed(results_dir, chrom, 'scoring')
        if not scoring_run:
            print(f"  {chrom}: no completed scoring run, skipping")
            continue

        boundaries_file = os.path.join(scoring_run, 'data', 'drop_boundaries.tsv')
        if not os.path.isfile(boundaries_file):
            print(f"  {chrom}: no drop_boundaries.tsv, skipping")
            continue

        regions = []
        with open(boundaries_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('chrom'):
                    continue
                parts = line.split('\t')
                if len(parts) < 11:
                    continue
                region = {
                    'chrom': parts[0],
                    'drop_start': int(parts[1]),
                    'drop_end': int(parts[2]),
                    'genomic_start': int(parts[3]),
                    'genomic_end': int(parts[4]),
                    'region_length': int(parts[5]),
                    'method': parts[6],
                    'start_confidence': float(parts[7]),
                    'end_confidence': float(parts[8]),
                    'mean_entropy': float(parts[9]),
                    'min_entropy': float(parts[10]),
                }
                regions.append(region)

        per_chrom[chrom] = regions
        all_regions.extend(regions)
        print(f"  {chrom}: {len(regions)} regions (max conf: {max(r['start_confidence'] for r in regions):.2f}, "
              f"median conf: {np.median([r['start_confidence'] for r in regions]):.2f})")

    return all_regions, per_chrom


def compute_fdr(all_regions):
    """Compute FDR-corrected significance for all regions."""
    confidences = np.array([r['start_confidence'] for r in all_regions])

    # Convert confidence (|z-score| or |MAD-score|) to two-sided p-values
    # Using normal distribution CDF (valid for z-scores; approximate for MAD)
    pvalues = 2.0 * stats.norm.sf(confidences)  # sf = 1 - cdf (survival function)

    # Benjamini-Hochberg FDR correction
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # BH procedure
    bh_threshold = np.arange(1, n + 1) / n
    fdr_levels = [0.01, 0.05, 0.10, 0.20]

    results = {}
    for fdr in fdr_levels:
        # Find largest k such that p_(k) <= k/n * fdr
        adjusted_threshold = bh_threshold * fdr
        passing = sorted_pvals <= adjusted_threshold
        if passing.any():
            max_k = np.max(np.where(passing)[0]) + 1
            # The confidence threshold that corresponds to this cutoff
            cutoff_pval = sorted_pvals[max_k - 1]
            cutoff_conf = stats.norm.isf(cutoff_pval / 2.0)  # inverse of 2*sf
            n_pass = max_k
        else:
            cutoff_conf = float('inf')
            n_pass = 0

        results[fdr] = {
            'n_pass': n_pass,
            'cutoff_confidence': float(cutoff_conf),
            'cutoff_pvalue': float(cutoff_pval) if n_pass > 0 else 0,
            'fraction': n_pass / n if n > 0 else 0,
        }

    return results, pvalues, confidences


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--all_human', action='store_true')
    parser.add_argument('--output', default=None, help='Output JSON file')
    args = parser.parse_args()

    if args.all_human:
        chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    else:
        print("Specify --all_human")
        sys.exit(1)

    print("Loading all drop regions...")
    all_regions, per_chrom = load_all_regions(args.results_dir, chroms)
    print(f"\nTotal regions across all chromosomes: {len(all_regions):,}")

    if not all_regions:
        print("No regions found!")
        sys.exit(1)

    # Confidence distribution
    confs = np.array([r['start_confidence'] for r in all_regions])
    print(f"\nConfidence distribution (all chromosomes combined):")
    print(f"  min: {confs.min():.2f}")
    print(f"  25th percentile: {np.percentile(confs, 25):.2f}")
    print(f"  median: {np.median(confs):.2f}")
    print(f"  75th percentile: {np.percentile(confs, 75):.2f}")
    print(f"  90th percentile: {np.percentile(confs, 90):.2f}")
    print(f"  95th percentile: {np.percentile(confs, 95):.2f}")
    print(f"  99th percentile: {np.percentile(confs, 99):.2f}")
    print(f"  max: {confs.max():.2f}")

    # FDR correction
    print(f"\n{'='*70}")
    print("FDR-Corrected Thresholds (Benjamini-Hochberg)")
    print(f"{'='*70}")
    fdr_results, pvalues, confidences = compute_fdr(all_regions)

    for fdr_level in sorted(fdr_results.keys()):
        r = fdr_results[fdr_level]
        print(f"\n  FDR = {fdr_level:.0%}:")
        print(f"    Regions passing: {r['n_pass']:,} / {len(all_regions):,} ({r['fraction']:.1%})")
        print(f"    Effective confidence cutoff: {r['cutoff_confidence']:.2f}")

    # Fixed threshold comparison
    print(f"\n{'='*70}")
    print("Fixed Confidence Threshold Comparison")
    print(f"{'='*70}")

    fixed_thresholds = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    for thresh in fixed_thresholds:
        n_pass = int(np.sum(confs >= thresh))
        n_chroms = len(set(r['chrom'] for r in all_regions if r['start_confidence'] >= thresh))
        print(f"\n  conf >= {thresh}:")
        print(f"    Regions: {n_pass:,} / {len(all_regions):,} ({n_pass/len(all_regions):.1%})")
        print(f"    Chromosomes represented: {n_chroms}/24")

    # Per-chromosome breakdown at key thresholds
    print(f"\n{'='*70}")
    print("Per-Chromosome Region Counts at Key Thresholds")
    print(f"{'='*70}")
    print(f"\n  {'Chrom':<8} {'Total':>8} {'>=4.0':>8} {'>=5.0':>8} {'>=8.0':>8} {'FDR5%':>8}")
    print(f"  {'-'*48}")

    fdr5_cutoff = fdr_results[0.05]['cutoff_confidence']
    total_4 = 0
    total_5 = 0
    total_8 = 0
    total_fdr = 0

    for chrom in chroms:
        if chrom not in per_chrom:
            print(f"  {chrom:<8} {'N/A':>8}")
            continue
        regs = per_chrom[chrom]
        c = np.array([r['start_confidence'] for r in regs])
        n_total = len(c)
        n_4 = int(np.sum(c >= 4.0))
        n_5 = int(np.sum(c >= 5.0))
        n_8 = int(np.sum(c >= 8.0))
        n_fdr = int(np.sum(c >= fdr5_cutoff))
        total_4 += n_4
        total_5 += n_5
        total_8 += n_8
        total_fdr += n_fdr
        print(f"  {chrom:<8} {n_total:>8,} {n_4:>8,} {n_5:>8,} {n_8:>8,} {n_fdr:>8,}")

    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<8} {len(all_regions):>8,} {total_4:>8,} {total_5:>8,} {total_8:>8,} {total_fdr:>8,}")

    # GPU compute estimates
    print(f"\n{'='*70}")
    print("GPU Compute Estimates for SAE Re-run")
    print(f"{'='*70}")

    # Rough estimate: ~2500 regions per shard, ~30 min per shard on GPU
    regions_per_gpu_hour = 5000  # approximate

    for label, count in [('conf >= 8.0', total_8),
                         ('conf >= 5.0', total_5),
                         ('conf >= 4.0', total_4),
                         ('FDR 5%', total_fdr)]:
        gpu_hours = count / regions_per_gpu_hour
        wall_hours_36gpu = gpu_hours / 36  # 36 parallel shards
        print(f"\n  {label}: {count:,} regions")
        print(f"    GPU-hours: ~{gpu_hours:.0f}")
        print(f"    Wall time (36 parallel GPUs): ~{wall_hours_36gpu:.1f} hours")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'total_regions': len(all_regions),
        'n_chromosomes': len(per_chrom),
        'confidence_distribution': {
            'min': float(confs.min()),
            'p25': float(np.percentile(confs, 25)),
            'median': float(np.median(confs)),
            'p75': float(np.percentile(confs, 75)),
            'p90': float(np.percentile(confs, 90)),
            'p95': float(np.percentile(confs, 95)),
            'p99': float(np.percentile(confs, 99)),
            'max': float(confs.max()),
        },
        'fdr_results': {str(k): v for k, v in fdr_results.items()},
        'fixed_thresholds': {
            str(t): int(np.sum(confs >= t)) for t in fixed_thresholds
        },
        'per_chrom_at_fdr5': {
            chrom: int(np.sum(np.array([r['start_confidence'] for r in regs]) >= fdr5_cutoff))
            for chrom, regs in per_chrom.items()
        },
    }

    out_path = args.output or os.path.join(args.results_dir, 'fdr_threshold_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
