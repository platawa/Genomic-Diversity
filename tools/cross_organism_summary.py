#!/usr/bin/env python3
"""
cross_organism_summary.py

Aggregate statistics across all scored chromosomes/organisms.
Produces cross-organism comparison table and plots.

Usage:
    python tools/cross_organism_summary.py \
        --results_dir results/ \
        [--output_dir results/_cross_organism/]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from results_utils import find_latest_completed


def discover_completed_runs(results_dir: str) -> Dict[str, Dict]:
    """Find all completed scoring runs organized by chromosome."""
    runs = {}
    results_path = Path(results_dir)

    for chrom_dir in sorted(results_path.iterdir()):
        if not chrom_dir.is_dir() or chrom_dir.name.startswith('_'):
            continue

        chrom = chrom_dir.name
        scoring_dir = chrom_dir / 'scoring'
        if not scoring_dir.is_dir():
            continue

        # Find latest completed run
        latest = find_latest_completed(results_dir, chrom, 'scoring')
        if latest is None:
            continue

        # Load metadata
        data_dir = os.path.join(latest, 'data')
        entropy_path = os.path.join(data_dir, 'entropy.npz')
        summary_path = os.path.join(data_dir, 'summary.json')

        if not os.path.exists(entropy_path):
            continue

        run_info = {
            'chrom': chrom,
            'run_dir': latest,
            'entropy_path': entropy_path,
        }

        if os.path.exists(summary_path):
            with open(summary_path) as f:
                run_info['summary'] = json.load(f)

        runs[chrom] = run_info

    return runs


def compute_entropy_stats(entropy_path: str) -> Dict:
    """Compute summary statistics for an entropy array."""
    data = np.load(entropy_path, allow_pickle=True)
    entropy = data['entropy']
    valid = ~np.isnan(entropy)

    if valid.sum() == 0:
        return {"n_positions": len(entropy), "n_valid": 0}

    e = entropy[valid]
    return {
        "n_positions": int(len(entropy)),
        "n_valid": int(valid.sum()),
        "n_nan": int((~valid).sum()),
        "pct_valid": float(valid.mean() * 100),
        "mean_entropy": float(np.mean(e)),
        "median_entropy": float(np.median(e)),
        "std_entropy": float(np.std(e)),
        "min_entropy": float(np.min(e)),
        "max_entropy": float(np.max(e)),
        "q25_entropy": float(np.percentile(e, 25)),
        "q75_entropy": float(np.percentile(e, 75)),
    }


def count_drops(entropy_path: str) -> Dict:
    """Count drops using zscore and mad methods."""
    from detection_methods import detect_drops_zscore, detect_drops_mad

    data = np.load(entropy_path, allow_pickle=True)
    entropy = data['entropy']

    drops_z = detect_drops_zscore(entropy)
    drops_m = detect_drops_mad(entropy)

    n_positions = len(entropy)
    length_mbp = n_positions / 1e6

    return {
        "zscore_drops": len(drops_z),
        "mad_drops": len(drops_m),
        "zscore_drops_per_mbp": len(drops_z) / length_mbp if length_mbp > 0 else 0,
        "mad_drops_per_mbp": len(drops_m) / length_mbp if length_mbp > 0 else 0,
        "length_mbp": float(length_mbp),
    }


def classify_organism(chrom: str) -> str:
    """Classify chromosome by organism based on naming convention."""
    if chrom.startswith('chr') or chrom.startswith('NC_0000'):
        return 'human'
    elif 'NC_000913' in chrom or 'ecoli' in chrom.lower():
        return 'ecoli'
    elif 'NC_000964' in chrom or 'bacillus' in chrom.lower():
        return 'bacillus'
    return 'unknown'


def main():
    ap = argparse.ArgumentParser(
        description="Cross-organism summary of scoring results"
    )
    ap.add_argument("--results_dir", default="results",
                    help="Root results directory (default: results)")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: results/_cross_organism)")
    ap.add_argument("--skip_drop_counting", action="store_true",
                    help="Skip drop counting (faster, entropy stats only)")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, '_cross_organism')
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover runs
    print(f"Scanning {args.results_dir} for completed scoring runs...")
    runs = discover_completed_runs(args.results_dir)
    print(f"Found {len(runs)} completed runs")

    if not runs:
        print("No completed scoring runs found.")
        return

    # Compute statistics
    all_stats = {}
    organism_stats = defaultdict(list)

    for chrom, run_info in sorted(runs.items()):
        print(f"\n  {chrom}:")
        entropy_stats = compute_entropy_stats(run_info['entropy_path'])
        mean_e = entropy_stats.get('mean_entropy', None)
        mean_str = f"{mean_e:.4f}" if mean_e is not None else "N/A"
        print(f"    {entropy_stats['n_valid']:,} valid positions, "
              f"mean entropy={mean_str}")

        entry = {
            'chrom': chrom,
            'organism': classify_organism(chrom),
            'entropy': entropy_stats,
        }

        if not args.skip_drop_counting:
            drop_stats = count_drops(run_info['entropy_path'])
            entry['drops'] = drop_stats
            print(f"    zscore: {drop_stats['zscore_drops']} drops "
                  f"({drop_stats['zscore_drops_per_mbp']:.1f}/Mbp), "
                  f"MAD: {drop_stats['mad_drops']} drops "
                  f"({drop_stats['mad_drops_per_mbp']:.1f}/Mbp)")

        all_stats[chrom] = entry
        organism_stats[entry['organism']].append(entry)

    # Organism-level aggregation
    print(f"\n{'='*70}")
    print("Cross-organism summary")
    print(f"{'='*70}")

    organism_summary = {}
    for org, entries in sorted(organism_stats.items()):
        total_mbp = sum(e['entropy']['n_valid'] / 1e6 for e in entries)
        mean_entropy = np.mean([e['entropy']['mean_entropy'] for e in entries
                                if 'mean_entropy' in e['entropy']])
        std_entropy = np.mean([e['entropy']['std_entropy'] for e in entries
                               if 'std_entropy' in e['entropy']])

        org_info = {
            "n_chromosomes": len(entries),
            "total_mbp": float(total_mbp),
            "mean_entropy": float(mean_entropy),
            "mean_std_entropy": float(std_entropy),
        }

        if not args.skip_drop_counting:
            total_z_drops = sum(e['drops']['zscore_drops'] for e in entries)
            total_m_drops = sum(e['drops']['mad_drops'] for e in entries)
            org_info["total_zscore_drops"] = total_z_drops
            org_info["total_mad_drops"] = total_m_drops
            org_info["zscore_drops_per_mbp"] = total_z_drops / total_mbp if total_mbp > 0 else 0
            org_info["mad_drops_per_mbp"] = total_m_drops / total_mbp if total_mbp > 0 else 0

        organism_summary[org] = org_info

        print(f"\n  {org}:")
        print(f"    Chromosomes: {len(entries)}")
        print(f"    Total: {total_mbp:.1f} Mbp")
        print(f"    Mean entropy: {mean_entropy:.4f} +/- {std_entropy:.4f}")
        if not args.skip_drop_counting:
            print(f"    Z-score drops: {org_info['total_zscore_drops']} "
                  f"({org_info['zscore_drops_per_mbp']:.1f}/Mbp)")
            print(f"    MAD drops: {org_info['total_mad_drops']} "
                  f"({org_info['mad_drops_per_mbp']:.1f}/Mbp)")

    # Save results
    output = {
        "per_chromosome": all_stats,
        "per_organism": organism_summary,
    }
    output_path = os.path.join(args.output_dir, 'cross_organism_summary.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to {output_path}")

    # Write TSV table
    tsv_path = os.path.join(args.output_dir, 'cross_organism_table.tsv')
    with open(tsv_path, 'w') as f:
        headers = ['organism', 'chrom', 'length_mbp', 'mean_entropy', 'std_entropy']
        if not args.skip_drop_counting:
            headers += ['zscore_drops', 'mad_drops', 'zscore_per_mbp', 'mad_per_mbp']
        f.write('\t'.join(headers) + '\n')

        for chrom, entry in sorted(all_stats.items(), key=lambda x: (x[1]['organism'], x[0])):
            row = [
                entry['organism'],
                chrom,
                f"{entry['entropy']['n_valid']/1e6:.2f}",
                f"{entry['entropy'].get('mean_entropy', 0):.4f}",
                f"{entry['entropy'].get('std_entropy', 0):.4f}",
            ]
            if not args.skip_drop_counting:
                row += [
                    str(entry['drops']['zscore_drops']),
                    str(entry['drops']['mad_drops']),
                    f"{entry['drops']['zscore_drops_per_mbp']:.1f}",
                    f"{entry['drops']['mad_drops_per_mbp']:.1f}",
                ]
            f.write('\t'.join(row) + '\n')
    print(f"Table saved to {tsv_path}")

    # Generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Bar chart: drops per Mbp by organism
        if not args.skip_drop_counting and len(organism_summary) > 1:
            orgs = list(organism_summary.keys())
            z_rates = [organism_summary[o]['zscore_drops_per_mbp'] for o in orgs]
            m_rates = [organism_summary[o]['mad_drops_per_mbp'] for o in orgs]

            x = np.arange(len(orgs))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width/2, z_rates, width, label='Z-score')
            ax.bar(x + width/2, m_rates, width, label='MAD')
            ax.set_ylabel('Drops per Mbp')
            ax.set_title('Drop density by organism')
            ax.set_xticks(x)
            ax.set_xticklabels(orgs)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'drops_per_mbp.png'), dpi=150)
            plt.close()

        # Entropy distribution by organism
        if len(organism_summary) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            orgs = list(organism_summary.keys())
            means = [organism_summary[o]['mean_entropy'] for o in orgs]
            stds = [organism_summary[o]['mean_std_entropy'] for o in orgs]
            ax.bar(orgs, means, yerr=stds, capsize=5)
            ax.set_ylabel('Mean entropy')
            ax.set_title('Mean entropy by organism')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'entropy_by_organism.png'), dpi=150)
            plt.close()

        print(f"Plots saved to {args.output_dir}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
