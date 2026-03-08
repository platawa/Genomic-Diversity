#!/usr/bin/env python3
"""
deep_locus_comparison.py

Deep comparison of all 6 detection methods on a single locus.
Runs each method with default params, then sweeps key parameters
and plots precision/recall curves using GTF boundaries as ground truth.

Usage:
    python tools/deep_locus_comparison.py \
        --entropy results/chr22/scoring/.../data/entropy.npz \
        --gtf /path/to/genomic.gtf \
        --chrom NC_000022.11 \
        --gene_name EWSR1 \
        [--padding 5000] \
        [--output_dir results/chr22/deep_comparison_EWSR1/]

    # Or specify coordinates directly:
    python tools/deep_locus_comparison.py \
        --entropy results/chr22/scoring/.../data/entropy.npz \
        --region_start 29663998 --region_end 29696515 \
        --gtf /path/to/genomic.gtf --chrom NC_000022.11 \
        --output_dir results/chr22/deep_comparison_region/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from detection_methods import METHODS, METHOD_DEFAULTS, run_method


def load_entropy_region(npz_path: str, start: int, end: int) -> Tuple[np.ndarray, int]:
    """Load a sub-region of the entropy array."""
    data = np.load(npz_path, allow_pickle=True)
    entropy = data['entropy']
    genomic_start = int(data['start']) if 'start' in data else 0

    # Convert genomic coordinates to array indices
    arr_start = max(0, start - genomic_start)
    arr_end = min(len(entropy), end - genomic_start)

    return entropy[arr_start:arr_end], genomic_start + arr_start


def find_gene_coordinates(gtf_path: str, chrom: str, gene_name: str) -> Optional[Tuple[int, int]]:
    """Find gene start/end from GTF."""
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[0] != chrom or fields[2] != 'gene':
                continue
            attrs = fields[8]
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name') or attr.startswith('gene_id'):
                    parts = attr.split('"')
                    if len(parts) >= 2 and parts[1] == gene_name:
                        return (int(fields[3]) - 1, int(fields[4]))
    return None


def extract_ground_truth(gtf_path: str, chrom: str,
                         region_start: int, region_end: int,
                         feature_types: List[str] = ['CDS', 'exon']
                         ) -> List[Tuple[int, str]]:
    """Extract ground truth boundary positions within a region.

    Returns list of (array_position, direction) tuples where direction
    is 'drop' for feature starts or 'rise' for feature ends (+ strand).
    """
    boundaries = []
    seen = set()

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[0] != chrom:
                continue
            feat_type = fields[2]
            if feat_type not in feature_types:
                continue

            feat_start = int(fields[3]) - 1
            feat_end = int(fields[4])
            strand = fields[6]

            if feat_end < region_start or feat_start > region_end:
                continue

            # Convert to array-relative positions
            arr_start = feat_start - region_start
            arr_end = feat_end - region_start

            if strand == '+':
                start_dir, end_dir = 'drop', 'rise'
            else:
                start_dir, end_dir = 'rise', 'drop'

            key_s = (arr_start, feat_type, 'start')
            if key_s not in seen and 0 <= arr_start:
                boundaries.append((arr_start, start_dir))
                seen.add(key_s)

            key_e = (arr_end, feat_type, 'end')
            if key_e not in seen and arr_end <= (region_end - region_start):
                boundaries.append((arr_end, end_dir))
                seen.add(key_e)

    boundaries.sort()
    return boundaries


def evaluate_detection(drops: List[Tuple[int, float]],
                       ground_truth: List[Tuple[int, str]],
                       tolerance: int,
                       direction: str = 'drop') -> Dict[str, float]:
    """Evaluate detected drops against ground truth boundaries."""
    gt_positions = [pos for pos, d in ground_truth if d == direction]
    det_positions = [pos for pos, _ in drops]

    if not gt_positions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(det_positions), 'fn': 0}

    tp = 0
    matched_gt = set()
    for dp in det_positions:
        for i, gp in enumerate(gt_positions):
            if i not in matched_gt and abs(dp - gp) <= tolerance:
                tp += 1
                matched_gt.add(i)
                break

    fp = len(det_positions) - tp
    fn = len(gt_positions) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


# Parameter sweep ranges for each method
PARAM_SWEEPS = {
    'zscore': {
        'zscore_threshold': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'smooth_w': [21, 31, 51, 75, 101],
    },
    'mad': {
        'mad_threshold': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
        'smooth_w': [21, 31, 51, 75, 101],
    },
    'derivative': {
        'thr_quantile': [0.005, 0.01, 0.02, 0.05],
        'smooth_w': [21, 31, 51, 75, 101],
    },
    'window_mean_shift': {
        'w': [50, 100, 200, 500],
        'top_k': [50, 100, 200, 500],
    },
    'cusum': {
        'h': [1.0, 2.0, 5.0, 10.0, 20.0],
        'smooth_w': [21, 31, 51, 75, 101],
    },
    'local_baseline': {
        'window_baseline': [200, 500, 1000],
        'threshold_sigma': [1.5, 2.0, 2.5, 3.0],
    },
}


def sweep_method(method_name: str, entropy: np.ndarray,
                 ground_truth: List[Tuple[int, str]],
                 tolerance: int = 100) -> List[Dict]:
    """Sweep parameters for one method, evaluate each combo."""
    sweep_params = PARAM_SWEEPS[method_name]
    defaults = METHOD_DEFAULTS[method_name].copy()
    param_names = list(sweep_params.keys())

    # Generate all combinations
    import itertools
    values = [sweep_params[p] for p in param_names]
    results = []

    for combo in itertools.product(*values):
        params = defaults.copy()
        for pname, pval in zip(param_names, combo):
            params[pname] = pval

        try:
            drops = METHODS[method_name](entropy, **params)
        except Exception:
            continue

        metrics = evaluate_detection(drops, ground_truth, tolerance, direction='drop')
        result = {
            'method': method_name,
            'params': {pname: pval for pname, pval in zip(param_names, combo)},
            'n_drops': len(drops),
            **metrics,
        }
        results.append(result)

    return results


def plot_locus_comparison(entropy: np.ndarray,
                          all_drops: Dict[str, List[Tuple[int, float]]],
                          ground_truth: List[Tuple[int, str]],
                          output_dir: str,
                          region_start: int = 0,
                          title: str = ''):
    """Plot all methods overlaid on entropy with ground truth annotations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    colors = {
        'zscore': '#e41a1c', 'mad': '#377eb8', 'derivative': '#4daf4a',
        'window_mean_shift': '#984ea3', 'cusum': '#ff7f00', 'local_baseline': '#a65628',
    }

    n_methods = len(all_drops)
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(16, 3 * (n_methods + 1)), sharex=True)
    x = np.arange(len(entropy)) + region_start

    # Top: entropy + ground truth
    axes[0].plot(x, entropy, color='gray', linewidth=0.5)
    drop_gt = [pos + region_start for pos, d in ground_truth if d == 'drop']
    rise_gt = [pos + region_start for pos, d in ground_truth if d == 'rise']
    for gp in drop_gt:
        axes[0].axvline(gp, color='green', alpha=0.3, linewidth=0.5)
    for gp in rise_gt:
        axes[0].axvline(gp, color='red', alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel('Entropy')
    axes[0].set_title(f'{title} — Entropy with GTF boundaries (green=drop, red=rise)')

    # Per-method panels
    for idx, (method, drops) in enumerate(sorted(all_drops.items()), 1):
        axes[idx].plot(x, entropy, color='gray', linewidth=0.3, alpha=0.5)
        color = colors.get(method, 'black')
        dp = [p + region_start for p, _ in drops]
        ds = [entropy[p] if 0 <= p < len(entropy) else 0 for p, _ in drops]
        if dp:
            axes[idx].scatter(dp, ds, color=color, s=20, zorder=5, alpha=0.8)
        for gp in drop_gt:
            axes[idx].axvline(gp, color='green', alpha=0.15, linewidth=0.5)
        axes[idx].set_ylabel(method)
        axes[idx].text(0.02, 0.85, f'n={len(drops)}', transform=axes[idx].transAxes, fontsize=9)

    axes[-1].set_xlabel('Genomic position')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'locus_overlay.png'), dpi=150)
    plt.close()


def plot_pr_curves(sweep_results: Dict[str, List[Dict]], output_dir: str):
    """Plot precision-recall curves from parameter sweeps."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {
        'zscore': '#e41a1c', 'mad': '#377eb8', 'derivative': '#4daf4a',
        'window_mean_shift': '#984ea3', 'cusum': '#ff7f00', 'local_baseline': '#a65628',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for method, results in sorted(sweep_results.items()):
        if not results:
            continue
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1s = [r['f1'] for r in results]

        color = colors.get(method, 'black')
        ax.scatter(recalls, precisions, color=color, s=15, alpha=0.5, label=method)

        # Highlight best F1
        best_idx = max(range(len(f1s)), key=lambda i: f1s[i])
        ax.scatter([recalls[best_idx]], [precisions[best_idx]],
                   color=color, s=80, marker='*', edgecolors='black', zorder=10)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall across parameter sweeps (star = best F1)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Deep comparison of all detection methods on a single locus"
    )
    ap.add_argument("--entropy", required=True, help="Path to entropy.npz")
    ap.add_argument("--gtf", required=True, help="GTF annotation file")
    ap.add_argument("--chrom", required=True, help="Chromosome name in GTF")
    ap.add_argument("--gene_name", default=None, help="Gene name to analyze")
    ap.add_argument("--region_start", type=int, default=None, help="Region start (genomic)")
    ap.add_argument("--region_end", type=int, default=None, help="Region end (genomic)")
    ap.add_argument("--padding", type=int, default=5000,
                    help="bp padding around gene (default: 5000)")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="bp tolerance for matching drops to ground truth (default: 100)")
    ap.add_argument("--output_dir", default=None, help="Output directory")
    ap.add_argument("--test_loci_json", default=None,
                    help="JSON from curate_test_loci.py — run all loci in one go")
    args = ap.parse_args()

    if args.test_loci_json:
        # Batch mode: process all loci
        with open(args.test_loci_json) as f:
            all_loci = json.load(f)

        for organism, loci in all_loci.items():
            for locus in loci:
                print(f"\n{'='*60}")
                print(f"Processing {organism}/{locus['gene_name']}")
                print(f"{'='*60}")
                out_dir = os.path.join(
                    args.output_dir or 'deep_comparison',
                    organism, locus['gene_name']
                )
                run_single_locus(
                    args.entropy, args.gtf, locus['chrom'],
                    locus['start'] - args.padding,
                    locus['end'] + args.padding,
                    locus['gene_name'], args.tolerance, out_dir
                )
        return

    # Single locus mode
    if args.gene_name:
        coords = find_gene_coordinates(args.gtf, args.chrom, args.gene_name)
        if coords is None:
            print(f"ERROR: Gene '{args.gene_name}' not found in GTF for {args.chrom}")
            sys.exit(1)
        region_start = coords[0] - args.padding
        region_end = coords[1] + args.padding
        title = args.gene_name
    elif args.region_start is not None and args.region_end is not None:
        region_start = args.region_start
        region_end = args.region_end
        title = f"{args.chrom}:{region_start}-{region_end}"
    else:
        print("ERROR: Provide --gene_name or --region_start/--region_end")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = f"deep_comparison_{title.replace(':', '_')}"

    run_single_locus(args.entropy, args.gtf, args.chrom,
                     region_start, region_end, title,
                     args.tolerance, args.output_dir)


def run_single_locus(entropy_path: str, gtf_path: str, chrom: str,
                     region_start: int, region_end: int,
                     title: str, tolerance: int, output_dir: str):
    """Run deep comparison on a single locus."""
    os.makedirs(output_dir, exist_ok=True)

    # Load entropy sub-region
    print(f"Loading entropy for {chrom}:{region_start}-{region_end}")
    entropy, actual_start = load_entropy_region(entropy_path, region_start, region_end)
    print(f"  Array length: {len(entropy):,} positions")

    # Extract ground truth
    print(f"Extracting ground truth from GTF...")
    ground_truth = extract_ground_truth(gtf_path, chrom, actual_start,
                                        actual_start + len(entropy))
    n_drops = sum(1 for _, d in ground_truth if d == 'drop')
    n_rises = sum(1 for _, d in ground_truth if d == 'rise')
    print(f"  {len(ground_truth)} boundaries ({n_drops} drops, {n_rises} rises)")

    # Run all methods with defaults
    print(f"\nRunning all methods with default parameters...")
    all_drops = {}
    default_results = {}
    for method_name in METHODS:
        drops = run_method(method_name, entropy)
        all_drops[method_name] = drops
        metrics = evaluate_detection(drops, ground_truth, tolerance, direction='drop')
        default_results[method_name] = {
            'n_drops': len(drops), **metrics,
            'params': METHOD_DEFAULTS[method_name],
        }
        print(f"  {method_name:20s}: {len(drops):4d} drops, "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    # Parameter sweeps
    print(f"\nRunning parameter sweeps...")
    sweep_results = {}
    for method_name in METHODS:
        print(f"  Sweeping {method_name}...")
        results = sweep_method(method_name, entropy, ground_truth, tolerance)
        sweep_results[method_name] = results
        if results:
            best = max(results, key=lambda r: r['f1'])
            print(f"    Best F1={best['f1']:.3f} with {best['params']}")

    # Generate plots
    plot_locus_comparison(entropy, all_drops, ground_truth, output_dir,
                          region_start=actual_start, title=title)
    plot_pr_curves(sweep_results, output_dir)

    # Save results
    summary = {
        'locus': title,
        'chrom': chrom,
        'region_start': actual_start,
        'region_end': actual_start + len(entropy),
        'n_positions': len(entropy),
        'n_ground_truth_drops': n_drops,
        'n_ground_truth_rises': n_rises,
        'tolerance_bp': tolerance,
        'default_results': default_results,
        'best_per_method': {},
    }
    for method, results in sweep_results.items():
        if results:
            best = max(results, key=lambda r: r['f1'])
            summary['best_per_method'][method] = best

    summary_path = os.path.join(output_dir, 'deep_comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {summary_path}")

    # Save full sweep results as TSV
    tsv_path = os.path.join(output_dir, 'parameter_sweep.tsv')
    with open(tsv_path, 'w') as f:
        f.write('method\tparams\tn_drops\tprecision\trecall\tf1\ttp\tfp\tfn\n')
        for method, results in sorted(sweep_results.items()):
            for r in results:
                params_str = json.dumps(r['params'])
                f.write(f"{method}\t{params_str}\t{r['n_drops']}\t"
                        f"{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t"
                        f"{r['tp']}\t{r['fp']}\t{r['fn']}\n")
    print(f"Sweep results: {tsv_path}")


if __name__ == "__main__":
    main()
