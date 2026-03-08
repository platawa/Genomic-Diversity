#!/usr/bin/env python3
"""
compare_ablations.py

Compare 2+ entropy.npz files from different scoring runs (ablations).
Computes correlation, per-position differences, detection count changes.

Usage:
    python tools/compare_ablations.py \
        --baseline results/chr22/scoring/.../data/entropy.npz \
        --ablations results/chr22/scoring/.../data/entropy.npz \
                    results/chr22/scoring/.../data/entropy.npz \
        [--labels "baseline" "no_rc" "stitch_mean"] \
        [--output_dir results/chr22/ablation_comparison/]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict
from scipy import stats

from detection_methods import detect_drops_zscore, detect_drops_mad


def load_entropy(npz_path: str) -> np.ndarray:
    """Load entropy array from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return data['entropy']


def compute_correlation(a: np.ndarray, b: np.ndarray):
    """Compute Pearson and Spearman correlation between two entropy arrays."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 10:
        return {"pearson_r": float('nan'), "spearman_rho": float('nan'), "n_valid": int(valid.sum())}

    a_valid, b_valid = a[valid], b[valid]
    pearson_r, pearson_p = stats.pearsonr(a_valid, b_valid)
    spearman_rho, spearman_p = stats.spearmanr(a_valid, b_valid)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n_valid": int(valid.sum()),
    }


def compute_diff_stats(a: np.ndarray, b: np.ndarray):
    """Compute per-position difference statistics."""
    valid = ~(np.isnan(a) | np.isnan(b))
    diff = np.full_like(a, np.nan)
    diff[valid] = b[valid] - a[valid]

    valid_diff = diff[valid]
    if len(valid_diff) == 0:
        return {"mean_diff": float('nan'), "std_diff": float('nan')}

    return {
        "mean_diff": float(np.mean(valid_diff)),
        "std_diff": float(np.std(valid_diff)),
        "median_diff": float(np.median(valid_diff)),
        "max_abs_diff": float(np.max(np.abs(valid_diff))),
        "pct_within_0.01": float(np.mean(np.abs(valid_diff) < 0.01) * 100),
        "pct_within_0.1": float(np.mean(np.abs(valid_diff) < 0.1) * 100),
    }


def count_detection_changes(baseline: np.ndarray, ablation: np.ndarray):
    """Compare drop counts between baseline and ablation."""
    drops_base_z = detect_drops_zscore(baseline)
    drops_abl_z = detect_drops_zscore(ablation)
    drops_base_m = detect_drops_mad(baseline)
    drops_abl_m = detect_drops_mad(ablation)

    return {
        "zscore_baseline": len(drops_base_z),
        "zscore_ablation": len(drops_abl_z),
        "zscore_change": len(drops_abl_z) - len(drops_base_z),
        "mad_baseline": len(drops_base_m),
        "mad_ablation": len(drops_abl_m),
        "mad_change": len(drops_abl_m) - len(drops_base_m),
    }


def plot_comparison(baseline: np.ndarray, ablations: Dict[str, np.ndarray],
                    output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    labels = list(ablations.keys())

    # --- Scatter plots ---
    n_ablations = len(ablations)
    fig, axes = plt.subplots(1, n_ablations, figsize=(6 * n_ablations, 5))
    if n_ablations == 1:
        axes = [axes]

    for ax, (label, abl) in zip(axes, ablations.items()):
        valid = ~(np.isnan(baseline) | np.isnan(abl))
        # Subsample for plotting
        n_plot = min(50000, valid.sum())
        idx = np.where(valid)[0]
        if len(idx) > n_plot:
            idx = np.random.choice(idx, n_plot, replace=False)

        ax.scatter(baseline[idx], abl[idx], s=1, alpha=0.1)
        ax.plot([0, np.nanmax(baseline)], [0, np.nanmax(baseline)],
                'r--', linewidth=0.5, alpha=0.5)
        corr = compute_correlation(baseline, abl)
        ax.set_title(f'vs {label}\nr={corr["pearson_r"]:.4f}')
        ax.set_xlabel('Baseline entropy')
        ax.set_ylabel(f'{label} entropy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_comparison.png'), dpi=150)
    plt.close()

    # --- Difference histogram ---
    fig, axes = plt.subplots(1, n_ablations, figsize=(6 * n_ablations, 4))
    if n_ablations == 1:
        axes = [axes]

    for ax, (label, abl) in zip(axes, ablations.items()):
        valid = ~(np.isnan(baseline) | np.isnan(abl))
        diff = abl[valid] - baseline[valid]
        ax.hist(diff, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=0.5)
        ax.set_title(f'Diff: {label} - baseline\nmean={np.mean(diff):.4f}')
        ax.set_xlabel('Entropy difference')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diff_histogram.png'), dpi=150)
    plt.close()

    # --- Side-by-side entropy traces at interesting regions ---
    # Find region with largest absolute difference
    all_diffs = []
    for label, abl in ablations.items():
        valid = ~(np.isnan(baseline) | np.isnan(abl))
        diff = np.zeros_like(baseline)
        diff[valid] = np.abs(abl[valid] - baseline[valid])
        all_diffs.append(diff)

    max_diff = np.nanmax(np.array(all_diffs), axis=0)
    # Smooth to find regions, not individual spikes
    from detection_methods import _rolling_mean
    smoothed_diff = _rolling_mean(np.nan_to_num(max_diff), 1000)
    interesting_pos = int(np.argmax(smoothed_diff))
    window = 5000
    lo = max(0, interesting_pos - window)
    hi = min(len(baseline), interesting_pos + window)

    fig, axes = plt.subplots(n_ablations + 1, 1,
                             figsize=(16, 3 * (n_ablations + 1)), sharex=True)
    x = np.arange(lo, hi)
    axes[0].plot(x, baseline[lo:hi], color='black', linewidth=0.5)
    axes[0].set_ylabel('Baseline')
    axes[0].set_title(f'Region with largest differences ({lo:,}-{hi:,})')

    for idx, (label, abl) in enumerate(ablations.items(), 1):
        axes[idx].plot(x, abl[lo:hi], color='blue', linewidth=0.5, alpha=0.7)
        axes[idx].plot(x, baseline[lo:hi], color='black', linewidth=0.3, alpha=0.3)
        axes[idx].set_ylabel(label)

    axes[-1].set_xlabel('Position')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trace_comparison.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare entropy.npz files from different scoring ablations"
    )
    ap.add_argument("--baseline", required=True,
                    help="Path to baseline entropy.npz")
    ap.add_argument("--ablations", nargs='+', required=True,
                    help="Paths to ablation entropy.npz files")
    ap.add_argument("--labels", nargs='+', default=None,
                    help="Labels for ablation runs (default: filenames)")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: ablation_comparison/ next to baseline)")
    args = ap.parse_args()

    # Load baseline
    print(f"Loading baseline: {args.baseline}")
    baseline = load_entropy(args.baseline)
    print(f"  Length: {len(baseline):,}")

    # Load ablations
    if args.labels is None:
        args.labels = [Path(p).parent.parent.name for p in args.ablations]

    ablations = {}
    for path, label in zip(args.ablations, args.labels):
        print(f"Loading ablation '{label}': {path}")
        abl = load_entropy(path)
        if len(abl) != len(baseline):
            print(f"  WARNING: length mismatch ({len(abl)} vs {len(baseline)}), truncating")
            min_len = min(len(abl), len(baseline))
            abl = abl[:min_len]
            baseline = baseline[:min_len]
        ablations[label] = abl

    # Setup output
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.baseline), 'ablation_comparison')
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute statistics
    results = {}
    for label, abl in ablations.items():
        print(f"\n{'='*60}")
        print(f"Comparison: baseline vs {label}")
        print(f"{'='*60}")

        corr = compute_correlation(baseline, abl)
        diff = compute_diff_stats(baseline, abl)
        det = count_detection_changes(baseline, abl)

        print(f"  Pearson r:    {corr['pearson_r']:.6f}")
        print(f"  Spearman rho: {corr['spearman_rho']:.6f}")
        print(f"  Mean diff:    {diff['mean_diff']:.6f}")
        print(f"  Max |diff|:   {diff['max_abs_diff']:.6f}")
        print(f"  Within 0.01:  {diff['pct_within_0.01']:.1f}%")
        print(f"  Z-score drops: {det['zscore_baseline']} -> {det['zscore_ablation']} "
              f"({det['zscore_change']:+d})")
        print(f"  MAD drops:     {det['mad_baseline']} -> {det['mad_ablation']} "
              f"({det['mad_change']:+d})")

        results[label] = {
            "correlation": corr,
            "difference": diff,
            "detection_changes": det,
        }

    # Save results
    results_path = os.path.join(args.output_dir, 'ablation_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {results_path}")

    # Generate plots
    plot_comparison(baseline, ablations, args.output_dir)


if __name__ == "__main__":
    main()
