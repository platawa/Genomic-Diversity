#!/usr/bin/env python3
"""
optimize_consensus_based.py

Parameter optimization using cross-method consensus instead of ground truth.

Strategy:
    - Run all 6 detection methods with varying parameters
    - Measure agreement between methods (drops within +/-tolerance)
    - Optimize for high consensus + reasonable drop count
    - No ground truth annotation required

Usage:
    python tools/optimize_consensus_based.py \
        --entropy results/chr22/scoring/.../data/entropy.npz \
        --output consensus_optimal_params.json \
        [--tolerance 100]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from detection_methods import METHODS, METHOD_DEFAULTS


@dataclass
class ConsensusResult:
    """Results of consensus-based parameter evaluation."""
    method: str
    params: Dict[str, Any]
    n_drops: int
    consensus_fraction: float  # fraction of drops confirmed by >= 2 other methods
    avg_n_confirming: float    # average number of other methods confirming each drop
    quality_score: float       # combined metric


def count_consensus(target_drops: List[Tuple[int, float]],
                    other_method_drops: Dict[str, List[Tuple[int, float]]],
                    tolerance: int = 100) -> Tuple[float, float]:
    """Count how many target drops are confirmed by other methods.

    Returns (consensus_fraction, avg_n_confirming).
    """
    if not target_drops:
        return 0.0, 0.0

    target_positions = sorted([p for p, _ in target_drops])

    # Pre-sort other method positions for binary search
    other_sorted = {}
    for method, drops in other_method_drops.items():
        positions = np.array(sorted([p for p, _ in drops]), dtype=np.int64)
        other_sorted[method] = positions

    confirmed = 0
    total_confirming = 0

    for tp in target_positions:
        n_confirming = 0
        for method, positions in other_sorted.items():
            if len(positions) == 0:
                continue
            idx = np.searchsorted(positions, tp)
            if idx < len(positions) and abs(positions[idx] - tp) <= tolerance:
                n_confirming += 1
            elif idx > 0 and abs(positions[idx - 1] - tp) <= tolerance:
                n_confirming += 1

        total_confirming += n_confirming
        if n_confirming >= 2:
            confirmed += 1

    consensus_frac = confirmed / len(target_positions)
    avg_confirming = total_confirming / len(target_positions)

    return consensus_frac, avg_confirming


# Parameter grids
PARAM_GRIDS = {
    'zscore': [
        {'smooth_w': sw, 'zscore_threshold': t, 'min_separation': 75}
        for t in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        for sw in [21, 31, 51, 75, 101]
    ],
    'mad': [
        {'smooth_w': sw, 'mad_threshold': t, 'min_separation': 75}
        for t in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        for sw in [21, 31, 51, 75, 101]
    ],
    'derivative': [
        {'smooth_w': sw, 'thr_quantile': q, 'min_separation': 75}
        for q in [0.005, 0.01, 0.02, 0.05]
        for sw in [21, 31, 51, 75, 101]
    ],
    'window_mean_shift': [
        {'w': w, 'top_k': k, 'min_separation': 75}
        for w in [50, 100, 200, 500]
        for k in [50, 100, 200, 500]
    ],
    'cusum': [
        {'smooth_w': sw, 'h': h, 'min_separation': 75}
        for h in [1.0, 2.0, 5.0, 10.0, 20.0]
        for sw in [21, 31, 51, 75, 101]
    ],
    'local_baseline': [
        {'window_baseline': w, 'threshold_sigma': t, 'min_separation': 75}
        for w in [200, 500, 1000]
        for t in [1.5, 2.0, 2.5, 3.0]
    ],
}


def optimize_method(method_name: str, entropy: np.ndarray,
                    reference_drops: Dict[str, List[Tuple[int, float]]],
                    tolerance: int = 100) -> List[ConsensusResult]:
    """Sweep parameters for one method and evaluate consensus."""
    param_grid = PARAM_GRIDS.get(method_name, [])
    if not param_grid:
        return []

    detect_fn = METHODS[method_name]
    results = []

    for params in param_grid:
        try:
            drops = detect_fn(entropy, **params)
        except Exception:
            continue

        if not drops:
            continue

        other_drops = {m: d for m, d in reference_drops.items() if m != method_name}
        consensus_frac, avg_confirming = count_consensus(drops, other_drops, tolerance)

        n_drops = len(drops)
        length_mbp = len(entropy) / 1e6
        density = n_drops / length_mbp if length_mbp > 0 else 0

        # Sweet spot: 10-500 drops per Mbp
        density_score = 1.0
        if density > 500:
            density_score = 500 / density
        elif density < 10:
            density_score = density / 10

        quality = 0.6 * consensus_frac + 0.25 * min(avg_confirming / 4, 1.0) + 0.15 * density_score

        results.append(ConsensusResult(
            method=method_name,
            params=params,
            n_drops=n_drops,
            consensus_fraction=consensus_frac,
            avg_n_confirming=avg_confirming,
            quality_score=quality,
        ))

    results.sort(key=lambda r: -r.quality_score)
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Optimize detection parameters via cross-method consensus"
    )
    ap.add_argument("--entropy", required=True, help="Path to entropy.npz")
    ap.add_argument("--output", required=True, help="Output JSON with optimal parameters")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="bp tolerance for consensus matching (default: 100)")
    ap.add_argument("--methods", nargs='+',
                    default=list(METHODS.keys()),
                    help="Methods to optimize (default: all 6)")
    args = ap.parse_args()

    # Load entropy
    print(f"Loading entropy from {args.entropy}")
    data = np.load(args.entropy, allow_pickle=True)
    entropy = data['entropy']
    print(f"  {len(entropy):,} positions, {np.isnan(entropy).sum():,} NaN")

    # First pass: run all methods with defaults to get reference drops
    print("\nRunning all methods with default parameters...")
    reference_drops = {}
    for method_name in METHODS:
        defaults = METHOD_DEFAULTS[method_name].copy()
        drops = METHODS[method_name](entropy, **defaults)
        reference_drops[method_name] = drops
        print(f"  {method_name:20s}: {len(drops):6d} drops")

    # Optimize each method
    all_results = {}
    for method_name in args.methods:
        if method_name not in METHODS:
            print(f"WARNING: Unknown method '{method_name}', skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Optimizing {method_name}")
        print(f"{'='*60}")

        results = optimize_method(method_name, entropy, reference_drops, args.tolerance)

        if not results:
            print(f"  No valid parameter combinations found")
            continue

        best = results[0]
        print(f"  Tested {len(results)} parameter combinations")
        print(f"  Best params: {best.params}")
        print(f"    Quality score:     {best.quality_score:.4f}")
        print(f"    Consensus:         {best.consensus_fraction:.3f}")
        print(f"    Avg confirming:    {best.avg_n_confirming:.2f}")
        print(f"    N drops:           {best.n_drops}")

        all_results[method_name] = {
            'best_params': best.params,
            'quality_score': best.quality_score,
            'consensus_fraction': best.consensus_fraction,
            'avg_n_confirming': best.avg_n_confirming,
            'n_drops': best.n_drops,
            'top_5': [
                {
                    'params': r.params,
                    'quality_score': r.quality_score,
                    'consensus_fraction': r.consensus_fraction,
                    'n_drops': r.n_drops,
                }
                for r in results[:5]
            ],
        }

    # Save results
    output = {
        'entropy_file': os.path.abspath(args.entropy),
        'tolerance_bp': args.tolerance,
        'n_positions': int(len(entropy)),
        'reference_drop_counts': {m: len(d) for m, d in reference_drops.items()},
        'optimal_parameters': all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary: optimal parameters per method")
    print(f"{'='*60}")
    print(f"{'Method':<20s} {'Quality':>8s} {'Consensus':>10s} {'Drops':>8s}")
    print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*8}")
    for method, info in sorted(all_results.items()):
        print(f"{method:<20s} {info['quality_score']:>8.4f} "
              f"{info['consensus_fraction']:>10.3f} {info['n_drops']:>8d}")


if __name__ == "__main__":
    main()
