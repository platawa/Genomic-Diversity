#!/usr/bin/env python3
"""
compare_detection_methods.py

Run all 6 detection methods on an entropy.npz file and compare results.
No GPU required — runs locally on existing scoring output.

Usage:
    python tools/compare_detection_methods.py \
        --entropy results/chr22/scoring/.../data/entropy.npz \
        [--gtf /path/to/genomic.gtf --chrom NC_000022.11] \
        [--output_dir results/chr22/detection_comparison/]
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


def load_entropy(npz_path: str) -> np.ndarray:
    """Load entropy array from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return data['entropy']


def parse_gtf_features(gtf_path: str, chrom: str,
                       start: int = 0, end: int = None
                       ) -> Dict[str, List[Tuple[int, int]]]:
    """Parse GTF to extract feature boundaries within a region.

    Returns dict mapping feature type -> list of (start, end) tuples.
    """
    features = defaultdict(list)
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[0] != chrom:
                continue
            feat_type = fields[2]
            feat_start = int(fields[3]) - 1  # GTF is 1-based
            feat_end = int(fields[4])
            if end is not None and feat_start > end:
                continue
            if feat_end < start:
                continue
            features[feat_type].append((feat_start - start, feat_end - start))
    return dict(features)


def compute_overlap_stats(drops: List[Tuple[int, float]],
                          features: Dict[str, List[Tuple[int, int]]],
                          tolerance: int = 100
                          ) -> Dict[str, Dict]:
    """Compute overlap between detected drops and GTF features.

    A drop at position P overlaps feature (start, end) if
    P is within [start - tolerance, end + tolerance].
    """
    stats = {}
    drop_positions = [pos for pos, _ in drops]

    for feat_type, intervals in features.items():
        hits = 0
        for dp in drop_positions:
            for fs, fe in intervals:
                if fs - tolerance <= dp <= fe + tolerance:
                    hits += 1
                    break
        stats[feat_type] = {
            "n_features": len(intervals),
            "n_drops_overlapping": hits,
            "fraction_drops_overlapping": hits / len(drop_positions) if drop_positions else 0,
        }
    return stats


def _count_matches_sorted(pos1_sorted: np.ndarray, pos2_sorted: np.ndarray,
                          tolerance: int) -> int:
    """Count how many positions in pos1 have a match within tolerance in pos2.

    Uses sorted arrays + binary search for O(n log m) instead of O(n*m).
    """
    count = 0
    for p in pos1_sorted:
        idx = np.searchsorted(pos2_sorted, p)
        # Check neighbors
        if idx < len(pos2_sorted) and abs(pos2_sorted[idx] - p) <= tolerance:
            count += 1
        elif idx > 0 and abs(pos2_sorted[idx - 1] - p) <= tolerance:
            count += 1
    return count


def pairwise_agreement(all_drops: Dict[str, List[Tuple[int, float]]],
                       tolerance: int = 100) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Jaccard-like agreement between methods.

    Two drops from different methods "agree" if they are within
    +-tolerance of each other. Uses binary search for efficiency.
    """
    methods = list(all_drops.keys())

    # Pre-sort all position arrays
    sorted_pos = {}
    for m in methods:
        positions = np.array([p for p, _ in all_drops[m]], dtype=np.int64)
        positions.sort()
        sorted_pos[m] = positions

    agreement = {}
    for m1 in methods:
        agreement[m1] = {}
        for m2 in methods:
            if len(sorted_pos[m1]) == 0 or len(sorted_pos[m2]) == 0:
                agreement[m1][m2] = 0.0
                continue

            matches_1to2 = _count_matches_sorted(sorted_pos[m1], sorted_pos[m2], tolerance)
            matches_2to1 = _count_matches_sorted(sorted_pos[m2], sorted_pos[m1], tolerance)

            intersection = (matches_1to2 + matches_2to1) / 2
            union = len(sorted_pos[m1]) + len(sorted_pos[m2]) - intersection
            agreement[m1][m2] = intersection / union if union > 0 else 0.0

    return agreement


def write_comparison_tsv(all_drops: Dict[str, List[Tuple[int, float]]],
                         output_path: str):
    """Write per-method drop summary to TSV."""
    with open(output_path, 'w') as f:
        f.write("method\tn_drops\tmin_score\tmax_score\tmedian_score\n")
        for method, drops in sorted(all_drops.items()):
            if not drops:
                f.write(f"{method}\t0\t\t\t\n")
                continue
            scores = [s for _, s in drops]
            f.write(f"{method}\t{len(drops)}\t{min(scores):.4f}\t"
                    f"{max(scores):.4f}\t{np.median(scores):.4f}\n")


def plot_comparison(entropy: np.ndarray,
                    all_drops: Dict[str, List[Tuple[int, float]]],
                    output_dir: str,
                    region: Optional[Tuple[int, int]] = None,
                    features: Optional[Dict[str, List[Tuple[int, int]]]] = None):
    """Generate overlay plot with all methods on same entropy trace."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping plots")
        return

    colors = {
        'zscore': '#e41a1c',
        'mad': '#377eb8',
        'derivative': '#4daf4a',
        'window_mean_shift': '#984ea3',
        'cusum': '#ff7f00',
        'local_baseline': '#a65628',
    }

    if region:
        lo, hi = region
    else:
        lo, hi = 0, len(entropy)

    ent_slice = entropy[lo:hi]
    x = np.arange(lo, hi)

    # --- Full overlay plot ---
    fig, axes = plt.subplots(len(all_drops) + 1, 1,
                             figsize=(16, 3 * (len(all_drops) + 1)),
                             sharex=True)

    # Top panel: entropy trace
    axes[0].plot(x, ent_slice, color='gray', linewidth=0.3, alpha=0.7)
    axes[0].set_ylabel('Entropy')
    axes[0].set_title(f'Entropy trace (positions {lo:,}–{hi:,})')

    # Add feature annotations if available
    if features:
        for feat_type, intervals in features.items():
            if feat_type in ('CDS', 'exon'):
                for fs, fe in intervals:
                    if lo <= fs < hi or lo <= fe < hi:
                        axes[0].axvspan(max(fs, lo), min(fe, hi),
                                       alpha=0.15, color='green' if feat_type == 'CDS' else 'blue')

    # Per-method panels
    for idx, (method, drops) in enumerate(sorted(all_drops.items()), 1):
        axes[idx].plot(x, ent_slice, color='gray', linewidth=0.3, alpha=0.5)
        color = colors.get(method, 'black')

        drop_positions = [p for p, _ in drops if lo <= p < hi]
        drop_scores = [s for p, s in drops if lo <= p < hi]

        if drop_positions:
            axes[idx].scatter(drop_positions,
                             ent_slice[np.array(drop_positions) - lo],
                             color=color, s=15, zorder=5, alpha=0.8)
        axes[idx].set_ylabel(method)
        axes[idx].text(0.02, 0.85, f'n={len(drop_positions)}',
                      transform=axes[idx].transAxes, fontsize=9)

    axes[-1].set_xlabel('Position')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_overlay.png'), dpi=150)
    plt.close()

    # --- Agreement heatmap ---
    agreement = pairwise_agreement(all_drops)
    methods = sorted(all_drops.keys())
    matrix = np.array([[agreement[m1][m2] for m2 in methods] for m1 in methods])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, label='Jaccard agreement')
    ax.set_title('Pairwise method agreement (tolerance=100bp)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_agreement_heatmap.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare all 6 detection methods on entropy data"
    )
    ap.add_argument("--entropy", required=True,
                    help="Path to entropy.npz file")
    ap.add_argument("--gtf", default=None,
                    help="Optional GTF file for annotation overlap analysis")
    ap.add_argument("--chrom", default=None,
                    help="Chromosome name in GTF (e.g., NC_000022.11)")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: alongside entropy.npz)")
    ap.add_argument("--region_start", type=int, default=None,
                    help="Plot only this region (start position)")
    ap.add_argument("--region_end", type=int, default=None,
                    help="Plot only this region (end position)")
    ap.add_argument("--tolerance", type=int, default=100,
                    help="Tolerance in bp for overlap/agreement (default: 100)")
    args = ap.parse_args()

    # Load entropy
    print(f"Loading entropy from {args.entropy}")
    entropy = load_entropy(args.entropy)
    print(f"  Array length: {len(entropy):,} positions")
    print(f"  NaN positions: {np.isnan(entropy).sum():,}")

    # Also try to load genomic offset
    data = np.load(args.entropy, allow_pickle=True)
    genomic_start = int(data['start']) if 'start' in data else 0

    # Setup output
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.entropy), 'detection_comparison')
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all methods
    print("\nRunning detection methods...")
    all_drops = {}
    for name in METHODS:
        drops = run_method(name, entropy)
        all_drops[name] = drops
        print(f"  {name:20s}: {len(drops):5d} drops")

    # Write comparison TSV
    tsv_path = os.path.join(args.output_dir, 'method_comparison.tsv')
    write_comparison_tsv(all_drops, tsv_path)
    print(f"\nComparison table: {tsv_path}")

    # Pairwise agreement
    agreement = pairwise_agreement(all_drops, tolerance=args.tolerance)
    agreement_path = os.path.join(args.output_dir, 'pairwise_agreement.json')
    with open(agreement_path, 'w') as f:
        json.dump(agreement, f, indent=2)
    print(f"Agreement matrix: {agreement_path}")

    # GTF overlap analysis
    features = None
    if args.gtf and args.chrom:
        print(f"\nParsing GTF annotations for {args.chrom}...")
        features = parse_gtf_features(args.gtf, args.chrom, genomic_start,
                                      genomic_start + len(entropy))
        for feat_type, intervals in sorted(features.items()):
            print(f"  {feat_type}: {len(intervals)} features")

        print("\nOverlap with annotations:")
        overlap_results = {}
        for method, drops in sorted(all_drops.items()):
            overlap = compute_overlap_stats(drops, features, tolerance=args.tolerance)
            overlap_results[method] = overlap
            cds_overlap = overlap.get('CDS', {})
            exon_overlap = overlap.get('exon', {})
            print(f"  {method:20s}: CDS overlap={cds_overlap.get('fraction_drops_overlapping', 0):.1%}, "
                  f"exon overlap={exon_overlap.get('fraction_drops_overlapping', 0):.1%}")

        overlap_path = os.path.join(args.output_dir, 'gtf_overlap.json')
        with open(overlap_path, 'w') as f:
            json.dump(overlap_results, f, indent=2)

    # Generate plots
    region = None
    if args.region_start is not None and args.region_end is not None:
        region = (args.region_start, args.region_end)

    plot_comparison(entropy, all_drops, args.output_dir, region=region,
                    features=features)

    # Summary
    summary = {
        "entropy_file": os.path.abspath(args.entropy),
        "n_positions": int(len(entropy)),
        "n_nan": int(np.isnan(entropy).sum()),
        "methods_run": list(all_drops.keys()),
        "drop_counts": {m: len(d) for m, d in all_drops.items()},
        "tolerance_bp": args.tolerance,
    }
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
