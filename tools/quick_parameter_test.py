#!/usr/bin/env python3
"""
quick_parameter_test.py

Quick parameter sweep on a single gene to visualize effect of thresholds.

Usage:
    python quick_parameter_test.py \
        --organism ecoli \
        --gene_id b0455 \
        --data_dir /path/to/scoring_outputs

Outputs:
    - parameter_sweep_results.png: Heatmap showing drop counts vs parameters
    - parameter_sweep_data.json: Detailed results for each combination
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_entropy_from_tsv(tsv_file: Path) -> np.ndarray:
    """Load entropy scores from genome_scoring TSV output."""
    data = []
    with open(tsv_file) as f:
        next(f)  # Skip header
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) > 3:
                try:
                    entropy = float(fields[3])
                    data.append(entropy)
                except ValueError:
                    data.append(np.nan)
    return np.array(data)


def test_zscore_parameters(entropy: np.ndarray) -> Dict[Tuple[float, int], int]:
    """
    Test range of z-score parameters.

    Returns:
        Dictionary mapping (threshold, min_sep) -> drop_count
    """
    # Import detection function
    sys.path.insert(0, str(Path(__file__).parent))
    from genome_scoring_jan26_drops import detect_drops_zscore

    results = {}

    # Test ranges
    thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    separations = [25, 50, 75, 100, 150]

    total_tests = len(thresholds) * len(separations)
    test_num = 0

    print("\nTesting Z-score parameter combinations...")
    print(f"{'Threshold':<12} {'MinSep':<10} {'Drops':<10}")
    print("-" * 32)

    for thresh in thresholds:
        for min_sep in separations:
            test_num += 1

            drops = detect_drops_zscore(
                entropy,
                smooth_w=51,
                zscore_threshold=thresh,
                min_separation=min_sep
            )

            n_drops = len(drops)
            results[(thresh, min_sep)] = n_drops

            print(f"{thresh:<12.1f} {min_sep:<10} {n_drops:<10} "
                  f"[{test_num}/{total_tests}]")

    return results


def test_mad_parameters(entropy: np.ndarray) -> Dict[Tuple[float, int], int]:
    """
    Test range of MAD parameters.

    Returns:
        Dictionary mapping (threshold, min_sep) -> drop_count
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from genome_scoring_jan26_drops import detect_drops_mad

    results = {}

    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    separations = [25, 50, 75, 100, 150]

    total_tests = len(thresholds) * len(separations)
    test_num = 0

    print("\nTesting MAD parameter combinations...")
    print(f"{'Threshold':<12} {'MinSep':<10} {'Drops':<10}")
    print("-" * 32)

    for thresh in thresholds:
        for min_sep in separations:
            test_num += 1

            drops = detect_drops_mad(
                entropy,
                smooth_w=51,
                mad_threshold=thresh,
                min_separation=min_sep
            )

            n_drops = len(drops)
            results[(thresh, min_sep)] = n_drops

            print(f"{thresh:<12.1f} {min_sep:<10} {n_drops:<10} "
                  f"[{test_num}/{total_tests}]")

    return results


def test_local_parameters(entropy: np.ndarray) -> Dict[Tuple[int, float, int], int]:
    """
    Test range of local baseline parameters.

    Returns:
        Dictionary mapping (window, threshold, min_sep) -> drop_count
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from genome_scoring_jan26_drops import detect_drops_local_baseline

    results = {}

    windows = [200, 500, 1000]
    thresholds = [1.5, 2.0, 2.5, 3.0]
    separations = [50, 75, 100]

    total_tests = len(windows) * len(thresholds) * len(separations)
    test_num = 0

    print("\nTesting Local Baseline parameter combinations...")
    print(f"{'Window':<10} {'Threshold':<12} {'MinSep':<10} {'Drops':<10}")
    print("-" * 42)

    for window in windows:
        for thresh in thresholds:
            for min_sep in separations:
                test_num += 1

                drops = detect_drops_local_baseline(
                    entropy,
                    window_baseline=window,
                    threshold_sigma=thresh,
                    min_separation=min_sep
                )

                n_drops = len(drops)
                results[(window, thresh, min_sep)] = n_drops

                print(f"{window:<10} {thresh:<12.1f} {min_sep:<10} {n_drops:<10} "
                      f"[{test_num}/{total_tests}]")

    return results


def plot_zscore_heatmap(results: Dict[Tuple[float, int], int],
                       output_file: Path):
    """Create heatmap of z-score parameter effects."""
    # Extract unique thresholds and separations
    thresholds = sorted(set(k[0] for k in results.keys()))
    separations = sorted(set(k[1] for k in results.keys()))

    # Create matrix
    matrix = np.zeros((len(thresholds), len(separations)))
    for i, thresh in enumerate(thresholds):
        for j, sep in enumerate(separations):
            matrix[i, j] = results.get((thresh, sep), 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=separations,
        yticklabels=[f'{t:.1f}' for t in thresholds],
        cbar_kws={'label': 'Number of Drops'},
        ax=ax
    )

    ax.set_xlabel('Min Separation (bp)', fontsize=12)
    ax.set_ylabel('Z-Score Threshold', fontsize=12)
    ax.set_title('Z-Score Method: Drop Count vs Parameters', fontsize=14, fontweight='bold')

    # Add interpretation guide
    fig.text(0.5, 0.02,
             'Lower left = more sensitive (more drops) | Upper right = more specific (fewer drops)',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nZ-score heatmap saved to: {output_file}")
    plt.close()


def plot_mad_heatmap(results: Dict[Tuple[float, int], int],
                    output_file: Path):
    """Create heatmap of MAD parameter effects."""
    thresholds = sorted(set(k[0] for k in results.keys()))
    separations = sorted(set(k[1] for k in results.keys()))

    matrix = np.zeros((len(thresholds), len(separations)))
    for i, thresh in enumerate(thresholds):
        for j, sep in enumerate(separations):
            matrix[i, j] = results.get((thresh, sep), 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap='YlGnBu',
        xticklabels=separations,
        yticklabels=[f'{t:.1f}' for t in thresholds],
        cbar_kws={'label': 'Number of Drops'},
        ax=ax
    )

    ax.set_xlabel('Min Separation (bp)', fontsize=12)
    ax.set_ylabel('MAD Threshold', fontsize=12)
    ax.set_title('MAD Method: Drop Count vs Parameters', fontsize=14, fontweight='bold')

    fig.text(0.5, 0.02,
             'Lower left = more sensitive | Upper right = more specific',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"MAD heatmap saved to: {output_file}")
    plt.close()


def plot_local_comparison(results: Dict[Tuple[int, float, int], int],
                         output_file: Path):
    """Create comparison plot for local baseline parameters."""
    windows = sorted(set(k[0] for k in results.keys()))
    thresholds = sorted(set(k[1] for k in results.keys()))

    fig, axes = plt.subplots(1, len(windows), figsize=(15, 5))

    if len(windows) == 1:
        axes = [axes]

    for idx, window in enumerate(windows):
        ax = axes[idx]

        # Filter results for this window
        window_results = {(t, s): n for (w, t, s), n in results.items() if w == window}

        thresholds_w = sorted(set(k[0] for k in window_results.keys()))
        separations_w = sorted(set(k[1] for k in window_results.keys()))

        matrix = np.zeros((len(thresholds_w), len(separations_w)))
        for i, thresh in enumerate(thresholds_w):
            for j, sep in enumerate(separations_w):
                matrix[i, j] = window_results.get((thresh, sep), 0)

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.0f',
            cmap='Greens',
            xticklabels=separations_w,
            yticklabels=[f'{t:.1f}' for t in thresholds_w],
            cbar_kws={'label': 'Drops'},
            ax=ax
        )

        ax.set_xlabel('Min Separation (bp)')
        ax.set_ylabel('Threshold (σ)')
        ax.set_title(f'Window = {window} bp', fontweight='bold')

    fig.suptitle('Local Baseline Method: Effect of Window Size',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Local baseline comparison saved to: {output_file}")
    plt.close()


def recommend_parameters(zscore_results: Dict, mad_results: Dict,
                        gene_length: int) -> Dict:
    """
    Recommend optimal parameters based on results.

    Strategy:
        - Target: 5-15 drops per gene (reasonable specificity)
        - Prefer higher thresholds when possible (higher confidence)
        - Balance between methods
    """
    target_drops = 10  # Ideal number
    tolerance = 5      # ±5 drops is acceptable

    recommendations = {}

    # Z-score recommendation
    best_zscore = None
    best_zscore_diff = float('inf')

    for (thresh, sep), n_drops in zscore_results.items():
        diff = abs(n_drops - target_drops)
        if diff < best_zscore_diff:
            # Prefer higher threshold if tie
            if diff == best_zscore_diff and best_zscore and thresh <= best_zscore[0]:
                continue
            best_zscore = (thresh, sep, n_drops)
            best_zscore_diff = diff

    recommendations['zscore'] = {
        'threshold': best_zscore[0],
        'min_separation': best_zscore[1],
        'expected_drops': best_zscore[2],
        'confidence': 'high' if best_zscore[0] >= 3.0 else 'medium' if best_zscore[0] >= 2.5 else 'low'
    }

    # MAD recommendation
    best_mad = None
    best_mad_diff = float('inf')

    for (thresh, sep), n_drops in mad_results.items():
        diff = abs(n_drops - target_drops)
        if diff < best_mad_diff:
            if diff == best_mad_diff and best_mad and thresh <= best_mad[0]:
                continue
            best_mad = (thresh, sep, n_drops)
            best_mad_diff = diff

    recommendations['mad'] = {
        'threshold': best_mad[0],
        'min_separation': best_mad[1],
        'expected_drops': best_mad[2],
        'confidence': 'high' if best_mad[0] >= 3.5 else 'medium' if best_mad[0] >= 3.0 else 'low'
    }

    return recommendations


def main():
    ap = argparse.ArgumentParser(
        description="Quick parameter test on single gene"
    )

    ap.add_argument("--organism", required=True,
                   help="Organism name")

    ap.add_argument("--gene_id", required=True,
                   help="Gene ID to test")

    ap.add_argument("--data_dir", type=Path, required=True,
                   help="Directory with genome_scoring outputs")

    ap.add_argument("--methods", nargs='+',
                   choices=["zscore", "mad", "local"],
                   default=["zscore", "mad"],
                   help="Methods to test")

    ap.add_argument("--output_prefix", type=Path,
                   default="parameter_sweep",
                   help="Output file prefix")

    args = ap.parse_args()

    # Load entropy data
    gene_data_dir = args.data_dir / args.gene_id
    tsv_file = gene_data_dir / "data" / f"{args.gene_id}.tsv"

    if not tsv_file.exists():
        print(f"Error: Data file not found: {tsv_file}")
        print(f"Please run genome_scoring_jan26_drops.py on {args.gene_id} first")
        return 1

    print(f"Loading entropy data from: {tsv_file}")
    entropy = load_entropy_from_tsv(tsv_file)
    print(f"Loaded {len(entropy)} positions")

    all_results = {}

    # Test Z-score
    if "zscore" in args.methods:
        zscore_results = test_zscore_parameters(entropy)
        all_results['zscore'] = {
            str(k): v for k, v in zscore_results.items()
        }
        plot_zscore_heatmap(
            zscore_results,
            Path(f"{args.output_prefix}_zscore.png")
        )

    # Test MAD
    if "mad" in args.methods:
        mad_results = test_mad_parameters(entropy)
        all_results['mad'] = {
            str(k): v for k, v in mad_results.items()
        }
        plot_mad_heatmap(
            mad_results,
            Path(f"{args.output_prefix}_mad.png")
        )

    # Test Local
    if "local" in args.methods:
        local_results = test_local_parameters(entropy)
        all_results['local'] = {
            str(k): v for k, v in local_results.items()
        }
        plot_local_comparison(
            local_results,
            Path(f"{args.output_prefix}_local.png")
        )

    # Save raw data
    with open(f"{args.output_prefix}_data.json", 'w') as f:
        json.dump({
            'organism': args.organism,
            'gene_id': args.gene_id,
            'gene_length': len(entropy),
            'results': all_results
        }, f, indent=2)

    print(f"\nRaw data saved to: {args.output_prefix}_data.json")

    # Generate recommendations
    if "zscore" in args.methods and "mad" in args.methods:
        print("\n" + "="*60)
        print("PARAMETER RECOMMENDATIONS")
        print("="*60)

        recommendations = recommend_parameters(
            zscore_results,
            mad_results,
            len(entropy)
        )

        print("\nZ-Score Method:")
        print(f"  --zscore_threshold {recommendations['zscore']['threshold']}")
        print(f"  --min_separation {recommendations['zscore']['min_separation']}")
        print(f"  Expected drops: {recommendations['zscore']['expected_drops']}")
        print(f"  Confidence level: {recommendations['zscore']['confidence']}")

        print("\nMAD Method:")
        print(f"  --mad_threshold {recommendations['mad']['threshold']}")
        print(f"  --min_separation {recommendations['mad']['min_separation']}")
        print(f"  Expected drops: {recommendations['mad']['expected_drops']}")
        print(f"  Confidence level: {recommendations['mad']['confidence']}")

        print("\nRecommended command:")
        print(f"\npython genome_scoring_jan26_drops.py \\")
        print(f"    --organism {args.organism} \\")
        print(f"    --gene_id {args.gene_id} \\")
        print(f"    --detection_methods zscore mad \\")
        print(f"    --zscore_threshold {recommendations['zscore']['threshold']} \\")
        print(f"    --mad_threshold {recommendations['mad']['threshold']} \\")
        print(f"    --min_separation {max(recommendations['zscore']['min_separation'], recommendations['mad']['min_separation'])}")

    return 0


if __name__ == "__main__":
    exit(main())
