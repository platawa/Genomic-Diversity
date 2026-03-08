#!/usr/bin/env python3
"""
analyze_benchmarks.py

Analyze and compare multiple benchmark runs.

Features:
    - Compare performance across different runs
    - Identify optimal chunk sizes
    - Track performance trends over time
    - Generate comparison plots

Usage:
    # Analyze latest benchmark
    python analyze_benchmarks.py

    # Analyze specific benchmark
    python analyze_benchmarks.py --input benchmark_results/benchmark_results_20260126.json

    # Compare multiple benchmarks
    python analyze_benchmarks.py \
        --compare \
        benchmark_results/benchmark_results_20260126_100000.json \
        benchmark_results/benchmark_results_20260126_110000.json

    # Find optimal chunk size for specific sequence length
    python analyze_benchmarks.py --find_optimal --sequence_length 10000
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import pandas as pd


def load_benchmark_results(json_file: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(json_file) as f:
        return json.load(f)


def find_optimal_chunk_size(
    results: List[Dict],
    sequence_length: int,
    metric: str = "positions_per_second"
) -> Tuple[int, float]:
    """
    Find optimal chunk size for given sequence length.

    Args:
        results: List of benchmark results
        sequence_length: Target sequence length
        metric: Optimization metric (throughput or memory)

    Returns:
        (optimal_chunk_size, metric_value)
    """
    # Filter to sequence length
    filtered = [r for r in results if r['sequence_length'] == sequence_length]

    if not filtered:
        # Find closest sequence length
        all_lens = set(r['sequence_length'] for r in results)
        closest = min(all_lens, key=lambda x: abs(x - sequence_length))
        print(f"Warning: No results for {sequence_length} bp, using closest: {closest} bp")
        filtered = [r for r in results if r['sequence_length'] == closest]

    if metric == "positions_per_second":
        # Maximize throughput
        best = max(filtered, key=lambda x: x[metric])
    elif metric == "gpu_memory_peak":
        # Minimize memory while maintaining reasonable throughput
        # Filter to top 50% by throughput, then find min memory
        sorted_by_throughput = sorted(filtered,
                                     key=lambda x: x['positions_per_second'],
                                     reverse=True)
        top_half = sorted_by_throughput[:len(sorted_by_throughput)//2 + 1]
        best = min(top_half, key=lambda x: x[metric])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return best['chunk_size'], best[metric]


def generate_recommendation_table(results: List[Dict]) -> str:
    """
    Generate table of recommended chunk sizes for different sequence lengths.

    Args:
        results: Benchmark results

    Returns:
        Formatted table string
    """
    sequence_lengths = sorted(set(r['sequence_length'] for r in results))

    lines = []
    lines.append("\n" + "="*80)
    lines.append("OPTIMAL CHUNK SIZE RECOMMENDATIONS")
    lines.append("="*80)
    lines.append(f"{'Seq Length':<15} {'Chunk Size':<15} {'Throughput':<20} {'GPU Memory':<15}")
    lines.append("-"*80)

    for seq_len in sequence_lengths:
        chunk_size, throughput = find_optimal_chunk_size(
            results, seq_len, metric="positions_per_second"
        )

        # Get memory for this configuration
        matching = [r for r in results
                   if r['sequence_length'] == seq_len and r['chunk_size'] == chunk_size]
        gpu_mem = matching[0]['gpu_memory_peak'] if matching else 0

        lines.append(f"{seq_len:<15,} {chunk_size:<15} "
                    f"{throughput:<20,.0f} {gpu_mem:<15,.0f}")

    lines.append("="*80)
    lines.append("\nKey:")
    lines.append("  - Chunk Size: Number of positions processed per inference call")
    lines.append("  - Throughput: Positions processed per second")
    lines.append("  - GPU Memory: Peak GPU memory usage (MB)")
    lines.append("")

    return "\n".join(lines)


def compare_benchmarks(json_files: List[Path]) -> str:
    """
    Compare multiple benchmark runs.

    Args:
        json_files: List of benchmark JSON files

    Returns:
        Comparison report string
    """
    all_data = []

    for json_file in json_files:
        data = load_benchmark_results(json_file)
        all_data.append({
            'file': json_file.name,
            'timestamp': data['benchmark_info']['timestamp'],
            'device': data['benchmark_info']['device'],
            'model_load_time': data['model_loading']['load_time'],
            'results': data['inference_results']
        })

    lines = []
    lines.append("\n" + "="*80)
    lines.append("BENCHMARK COMPARISON")
    lines.append("="*80)

    # Model loading comparison
    lines.append("\nModel Loading Time:")
    lines.append("-"*40)
    for i, d in enumerate(all_data, 1):
        lines.append(f"  Run {i} ({d['timestamp']}): {d['model_load_time']:.2f}s")

    # Average throughput comparison
    lines.append("\nAverage Throughput (positions/second):")
    lines.append("-"*40)

    for i, d in enumerate(all_data, 1):
        avg_throughput = np.mean([r['positions_per_second'] for r in d['results']])
        lines.append(f"  Run {i}: {avg_throughput:,.0f}")

    # Peak memory comparison
    lines.append("\nPeak GPU Memory (MB):")
    lines.append("-"*40)

    for i, d in enumerate(all_data, 1):
        max_mem = max(r['gpu_memory_peak'] for r in d['results'])
        lines.append(f"  Run {i}: {max_mem:,.0f}")

    lines.append("="*80)

    return "\n".join(lines)


def plot_comparison(json_files: List[Path], output_file: Path):
    """
    Create comparison plots for multiple benchmark runs.

    Args:
        json_files: List of benchmark JSON files
        output_file: Output plot file
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    all_data = [load_benchmark_results(f) for f in json_files]

    # Plot 1: Throughput comparison
    ax = axes[0, 0]

    for i, (data, json_file) in enumerate(zip(all_data, json_files)):
        results = data['inference_results']

        # Group by sequence length, average across chunk sizes
        seq_lens = sorted(set(r['sequence_length'] for r in results))
        avg_throughput = []

        for seq_len in seq_lens:
            seq_results = [r for r in results if r['sequence_length'] == seq_len]
            avg = np.mean([r['positions_per_second'] for r in seq_results])
            avg_throughput.append(avg)

        ax.plot(seq_lens, avg_throughput, marker='o',
               label=f'Run {i+1}: {json_file.stem[-8:]}')

    ax.set_xlabel('Sequence Length (bp)')
    ax.set_ylabel('Avg Throughput (positions/s)')
    ax.set_title('Throughput vs Sequence Length')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Memory usage comparison
    ax = axes[0, 1]

    for i, (data, json_file) in enumerate(zip(all_data, json_files)):
        results = data['inference_results']

        seq_lens = sorted(set(r['sequence_length'] for r in results))
        max_memory = []

        for seq_len in seq_lens:
            seq_results = [r for r in results if r['sequence_length'] == seq_len]
            max_mem = max(r['gpu_memory_peak'] for r in seq_results)
            max_memory.append(max_mem)

        ax.plot(seq_lens, max_memory, marker='s',
               label=f'Run {i+1}: {json_file.stem[-8:]}')

    ax.set_xlabel('Sequence Length (bp)')
    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_title('Memory Usage vs Sequence Length')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Model loading time
    ax = axes[1, 0]

    load_times = [d['model_loading']['load_time'] for d in all_data]
    timestamps = [d['benchmark_info']['timestamp'][:10] for d in all_data]

    ax.bar(range(len(load_times)), load_times, color='steelblue')
    ax.set_xlabel('Benchmark Run')
    ax.set_ylabel('Model Load Time (s)')
    ax.set_title('Model Loading Time Comparison')
    ax.set_xticks(range(len(load_times)))
    ax.set_xticklabels([f'Run {i+1}\n{ts}' for i, ts in enumerate(timestamps)],
                       rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    # Plot 4: Efficiency (throughput per GB of GPU memory)
    ax = axes[1, 1]

    for i, (data, json_file) in enumerate(zip(all_data, json_files)):
        results = data['inference_results']

        # Calculate efficiency: throughput per GB GPU memory
        efficiency = []
        chunk_sizes = sorted(set(r['chunk_size'] for r in results))

        for chunk_size in chunk_sizes:
            chunk_results = [r for r in results if r['chunk_size'] == chunk_size]
            avg_throughput = np.mean([r['positions_per_second'] for r in chunk_results])
            avg_memory_gb = np.mean([r['gpu_memory_peak'] / 1024 for r in chunk_results])

            if avg_memory_gb > 0:
                efficiency.append(avg_throughput / avg_memory_gb)
            else:
                efficiency.append(0)

        ax.plot(chunk_sizes, efficiency, marker='^',
               label=f'Run {i+1}')

    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Efficiency (pos/s per GB GPU)')
    ax.set_title('Memory Efficiency vs Chunk Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to: {output_file}")
    plt.close()


def create_pandas_summary(results: List[Dict]) -> pd.DataFrame:
    """
    Create pandas DataFrame summary of results.

    Args:
        results: Benchmark results

    Returns:
        DataFrame with key metrics
    """
    df = pd.DataFrame(results)

    # Select key columns
    columns = [
        'sequence_length',
        'chunk_size',
        'total_inference_time',
        'positions_per_second',
        'gpu_memory_peak',
        'chunks_processed'
    ]

    df_summary = df[columns].copy()

    # Add efficiency metric
    df_summary['efficiency'] = df_summary['positions_per_second'] / (df_summary['gpu_memory_peak'] / 1024)

    return df_summary


def main():
    ap = argparse.ArgumentParser(
        description="Analyze genome scoring benchmark results"
    )

    ap.add_argument("--input", type=Path,
                   help="Input benchmark JSON file (default: latest)")

    ap.add_argument("--compare", nargs='+', type=Path,
                   help="Compare multiple benchmark files")

    ap.add_argument("--find_optimal", action="store_true",
                   help="Find optimal chunk size for sequence length")

    ap.add_argument("--sequence_length", type=int, default=10000,
                   help="Sequence length for optimal chunk size search")

    ap.add_argument("--output_dir", type=Path, default="benchmark_analysis",
                   help="Output directory for analysis results")

    args = ap.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Load benchmark data
    if args.compare:
        # Comparison mode
        json_files = args.compare

        if len(json_files) < 2:
            print("Error: Need at least 2 files to compare")
            return 1

        print(f"Comparing {len(json_files)} benchmark runs...")

        # Generate comparison report
        report = compare_benchmarks(json_files)
        print(report)

        # Save report
        report_file = args.output_dir / "benchmark_comparison.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Comparison report saved to: {report_file}")

        # Create comparison plots
        plot_file = args.output_dir / "benchmark_comparison.png"
        plot_comparison(json_files, plot_file)

    else:
        # Single benchmark analysis
        if args.input:
            json_file = args.input
        else:
            # Find latest benchmark file
            benchmark_dir = Path("benchmark_results")
            if not benchmark_dir.exists():
                print("Error: No benchmark results found")
                return 1

            json_files = sorted(benchmark_dir.glob("benchmark_results_*.json"))
            if not json_files:
                print("Error: No benchmark JSON files found")
                return 1

            json_file = json_files[-1]
            print(f"Using latest benchmark: {json_file}")

        # Load data
        data = load_benchmark_results(json_file)
        results = data['inference_results']

        # Generate recommendation table
        recommendations = generate_recommendation_table(results)
        print(recommendations)

        # Save recommendations
        rec_file = args.output_dir / "recommendations.txt"
        with open(rec_file, 'w') as f:
            f.write(recommendations)
        print(f"✓ Recommendations saved to: {rec_file}")

        # Find optimal for specific sequence length
        if args.find_optimal:
            chunk_size, throughput = find_optimal_chunk_size(
                results,
                args.sequence_length,
                metric="positions_per_second"
            )

            print(f"\n{'='*60}")
            print(f"OPTIMAL CHUNK SIZE FOR {args.sequence_length:,} bp")
            print(f"{'='*60}")
            print(f"Recommended chunk size: {chunk_size}")
            print(f"Expected throughput: {throughput:,.0f} positions/second")

            # Get memory for this configuration
            matching = [r for r in results
                       if abs(r['sequence_length'] - args.sequence_length) < 100
                       and r['chunk_size'] == chunk_size]
            if matching:
                gpu_mem = matching[0]['gpu_memory_peak']
                print(f"Expected GPU memory: {gpu_mem:,.0f} MB")

        # Create pandas summary
        df = create_pandas_summary(results)

        # Save CSV
        csv_file = args.output_dir / "benchmark_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ CSV summary saved to: {csv_file}")

        # Print top 10 configurations by throughput
        print("\nTop 10 configurations by throughput:")
        print("-" * 80)
        top10 = df.nlargest(10, 'positions_per_second')
        print(top10.to_string(index=False))

    return 0


if __name__ == "__main__":
    exit(main())
