#!/usr/bin/env python3
"""
genome_scoring_jan26_benchmark.py

Performance benchmarking for genome scoring pipeline.

Tests:
    1. Model loading time (ESM2/Evo2)
    2. Inference time vs chunk size (1, 5, 50, 100, 1000, 2000, 5000)
    3. Memory usage (GPU and RAM)
    4. Throughput (positions/second)
    5. Sequence length scaling

Outputs:
    - benchmark_results.json: Detailed metrics
    - benchmark_summary.txt: Human-readable summary
    - benchmark_plots.png: Visualization

Usage:
    # Quick test with random sequences (3 sequences, 4 chunk sizes)
    python benchmark_performance.py --mode quick

    # Full benchmark with random sequences
    python benchmark_performance.py --mode full

    # Use REAL human genome sequences
    python benchmark_performance.py --mode quick --organism human

    # Use REAL E. coli genome sequences
    python benchmark_performance.py --mode quick --organism ecoli

    # Custom FASTA file
    python benchmark_performance.py --mode quick --fasta /path/to/genome.fna

    # Custom test with specific lengths
    python benchmark_performance.py \
        --sequence_lengths 1000 5000 10000 \
        --chunk_sizes 50 100 1000 \
        --n_repeats 3 \
        --organism human
"""

import torch
import numpy as np
import time
import json
import argparse
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    sequence_length: int
    chunk_size: int

    # Timing (seconds)
    model_load_time: float
    total_inference_time: float
    avg_chunk_time: float

    # Throughput
    positions_per_second: float
    chunks_processed: int

    # Memory (MB)
    gpu_memory_used: float
    gpu_memory_peak: float
    ram_memory_used: float

    # GPU utilization (%)
    gpu_utilization: float

    # Metadata
    device: str
    timestamp: str


@dataclass
class ModelLoadBenchmark:
    """Results from model loading benchmark."""
    model_name: str
    load_time: float
    model_size_mb: float
    device: str
    success: bool
    error_message: str = ""


def get_gpu_memory_mb() -> Tuple[float, float]:
    """
    Get current and peak GPU memory usage in MB.

    Returns:
        (current_mb, peak_mb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    current = torch.cuda.memory_allocated() / 1024 / 1024
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    return current, peak


def get_ram_memory_mb() -> float:
    """Get current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_utilization() -> float:
    """
    Get GPU utilization percentage.

    Returns:
        GPU utilization (0-100), or 0 if unavailable
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except:
        return 0.0


def generate_random_sequence(length: int) -> str:
    """
    Generate random DNA sequence for testing.

    Args:
        length: Sequence length in bp

    Returns:
        Random DNA sequence
    """
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases, size=length))


def load_genome_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Load genome sequences from FASTA file.

    Args:
        fasta_path: Path to genome FASTA file

    Returns:
        Dictionary mapping chromosome names to sequences
    """
    print(f"Loading genome from: {fasta_path}")
    sequences = {}
    current_chrom = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_chrom is not None:
                    sequences[current_chrom] = ''.join(current_seq)
                current_chrom = line[1:].split()[0]  # Get chromosome name
                current_seq = []
            else:
                current_seq.append(line.upper())

        if current_chrom is not None:
            sequences[current_chrom] = ''.join(current_seq)

    print(f"Loaded {len(sequences)} chromosomes")
    return sequences


def extract_genomic_sequence(genome: Dict[str, str], length: int,
                              chromosome: str = None) -> str:
    """
    Extract a subsequence of specified length from the genome.

    Args:
        genome: Dictionary of chromosome sequences
        length: Desired sequence length
        chromosome: Specific chromosome to use (default: random choice)

    Returns:
        Genomic subsequence
    """
    # Pick a chromosome that's long enough
    if chromosome is None:
        valid_chroms = [c for c, seq in genome.items() if len(seq) >= length]
        if not valid_chroms:
            raise ValueError(f"No chromosome long enough for {length} bp")
        chromosome = np.random.choice(valid_chroms)

    seq = genome[chromosome]
    if len(seq) < length:
        raise ValueError(f"Chromosome {chromosome} too short ({len(seq)} < {length})")

    # Random start position
    max_start = len(seq) - length
    start = np.random.randint(0, max_start + 1)

    extracted = seq[start:start + length]
    print(f"  Extracted {length} bp from {chromosome}:{start}-{start+length}")

    return extracted


def benchmark_model_loading(model_name: str = "evo2_7b") -> ModelLoadBenchmark:
    """
    Benchmark model loading time.

    Args:
        model_name: Model to load (evo2_7b)

    Returns:
        ModelLoadBenchmark with timing results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking Model Loading: {model_name}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        # Import and load Evo2 model
        from evo2 import Evo2

        print(f"Loading Evo2 model: {model_name}")

        model = Evo2(model_name)
        model.model.eval()

        load_time = time.time() - start_time

        # Get model size (estimate)
        param_count = sum(p.numel() for p in model.model.parameters())
        model_size_mb = param_count * 2 / 1024 / 1024  # bfloat16

        gpu_mem_current, gpu_mem_peak = get_gpu_memory_mb()

        print(f"✓ Model loaded successfully in {load_time:.2f}s")
        print(f"  - Parameters: {param_count:,}")
        print(f"  - Model size: {model_size_mb:.1f} MB")
        print(f"  - GPU memory: {gpu_mem_peak:.1f} MB")
        print(f"  - Device: {device}")

        return ModelLoadBenchmark(
            model_name=model_name,
            load_time=load_time,
            model_size_mb=model_size_mb,
            device=device,
            success=True
        )

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return ModelLoadBenchmark(
            model_name=model_name,
            load_time=0.0,
            model_size_mb=0.0,
            device=device,
            success=False,
            error_message=str(e)
        )


def benchmark_inference_chunking(
    sequence: str,
    chunk_size: int,
    model,
    tokenizer,
    device: str,
    n_repeats: int = 3
) -> BenchmarkResult:
    """
    Benchmark inference with specific chunk size.

    Args:
        sequence: DNA sequence to process
        chunk_size: Number of positions per chunk
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device (cuda/cpu)
        n_repeats: Number of times to repeat for averaging

    Returns:
        BenchmarkResult with detailed metrics
    """
    seq_len = len(sequence)

    print(f"\n  Testing chunk_size={chunk_size:>5} (seq_len={seq_len:>6})...",
          end=' ', flush=True)

    # Reset GPU stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    ram_before = get_ram_memory_mb()

    # Warm-up run (not timed)
    with torch.no_grad():
        test_chunk = sequence[:min(chunk_size, seq_len)]
        input_ids = torch.tensor([tokenizer.tokenize(test_chunk)], dtype=torch.long)
        if device == "cuda":
            input_ids = input_ids.cuda()
        _ = model.model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Actual benchmark runs
    chunk_times = []
    total_chunks = 0

    for repeat in range(n_repeats):
        start_time = time.time()

        with torch.no_grad():
            pos = 0
            while pos < seq_len:
                chunk_end = min(pos + chunk_size, seq_len)
                chunk_seq = sequence[pos:chunk_end]

                # Tokenize using Evo2 tokenizer
                input_ids = torch.tensor([tokenizer.tokenize(chunk_seq)], dtype=torch.long)
                if device == "cuda":
                    input_ids = input_ids.cuda()

                # Inference
                outputs = model.model(input_ids)

                total_chunks += 1
                pos = chunk_end

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        chunk_times.append(time.time() - start_time)

    # Compute statistics
    avg_time = np.mean(chunk_times)
    avg_chunk_time = avg_time / (total_chunks / n_repeats)
    positions_per_sec = seq_len / avg_time

    # Memory metrics
    gpu_mem_current, gpu_mem_peak = get_gpu_memory_mb()
    ram_after = get_ram_memory_mb()
    ram_used = ram_after - ram_before

    # GPU utilization
    gpu_util = get_gpu_utilization()

    print(f"{avg_time:.3f}s ({positions_per_sec:.0f} pos/s, "
          f"{gpu_mem_peak:.0f}MB GPU)")

    return BenchmarkResult(
        sequence_length=seq_len,
        chunk_size=chunk_size,
        model_load_time=0.0,  # Not measured here
        total_inference_time=avg_time,
        avg_chunk_time=avg_chunk_time,
        positions_per_second=positions_per_sec,
        chunks_processed=total_chunks // n_repeats,
        gpu_memory_used=gpu_mem_current,
        gpu_memory_peak=gpu_mem_peak,
        ram_memory_used=ram_used,
        gpu_utilization=gpu_util,
        device=device,
        timestamp=datetime.now().isoformat()
    )


def run_chunking_benchmarks(
    sequence_lengths: List[int],
    chunk_sizes: List[int],
    model_name: str = "evo2_7b",
    n_repeats: int = 3,
    genome: Dict[str, str] = None
) -> Tuple[List[BenchmarkResult], ModelLoadBenchmark]:
    """
    Run comprehensive chunking benchmarks.

    Args:
        sequence_lengths: List of sequence lengths to test
        chunk_sizes: List of chunk sizes to test
        model_name: Model to use
        n_repeats: Repetitions per test
        genome: Optional loaded genome dict (if None, uses random sequences)

    Returns:
        (list of BenchmarkResults, ModelLoadBenchmark)
    """
    # Load model once
    model_load_result = benchmark_model_loading(model_name)

    if not model_load_result.success:
        print("\n✗ Model loading failed, cannot proceed with benchmarks")
        return [], model_load_result

    # Re-load model for actual benchmarking
    from evo2 import Evo2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Evo2 model for benchmarking: {model_name}")
    model = Evo2(model_name)
    model.model.eval()
    tokenizer = model.tokenizer

    # Run benchmarks
    results = []

    print(f"\n{'='*60}")
    print(f"Running Chunking Benchmarks")
    print(f"{'='*60}")
    print(f"Sequences: {len(sequence_lengths)}")
    print(f"Chunk sizes: {len(chunk_sizes)}")
    print(f"Repeats: {n_repeats}")
    print(f"Sequence source: {'Real genome' if genome else 'Random'}")
    print(f"Total tests: {len(sequence_lengths) * len(chunk_sizes)}")

    for seq_len in sequence_lengths:
        print(f"\n--- Sequence Length: {seq_len} bp ---")

        # Generate test sequence (real or random)
        if genome:
            sequence = extract_genomic_sequence(genome, seq_len)
        else:
            sequence = generate_random_sequence(seq_len)

        for chunk_size in chunk_sizes:
            result = benchmark_inference_chunking(
                sequence=sequence,
                chunk_size=chunk_size,
                model=model,
                tokenizer=tokenizer,
                device=device,
                n_repeats=n_repeats
            )

            results.append(result)

    return results, model_load_result


def save_results(
    results: List[BenchmarkResult],
    model_load: ModelLoadBenchmark,
    output_file: Path
):
    """
    Save benchmark results to JSON.

    Args:
        results: List of benchmark results
        model_load: Model loading benchmark
        output_file: Output JSON file path
    """
    output_data = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
            "n_tests": len(results)
        },
        "model_loading": asdict(model_load),
        "inference_results": [asdict(r) for r in results]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def generate_summary_report(
    results: List[BenchmarkResult],
    model_load: ModelLoadBenchmark,
    output_file: Path
):
    """
    Generate human-readable summary report.

    Args:
        results: Benchmark results
        model_load: Model loading results
        output_file: Output text file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GENOME SCORING BENCHMARK RESULTS\n")
        f.write("="*80 + "\n\n")

        # Model loading
        f.write("MODEL LOADING\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {model_load.model_name}\n")
        f.write(f"Load time: {model_load.load_time:.2f}s\n")
        f.write(f"Model size: {model_load.model_size_mb:.1f} MB\n")
        f.write(f"Device: {model_load.device}\n")
        f.write(f"Success: {'✓' if model_load.success else '✗'}\n\n")

        # Inference results summary
        f.write("INFERENCE PERFORMANCE\n")
        f.write("-"*80 + "\n\n")

        # Group by sequence length
        seq_lengths = sorted(set(r.sequence_length for r in results))

        for seq_len in seq_lengths:
            f.write(f"Sequence Length: {seq_len:,} bp\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Chunk Size':<12} {'Time (s)':<12} {'Throughput':<15} {'GPU Mem (MB)':<15}\n")
            f.write("-"*40 + "\n")

            seq_results = [r for r in results if r.sequence_length == seq_len]
            seq_results.sort(key=lambda x: x.chunk_size)

            for r in seq_results:
                f.write(f"{r.chunk_size:<12} "
                       f"{r.total_inference_time:<12.3f} "
                       f"{r.positions_per_second:<15.0f} "
                       f"{r.gpu_memory_peak:<15.0f}\n")

            f.write("\n")

        # Optimal chunk sizes
        f.write("OPTIMAL CHUNK SIZES (by throughput)\n")
        f.write("-"*80 + "\n")

        for seq_len in seq_lengths:
            seq_results = [r for r in results if r.sequence_length == seq_len]
            best = max(seq_results, key=lambda x: x.positions_per_second)

            f.write(f"Seq len {seq_len:>6} bp: "
                   f"chunk_size={best.chunk_size:>5} "
                   f"({best.positions_per_second:.0f} pos/s, "
                   f"{best.gpu_memory_peak:.0f} MB GPU)\n")

        f.write("\n")

        # Memory usage
        f.write("MEMORY USAGE\n")
        f.write("-"*80 + "\n")
        max_gpu = max(r.gpu_memory_peak for r in results)
        max_ram = max(r.ram_memory_used for r in results)
        f.write(f"Peak GPU memory: {max_gpu:.1f} MB\n")
        f.write(f"Peak RAM usage: {max_ram:.1f} MB\n")
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")

        # Find sweet spot (best throughput without excessive memory)
        all_results_sorted = sorted(results,
                                    key=lambda x: x.positions_per_second,
                                    reverse=True)

        # Filter to reasonable memory usage (<10GB GPU)
        reasonable = [r for r in all_results_sorted if r.gpu_memory_peak < 10000]

        if reasonable:
            best_overall = reasonable[0]
            f.write(f"Recommended chunk size: {best_overall.chunk_size}\n")
            f.write(f"  - Throughput: {best_overall.positions_per_second:.0f} pos/s\n")
            f.write(f"  - GPU memory: {best_overall.gpu_memory_peak:.0f} MB\n")
            f.write(f"  - Tested on: {best_overall.sequence_length} bp sequence\n")

        f.write("\n")
        f.write("="*80 + "\n")

    print(f"✓ Summary saved to: {output_file}")


def plot_benchmark_results(
    results: List[BenchmarkResult],
    output_file: Path
):
    """
    Create visualization of benchmark results.

    Args:
        results: Benchmark results
        output_file: Output PNG file
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get unique sequence lengths
    seq_lengths = sorted(set(r.sequence_length for r in results))

    # Plot 1: Throughput vs Chunk Size
    ax = axes[0, 0]
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r.sequence_length == seq_len]
        seq_results.sort(key=lambda x: x.chunk_size)

        chunk_sizes = [r.chunk_size for r in seq_results]
        throughput = [r.positions_per_second for r in seq_results]

        ax.plot(chunk_sizes, throughput, marker='o', label=f'{seq_len} bp')

    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Throughput (positions/second)')
    ax.set_title('Throughput vs Chunk Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: GPU Memory vs Chunk Size
    ax = axes[0, 1]
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r.sequence_length == seq_len]
        seq_results.sort(key=lambda x: x.chunk_size)

        chunk_sizes = [r.chunk_size for r in seq_results]
        gpu_mem = [r.gpu_memory_peak for r in seq_results]

        ax.plot(chunk_sizes, gpu_mem, marker='s', label=f'{seq_len} bp')

    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('GPU Memory Usage vs Chunk Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Inference Time vs Chunk Size
    ax = axes[1, 0]
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r.sequence_length == seq_len]
        seq_results.sort(key=lambda x: x.chunk_size)

        chunk_sizes = [r.chunk_size for r in seq_results]
        times = [r.total_inference_time for r in seq_results]

        ax.plot(chunk_sizes, times, marker='^', label=f'{seq_len} bp')

    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Inference Time vs Chunk Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Heatmap of Throughput
    ax = axes[1, 1]

    # Create matrix
    chunk_sizes = sorted(set(r.chunk_size for r in results))
    matrix = np.zeros((len(seq_lengths), len(chunk_sizes)))

    for i, seq_len in enumerate(seq_lengths):
        for j, chunk_size in enumerate(chunk_sizes):
            matching = [r for r in results
                       if r.sequence_length == seq_len and r.chunk_size == chunk_size]
            if matching:
                matrix[i, j] = matching[0].positions_per_second

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        xticklabels=chunk_sizes,
        yticklabels=seq_lengths,
        cbar_kws={'label': 'Positions/Second'},
        ax=ax
    )

    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Sequence Length (bp)')
    ax.set_title('Throughput Heatmap')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to: {output_file}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark genome scoring performance"
    )

    ap.add_argument("--mode", choices=["quick", "full", "custom"],
                   default="quick",
                   help="Benchmark mode (quick=fast, full=comprehensive)")

    ap.add_argument("--sequence_lengths", nargs='+', type=int,
                   help="Sequence lengths to test (bp)")

    ap.add_argument("--chunk_sizes", nargs='+', type=int,
                   help="Chunk sizes to test")

    ap.add_argument("--n_repeats", type=int, default=3,
                   help="Number of repetitions per test")

    ap.add_argument("--model_name", default="evo2_7b",
                   help="Model to benchmark")

    ap.add_argument("--output_dir", type=Path, default="results/_benchmarks",
                   help="Output directory (default: results/_benchmarks)")

    ap.add_argument("--fasta", type=Path, default=None,
                   help="Path to genome FASTA file (if not provided, uses random sequences)")

    ap.add_argument("--organism", choices=["human", "ecoli", "bacillus"], default=None,
                   help="Use pre-configured genome path for organism")

    args = ap.parse_args()

    # Set parameters based on mode
    if args.mode == "quick":
        sequence_lengths = args.sequence_lengths or [1000, 5000, 10000]
        chunk_sizes = args.chunk_sizes or [50, 100, 500, 1000]
        n_repeats = 2
        print("Running QUICK benchmark (3 sequences, 4 chunk sizes)")

    elif args.mode == "full":
        sequence_lengths = args.sequence_lengths or [500, 1000, 2000, 5000, 10000, 20000, 50000]
        chunk_sizes = args.chunk_sizes or [1, 5, 50, 100, 500, 1000, 2000, 5000]
        n_repeats = 3
        print("Running FULL benchmark (7 sequences, 8 chunk sizes)")

    else:  # custom
        if not args.sequence_lengths or not args.chunk_sizes:
            print("Error: --sequence_lengths and --chunk_sizes required for custom mode")
            return 1

        sequence_lengths = args.sequence_lengths
        chunk_sizes = args.chunk_sizes
        n_repeats = args.n_repeats
        print(f"Running CUSTOM benchmark ({len(sequence_lengths)} sequences, "
              f"{len(chunk_sizes)} chunk sizes)")

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Load genome if specified
    genome = None
    fasta_path = args.fasta

    # Pre-configured paths for organisms
    ORGANISM_FASTA = {
        "human": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna",
        "ecoli": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
        "bacillus": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna",
    }

    if args.organism:
        fasta_path = Path(ORGANISM_FASTA[args.organism])
        print(f"Using {args.organism} genome: {fasta_path}")

    if fasta_path:
        if not fasta_path.exists():
            print(f"Error: FASTA file not found: {fasta_path}")
            return 1
        genome = load_genome_fasta(str(fasta_path))

    # Run benchmarks
    print(f"\nDevice: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    results, model_load = run_chunking_benchmarks(
        sequence_lengths=sequence_lengths,
        chunk_sizes=chunk_sizes,
        model_name=args.model_name,
        n_repeats=n_repeats,
        genome=genome
    )

    if not results:
        print("\n✗ No results to save")
        return 1

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = args.output_dir / f"benchmark_results_{timestamp}.json"
    summary_file = args.output_dir / f"benchmark_summary_{timestamp}.txt"
    plot_file = args.output_dir / f"benchmark_plots_{timestamp}.png"

    save_results(results, model_load, json_file)
    generate_summary_report(results, model_load, summary_file)
    plot_benchmark_results(results, plot_file)

    # Print summary to console
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Total tests run: {len(results)}")
    print(f"Model load time: {model_load.load_time:.2f}s")

    # Best throughput
    best = max(results, key=lambda x: x.positions_per_second)
    print(f"\nBest throughput:")
    print(f"  Chunk size: {best.chunk_size}")
    print(f"  Sequence length: {best.sequence_length} bp")
    print(f"  Throughput: {best.positions_per_second:.0f} positions/second")
    print(f"  GPU memory: {best.gpu_memory_peak:.0f} MB")

    print(f"\nResults saved to: {args.output_dir}/")

    return 0


if __name__ == "__main__":
    exit(main())
