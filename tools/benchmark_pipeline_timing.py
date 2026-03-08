#!/usr/bin/env python3
"""
benchmark_pipeline_timing.py

End-to-end timing benchmark for the genome scoring pipeline.

Measures wall-clock time at each stage of the REAL scoring pipeline
(not just raw forward passes) using actual genome data:

  T1: Model loading (Evo2 init + tokenizer + ACGT IDs)
  T2: Single-chunk inference at various chunk sizes
      - T2a: Forward-strand entropy (1 model call)
      - T2b: RC-averaged entropy  (2 model calls: fwd + rc)
      - T2c: Next-token logprobs + gather + probs_subset (1 model call)
  T3: Full-locus scoring via score_locus_aligned_overlap()
  T4: File writing (TSV, drops, rises, window summary) with synthetic data
  T5: FULL PIPELINE on whole genome (--whole_genome):
      - Inference (scoring entire chromosome)
      - Unit conversion (nats -> bits)
      - Drop detection (all 6 methods: zscore, mad, local, derivative, win_shift, cusum)
      - Rise detection (all 6 methods)
      - Plot suite generation
      - Method comparison report
      - All file I/O (TSV, drops, rises, window summary, metadata JSON)

Also benchmarks multi-GPU data parallelism when available
(via score_locus_aligned_overlap_multigpu).

Usage:
    # Quick test (small chunk sizes only)
    python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --mode quick

    # Full benchmark across all chunk sizes
    python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --mode full

    # WHOLE GENOME: full pipeline with detection, plots, and file I/O
    python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --whole_genome

    # Whole genome with custom chunk size for the pipeline
    python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --whole_genome \
        --pipeline_chunk_size 50000

    # Everything: whole genome + chunk sweep + multi-GPU
    python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --whole_genome \
        --mode full --benchmark_multigpu
"""

import sys
import os
import time
import json
import gc
import math
import argparse
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager

import numpy as np

# Add parent directory to path so we can import from genome_scoring_jan26_drops
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from score_chromosome import find_max_chunk_size as _find_max_chunk_size

from genome_scoring_jan26_drops import (
    entropy_like_reference_acgt,
    next_token_logprobs_and_targets_aligned,
    next_token_probs_subset,
    score_locus_aligned_overlap,
    write_window_summary,
    MAX_CHUNK_LEN_DEFAULT,
    CHUNK_OVERLAP_DEFAULT,
    # Detection functions
    detect_drops_zscore,
    detect_drops_mad,
    detect_drops_local_baseline,
    detect_drops_derivative,
    detect_drops_window_mean_shift,
    detect_drops_cusum,
    detect_rises_zscore,
    detect_rises_mad,
    detect_rises_local_baseline,
    detect_rises_derivative,
    detect_rises_window_mean_shift,
    detect_rises_cusum,
    _drops_scored_to_positions,
    # Plotting & reporting
    OutputManager,
    plot_suite,
    generate_method_comparison_report,
    # Constants
    DROP_SMOOTH_W,
    DROP_DERIV_Q,
    DROP_SHIFT_W,
    DROP_SHIFT_TOPK,
    DROP_CUSUM_H,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CHUNK_SIZES_QUICK = [5_000, 15_000, 50_000, 100_000]

CHUNK_SIZES_FULL = [
    5_000,
    10_000,
    15_000,       # current default
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    750_000,
    1_000_000,
    1_048_576,    # model max_seqlen
    1_500_000,    # past max
    2_000_000,    # well past max
]

FULL_LOCUS_CHUNK_SIZES = [15_000, 50_000, 100_000, 250_000, 500_000]

DEFAULT_LOCUS_LENGTH = 100_000  # bp for full-locus tests
DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP_DEFAULT  # 1024


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelLoadResult:
    """Timing for model loading."""
    load_time_seconds: float
    tokenizer_setup_seconds: float
    total_seconds: float
    model_name: str
    device: str
    n_gpus_available: int
    gpu_memory_after_mb: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class SingleChunkResult:
    """Timing for scoring one chunk through the full pipeline."""
    chunk_size_bp: int
    status: str  # "success", "oom", "error"
    entropy_fwd_seconds: float = 0.0
    entropy_rc_seconds: float = 0.0
    next_token_seconds: float = 0.0
    total_inference_seconds: float = 0.0
    n_forward_passes: int = 4
    throughput_bp_per_sec: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    error: str = ""


@dataclass
class FullLocusResult:
    """Timing for scoring a full locus with chunking."""
    chunk_size_bp: int
    locus_length_bp: int
    status: str
    total_seconds: float = 0.0
    n_chunks: int = 0
    avg_seconds_per_chunk: float = 0.0
    throughput_bp_per_sec: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    mode: str = "sequential"  # "sequential" or "multigpu"
    n_gpus_used: int = 1
    error: str = ""


@dataclass
class FileWriteResult:
    """Timing for file I/O operations."""
    tsv_write_seconds: float = 0.0
    drops_write_seconds: float = 0.0
    rises_write_seconds: float = 0.0
    window_summary_seconds: float = 0.0
    total_seconds: float = 0.0
    tsv_rows: int = 0
    tsv_file_size_bytes: int = 0


@dataclass
class FullPipelineResult:
    """Timing for the complete analysis pipeline on a locus (replicating run_one_locus)."""
    locus_length_bp: int = 0
    chunk_size_bp: int = 15_000
    status: str = "pending"
    # Inference (scoring)
    inference_seconds: float = 0.0
    n_chunks: int = 0
    # Unit conversion (nats -> bits)
    unit_conversion_seconds: float = 0.0
    # Drop detection
    drop_detection_seconds: float = 0.0
    drop_detection_by_method: Dict[str, float] = field(default_factory=dict)
    n_drops_by_method: Dict[str, int] = field(default_factory=dict)
    # Rise detection
    rise_detection_seconds: float = 0.0
    rise_detection_by_method: Dict[str, float] = field(default_factory=dict)
    n_rises_by_method: Dict[str, int] = field(default_factory=dict)
    # Plot generation
    plot_suite_seconds: float = 0.0
    method_comparison_seconds: float = 0.0
    # File I/O
    tsv_write_seconds: float = 0.0
    drops_file_seconds: float = 0.0
    rises_file_seconds: float = 0.0
    window_summary_seconds: float = 0.0
    metadata_json_seconds: float = 0.0
    # Totals
    total_analysis_seconds: float = 0.0  # drops + rises + plots + file I/O
    total_seconds: float = 0.0           # everything including inference
    gpu_memory_peak_mb: float = 0.0
    error: str = ""


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str = ""
    genome_source: str = ""
    genome_length_bp: int = 0
    model_name: str = "evo2_7b"
    device: str = ""
    cuda_available: bool = False
    n_gpus: int = 0
    pytorch_version: str = ""
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    model_loading: Optional[ModelLoadResult] = None
    single_chunk_results: List[SingleChunkResult] = field(default_factory=list)
    full_locus_results: List[FullLocusResult] = field(default_factory=list)
    file_writing: Optional[FileWriteResult] = None
    full_pipeline: Optional[FullPipelineResult] = None


# ============================================================================
# UTILITIES
# ============================================================================

def gpu_sync():
    """Synchronize GPU if available (ensures accurate timing)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> Tuple[float, float]:
    """Return (current_mb, peak_mb) GPU memory usage."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    current = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return current, peak


def reset_gpu_stats():
    """Reset peak GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def load_genome_from_genbank(gbff_path: str, chrom: str = None) -> Tuple[str, str]:
    """
    Load a genome sequence from a GenBank (.gbff/.gb) or FASTA (.fna/.fa/.fasta) file.

    Auto-detects format from file extension. Supports selecting a specific
    chromosome/contig by ID; otherwise picks the largest record.

    Args:
        gbff_path: Path to GenBank or FASTA file
        chrom: Optional chromosome/contig ID to select (e.g., 'NC_000001.11' for
               human chr1). If None, picks the largest record.

    Returns:
        (sequence_string, record_id)
    """
    from Bio import SeqIO
    import re

    print(f"[BENCH] Loading genome from: {gbff_path}")

    # Auto-detect format from extension
    ext = os.path.splitext(gbff_path)[1].lower()
    if ext in (".fna", ".fa", ".fasta"):
        fmt = "fasta"
    elif ext in (".gbff", ".gb", ".gbk", ".genbank"):
        fmt = "genbank"
    else:
        # Try both, GenBank first
        fmt = "genbank"
        try:
            test = list(SeqIO.parse(gbff_path, "genbank"))
            if not test:
                fmt = "fasta"
        except Exception:
            fmt = "fasta"

    print(f"[BENCH]   Format: {fmt}")

    # For large FASTA files (human genome ~3GB), stream rather than load all
    if chrom is not None:
        # User wants a specific chromosome -- stream and find it
        print(f"[BENCH]   Searching for chromosome: {chrom}")
        record = None
        for rec in SeqIO.parse(gbff_path, fmt):
            if rec.id == chrom or rec.name == chrom:
                record = rec
                break
            # Also try matching without version suffix (e.g., 'NC_000001' matches 'NC_000001.11')
            if chrom in rec.id or rec.id.startswith(chrom):
                record = rec
                break
        if record is None:
            # Print available IDs to help user
            print(f"[BENCH]   ERROR: Chromosome '{chrom}' not found.")
            print(f"[BENCH]   Available records (first 30):")
            for i, rec in enumerate(SeqIO.parse(gbff_path, fmt)):
                print(f"[BENCH]     {rec.id}  ({len(rec.seq):,} bp)")
                if i >= 29:
                    print(f"[BENCH]     ... (more records exist)")
                    break
            raise ValueError(f"Chromosome '{chrom}' not found in {gbff_path}")
    else:
        # No specific chromosome -- pick the largest
        print(f"[BENCH]   Loading all records to find largest...")
        records = list(SeqIO.parse(gbff_path, fmt))
        if not records:
            raise ValueError(f"No records found in {gbff_path}")
        record = max(records, key=lambda r: len(r.seq))
        print(f"[BENCH]   Found {len(records)} records, picking largest")

    seq = str(record.seq).upper()

    # Replace non-ACGT with random ACGT (Evo2 expects clean sequence)
    non_acgt = re.compile(r'[^ACGT]')
    bases = ['A', 'C', 'G', 'T']
    if non_acgt.search(seq):
        n_replaced = len(non_acgt.findall(seq))
        seq = non_acgt.sub(lambda m: np.random.choice(bases), seq)
        print(f"[BENCH]   Replaced {n_replaced:,} non-ACGT characters with random bases")

    print(f"[BENCH]   Loaded: {record.id} ({len(seq):,} bp)")
    return seq, record.id


def extract_clean_subsequence(genome_seq: str, length: int, start: int = 0) -> str:
    """Extract a contiguous subsequence from the genome."""
    if start + length > len(genome_seq):
        # If requested length exceeds genome, wrap or truncate
        available = len(genome_seq) - start
        if available < length:
            print(f"[BENCH]   Warning: genome only has {available:,} bp from position {start}, "
                  f"using full available region")
            return genome_seq[start:]
    return genome_seq[start:start + length]


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_model_loading(model_name: str = "evo2_7b") -> Tuple[ModelLoadResult, Any, Any, Any]:
    """
    Benchmark model loading time.

    Returns:
        (result, evo2_model, ACGT_IDS, device)
    """
    print(f"\n{'='*70}")
    print(f"T1: MODEL LOADING BENCHMARK")
    print(f"{'='*70}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    try:
        reset_gpu_stats()
        gpu_sync()

        # Time model loading
        t0 = time.perf_counter()
        from evo2 import Evo2
        evo2_model = Evo2(model_name)
        if hasattr(evo2_model, "eval"):
            evo2_model.eval()
        elif hasattr(evo2_model, "model"):
            evo2_model.model.eval()
        gpu_sync()
        t_model = time.perf_counter()

        # Time tokenizer / ACGT ID setup
        device = str(getattr(evo2_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        idx_A = evo2_model.tokenizer.tokenize("A")[0]
        idx_C = evo2_model.tokenizer.tokenize("C")[0]
        idx_G = evo2_model.tokenizer.tokenize("G")[0]
        idx_T = evo2_model.tokenizer.tokenize("T")[0]
        ACGT_IDS = torch.tensor([idx_A, idx_C, idx_G, idx_T], dtype=torch.long, device=device)
        t_tok = time.perf_counter()

        load_time = t_model - t0
        tok_time = t_tok - t_model
        total_time = t_tok - t0

        _, gpu_peak = get_gpu_memory_mb()

        print(f"  Model load:     {load_time:.2f}s")
        print(f"  Tokenizer setup: {tok_time:.4f}s")
        print(f"  Total:          {total_time:.2f}s")
        print(f"  Device:         {device}")
        print(f"  GPUs available: {n_gpus}")
        if gpu_peak > 0:
            print(f"  GPU memory:     {gpu_peak:.0f} MB")

        result = ModelLoadResult(
            load_time_seconds=load_time,
            tokenizer_setup_seconds=tok_time,
            total_seconds=total_time,
            model_name=model_name,
            device=device,
            n_gpus_available=n_gpus,
            gpu_memory_after_mb=gpu_peak,
        )
        return result, evo2_model, ACGT_IDS, device

    except Exception as e:
        print(f"  FAILED: {e}")
        result = ModelLoadResult(
            load_time_seconds=0.0,
            tokenizer_setup_seconds=0.0,
            total_seconds=0.0,
            model_name=model_name,
            device=device_str,
            n_gpus_available=n_gpus,
            success=False,
            error=str(e),
        )
        return result, None, None, device_str


def benchmark_single_chunk(
    chunk_seq: str,
    evo2_model,
    ACGT_IDS: torch.Tensor,
    device: str,
    chunk_size_label: int,
) -> SingleChunkResult:
    """
    Benchmark the full scoring pipeline for a single chunk.

    Replicates the exact 4-forward-pass pattern from score_locus_aligned_overlap():
      1. entropy_like_reference_acgt(reverse_complement=False)  -- 1 fwd pass
      2. entropy_like_reference_acgt(reverse_complement=True)   -- 2 fwd passes (fwd+rc)
      3. next_token_logprobs_and_targets_aligned + gather + probs_subset -- 1 fwd pass
    """
    try:
        reset_gpu_stats()

        # Pass 1: Forward-only entropy
        gpu_sync()
        t0 = time.perf_counter()
        Hf, Pf = entropy_like_reference_acgt(
            chunk_seq, evo2_model, ACGT_IDS, device,
            prepend_bos=True, reverse_complement=False
        )
        gpu_sync()
        t_fwd = time.perf_counter()

        # Pass 2+3: RC-averaged entropy (forward + reverse complement)
        gpu_sync()
        t1 = time.perf_counter()
        Hr, Pr = entropy_like_reference_acgt(
            chunk_seq, evo2_model, ACGT_IDS, device,
            prepend_bos=True, reverse_complement=True
        )
        gpu_sync()
        t_rc = time.perf_counter()

        # Pass 4: Next-token logprobs + post-processing
        gpu_sync()
        t2 = time.perf_counter()
        logprobs_next, target_next = next_token_logprobs_and_targets_aligned(
            chunk_seq, evo2_model, device
        )
        ll = (
            logprobs_next.float()
            .gather(-1, target_next.unsqueeze(-1))
            .squeeze(-1)
            .detach().cpu()
        )
        p4_run = (
            next_token_probs_subset(logprobs_next.float(), ACGT_IDS)
            .detach().cpu()
        )
        gpu_sync()
        t_ntp = time.perf_counter()

        entropy_fwd_s = t_fwd - t0
        entropy_rc_s = t_rc - t1
        next_token_s = t_ntp - t2
        total_s = t_ntp - t0

        _, gpu_peak = get_gpu_memory_mb()
        chunk_len = len(chunk_seq)
        throughput = chunk_len / total_s if total_s > 0 else 0

        return SingleChunkResult(
            chunk_size_bp=chunk_size_label,
            status="success",
            entropy_fwd_seconds=entropy_fwd_s,
            entropy_rc_seconds=entropy_rc_s,
            next_token_seconds=next_token_s,
            total_inference_seconds=total_s,
            n_forward_passes=4,
            throughput_bp_per_sec=throughput,
            gpu_memory_peak_mb=gpu_peak,
        )

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return SingleChunkResult(
            chunk_size_bp=chunk_size_label,
            status="oom",
            error=str(e),
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return SingleChunkResult(
                chunk_size_bp=chunk_size_label,
                status="oom",
                error=str(e),
            )
        return SingleChunkResult(
            chunk_size_bp=chunk_size_label,
            status="error",
            error=str(e),
        )
    except Exception as e:
        return SingleChunkResult(
            chunk_size_bp=chunk_size_label,
            status="error",
            error=f"{type(e).__name__}: {e}",
        )


def benchmark_single_chunk_tests(
    genome_seq: str,
    chunk_sizes: List[int],
    evo2_model,
    ACGT_IDS: torch.Tensor,
    device: str,
) -> List[SingleChunkResult]:
    """Run single-chunk benchmarks across all chunk sizes."""

    print(f"\n{'='*70}")
    print(f"T2: SINGLE-CHUNK INFERENCE BENCHMARKS")
    print(f"{'='*70}")
    print(f"  Chunk sizes: {len(chunk_sizes)}")
    print(f"  Genome available: {len(genome_seq):,} bp")
    print()

    results = []
    for chunk_size in chunk_sizes:
        # Extract a sequence of this length
        seq_len = min(chunk_size, len(genome_seq))
        chunk_seq = extract_clean_subsequence(genome_seq, seq_len)

        print(f"  Chunk size: {chunk_size:>10,} bp ... ", end="", flush=True)

        if len(chunk_seq) < chunk_size:
            print(f"SKIP (genome only {len(genome_seq):,} bp)")
            results.append(SingleChunkResult(
                chunk_size_bp=chunk_size,
                status="skip",
                error=f"Genome too short ({len(genome_seq)} < {chunk_size})",
            ))
            continue

        result = benchmark_single_chunk(
            chunk_seq, evo2_model, ACGT_IDS, device, chunk_size
        )

        if result.status == "success":
            print(f"{result.total_inference_seconds:.2f}s "
                  f"(fwd={result.entropy_fwd_seconds:.2f}s, "
                  f"rc={result.entropy_rc_seconds:.2f}s, "
                  f"ntp={result.next_token_seconds:.2f}s) "
                  f"| {result.throughput_bp_per_sec:.0f} bp/s "
                  f"| {result.gpu_memory_peak_mb:.0f} MB")
        else:
            print(f"{result.status.upper()}: {result.error[:80]}")

        results.append(result)

        # Clean up between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def benchmark_full_locus(
    genome_seq: str,
    locus_length: int,
    chunk_sizes: List[int],
    evo2_model,
    ACGT_IDS: torch.Tensor,
    device: str,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[FullLocusResult]:
    """Benchmark score_locus_aligned_overlap with various chunk sizes."""

    print(f"\n{'='*70}")
    print(f"T3: FULL LOCUS SCORING BENCHMARKS (sequential)")
    print(f"{'='*70}")
    print(f"  Locus length: {locus_length:,} bp")
    print(f"  Chunk overlap: {chunk_overlap}")
    print(f"  Chunk sizes to test: {chunk_sizes}")
    print()

    locus_seq = extract_clean_subsequence(genome_seq, locus_length)
    actual_len = len(locus_seq)

    results = []
    for chunk_size in chunk_sizes:
        print(f"  chunk_size={chunk_size:>10,} ... ", end="", flush=True)

        try:
            reset_gpu_stats()

            # Calculate expected number of chunks
            step = max(1, chunk_size - chunk_overlap)
            n_chunks = len(list(range(0, actual_len, step)))

            gpu_sync()
            t0 = time.perf_counter()

            entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = \
                score_locus_aligned_overlap(
                    locus_seq,
                    evo2_model,
                    ACGT_IDS,
                    device,
                    max_chunk_len=chunk_size,
                    chunk_overlap=chunk_overlap,
                    compute_rcavg_entropy=True,
                )

            gpu_sync()
            total_s = time.perf_counter() - t0

            _, gpu_peak = get_gpu_memory_mb()
            avg_per_chunk = total_s / n_chunks if n_chunks > 0 else 0
            throughput = actual_len / total_s if total_s > 0 else 0

            print(f"{total_s:.2f}s | {n_chunks} chunks | "
                  f"{avg_per_chunk:.2f}s/chunk | "
                  f"{throughput:.0f} bp/s | {gpu_peak:.0f} MB")

            results.append(FullLocusResult(
                chunk_size_bp=chunk_size,
                locus_length_bp=actual_len,
                status="success",
                total_seconds=total_s,
                n_chunks=n_chunks,
                avg_seconds_per_chunk=avg_per_chunk,
                throughput_bp_per_sec=throughput,
                gpu_memory_peak_mb=gpu_peak,
            ))

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"OOM: {e}")
            results.append(FullLocusResult(
                chunk_size_bp=chunk_size,
                locus_length_bp=actual_len,
                status="oom",
                error=str(e),
            ))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(FullLocusResult(
                chunk_size_bp=chunk_size,
                locus_length_bp=actual_len,
                status="error",
                error=f"{type(e).__name__}: {e}",
            ))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results


def benchmark_full_locus_multigpu(
    genome_seq: str,
    locus_length: int,
    chunk_sizes: List[int],
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[FullLocusResult]:
    """Benchmark multi-GPU scoring if available."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus < 2:
        print(f"\n[BENCH] Skipping multi-GPU benchmark (only {n_gpus} GPU(s) available)")
        return []

    print(f"\n{'='*70}")
    print(f"T3b: FULL LOCUS SCORING BENCHMARKS (multi-GPU, {n_gpus} GPUs)")
    print(f"{'='*70}")

    locus_seq = extract_clean_subsequence(genome_seq, locus_length)
    actual_len = len(locus_seq)

    # Try importing the multi-GPU function
    try:
        from genome_scoring_jan26_drops import score_locus_aligned_overlap_multigpu
    except ImportError:
        print("  score_locus_aligned_overlap_multigpu not available, skipping")
        return []

    results = []
    for chunk_size in chunk_sizes:
        print(f"  chunk_size={chunk_size:>10,}, n_gpus={n_gpus} ... ", end="", flush=True)

        try:
            reset_gpu_stats()

            step = max(1, chunk_size - chunk_overlap)
            n_chunks = len(list(range(0, actual_len, step)))

            gpu_sync()
            t0 = time.perf_counter()

            entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = \
                score_locus_aligned_overlap_multigpu(
                    locus_seq,
                    n_gpus=n_gpus,
                    max_chunk_len=chunk_size,
                    chunk_overlap=chunk_overlap,
                    compute_rcavg_entropy=True,
                )

            gpu_sync()
            total_s = time.perf_counter() - t0

            _, gpu_peak = get_gpu_memory_mb()
            avg_per_chunk = total_s / n_chunks if n_chunks > 0 else 0
            throughput = actual_len / total_s if total_s > 0 else 0

            print(f"{total_s:.2f}s | {n_chunks} chunks | "
                  f"{throughput:.0f} bp/s | {gpu_peak:.0f} MB")

            results.append(FullLocusResult(
                chunk_size_bp=chunk_size,
                locus_length_bp=actual_len,
                status="success",
                total_seconds=total_s,
                n_chunks=n_chunks,
                avg_seconds_per_chunk=avg_per_chunk,
                throughput_bp_per_sec=throughput,
                gpu_memory_peak_mb=gpu_peak,
                mode="multigpu",
                n_gpus_used=n_gpus,
            ))

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"ERROR: {e}")
            results.append(FullLocusResult(
                chunk_size_bp=chunk_size,
                locus_length_bp=actual_len,
                status="error",
                error=f"{type(e).__name__}: {e}",
                mode="multigpu",
                n_gpus_used=n_gpus,
            ))

    return results


def benchmark_file_writing(
    locus_length: int,
    output_dir: str,
) -> FileWriteResult:
    """
    Benchmark file I/O using synthetic data matching pipeline output shapes.

    Generates arrays of the same size that the pipeline would produce for
    a locus of the given length, then times writing them in the same formats.
    """
    print(f"\n{'='*70}")
    print(f"T4: FILE WRITING BENCHMARKS")
    print(f"{'='*70}")
    print(f"  Locus length: {locus_length:,} positions")

    L = locus_length

    # Generate synthetic data matching pipeline output shapes
    entropy_fwd = np.random.rand(L).astype(np.float32) * 1.4
    ppx_fwd = np.exp(entropy_fwd)
    entropy_rc = np.random.rand(L).astype(np.float32) * 1.4
    ppx_rc = np.exp(entropy_rc)
    p4 = np.random.rand(L, 4).astype(np.float32)
    p4 = p4 / p4.sum(axis=1, keepdims=True)  # normalize
    ll_next = np.random.rand(L).astype(np.float32) * -5
    true_tok = np.array([np.random.choice(["A", "C", "G", "T"]) for _ in range(L)], dtype=object)
    locus_seq = "".join(true_tok)
    is_exon = np.random.randint(0, 2, size=L).astype(np.int32)
    exon_id = np.zeros(L, dtype=np.int32)
    pos = np.arange(1, L + 1)
    dist_start = np.random.rand(L).astype(np.float32) * 1000
    dist_end = np.random.rand(L).astype(np.float32) * 1000

    os.makedirs(output_dir, exist_ok=True)

    # T4a: Write main TSV
    tsv_path = os.path.join(output_dir, "benchmark_data.tsv")
    unit = "bits"
    scale = 1.0 / math.log(2.0)

    t0 = time.perf_counter()
    with open(tsv_path, "w") as f:
        f.write(f"Pos\tEntropy({unit})\tPerplexity(e)\t"
                f"P(A)\tP(C)\tP(G)\tP(T)\tTrueToken\tLL_next(nats)"
                f"\tEntropy_RCavg({unit})\tPerplexity_RCavg(e)"
                "\tBase\tOrientedIdx\tIsExon\tExonID\t"
                "DistToExonStart\tDistToExonEnd\n")
        for i in range(L):
            ent = float(entropy_fwd[i]) * scale
            px = float(ppx_fwd[i])
            ll = float(ll_next[i])
            a, c, g, t = p4[i, :].tolist()
            ent_rc = float(entropy_rc[i]) * scale
            px_rc = float(ppx_rc[i])
            f.write(
                f"{int(pos[i])}\t"
                f"{ent:.6f}\t{px:.6f}\t"
                f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\t"
                f"{true_tok[i]}\t{ll:.6f}\t"
                f"{ent_rc:.6f}\t{px_rc:.6f}\t"
                f"{locus_seq[i]}\t{i}\t{int(is_exon[i])}\t{int(exon_id[i])}\t"
                f"{dist_start[i]:.1f}\t{dist_end[i]:.1f}\n"
            )
    t_tsv = time.perf_counter() - t0
    tsv_size = os.path.getsize(tsv_path)
    print(f"  TSV write:        {t_tsv:.3f}s ({tsv_size/1024/1024:.1f} MB, {L:,} rows)")

    # T4b: Write drops.txt
    drops_path = os.path.join(output_dir, "benchmark_drops.txt")
    # Generate synthetic drop data
    n_drops = max(1, L // 5000)
    drop_positions = sorted(np.random.choice(L, size=min(n_drops, L), replace=False))
    drop_scores = np.random.rand(len(drop_positions)) * -4

    t0 = time.perf_counter()
    with open(drops_path, "w") as f:
        f.write("# Drop detection results (benchmark)\n")
        f.write("# Format: method_name<TAB>pos1:score1,pos2:score2,...\n\n")
        for method in ["zscore", "mad", "local"]:
            entries = [f"{pos}:{score:.4f}" for pos, score in zip(drop_positions, drop_scores)]
            f.write(f"{method}\t" + ",".join(entries) + "\n")
    t_drops = time.perf_counter() - t0
    print(f"  Drops write:      {t_drops:.4f}s ({n_drops} drops)")

    # T4c: Write rises.txt
    rises_path = os.path.join(output_dir, "benchmark_rises.txt")
    t0 = time.perf_counter()
    with open(rises_path, "w") as f:
        f.write("# Rise detection results (benchmark)\n\n")
        for method in ["zscore", "mad"]:
            entries = [f"{pos}:{score:.4f}" for pos, score in zip(drop_positions, drop_scores)]
            f.write(f"{method}\t" + ",".join(entries) + "\n")
    t_rises = time.perf_counter() - t0
    print(f"  Rises write:      {t_rises:.4f}s")

    # T4d: Write window summary
    summary_path = os.path.join(output_dir, "benchmark_window_summary.tsv")
    entropy_main = entropy_fwd * scale

    t0 = time.perf_counter()
    write_window_summary(summary_path, entropy_main, is_exon, win=200, step=50)
    t_summary = time.perf_counter() - t0
    print(f"  Window summary:   {t_summary:.4f}s")

    total = t_tsv + t_drops + t_rises + t_summary
    print(f"  Total file I/O:   {total:.3f}s")

    # Clean up temp files
    for p in [tsv_path, drops_path, rises_path, summary_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    return FileWriteResult(
        tsv_write_seconds=t_tsv,
        drops_write_seconds=t_drops,
        rises_write_seconds=t_rises,
        window_summary_seconds=t_summary,
        total_seconds=total,
        tsv_rows=L,
        tsv_file_size_bytes=tsv_size,
    )


# ============================================================================
# FULL PIPELINE BENCHMARK (replicates run_one_locus end-to-end)
# ============================================================================

def benchmark_full_pipeline(
    genome_seq: str,
    genome_id: str,
    evo2_model,
    ACGT_IDS: torch.Tensor,
    device: str,
    output_dir: str,
    chunk_size: int = 15_000,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> FullPipelineResult:
    """
    Benchmark the COMPLETE analysis pipeline on the whole genome.

    Replicates what run_one_locus() does end-to-end:
      1. Inference: score_locus_aligned_overlap on entire genome
      2. Unit conversion: nats -> bits
      3. Drop detection: all 6 methods (zscore, mad, local, derivative, win_shift, cusum)
      4. Rise detection: all 6 methods
      5. Plot suite generation
      6. Method comparison report
      7. File I/O: TSV, drops.txt, rises.txt, window_summary, metadata JSON
    """
    locus_length = len(genome_seq)
    result = FullPipelineResult(
        locus_length_bp=locus_length,
        chunk_size_bp=chunk_size,
    )

    print(f"\n{'='*70}")
    print(f"T5: FULL PIPELINE BENCHMARK (whole genome)")
    print(f"{'='*70}")
    print(f"  Genome: {genome_id} ({locus_length:,} bp)")
    print(f"  Chunk size: {chunk_size:,} bp")
    print(f"  Chunk overlap: {chunk_overlap} bp")
    step = max(1, chunk_size - chunk_overlap)
    n_chunks_expected = len(list(range(0, locus_length, step)))
    print(f"  Expected chunks: {n_chunks_expected}")
    print()

    pipeline_output_dir = os.path.join(output_dir, "full_pipeline_output")
    base_name = f"benchmark_{genome_id}"

    # Create OutputManager for plot_suite and file I/O
    output_mgr = OutputManager(
        out_dir=pipeline_output_dir,
        gene_tag=f"benchmark_{genome_id}",
        detection_methods=["zscore", "mad", "local", "derivative", "win_shift", "cusum"],
        organism="ecoli",
        include_timestamp=True,
    )

    t_pipeline_start = time.perf_counter()

    # ---- Step 1: Inference (scoring) ----
    print(f"  [STEP 1/7] Inference: scoring {locus_length:,} bp ...", flush=True)
    try:
        reset_gpu_stats()
        gpu_sync()
        t0 = time.perf_counter()

        entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = \
            score_locus_aligned_overlap(
                genome_seq,
                evo2_model,
                ACGT_IDS,
                device,
                max_chunk_len=chunk_size,
                chunk_overlap=chunk_overlap,
                compute_rcavg_entropy=True,
            )

        gpu_sync()
        result.inference_seconds = time.perf_counter() - t0
        result.n_chunks = n_chunks_expected
        _, result.gpu_memory_peak_mb = get_gpu_memory_mb()

        print(f"            {result.inference_seconds:.2f}s "
              f"({result.n_chunks} chunks, "
              f"{locus_length / result.inference_seconds:.0f} bp/s)")

    except Exception as e:
        result.status = "error"
        result.error = f"Inference failed: {type(e).__name__}: {e}"
        result.total_seconds = time.perf_counter() - t_pipeline_start
        print(f"            FAILED: {e}")
        traceback.print_exc()
        return result

    # ---- Step 2: Unit conversion (nats -> bits) ----
    print(f"  [STEP 2/7] Unit conversion (nats -> bits) ...", end="", flush=True)
    t0 = time.perf_counter()
    scale = 1.0 / math.log(2.0)
    entropy_main_u = entropy_fwd * scale if entropy_fwd is not None else np.zeros(locus_length)
    ppx_main_u = ppx_fwd
    entropy_rc_u = entropy_rc * scale if entropy_rc is not None else np.zeros(locus_length)
    ppx_rc_u = ppx_rc
    result.unit_conversion_seconds = time.perf_counter() - t0
    print(f" {result.unit_conversion_seconds:.4f}s")

    # Create synthetic is_exon (E. coli is prokaryotic, ~87% coding)
    is_exon = np.ones(locus_length, dtype=np.int32)
    exon_id = np.zeros(locus_length, dtype=np.int32)

    # ---- Step 3: Drop detection (all methods) ----
    print(f"  [STEP 3/7] Drop detection (6 methods) ...")
    t_drop_start = time.perf_counter()
    drops = {}
    scored_drops = {}
    detection_methods = ["zscore", "mad", "local", "derivative", "win_shift", "cusum"]

    # Z-score
    t0 = time.perf_counter()
    scored_drops["zscore"] = detect_drops_zscore(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, zscore_threshold=2.0, min_separation=100
    )
    drops["zscore"] = _drops_scored_to_positions(scored_drops["zscore"])
    t_method = time.perf_counter() - t0
    result.drop_detection_by_method["zscore"] = t_method
    result.n_drops_by_method["zscore"] = len(drops["zscore"])
    print(f"            zscore:     {t_method:.3f}s ({len(drops['zscore'])} drops)")

    # MAD
    t0 = time.perf_counter()
    scored_drops["mad"] = detect_drops_mad(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, mad_threshold=3.0, min_separation=100
    )
    drops["mad"] = _drops_scored_to_positions(scored_drops["mad"])
    t_method = time.perf_counter() - t0
    result.drop_detection_by_method["mad"] = t_method
    result.n_drops_by_method["mad"] = len(drops["mad"])
    print(f"            mad:        {t_method:.3f}s ({len(drops['mad'])} drops)")

    # Local baseline
    t0 = time.perf_counter()
    scored_drops["local"] = detect_drops_local_baseline(
        entropy_main_u, window_baseline=5000, threshold_sigma=2.0, min_separation=100
    )
    drops["local"] = _drops_scored_to_positions(scored_drops["local"])
    t_method = time.perf_counter() - t0
    result.drop_detection_by_method["local"] = t_method
    result.n_drops_by_method["local"] = len(drops["local"])
    print(f"            local:      {t_method:.3f}s ({len(drops['local'])} drops)")

    # Derivative
    t0 = time.perf_counter()
    drops["derivative"] = detect_drops_derivative(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, thr_quantile=DROP_DERIV_Q
    )
    t_method = time.perf_counter() - t0
    result.drop_detection_by_method["derivative"] = t_method
    result.n_drops_by_method["derivative"] = len(drops["derivative"])
    print(f"            derivative: {t_method:.3f}s ({len(drops['derivative'])} drops)")

    # Window mean shift
    t0 = time.perf_counter()
    drops["win_shift"] = detect_drops_window_mean_shift(
        entropy_main_u, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK
    )
    t_method = time.perf_counter() - t0
    result.drop_detection_by_method["win_shift"] = t_method
    result.n_drops_by_method["win_shift"] = len(drops["win_shift"])
    print(f"            win_shift:  {t_method:.3f}s ({len(drops['win_shift'])} drops)")

    # CUSUM
    t0 = time.perf_counter()
    drops["cusum"] = detect_drops_cusum(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H
    )
    t_method = time.perf_counter() - t0
    result.drop_detection_by_method["cusum"] = t_method
    result.n_drops_by_method["cusum"] = len(drops["cusum"])
    print(f"            cusum:      {t_method:.3f}s ({len(drops['cusum'])} drops)")

    result.drop_detection_seconds = time.perf_counter() - t_drop_start
    print(f"            TOTAL:      {result.drop_detection_seconds:.3f}s")

    # ---- Step 4: Rise detection (all methods) ----
    print(f"  [STEP 4/7] Rise detection (6 methods) ...")
    t_rise_start = time.perf_counter()
    rises = {}
    scored_rises = {}

    # Derivative rises
    t0 = time.perf_counter()
    rises["derivative"] = detect_rises_derivative(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, thr_quantile=(1.0 - DROP_DERIV_Q)
    )
    t_method = time.perf_counter() - t0
    result.rise_detection_by_method["derivative"] = t_method
    result.n_rises_by_method["derivative"] = len(rises["derivative"])
    print(f"            derivative: {t_method:.3f}s ({len(rises['derivative'])} rises)")

    # Window mean shift rises
    t0 = time.perf_counter()
    rises["win_shift"] = detect_rises_window_mean_shift(
        entropy_main_u, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK
    )
    t_method = time.perf_counter() - t0
    result.rise_detection_by_method["win_shift"] = t_method
    result.n_rises_by_method["win_shift"] = len(rises["win_shift"])
    print(f"            win_shift:  {t_method:.3f}s ({len(rises['win_shift'])} rises)")

    # CUSUM rises
    t0 = time.perf_counter()
    rises["cusum"] = detect_rises_cusum(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H
    )
    t_method = time.perf_counter() - t0
    result.rise_detection_by_method["cusum"] = t_method
    result.n_rises_by_method["cusum"] = len(rises["cusum"])
    print(f"            cusum:      {t_method:.3f}s ({len(rises['cusum'])} rises)")

    # Z-score rises
    t0 = time.perf_counter()
    scored_rises["zscore"] = detect_rises_zscore(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, zscore_threshold=2.0, min_separation=100
    )
    rises["zscore"] = _drops_scored_to_positions(scored_rises["zscore"])
    t_method = time.perf_counter() - t0
    result.rise_detection_by_method["zscore"] = t_method
    result.n_rises_by_method["zscore"] = len(rises["zscore"])
    print(f"            zscore:     {t_method:.3f}s ({len(rises['zscore'])} rises)")

    # MAD rises
    t0 = time.perf_counter()
    scored_rises["mad"] = detect_rises_mad(
        entropy_main_u, smooth_w=DROP_SMOOTH_W, mad_threshold=3.0, min_separation=100
    )
    rises["mad"] = _drops_scored_to_positions(scored_rises["mad"])
    t_method = time.perf_counter() - t0
    result.rise_detection_by_method["mad"] = t_method
    result.n_rises_by_method["mad"] = len(rises["mad"])
    print(f"            mad:        {t_method:.3f}s ({len(rises['mad'])} rises)")

    # Local baseline rises
    t0 = time.perf_counter()
    scored_rises["local"] = detect_rises_local_baseline(
        entropy_main_u, window_baseline=5000, threshold_sigma=2.0, min_separation=100
    )
    rises["local"] = _drops_scored_to_positions(scored_rises["local"])
    t_method = time.perf_counter() - t0
    result.rise_detection_by_method["local"] = t_method
    result.n_rises_by_method["local"] = len(rises["local"])
    print(f"            local:      {t_method:.3f}s ({len(rises['local'])} rises)")

    result.rise_detection_seconds = time.perf_counter() - t_rise_start
    print(f"            TOTAL:      {result.rise_detection_seconds:.3f}s")

    # ---- Step 5: Plot suite ----
    print(f"  [STEP 5/7] Plot suite generation ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        title_prefix = f"Benchmark ({genome_id})"
        plot_suite(
            output_mgr=output_mgr,
            base_name=base_name,
            entropy_main=entropy_main_u,
            is_exon=is_exon,
            drop_points=drops,
            scored_drops=scored_drops,
            title_prefix=title_prefix,
            smooth_w=DROP_SMOOTH_W,
            zoom_bp=0,
            max_zoom_plots=0,  # skip zoom plots for speed
            plot_style="plain",
            unit="bits",
            ylim=None,
            annotate_top_n=5,
            genomic_start=0,
            rise_points=rises,
            scored_rises=scored_rises,
            gff_intervals=None,
        )
        result.plot_suite_seconds = time.perf_counter() - t0
        print(f" {result.plot_suite_seconds:.2f}s")
    except Exception as e:
        result.plot_suite_seconds = time.perf_counter() - t0
        print(f" FAILED ({e})")
        traceback.print_exc()

    # ---- Step 5b: Method comparison report ----
    print(f"  [STEP 5b]  Method comparison report ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        generate_method_comparison_report(
            output_mgr=output_mgr,
            base_name=base_name,
            scored_drops=scored_drops,
            drop_points=drops,
            entropy=entropy_main_u,
            is_exon=is_exon,
            title_prefix=f"Benchmark ({genome_id})",
            tolerance_bp=100,
            smooth_w=DROP_SMOOTH_W,
        )
        result.method_comparison_seconds = time.perf_counter() - t0
        print(f" {result.method_comparison_seconds:.2f}s")
    except Exception as e:
        result.method_comparison_seconds = time.perf_counter() - t0
        print(f" FAILED ({e})")

    # ---- Step 6: File I/O (TSV) ----
    print(f"  [STEP 6/7] File I/O ...")
    pos = np.arange(1, locus_length + 1)
    dist_to_exon_start = np.zeros(locus_length, dtype=np.float32)
    dist_to_exon_end = np.zeros(locus_length, dtype=np.float32)

    # 6a: Main TSV
    tsv_path = output_mgr.data_path(f"{base_name}.tsv")
    t0 = time.perf_counter()
    with open(tsv_path, "w") as f:
        f.write(f"Pos\tEntropy(bits)\tPerplexity(e)\t"
                f"P(A)\tP(C)\tP(G)\tP(T)\tTrueToken\tLL_next(nats)"
                f"\tEntropy_RCavg(bits)\tPerplexity_RCavg(e)"
                "\tBase\tOrientedIdx\tIsExon\tExonID\t"
                "DistToExonStart\tDistToExonEnd\n")
        for i in range(locus_length):
            ent = float(entropy_main_u[i]) if not np.isnan(entropy_main_u[i]) else np.nan
            px = float(ppx_main_u[i]) if not np.isnan(ppx_main_u[i]) else np.nan
            ll_val = float(ll_next[i]) if ll_next is not None and not np.isnan(ll_next[i]) else np.nan
            a_val = float(p4[i, 0]) if p4 is not None else np.nan
            c_val = float(p4[i, 1]) if p4 is not None else np.nan
            g_val = float(p4[i, 2]) if p4 is not None else np.nan
            t_val = float(p4[i, 3]) if p4 is not None else np.nan
            ent_rc = float(entropy_rc_u[i]) if not np.isnan(entropy_rc_u[i]) else np.nan
            px_rc = float(ppx_rc_u[i]) if ppx_rc_u is not None and not np.isnan(ppx_rc_u[i]) else np.nan
            tok = true_tok[i] if true_tok is not None else "N"
            f.write(
                f"{int(pos[i])}\t"
                f"{ent:.6f}\t{px:.6f}\t"
                f"{a_val:.6f}\t{c_val:.6f}\t{g_val:.6f}\t{t_val:.6f}\t"
                f"{tok}\t{ll_val:.6f}\t"
                f"{ent_rc:.6f}\t{px_rc:.6f}\t"
                f"{genome_seq[i]}\t{i}\t{int(is_exon[i])}\t{int(exon_id[i])}\t"
                f"{dist_to_exon_start[i]:.1f}\t{dist_to_exon_end[i]:.1f}\n"
            )
    result.tsv_write_seconds = time.perf_counter() - t0
    tsv_size = os.path.getsize(tsv_path)
    print(f"            TSV:              {result.tsv_write_seconds:.2f}s "
          f"({tsv_size / 1024 / 1024:.1f} MB, {locus_length:,} rows)")

    # 6b: Drops file
    out_drops = output_mgr.data_path(f"{base_name}.drops.txt")
    t0 = time.perf_counter()
    with open(out_drops, "w") as f:
        f.write("# Drop detection results (benchmark full pipeline)\n")
        f.write("# Format: method_name<TAB>pos1:score1,pos2:score2,...\n\n")
        for method in ["derivative", "win_shift", "cusum"]:
            if method in drops and method not in scored_drops:
                pts = drops[method]
                f.write(f"{method}\t" + ",".join(map(str, pts)) + "\n")
        for method, scored_pts in scored_drops.items():
            if scored_pts:
                entries = [f"{p}:{s:.4f}" for p, s in scored_pts]
                f.write(f"{method}\t" + ",".join(entries) + "\n")
    result.drops_file_seconds = time.perf_counter() - t0
    print(f"            Drops:            {result.drops_file_seconds:.4f}s")

    # 6c: Rises file
    out_rises = output_mgr.data_path(f"{base_name}.rises.txt")
    t0 = time.perf_counter()
    with open(out_rises, "w") as f:
        f.write("# Rise detection results (benchmark full pipeline)\n")
        f.write("# Format: method_name<TAB>pos1:score1,pos2:score2,...\n\n")
        for method in ["derivative", "win_shift", "cusum"]:
            if method in rises and method not in scored_rises:
                pts = rises[method]
                f.write(f"{method}\t" + ",".join(map(str, pts)) + "\n")
        for method, scored_pts in scored_rises.items():
            if scored_pts:
                entries = [f"{p}:{s:.4f}" for p, s in scored_pts]
                f.write(f"{method}\t" + ",".join(entries) + "\n")
    result.rises_file_seconds = time.perf_counter() - t0
    print(f"            Rises:            {result.rises_file_seconds:.4f}s")

    # 6d: Window summary
    out_summary = output_mgr.data_path(f"{base_name}.window_summary.tsv")
    t0 = time.perf_counter()
    write_window_summary(out_summary, entropy_main_u, is_exon, win=200, step=50)
    result.window_summary_seconds = time.perf_counter() - t0
    print(f"            Window summary:   {result.window_summary_seconds:.3f}s")

    # 6e: Metadata JSON
    meta_path = output_mgr.meta_path(f"{base_name}.metadata.json")
    t0 = time.perf_counter()
    meta = {
        "genome_id": genome_id,
        "locus_length_bp": locus_length,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_chunks": n_chunks_expected,
        "detection_methods": detection_methods,
        "n_drops": {m: len(drops.get(m, [])) for m in detection_methods},
        "n_rises": {m: len(rises.get(m, [])) for m in detection_methods},
        "unit": "bits",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    result.metadata_json_seconds = time.perf_counter() - t0
    print(f"            Metadata JSON:    {result.metadata_json_seconds:.4f}s")

    # ---- Totals ----
    result.total_analysis_seconds = (
        result.unit_conversion_seconds +
        result.drop_detection_seconds +
        result.rise_detection_seconds +
        result.plot_suite_seconds +
        result.method_comparison_seconds +
        result.tsv_write_seconds +
        result.drops_file_seconds +
        result.rises_file_seconds +
        result.window_summary_seconds +
        result.metadata_json_seconds
    )
    result.total_seconds = time.perf_counter() - t_pipeline_start
    result.status = "success"

    print(f"\n  [STEP 7/7] SUMMARY")
    print(f"            Inference:         {result.inference_seconds:.2f}s")
    print(f"            Drop detection:    {result.drop_detection_seconds:.3f}s")
    print(f"            Rise detection:    {result.rise_detection_seconds:.3f}s")
    print(f"            Plot suite:        {result.plot_suite_seconds:.2f}s")
    print(f"            Method comparison: {result.method_comparison_seconds:.2f}s")
    print(f"            File I/O total:    "
          f"{result.tsv_write_seconds + result.drops_file_seconds + result.rises_file_seconds + result.window_summary_seconds + result.metadata_json_seconds:.2f}s")
    print(f"            ─────────────────────────────")
    print(f"            Analysis (non-inf):{result.total_analysis_seconds:.2f}s")
    print(f"            TOTAL (end-to-end):{result.total_seconds:.2f}s")
    print(f"            Output dir: {output_mgr.base_dir}")

    return result


# ============================================================================
# OUTPUT
# ============================================================================

def _save_partial(report: BenchmarkReport, output_dir: str, timestamp: str):
    """Save partial results after each benchmark phase (overwrites same file)."""
    json_path = os.path.join(output_dir, f"pipeline_benchmark_{timestamp}.json")
    save_json_report(report, json_path)
    print(f"  [checkpoint saved]")


def save_json_report(report: BenchmarkReport, output_path: str):
    """Save structured benchmark results to JSON."""
    data = {
        "benchmark_info": {
            "timestamp": report.timestamp,
            "genome_source": report.genome_source,
            "genome_length_bp": report.genome_length_bp,
            "model_name": report.model_name,
            "device": report.device,
            "cuda_available": report.cuda_available,
            "n_gpus": report.n_gpus,
            "pytorch_version": report.pytorch_version,
            "chunk_overlap": report.chunk_overlap,
            "multi_gpu_data_parallel": False,
            "multi_gpu_model_parallel": True,
            "note": "Current pipeline processes chunks SEQUENTIALLY. "
                    "Vortex handles model parallelism (sharding weights), "
                    "not data parallelism (distributing sequences)."
        },
        "model_loading": asdict(report.model_loading) if report.model_loading else None,
        "single_chunk_results": [asdict(r) for r in report.single_chunk_results],
        "full_locus_results": [asdict(r) for r in report.full_locus_results],
        "file_writing": asdict(report.file_writing) if report.file_writing else None,
        "full_pipeline": asdict(report.full_pipeline) if report.full_pipeline else None,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[BENCH] JSON report saved: {output_path}")


def generate_summary_text(report: BenchmarkReport, output_path: str):
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("GENOME SCORING PIPELINE - TIMING BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append(f"Timestamp:       {report.timestamp}")
    lines.append(f"Genome:          {report.genome_source} ({report.genome_length_bp:,} bp)")
    lines.append(f"Model:           {report.model_name}")
    lines.append(f"Device:          {report.device}")
    lines.append(f"GPUs:            {report.n_gpus}")
    lines.append(f"PyTorch:         {report.pytorch_version}")
    lines.append(f"Chunk overlap:   {report.chunk_overlap} bp")
    lines.append("")

    # GPU parallelism note
    lines.append("NOTE: The current pipeline does NOT use multi-GPU data parallelism.")
    lines.append("All chunks are processed sequentially on one model instance.")
    lines.append("Vortex handles model parallelism (weight sharding) only.")
    lines.append("")

    # T1: Model loading
    if report.model_loading:
        ml = report.model_loading
        lines.append("-" * 70)
        lines.append("T1: MODEL LOADING")
        lines.append("-" * 70)
        lines.append(f"  Model load time:      {ml.load_time_seconds:.2f}s")
        lines.append(f"  Tokenizer setup:      {ml.tokenizer_setup_seconds:.4f}s")
        lines.append(f"  Total:                {ml.total_seconds:.2f}s")
        lines.append(f"  GPU memory after:     {ml.gpu_memory_after_mb:.0f} MB")
        lines.append("")

    # T2: Single-chunk results
    if report.single_chunk_results:
        lines.append("-" * 70)
        lines.append("T2: SINGLE-CHUNK INFERENCE (4 forward passes per chunk)")
        lines.append("-" * 70)
        header = (f"  {'Chunk Size':>12}  {'Status':>7}  {'Fwd(s)':>8}  {'RC(s)':>8}  "
                  f"{'NTP(s)':>8}  {'Total(s)':>9}  {'bp/sec':>10}  {'GPU MB':>8}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for r in report.single_chunk_results:
            if r.status == "success":
                lines.append(
                    f"  {r.chunk_size_bp:>12,}  {r.status:>7}  "
                    f"{r.entropy_fwd_seconds:>8.2f}  {r.entropy_rc_seconds:>8.2f}  "
                    f"{r.next_token_seconds:>8.2f}  {r.total_inference_seconds:>9.2f}  "
                    f"{r.throughput_bp_per_sec:>10.0f}  {r.gpu_memory_peak_mb:>8.0f}"
                )
            else:
                lines.append(
                    f"  {r.chunk_size_bp:>12,}  {r.status:>7}  "
                    f"{'--':>8}  {'--':>8}  {'--':>8}  {'--':>9}  "
                    f"{'--':>10}  {'--':>8}  {r.error[:40]}"
                )
        lines.append("")

    # T3: Full locus results
    if report.full_locus_results:
        lines.append("-" * 70)
        lines.append("T3: FULL LOCUS SCORING")
        lines.append("-" * 70)
        header = (f"  {'Chunk Size':>12}  {'Mode':>10}  {'Status':>7}  "
                  f"{'Chunks':>6}  {'Total(s)':>9}  {'Avg/Chunk':>9}  "
                  f"{'bp/sec':>10}  {'GPU MB':>8}")
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for r in report.full_locus_results:
            if r.status == "success":
                lines.append(
                    f"  {r.chunk_size_bp:>12,}  {r.mode:>10}  {r.status:>7}  "
                    f"{r.n_chunks:>6}  {r.total_seconds:>9.2f}  "
                    f"{r.avg_seconds_per_chunk:>9.2f}  "
                    f"{r.throughput_bp_per_sec:>10.0f}  {r.gpu_memory_peak_mb:>8.0f}"
                )
            else:
                lines.append(
                    f"  {r.chunk_size_bp:>12,}  {r.mode:>10}  {r.status:>7}  "
                    f"{'--':>6}  {'--':>9}  {'--':>9}  {'--':>10}  {'--':>8}"
                )
        lines.append("")

    # T4: File writing
    if report.file_writing:
        fw = report.file_writing
        lines.append("-" * 70)
        lines.append("T4: FILE WRITING (synthetic data)")
        lines.append("-" * 70)
        lines.append(f"  TSV write:            {fw.tsv_write_seconds:.3f}s "
                     f"({fw.tsv_file_size_bytes/1024/1024:.1f} MB, {fw.tsv_rows:,} rows)")
        lines.append(f"  Drops write:          {fw.drops_write_seconds:.4f}s")
        lines.append(f"  Rises write:          {fw.rises_write_seconds:.4f}s")
        lines.append(f"  Window summary:       {fw.window_summary_seconds:.4f}s")
        lines.append(f"  Total file I/O:       {fw.total_seconds:.3f}s")
        lines.append("")

    # T5: Full pipeline
    if report.full_pipeline and report.full_pipeline.status == "success":
        fp = report.full_pipeline
        lines.append("-" * 70)
        lines.append(f"T5: FULL PIPELINE (whole genome, {fp.locus_length_bp:,} bp)")
        lines.append("-" * 70)
        lines.append(f"  Chunk size:           {fp.chunk_size_bp:,} bp ({fp.n_chunks} chunks)")
        lines.append("")
        lines.append(f"  Inference:            {fp.inference_seconds:.2f}s")
        lines.append(f"  Unit conversion:      {fp.unit_conversion_seconds:.4f}s")
        lines.append("")
        lines.append(f"  Drop detection:       {fp.drop_detection_seconds:.3f}s")
        for method, t in sorted(fp.drop_detection_by_method.items()):
            n = fp.n_drops_by_method.get(method, 0)
            lines.append(f"    {method:>12}:       {t:.3f}s  ({n} drops)")
        lines.append("")
        lines.append(f"  Rise detection:       {fp.rise_detection_seconds:.3f}s")
        for method, t in sorted(fp.rise_detection_by_method.items()):
            n = fp.n_rises_by_method.get(method, 0)
            lines.append(f"    {method:>12}:       {t:.3f}s  ({n} rises)")
        lines.append("")
        lines.append(f"  Plot suite:           {fp.plot_suite_seconds:.2f}s")
        lines.append(f"  Method comparison:    {fp.method_comparison_seconds:.2f}s")
        lines.append("")
        file_io = (fp.tsv_write_seconds + fp.drops_file_seconds +
                   fp.rises_file_seconds + fp.window_summary_seconds +
                   fp.metadata_json_seconds)
        lines.append(f"  File I/O:             {file_io:.2f}s")
        lines.append(f"    TSV:                {fp.tsv_write_seconds:.2f}s")
        lines.append(f"    Drops file:         {fp.drops_file_seconds:.4f}s")
        lines.append(f"    Rises file:         {fp.rises_file_seconds:.4f}s")
        lines.append(f"    Window summary:     {fp.window_summary_seconds:.3f}s")
        lines.append(f"    Metadata JSON:      {fp.metadata_json_seconds:.4f}s")
        lines.append("")
        lines.append(f"  Analysis (non-inf):   {fp.total_analysis_seconds:.2f}s")
        lines.append(f"  TOTAL (end-to-end):   {fp.total_seconds:.2f}s")
        lines.append(f"  GPU memory peak:      {fp.gpu_memory_peak_mb:.0f} MB")
        lines.append("")

        # Proportion breakdown
        lines.append("  Time Breakdown:")
        inf_pct = fp.inference_seconds / fp.total_seconds * 100
        drop_pct = fp.drop_detection_seconds / fp.total_seconds * 100
        rise_pct = fp.rise_detection_seconds / fp.total_seconds * 100
        plot_pct = (fp.plot_suite_seconds + fp.method_comparison_seconds) / fp.total_seconds * 100
        io_pct = file_io / fp.total_seconds * 100
        other_pct = 100 - inf_pct - drop_pct - rise_pct - plot_pct - io_pct
        lines.append(f"    Inference:          {inf_pct:5.1f}%")
        lines.append(f"    Drop detection:     {drop_pct:5.1f}%")
        lines.append(f"    Rise detection:     {rise_pct:5.1f}%")
        lines.append(f"    Plotting:           {plot_pct:5.1f}%")
        lines.append(f"    File I/O:           {io_pct:5.1f}%")
        lines.append(f"    Other:              {other_pct:5.1f}%")
        lines.append("")

    # Summary
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    if report.model_loading:
        lines.append(f"  Model loading:        {report.model_loading.total_seconds:.2f}s")
    if report.single_chunk_results:
        successful = [r for r in report.single_chunk_results if r.status == "success"]
        if successful:
            best = min(successful, key=lambda r: r.total_inference_seconds / r.chunk_size_bp)
            lines.append(f"  Best throughput:      {best.throughput_bp_per_sec:.0f} bp/s "
                        f"(chunk_size={best.chunk_size_bp:,})")
            failed = [r for r in report.single_chunk_results if r.status != "success"]
            if failed:
                lines.append(f"  Failed chunk sizes:   "
                            + ", ".join(f"{r.chunk_size_bp:,}" for r in failed))
    if report.full_pipeline and report.full_pipeline.status == "success":
        fp = report.full_pipeline
        lines.append(f"  Whole genome pipeline: {fp.total_seconds:.2f}s "
                     f"({fp.locus_length_bp:,} bp)")
        lines.append(f"    Inference:          {fp.inference_seconds:.2f}s "
                     f"({fp.inference_seconds / fp.total_seconds * 100:.0f}%)")
        lines.append(f"    Analysis+I/O:       {fp.total_analysis_seconds:.2f}s "
                     f"({fp.total_analysis_seconds / fp.total_seconds * 100:.0f}%)")
    elif report.file_writing:
        lines.append(f"  File writing:         {report.file_writing.total_seconds:.3f}s")
    lines.append("")

    text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(text)
    print(f"[BENCH] Summary report saved: {output_path}")

    # Also print to stdout
    print()
    print(text)


# ============================================================================
# PUBLICATION-QUALITY PLOTS & TABLES
# ============================================================================

def _setup_plot_style():
    """Configure matplotlib for publication-quality (ICML/NeurIPS) figures."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.figsize": (7, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _format_bp(x, _pos=None):
    """Format base-pair values for axis ticks: 5000 -> '5K', 1000000 -> '1M'."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M" if x % 1_000_000 else f"{int(x // 1_000_000)}M"
    elif x >= 1_000:
        return f"{x / 1_000:.0f}K" if x % 1_000 == 0 else f"{x / 1_000:.1f}K"
    return str(int(x))


def generate_plots(report: BenchmarkReport, output_dir: str, timestamp: str):
    """
    Generate publication-quality figures for the benchmark report.

    Produces 5 figures:
      1. Inference time vs chunk size (log-log) with component breakdown
      2. Throughput (bp/sec) vs chunk size
      3. GPU memory vs chunk size
      4. Stacked bar: time breakdown per component
      5. Pipeline stage waterfall (model load, inference, file I/O)
      6. Full-locus: sequential vs multi-GPU comparison (if data available)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import FancyBboxPatch

    _setup_plot_style()

    successful = [r for r in report.single_chunk_results if r.status == "success"]
    failed = [r for r in report.single_chunk_results if r.status != "success"]

    if not successful:
        print("[BENCH] No successful single-chunk results; skipping plots.")
        return

    chunk_sizes = np.array([r.chunk_size_bp for r in successful])
    t_fwd = np.array([r.entropy_fwd_seconds for r in successful])
    t_rc = np.array([r.entropy_rc_seconds for r in successful])
    t_ntp = np.array([r.next_token_seconds for r in successful])
    t_total = np.array([r.total_inference_seconds for r in successful])
    throughput = np.array([r.throughput_bp_per_sec for r in successful])
    gpu_mem = np.array([r.gpu_memory_peak_mb for r in successful])

    # Color palette (colorblind-friendly, ICML-style)
    C_FWD = "#2196F3"      # blue
    C_RC = "#FF9800"       # orange
    C_NTP = "#4CAF50"      # green
    C_TOTAL = "#212121"    # dark grey
    C_FAIL = "#F44336"     # red
    C_MEM = "#9C27B0"      # purple
    C_THRU = "#00796B"     # teal
    C_SEQ = "#1565C0"      # dark blue
    C_MGPU = "#E65100"     # dark orange

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Figure 1: Inference time vs chunk size (log-log) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(chunk_sizes, t_fwd, "o-", color=C_FWD, label="Fwd entropy (1 pass)", zorder=3)
    ax.plot(chunk_sizes, t_rc, "s-", color=C_RC, label="RC entropy (2 passes)", zorder=3)
    ax.plot(chunk_sizes, t_ntp, "^-", color=C_NTP, label="Next-token logprobs (1 pass)", zorder=3)
    ax.plot(chunk_sizes, t_total, "D-", color=C_TOTAL, label="Total (4 passes)", linewidth=2, zorder=4)

    # Mark failed chunk sizes
    if failed:
        fail_sizes = [r.chunk_size_bp for r in failed]
        for fs in fail_sizes:
            ax.axvline(fs, color=C_FAIL, linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(fail_sizes[0], color=C_FAIL, linestyle="--", alpha=0.5,
                   linewidth=0.8, label=f"OOM / Error")

    # Mark model max_seqlen
    ax.axvline(1_048_576, color="#757575", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.text(1_048_576, ax.get_ylim()[1] * 0.95, " max_seqlen", fontsize=7,
            color="#757575", va="top", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_title("Evo2-7B Inference Time vs. Chunk Size")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.8")
    fig.savefig(os.path.join(plots_dir, f"fig1_time_vs_chunksize_{timestamp}.pdf"))
    fig.savefig(os.path.join(plots_dir, f"fig1_time_vs_chunksize_{timestamp}.png"))
    plt.close(fig)
    print(f"  [PLOT] Fig 1: time vs chunk size")

    # ---- Figure 2: Throughput vs chunk size ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(chunk_sizes, throughput / 1000, "o-", color=C_THRU, linewidth=2, zorder=3)
    ax.fill_between(chunk_sizes, 0, throughput / 1000, alpha=0.12, color=C_THRU)

    best_idx = np.argmax(throughput)
    ax.annotate(
        f"Peak: {throughput[best_idx]:,.0f} bp/s\n({_format_bp(chunk_sizes[best_idx])})",
        xy=(chunk_sizes[best_idx], throughput[best_idx] / 1000),
        xytext=(20, 15), textcoords="offset points",
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("Throughput (Kbp / sec)")
    ax.set_title("Evo2-7B Scoring Throughput vs. Chunk Size")
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(plots_dir, f"fig2_throughput_{timestamp}.pdf"))
    fig.savefig(os.path.join(plots_dir, f"fig2_throughput_{timestamp}.png"))
    plt.close(fig)
    print(f"  [PLOT] Fig 2: throughput")

    # ---- Figure 3: GPU peak memory vs chunk size ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(chunk_sizes, gpu_mem / 1024, "o-", color=C_MEM, linewidth=2, zorder=3)
    ax.fill_between(chunk_sizes, 0, gpu_mem / 1024, alpha=0.10, color=C_MEM)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Evo2-7B Peak GPU Memory vs. Chunk Size")
    ax.set_ylim(bottom=0)
    fig.savefig(os.path.join(plots_dir, f"fig3_gpu_memory_{timestamp}.pdf"))
    fig.savefig(os.path.join(plots_dir, f"fig3_gpu_memory_{timestamp}.png"))
    plt.close(fig)
    print(f"  [PLOT] Fig 3: GPU memory")

    # ---- Figure 4: Stacked bar breakdown per component ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_labels = [_format_bp(cs) for cs in chunk_sizes]
    x = np.arange(len(chunk_sizes))
    bar_w = 0.6

    bars_fwd = ax.bar(x, t_fwd, bar_w, label="Fwd Entropy", color=C_FWD, zorder=3)
    bars_rc = ax.bar(x, t_rc, bar_w, bottom=t_fwd, label="RC Entropy", color=C_RC, zorder=3)
    bars_ntp = ax.bar(x, t_ntp, bar_w, bottom=t_fwd + t_rc,
                      label="Next-Token", color=C_NTP, zorder=3)

    # Annotate total time above each bar
    for i in range(len(chunk_sizes)):
        ax.text(x[i], t_total[i] + t_total.max() * 0.02,
                f"{t_total[i]:.1f}s", ha="center", va="bottom", fontsize=7, color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_title("Inference Time Breakdown by Pipeline Component")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"fig4_component_breakdown_{timestamp}.pdf"))
    fig.savefig(os.path.join(plots_dir, f"fig4_component_breakdown_{timestamp}.png"))
    plt.close(fig)
    print(f"  [PLOT] Fig 4: component breakdown")

    # ---- Figure 5: Pipeline stage waterfall ----
    fig, ax = plt.subplots(figsize=(7, 3.5))
    stages = []
    times = []
    colors = []

    if report.model_loading:
        stages.append("Model\nLoading")
        times.append(report.model_loading.total_seconds)
        colors.append("#455A64")

    # Use the default chunk size result for representative inference time
    default_results = [r for r in successful if r.chunk_size_bp == 15_000]
    if default_results:
        rep = default_results[0]
    else:
        rep = successful[len(successful) // 2]  # middle entry

    stages.extend(["Fwd\nEntropy", "RC\nEntropy", "Next-Token\nLogprobs"])
    times.extend([rep.entropy_fwd_seconds, rep.entropy_rc_seconds, rep.next_token_seconds])
    colors.extend([C_FWD, C_RC, C_NTP])

    if report.file_writing:
        stages.append("File\nI/O")
        times.append(report.file_writing.total_seconds)
        colors.append("#795548")

    times = np.array(times)
    y_pos = np.arange(len(stages))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    for i, (bar, t) in enumerate(zip(bars, times)):
        ax.text(bar.get_width() + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.set_xlabel("Wall-Clock Time (seconds)")
    ax.set_title(f"Pipeline Stage Timing (chunk size = {_format_bp(rep.chunk_size_bp)})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"fig5_pipeline_waterfall_{timestamp}.pdf"))
    fig.savefig(os.path.join(plots_dir, f"fig5_pipeline_waterfall_{timestamp}.png"))
    plt.close(fig)
    print(f"  [PLOT] Fig 5: pipeline waterfall")

    # ---- Figure 6: Sequential vs multi-GPU (if data available) ----
    seq_locus = [r for r in report.full_locus_results
                 if r.mode == "sequential" and r.status == "success"]
    mgpu_locus = [r for r in report.full_locus_results
                  if r.mode == "multigpu" and r.status == "success"]

    if seq_locus and mgpu_locus:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # Find matching chunk sizes
        seq_dict = {r.chunk_size_bp: r for r in seq_locus}
        mgpu_dict = {r.chunk_size_bp: r for r in mgpu_locus}
        common = sorted(set(seq_dict) & set(mgpu_dict))

        if common:
            x = np.arange(len(common))
            labels = [_format_bp(cs) for cs in common]
            bar_w = 0.35

            seq_times = [seq_dict[cs].total_seconds for cs in common]
            mgpu_times = [mgpu_dict[cs].total_seconds for cs in common]
            n_gpus_used = mgpu_locus[0].n_gpus_used

            # Panel A: Wall-clock comparison
            ax1.bar(x - bar_w / 2, seq_times, bar_w, label="1 GPU (sequential)",
                    color=C_SEQ, zorder=3)
            ax1.bar(x + bar_w / 2, mgpu_times, bar_w, label=f"{n_gpus_used} GPUs (parallel)",
                    color=C_MGPU, zorder=3)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.set_xlabel("Chunk Size")
            ax1.set_ylabel("Total Time (seconds)")
            ax1.set_title("Wall-Clock Time")
            ax1.legend(framealpha=0.9, edgecolor="0.8")

            # Panel B: Speedup
            speedups = [seq_dict[cs].total_seconds / mgpu_dict[cs].total_seconds
                        for cs in common]
            ax2.bar(x, speedups, 0.5, color=C_MGPU, zorder=3)
            ax2.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8, zorder=2)
            ax2.axhline(n_gpus_used, color=C_FAIL, linestyle=":", linewidth=0.8, zorder=2,
                        label=f"Linear scaling ({n_gpus_used}x)")
            for i, sp in enumerate(speedups):
                ax2.text(i, sp + 0.05, f"{sp:.1f}x", ha="center", fontsize=8, fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_xlabel("Chunk Size")
            ax2.set_ylabel("Speedup")
            ax2.set_title(f"Speedup ({n_gpus_used}-GPU vs 1-GPU)")
            ax2.legend(framealpha=0.9, edgecolor="0.8")

            fig.suptitle("Multi-GPU Data Parallelism: Full-Locus Scoring", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"fig6_multigpu_comparison_{timestamp}.pdf"))
            fig.savefig(os.path.join(plots_dir, f"fig6_multigpu_comparison_{timestamp}.png"))
            plt.close(fig)
            print(f"  [PLOT] Fig 6: multi-GPU comparison")

    # ---- Figure 7: Combined 2x2 summary panel (good for paper main figure) ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Evo2-7B Genome Scoring Pipeline: Runtime Analysis", fontsize=13, y=0.98)

    # Panel (a): Time vs chunk size
    ax = axes[0, 0]
    ax.plot(chunk_sizes, t_total, "D-", color=C_TOTAL, linewidth=2, zorder=4, label="Total")
    ax.plot(chunk_sizes, t_fwd, "o-", color=C_FWD, label="Fwd entropy")
    ax.plot(chunk_sizes, t_rc, "s-", color=C_RC, label="RC entropy")
    ax.plot(chunk_sizes, t_ntp, "^-", color=C_NTP, label="Next-token")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("Time (s)")
    ax.set_title("(a) Inference Time")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # Panel (b): Throughput
    ax = axes[0, 1]
    ax.plot(chunk_sizes, throughput / 1000, "o-", color=C_THRU, linewidth=2, zorder=3)
    ax.fill_between(chunk_sizes, 0, throughput / 1000, alpha=0.12, color=C_THRU)
    best_idx = np.argmax(throughput)
    ax.plot(chunk_sizes[best_idx], throughput[best_idx] / 1000, "*",
            color=C_FAIL, markersize=12, zorder=5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("Throughput (Kbp/s)")
    ax.set_title("(b) Scoring Throughput")
    ax.set_ylim(bottom=0)

    # Panel (c): GPU memory
    ax = axes[1, 0]
    ax.plot(chunk_sizes, gpu_mem / 1024, "o-", color=C_MEM, linewidth=2, zorder=3)
    ax.fill_between(chunk_sizes, 0, gpu_mem / 1024, alpha=0.10, color=C_MEM)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("(c) GPU Memory")
    ax.set_ylim(bottom=0)

    # Panel (d): Component proportion (stacked area)
    ax = axes[1, 1]
    frac_fwd = t_fwd / t_total * 100
    frac_rc = t_rc / t_total * 100
    frac_ntp = t_ntp / t_total * 100
    ax.stackplot(chunk_sizes, frac_fwd, frac_rc, frac_ntp,
                 labels=["Fwd entropy", "RC entropy", "Next-token"],
                 colors=[C_FWD, C_RC, C_NTP], alpha=0.8, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_bp))
    ax.set_xlabel("Chunk Size (bp)")
    ax.set_ylabel("% of Total Time")
    ax.set_title("(d) Time Proportion")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="center right", framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plots_dir, f"fig7_summary_panel_{timestamp}.pdf"))
    fig.savefig(os.path.join(plots_dir, f"fig7_summary_panel_{timestamp}.png"))
    plt.close(fig)
    print(f"  [PLOT] Fig 7: 2x2 summary panel")

    # ---- Figure 8: Full pipeline stage breakdown (if available) ----
    if report.full_pipeline and report.full_pipeline.status == "success":
        fp = report.full_pipeline
        file_io_total = (fp.tsv_write_seconds + fp.drops_file_seconds +
                         fp.rises_file_seconds + fp.window_summary_seconds +
                         fp.metadata_json_seconds)
        plot_total = fp.plot_suite_seconds + fp.method_comparison_seconds

        # 8a: Horizontal bar chart of pipeline stages
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        stage_names = [
            "Inference\n(scoring)",
            "Drop\ndetection",
            "Rise\ndetection",
            "Plot\nsuite",
            "Method\ncomparison",
            "TSV\nwrite",
            "Window\nsummary",
            "Other\nI/O",
        ]
        stage_times = np.array([
            fp.inference_seconds,
            fp.drop_detection_seconds,
            fp.rise_detection_seconds,
            fp.plot_suite_seconds,
            fp.method_comparison_seconds,
            fp.tsv_write_seconds,
            fp.window_summary_seconds,
            fp.drops_file_seconds + fp.rises_file_seconds + fp.metadata_json_seconds,
        ])
        stage_colors = [
            C_TOTAL,   # inference
            "#E53935", # drops
            "#FB8C00", # rises
            "#43A047", # plots
            "#00897B", # method comparison
            "#5E35B1", # TSV
            "#8E24AA", # window summary
            "#795548", # other I/O
        ]

        y_pos = np.arange(len(stage_names))
        bars = ax1.barh(y_pos, stage_times, color=stage_colors,
                        edgecolor="white", linewidth=0.5, zorder=3)
        for bar, t in zip(bars, stage_times):
            if t > 0:
                label = f"{t:.2f}s" if t >= 0.01 else f"{t*1000:.1f}ms"
                ax1.text(bar.get_width() + max(stage_times) * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         label, va="center", fontsize=8, fontweight="bold")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(stage_names, fontsize=8)
        ax1.set_xlabel("Wall-Clock Time (seconds)")
        ax1.set_title(f"Full Pipeline Stage Timing\n({fp.locus_length_bp:,} bp genome)")
        ax1.invert_yaxis()

        # 8b: Pie chart of time proportions
        pie_labels = ["Inference", "Drop det.", "Rise det.",
                      "Plotting", "File I/O"]
        pie_sizes = [
            fp.inference_seconds,
            fp.drop_detection_seconds,
            fp.rise_detection_seconds,
            plot_total,
            file_io_total,
        ]
        pie_colors = [C_TOTAL, "#E53935", "#FB8C00", "#43A047", "#5E35B1"]

        # Filter out near-zero slices
        filtered = [(l, s, c) for l, s, c in zip(pie_labels, pie_sizes, pie_colors) if s > 0.001]
        if filtered:
            p_labels, p_sizes, p_colors = zip(*filtered)
            wedges, texts, autotexts = ax2.pie(
                p_sizes, labels=p_labels, colors=p_colors,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
                startangle=90, textprops={"fontsize": 8},
            )
            for at in autotexts:
                at.set_fontsize(7)
                at.set_fontweight("bold")
            ax2.set_title(f"Time Distribution\n(Total: {fp.total_seconds:.1f}s)")

        fig.suptitle("Evo2-7B Full Analysis Pipeline: End-to-End Timing", fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(plots_dir, f"fig8_full_pipeline_{timestamp}.pdf"))
        fig.savefig(os.path.join(plots_dir, f"fig8_full_pipeline_{timestamp}.png"))
        plt.close(fig)
        print(f"  [PLOT] Fig 8: full pipeline breakdown")

        # 8c: Drop detection method comparison
        if fp.drop_detection_by_method:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

            methods = sorted(fp.drop_detection_by_method.keys())
            drop_times = [fp.drop_detection_by_method[m] for m in methods]
            drop_counts = [fp.n_drops_by_method.get(m, 0) for m in methods]
            rise_times = [fp.rise_detection_by_method.get(m, 0) for m in methods]
            rise_counts = [fp.n_rises_by_method.get(m, 0) for m in methods]

            x = np.arange(len(methods))
            bar_w = 0.35

            ax1.bar(x - bar_w/2, drop_times, bar_w, label="Drop detection",
                    color="#E53935", zorder=3)
            ax1.bar(x + bar_w/2, rise_times, bar_w, label="Rise detection",
                    color="#FB8C00", zorder=3)
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=30, ha="right")
            ax1.set_ylabel("Time (seconds)")
            ax1.set_title("Detection Runtime by Method")
            ax1.legend(framealpha=0.9, edgecolor="0.8")

            ax2.bar(x - bar_w/2, drop_counts, bar_w, label="Drops",
                    color="#E53935", zorder=3)
            ax2.bar(x + bar_w/2, rise_counts, bar_w, label="Rises",
                    color="#FB8C00", zorder=3)
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, rotation=30, ha="right")
            ax2.set_ylabel("Count")
            ax2.set_title("Detections Found by Method")
            ax2.legend(framealpha=0.9, edgecolor="0.8")

            fig.suptitle(f"Detection Method Analysis ({fp.locus_length_bp:,} bp genome)",
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(os.path.join(plots_dir, f"fig9_detection_methods_{timestamp}.pdf"))
            fig.savefig(os.path.join(plots_dir, f"fig9_detection_methods_{timestamp}.png"))
            plt.close(fig)
            print(f"  [PLOT] Fig 9: detection method comparison")


def generate_latex_table(report: BenchmarkReport, output_dir: str, timestamp: str):
    """
    Generate LaTeX tables suitable for ICML/NeurIPS papers.

    Produces two tables:
      Table 1: Single-chunk inference timing breakdown
      Table 2: Full-locus scoring comparison (sequential vs multi-GPU)
    """
    successful = [r for r in report.single_chunk_results if r.status == "success"]
    failed = [r for r in report.single_chunk_results if r.status != "success"]

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # ---- Table 1: Single-chunk inference ----
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Evo2-7B inference time breakdown by chunk size. "
                 r"Each chunk requires 4 forward passes: forward-strand entropy (Fwd), "
                 r"reverse-complement averaged entropy (RC, 2 passes), and "
                 r"next-token log-probabilities (NTP). "
                 r"Throughput measured as base pairs scored per second of wall-clock time.}")
    lines.append(r"\label{tab:chunk-timing}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Chunk Size & Fwd (s) & RC (s) & NTP (s) & Total (s) "
                 r"& Throughput & GPU Mem. \\")
    lines.append(r"(bp) & & & & & (bp/s) & (GB) \\")
    lines.append(r"\midrule")

    for r in successful:
        cs_str = f"{r.chunk_size_bp:,}"
        lines.append(
            f"{cs_str} & {r.entropy_fwd_seconds:.2f} & {r.entropy_rc_seconds:.2f} & "
            f"{r.next_token_seconds:.2f} & {r.total_inference_seconds:.2f} & "
            f"{r.throughput_bp_per_sec:,.0f} & {r.gpu_memory_peak_mb / 1024:.1f} \\\\"
        )

    if failed:
        lines.append(r"\midrule")
        for r in failed:
            cs_str = f"{r.chunk_size_bp:,}"
            status = r.status.upper()
            lines.append(
                f"{cs_str} & \\multicolumn{{6}}{{c}}{{\\textit{{{status}: {r.error[:50]}}}}}" r" \\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table1_path = os.path.join(tables_dir, f"table1_chunk_timing_{timestamp}.tex")
    with open(table1_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [TABLE] Table 1: {table1_path}")

    # ---- Table 2: Full-locus comparison ----
    seq_locus = [r for r in report.full_locus_results
                 if r.mode == "sequential" and r.status == "success"]
    mgpu_locus = [r for r in report.full_locus_results
                  if r.mode == "multigpu" and r.status == "success"]

    if seq_locus:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")

        if mgpu_locus:
            n_gpus_used = mgpu_locus[0].n_gpus_used
            locus_len = seq_locus[0].locus_length_bp
            lines.append(
                r"\caption{Full-locus scoring (" + f"{locus_len:,}" + r" bp): "
                r"sequential (1~GPU) vs.\ data-parallel (" + str(n_gpus_used) +
                r"~GPUs). Speedup defined as $T_{\text{seq}} / T_{\text{parallel}}$.}"
            )
            lines.append(r"\label{tab:multigpu}")
            lines.append(r"\small")
            lines.append(r"\begin{tabular}{r r r r r r}")
            lines.append(r"\toprule")
            lines.append(r"Chunk & \multicolumn{2}{c}{1 GPU (seq.)} "
                         r"& \multicolumn{2}{c}{" + str(n_gpus_used) +
                         r" GPUs (par.)} & Speedup \\")
            lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
            lines.append(r"Size (bp) & Time (s) & bp/s & Time (s) & bp/s & \\")
            lines.append(r"\midrule")

            seq_dict = {r.chunk_size_bp: r for r in seq_locus}
            mgpu_dict = {r.chunk_size_bp: r for r in mgpu_locus}

            for cs in sorted(set(seq_dict) | set(mgpu_dict)):
                cs_str = f"{cs:,}"
                if cs in seq_dict and cs in mgpu_dict:
                    s = seq_dict[cs]
                    m = mgpu_dict[cs]
                    speedup = s.total_seconds / m.total_seconds
                    lines.append(
                        f"{cs_str} & {s.total_seconds:.1f} & {s.throughput_bp_per_sec:,.0f} "
                        f"& {m.total_seconds:.1f} & {m.throughput_bp_per_sec:,.0f} "
                        f"& {speedup:.2f}$\\times$ \\\\"
                    )
                elif cs in seq_dict:
                    s = seq_dict[cs]
                    lines.append(
                        f"{cs_str} & {s.total_seconds:.1f} & {s.throughput_bp_per_sec:,.0f} "
                        r"& --- & --- & --- \\"
                    )
        else:
            locus_len = seq_locus[0].locus_length_bp
            lines.append(
                r"\caption{Full-locus sequential scoring (" + f"{locus_len:,}" +
                r" bp) by chunk size.}"
            )
            lines.append(r"\label{tab:locus-timing}")
            lines.append(r"\small")
            lines.append(r"\begin{tabular}{r r r r r r}")
            lines.append(r"\toprule")
            lines.append(r"Chunk Size & Chunks & Total (s) & Avg/Chunk (s) "
                         r"& Throughput (bp/s) & GPU (GB) \\")
            lines.append(r"\midrule")

            for r in seq_locus:
                cs_str = f"{r.chunk_size_bp:,}"
                lines.append(
                    f"{cs_str} & {r.n_chunks} & {r.total_seconds:.1f} & "
                    f"{r.avg_seconds_per_chunk:.2f} & {r.throughput_bp_per_sec:,.0f} & "
                    f"{r.gpu_memory_peak_mb / 1024:.1f} \\\\"
                )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        table2_path = os.path.join(tables_dir, f"table2_locus_comparison_{timestamp}.tex")
        with open(table2_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  [TABLE] Table 2: {table2_path}")

    # ---- Table 3: Pipeline stage summary ----
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{End-to-end pipeline timing for a single genome locus. "
                 r"Model loading is a one-time cost amortized across all loci.}")
    lines.append(r"\label{tab:pipeline-stages}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r}")
    lines.append(r"\toprule")
    lines.append(r"Pipeline Stage & Time (s) & \% of Total \\")
    lines.append(r"\midrule")

    stage_times = []
    if report.model_loading:
        stage_times.append(("Model loading (one-time)", report.model_loading.total_seconds))
    if successful:
        # Use default chunk size or median result
        default_r = ([r for r in successful if r.chunk_size_bp == 15_000] or
                     [successful[len(successful) // 2]])[0]
        stage_times.append((f"Inference ({_format_bp(default_r.chunk_size_bp)} chunk)",
                           default_r.total_inference_seconds))
    if report.file_writing:
        stage_times.append(("File I/O (TSV + drops + summary)", report.file_writing.total_seconds))

    total_all = sum(t for _, t in stage_times) if stage_times else 1.0
    for name, t in stage_times:
        pct = t / total_all * 100
        lines.append(f"{name} & {t:.2f} & {pct:.1f}\\% \\\\")

    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total_all:.2f}}} & 100.0\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table3_path = os.path.join(tables_dir, f"table3_pipeline_stages_{timestamp}.tex")
    with open(table3_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [TABLE] Table 3: {table3_path}")

    # ---- Table 4: Full pipeline breakdown (if available) ----
    if report.full_pipeline and report.full_pipeline.status == "success":
        fp = report.full_pipeline
        file_io_total = (fp.tsv_write_seconds + fp.drops_file_seconds +
                         fp.rises_file_seconds + fp.window_summary_seconds +
                         fp.metadata_json_seconds)
        plot_total = fp.plot_suite_seconds + fp.method_comparison_seconds

        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Full pipeline timing for whole-genome analysis "
                     f"({fp.locus_length_bp:,}" r" bp, "
                     f"chunk size {fp.chunk_size_bp:,}" r" bp). "
                     r"Inference dominates runtime; detection and I/O are negligible.}")
        lines.append(r"\label{tab:full-pipeline}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{l r r}")
        lines.append(r"\toprule")
        lines.append(r"Pipeline Stage & Time (s) & \% of Total \\")
        lines.append(r"\midrule")

        stages = [
            ("Inference (scoring)", fp.inference_seconds),
            ("Unit conversion", fp.unit_conversion_seconds),
            ("Drop detection (6 methods)", fp.drop_detection_seconds),
            ("Rise detection (6 methods)", fp.rise_detection_seconds),
            ("Plot suite", fp.plot_suite_seconds),
            ("Method comparison report", fp.method_comparison_seconds),
            ("TSV file write", fp.tsv_write_seconds),
            ("Window summary", fp.window_summary_seconds),
            ("Other file I/O", fp.drops_file_seconds + fp.rises_file_seconds + fp.metadata_json_seconds),
        ]
        total = fp.total_seconds
        for name, t in stages:
            pct = t / total * 100 if total > 0 else 0
            lines.append(f"{name} & {t:.2f} & {pct:.1f}\\% \\\\")

        lines.append(r"\midrule")
        lines.append(f"\\textbf{{Total (end-to-end)}} & \\textbf{{{total:.2f}}} & 100.0\\% \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        table4_path = os.path.join(tables_dir, f"table4_full_pipeline_{timestamp}.tex")
        with open(table4_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  [TABLE] Table 4: {table4_path}")

        # ---- Table 5: Detection method breakdown ----
        if fp.drop_detection_by_method:
            lines = []
            lines.append(r"\begin{table}[t]")
            lines.append(r"\centering")
            lines.append(r"\caption{Detection method runtime and yield "
                         f"({fp.locus_length_bp:,}" r" bp genome). "
                         r"All methods run on CPU after GPU inference completes.}")
            lines.append(r"\label{tab:detection-methods}")
            lines.append(r"\small")
            lines.append(r"\begin{tabular}{l r r r r}")
            lines.append(r"\toprule")
            lines.append(r"Method & Drop Time (s) & \# Drops & Rise Time (s) & \# Rises \\")
            lines.append(r"\midrule")

            methods = sorted(fp.drop_detection_by_method.keys())
            for m in methods:
                dt = fp.drop_detection_by_method.get(m, 0)
                nd = fp.n_drops_by_method.get(m, 0)
                rt = fp.rise_detection_by_method.get(m, 0)
                nr = fp.n_rises_by_method.get(m, 0)
                lines.append(f"{m} & {dt:.3f} & {nd:,} & {rt:.3f} & {nr:,} \\\\")

            lines.append(r"\midrule")
            total_dt = sum(fp.drop_detection_by_method.values())
            total_rt = sum(fp.rise_detection_by_method.values())
            lines.append(f"\\textbf{{Total}} & \\textbf{{{total_dt:.3f}}} & "
                         f"& \\textbf{{{total_rt:.3f}}} & \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")

            table5_path = os.path.join(tables_dir, f"table5_detection_methods_{timestamp}.tex")
            with open(table5_path, "w") as f:
                f.write("\n".join(lines))
            print(f"  [TABLE] Table 5: {table5_path}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_benchmarks(
    genbank_path: str,
    chunk_sizes: List[int],
    locus_length: int,
    output_dir: str,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    benchmark_multigpu: bool = False,
    whole_genome: bool = False,
    pipeline_chunk_size: int = 15_000,
    model_name: str = "evo2_7b",
    chrom: str = None,
    auto_probe_chunk: bool = False,
):
    """Run the complete benchmark suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        cuda_available=torch.cuda.is_available(),
        n_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        pytorch_version=torch.__version__,
        chunk_overlap=chunk_overlap,
    )

    print(f"\n{'#'*70}")
    print(f"  GENOME SCORING PIPELINE - TIMING BENCHMARK")
    print(f"{'#'*70}")
    print(f"  Time:       {report.timestamp}")
    print(f"  CUDA:       {report.cuda_available}")
    print(f"  GPUs:       {report.n_gpus}")
    if report.cuda_available:
        for i in range(report.n_gpus):
            print(f"    GPU {i}:    {torch.cuda.get_device_name(i)}")
    print(f"  PyTorch:    {report.pytorch_version}")
    print(f"  Overlap:    {chunk_overlap}")
    print(f"  Chunk sizes: {chunk_sizes}")
    print(f"  Locus len:  {locus_length:,}")
    print(f"  Whole genome: {whole_genome}")
    print()

    # --- Load genome data ---
    genome_seq, genome_id = load_genome_from_genbank(genbank_path, chrom=chrom)
    report.genome_source = f"{genome_id} ({len(genome_seq):,} bp)"
    report.genome_length_bp = len(genome_seq)
    report.device = "cuda" if report.cuda_available else "cpu"

    # If whole_genome mode, set locus_length to full genome
    if whole_genome:
        locus_length = len(genome_seq)
        print(f"[BENCH] Whole-genome mode: locus_length set to {locus_length:,} bp")

    # --- Build per-run output folder ---
    # Format: <output_dir>/<genome_id>_<mode>_<length>_<timestamp>/
    # e.g.  benchmark_results/NC_012920.1_whole_16Kbp_20260224_164603/
    bp = len(genome_seq)
    if bp >= 1_000_000_000:
        bp_label = f"{bp / 1_000_000_000:.1f}Gbp"
    elif bp >= 1_000_000:
        bp_label = f"{bp / 1_000_000:.0f}Mbp"
    elif bp >= 1_000:
        bp_label = f"{bp / 1_000:.0f}Kbp"
    else:
        bp_label = f"{bp}bp"

    mode_label = "whole" if whole_genome else "locus"
    run_folder = f"{genome_id}_{mode_label}_{bp_label}_{timestamp}"
    output_dir = os.path.join(output_dir, run_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[BENCH] Output directory: {output_dir}")

    # --- T1: Model loading ---
    model_result, evo2_model, ACGT_IDS, device = benchmark_model_loading(model_name)
    report.model_loading = model_result
    _save_partial(report, output_dir, timestamp)

    if not model_result.success or evo2_model is None:
        print("\nModel loading failed. Cannot proceed with inference benchmarks.")
        return report

    report.device = device

    # --- Optional: Auto-probe max chunk size before T2 ---
    if auto_probe_chunk:
        print(f"\n[BENCH] Auto-probing max chunk size (starting from {max(chunk_sizes):,} bp)...")
        try:
            max_viable = _find_max_chunk_size(
                evo2_model, ACGT_IDS, device,
                target_size=max(chunk_sizes),
            )
            before = len(chunk_sizes)
            chunk_sizes = [cs for cs in chunk_sizes if cs <= max_viable]
            print(f"[BENCH] Max viable chunk size: {max_viable:,} bp")
            print(f"[BENCH] Filtered: {before} -> {len(chunk_sizes)} chunk sizes")
        except Exception as e:
            print(f"[BENCH] Auto-probe failed: {e}. Proceeding with all chunk sizes.")

    # --- T2: Single-chunk benchmarks ---
    report.single_chunk_results = benchmark_single_chunk_tests(
        genome_seq, chunk_sizes, evo2_model, ACGT_IDS, device
    )
    _save_partial(report, output_dir, timestamp)

    # Collect chunk sizes that OOM'd in T2 so we can skip them in T3/T5
    oom_sizes = {r.chunk_size_bp for r in report.single_chunk_results if r.status == "oom"}
    if oom_sizes:
        print(f"\n[BENCH] Chunk sizes that OOM'd in T2: {sorted(oom_sizes)}")
        print(f"        These will be auto-skipped in T3/T5.")

    # --- T3: Full locus benchmarks (sequential) ---
    # Use chunk sizes that are <= locus_length for full-locus tests
    locus_chunk_sizes = [cs for cs in FULL_LOCUS_CHUNK_SIZES if cs <= locus_length]
    if not locus_chunk_sizes:
        locus_chunk_sizes = [min(chunk_sizes)]

    # Skip chunk sizes that already OOM'd in T2
    if oom_sizes:
        skipped = [cs for cs in locus_chunk_sizes if cs in oom_sizes]
        locus_chunk_sizes = [cs for cs in locus_chunk_sizes if cs not in oom_sizes]
        if skipped:
            print(f"[BENCH] T3: skipping {skipped} (OOM in T2)")
        if not locus_chunk_sizes:
            print(f"[BENCH] T3: no viable chunk sizes remain after OOM filtering")
            locus_chunk_sizes = []

    if locus_chunk_sizes:
        report.full_locus_results = benchmark_full_locus(
            genome_seq, locus_length, locus_chunk_sizes,
            evo2_model, ACGT_IDS, device, chunk_overlap
        )
    else:
        report.full_locus_results = []
    _save_partial(report, output_dir, timestamp)

    # --- T3b: Full locus benchmarks (multi-GPU) ---
    if benchmark_multigpu:
        multigpu_results = benchmark_full_locus_multigpu(
            genome_seq, locus_length, locus_chunk_sizes, chunk_overlap
        )
        report.full_locus_results.extend(multigpu_results)
        _save_partial(report, output_dir, timestamp)

    # --- T4: File writing benchmarks (synthetic) ---
    write_bench_dir = os.path.join(output_dir, "bench_tmp")
    os.makedirs(write_bench_dir, exist_ok=True)
    report.file_writing = benchmark_file_writing(locus_length, write_bench_dir)
    # Clean up temp dir
    try:
        os.rmdir(write_bench_dir)
    except OSError:
        pass
    _save_partial(report, output_dir, timestamp)

    # --- T5: Full pipeline benchmark (whole genome with detection + plots + I/O) ---
    if whole_genome:
        # Check if the pipeline chunk size OOM'd in T2
        if pipeline_chunk_size in oom_sizes:
            # Find largest successful chunk size from T2
            successful_sizes = sorted(
                [r.chunk_size_bp for r in report.single_chunk_results if r.status == "success"]
            )
            if successful_sizes:
                fallback = successful_sizes[-1]
                print(f"\n[BENCH] T5: pipeline_chunk_size={pipeline_chunk_size} OOM'd in T2.")
                print(f"        Falling back to largest successful size: {fallback}")
                pipeline_chunk_size = fallback
            else:
                print(f"\n[BENCH] T5: skipped — no chunk sizes succeeded in T2.")
                whole_genome = False

    if whole_genome:
        print(f"\n[BENCH] Running full analysis pipeline on entire genome ({len(genome_seq):,} bp)...")
        try:
            report.full_pipeline = benchmark_full_pipeline(
                genome_seq=genome_seq,
                genome_id=genome_id,
                evo2_model=evo2_model,
                ACGT_IDS=ACGT_IDS,
                device=device,
                output_dir=output_dir,
                chunk_size=pipeline_chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            print(f"[WARN] Full pipeline benchmark failed: {e}")
            traceback.print_exc()
            report.full_pipeline = FullPipelineResult(
                locus_length_bp=len(genome_seq),
                chunk_size_bp=pipeline_chunk_size,
                status="error",
                error=f"{type(e).__name__}: {e}",
            )
        _save_partial(report, output_dir, timestamp)

    # --- Save results ---
    json_path = os.path.join(output_dir, f"pipeline_benchmark_{timestamp}.json")
    summary_path = os.path.join(output_dir, f"pipeline_benchmark_{timestamp}_summary.txt")
    save_json_report(report, json_path)
    generate_summary_text(report, summary_path)

    # --- Generate publication-quality plots and LaTeX tables ---
    print(f"\n{'='*70}")
    print(f"GENERATING PLOTS & TABLES")
    print(f"{'='*70}")
    try:
        generate_plots(report, output_dir, timestamp)
    except Exception as e:
        print(f"[WARN] Plot generation failed: {e}")
        traceback.print_exc()

    try:
        generate_latex_table(report, output_dir, timestamp)
    except Exception as e:
        print(f"[WARN] LaTeX table generation failed: {e}")
        traceback.print_exc()

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end timing benchmark for the genome scoring pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --mode quick

  # Full benchmark with chunk size sweep
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --mode full

  # WHOLE GENOME: full pipeline with drop/rise detection, plots, and file I/O
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --whole_genome

  # Whole genome with custom pipeline chunk size
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --whole_genome \\
      --pipeline_chunk_size 50000

  # Custom chunk sizes for sweep
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff \\
      --chunk_sizes 5000 15000 100000 500000 1000000 1500000

  # With multi-GPU comparison
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --mode full --benchmark_multigpu

  # Everything: whole genome pipeline + chunk sweep + multi-GPU
  python tools/benchmark_pipeline_timing.py --genbank genomic.gbff --whole_genome \\
      --mode full --benchmark_multigpu
        """
    )

    ap.add_argument("--genbank", "--fasta", type=str, dest="genbank",
                    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "genomic.gbff"),
                    help="Path to genome file (GenBank .gbff/.gb or FASTA .fna/.fa/.fasta). "
                         "Format auto-detected from extension. Default: genomic.gbff")

    ap.add_argument("--chrom", type=str, default=None,
                    help="Specific chromosome/contig ID to select (e.g., 'NC_000001.11' for "
                         "human chr1). If not specified, picks the largest record. "
                         "For human GRCh38: NC_000001.11 (chr1, 248M bp), "
                         "NC_000022.11 (chr22, 50M bp, good for testing).")

    ap.add_argument("--mode", choices=["quick", "full"], default="full",
                    help="Benchmark mode: 'quick' (4 chunk sizes) or 'full' (13 chunk sizes)")

    ap.add_argument("--chunk_sizes", nargs="+", type=int, default=None,
                    help="Custom chunk sizes to test (overrides --mode)")

    ap.add_argument("--locus_length", type=int, default=DEFAULT_LOCUS_LENGTH,
                    help=f"Length of test locus for full-pipeline scoring (default: {DEFAULT_LOCUS_LENGTH:,})")

    ap.add_argument("--whole_genome", action="store_true",
                    help="Run the FULL analysis pipeline across the entire genome chromosome. "
                         "This replicates what genome_scoring_jan26_drops.py does end-to-end: "
                         "inference + drop/rise detection (all 6 methods) + plot suite + "
                         "method comparison + all file I/O. Times each stage separately.")

    ap.add_argument("--pipeline_chunk_size", type=int, default=15_000,
                    help="Chunk size for the full pipeline benchmark (default: 15000). "
                         "Only used with --whole_genome.")

    ap.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                    help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})")

    ap.add_argument("--output_dir", type=str, default="results/_benchmarks",
                    help="Output directory for results (default: results/_benchmarks)")

    ap.add_argument("--benchmark_multigpu", action="store_true",
                    help="Also benchmark multi-GPU scoring (requires multiple GPUs and "
                         "score_locus_aligned_overlap_multigpu)")

    ap.add_argument("--auto_probe_chunk", action="store_true",
                    help="Auto-probe max chunk size before T2 to skip sizes that will OOM. "
                         "Runs a test forward pass at decreasing sizes to find the largest "
                         "that fits in GPU memory.")

    ap.add_argument("--model", type=str, default="evo2_7b",
                    help="Model name (default: evo2_7b)")

    args = ap.parse_args()

    # Determine chunk sizes
    if args.chunk_sizes is not None:
        chunk_sizes = args.chunk_sizes
    elif args.mode == "quick":
        chunk_sizes = CHUNK_SIZES_QUICK
    else:
        chunk_sizes = CHUNK_SIZES_FULL

    run_benchmarks(
        genbank_path=args.genbank,
        chunk_sizes=chunk_sizes,
        locus_length=args.locus_length,
        output_dir=args.output_dir,
        chunk_overlap=args.chunk_overlap,
        benchmark_multigpu=args.benchmark_multigpu,
        whole_genome=args.whole_genome,
        pipeline_chunk_size=args.pipeline_chunk_size,
        model_name=args.model,
        chrom=args.chrom,
        auto_probe_chunk=args.auto_probe_chunk,
    )


if __name__ == "__main__":
    main()
