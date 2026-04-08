#!/usr/bin/env python3
"""
scan_sae_global_stats.py — Genome-wide SAE feature min/max statistics

Scans an entire chromosome through the SAE in chunks, tracking per-feature
global min and max activations. No per-position storage — just two (32768,)
arrays updated per chunk, so memory usage is minimal.

Run per-chromosome on GPU, then use aggregate_genome_sae_stats.py (or the
built-in --aggregate mode) to merge across chromosomes.

Usage:
    # Single chromosome
    python tools/scan_sae_global_stats.py \\
        --fasta /path/to/genome.fna \\
        --chrom chr22 \\
        --output_dir results/

    # All human chromosomes (submit via SLURM — see run_sae_global_stats.sh)
    python tools/scan_sae_global_stats.py \\
        --fasta /path/to/genome.fna \\
        --chrom chr1 \\
        --output_dir results/

    # Aggregate previously computed per-chromosome stats (CPU only, no GPU)
    python tools/scan_sae_global_stats.py \\
        --aggregate \\
        --results_dir results/ \\
        --chroms chr1 chr2 chr22 \\
        [--all_human]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, find_latest_completed, write_completed, write_source

CHUNK_SIZE = 16384  # 2x original; safe on 44GB GPUs, ~2x fewer forward passes
OVERLAP = 256
BATCH_SIZE = 2  # Number of chunks per batched forward pass; falls back to 1 on OOM
N_FEATURES = 32768

ALL_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]

logger = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


class _EarlyExitException(Exception):
    """Abort forward pass after capturing target layer activations."""
    pass


def _get_feature_stats_batch(model, sae, sequences):
    """Run SAE on a batch of sequences, return per-chunk min/max on GPU.

    Uses early exit after layer 26 to skip blocks 27-47 and the output head,
    saving ~40% of forward pass compute. Works for any batch size (including 1).

    Returns:
        List of (chunk_min, chunk_max) numpy arrays, shape (N_FEATURES,) each
    """
    import torch
    from sae_utils import SAE_LAYER_NAME

    layer_name = SAE_LAYER_NAME

    # Tokenize all sequences
    all_toks = []
    for seq in sequences:
        toks = model.tokenizer.tokenize(seq)
        all_toks.append(torch.tensor(toks, dtype=torch.long))

    # Pad to same length for batching
    max_len = max(t.shape[0] for t in all_toks)
    batch = torch.zeros(len(all_toks), max_len, dtype=torch.long, device=model.device)
    for j, t in enumerate(all_toks):
        batch[j, :t.shape[0]] = t.to(model.device)

    # Compile SAE encoder once
    if not getattr(sae, '_encode_compiled', False):
        try:
            sae.encode = torch.compile(sae.encode)
            sae._encode_compiled = True
        except Exception:
            sae._encode_compiled = True

    sae_device = next(iter(sae.parameters())).device

    # Early exit: hook on layer 26 captures activations and aborts forward pass
    # so blocks 27-47 and the LM head never execute
    captured = {}
    def _exit_hook(module, input, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured['acts'] = acts.detach()
        raise _EarlyExitException()

    target_module = model.scope._module_dict[layer_name]
    handle = target_module.register_forward_hook(_exit_hook)

    results = []
    try:
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                early_exited = False
                try:
                    model.model(batch)
                except _EarlyExitException:
                    early_exited = True

                if not early_exited and not getattr(model, '_early_exit_warned', False):
                    logger.warning("Early exit not effective — running full forward pass")
                    model._early_exit_warned = True
                elif early_exited and not getattr(model, '_early_exit_logged', False):
                    logger.info("Early exit after layer 26 active — skipping blocks 27-47")
                    model._early_exit_logged = True

                layer_acts = captured['acts']  # (batch, seq_len, d_hidden)

                for j in range(len(sequences)):
                    seq_len = len(all_toks[j])
                    act_j = layer_acts[j, :seq_len, :].to(sae_device)
                    features = sae.encode(act_j)  # (seq_len, N_FEATURES)

                    # Min/max on GPU before transfer
                    chunk_max = features.max(dim=0).values.float().cpu().numpy().astype(np.float64)
                    chunk_min = features.min(dim=0).values.float().cpu().numpy().astype(np.float64)
                    # Per-nucleotide mean/var on GPU before transfer
                    chunk_mean = features.mean(dim=0).float().cpu().numpy().astype(np.float64)
                    chunk_var = features.var(dim=0, unbiased=False).float().cpu().numpy().astype(np.float64)
                    chunk_n = features.shape[0]  # number of positions (tokens)
                    results.append((chunk_min, chunk_max, chunk_mean, chunk_var, chunk_n))

                del layer_acts
    finally:
        handle.remove()

    return results


def scan_chromosome_stats(sequence, model, sae, chunk_size=CHUNK_SIZE,
                          overlap=OVERLAP, batch_size=BATCH_SIZE,
                          checkpoint_dir=None, checkpoint_interval=200):
    """Scan a chromosome in batched chunks, tracking per-feature global min and max.

    Only keeps two (32768,) arrays — no per-position or per-chunk storage.
    Processes multiple chunks per forward pass for better GPU utilization.
    Supports checkpointing for resumption after preemption.

    Args:
        sequence: DNA string
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        chunk_size: bp per chunk
        overlap: bp overlap between chunks
        batch_size: number of chunks per batched forward pass
        checkpoint_dir: directory to save/load checkpoints (None = no checkpointing)
        checkpoint_interval: save checkpoint every N chunks (default 200)

    Returns:
        dict with keys: global_min, global_max, global_mean, global_std,
                        n_chunks, genome_length, n_nonzero_chunks
    """
    genome_len = len(sequence)
    stride = chunk_size - overlap
    n_chunks = max(1, (genome_len - overlap + stride - 1) // stride)

    # Running stats arrays
    global_min = np.full(N_FEATURES, np.inf, dtype=np.float64)
    global_max = np.full(N_FEATURES, -np.inf, dtype=np.float64)

    # For online mean/variance of chunk-max values (Welford's algorithm)
    running_mean = np.zeros(N_FEATURES, dtype=np.float64)
    running_m2 = np.zeros(N_FEATURES, dtype=np.float64)
    n_nonzero_chunks = np.zeros(N_FEATURES, dtype=np.int64)
    n_processed = 0
    start_chunk = 0

    # Per-nucleotide stats: weighted Welford merge across all positions
    nuc_mean = np.zeros(N_FEATURES, dtype=np.float64)
    nuc_m2 = np.zeros(N_FEATURES, dtype=np.float64)  # sum of (var_i * n_i) + cross-terms
    total_positions = 0

    # Try to resume from checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "_scan_checkpoint.npz") if checkpoint_dir else None
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = np.load(ckpt_path)
        global_min = ckpt["global_min"].astype(np.float64)
        global_max = ckpt["global_max"].astype(np.float64)
        running_mean = ckpt["running_mean"].astype(np.float64)
        running_m2 = ckpt["running_m2"].astype(np.float64)
        n_nonzero_chunks = ckpt["n_nonzero_chunks"].astype(np.int64)
        n_processed = int(ckpt["n_processed"])
        start_chunk = n_processed  # chunk index equals n_processed (0-based)
        # Restore per-nucleotide stats if present (backward compat with old checkpoints)
        if "nuc_mean" in ckpt:
            nuc_mean = ckpt["nuc_mean"].astype(np.float64)
            nuc_m2 = ckpt["nuc_m2"].astype(np.float64)
            total_positions = int(ckpt["total_positions"])
        logger.info(f"RESUMED from checkpoint: {n_processed}/{n_chunks} chunks already done")

    logger.info(f"Scanning {genome_len:,} bp in {n_chunks} chunks "
                f"({chunk_size} bp, {overlap} bp overlap, batch_size={batch_size})")

    t0 = time.time()
    i = start_chunk
    while i < n_chunks:
        # Collect a batch of chunk sequences
        batch_seqs = []
        for b in range(batch_size):
            ci = i + b
            if ci >= n_chunks:
                break
            chunk_start = ci * stride
            chunk_end = min(chunk_start + chunk_size, genome_len)
            chunk_seq = sequence[chunk_start:chunk_end]
            if len(chunk_seq) >= 10:
                batch_seqs.append(chunk_seq)

        if not batch_seqs:
            i += batch_size
            continue

        # Process batch — try batched, fall back to sequential on OOM
        try:
            stats_list = _get_feature_stats_batch(model, sae, batch_seqs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                if batch_size > 1:
                    logger.warning(f"OOM on batch of {len(batch_seqs)}, "
                                   f"reducing batch_size to 1 for remaining chunks")
                    batch_size = 1
                    # Retry sequentially (still uses early exit)
                    stats_list = []
                    for seq in batch_seqs:
                        stats_list.extend(
                            _get_feature_stats_batch(model, sae, [seq]))
                else:
                    raise  # Single chunk OOM — chunk_size is too large for this GPU
            else:
                raise

        for chunk_min, chunk_max, chunk_mean, chunk_var, chunk_n in stats_list:
            np.minimum(global_min, chunk_min, out=global_min)
            np.maximum(global_max, chunk_max, out=global_max)

            # Chunk-max Welford (existing)
            n_processed += 1
            delta = chunk_max - running_mean
            running_mean += delta / n_processed
            delta2 = chunk_max - running_mean
            running_m2 += delta * delta2
            n_nonzero_chunks += (chunk_max > 0).astype(np.int64)

            # Per-nucleotide weighted Welford merge
            # Merges (nuc_mean, nuc_var, total_positions) with (chunk_mean, chunk_var, chunk_n)
            if total_positions == 0:
                nuc_mean = chunk_mean.copy()
                nuc_m2 = chunk_var * chunk_n
                total_positions = chunk_n
            else:
                new_total = total_positions + chunk_n
                delta_nuc = chunk_mean - nuc_mean
                nuc_mean += delta_nuc * (chunk_n / new_total)
                nuc_m2 += (chunk_var * chunk_n) + (delta_nuc ** 2) * (total_positions * chunk_n / new_total)
                total_positions = new_total

        i += batch_size

        if n_processed > 0 and n_processed % 50 == 0:
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta = (n_chunks - n_processed) / rate if rate > 0 else 0
            n_active = np.sum(global_max > 0)
            logger.info(f"  Chunk {n_processed}/{n_chunks} ({elapsed:.0f}s elapsed, "
                        f"~{eta:.0f}s remaining, {n_active} features active so far)")

        # Save checkpoint periodically
        if ckpt_path and n_processed > 0 and n_processed % checkpoint_interval == 0:
            np.savez(ckpt_path,
                     global_min=global_min, global_max=global_max,
                     running_mean=running_mean, running_m2=running_m2,
                     n_nonzero_chunks=n_nonzero_chunks,
                     n_processed=np.array(n_processed),
                     nuc_mean=nuc_mean, nuc_m2=nuc_m2,
                     total_positions=np.array(total_positions))
            logger.info(f"  Checkpoint saved at chunk {n_processed}/{n_chunks}")

    elapsed = time.time() - t0
    logger.info(f"Scan complete in {elapsed:.1f}s ({n_processed} chunks processed)")

    # Finalize chunk-max variance
    if n_processed > 1:
        running_var = running_m2 / (n_processed - 1)
    else:
        running_var = np.zeros(N_FEATURES, dtype=np.float64)

    # Finalize per-nucleotide variance
    if total_positions > 0:
        nuc_var = nuc_m2 / total_positions
        nuc_std = np.sqrt(nuc_var)
    else:
        nuc_std = np.zeros(N_FEATURES, dtype=np.float64)

    # Replace inf with 0 for features never seen
    global_min[global_min == np.inf] = 0.0
    global_max[global_max == -np.inf] = 0.0

    n_active = int(np.sum(global_max > 0))
    logger.info(f"Features with any activation: {n_active}/{N_FEATURES}")
    logger.info(f"Global max range: [{global_max.min():.4f}, {global_max.max():.4f}]")
    logger.info(f"Per-nucleotide stats: {total_positions:,} positions processed")
    logger.info(f"  nuc_mean range: [{nuc_mean.min():.6f}, {nuc_mean.max():.6f}]")
    logger.info(f"  nuc_std range:  [{nuc_std.min():.6f}, {nuc_std.max():.6f}]")

    return {
        "global_min": global_min.astype(np.float32),
        "global_max": global_max.astype(np.float32),
        "chunk_max_mean": running_mean.astype(np.float32),
        "chunk_max_std": np.sqrt(running_var).astype(np.float32),
        "n_nonzero_chunks": n_nonzero_chunks,
        "n_chunks": n_processed,
        "genome_length": genome_len,
        "nuc_mean": nuc_mean.astype(np.float32),
        "nuc_std": nuc_std.astype(np.float32),
        "total_positions": total_positions,
    }


def merge_stats(mean_a, var_a, n_a, mean_b, var_b, n_b):
    """Merge two sets of sufficient statistics into one.

    Given (mean, variance, count) for two disjoint populations A and B,
    compute the combined (mean, variance, count) as if all observations
    had been processed in a single Welford pass.

    All array inputs must be the same shape; scalars are broadcast.

    Returns:
        (combined_mean, combined_var, combined_n) — same shape as inputs.
    """
    n_ab = n_a + n_b
    delta = mean_b - mean_a
    combined_mean = mean_a + delta * (n_b / n_ab)
    combined_m2 = (var_a * n_a) + (var_b * n_b) + (delta ** 2) * (n_a * n_b / n_ab)
    combined_var = combined_m2 / n_ab
    return combined_mean, combined_var, n_ab


def aggregate_chromosome_stats_corrected(results_dir, chroms):
    """Merge per-chromosome chunk-level stats into genome-wide mean/std.

    Uses weighted Welford-style merging so the result is mathematically
    equivalent to computing mean/std over ALL chunks across ALL chromosomes
    in a single streaming pass.

    Each chromosome's global_sae_stats.npz provides:
        chunk_max_mean  — per-feature mean of chunk-max values (shape 32768)
        chunk_max_std   — per-feature std of chunk-max values  (shape 32768)
        n_chunks        — number of chunks in that chromosome  (scalar)
        nuc_mean        — per-feature mean across all nucleotide positions (shape 32768)
        nuc_std         — per-feature std across all nucleotide positions  (shape 32768)
        total_positions — number of nucleotide positions in that chromosome (scalar)
    """
    genome_min = np.full(N_FEATURES, np.inf, dtype=np.float64)
    genome_max = np.full(N_FEATURES, -np.inf, dtype=np.float64)

    # Running sufficient statistics across all chunks (chunk-max)
    global_mean = np.zeros(N_FEATURES, dtype=np.float64)
    global_m2 = np.zeros(N_FEATURES, dtype=np.float64)
    total_n = 0

    # Running sufficient statistics across all nucleotide positions
    genome_nuc_mean = np.zeros(N_FEATURES, dtype=np.float64)
    genome_nuc_m2 = np.zeros(N_FEATURES, dtype=np.float64)
    genome_total_positions = 0
    has_nuc_stats = False

    total_bp = 0
    chrom_summaries = []
    found_chroms = []
    missing_chroms = []

    for chrom in chroms:
        run_dir = find_latest_completed(results_dir, chrom, "sae_global_stats")
        if run_dir is None:
            logger.warning(f"  {chrom}: no completed sae_global_stats run")
            missing_chroms.append(chrom)
            continue

        stats_path = os.path.join(run_dir, "data", "global_sae_stats.npz")
        if not os.path.isfile(stats_path):
            logger.warning(f"  {chrom}: global_sae_stats.npz not found in {run_dir}")
            missing_chroms.append(chrom)
            continue

        data = np.load(stats_path)
        chrom_min = data["global_min"].astype(np.float64)
        chrom_max = data["global_max"].astype(np.float64)
        chrom_mean = data["chunk_max_mean"].astype(np.float64)
        chrom_std = data["chunk_max_std"].astype(np.float64)
        n_chunks = int(data["n_chunks"])
        genome_length = int(data["genome_length"])

        np.minimum(genome_min, chrom_min, out=genome_min)
        np.maximum(genome_max, chrom_max, out=genome_max)

        # Weighted merge of chunk-max stats into running totals
        chrom_var = chrom_std ** 2
        if total_n == 0:
            global_mean = chrom_mean.copy()
            global_m2 = chrom_var * n_chunks
            total_n = n_chunks
        else:
            delta = chrom_mean - global_mean
            new_n = total_n + n_chunks
            global_mean += delta * (n_chunks / new_n)
            global_m2 += (chrom_var * n_chunks) + (delta ** 2) * (total_n * n_chunks / new_n)
            total_n = new_n

        # Weighted merge of per-nucleotide stats (if available)
        if "nuc_mean" in data and "nuc_std" in data and "total_positions" in data:
            has_nuc_stats = True
            chrom_nuc_mean = data["nuc_mean"].astype(np.float64)
            chrom_nuc_std = data["nuc_std"].astype(np.float64)
            chrom_nuc_var = chrom_nuc_std ** 2
            chrom_positions = int(data["total_positions"])

            if genome_total_positions == 0:
                genome_nuc_mean = chrom_nuc_mean.copy()
                genome_nuc_m2 = chrom_nuc_var * chrom_positions
                genome_total_positions = chrom_positions
            else:
                delta_nuc = chrom_nuc_mean - genome_nuc_mean
                new_total = genome_total_positions + chrom_positions
                genome_nuc_mean += delta_nuc * (chrom_positions / new_total)
                genome_nuc_m2 += (chrom_nuc_var * chrom_positions) + \
                    (delta_nuc ** 2) * (genome_total_positions * chrom_positions / new_total)
                genome_total_positions = new_total

        total_bp += genome_length
        found_chroms.append(chrom)
        chrom_summaries.append({
            "chrom": chrom,
            "n_chunks": n_chunks,
            "genome_length": genome_length,
            "run_dir": run_dir,
        })
        logger.info(f"  {chrom}: {genome_length:,} bp, {n_chunks:,} chunks")

    genome_min[genome_min == np.inf] = 0.0
    genome_max[genome_max == -np.inf] = 0.0

    if total_n > 0:
        global_var = global_m2 / total_n
        global_std = np.sqrt(global_var).astype(np.float32)
    else:
        global_std = np.zeros(N_FEATURES, dtype=np.float32)

    # Finalize per-nucleotide stats
    if genome_total_positions > 0:
        genome_nuc_var = genome_nuc_m2 / genome_total_positions
        genome_nuc_std = np.sqrt(genome_nuc_var).astype(np.float32)
    else:
        genome_nuc_std = np.zeros(N_FEATURES, dtype=np.float32)

    logger.info(f"\nAggregated {len(found_chroms)} chromosomes: "
                f"{total_bp:,} bp, {total_n:,} total chunks")
    n_active = int(np.sum(genome_max > 0))
    logger.info(f"Features with any activation genome-wide: {n_active}/{N_FEATURES}")
    logger.info(f"Expected chunks (~genome/8192): ~{total_bp // 8192:,}")
    logger.info(f"Chunk-max stats:")
    logger.info(f"  Global mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
    logger.info(f"  Global std range:  [{global_std.min():.4f}, {global_std.max():.4f}]")
    if has_nuc_stats:
        logger.info(f"Per-nucleotide stats ({genome_total_positions:,} positions):")
        logger.info(f"  nuc_mean range: [{genome_nuc_mean.min():.6f}, {genome_nuc_mean.max():.6f}]")
        logger.info(f"  nuc_std range:  [{genome_nuc_std.min():.6f}, {genome_nuc_std.max():.6f}]")
    else:
        logger.warning("No per-nucleotide stats found — re-run scans with updated script")

    result = {
        "genome_min": genome_min.astype(np.float32),
        "genome_max": genome_max.astype(np.float32),
        "global_mean": global_mean.astype(np.float32),
        "global_std": global_std,
        "total_chunks": total_n,
        "total_bp": total_bp,
        "found_chroms": found_chroms,
        "missing_chroms": missing_chroms,
        "chrom_summaries": chrom_summaries,
    }

    if has_nuc_stats:
        result["nuc_mean"] = genome_nuc_mean.astype(np.float32)
        result["nuc_std"] = genome_nuc_std
        result["total_positions"] = genome_total_positions

    return result


def aggregate_chromosome_stats(results_dir, chroms):
    """LEGACY: Merge per-chromosome stats into genome-wide min/max.

    WARNING: This uses mean-of-means (unweighted Welford across chromosome-
    level averages), giving only 22 data points for the std.  Use
    aggregate_chromosome_stats_corrected() instead for chunk-weighted stats.
    """
    genome_min = np.full(N_FEATURES, np.inf, dtype=np.float64)
    genome_max = np.full(N_FEATURES, -np.inf, dtype=np.float64)
    total_chunks = 0
    total_bp = 0
    chrom_summaries = []
    found_chroms = []
    missing_chroms = []

    # Welford's across chromosomes (on chunk_max_mean as proxy)
    running_mean = np.zeros(N_FEATURES, dtype=np.float64)
    running_m2 = np.zeros(N_FEATURES, dtype=np.float64)
    n_chroms_seen = 0

    for chrom in chroms:
        run_dir = find_latest_completed(results_dir, chrom, "sae_global_stats")
        if run_dir is None:
            logger.warning(f"  {chrom}: no completed sae_global_stats run")
            missing_chroms.append(chrom)
            continue

        stats_path = os.path.join(run_dir, "data", "global_sae_stats.npz")
        if not os.path.isfile(stats_path):
            logger.warning(f"  {chrom}: global_sae_stats.npz not found in {run_dir}")
            missing_chroms.append(chrom)
            continue

        data = np.load(stats_path)
        chrom_min = data["global_min"].astype(np.float64)
        chrom_max = data["global_max"].astype(np.float64)
        n_chunks = int(data["n_chunks"])
        genome_length = int(data["genome_length"])

        np.minimum(genome_min, chrom_min, out=genome_min)
        np.maximum(genome_max, chrom_max, out=genome_max)

        # Welford's on the per-chromosome chunk_max_mean
        chrom_mean = data["chunk_max_mean"].astype(np.float64)
        n_chroms_seen += 1
        delta = chrom_mean - running_mean
        running_mean += delta / n_chroms_seen
        delta2 = chrom_mean - running_mean
        running_m2 += delta * delta2

        total_chunks += n_chunks
        total_bp += genome_length
        found_chroms.append(chrom)
        chrom_summaries.append({
            "chrom": chrom,
            "n_chunks": n_chunks,
            "genome_length": genome_length,
            "run_dir": run_dir,
        })
        logger.info(f"  {chrom}: {genome_length:,} bp, {n_chunks} chunks")

    genome_min[genome_min == np.inf] = 0.0
    genome_max[genome_max == -np.inf] = 0.0

    if n_chroms_seen > 1:
        cross_chrom_std = np.sqrt(running_m2 / (n_chroms_seen - 1)).astype(np.float32)
    else:
        cross_chrom_std = np.zeros(N_FEATURES, dtype=np.float32)

    logger.info(f"\nAggregated {len(found_chroms)} chromosomes: "
                f"{total_bp:,} bp, {total_chunks} chunks")
    n_active = int(np.sum(genome_max > 0))
    logger.info(f"Features with any activation genome-wide: {n_active}/{N_FEATURES}")

    return {
        "genome_min": genome_min.astype(np.float32),
        "genome_max": genome_max.astype(np.float32),
        "cross_chrom_mean": running_mean.astype(np.float32),
        "cross_chrom_std": cross_chrom_std,
        "total_chunks": total_chunks,
        "total_bp": total_bp,
        "found_chroms": found_chroms,
        "missing_chroms": missing_chroms,
        "chrom_summaries": chrom_summaries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Genome-wide SAE feature min/max statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Scan mode (per-chromosome, GPU) ---
    parser.add_argument("--fasta", type=str, default=None,
                        help="Path to genome FASTA file")
    parser.add_argument("--chrom", type=str, default=None,
                        help="Chromosome to scan (e.g. chr22)")
    parser.add_argument("--chrom_name", type=str, default=None,
                        help="Friendly name for output dir (defaults to --chrom)")

    # --- Aggregate mode (cross-chromosome, CPU only) ---
    parser.add_argument("--aggregate", action="store_true",
                        help="LEGACY aggregate (mean-of-means, 22 data points)")
    parser.add_argument("--aggregate_corrected", action="store_true",
                        help="Correct chunk-weighted aggregate across all chromosomes")
    parser.add_argument("--chroms", nargs="+", type=str, default=None,
                        help="Chromosomes to aggregate")
    parser.add_argument("--all_human", action="store_true",
                        help="Aggregate all 24 human chromosomes")

    # --- Common ---
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base output/results directory (default: results/)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results dir for --aggregate (defaults to --output_dir)")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help=f"Chunk size in bp (default: {CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"Overlap between chunks (default: {OVERLAP})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Chunks per batched forward pass (default: {BATCH_SIZE}). "
                             f"Falls back to sequential on OOM.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    global logger
    logger = setup_logging(args.log_level)

    t_start = time.time()

    if args.aggregate_corrected:
        # ── Corrected aggregate mode (chunk-weighted) ────────────────────
        results_dir = os.path.abspath(args.results_dir or args.output_dir)

        if args.all_human:
            chroms = ALL_HUMAN_CHROMS
        elif args.chroms:
            chroms = args.chroms
        else:
            parser.error("--aggregate_corrected requires --chroms or --all_human")

        logger.info("=" * 70)
        logger.info("Genome-Wide SAE Stats — Corrected Chunk-Weighted Aggregation")
        logger.info("=" * 70)
        logger.info(f"Results dir: {results_dir}")
        logger.info(f"Chromosomes requested: {len(chroms)}")

        result = aggregate_chromosome_stats_corrected(results_dir, chroms)

        # Save output
        output_base = os.path.join(results_dir, "_genome_sae_stats")
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_found = len(result["found_chroms"])
        run_dir = os.path.join(output_base, f"{ts_str}_corrected_{n_found}chroms")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Stats file — uses global_mean/global_std keys (not cross_chrom_*)
        save_dict = {
            "genome_min": result["genome_min"],
            "genome_max": result["genome_max"],
            "global_mean": result["global_mean"],
            "global_std": result["global_std"],
            # Also save as mean/std so compute_sae_latent.py can load directly
            "mean": result["global_mean"],
            "std": result["global_std"],
        }
        # Add per-nucleotide stats if available
        if "nuc_mean" in result:
            save_dict["nuc_mean"] = result["nuc_mean"]
            save_dict["nuc_std"] = result["nuc_std"]
            save_dict["total_positions"] = np.array(result["total_positions"])
        np.savez_compressed(
            os.path.join(data_dir, "genome_wide_sae_stats_corrected.npz"),
            **save_dict,
        )

        # Chromosome summary
        with open(os.path.join(data_dir, "chromosome_summary.tsv"), "w") as f:
            f.write("chrom\tn_chunks\tgenome_length\trun_dir\n")
            for row in result["chrom_summaries"]:
                f.write(f"{row['chrom']}\t{row['n_chunks']}\t{row['genome_length']}\t{row['run_dir']}\n")

        # Metadata
        meta = {
            "chroms_found": result["found_chroms"],
            "chroms_missing": result["missing_chroms"],
            "total_chunks": result["total_chunks"],
            "total_bp": result["total_bp"],
            "total_positions": result.get("total_positions", 0),
            "has_nuc_stats": "nuc_mean" in result,
            "n_features": N_FEATURES,
            "n_active_features": int(np.sum(result["genome_max"] > 0)),
            "aggregation_method": "chunk_weighted_welford",
        }
        with open(os.path.join(data_dir, "aggregation_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

        # Source + completed
        source_inputs = {}
        for row in result["chrom_summaries"]:
            source_inputs[f"sae_global_stats_{row['chrom']}"] = row["run_dir"]
        write_source(run_dir, **source_inputs)
        write_completed(run_dir, "scan_sae_global_stats.py", time.time() - t_start)

        logger.info(f"\nOutput: {run_dir}")

        # ── Validation: compare to old aggregation ───────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION: Comparing corrected vs legacy aggregation")
        logger.info("=" * 70)

        old_result = aggregate_chromosome_stats(results_dir, chroms)

        old_mean = old_result["cross_chrom_mean"]
        new_mean = result["global_mean"]
        old_std = old_result["cross_chrom_std"]
        new_std = result["global_std"]

        valid = new_std > 0
        logger.info(f"Features with std > 0: {valid.sum()}/{N_FEATURES}")
        logger.info(f"Old method: mean-of-means across {len(old_result['found_chroms'])} chromosomes")
        logger.info(f"New method: chunk-weighted merge across {result['total_chunks']:,} chunks")
        logger.info(f"")
        logger.info(f"{'Metric':<30} {'Old (legacy)':<25} {'New (corrected)':<25}")
        logger.info(f"{'-'*80}")
        logger.info(f"{'Mean range':<30} [{old_mean.min():.4f}, {old_mean.max():.4f}]{'':>3} [{new_mean.min():.4f}, {new_mean.max():.4f}]")
        logger.info(f"{'Std range':<30} [{old_std.min():.4f}, {old_std.max():.4f}]{'':>3} [{new_std.min():.4f}, {new_std.max():.4f}]")
        logger.info(f"{'Avg mean (active features)':<30} {old_mean[valid].mean():.4f}{'':>17} {new_mean[valid].mean():.4f}")
        logger.info(f"{'Avg std (active features)':<30} {old_std[valid].mean():.4f}{'':>17} {new_std[valid].mean():.4f}")

        mean_corr = np.corrcoef(old_mean[valid], new_mean[valid])[0, 1]
        std_corr = np.corrcoef(old_std[valid], new_std[valid])[0, 1]
        logger.info(f"{'Mean correlation':<30} {mean_corr:.6f}")
        logger.info(f"{'Std correlation':<30} {std_corr:.6f}")

        if valid.sum() > 0:
            std_ratio = new_std[valid] / np.maximum(old_std[valid], 1e-10)
            logger.info(f"{'Median new_std / old_std':<30} {np.median(std_ratio):.4f}")
            logger.info(f"{'  (>1 = old underestimated)':<30}")

    elif args.aggregate:
        # ── Legacy aggregate mode ────────────────────────────────────────
        results_dir = os.path.abspath(args.results_dir or args.output_dir)

        if args.all_human:
            chroms = ALL_HUMAN_CHROMS
        elif args.chroms:
            chroms = args.chroms
        else:
            parser.error("--aggregate requires --chroms or --all_human")

        logger.info("=" * 70)
        logger.info("Genome-Wide SAE Stats Aggregation (LEGACY)")
        logger.info("=" * 70)
        logger.info(f"Results dir: {results_dir}")
        logger.info(f"Chromosomes: {len(chroms)}")

        result = aggregate_chromosome_stats(results_dir, chroms)

        # Save output
        output_base = os.path.join(results_dir, "_genome_sae_stats")
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_found = len(result["found_chroms"])
        run_dir = os.path.join(output_base, f"{ts_str}_genome_minmax_{n_found}chroms")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Stats file
        np.savez_compressed(
            os.path.join(data_dir, "genome_wide_sae_stats.npz"),
            genome_min=result["genome_min"],
            genome_max=result["genome_max"],
            cross_chrom_mean=result["cross_chrom_mean"],
            cross_chrom_std=result["cross_chrom_std"],
        )

        # Chromosome summary
        with open(os.path.join(data_dir, "chromosome_summary.tsv"), "w") as f:
            f.write("chrom\tn_chunks\tgenome_length\trun_dir\n")
            for row in result["chrom_summaries"]:
                f.write(f"{row['chrom']}\t{row['n_chunks']}\t{row['genome_length']}\t{row['run_dir']}\n")

        # Metadata
        meta = {
            "chroms_found": result["found_chroms"],
            "chroms_missing": result["missing_chroms"],
            "total_chunks": result["total_chunks"],
            "total_bp": result["total_bp"],
            "n_features": N_FEATURES,
            "n_active_features": int(np.sum(result["genome_max"] > 0)),
        }
        with open(os.path.join(data_dir, "aggregation_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

        # Source + completed
        source_inputs = {}
        for row in result["chrom_summaries"]:
            source_inputs[f"sae_global_stats_{row['chrom']}"] = row["run_dir"]
        write_source(run_dir, **source_inputs)
        write_completed(run_dir, "scan_sae_global_stats.py", time.time() - t_start)

        logger.info(f"\nOutput: {run_dir}")

    else:
        # ── Scan mode (single chromosome, GPU) ──────────────────────────
        if not args.fasta or not args.chrom:
            parser.error("Scan mode requires --fasta and --chrom")

        logger.info("=" * 70)
        logger.info("SAE Global Stats Scan")
        logger.info("=" * 70)
        logger.info(f"Chromosome: {args.chrom}")
        logger.info(f"FASTA: {args.fasta}")

        # Load sequence
        from run_sae_on_chromosome_drops import load_chromosome_sequence
        sequence = load_chromosome_sequence(args.fasta, args.chrom, logger)
        logger.info(f"Sequence length: {len(sequence):,} bp")

        # Initialize model and SAE
        logger.info("Initializing Evo2 model and SAE...")
        import torch
        from sae_utils import ObservableEvo2, load_topk_sae_from_hf
        model = ObservableEvo2("evo2_7b")
        sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)
        logger.info("Model and SAE loaded")

        # Set up checkpoint directory (inside output dir, before run_dir is created)
        chrom_name = args.chrom_name or args.chrom
        output_dir = os.path.abspath(args.output_dir)
        ckpt_dir = os.path.join(output_dir, chrom_name, "sae_global_stats", "_checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Scan (with checkpointing)
        stats = scan_chromosome_stats(
            sequence, model, sae, batch_size=args.batch_size,
            chunk_size=args.chunk_size, overlap=args.overlap,
            checkpoint_dir=ckpt_dir, checkpoint_interval=200,
        )

        # Clean up checkpoint after successful completion
        ckpt_file = os.path.join(ckpt_dir, "_scan_checkpoint.npz")
        if os.path.isfile(ckpt_file):
            os.remove(ckpt_file)
            logger.info("Checkpoint file removed (scan complete)")

        # Save output
        run_dir = build_run_dir(output_dir, chrom_name, "sae_global_stats", "minmax")
        data_dir = os.path.join(run_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(data_dir, "global_sae_stats.npz"),
            global_min=stats["global_min"],
            global_max=stats["global_max"],
            chunk_max_mean=stats["chunk_max_mean"],
            chunk_max_std=stats["chunk_max_std"],
            n_nonzero_chunks=stats["n_nonzero_chunks"],
            n_chunks=np.array(stats["n_chunks"]),
            genome_length=np.array(stats["genome_length"]),
            nuc_mean=stats["nuc_mean"],
            nuc_std=stats["nuc_std"],
            total_positions=np.array(stats["total_positions"]),
        )

        summary = {
            "chrom": args.chrom,
            "genome_length": stats["genome_length"],
            "n_chunks": stats["n_chunks"],
            "total_positions": stats["total_positions"],
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "n_features": N_FEATURES,
            "n_active_features": int(np.sum(stats["global_max"] > 0)),
            "global_max_range": [float(stats["global_max"].min()),
                                 float(stats["global_max"].max())],
            "nuc_mean_range": [float(stats["nuc_mean"].min()),
                               float(stats["nuc_mean"].max())],
            "nuc_std_range": [float(stats["nuc_std"].min()),
                              float(stats["nuc_std"].max())],
        }
        with open(os.path.join(data_dir, "scan_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

        write_source(run_dir, fasta=args.fasta)
        write_completed(run_dir, "scan_sae_global_stats.py", time.time() - t_start)

        logger.info(f"\nOutput: {run_dir}")
        logger.info(f"Active features: {summary['n_active_features']}/{N_FEATURES}")

    logger.info(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
