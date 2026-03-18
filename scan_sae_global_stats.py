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
                    results.append((chunk_min, chunk_max))

                del layer_acts
    finally:
        handle.remove()

    return results


def scan_chromosome_stats(sequence, model, sae, chunk_size=CHUNK_SIZE,
                          overlap=OVERLAP, batch_size=BATCH_SIZE):
    """Scan a chromosome in batched chunks, tracking per-feature global min and max.

    Only keeps two (32768,) arrays — no per-position or per-chunk storage.
    Processes multiple chunks per forward pass for better GPU utilization.

    Args:
        sequence: DNA string
        model: ObservableEvo2 instance
        sae: BatchTopKTiedSAE instance
        chunk_size: bp per chunk
        overlap: bp overlap between chunks
        batch_size: number of chunks per batched forward pass

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

    # For online mean/variance (Welford's algorithm)
    running_mean = np.zeros(N_FEATURES, dtype=np.float64)
    running_m2 = np.zeros(N_FEATURES, dtype=np.float64)
    n_nonzero_chunks = np.zeros(N_FEATURES, dtype=np.int64)
    n_processed = 0

    logger.info(f"Scanning {genome_len:,} bp in {n_chunks} chunks "
                f"({chunk_size} bp, {overlap} bp overlap, batch_size={batch_size})")

    t0 = time.time()
    i = 0
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

        for chunk_min, chunk_max in stats_list:
            np.minimum(global_min, chunk_min, out=global_min)
            np.maximum(global_max, chunk_max, out=global_max)

            n_processed += 1
            delta = chunk_max - running_mean
            running_mean += delta / n_processed
            delta2 = chunk_max - running_mean
            running_m2 += delta * delta2
            n_nonzero_chunks += (chunk_max > 0).astype(np.int64)

        i += batch_size

        if n_processed > 0 and n_processed % 50 == 0:
            elapsed = time.time() - t0
            rate = n_processed / elapsed
            eta = (n_chunks - n_processed) / rate if rate > 0 else 0
            n_active = np.sum(global_max > 0)
            logger.info(f"  Chunk {n_processed}/{n_chunks} ({elapsed:.0f}s elapsed, "
                        f"~{eta:.0f}s remaining, {n_active} features active so far)")

    elapsed = time.time() - t0
    logger.info(f"Scan complete in {elapsed:.1f}s ({n_processed} chunks processed)")

    # Finalize variance
    if n_processed > 1:
        running_var = running_m2 / (n_processed - 1)
    else:
        running_var = np.zeros(N_FEATURES, dtype=np.float64)

    # Replace inf with 0 for features never seen
    global_min[global_min == np.inf] = 0.0
    global_max[global_max == -np.inf] = 0.0

    n_active = int(np.sum(global_max > 0))
    logger.info(f"Features with any activation: {n_active}/{N_FEATURES}")
    logger.info(f"Global max range: [{global_max.min():.4f}, {global_max.max():.4f}]")

    return {
        "global_min": global_min.astype(np.float32),
        "global_max": global_max.astype(np.float32),
        "chunk_max_mean": running_mean.astype(np.float32),
        "chunk_max_std": np.sqrt(running_var).astype(np.float32),
        "n_nonzero_chunks": n_nonzero_chunks,
        "n_chunks": n_processed,
        "genome_length": genome_len,
    }


def aggregate_chromosome_stats(results_dir, chroms):
    """Merge per-chromosome stats into genome-wide min/max.

    Loads each chromosome's global_sae_stats.npz and takes element-wise
    min/max across all chromosomes.
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
                        help="Aggregate per-chromosome stats (no GPU needed)")
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

    if args.aggregate:
        # ── Aggregate mode ───────────────────────────────────────────────
        results_dir = os.path.abspath(args.results_dir or args.output_dir)

        if args.all_human:
            chroms = ALL_HUMAN_CHROMS
        elif args.chroms:
            chroms = args.chroms
        else:
            parser.error("--aggregate requires --chroms or --all_human")

        logger.info("=" * 70)
        logger.info("Genome-Wide SAE Stats Aggregation")
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

        # Scan
        stats = scan_chromosome_stats(
            sequence, model, sae, batch_size=args.batch_size,
            chunk_size=args.chunk_size, overlap=args.overlap,
        )

        # Save output
        chrom_name = args.chrom_name or args.chrom
        output_dir = os.path.abspath(args.output_dir)
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
        )

        summary = {
            "chrom": args.chrom,
            "genome_length": stats["genome_length"],
            "n_chunks": stats["n_chunks"],
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "n_features": N_FEATURES,
            "n_active_features": int(np.sum(stats["global_max"] > 0)),
            "global_max_range": [float(stats["global_max"].min()),
                                 float(stats["global_max"].max())],
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
