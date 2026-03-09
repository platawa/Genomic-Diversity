#!/usr/bin/env python3
"""
score_chromosome.py

================================================================================
OVERVIEW
================================================================================
Chromosome-wide entropy scoring and drop boundary detection using Evo2.

This script scores an entire human chromosome (or a specified region) and
identifies entropy "drop regions" - areas where the model shows high confidence
(low entropy) that may correspond to functional elements.

Key Features:
- Scores entire chromosome in overlapping chunks (GPU memory efficient)
- Uses MAD and z-score methods for robust drop detection
- Identifies BOTH drop starts (entropy decreasing) and drop ends (entropy rising)
- Pairs drops and rises to define complete "low-entropy regions"
- Outputs comprehensive TSV with all detected boundaries

================================================================================
OUTPUT FILES
================================================================================
1. <output_prefix>.entropy.npz
   - Compressed numpy archive with per-position entropy values
   - Keys: 'entropy', 'positions', 'chrom', 'start', 'end'

2. <output_prefix>.drop_boundaries.tsv
   - Main output: detected drop regions with boundaries
   - Columns: chrom, drop_start, drop_end, region_length, method,
              start_confidence, end_confidence, mean_entropy, min_entropy

3. <output_prefix>.drops.tsv
   - All detected drop points (entropy starts decreasing)
   - Columns: chrom, position, genomic_position, method, confidence

4. <output_prefix>.rises.tsv
   - All detected rise points (entropy starts increasing)
   - Columns: chrom, position, genomic_position, method, confidence

5. <output_prefix>.summary.json
   - Run metadata and summary statistics

================================================================================
USAGE
================================================================================
    # Score human chromosome 21 (smallest autosome, good for testing)
    python score_chromosome.py --chrom NC_000021.9 --output_prefix chr21_test

    # Score a specific region of chromosome 1
    python score_chromosome.py --chrom NC_000001.11 --start 1000000 --end 2000000

    # Score with custom thresholds
    python score_chromosome.py --chrom NC_000021.9 --zscore_threshold 3.0 --mad_threshold 3.5

    # Score entire chromosome (will take a while)
    python score_chromosome.py --chrom NC_000001.11 --output_prefix chr1_full

================================================================================
"""

import os
import sys

# If spawned as a multi-GPU worker, restrict CUDA to the assigned GPU.
# Must happen BEFORE `import torch` so Vortex sees only 1 GPU.
_WORKER_GPU = os.environ.pop("_SCORE_CHROM_GPU_ID", None)
if _WORKER_GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _WORKER_GPU

import gc
import json
import random
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.multiprocessing as mp

from Bio import SeqIO
from Bio.Seq import Seq


# =============================================================================
# VECTORIZED HELPERS (avoid Python-level character loops)
# =============================================================================

def _find_non_n_runs(sequence: str) -> List[Tuple[int, int]]:
    """Find contiguous non-N runs using numpy. Returns list of (start, end) tuples.

    ~100x faster than character-by-character Python loop for large sequences.
    """
    arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    is_valid = (arr != ord('N'))
    if not np.any(is_valid):
        return []
    # Find boundaries: prepend/append False to detect edges
    padded = np.concatenate(([False], is_valid, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))

# Import Evo2 model
from evo2 import Evo2

from results_utils import build_run_dir, write_completed
from detection_methods import (
    detect_drops_zscore, detect_drops_mad,
    detect_rises_zscore, detect_rises_mad,
    _rolling_mean, _cluster_and_pick_best,
    METHODS as DETECTION_METHODS,
    RISE_METHODS as DETECTION_RISE_METHODS,
    run_method as run_detection_method,
)


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configure logging with console and optional file output."""
    logger = logging.getLogger("score_chromosome")
    logger.setLevel(getattr(logging, log_level.upper()))

    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    # File handler (if specified)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# =============================================================================
# CONFIGURATION
# =============================================================================

# Human reference genome (GRCh38)
DEFAULT_FASTA = "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna"

# Chromosome name mapping (common names to RefSeq accessions)
CHROM_MAP = {
    "chr1": "NC_000001.11", "1": "NC_000001.11",
    "chr2": "NC_000002.12", "2": "NC_000002.12",
    "chr3": "NC_000003.12", "3": "NC_000003.12",
    "chr4": "NC_000004.12", "4": "NC_000004.12",
    "chr5": "NC_000005.10", "5": "NC_000005.10",
    "chr6": "NC_000006.12", "6": "NC_000006.12",
    "chr7": "NC_000007.14", "7": "NC_000007.14",
    "chr8": "NC_000008.11", "8": "NC_000008.11",
    "chr9": "NC_000009.12", "9": "NC_000009.12",
    "chr10": "NC_000010.11", "10": "NC_000010.11",
    "chr11": "NC_000011.10", "11": "NC_000011.10",
    "chr12": "NC_000012.12", "12": "NC_000012.12",
    "chr13": "NC_000013.11", "13": "NC_000013.11",
    "chr14": "NC_000014.9", "14": "NC_000014.9",
    "chr15": "NC_000015.10", "15": "NC_000015.10",
    "chr16": "NC_000016.10", "16": "NC_000016.10",
    "chr17": "NC_000017.11", "17": "NC_000017.11",
    "chr18": "NC_000018.10", "18": "NC_000018.10",
    "chr19": "NC_000019.10", "19": "NC_000019.10",
    "chr20": "NC_000020.11", "20": "NC_000020.11",
    "chr21": "NC_000021.9", "21": "NC_000021.9",
    "chr22": "NC_000022.11", "22": "NC_000022.11",
    "chrX": "NC_000023.11", "X": "NC_000023.11",
    "chrY": "NC_000024.10", "Y": "NC_000024.10",
    "chrM": "NC_012920.1", "M": "NC_012920.1", "MT": "NC_012920.1",
}

# Algorithm parameters
MAX_CHUNK_LEN = 15000
CHUNK_OVERLAP = 1024
SMOOTH_W = 51
ZSCORE_THRESHOLD = 2.5
MAD_THRESHOLD = 3.0
MIN_SEPARATION = 75


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DropRegion:
    """Represents a detected low-entropy region with boundaries."""
    chrom: str
    drop_start: int  # Position where entropy starts dropping
    drop_end: int    # Position where entropy recovers
    region_length: int
    method: str
    start_confidence: float
    end_confidence: float
    mean_entropy: float
    min_entropy: float
    genomic_start: int  # Genomic coordinate (1-based)
    genomic_end: int


@dataclass
class DetectionResult:
    """Container for all detection results."""
    drops_zscore: List[Tuple[int, float]] = field(default_factory=list)
    drops_mad: List[Tuple[int, float]] = field(default_factory=list)
    rises_zscore: List[Tuple[int, float]] = field(default_factory=list)
    rises_mad: List[Tuple[int, float]] = field(default_factory=list)
    regions_zscore: List[DropRegion] = field(default_factory=list)
    regions_mad: List[DropRegion] = field(default_factory=list)


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def _bos_id(tok) -> int:
    """Get BOS token ID from tokenizer."""
    bid = getattr(tok, "bos_id", getattr(tok, "eod_id", None))
    if bid is None:
        raise AttributeError("Tokenizer has neither bos_id nor eod_id")
    return bid


def _extract_logits(model_out) -> torch.Tensor:
    """Robustly extract logits tensor from potentially nested model output.

    Evo2 returns nested tuples. This recursively searches for a 3D
    tensor [Batch, Time, Vocab]. Matches genome_scoring_jan26_drops.py.
    """
    def walk(x):
        if isinstance(x, torch.Tensor):
            yield x
            return
        lg = getattr(x, "logits", None)
        if isinstance(lg, torch.Tensor):
            yield lg
        if isinstance(x, dict):
            if isinstance(x.get("logits"), torch.Tensor):
                yield x["logits"]
            for v in x.values():
                yield from walk(v)
            return
        if isinstance(x, (tuple, list)):
            for v in x:
                yield from walk(v)
            return

    candidates = list(walk(model_out))
    if not candidates:
        raise TypeError(f"Could not extract logits from model output of type {type(model_out)}")
    for t in candidates:
        if t.ndim == 3:
            return t
    return candidates[0]


def _id_to_token_str(tok, idx: int) -> str:
    """Convert a token ID back to its string representation."""
    for attr in ("id_to_token", "decode", "detokenize", "convert_ids_to_tokens"):
        fn = getattr(tok, attr, None)
        if callable(fn):
            try:
                out = fn([idx]) if attr in ("decode", "detokenize", "convert_ids_to_tokens") else fn(idx)
                return out[0] if isinstance(out, (list, tuple)) else str(out)
            except Exception:
                pass
    return str(idx)


def get_acgt_token_ids(model: Evo2, device: str) -> torch.Tensor:
    """Get token IDs for A, C, G, T nucleotides."""
    tok = model.tokenizer
    acgt_ids = []
    for nuc in ["A", "C", "G", "T"]:
        toks = tok.tokenize(nuc)
        if len(toks) == 1:
            acgt_ids.append(toks[0])
        else:
            raise ValueError(f"Nucleotide {nuc} tokenized to multiple tokens: {toks}")
    return torch.tensor(acgt_ids, dtype=torch.long, device=device)


# =============================================================================
# AUTO CHUNK SIZE PROBING
# =============================================================================

def find_max_chunk_size(
    model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    target_size: int = 100_000,
    min_size: int = 1_000,
    logger: logging.Logger = None,
) -> int:
    """
    Probe GPU memory to find the largest chunk size that fits.

    Tries a forward pass at `target_size`, halving on OOM until it works.
    Uses random ACGT sequence (worst-case memory, no N-skipping).

    Returns:
        Largest chunk size (bp) that completes without OOM.
    """
    current = target_size
    while current >= min_size:
        try:
            test_seq = "".join(random.choices("ACGT", k=current))
            _ = compute_entropy_chunk(test_seq, model, ACGT_IDS, device)
            torch.cuda.empty_cache()
            if logger:
                logger.info(f"  Chunk size {current:,} bp: OK")
            return current
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            gc.collect()
            if logger:
                logger.info(f"  Chunk size {current:,} bp: OOM, halving")
            current = current // 2

    raise RuntimeError(f"Cannot fit even {min_size:,} bp chunk in GPU memory")


def _probe_chunk_size_worker(target_size: int, model_name: str, result_queue):
    """Probe max chunk size on a single GPU in a spawned subprocess.

    Must be spawned with ``_SCORE_CHROM_GPU_ID`` set in the environment so
    that the module-level code (lines 67-69) restricts CUDA visibility
    BEFORE ``import torch``.  This ensures the probe model loads on exactly
    one GPU — matching what the real per-GPU workers will see.
    """
    model = Evo2(model_name)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model"):
        model.model.eval()
    device = "cuda:0"
    ACGT_IDS = get_acgt_token_ids(model, device)

    try:
        chunk_size = find_max_chunk_size(
            model, ACGT_IDS, device, target_size=target_size,
        )
    except RuntimeError:
        chunk_size = 0  # signal failure

    result_queue.put(chunk_size)


# =============================================================================
# ENTROPY SCORING
# =============================================================================

@torch.inference_mode()
def compute_entropy_chunk(
    sequence: str,
    model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    reverse_complement: bool = False,
    compute_logprobs: bool = False,
) -> Union[np.ndarray, dict]:
    """
    Compute per-position entropy for a sequence chunk.

    When compute_logprobs=True, also extracts P(A/C/G/T) and next-token
    log-likelihood from the SAME forward pass (zero extra cost).

    Args:
        sequence: DNA sequence string (ACGT only, no N's)
        model: Evo2 model instance
        ACGT_IDS: Token IDs for A, C, G, T
        device: Compute device
        reverse_complement: If True, also compute RC entropy and average
            with forward entropy for strand-independent signal.
        compute_logprobs: If True, return dict with entropy, P(ACGT),
            next-token log-likelihood, and true tokens.

    Returns:
        If compute_logprobs=False: np.ndarray of entropy values (nats)
        If compute_logprobs=True: dict with keys:
            "entropy": np.ndarray [L] per-position entropy (nats)
            "p_acgt":  np.ndarray [L, 4] P(A), P(C), P(G), P(T)
            "ll_next": np.ndarray [L-1] log-likelihood of true next token
            "true_token": list[str] [L-1] actual token at each position
    """
    tok = model.tokenizer
    _acgt_ids = ACGT_IDS.to(device)

    def _forward_pass(seq):
        """Single forward pass → entropy + optional logprobs from same logits."""
        toks_list = tok.tokenize(seq)
        toks_list = [_bos_id(tok)] + toks_list
        input_ids = torch.tensor(toks_list, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids)
        logits = _extract_logits(out).float()

        # ACGT entropy (always computed)
        logits_sub = logits.index_select(-1, _acgt_ids)
        logZ = torch.logsumexp(logits_sub, dim=-1, keepdim=True)
        logp = logits_sub - logZ
        H = -(logp.exp() * logp).sum(dim=-1)
        H = H[:, 1:]  # Remove BOS position
        entropy = H.squeeze(0).detach().cpu().numpy()

        if not compute_logprobs:
            return entropy

        # P(A/C/G/T) — already computed from ACGT logits
        p_acgt = logp.exp()[:, 1:, :]  # Remove BOS, [1, L, 4]
        p_acgt_np = p_acgt.squeeze(0).detach().cpu().numpy()

        # Next-token log-likelihood from full vocabulary (same logits tensor)
        full_logprobs = torch.log_softmax(logits, dim=-1)  # [1, T, V]
        # Shift: logprobs[t] predicts token[t+1]
        ll_next = full_logprobs[:, :-1, :].gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [1, L]
        ll_next_np = ll_next.squeeze(0).detach().cpu().numpy()

        # True tokens at each position (excluding BOS)
        true_ids = input_ids[0, 1:].cpu().tolist()
        true_tokens = [_id_to_token_str(tok, tid) for tid in true_ids]

        return {
            "entropy": entropy,
            "p_acgt": p_acgt_np,
            "ll_next": ll_next_np,
            "true_token": true_tokens,
        }

    def _batched_fwd_rc_pass(seq_fwd, seq_rc):
        """Batched forward pass: fwd + RC in a single model call (batch=2).

        Both sequences must be the same length (always true for fwd/RC pairs).
        Returns (result_fwd, result_rc) in the same format as _forward_pass.
        """
        bos = _bos_id(tok)
        toks_fwd = [bos] + tok.tokenize(seq_fwd)
        toks_rc = [bos] + tok.tokenize(seq_rc)
        # Stack as [2, T]
        input_ids = torch.tensor([toks_fwd, toks_rc], dtype=torch.long, device=device)
        out = model(input_ids)
        logits = _extract_logits(out).float()  # [2, T, V]

        # ACGT entropy for both
        logits_sub = logits.index_select(-1, _acgt_ids)  # [2, T, 4]
        logZ = torch.logsumexp(logits_sub, dim=-1, keepdim=True)
        logp = logits_sub - logZ
        H = -(logp.exp() * logp).sum(dim=-1)
        H = H[:, 1:]  # Remove BOS, [2, L]
        entropy_both = H.detach().cpu().numpy()  # [2, L]

        if not compute_logprobs:
            return entropy_both[0], entropy_both[1]

        # P(ACGT) for both
        p_acgt_both = logp.exp()[:, 1:, :].detach().cpu().numpy()  # [2, L, 4]

        # Next-token log-likelihood from full vocabulary
        full_logprobs = torch.log_softmax(logits, dim=-1)  # [2, T, V]
        ll_next = full_logprobs[:, :-1, :].gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [2, L]
        ll_next_both = ll_next.detach().cpu().numpy()

        # True tokens (forward strand only)
        true_ids = input_ids[0, 1:].cpu().tolist()
        true_tokens = [_id_to_token_str(tok, tid) for tid in true_ids]

        result_fwd = {
            "entropy": entropy_both[0],
            "p_acgt": p_acgt_both[0],
            "ll_next": ll_next_both[0],
            "true_token": true_tokens,
        }
        result_rc = {
            "entropy": entropy_both[1],
            "p_acgt": p_acgt_both[1],
            "ll_next": ll_next_both[1],
            "true_token": [],  # Not needed for RC
        }
        return result_fwd, result_rc

    if not reverse_complement:
        return _forward_pass(sequence)

    # Batched fwd + RC in a single model call
    seq_rc = str(Seq(sequence).reverse_complement())
    try:
        result_fwd, result_rc = _batched_fwd_rc_pass(sequence, seq_rc)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        # Fall back to sequential if batch=2 OOMs
        torch.cuda.empty_cache()
        result_fwd = _forward_pass(sequence)
        result_rc = _forward_pass(seq_rc)

    if not compute_logprobs:
        return 0.5 * (result_fwd + result_rc[::-1])

    # Fused logprobs with RC averaging:
    # - Entropy: average fwd and RC (flipped)
    # - P(ACGT): average fwd and RC (flipped, with complement swap: A↔T, C↔G)
    # - LL_next: forward strand only (RC LL is meaningless for this purpose)
    # - true_token: forward strand only
    H_fwd = result_fwd["entropy"]
    H_rc = result_rc["entropy"][::-1]
    entropy_avg = 0.5 * (H_fwd + H_rc)

    p_fwd = result_fwd["p_acgt"]
    p_rc = result_rc["p_acgt"][::-1, :]  # Flip positions
    # RC complement swap: A(0)↔T(3), C(1)↔G(2)
    p_rc_complement = p_rc[:, [3, 2, 1, 0]]
    p_avg = 0.5 * (p_fwd + p_rc_complement)

    return {
        "entropy": entropy_avg,
        "p_acgt": p_avg,
        "ll_next": result_fwd["ll_next"],
        "true_token": result_fwd["true_token"],
    }


def score_chromosome_region(
    sequence: str,
    model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    max_chunk_len: int = MAX_CHUNK_LEN,
    chunk_overlap: int = CHUNK_OVERLAP,
    reverse_complement: bool = False,
    compute_logprobs: bool = False,
    logger: logging.Logger = None,
    stitch_method: str = "core",
) -> Union[np.ndarray, dict]:
    """
    Score a chromosome region using overlapping chunks.

    Long sequences are split into overlapping chunks. Each chunk is scored,
    and the overlap zones are handled according to stitch_method:
      - "core": only the center of each chunk is used (default, discards edges)
      - "mean": average entropy values from overlapping chunks
      - "min":  take lowest entropy (highest confidence) in overlap zones

    Args:
        sequence: DNA sequence string
        model: Evo2 model
        ACGT_IDS: Token IDs for ACGT
        device: Compute device
        max_chunk_len: Maximum bases per chunk
        chunk_overlap: Overlap between chunks
        reverse_complement: If True, average forward and RC entropy
        compute_logprobs: If True, also return P(ACGT) and LL_next
        logger: Logger instance
        stitch_method: How to handle overlap zones ("core", "mean", "min")

    Returns:
        If compute_logprobs=False: Per-position entropy array (nats)
        If compute_logprobs=True: dict with "entropy", "p_acgt", "ll_next", "true_token"
    """
    L = len(sequence)

    if logger:
        logger.info(f"Scoring sequence of length {L:,} bp")

    # Initialize output arrays
    entropy = np.full(L, np.nan, dtype=np.float32)
    if compute_logprobs:
        p_acgt = np.full((L, 4), np.nan, dtype=np.float32)
        ll_next = np.full(L, np.nan, dtype=np.float32)
        true_token = [""] * L

    # Experimental stitch methods: accumulation arrays
    if stitch_method == "mean":
        entropy_sum = np.zeros(L, dtype=np.float64)
        entropy_count = np.zeros(L, dtype=np.int32)
    elif stitch_method == "min":
        entropy[:] = np.inf

    def _extract_entropy(result):
        """Get entropy array from chunk result (ndarray or dict)."""
        if isinstance(result, dict):
            return result["entropy"]
        return result

    def _write_core(result, run_start_in_chunk, run_end_in_chunk,
                    chunk_start, core_s, core_e):
        """Write positions from a chunk result to output arrays (vectorized)."""
        ent = _extract_entropy(result)
        # Compute the intersection of the run's global range and the core range
        g_start = chunk_start + run_start_in_chunk
        g_end = chunk_start + run_end_in_chunk
        w_start = max(g_start, core_s)
        w_end = min(g_end, core_e)
        if w_start >= w_end:
            return
        # Offsets into the run result
        r_start = w_start - g_start
        r_end = r_start + (w_end - w_start)
        r_end = min(r_end, len(ent))
        if r_start >= r_end:
            return
        n = r_end - r_start
        src = ent[r_start:r_end]
        if stitch_method == "mean":
            valid = ~np.isnan(src)
            entropy_sum[w_start:w_start + n][valid] += src[valid]
            entropy_count[w_start:w_start + n][valid] += 1
        elif stitch_method == "min":
            valid = ~np.isnan(src)
            dest = entropy[w_start:w_start + n]
            dest[valid] = np.minimum(dest[valid], src[valid])
        else:
            entropy[w_start:w_start + n] = src
        if compute_logprobs and isinstance(result, dict):
            rp = min(r_end, len(result["p_acgt"]))
            if rp > r_start:
                p_acgt[w_start:w_start + (rp - r_start)] = result["p_acgt"][r_start:rp]
            rl = min(r_end, len(result["ll_next"]))
            if rl > r_start:
                ll_next[w_start:w_start + (rl - r_start)] = result["ll_next"][r_start:rl]
            rt = min(r_end, len(result["true_token"]))
            if rt > r_start:
                for ti in range(r_start, rt):
                    true_token[w_start + ti - r_start] = result["true_token"][ti]

    if L <= max_chunk_len:
        # Process contiguous non-N runs (vectorized N-boundary detection)
        runs = _find_non_n_runs(sequence)
        for i, j in runs:
            run_seq = sequence[i:j]
            result = compute_entropy_chunk(run_seq, model, ACGT_IDS, device,
                                           reverse_complement=reverse_complement,
                                           compute_logprobs=compute_logprobs)
            ent = _extract_entropy(result)
            entropy[i:i+len(ent)] = ent
            if compute_logprobs and isinstance(result, dict):
                p_acgt[i:i+len(result["p_acgt"])] = result["p_acgt"]
                n_ll = len(result["ll_next"])
                ll_next[i:i+n_ll] = result["ll_next"]
                for ti, t in enumerate(result["true_token"]):
                    if i + ti < L:
                        true_token[i + ti] = t

        if not compute_logprobs:
            return entropy
        return {"entropy": entropy, "p_acgt": p_acgt,
                "ll_next": ll_next, "true_token": true_token}

    # Calculate step size
    step = max(1, max_chunk_len - chunk_overlap)

    # Process chunks
    chunk_starts = list(range(0, L, step))
    n_chunks = len(chunk_starts)

    if logger:
        logger.info(f"Processing {n_chunks} chunks (max_len={max_chunk_len}, overlap={chunk_overlap})")

    scoring_start = time.time()
    for i, s in enumerate(chunk_starts):
        e = min(s + max_chunk_len, L)
        chunk_seq = sequence[s:e]

        # Core region bounds depend on stitch method
        if stitch_method == "core":
            core_s = s if s == 0 else s + chunk_overlap // 2
            core_e = e if e == L else e - chunk_overlap // 2
        else:
            # mean/min: use full chunk, handle overlap in accumulation
            core_s = s
            core_e = e
        core_s = min(core_s, core_e)

        if logger and (i % 10 == 0 or i == n_chunks - 1):
            elapsed = time.time() - scoring_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_chunks - i - 1) / rate if rate > 0 else 0
            logger.info(f"  Chunk {i+1}/{n_chunks}: positions {s:,}-{e:,} "
                        f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        # Process contiguous non-N runs within this chunk (vectorized)
        runs = _find_non_n_runs(chunk_seq)
        for ci, cj in runs:
            run_seq = chunk_seq[ci:cj]
            try:
                result = compute_entropy_chunk(run_seq, model, ACGT_IDS, device,
                                               reverse_complement=reverse_complement,
                                               compute_logprobs=compute_logprobs)
            except Exception as ex:
                if logger:
                    logger.error(f"  Chunk {i+1} run scoring failed: {ex}")
                continue

            _write_core(result, ci, cj, s, core_s, core_e)

    # Post-process experimental stitch methods
    if stitch_method == "mean":
        valid = entropy_count > 0
        entropy = np.where(valid, (entropy_sum / entropy_count).astype(np.float32), np.nan)
        if logger:
            n_overlap = int((entropy_count > 1).sum())
            logger.info(f"  Stitch method 'mean': {n_overlap:,} positions averaged from 2 chunks")
    elif stitch_method == "min":
        n_overlap = int(np.isfinite(entropy).sum())  # count before replacing inf
        entropy[np.isinf(entropy)] = np.nan
        if logger:
            logger.info(f"  Stitch method 'min': lowest entropy selected in overlap zones")

    if not compute_logprobs:
        return entropy
    return {"entropy": entropy, "p_acgt": p_acgt,
            "ll_next": ll_next, "true_token": true_token}


# =============================================================================
# MULTI-GPU DATA PARALLELISM
# =============================================================================

def _chromosome_gpu_worker(
    gpu_id: int,
    work_items: list,
    seq_shm_name: str,
    result_shm_name: str,
    seq_len: int,
    total_len: int,
    model_name: str = "evo2_7b",
    reverse_complement: bool = False,
    compute_logprobs: bool = False,
    p_acgt_shm_name: str = None,
    ll_next_shm_name: str = None,
    stitch_method: str = "core",
    result_shm_alt_name: str = None,
    use_torch_compile: bool = False,
):
    """
    Worker process that loads a model on a specific GPU and scores assigned chunks.

    Uses shared memory for zero-copy access to the sequence and results,
    avoiding costly serialization of large arrays through Manager proxies.

    Args:
        gpu_id: CUDA device index for this worker
        work_items: List of (chunk_idx, start, end, core_start, core_end)
        seq_shm_name: Name of shared memory block holding the sequence
        result_shm_name: Name of shared memory block for entropy output
        seq_len: Length of the sequence in shared memory
        total_len: Total length of the entropy output array
        model_name: Evo2 model to load
        reverse_complement: If True, average forward and RC entropy
        compute_logprobs: If True, also write P(ACGT) and LL_next to shared memory
        p_acgt_shm_name: Shared memory name for P(ACGT) [L, 4] float32
        ll_next_shm_name: Shared memory name for LL_next [L] float32
        stitch_method: Overlap handling ("core", "mean", "min")
        result_shm_alt_name: Alternate shared memory for even/odd layer separation
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import re as _re
    import time as _time
    import numpy as _np
    import torch as _torch
    from multiprocessing import shared_memory as _shm
    from evo2 import Evo2
    from Bio.Seq import Seq as _Seq

    # Attach to shared memory
    seq_shm = _shm.SharedMemory(name=seq_shm_name)
    sequence = bytes(seq_shm.buf[:seq_len]).decode('ascii')
    result_shm = _shm.SharedMemory(name=result_shm_name)
    entropy_out = _np.ndarray(total_len, dtype=_np.float32, buffer=result_shm.buf)

    # Alternate layer for experimental stitch methods (even/odd chunk separation)
    result_shm_alt = None
    entropy_out_alt = None
    if stitch_method != "core" and result_shm_alt_name:
        result_shm_alt = _shm.SharedMemory(name=result_shm_alt_name)
        entropy_out_alt = _np.ndarray(total_len, dtype=_np.float32, buffer=result_shm_alt.buf)

    # Optional logprobs shared memory
    p_acgt_shm = None
    p_acgt_out = None
    ll_next_shm = None
    ll_next_out = None
    if compute_logprobs and p_acgt_shm_name and ll_next_shm_name:
        p_acgt_shm = _shm.SharedMemory(name=p_acgt_shm_name)
        p_acgt_out = _np.ndarray((total_len, 4), dtype=_np.float32, buffer=p_acgt_shm.buf)
        ll_next_shm = _shm.SharedMemory(name=ll_next_shm_name)
        ll_next_out = _np.ndarray(total_len, dtype=_np.float32, buffer=ll_next_shm.buf)

    print(f"[GPU {gpu_id}] Loading model on CUDA device {gpu_id}...")
    model = Evo2(model_name)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model"):
        model.model.eval()

    # Try torch.compile for fused kernels
    if use_torch_compile:
        try:
            inner = model.model if hasattr(model, "model") else model
            compiled = _torch.compile(inner, mode="default")
            if hasattr(model, "model"):
                model.model = compiled
            else:
                model = compiled
            print(f"[GPU {gpu_id}] torch.compile applied successfully")
        except Exception as ex:
            print(f"[GPU {gpu_id}] torch.compile failed ({ex}), continuing without it")

    device = "cuda:0"  # Each worker sees only its assigned GPU
    tok = model.tokenizer
    acgt_ids = []
    for nuc in ["A", "C", "G", "T"]:
        toks_list = tok.tokenize(nuc)
        acgt_ids.append(toks_list[0])
    ACGT_IDS = _torch.tensor(acgt_ids, dtype=_torch.long, device=device)

    rc_label = " (RC-averaged)" if reverse_complement else ""
    lp_label = " +logprobs" if compute_logprobs else ""
    print(f"[GPU {gpu_id}] Model loaded. Processing {len(work_items)} chunks{rc_label}{lp_label}...")

    non_acgt_re = _re.compile(r'[^ACGT]')
    bases = ['A', 'C', 'G', 'T']

    def _id_to_str(idx):
        for attr in ("id_to_token", "decode", "detokenize", "convert_ids_to_tokens"):
            fn = getattr(tok, attr, None)
            if callable(fn):
                try:
                    out = fn([idx]) if attr in ("decode", "detokenize", "convert_ids_to_tokens") else fn(idx)
                    return out[0] if isinstance(out, (list, tuple)) else str(out)
                except Exception:
                    pass
        return str(idx)

    def _forward_pass(seq):
        """Single forward pass → entropy (+ optional logprobs) from same logits."""
        toks_list = tok.tokenize(seq)
        bos = getattr(tok, "bos_id", getattr(tok, "eod_id", None))
        toks_list = [bos] + toks_list
        input_ids = _torch.tensor(toks_list, dtype=_torch.long, device=device).unsqueeze(0)
        out = model(input_ids)

        # Extract logits robustly (reuse module-level helper that
        # recursively unwraps nested tuples from Evo2/Vortex)
        logits = _extract_logits(out).float()

        # ACGT entropy
        logits_sub = logits.index_select(-1, ACGT_IDS.to(logits.device))
        logZ = _torch.logsumexp(logits_sub, dim=-1, keepdim=True)
        logp = logits_sub - logZ
        H = -(logp.exp() * logp).sum(dim=-1)
        H = H[:, 1:]  # Remove BOS
        entropy = H.squeeze(0).detach().cpu().numpy()

        if not compute_logprobs:
            return entropy

        # P(ACGT) — from same logits
        p_acgt = logp.exp()[:, 1:, :]
        p_acgt_np = p_acgt.squeeze(0).detach().cpu().numpy()

        # Next-token log-likelihood from full vocabulary
        full_lp = _torch.log_softmax(logits, dim=-1)
        ll = full_lp[:, :-1, :].gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ll_np = ll.squeeze(0).detach().cpu().numpy()

        # True tokens
        true_ids = input_ids[0, 1:].cpu().tolist()
        true_toks = [_id_to_str(tid) for tid in true_ids]

        return {
            "entropy": entropy,
            "p_acgt": p_acgt_np,
            "ll_next": ll_np,
            "true_token": true_toks,
        }

    def _batched_fwd_rc_pass(seq_fwd, seq_rc):
        """Batched forward pass: fwd + RC in a single model call (batch=2).

        Both sequences are the same length (always true for fwd/RC pairs).
        Returns (result_fwd, result_rc) in the same format as _forward_pass.
        """
        bos = getattr(tok, "bos_id", getattr(tok, "eod_id", None))
        toks_fwd = [bos] + tok.tokenize(seq_fwd)
        toks_rc = [bos] + tok.tokenize(seq_rc)
        input_ids = _torch.tensor([toks_fwd, toks_rc], dtype=_torch.long, device=device)
        out = model(input_ids)
        logits = _extract_logits(out).float()  # [2, T, V]

        # ACGT entropy for both
        logits_sub = logits.index_select(-1, ACGT_IDS.to(logits.device))
        logZ = _torch.logsumexp(logits_sub, dim=-1, keepdim=True)
        logp = logits_sub - logZ
        H = -(logp.exp() * logp).sum(dim=-1)
        H = H[:, 1:]  # [2, L]
        entropy_both = H.detach().cpu().numpy()

        if not compute_logprobs:
            return entropy_both[0], entropy_both[1]

        p_acgt_both = logp.exp()[:, 1:, :].detach().cpu().numpy()
        full_lp = _torch.log_softmax(logits, dim=-1)
        ll = full_lp[:, :-1, :].gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        ll_both = ll.detach().cpu().numpy()

        true_ids = input_ids[0, 1:].cpu().tolist()
        true_toks = [_id_to_str(tid) for tid in true_ids]

        result_fwd = {
            "entropy": entropy_both[0], "p_acgt": p_acgt_both[0],
            "ll_next": ll_both[0], "true_token": true_toks,
        }
        result_rc = {
            "entropy": entropy_both[1], "p_acgt": p_acgt_both[1],
            "ll_next": ll_both[1], "true_token": [],
        }
        return result_fwd, result_rc

    # Track whether batched RC works (OOM falls back to sequential)
    _batched_rc_ok = True

    t0 = _time.time()
    n_scored = 0
    n_skipped = 0

    for i, (chunk_idx, s, e, core_s, core_e) in enumerate(work_items):
        chunk_seq = sequence[s:e]
        chunk_len = e - s

        # Handle non-ACGT characters
        non_acgt_positions = [m.start() for m in non_acgt_re.finditer(chunk_seq)]
        n_bad = len(non_acgt_positions)

        if n_bad > 0 and n_bad / chunk_len > 0.5:
            n_skipped += 1
            continue

        if n_bad > 0:
            chunk_seq = non_acgt_re.sub(
                lambda m: bases[_np.random.randint(4)], chunk_seq
            )

        try:
            with _torch.inference_mode():
                if reverse_complement:
                    seq_rc = str(_Seq(chunk_seq).reverse_complement())

                    # Try batched fwd+RC (batch=2, single model call)
                    if _batched_rc_ok:
                        try:
                            result_fwd, result_rc = _batched_fwd_rc_pass(chunk_seq, seq_rc)
                        except (_torch.cuda.OutOfMemoryError, RuntimeError) as batch_exc:
                            if isinstance(batch_exc, _torch.cuda.OutOfMemoryError) or "out of memory" in str(batch_exc).lower():
                                _torch.cuda.empty_cache()
                                _batched_rc_ok = False
                                print(f"[GPU {gpu_id}] Batched fwd+RC OOM, falling back to sequential")
                                result_fwd = _forward_pass(chunk_seq)
                                result_rc = _forward_pass(seq_rc)
                            else:
                                raise
                    else:
                        result_fwd = _forward_pass(chunk_seq)
                        result_rc = _forward_pass(seq_rc)

                    if not compute_logprobs:
                        chunk_entropy = 0.5 * (result_fwd + result_rc[::-1])
                        chunk_p_acgt = None
                        chunk_ll = None
                    else:
                        chunk_entropy = 0.5 * (result_fwd["entropy"] +
                                               result_rc["entropy"][::-1])
                        p_rc = result_rc["p_acgt"][::-1, :]
                        p_rc_complement = p_rc[:, [3, 2, 1, 0]]
                        chunk_p_acgt = 0.5 * (result_fwd["p_acgt"] + p_rc_complement)
                        chunk_ll = result_fwd["ll_next"]
                else:
                    result_fwd = _forward_pass(chunk_seq)
                    if not compute_logprobs:
                        chunk_entropy = result_fwd
                        chunk_p_acgt = None
                        chunk_ll = None
                    else:
                        chunk_entropy = result_fwd["entropy"]
                        chunk_p_acgt = result_fwd["p_acgt"]
                        chunk_ll = result_fwd["ll_next"]

            # NaN out non-ACGT positions
            if n_bad > 0:
                for pos in non_acgt_positions:
                    if pos < len(chunk_entropy):
                        chunk_entropy[pos] = _np.nan
                    if chunk_p_acgt is not None and pos < len(chunk_p_acgt):
                        chunk_p_acgt[pos] = _np.nan
                    if chunk_ll is not None and pos < len(chunk_ll):
                        chunk_ll[pos] = _np.nan

            # Write to shared memory (select layer for experimental stitch)
            local_s = core_s - s
            local_e = core_e - s
            if local_e > local_s and local_e <= len(chunk_entropy):
                if stitch_method != "core" and entropy_out_alt is not None:
                    target = entropy_out if chunk_idx % 2 == 0 else entropy_out_alt
                else:
                    target = entropy_out
                target[core_s:core_e] = chunk_entropy[local_s:local_e]
                n_scored += core_e - core_s

                if chunk_p_acgt is not None and p_acgt_out is not None:
                    if local_e <= len(chunk_p_acgt):
                        p_acgt_out[core_s:core_e] = chunk_p_acgt[local_s:local_e]
                if chunk_ll is not None and ll_next_out is not None:
                    if local_e <= len(chunk_ll):
                        ll_next_out[core_s:core_e] = chunk_ll[local_s:local_e]

        except (_torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if isinstance(exc, _torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                _torch.cuda.empty_cache()
                print(f"[GPU {gpu_id}] OOM on chunk {chunk_idx} (pos {s:,}-{e:,}), skipping")
                n_skipped += 1
            else:
                raise
        except Exception as exc:
            print(f"[GPU {gpu_id}] Error on chunk {chunk_idx}: {exc}")
            n_skipped += 1

        # Progress logging with ETA
        if (i + 1) % 10 == 0 or i == len(work_items) - 1:
            elapsed = _time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(work_items) - i - 1) / rate if rate > 0 else 0
            print(f"[GPU {gpu_id}] Chunk {i+1}/{len(work_items)} "
                  f"({rate:.2f} chunks/s, ETA {remaining/60:.1f} min)")

    elapsed = _time.time() - t0
    print(f"[GPU {gpu_id}] Done: {n_scored:,} positions scored, "
          f"{n_skipped} skipped, {elapsed:.1f}s total")

    # Close shared memory handles (don't unlink — main process does that)
    seq_shm.close()
    result_shm.close()
    if result_shm_alt is not None:
        result_shm_alt.close()
    if p_acgt_shm is not None:
        p_acgt_shm.close()
    if ll_next_shm is not None:
        ll_next_shm.close()


def score_chromosome_region_multigpu(
    sequence: str,
    n_gpus: int = None,
    model_name: str = "evo2_7b",
    max_chunk_len: int = MAX_CHUNK_LEN,
    chunk_overlap: int = CHUNK_OVERLAP,
    reverse_complement: bool = False,
    compute_logprobs: bool = False,
    logger: logging.Logger = None,
    stitch_method: str = "core",
    use_torch_compile: bool = False,
) -> Union[np.ndarray, dict]:
    """
    Multi-GPU data-parallel version of score_chromosome_region.

    Uses shared memory for zero-copy IPC: the sequence is placed in shared
    memory once, and workers write entropy values directly to a shared output
    array.  This eliminates the Manager-dict pickling overhead that was
    previously the second-largest time sink after the forward passes.

    For experimental stitch methods ("mean"/"min"), uses two shared entropy
    arrays (even/odd layers) to avoid race conditions between workers writing
    to overlapping positions. Adjacent chunks always have different parity,
    so they write to different layers.

    Args:
        sequence: DNA sequence string
        n_gpus: Number of GPUs (None = auto-detect all available)
        model_name: Evo2 model name
        max_chunk_len: Maximum bases per chunk
        chunk_overlap: Overlap between chunks
        reverse_complement: If True, average forward and RC entropy
        compute_logprobs: If True, also return P(ACGT) and LL_next
        logger: Logger instance
        stitch_method: Overlap handling ("core", "mean", "min")

    Returns:
        If compute_logprobs=False: Per-position entropy array (nats)
        If compute_logprobs=True: dict with "entropy", "p_acgt", "ll_next"
    """
    import torch.multiprocessing as mp
    from multiprocessing import shared_memory as shm

    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise ValueError("No GPUs available for multi-GPU scoring")

    L = len(sequence)
    rc_label = " (RC-averaged)" if reverse_complement else ""
    lp_label = " +logprobs" if compute_logprobs else ""
    if logger:
        logger.info(f"Multi-GPU scoring{rc_label}{lp_label}: {L:,} bp across {n_gpus} GPUs")

    if L == 0:
        empty = np.full(L, np.nan, dtype=np.float32)
        if not compute_logprobs:
            return empty
        return {"entropy": empty, "p_acgt": np.full((L, 4), np.nan, dtype=np.float32),
                "ll_next": empty}

    # Compute chunk boundaries (same logic as sequential version)
    step = max(1, max_chunk_len - chunk_overlap)
    chunk_starts = list(range(0, L, step))

    work_items_all = []
    for chunk_idx, s in enumerate(chunk_starts):
        e = min(L, s + max_chunk_len)
        if stitch_method == "core":
            core_s = s if s == 0 else s + chunk_overlap // 2
            core_e = e if e == L else e - chunk_overlap // 2
        else:
            core_s = s
            core_e = e
        core_s = min(core_s, core_e)
        work_items_all.append((chunk_idx, s, e, core_s, core_e))

    # Distribute round-robin across GPUs
    gpu_work = [[] for _ in range(n_gpus)]
    for i, item in enumerate(work_items_all):
        gpu_work[i % n_gpus].append(item)

    if logger:
        logger.info(f"  {len(work_items_all)} chunks distributed across {n_gpus} GPUs:")
        for gid in range(n_gpus):
            logger.info(f"    GPU {gid}: {len(gpu_work[gid])} chunks")

    # Set spawn method (required for CUDA)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # --- Shared memory setup ---
    # Sequence: share as bytes (avoids 50MB+ pickle per worker)
    seq_bytes = sequence.encode('ascii')
    seq_shm = shm.SharedMemory(create=True, size=len(seq_bytes))
    seq_shm.buf[:len(seq_bytes)] = seq_bytes

    # Results: shared float32 array, pre-filled with NaN
    result_shm = shm.SharedMemory(create=True, size=L * 4)  # float32 = 4 bytes
    entropy_shared = np.ndarray(L, dtype=np.float32, buffer=result_shm.buf)
    entropy_shared[:] = np.nan

    # Alternate layer for experimental stitch methods (even/odd chunk separation)
    result_shm_alt = None
    entropy_alt_shared = None
    if stitch_method != "core":
        result_shm_alt = shm.SharedMemory(create=True, size=L * 4)
        entropy_alt_shared = np.ndarray(L, dtype=np.float32, buffer=result_shm_alt.buf)
        entropy_alt_shared[:] = np.nan

    # Additional shared memory for logprobs
    p_acgt_shm = None
    ll_next_shm = None
    if compute_logprobs:
        p_acgt_shm = shm.SharedMemory(create=True, size=L * 4 * 4)  # [L, 4] float32
        p_acgt_shared = np.ndarray((L, 4), dtype=np.float32, buffer=p_acgt_shm.buf)
        p_acgt_shared[:] = np.nan

        ll_next_shm = shm.SharedMemory(create=True, size=L * 4)  # [L] float32
        ll_next_shared = np.ndarray(L, dtype=np.float32, buffer=ll_next_shm.buf)
        ll_next_shared[:] = np.nan

    shm_total = len(seq_bytes) + L * 4
    if result_shm_alt is not None:
        shm_total += L * 4
    if compute_logprobs:
        shm_total += L * 4 * 4 + L * 4
    if logger:
        logger.info(f"  Shared memory: {shm_total/1e6:.1f}MB total")

    # Spawn workers
    processes = []
    active_gpu_ids = []
    for gid in range(n_gpus):
        if not gpu_work[gid]:
            continue
        p = mp.Process(
            target=_chromosome_gpu_worker,
            args=(gid, gpu_work[gid], seq_shm.name, result_shm.name,
                  len(seq_bytes), L, model_name, reverse_complement,
                  compute_logprobs,
                  p_acgt_shm.name if p_acgt_shm else None,
                  ll_next_shm.name if ll_next_shm else None,
                  stitch_method,
                  result_shm_alt.name if result_shm_alt else None,
                  use_torch_compile),
        )
        processes.append(p)
        active_gpu_ids.append(gid)

    # Start each worker with its GPU ID in the environment so the module-level
    # code (lines 67-69) sets CUDA_VISIBLE_DEVICES before torch is imported
    # during spawn's reimport of this module.
    mgpu_start = time.time()
    for gid, p in zip(active_gpu_ids, processes):
        os.environ["_SCORE_CHROM_GPU_ID"] = str(gid)
        p.start()
    # Clean up env var in parent process
    os.environ.pop("_SCORE_CHROM_GPU_ID", None)

    for p in processes:
        p.join()

    if logger:
        mgpu_time = time.time() - mgpu_start
        logger.info(f"  All {len(processes)} workers completed in {mgpu_time:.2f}s "
                    f"({mgpu_time/60:.1f} min)")

    # Check for worker failures
    for i, p in enumerate(processes):
        if p.exitcode != 0:
            if logger:
                logger.warning(f"  Worker {i} exited with code {p.exitcode}")

    # Copy results from shared memory and combine layers for stitch methods
    if stitch_method == "mean" and entropy_alt_shared is not None:
        layer0 = np.array(entropy_shared, copy=True)
        layer1 = np.array(entropy_alt_shared, copy=True)
        mask0, mask1 = ~np.isnan(layer0), ~np.isnan(layer1)
        both = mask0 & mask1
        entropy = np.full(L, np.nan, dtype=np.float32)
        entropy[mask0 & ~mask1] = layer0[mask0 & ~mask1]
        entropy[~mask0 & mask1] = layer1[~mask0 & mask1]
        entropy[both] = 0.5 * (layer0[both] + layer1[both])
        if logger:
            logger.info(f"  Stitch 'mean': {int(both.sum()):,} positions averaged from 2 chunks")
    elif stitch_method == "min" and entropy_alt_shared is not None:
        layer0 = np.array(entropy_shared, copy=True)
        layer1 = np.array(entropy_alt_shared, copy=True)
        entropy = np.fmin(layer0, layer1)  # fmin ignores NaN
        if logger:
            n_both = int((~np.isnan(layer0) & ~np.isnan(layer1)).sum())
            logger.info(f"  Stitch 'min': lowest entropy selected at {n_both:,} overlap positions")
    else:
        entropy = np.array(entropy_shared, copy=True)
    result_dict = None
    if compute_logprobs:
        p_acgt = np.array(p_acgt_shared, copy=True)
        ll_next = np.array(ll_next_shared, copy=True)
        result_dict = {"entropy": entropy, "p_acgt": p_acgt, "ll_next": ll_next}

    n_scored = int(np.isfinite(entropy).sum())
    if logger:
        logger.info(f"  Result: {n_scored:,} positions scored, "
                    f"{np.isnan(entropy).sum():,} NaN positions")

    # Clean up shared memory
    seq_shm.close()
    seq_shm.unlink()
    result_shm.close()
    result_shm.unlink()
    if result_shm_alt is not None:
        result_shm_alt.close()
        result_shm_alt.unlink()
    if p_acgt_shm is not None:
        p_acgt_shm.close()
        p_acgt_shm.unlink()
    if ll_next_shm is not None:
        ll_next_shm.close()
        ll_next_shm.unlink()

    if compute_logprobs:
        return result_dict
    return entropy


# =============================================================================
# DROP DETECTION — imported from detection_methods.py
# =============================================================================
# Functions imported at top of file:
#   detect_drops_zscore, detect_drops_mad,
#   detect_rises_zscore, detect_rises_mad,
#   _rolling_mean, _cluster_and_pick_best


# =============================================================================
# REGION PAIRING
# =============================================================================

def pair_drops_and_rises(
    drops: List[Tuple[int, float]],
    rises: List[Tuple[int, float]],
    entropy: np.ndarray,
    chrom: str,
    genomic_offset: int,
    method: str,
    max_region_length: int = 50000
) -> List[DropRegion]:
    """Pair drop points with subsequent rise points to define complete regions."""
    regions = []

    rise_positions = [pos for pos, _ in rises]
    rise_scores = {pos: score for pos, score in rises}

    for drop_pos, drop_score in drops:
        subsequent_rises = [p for p in rise_positions if p > drop_pos]

        if subsequent_rises:
            rise_pos = subsequent_rises[0]
            rise_score = rise_scores[rise_pos]
        else:
            rise_pos = min(drop_pos + max_region_length, len(entropy) - 1)
            rise_score = 0.0

        region_length = rise_pos - drop_pos
        if region_length > max_region_length or region_length < 1:
            continue

        region_entropy = entropy[drop_pos:rise_pos]
        valid_entropy = region_entropy[~np.isnan(region_entropy)]
        if len(valid_entropy) == 0:
            continue

        region = DropRegion(
            chrom=chrom,
            drop_start=drop_pos,
            drop_end=rise_pos,
            region_length=region_length,
            method=method,
            start_confidence=abs(drop_score),
            end_confidence=abs(rise_score),
            mean_entropy=float(np.mean(valid_entropy)),
            min_entropy=float(np.min(valid_entropy)),
            genomic_start=genomic_offset + drop_pos + 1,
            genomic_end=genomic_offset + rise_pos + 1
        )
        regions.append(region)

    return regions


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def run_detection(
    entropy: np.ndarray,
    chrom: str,
    genomic_offset: int,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    mad_threshold: float = MAD_THRESHOLD,
    smooth_w: int = SMOOTH_W,
    min_separation: int = MIN_SEPARATION,
    logger: logging.Logger = None,
    detection_methods: Optional[List[str]] = None
) -> DetectionResult:
    """Run drop and rise detection.

    Z-score and MAD detection run in parallel using threads (they are
    CPU-bound numpy code that releases the GIL during array operations).

    Args:
        detection_methods: List of method names to run (default: ["zscore", "mad"]).
            Available: zscore, mad, derivative, window_mean_shift, cusum, local_baseline.
            Rise detection and region pairing are only performed for zscore and mad
            (the only methods with corresponding rise detectors).
    """
    if detection_methods is None:
        detection_methods = ["zscore", "mad"]

    result = DetectionResult()

    # Run zscore and mad detection in parallel (independent numpy operations)
    run_zscore = "zscore" in detection_methods
    run_mad = "mad" in detection_methods

    def _detect_zscore():
        drops = detect_drops_zscore(entropy, smooth_w, zscore_threshold, min_separation)
        rises = detect_rises_zscore(entropy, smooth_w, zscore_threshold, min_separation)
        return drops, rises

    def _detect_mad():
        drops = detect_drops_mad(entropy, smooth_w, mad_threshold, min_separation)
        rises = detect_rises_mad(entropy, smooth_w, mad_threshold, min_separation)
        return drops, rises

    if run_zscore and run_mad:
        if logger:
            logger.info("Running z-score and MAD detection in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_z = executor.submit(_detect_zscore)
            fut_m = executor.submit(_detect_mad)
            result.drops_zscore, result.rises_zscore = fut_z.result()
            result.drops_mad, result.rises_mad = fut_m.result()
        if logger:
            logger.info(f"  Z-score: {len(result.drops_zscore)} drops, "
                        f"{len(result.rises_zscore)} rises")
            logger.info(f"  MAD: {len(result.drops_mad)} drops, "
                        f"{len(result.rises_mad)} rises")
    else:
        if run_zscore:
            if logger:
                logger.info("Running z-score detection...")
            result.drops_zscore, result.rises_zscore = _detect_zscore()
            if logger:
                logger.info(f"  Z-score: {len(result.drops_zscore)} drops, "
                            f"{len(result.rises_zscore)} rises")
        if run_mad:
            if logger:
                logger.info("Running MAD detection...")
            result.drops_mad, result.rises_mad = _detect_mad()
            if logger:
                logger.info(f"  MAD: {len(result.drops_mad)} drops, "
                            f"{len(result.rises_mad)} rises")

    # Run any additional methods (drop-only, no rise pairing)
    extra_methods = [m for m in detection_methods if m not in ("zscore", "mad")]
    for method_name in extra_methods:
        if method_name not in DETECTION_METHODS:
            if logger:
                logger.warning(f"Unknown detection method '{method_name}', skipping")
            continue
        if logger:
            logger.info(f"Running {method_name} drop detection...")
        drops = run_detection_method(method_name, entropy,
                                     smooth_w=smooth_w, min_separation=min_separation)
        if logger:
            logger.info(f"  Found {len(drops)} drops")
        # Store in result as extra attribute for downstream access
        setattr(result, f"drops_{method_name}", drops)

    # Pair drops and rises into regions (also parallelized)
    if logger:
        logger.info("Pairing drops and rises into regions...")

    if run_zscore and run_mad:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_zr = executor.submit(pair_drops_and_rises,
                                     result.drops_zscore, result.rises_zscore,
                                     entropy, chrom, genomic_offset, "zscore")
            fut_mr = executor.submit(pair_drops_and_rises,
                                     result.drops_mad, result.rises_mad,
                                     entropy, chrom, genomic_offset, "mad")
            result.regions_zscore = fut_zr.result()
            result.regions_mad = fut_mr.result()
        if logger:
            logger.info(f"  Z-score: {len(result.regions_zscore)} regions")
            logger.info(f"  MAD: {len(result.regions_mad)} regions")
    else:
        if run_zscore:
            result.regions_zscore = pair_drops_and_rises(
                result.drops_zscore, result.rises_zscore,
                entropy, chrom, genomic_offset, "zscore"
            )
            if logger:
                logger.info(f"  Z-score: {len(result.regions_zscore)} regions")
        if run_mad:
            result.regions_mad = pair_drops_and_rises(
                result.drops_mad, result.rises_mad,
                entropy, chrom, genomic_offset, "mad"
            )
            if logger:
                logger.info(f"  MAD: {len(result.regions_mad)} regions")

    return result


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def _outpath(output_prefix, suffix, output_dir=None):
    """Build output file path. Clean name when output_dir is set."""
    if output_dir:
        return os.path.join(output_dir, suffix)
    return f"{output_prefix}.{suffix}"


def save_results(
    output_prefix: str,
    chrom: str,
    start: int,
    end: int,
    entropy: np.ndarray,
    result: DetectionResult,
    run_metadata: Dict[str, Any],
    logger: logging.Logger = None,
    logprobs_data: Optional[dict] = None,
    save_per_position_tsv: bool = False,
    output_dir: str = None,
):
    """Save all detection results to files.

    Args:
        logprobs_data: Optional dict with "p_acgt" and "ll_next" arrays.
        save_per_position_tsv: If True and logprobs_data provided, write
            a per-position TSV with P(A/C/G/T) and LL_next columns.
        output_dir: If set, write clean-named files (entropy.npz, etc.)
            into this directory instead of using {output_prefix}.xxx.
    """

    # 1. Save entropy array (+ logprobs arrays if available)
    entropy_file = _outpath(output_prefix, "entropy.npz", output_dir)
    if logger:
        logger.info(f"Saving entropy to: {entropy_file}")
    save_kwargs = dict(
        entropy=entropy,
        positions=np.arange(len(entropy)),
        chrom=chrom,
        start=start,
        end=end,
    )
    if logprobs_data is not None:
        save_kwargs["p_acgt"] = logprobs_data["p_acgt"]
        save_kwargs["ll_next"] = logprobs_data["ll_next"]
    np.savez_compressed(entropy_file, **save_kwargs)

    # 1b. Optional per-position TSV (can be large: ~2-3GB for 50M positions)
    if save_per_position_tsv and logprobs_data is not None:
        tsv_file = _outpath(output_prefix, "per_position.tsv", output_dir)
        if logger:
            logger.info(f"Saving per-position TSV to: {tsv_file} "
                        f"({len(entropy):,} rows)")
        p_acgt = logprobs_data["p_acgt"]
        ll_next = logprobs_data["ll_next"]
        with open(tsv_file, "w") as f:
            f.write("Pos\tGenomic_Pos\tEntropy(nats)\t"
                    "P(A)\tP(C)\tP(G)\tP(T)\tLL_next(nats)\n")
            for i in range(len(entropy)):
                gpos = start + i + 1  # 1-based genomic coordinate
                f.write(f"{i}\t{gpos}\t{entropy[i]:.6f}\t"
                        f"{p_acgt[i, 0]:.6f}\t{p_acgt[i, 1]:.6f}\t"
                        f"{p_acgt[i, 2]:.6f}\t{p_acgt[i, 3]:.6f}\t"
                        f"{ll_next[i]:.6f}\n")
        if logger:
            logger.info(f"  Per-position TSV written ({os.path.getsize(tsv_file)/1e6:.0f}MB)")

    # 2. Save drop boundaries (main output)
    boundaries_file = _outpath(output_prefix, "drop_boundaries.tsv", output_dir)
    if logger:
        logger.info(f"Saving drop boundaries to: {boundaries_file}")

    all_regions = result.regions_zscore + result.regions_mad
    all_regions.sort(key=lambda r: r.drop_start)

    with open(boundaries_file, "w") as f:
        # Header
        f.write("# Drop boundary detection results\n")
        f.write("# Generated: {}\n".format(datetime.now().isoformat()))
        f.write("# Chromosome: {} (positions {}-{})\n".format(chrom, start, end))
        f.write("#\n")
        f.write("chrom\tdrop_start\tdrop_end\tgenomic_start\tgenomic_end\t")
        f.write("region_length\tmethod\tstart_confidence\tend_confidence\t")
        f.write("mean_entropy\tmin_entropy\n")

        for region in all_regions:
            f.write(f"{region.chrom}\t{region.drop_start}\t{region.drop_end}\t")
            f.write(f"{region.genomic_start}\t{region.genomic_end}\t")
            f.write(f"{region.region_length}\t{region.method}\t")
            f.write(f"{region.start_confidence:.4f}\t{region.end_confidence:.4f}\t")
            f.write(f"{region.mean_entropy:.6f}\t{region.min_entropy:.6f}\n")

    # 3. Save individual drops
    drops_file = _outpath(output_prefix, "drops.tsv", output_dir)
    if logger:
        logger.info(f"Saving drops to: {drops_file}")

    with open(drops_file, "w") as f:
        f.write("# All detected drop points (entropy starts decreasing)\n")
        f.write("chrom\tposition\tgenomic_position\tmethod\tconfidence\n")

        for pos, score in result.drops_zscore:
            f.write(f"{chrom}\t{pos}\t{start + pos + 1}\tzscore\t{abs(score):.4f}\n")
        for pos, score in result.drops_mad:
            f.write(f"{chrom}\t{pos}\t{start + pos + 1}\tmad\t{abs(score):.4f}\n")

    # 4. Save individual rises
    rises_file = _outpath(output_prefix, "rises.tsv", output_dir)
    if logger:
        logger.info(f"Saving rises to: {rises_file}")

    with open(rises_file, "w") as f:
        f.write("# All detected rise points (entropy starts increasing)\n")
        f.write("chrom\tposition\tgenomic_position\tmethod\tconfidence\n")

        for pos, score in result.rises_zscore:
            f.write(f"{chrom}\t{pos}\t{start + pos + 1}\tzscore\t{abs(score):.4f}\n")
        for pos, score in result.rises_mad:
            f.write(f"{chrom}\t{pos}\t{start + pos + 1}\tmad\t{abs(score):.4f}\n")

    # 5. Save summary metadata
    summary_file = _outpath(output_prefix, "summary.json", output_dir)
    if logger:
        logger.info(f"Saving summary to: {summary_file}")

    summary = {
        **run_metadata,
        "results": {
            "zscore": {
                "n_drops": len(result.drops_zscore),
                "n_rises": len(result.rises_zscore),
                "n_regions": len(result.regions_zscore),
            },
            "mad": {
                "n_drops": len(result.drops_mad),
                "n_rises": len(result.rises_mad),
                "n_regions": len(result.regions_mad),
            },
            "total_regions": len(all_regions),
        },
        "entropy_stats": {
            "mean": float(np.nanmean(entropy)),
            "std": float(np.nanstd(entropy)),
            "min": float(np.nanmin(entropy)),
            "max": float(np.nanmax(entropy)),
            "nan_fraction": float(np.isnan(entropy).sum() / len(entropy)),
        }
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if logger:
        logger.info(f"Results saved. Total regions detected: {len(all_regions)}")


# =============================================================================
# SEQUENCE LOADING
# =============================================================================

def load_chromosome_sequence(
    fasta_path: str,
    chrom: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    logger: logging.Logger = None
) -> Tuple[str, int, int]:
    """
    Load chromosome sequence from FASTA file.

    Uses pysam for O(1) indexed access if a .fai index exists (much faster
    for large genomes). Falls back to BioPython SeqIO.parse if pysam is
    unavailable or the index is missing.

    Args:
        fasta_path: Path to genome FASTA
        chrom: Chromosome name (RefSeq accession or common name)
        start: Start position (0-based, inclusive). None = beginning
        end: End position (0-based, exclusive). None = end of chromosome
        logger: Logger instance

    Returns:
        Tuple of (sequence, actual_start, actual_end)
    """
    # Map common names to RefSeq
    if chrom in CHROM_MAP:
        chrom_id = CHROM_MAP[chrom]
    else:
        chrom_id = chrom

    if logger:
        logger.info(f"Loading chromosome {chrom_id} from {fasta_path}")

    # Try pysam for indexed O(1) access (requires .fai index)
    try:
        import pysam
        fai_path = fasta_path + ".fai"
        if os.path.exists(fai_path):
            if logger:
                logger.info("  Using pysam indexed FASTA (fast path)")
            fa = pysam.FastaFile(fasta_path)
            try:
                full_length = fa.get_reference_length(chrom_id)
                actual_start = start if start is not None else 0
                actual_end = end if end is not None else full_length
                if actual_start < 0:
                    actual_start = 0
                if actual_end > full_length:
                    actual_end = full_length
                if actual_start >= actual_end:
                    raise ValueError(f"Invalid range: start={actual_start}, end={actual_end}")
                sequence = fa.fetch(chrom_id, actual_start, actual_end).upper()
                if logger:
                    logger.info(f"Chromosome length: {full_length:,} bp")
                    logger.info(f"Extracted region: {actual_start:,}-{actual_end:,} ({len(sequence):,} bp)")
                return sequence, actual_start, actual_end
            finally:
                fa.close()
        else:
            if logger:
                logger.info("  No .fai index found, falling back to BioPython")
    except ImportError:
        if logger:
            logger.info("  pysam not available, using BioPython (slower)")

    # Fallback: BioPython sequential scan
    seq_record = None
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == chrom_id or record.id.startswith(chrom_id):
            seq_record = record
            break

    if seq_record is None:
        raise ValueError(f"Chromosome {chrom_id} not found in FASTA")

    full_length = len(seq_record.seq)
    if logger:
        logger.info(f"Chromosome length: {full_length:,} bp")

    # Handle start/end
    actual_start = start if start is not None else 0
    actual_end = end if end is not None else full_length

    # Validate
    if actual_start < 0:
        actual_start = 0
    if actual_end > full_length:
        actual_end = full_length
    if actual_start >= actual_end:
        raise ValueError(f"Invalid range: start={actual_start}, end={actual_end}")

    # Extract sequence
    sequence = str(seq_record.seq[actual_start:actual_end]).upper()

    if logger:
        logger.info(f"Extracted region: {actual_start:,}-{actual_end:,} ({len(sequence):,} bp)")

    return sequence, actual_start, actual_end


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score chromosome regions and detect entropy drop boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score human chromosome 21 (smallest autosome)
  python score_chromosome.py --chrom chr21 --output_prefix chr21_test

  # Score a 1Mb region of chromosome 1
  python score_chromosome.py --chrom chr1 --start 1000000 --end 2000000 --output_prefix chr1_region

  # Use custom detection thresholds
  python score_chromosome.py --chrom chr21 --zscore_threshold 3.0 --mad_threshold 3.5

  # Multi-GPU with auto chunk size (fastest forward-only)
  python score_chromosome.py --chrom chr22 --output_prefix chr22_fast --n_gpus 0 --auto_chunk_size

  # Multi-GPU with RC-averaged entropy (better biological signal)
  python score_chromosome.py --chrom chr22 --output_prefix chr22_rc --n_gpus 0 --auto_chunk_size --rc_average

  # Full scoring with fused logprobs (zero extra cost)
  python score_chromosome.py --chrom chr22 --output_prefix chr22_full \
      --n_gpus 0 --auto_chunk_size --rc_average --compute_logprobs

  # Also save per-position TSV (large file, ~2-3GB for chr22)
  python score_chromosome.py --chrom chr22 --output_prefix chr22_full \
      --n_gpus 0 --auto_chunk_size --rc_average --compute_logprobs --save_per_position_tsv
        """
    )

    # Required arguments
    parser.add_argument("--chrom", required=True,
                        help="Chromosome name (e.g., chr1, NC_000001.11, 1)")
    parser.add_argument("--output_prefix", default=None,
                        help="(Deprecated, ignored) Old flat-mode prefix.")

    # Optional region specification
    parser.add_argument("--start", type=int, default=None,
                        help="Start position (0-based). Default: beginning of chromosome")
    parser.add_argument("--end", type=int, default=None,
                        help="End position (0-based, exclusive). Default: end of chromosome")

    # File paths
    parser.add_argument("--fasta", type=str, default=DEFAULT_FASTA,
                        help=f"Path to genome FASTA (default: {DEFAULT_FASTA})")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Root results directory (default: ./results)")

    # Detection parameters
    parser.add_argument("--zscore_threshold", type=float, default=ZSCORE_THRESHOLD,
                        help=f"Z-score threshold for detection (default: {ZSCORE_THRESHOLD})")
    parser.add_argument("--mad_threshold", type=float, default=MAD_THRESHOLD,
                        help=f"MAD threshold for detection (default: {MAD_THRESHOLD})")
    parser.add_argument("--smooth_w", type=int, default=SMOOTH_W,
                        help=f"Smoothing window size (default: {SMOOTH_W})")
    parser.add_argument("--min_separation", type=int, default=MIN_SEPARATION,
                        help=f"Minimum separation between detections (default: {MIN_SEPARATION})")
    parser.add_argument("--detection_methods", type=str, default="zscore,mad",
                        help="Comma-separated detection methods to run. "
                             "Available: zscore, mad, derivative, window_mean_shift, "
                             "cusum, local_baseline (default: zscore,mad)")

    # Scoring parameters
    parser.add_argument("--max_chunk_len", type=int, default=MAX_CHUNK_LEN,
                        help=f"Maximum chunk length for scoring (default: {MAX_CHUNK_LEN})")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP,
                        help=f"Chunk overlap (default: {CHUNK_OVERLAP})")
    parser.add_argument("--stitch_method", type=str, default="core",
                        choices=["core", "mean", "min"],
                        help="How to handle overlapping chunk regions. "
                             "'core': use only the center of each chunk (default). "
                             "'mean': average entropy values in overlap zones. "
                             "'min': take lowest entropy (highest confidence) in overlaps.")

    # Model parameters
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device (default: cuda:0)")

    # Multi-GPU and chunk optimization
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs for data-parallel scoring. "
                             "Default: 1 (sequential). Set >1 to distribute chunks "
                             "across multiple GPUs (one model per GPU). "
                             "Set to 0 to auto-detect all available GPUs.")
    parser.add_argument("--auto_chunk_size", action="store_true",
                        help="Automatically find the largest chunk size that fits "
                             "in GPU memory. Overrides --max_chunk_len.")
    parser.add_argument("--target_chunk_size", type=int, default=100_000,
                        help="Starting chunk size for auto-probing (default: 100000). "
                             "Only used with --auto_chunk_size.")

    # Entropy mode
    parser.add_argument("--rc_average", action="store_true", default=False,
                        help="Enable RC-averaged entropy: forward + reverse complement "
                             "averaged for strand-independent signal. Uses 2 forward "
                             "passes per chunk instead of 1.")

    # Logprobs (fused into same forward pass — zero extra cost)
    parser.add_argument("--compute_logprobs", action="store_true", default=False,
                        help="Compute next-token log-likelihood and P(A/C/G/T) "
                             "from the same forward pass (zero extra cost). "
                             "Results saved in entropy.npz.")
    parser.add_argument("--save_per_position_tsv", action="store_true", default=False,
                        help="Save per-position TSV with P(A/C/G/T) and LL_next. "
                             "Warning: ~2-3GB for 50M positions. "
                             "Requires --compute_logprobs.")

    # Performance options
    parser.add_argument("--torch_compile", action="store_true", default=False,
                        help="Apply torch.compile to the model for fused kernels. "
                             "Can give 10-30%% speedup on the forward pass. "
                             "Falls back gracefully if compilation fails.")

    # Other options
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    args = parser.parse_args()

    # Create output directory and setup logging
    # Build flags descriptor for the run directory name
    flags = []
    if args.rc_average:
        flags.append("rc")
    if args.compute_logprobs:
        flags.append("logprobs")
    if args.stitch_method != "core":
        flags.append(f"stitch_{args.stitch_method}")
    gpu_label = f"{args.n_gpus}gpu" if args.n_gpus > 0 else "autogpu"
    flags.append(gpu_label)
    flags_str = "_".join(flags)

    run_dir = build_run_dir(args.output_dir, args.chrom, "scoring", flags_str)
    organized_data_dir = os.path.join(run_dir, "data")
    for subdir in ("data", "logs"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    output_prefix = os.path.join(run_dir, "data", "scoring")
    log_file = os.path.join(run_dir, "logs", "scoring.log")
    logger = setup_logging(args.log_level, log_file=log_file)
    logger.info("=" * 70)
    logger.info("CHROMOSOME SCORING AND DROP BOUNDARY DETECTION")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")

    # Record run metadata
    run_metadata = {
        "script": "score_chromosome.py",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "chrom": args.chrom,
            "start": args.start,
            "end": args.end,
            "zscore_threshold": args.zscore_threshold,
            "mad_threshold": args.mad_threshold,
            "smooth_w": args.smooth_w,
            "min_separation": args.min_separation,
            "detection_methods": args.detection_methods,
            "max_chunk_len": args.max_chunk_len,
            "chunk_overlap": args.chunk_overlap,
            "device": args.device,
            "n_gpus": args.n_gpus,
            "auto_chunk_size": args.auto_chunk_size,
            "rc_average": args.rc_average,
            "compute_logprobs": args.compute_logprobs,
            "stitch_method": args.stitch_method,
        }
    }

    # Load chromosome sequence
    logger.info("-" * 70)
    logger.info("STEP 1: Loading chromosome sequence")
    logger.info("-" * 70)

    wall_start = time.time()
    step1_start = time.time()
    try:
        sequence, actual_start, actual_end = load_chromosome_sequence(
            args.fasta, args.chrom, args.start, args.end, logger
        )
    except Exception as e:
        logger.error(f"Failed to load chromosome: {e}")
        sys.exit(1)
    step1_time = time.time() - step1_start
    logger.info(f"  STEP 1 completed in {step1_time:.2f}s")

    # Resolve chromosome name
    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)

    run_metadata["resolved_chrom"] = chrom_id
    run_metadata["actual_start"] = actual_start
    run_metadata["actual_end"] = actual_end
    run_metadata["sequence_length"] = len(sequence)

    # Resolve GPU count
    n_gpus = args.n_gpus
    if n_gpus == 0:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Auto-detected {n_gpus} GPU(s)")

    # Load model and prepare for scoring
    logger.info("-" * 70)
    logger.info("STEP 2: Loading Evo2 model")
    logger.info("-" * 70)

    step2_start = time.time()
    if n_gpus > 1:
        logger.info(f"Multi-GPU mode ({n_gpus} GPUs): workers will load their own models")
        if torch.cuda.device_count() < n_gpus:
            logger.warning(f"Requested {n_gpus} GPUs but only "
                           f"{torch.cuda.device_count()} available. "
                           f"Falling back to {torch.cuda.device_count()} GPUs.")
            n_gpus = torch.cuda.device_count()

    model = None
    ACGT_IDS = None
    device = args.device
    step2b_time = None

    if n_gpus > 1 and args.auto_chunk_size:
        # Probe chunk size in a SPAWNED SUBPROCESS so that the module-level
        # CUDA_VISIBLE_DEVICES trick (lines 67-69) takes effect BEFORE torch
        # is imported.  Setting the env var in the parent process after torch
        # is already imported has NO EFFECT — the probe model would shard
        # across all GPUs, overestimating the safe chunk size for single-GPU
        # workers and causing OOM.
        logger.info("-" * 70)
        logger.info("STEP 2b: Auto-detecting maximum chunk size "
                     "(single-GPU subprocess probe)")
        logger.info("-" * 70)
        step2b_start = time.time()

        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        os.environ["_SCORE_CHROM_GPU_ID"] = "0"
        probe_proc = ctx.Process(
            target=_probe_chunk_size_worker,
            args=(args.target_chunk_size, "evo2_7b", result_q),
        )
        probe_proc.start()
        probe_proc.join()
        os.environ.pop("_SCORE_CHROM_GPU_ID", None)

        if not result_q.empty():
            args.max_chunk_len = result_q.get()
        else:
            logger.warning("  Probe subprocess failed — using default chunk size")
            args.max_chunk_len = MAX_CHUNK_LEN

        if args.max_chunk_len <= 0:
            raise RuntimeError("Probe could not fit any chunk size in GPU memory")

        step2b_time = time.time() - step2b_start
        logger.info(f"  Selected chunk size: {args.max_chunk_len:,} bp")
        logger.info(f"  STEP 2b completed in {step2b_time:.2f}s")
        run_metadata["parameters"]["max_chunk_len"] = args.max_chunk_len

    elif n_gpus > 1:
        # Multi-GPU without auto-probe: no model needed in main process
        logger.info("  Skipping main model load (workers will load their own)")

    else:
        # Single-GPU: load model in main process
        model = Evo2("evo2_7b")
        device = str(getattr(model, "device", args.device))
        if hasattr(model, "eval"):
            model.eval()
        elif hasattr(model, "model"):
            model.model.eval()
        logger.info(f"  Model loaded. Device: {device}")

        # Try torch.compile for fused kernels (10-30% speedup if it works)
        if args.torch_compile:
            try:
                inner = model.model if hasattr(model, "model") else model
                compiled = torch.compile(inner, mode="default")
                if hasattr(model, "model"):
                    model.model = compiled
                else:
                    model = compiled
                logger.info("  torch.compile applied successfully")
            except Exception as ex:
                logger.warning(f"  torch.compile failed ({ex}), continuing without it")

        ACGT_IDS = get_acgt_token_ids(model, device)

        # Auto-probe on single GPU (model is already on 1 GPU)
        if args.auto_chunk_size:
            logger.info("-" * 70)
            logger.info("STEP 2b: Auto-detecting maximum chunk size")
            logger.info("-" * 70)
            step2b_start = time.time()
            args.max_chunk_len = find_max_chunk_size(
                model, ACGT_IDS, device,
                target_size=args.target_chunk_size,
                logger=logger,
            )
            step2b_time = time.time() - step2b_start
            logger.info(f"  Selected chunk size: {args.max_chunk_len:,} bp")
            logger.info(f"  STEP 2b completed in {step2b_time:.2f}s")
            run_metadata["parameters"]["max_chunk_len"] = args.max_chunk_len

    step2_time = time.time() - step2_start
    logger.info(f"  STEP 2 completed in {step2_time:.2f}s")

    # Score sequence
    logger.info("-" * 70)
    rc_label = " (RC-averaged)" if args.rc_average else ""
    stitch_label = f" [stitch={args.stitch_method}]" if args.stitch_method != "core" else ""
    logger.info(f"STEP 3: Scoring chromosome sequence{rc_label}{stitch_label}")
    logger.info("-" * 70)

    step3_start = time.time()
    logprobs_data = None
    if n_gpus > 1:
        logger.info(f"Using MULTI-GPU scoring ({n_gpus} GPUs, "
                     f"chunk_size={args.max_chunk_len:,} bp)...")
        scoring_result = score_chromosome_region_multigpu(
            sequence,
            n_gpus=n_gpus,
            model_name="evo2_7b",
            max_chunk_len=args.max_chunk_len,
            chunk_overlap=args.chunk_overlap,
            reverse_complement=args.rc_average,
            compute_logprobs=args.compute_logprobs,
            logger=logger,
            stitch_method=args.stitch_method,
            use_torch_compile=args.torch_compile,
        )
    else:
        scoring_result = score_chromosome_region(
            sequence, model, ACGT_IDS, device,
            max_chunk_len=args.max_chunk_len,
            chunk_overlap=args.chunk_overlap,
            reverse_complement=args.rc_average,
            compute_logprobs=args.compute_logprobs,
            logger=logger,
            stitch_method=args.stitch_method,
        )

    # Unpack result (dict if compute_logprobs, else ndarray)
    if isinstance(scoring_result, dict):
        entropy = scoring_result["entropy"]
        logprobs_data = scoring_result
    else:
        entropy = scoring_result

    step3_time = time.time() - step3_start

    n_scored = int(np.isfinite(entropy).sum())
    logger.info(f"Scoring complete. Entropy shape: {entropy.shape}")
    logger.info(f"  Mean entropy: {np.nanmean(entropy):.4f} nats")
    logger.info(f"  Std entropy: {np.nanstd(entropy):.4f} nats")
    logger.info(f"  NaN fraction: {np.isnan(entropy).sum() / len(entropy):.4%}")
    logger.info(f"  STEP 3 completed in {step3_time:.2f}s "
                f"({n_scored / step3_time:.0f} positions/s)")

    # Run detection
    logger.info("-" * 70)
    logger.info("STEP 4: Running drop and rise detection")
    logger.info("-" * 70)

    step4_start = time.time()
    det_methods = [m.strip() for m in args.detection_methods.split(",")]
    result = run_detection(
        entropy, chrom_id, actual_start,
        zscore_threshold=args.zscore_threshold,
        mad_threshold=args.mad_threshold,
        smooth_w=args.smooth_w,
        min_separation=args.min_separation,
        logger=logger,
        detection_methods=det_methods
    )
    step4_time = time.time() - step4_start
    logger.info(f"  STEP 4 completed in {step4_time:.2f}s")

    # Save results
    logger.info("-" * 70)
    logger.info("STEP 5: Saving results")
    logger.info("-" * 70)

    # Add timing data to metadata before saving
    total_wall_time = time.time() - wall_start
    run_metadata["timing"] = {
        "step1_load_sequence_s": round(step1_time, 2),
        "step2_load_model_s": round(step2_time, 2),
        "step2b_auto_chunk_s": round(step2b_time, 2) if step2b_time is not None else None,
        "step3_scoring_s": round(step3_time, 2),
        "step4_detection_s": round(step4_time, 2),
        "step3_throughput_bp_per_s": round(n_scored / step3_time, 1) if step3_time > 0 else None,
        "n_gpus_used": n_gpus,
    }

    step5_start = time.time()
    save_results(
        output_prefix, chrom_id, actual_start, actual_end,
        entropy, result, run_metadata, logger,
        logprobs_data=logprobs_data,
        save_per_position_tsv=args.save_per_position_tsv,
        output_dir=organized_data_dir,
    )
    step5_time = time.time() - step5_start
    logger.info(f"  STEP 5 completed in {step5_time:.2f}s")

    # Finalize wall time (include save step)
    total_wall_time = time.time() - wall_start
    run_metadata["timing"]["step5_save_s"] = round(step5_time, 2)
    run_metadata["timing"]["total_wall_s"] = round(total_wall_time, 2)

    # Summary
    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.1f} min)")
    logger.info(f"  Step 1 (load sequence):  {step1_time:.2f}s")
    logger.info(f"  Step 2 (load model):     {step2_time:.2f}s")
    if step2b_time is not None:
        logger.info(f"  Step 2b (auto chunk):    {step2b_time:.2f}s")
    logger.info(f"  Step 3 (scoring):        {step3_time:.2f}s "
                f"({n_scored/step3_time:.0f} bp/s, {n_gpus} GPU(s))")
    logger.info(f"  Step 4 (detection):      {step4_time:.2f}s")
    logger.info(f"  Step 5 (save):           {step5_time:.2f}s")
    logger.info(f"Output files:")
    logger.info(f"  - {output_prefix}.entropy.npz (entropy array"
                f"{' + P(ACGT) + LL_next' if logprobs_data else ''})")
    logger.info(f"  - {output_prefix}.drop_boundaries.tsv (main results)")
    logger.info(f"  - {output_prefix}.drops.tsv (all drops)")
    logger.info(f"  - {output_prefix}.rises.tsv (all rises)")
    logger.info(f"  - {output_prefix}.summary.json (metadata)")
    if args.save_per_position_tsv and logprobs_data:
        logger.info(f"  - {output_prefix}.per_position.tsv (per-position logprobs)")
    logger.info(f"  - {log_file} (this log)")
    logger.info("")
    logger.info(f"Total regions detected:")
    logger.info(f"  - Z-score: {len(result.regions_zscore)}")
    logger.info(f"  - MAD: {len(result.regions_mad)}")
    logger.info(f"  - Combined: {len(result.regions_zscore) + len(result.regions_mad)}")

    # Write COMPLETED sentinel (must be the very last action)
    write_completed(run_dir, "score_chromosome.py", total_wall_time)
    logger.info(f"  COMPLETED sentinel written to {run_dir}/COMPLETED")


if __name__ == "__main__":
    main()
