#!/usr/bin/env python3
"""
run_sae_fast.py  (optimized copy of run_sae_on_chromosome_drops.py)

================================================================================
OVERVIEW
================================================================================
End-to-end pipeline that connects score_chromosome.py outputs to SAE analysis.

Takes the detected high-confidence drop regions from chromosome scoring,
extracts the corresponding sequences, runs them through Evo2's Sparse
Autoencoder (layer 26, 32K features), and generates visualizations showing
which biological "tracks" are triggered in those regions.

Built directly on the patterns from:
    evo2/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb

Uses the same:
    - ObservableEvo2 model wrapper
    - BatchTopKTiedSAE sparse autoencoder
    - get_feature_ts() feature extraction
    - Stacked per-feature line plot visualization
    - GenBank annotation overlay with ANNOTATION_COLORS

================================================================================
INPUTS
================================================================================
From score_chromosome.py:
  - <prefix>.drop_boundaries.tsv   (paired drop-rise regions)
  - <prefix>.entropy.npz           (per-position entropy array)

Additionally:
  - Chromosome FASTA file           (to extract region sequences)
  - GenBank file (optional)         (for annotation overlays)

================================================================================
OUTPUTS
================================================================================
<output_dir>/
    data/
        sae_results.tsv             - Per-region top features
        signature_features.tsv      - Features recurring across regions
        feature_matrices.npz        - Raw SAE feature matrices (for notebook)
    plots/
        region_<N>_features.png     - Figure 4g style feature plots (filled area + gene track)
        region_<N>_entropy.png      - Entropy + drop boundary markers
        signature_summary.png       - Cross-region signature bar chart
    sae_exploration.ipynb           - Interactive Jupyter notebook

================================================================================
USAGE
================================================================================
    # Basic usage (after running score_chromosome.py)
    python run_sae_on_chromosome_drops.py \\
        --boundaries chr21_test.drop_boundaries.tsv \\
        --entropy chr21_test.entropy.npz \\
        --fasta /path/to/genome.fna \\
        --chrom chr21

    # With GenBank annotation overlay
    python run_sae_on_chromosome_drops.py \\
        --boundaries chr21_test.drop_boundaries.tsv \\
        --entropy chr21_test.entropy.npz \\
        --fasta /path/to/genome.fna \\
        --chrom chr21 \\
        --genbank /path/to/chromosome.gb

    # With filtering
    python run_sae_on_chromosome_drops.py \\
        --boundaries chr21_test.drop_boundaries.tsv \\
        --entropy chr21_test.entropy.npz \\
        --fasta /path/to/genome.fna \\
        --chrom chr21 \\
        --min_confidence 3.0 \\
        --max_regions 50

================================================================================
"""

import os
import sys
import json
import argparse
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np

from results_utils import build_run_dir, write_completed, write_source, find_latest_completed

# Defer torch to avoid slow startup for --help
_torch_imported = False


def _import_torch():
    global torch, _torch_imported
    if not _torch_imported:
        import torch as _torch
        torch = _torch
        _torch_imported = True
    return torch


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("sae_chromosome")
    logger.setLevel(getattr(logging, log_level.upper()))

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default FASTA path (same as score_chromosome.py)
DEFAULT_FASTA = (
    "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/"
    "ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/"
    "GCF_000001405.26_GRCh38_genomic.fna"
)

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

# Annotation colors — directly from the SAE notebook
ANNOTATION_COLORS = {
    'CDS': 'white',
    'gene': 'gray',
    'mobile_element': 'green',
    'misc_feature': 'yellow',
    'rRNA': '#7AC8AC',
    'tRNA': '#662D91',
    'ncRNA': 'white',
    'Regulatory': 'red',
    'tmRNA': 'red',
}

# SAE analysis defaults
DEFAULT_MIN_CONFIDENCE = 2.5
DEFAULT_MAX_REGIONS = 0  # 0 = no cap (process ALL qualifying regions)
DEFAULT_PADDING = 500
DEFAULT_TOP_FEATURES_PER_REGION = 10
DEFAULT_N_PLOT_FEATURES = 10
DEFAULT_SIGNATURE_MIN_PREVALENCE = 0.3

# Known biological SAE features from Evo2 paper (Figure 4g).
# These are always plotted to track interpretable biology across regions.
# Maps feature_id -> (short_label, description)
KNOWN_BIO_FEATURES = {
    15680: ("CDS",        "coding regions"),
    28339: ("Intron",     "introns"),
    1050:  ("Exon start", "first base of exon following intron"),
    25666: ("Exon end",   "last base of exon followed by intron"),
    24278: ("Frameshift", "mutation-sensitive, frameshifts & premature stops"),
}


# =============================================================================
# GENBANK ANNOTATION — directly from the SAE notebook
# =============================================================================

def find_relevant_gb_annotations(
    records,
    window_start: int,
    window_size: int,
    valid_features: set = None,
    valid_qualifiers: set = None,
) -> List[List]:
    """
    Extract annotations from GenBank records within a specified window.

    Directly from: evo2/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb

    Args:
        records: List of GenBank records (from SeqIO.parse)
        window_start: Start position of window (int)
        window_size: Size of window (int)
        valid_features: Set of feature types to include
        valid_qualifiers: Set of qualifiers to extract

    Returns:
        List of annotations: [start, end, type, qualifiers_dict]
    """
    if valid_features is None:
        valid_features = {
            'CDS', 'gene', 'mobile_element', 'misc_feature',
            'rRNA', 'tRNA', 'ncRNA', 'Regulatory', 'tmRNA',
        }
    if valid_qualifiers is None:
        valid_qualifiers = {'gene', 'locus_id', 'product', 'mobile_element_type'}

    window_end = window_start + window_size
    annotations = []

    for record in records:
        for feature in record.features:
            # Skip features outside window
            if feature.location.end < window_start or feature.location.start > window_end:
                continue

            if feature.type in valid_features:
                # Calculate relative positions within window
                start = max(0, feature.location.start - window_start)
                end = min(window_size, feature.location.end - window_start)

                # Extract relevant qualifiers
                qualifiers = {
                    q: feature.qualifiers[q]
                    for q in valid_qualifiers
                    if q in feature.qualifiers
                }

                annotations.append([start, end, feature.type, qualifiers])

    return annotations


# =============================================================================
# SEQUENCE LOADING
# =============================================================================

def load_chromosome_sequence(
    fasta_path: str,
    chrom: str,
    logger: logging.Logger = None,
) -> str:
    """Load full chromosome sequence from FASTA file."""
    from Bio import SeqIO

    chrom_id = CHROM_MAP.get(chrom, chrom)

    if logger:
        logger.info(f"Loading chromosome {chrom_id} from {fasta_path}")

    seq_record = None
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == chrom_id or record.id.startswith(chrom_id):
            seq_record = record
            break

    if seq_record is None:
        raise ValueError(f"Chromosome {chrom_id} not found in FASTA")

    sequence = str(seq_record.seq).upper()
    if logger:
        logger.info(f"Loaded {len(sequence):,} bp")

    return sequence


# =============================================================================
# REGION EXTRACTION
# =============================================================================

def extract_region_sequences(
    regions: List[Dict[str, Any]],
    chromosome_seq: str,
    padding: int = DEFAULT_PADDING,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    Extract DNA sequences for each detected region, with padding.

    Returns:
        Regions list with added keys: 'seq', 'padded_start', 'padded_end',
        'drop_local_pos', 'rise_local_pos'
    """
    chrom_len = len(chromosome_seq)

    for i, region in enumerate(regions):
        padded_start = max(0, region['drop_start'] - padding)
        padded_end = min(chrom_len, region['drop_end'] + padding)

        seq = chromosome_seq[padded_start:padded_end]

        region['seq'] = seq
        region['padded_start'] = padded_start
        region['padded_end'] = padded_end
        region['drop_local_pos'] = region['drop_start'] - padded_start
        region['rise_local_pos'] = region['drop_end'] - padded_start

        if logger:
            logger.debug(
                f"  Region {i+1}: {region['drop_start']}-{region['drop_end']} "
                f"({region['region_length']} bp) padded to "
                f"{padded_start}-{padded_end} ({len(seq)} bp)"
            )

    if logger:
        logger.info(f"Extracted sequences for {len(regions)} regions")

    return regions


# =============================================================================
# REGION FILTERING — method-based filtering and overlap detection
# =============================================================================

def find_overlapping_regions(
    regions: List[Dict[str, Any]],
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """Find MAD regions overlapping zscore regions and merge them.

    Uses a sweep-line O(N log N) algorithm:
    1. Separate regions by method
    2. Sort both lists by start position
    3. Sweep through to find overlapping MAD+zscore pairs
    4. Merge overlapping pairs using union of coordinates

    Returns:
        List of merged regions (method='both'), sorted by combined confidence.
    """
    mad_regions = sorted(
        [r for r in regions if r['method'] == 'mad'],
        key=lambda r: r['drop_start']
    )
    zscore_regions = sorted(
        [r for r in regions if r['method'] == 'zscore'],
        key=lambda r: r['drop_start']
    )

    if logger:
        logger.info(f"Finding overlaps: {len(mad_regions)} MAD × "
                     f"{len(zscore_regions)} zscore regions")

    if not mad_regions or not zscore_regions:
        if logger:
            logger.warning("One method has no regions — no overlaps possible")
        return []

    merged = []
    j_start = 0  # pointer into zscore_regions

    for m in mad_regions:
        m_start, m_end = m['drop_start'], m['drop_end']

        for j in range(j_start, len(zscore_regions)):
            z = zscore_regions[j]
            z_start, z_end = z['drop_start'], z['drop_end']

            # zscore region is entirely before MAD region — advance j_start
            if z_end <= m_start:
                j_start = j + 1
                continue

            # zscore region is entirely after MAD region — stop
            if z_start >= m_end:
                break

            # Overlap found — merge
            union_start = min(m_start, z_start)
            union_end = max(m_end, z_end)
            combined_conf = max(m['start_confidence'], z['start_confidence'])

            merged_region = {
                'chrom': m['chrom'],
                'drop_start': union_start,
                'drop_end': union_end,
                'genomic_start': union_start,
                'genomic_end': union_end,
                'region_length': union_end - union_start,
                'method': 'both',
                'start_confidence': combined_conf,
                'end_confidence': max(m['end_confidence'], z['end_confidence']),
                'mean_entropy': min(m['mean_entropy'], z['mean_entropy']),
                'min_entropy': min(m['min_entropy'], z['min_entropy']),
                'mad_confidence': m['start_confidence'],
                'zscore_confidence': z['start_confidence'],
            }
            merged.append(merged_region)

    # Deduplicate overlapping merges (a single region may overlap multiple partners)
    if merged:
        merged.sort(key=lambda r: (r['drop_start'], r['drop_end']))
        deduped = [merged[0]]
        for r in merged[1:]:
            prev = deduped[-1]
            # If this merged region overlaps the previous, take the better one
            if r['drop_start'] < prev['drop_end']:
                if r['start_confidence'] > prev['start_confidence']:
                    deduped[-1] = r
            else:
                deduped.append(r)
        merged = deduped

    # Sort by combined confidence (highest first)
    merged.sort(key=lambda r: -r['start_confidence'])

    if logger:
        logger.info(f"Found {len(merged)} overlapping MAD+zscore regions")

    return merged


def stratified_sample_regions(
    regions: List[Dict[str, Any]],
    max_regions: int,
    n_bins: int = 10,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """Sample regions across the full confidence range using stratified bins.

    Instead of taking the top-N by confidence (which produces homogeneous
    clusters in t-SNE), this divides the confidence range into equal bins
    and samples proportionally from each bin.

    Args:
        regions: All regions, sorted by confidence (highest first).
        max_regions: Total number of regions to select.
        n_bins: Number of equal-width bins across the confidence range.
        logger: Optional logger.

    Returns:
        Stratified sample of regions, sorted by confidence (highest first).
    """
    if len(regions) <= max_regions:
        if logger:
            logger.info(f"Stratified sampling: only {len(regions)} regions, keeping all")
        return regions

    confidences = np.array([r['start_confidence'] for r in regions])
    conf_min, conf_max = confidences.min(), confidences.max()

    if conf_min == conf_max:
        if logger:
            logger.info("Stratified sampling: all regions have same confidence, using top-N")
        return regions[:max_regions]

    bin_edges = np.linspace(conf_min, conf_max, n_bins + 1)
    per_bin = max(1, max_regions // n_bins)
    sampled = []

    rng = np.random.RandomState(42)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        bin_indices = np.where(mask)[0]
        if len(bin_indices) == 0:
            continue
        n_take = min(per_bin, len(bin_indices))
        chosen = rng.choice(bin_indices, size=n_take, replace=False)
        sampled.extend(chosen)

    # If we have fewer than max_regions, fill from remaining
    sampled_set = set(sampled)
    if len(sampled) < max_regions:
        remaining = [i for i in range(len(regions)) if i not in sampled_set]
        rng.shuffle(remaining)
        sampled.extend(remaining[:max_regions - len(sampled)])

    # Truncate if we overshot
    sampled = sampled[:max_regions]

    result = [regions[i] for i in sorted(sampled)]
    # Re-sort by confidence (highest first) for consistent downstream behavior
    result.sort(key=lambda r: -r['start_confidence'])

    if logger:
        result_confs = [r['start_confidence'] for r in result]
        logger.info(f"Stratified sampling: {len(result)} regions from {n_bins} bins "
                     f"(conf range: {min(result_confs):.2f} - {max(result_confs):.2f})")

    return result


def filter_regions_by_method(
    regions: List[Dict[str, Any]],
    method_filter: str,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """Filter regions by detection method.

    Args:
        regions: All parsed regions (any method)
        method_filter: 'zscore', 'mad', or 'both' (overlapping only)
        logger: Optional logger

    Returns:
        Filtered list of regions, sorted by confidence (highest first).
    """
    if method_filter == 'both':
        filtered = find_overlapping_regions(regions, logger)
    elif method_filter in ('zscore', 'mad'):
        filtered = [r for r in regions if r['method'] == method_filter]
        filtered.sort(key=lambda r: -r['start_confidence'])
        if logger:
            logger.info(f"Filtered to {len(filtered)} {method_filter} regions")
    else:
        raise ValueError(f"Unknown method_filter: {method_filter!r} "
                         f"(expected 'zscore', 'mad', or 'both')")
    return filtered


def gtf_to_genbank_annotations(
    gtf_features: List[Dict[str, Any]],
    window_start: int,
    window_size: int,
) -> List[List]:
    """Convert GTF feature dicts to GenBank-style annotation lists.

    This adapter converts the output of load_annotation_features() into
    the [start, end, type, qualifiers] format expected by plot_region_features().

    The returned positions are relative to the window (0-based), matching
    the GenBank annotation convention used by find_relevant_gb_annotations().
    """
    # Expanded keep_types for GTF
    keep_types = {
        'CDS', 'gene', 'exon', 'mRNA', 'transcript',
        'five_prime_UTR', 'three_prime_UTR',
        'start_codon', 'stop_codon',
        'tRNA', 'rRNA', 'ncRNA',
    }

    window_end = window_start + window_size
    annotations = []

    for feat in gtf_features:
        if feat['feature_type'] not in keep_types:
            continue
        if feat['start'] >= window_end or feat['end_exclusive'] <= window_start:
            continue

        # Convert to window-relative coordinates
        rel_start = max(0, feat['start'] - window_start)
        rel_end = min(window_size, feat['end_exclusive'] - window_start)

        # Map GTF type to GenBank-compatible type for ANNOTATION_COLORS
        ftype = feat['feature_type']
        type_map = {
            'five_prime_UTR': 'misc_feature',
            'three_prime_UTR': 'misc_feature',
            'start_codon': 'CDS',
            'stop_codon': 'CDS',
            'mRNA': 'gene',
            'transcript': 'gene',
            'exon': 'CDS',
        }
        gb_type = type_map.get(ftype, ftype)

        qualifiers = {}
        if feat.get('name'):
            qualifiers['gene'] = [feat['name']]
        if feat.get('attributes', {}).get('product'):
            qualifiers['product'] = [feat['attributes']['product']]

        annotations.append([rel_start, rel_end, gb_type, qualifiers])

    return annotations


# =============================================================================
# SAE ANALYSIS — uses the same get_feature_ts pattern from the notebook
# =============================================================================

def run_sae_on_regions(
    regions: List[Dict[str, Any]],
    model,  # ObservableEvo2
    sae,    # BatchTopKTiedSAE
    top_n: int = DEFAULT_TOP_FEATURES_PER_REGION,
    batch_size: int = 1,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 200,
    extract_only: bool = False,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    Run SAE feature extraction on regions, with optional batching, checkpointing,
    and memory-efficient extract-only mode.

    Args:
        batch_size: Sequences per GPU forward pass. Regions within each
            checkpoint chunk are sorted by length before batching to minimize
            padding waste. Default 1 = original single-sequence behaviour.
        checkpoint_dir: Directory to write _checkpoint.npz / _checkpoint_meta.json.
            On re-run the function resumes from the last completed chunk.
        checkpoint_interval: How many regions to process before each checkpoint.
        extract_only: When True, feature_ts arrays are saved to the checkpoint
            file and then released from memory after each chunk, keeping peak
            RAM proportional to checkpoint_interval rather than total regions.
            Downstream callers must load feature_ts from disk via feature_matrices.npz
            if they need it (plotting is assumed skipped in this mode).
    """
    from sae_utils import get_feature_ts, get_feature_ts_batch, SAE_LAYER_NAME

    n = len(regions)
    # results[i] holds the dict for regions[i]; None = not yet processed
    results = [None] * n
    start_idx = 0

    # ---- checkpoint paths ----
    # In extract_only mode we write per-chunk feature files so that no chunk's
    # feature_ts is ever overwritten with None.  The meta file tracks progress.
    chk_meta = os.path.join(checkpoint_dir, '_checkpoint_meta.json') if checkpoint_dir else None

    def _chunk_npz_path(chunk_start: int, chunk_end: int) -> str:
        return os.path.join(checkpoint_dir, f'_chunk_{chunk_start:07d}_{chunk_end:07d}.npz')

    # ---- resume from existing checkpoint ----
    if chk_meta and os.path.exists(chk_meta):
        try:
            with open(chk_meta) as f:
                meta = json.load(f)
            n_done = meta['n_done']
            for i in range(n_done):
                results[i] = {
                    'region':          regions[i],
                    'feature_ts':      None,   # always None on resume (reloaded in step 6b)
                    'top_feature_idx': meta['top_feature_idx'][i],
                    'drop_features':   [tuple(x) for x in meta['drop_features'][i]],
                    'rise_features':   [tuple(x) for x in meta['rise_features'][i]],
                }
            start_idx = n_done
            if logger:
                logger.info(f"[checkpoint] Resumed: {n_done}/{n} regions already done")
        except Exception as exc:
            if logger:
                logger.warning(f"[checkpoint] Load failed ({exc}), starting fresh")
            results   = [None] * n
            start_idx = 0

    # ---- async checkpoint: overlap disk I/O with next GPU chunk ----
    _chk_thread: Optional[threading.Thread] = None

    def _save_checkpoint(chunk_start: int, chunk_end: int):
        """Fire-and-forget checkpoint — overlaps with the next GPU chunk.

        Feature matrices for this chunk go to a per-chunk NPZ file so that
        cleared (None) regions from earlier chunks are never overwritten.
        Cumulative metadata goes to the shared JSON for resume logic.
        """
        nonlocal _chk_thread
        if not checkpoint_dir:
            return

        # Wait for any in-flight write before starting a new one
        if _chk_thread is not None:
            _chk_thread.join()

        # Per-chunk feature snapshot (only current chunk, all non-None)
        chunk_matrix = {
            f'region_{i}': results[i]['feature_ts']
            for i in range(chunk_start, chunk_end)
            if results[i] is not None and results[i].get('feature_ts') is not None
        }

        # Cumulative metadata snapshot (all completed regions so far)
        n_done = chunk_end
        meta_snapshot = {
            'n_done':          n_done,
            'top_feature_idx': [results[i]['top_feature_idx'] for i in range(n_done)],
            'drop_features':   [results[i]['drop_features']   for i in range(n_done)],
            'rise_features':   [results[i]['rise_features']   for i in range(n_done)],
        }

        chunk_path = _chunk_npz_path(chunk_start, chunk_end)

        def _write():
            os.makedirs(checkpoint_dir, exist_ok=True)
            if chunk_matrix:
                np.savez_compressed(chunk_path, **chunk_matrix)
            with open(chk_meta, 'w') as f:
                json.dump(meta_snapshot, f)
            if logger:
                logger.info(f"[checkpoint] Saved chunk {chunk_start}-{chunk_end} / {n}")

        _chk_thread = threading.Thread(target=_write, daemon=True)
        _chk_thread.start()

    # ---- helper: build result dict for one region ----
    def _make_result(i: int, feature_ts: np.ndarray) -> Dict[str, Any]:
        region = regions[i]
        total  = feature_ts.sum(axis=0)
        top_idx = np.argsort(total)[::-1][:top_n]
        top_idx = top_idx[total[top_idx] > 0]
        return {
            'region':          region,
            'feature_ts':      feature_ts,
            'top_feature_idx': top_idx.tolist(),
            'drop_features':   _get_top_features_at_pos(feature_ts, region['drop_local_pos'], top_n),
            'rise_features':   _get_top_features_at_pos(feature_ts, region['rise_local_pos'], top_n),
        }

    # ---- process in checkpoint-sized chunks ----
    # Each chunk is processed completely before checkpointing, giving a clean
    # consecutive prefix for resume. Within each chunk, regions are sorted by
    # sequence length to minimise padding when batch_size > 1.
    for chunk_start in range(start_idx, n, checkpoint_interval):
        chunk_end     = min(chunk_start + checkpoint_interval, n)
        chunk_indices = list(range(chunk_start, chunk_end))

        if batch_size > 1:
            # Sort by seq length within chunk for efficient padding
            chunk_sorted = sorted(chunk_indices, key=lambda i: len(regions[i]['seq']))

            for b in range(0, len(chunk_sorted), batch_size):
                sub   = chunk_sorted[b : b + batch_size]
                seqs  = [regions[i]['seq'] for i in sub]
                lens  = [len(s) for s in seqs]
                done_so_far = chunk_start + b
                if logger:
                    logger.info(
                        f"[SAE] {done_so_far+1}-{done_so_far+len(sub)}/{n}  "
                        f"({len(sub)} seqs, {min(lens)}-{max(lens)} bp)"
                    )
                fts_list = get_feature_ts_batch(model, sae, seqs, SAE_LAYER_NAME)
                for orig_i, fts in zip(sub, fts_list):
                    results[orig_i] = _make_result(orig_i, fts)
        else:
            for i in chunk_indices:
                region = regions[i]
                if logger:
                    logger.info(
                        f"[SAE] Region {i+1}/{n}: "
                        f"{region['drop_start']}-{region['drop_end']} "
                        f"({len(region['seq'])} bp, conf={region['start_confidence']:.2f})"
                    )
                fts = get_feature_ts(model, sae, region['seq'], SAE_LAYER_NAME)
                results[i] = _make_result(i, fts)

        # Checkpoint after completing the chunk
        _save_checkpoint(chunk_start, chunk_end)

        # In extract_only mode, release feature_ts from memory after saving.
        # Join the thread first so the NPZ is fully written before we clear.
        if extract_only:
            if _chk_thread is not None:
                _chk_thread.join()
            for i in range(chunk_start, chunk_end):
                results[i]['feature_ts'] = None

    # Ensure the final checkpoint write finishes before returning
    if _chk_thread is not None:
        _chk_thread.join()

    return results


def _get_top_features_at_pos(
    feature_ts: np.ndarray,
    pos: int,
    top_n: int,
) -> List[Tuple[int, float]]:
    """Get top-N active features at a specific position."""
    if pos < 0 or pos >= len(feature_ts):
        return []

    features = feature_ts[pos, :]
    active_idx = np.where(features > 0)[0]

    if len(active_idx) == 0:
        return []

    active_vals = features[active_idx]
    sorted_order = np.argsort(active_vals)[::-1][:top_n]

    return [
        (int(active_idx[j]), float(active_vals[j]))
        for j in sorted_order
    ]


def find_signature_features_across_regions(
    results: List[Dict[str, Any]],
    min_prevalence: float = DEFAULT_SIGNATURE_MIN_PREVALENCE,
) -> List[Dict[str, Any]]:
    """
    Find SAE features that consistently activate at drop boundaries.

    Aggregates top features from all regions to find features
    that appear across many drop sites.
    """
    feature_stats = defaultdict(lambda: {
        'drop_activations': [],
        'rise_activations': [],
        'positions': [],
        'zscore_activations': [],
        'mad_activations': [],
    })

    for result in results:
        region = result['region']
        method = region.get('method', 'unknown')

        for feat_id, activation in result['drop_features']:
            feature_stats[feat_id]['drop_activations'].append(activation)
            feature_stats[feat_id]['positions'].append(region['genomic_start'])
            if method == 'zscore':
                feature_stats[feat_id]['zscore_activations'].append(activation)
            elif method == 'mad':
                feature_stats[feat_id]['mad_activations'].append(activation)

        for feat_id, activation in result['rise_features']:
            feature_stats[feat_id]['rise_activations'].append(activation)
            if method == 'zscore':
                feature_stats[feat_id]['zscore_activations'].append(activation)
            elif method == 'mad':
                feature_stats[feat_id]['mad_activations'].append(activation)

    n_regions = len(results)
    min_count = max(1, int(n_regions * min_prevalence))

    signatures = []
    for feat_id, stats in feature_stats.items():
        total_appearances = len(stats['drop_activations']) + len(stats['rise_activations'])
        if total_appearances < min_count:
            continue

        all_acts = stats['drop_activations'] + stats['rise_activations']
        zs_acts = stats['zscore_activations']
        mad_acts = stats['mad_activations']
        signatures.append({
            'feature_id': feat_id,
            'total_count': total_appearances,
            'drop_count': len(stats['drop_activations']),
            'rise_count': len(stats['rise_activations']),
            'prevalence': total_appearances / n_regions,
            'mean_activation': float(np.mean(all_acts)) if all_acts else 0.0,
            'max_activation': float(np.max(all_acts)) if all_acts else 0.0,
            'positions': stats['positions'],
            'zscore_count': len(zs_acts),
            'mad_count': len(mad_acts),
            'zscore_mean_activation': float(np.mean(zs_acts)) if zs_acts else 0.0,
            'mad_mean_activation': float(np.mean(mad_acts)) if mad_acts else 0.0,
        })

    signatures.sort(key=lambda x: -x['mean_activation'])
    return signatures


# =============================================================================
# VISUALIZATION — Notebook-style stacked per-feature line plots
# =============================================================================

def plot_region_features(
    result: Dict[str, Any],
    region_idx: int,
    output_path: str,
    annotations: Optional[List[List]] = None,
    entropy: Optional[np.ndarray] = None,
    gtf_features: Optional[List] = None,
    n_plot_features: int = DEFAULT_N_PLOT_FEATURES,
    feature_stats: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Plot SAE features for a region — Figure 4C style with gene track.

    Stacked panels (top to bottom):
      1. N feature activation traces (with GenBank/GTF annotation shading)
      2. Entropy trace with drop/rise markers
      3. GTF gene track (if provided)

    Directly mirrors the Evo2 SAE notebook's visualization:
        fig, axes = plt.subplots(len(selected_features), 1, ...)
        for ind, feature in enumerate(selected_features):
            axes[ind].plot(feature_ts[:, feature], lw=0.5, ...)
            for start, end, feature_type, _ in annotations:
                axes[ind].axvspan(start, end, color=ANNOTATION_COLORS[feature_type], alpha=0.2)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    region = result['region']
    feature_ts = result['feature_ts']
    selected_features = result['top_feature_idx'][:n_plot_features]

    if len(selected_features) == 0:
        return

    n_features = len(selected_features)
    has_entropy = entropy is not None
    has_gene_track = gtf_features is not None and len(gtf_features) > 0

    # Build panel layout: features + entropy + gene track
    n_panels = n_features + (1 if has_entropy else 0) + (1 if has_gene_track else 0)
    height_ratios = [1.0] * n_features
    if has_entropy:
        height_ratios.append(1.0)
    if has_gene_track:
        height_ratios.append(0.6)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(30, 1 * n_panels),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
    )
    if n_panels == 1:
        axes = [axes]

    # Use genomic coordinates on x-axis if we have them
    padded_start = region.get('padded_start', 0)
    padded_end = region.get('padded_end', feature_ts.shape[0])
    seq_len = feature_ts.shape[0]
    x = np.linspace(padded_start, padded_end, seq_len, endpoint=False)
    drop_x = padded_start + region['drop_local_pos']
    rise_x = padded_start + region['rise_local_pos']

    # GTF feature colors (matching Figure 4C style)
    gff_colors = {
        "CDS": "#3498db", "gene": "#2ecc71", "mRNA": "#1abc9c",
        "exon": "#a8e6cf", "transcript": "#1abc9c",
        "five_prime_UTR": "#e67e22", "three_prime_UTR": "#e74c3c",
        "start_codon": "#9b59b6", "stop_codon": "#8e44ad",
        "tRNA": "#662D91", "rRNA": "#7AC8AC", "ncRNA": "#95a5a6",
    }

    for ind, feature_id in enumerate(selected_features):
        ax = axes[ind]
        trace = feature_ts[:, feature_id]
        if feature_stats is not None:
            trace = (trace - feature_stats['mean'][feature_id]) / feature_stats['std'][feature_id]
        ax.plot(x, trace, lw=0.5, color='black', alpha=0.9)

        # Overlay GenBank annotations (same as notebook)
        if annotations:
            for start, end, feature_type, _ in annotations:
                ax.axvspan(
                    padded_start + start, padded_start + end,
                    color=ANNOTATION_COLORS.get(feature_type, 'black'),
                    alpha=0.2,
                )

        # GTF gene region shading (Figure 4C style)
        if gtf_features:
            for feat in gtf_features:
                if feat["feature_type"] == "gene":
                    s = max(feat["start"], padded_start)
                    e = min(feat["end_exclusive"], padded_end)
                    ax.axvspan(s, e, alpha=0.08, facecolor="#2ecc71", edgecolor="none")

        # Mark drop/rise boundaries
        ax.axvline(drop_x, color='red', linestyle='--', lw=0.8, alpha=0.7)
        ax.axvline(rise_x, color='blue', linestyle='--', lw=0.8, alpha=0.7)

        ax.set_xlim(padded_start, padded_end)
        if feature_stats is not None:
            ymax = max(np.percentile(np.abs(trace), 99) if len(trace) > 0 else 3.0, 2.0)
            ax.set_ylim([-ymax, ymax])
            ax.set_yticks([-2, 0, 2])
            ax.axhline(0, color='#aaaaaa', lw=0.4, zorder=0)
        else:
            ax.set_ylim([0, 7])
            ax.set_yticks([0, 5])
        ax.set_ylabel(f"F{feature_id}", fontsize=8, rotation=0, labelpad=30, va='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelbottom=False)

    # Entropy panel
    if has_entropy:
        ax = axes[n_features]
        ent_slice = entropy[padded_start:padded_end]
        ent_x = np.linspace(padded_start, padded_end, len(ent_slice), endpoint=False)
        ax.plot(ent_x, ent_slice, lw=0.4, color='#2c3e50', alpha=0.8)
        ax.axvspan(drop_x, rise_x, alpha=0.15, color='#e74c3c')
        ax.axvline(drop_x, color='red', linestyle='--', lw=0.8, alpha=0.7)
        ax.axvline(rise_x, color='blue', linestyle='--', lw=0.8, alpha=0.7)
        ax.set_ylabel("Entropy", fontsize=8, rotation=0, labelpad=30, va='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelbottom=False)
        if gtf_features:
            for feat in gtf_features:
                if feat["feature_type"] == "gene":
                    s = max(feat["start"], padded_start)
                    e = min(feat["end_exclusive"], padded_end)
                    ax.axvspan(s, e, alpha=0.08, facecolor="#2ecc71", edgecolor="none")

    # GTF gene track panel (bottom, Figure 4C style)
    if has_gene_track:
        gene_ax = axes[-1]
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
            from analyze_scoring_results import draw_gene_track
            draw_gene_track(gene_ax, gtf_features, padded_start, padded_end)
        except ImportError:
            # Fallback: draw simple gene bars
            for feat in gtf_features:
                color = gff_colors.get(feat["feature_type"], "#999999")
                s = max(feat["start"], padded_start)
                e = min(feat["end_exclusive"], padded_end)
                gene_ax.barh(0, e - s, left=s, height=0.6, color=color, alpha=0.7)
            gene_ax.set_ylim(-1, 1)
            gene_ax.set_yticks([])
        gene_ax.set_xlim(padded_start, padded_end)
        gene_ax.tick_params(axis='x', labelbottom=True, labelsize=8)
        gene_ax.set_xlabel(f"Genomic position", fontsize=10)
    else:
        axes[-1].tick_params(axis='x', labelbottom=True, labelsize=8)
        axes[-1].set_xlabel(f"Genomic position", fontsize=10)

    # Format x-axis as Mb
    from matplotlib.ticker import FuncFormatter
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1e6:.3f} Mb"))

    fig.suptitle(
        f"Region {region_idx+1}: "
        f"{region['genomic_start']:,}-{region['genomic_end']:,} | "
        f"method={region['method']} | confidence={region['start_confidence']:.2f}",
        fontsize=12, fontweight='bold', y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_region_entropy(
    result: Dict[str, Any],
    entropy: np.ndarray,
    region_idx: int,
    output_path: str,
):
    """
    Plot the entropy curve for a region with drop/rise boundary markers.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    region = result['region']
    padded_start = region['padded_start']
    padded_end = region['padded_end']
    ent_slice = entropy[padded_start:padded_end]

    fig, ax = plt.subplots(figsize=(30, 2))

    positions = np.arange(len(ent_slice))
    ax.plot(positions, ent_slice, color='#2c3e50', lw=0.5, alpha=0.9)

    # Shade the low-entropy region
    drop_local = region['drop_local_pos']
    rise_local = region['rise_local_pos']
    ax.axvspan(drop_local, rise_local, alpha=0.15, color='#e74c3c')
    ax.axvline(drop_local, color='red', linestyle='--', lw=0.8, alpha=0.7,
               label='Drop start')
    ax.axvline(rise_local, color='blue', linestyle='--', lw=0.8, alpha=0.7,
               label='Drop end')

    ax.set_xlim(0, len(ent_slice))
    ax.set_ylabel('Entropy (nats)', fontsize=10)
    ax.set_xlabel('Position (bp)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Region {region_idx+1}: Entropy | "
        f"chr {region['genomic_start']:,}-{region['genomic_end']:,}",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_region_figure4g(
    result: Dict[str, Any],
    region_idx: int,
    output_path: str,
    annotations: Optional[List[List]] = None,
    entropy: Optional[np.ndarray] = None,
    gtf_features: Optional[List] = None,
    n_plot_features: int = DEFAULT_N_PLOT_FEATURES,
    chrom: str = "",
    feature_stats: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Figure 4g-style plot: filled area SAE feature traces + annotation track.

    Matches the Evo2 paper Figure 4g visual style:
      - Blue filled area plots (fill_between) for each feature
      - Orange labels for known bio features, gray for region-specific
      - Gray exon/CDS shading behind traces (like Fig 4g)
      - Scoring-style annotation track at bottom (colored rows per feature type)
      - Optional entropy panel
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    region = result['region']
    feature_ts = result['feature_ts']

    # Build feature list: known biological features first, then top region features
    bio_feature_ids = list(KNOWN_BIO_FEATURES.keys())
    region_top = [f for f in result['top_feature_idx'][:n_plot_features]
                  if f not in KNOWN_BIO_FEATURES]
    selected_features = bio_feature_ids + region_top

    if len(selected_features) == 0:
        return

    n_bio = len(bio_feature_ids)
    n_features = len(selected_features)
    has_entropy = entropy is not None
    has_gene_track = gtf_features is not None and len(gtf_features) > 0

    # Panel layout: features + optional entropy + annotation track
    n_panels = n_features + (1 if has_entropy else 0) + (1 if has_gene_track else 0)
    height_ratios = [1.0] * n_features
    if has_entropy:
        height_ratios.append(1.0)
    if has_gene_track:
        height_ratios.append(1.2)  # taller annotation track (multiple rows)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(30, 1.0 * n_panels + 0.5),
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05},
    )
    if n_panels == 1:
        axes = [axes]

    # Genomic coordinates
    padded_start = region.get('padded_start', 0)
    padded_end = region.get('padded_end', feature_ts.shape[0])
    seq_len = feature_ts.shape[0]
    x = np.linspace(padded_start, padded_end, seq_len, endpoint=False)
    drop_x = padded_start + region['drop_local_pos']
    rise_x = padded_start + region['rise_local_pos']

    # Color palette (matching Fig 4g)
    fill_color = '#4A90D9'       # Blue (matching paper)
    fill_alpha = 0.85
    line_color = '#3A7BC8'
    line_width = 0.3
    bio_label_color = '#E67E22'  # Orange for known bio feature labels
    region_label_color = '#555555'  # Gray for region-specific feature labels

    # Collect exon/CDS regions for gray shading on feature panels (Fig 4g style)
    exon_regions = []
    if gtf_features:
        for feat in gtf_features:
            if feat["feature_type"] in ("exon", "CDS"):
                s = max(feat["start"], padded_start)
                e = min(feat["end_exclusive"], padded_end)
                if s < e:
                    exon_regions.append((s, e))

    # ── Feature panels (filled area style) ──
    for ind, feature_id in enumerate(selected_features):
        ax = axes[ind]
        trace = feature_ts[:, feature_id]

        # Z-score normalize using chromosome-level stats if provided
        if feature_stats is not None:
            trace = (trace - feature_stats['mean'][feature_id]) / feature_stats['std'][feature_id]
            normalized = True
        else:
            normalized = False

        # Gray exon/CDS shading behind traces (like Fig 4g)
        for s, e in exon_regions:
            ax.axvspan(s, e, alpha=0.12, facecolor='#888888', edgecolor='none', zorder=0)

        # Filled area plot (Figure 4g style) — baseline 0 for both raw and z-scored
        ax.fill_between(x, 0, trace, facecolor=fill_color, alpha=fill_alpha,
                         edgecolor=line_color, linewidth=line_width)

        # Drop/rise boundary markers
        ax.axvline(drop_x, color='#D62828', linestyle='--', lw=0.7, alpha=0.5)
        ax.axvline(rise_x, color='#1D3557', linestyle='--', lw=0.7, alpha=0.5)

        if normalized:
            ymax = max(np.percentile(np.abs(trace), 99) if len(trace) > 0 else 3.0, 2.0)
            ax.set_xlim(padded_start, padded_end)
            ax.set_ylim([-ymax, ymax])
            ax.set_yticks([-2, 0, 2])
            ax.axhline(0, color='#aaaaaa', lw=0.4, zorder=0)
        else:
            ymax = max(np.percentile(trace[trace > 0], 99) if np.any(trace > 0) else 3, 3)
            ax.set_xlim(padded_start, padded_end)
            ax.set_ylim([0, ymax])
            ax.set_yticks([0, int(ymax)])

        # Label: orange for known bio features, gray for region-specific
        is_bio = feature_id in KNOWN_BIO_FEATURES
        if is_bio:
            bio_name = KNOWN_BIO_FEATURES[feature_id][0]
            label = f"{bio_name}\nf/{feature_id}"
            label_color = bio_label_color
        else:
            label = f"f/{feature_id}"
            label_color = region_label_color
        ax.set_ylabel(label, fontsize=8, rotation=0,
                       labelpad=50, va='center', fontweight='bold',
                       color=label_color)

        # "Feature activations" label on the right of middle panel
        if ind == n_features // 2:
            ax2 = ax.twinx()
            ylabel = "Feature activations (z-scored)" if feature_stats is not None else "Feature activations"
            ax2.set_ylabel(ylabel, fontsize=10, rotation=270,
                           labelpad=15, color='#555555')
            ax2.set_yticks([])

        ax.tick_params(axis='y', labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelbottom=False)

    # ── Entropy panel ──
    if has_entropy:
        ax = axes[n_features]
        ent_slice = entropy[padded_start:padded_end]
        ent_x = np.linspace(padded_start, padded_end, len(ent_slice), endpoint=False)

        # Gray exon shading on entropy too
        for s, e in exon_regions:
            ax.axvspan(s, e, alpha=0.12, facecolor='#888888', edgecolor='none', zorder=0)

        ax.fill_between(ent_x, 0, ent_slice, facecolor='#2c3e50', alpha=0.5,
                         edgecolor='#2c3e50', linewidth=0.3)
        ax.axvspan(drop_x, rise_x, alpha=0.15, color='#e74c3c')
        ax.axvline(drop_x, color='#D62828', linestyle='--', lw=0.7, alpha=0.5)
        ax.axvline(rise_x, color='#1D3557', linestyle='--', lw=0.7, alpha=0.5)
        ax.set_ylabel("Entropy", fontsize=8, rotation=0, labelpad=50,
                       va='center', fontweight='bold', color='#555555')
        ax.tick_params(axis='y', labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelbottom=False)

    # ── Annotation track (scoring style: colored rows per feature type) ──
    if has_gene_track:
        gene_ax = axes[-1]
        # Use the same draw_gene_track from analyze_scoring_results
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
            from analyze_scoring_results import draw_gene_track
            draw_gene_track(gene_ax, gtf_features, padded_start, padded_end)
        except ImportError:
            # Fallback: simple bars
            from matplotlib.patches import Rectangle, Patch
            _gff_colors = {
                "CDS": "#3498db", "gene": "#2ecc71", "exon": "#a8e6cf",
                "transcript": "#1abc9c", "five_prime_UTR": "#e67e22",
                "three_prime_UTR": "#e74c3c",
            }
            vis = [f for f in gtf_features
                   if f["start"] < padded_end and f["end_exclusive"] > padded_start]
            types_present = sorted(set(f["feature_type"] for f in vis))
            n_types = max(len(types_present), 1)
            sub_h = 1.0 / n_types
            type_to_row = {ft: i for i, ft in enumerate(types_present)}
            gene_ax.set_ylim(0, 1)
            gene_ax.set_yticks([])
            for feat in vis:
                row = type_to_row[feat["feature_type"]]
                color = _gff_colors.get(feat["feature_type"], "#95a5a6")
                s = max(feat["start"], padded_start)
                e = min(feat["end_exclusive"], padded_end)
                gene_ax.add_patch(Rectangle(
                    (s, row * sub_h + sub_h * 0.075), e - s, sub_h * 0.85,
                    facecolor=color, edgecolor="none", alpha=0.85))
            for ftype, row in type_to_row.items():
                color = _gff_colors.get(ftype, "#95a5a6")
                gene_ax.text(padded_start, (row + 0.5) * sub_h, ftype,
                             ha="left", va="center", fontsize=6,
                             fontweight="bold", color=color)

        gene_ax.set_xlim(padded_start, padded_end)
        gene_ax.set_ylabel("Annotations", fontsize=8, rotation=0, labelpad=50,
                           va='center', color='#555555')
        gene_ax.tick_params(axis='x', labelbottom=True, labelsize=8)
        chrom_label = chrom if chrom else "chr"
        gene_ax.set_xlabel(f"Position (bp)", fontsize=10)
    else:
        axes[-1].tick_params(axis='x', labelbottom=True, labelsize=8)
        axes[-1].set_xlabel("Position (bp)", fontsize=10)

    # X-axis formatting
    for ax in axes:
        ax.set_xlim(padded_start, padded_end)

    def _fmt_pos(v, _):
        return f"{int(v):,}"
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_fmt_pos))

    # Title — inside the figure, tight to top
    axes[0].set_title(
        f"Region {region_idx+1}: "
        f"{region['genomic_start']:,}-{region['genomic_end']:,} | "
        f"method={region['method']} | confidence={region['start_confidence']:.2f}",
        fontsize=11, fontweight='bold', pad=4,
    )

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close(fig)


def plot_signature_summary(
    signatures: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 50,
):
    """Bar chart of signature features with prevalence and activation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not signatures:
        return

    sigs = signatures[:top_n]
    feat_ids = [f"F{s['feature_id']}" for s in sigs]
    prevalences = [s['prevalence'] for s in sigs]
    mean_acts = [s['mean_activation'] for s in sigs]
    drop_counts = [s['drop_count'] for s in sigs]
    rise_counts = [s['rise_count'] for s in sigs]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(sigs) * 0.3)))

    y = range(len(sigs))

    # Left: Prevalence
    bars = axes[0].barh(y, prevalences, color='#3498db', alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(feat_ids, fontsize=8)
    axes[0].set_xlabel('Prevalence (fraction of regions)', fontsize=10)
    axes[0].set_title('Feature Prevalence Across Drop Regions', fontsize=11, fontweight='bold')
    axes[0].invert_yaxis()

    for i, (bar, dc, rc) in enumerate(zip(bars, drop_counts, rise_counts)):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'd={dc} r={rc}', va='center', fontsize=7, color='#555')

    # Right: Mean activation
    axes[1].barh(y, mean_acts, color='#e74c3c', alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(feat_ids, fontsize=8)
    axes[1].set_xlabel('Mean Activation Strength', fontsize=10)
    axes[1].set_title('Feature Activation at Drop Boundaries', fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_heatmap(
    results: List[Dict[str, Any]],
    signatures: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 50,
):
    """
    Feature x Region heatmap.

    Rows: top N signature features (by mean_activation).
    Columns: regions sorted by genomic position.
    Cells: activation at drop position.
    Column top bar color-coded by detection method (red=zscore, blue=MAD).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not signatures or not results:
        return

    # Select top features
    top_sigs = signatures[:top_n]
    feat_ids = [s['feature_id'] for s in top_sigs]

    # Sort regions by genomic position
    sorted_results = sorted(results, key=lambda r: r['region']['genomic_start'])

    # Build activation matrix (features x regions)
    matrix = np.zeros((len(feat_ids), len(sorted_results)), dtype=np.float32)
    methods = []

    for j, result in enumerate(sorted_results):
        methods.append(result['region'].get('method', 'unknown'))
        # Build lookup from drop_features
        drop_dict = {fid: act for fid, act in result['drop_features']}
        for i, fid in enumerate(feat_ids):
            matrix[i, j] = drop_dict.get(fid, 0.0)

    # Plot
    fig_height = max(6, len(feat_ids) * 0.35)
    fig_width = max(10, len(sorted_results) * 0.15 + 3)
    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(fig_width, fig_height + 0.8),
        gridspec_kw={'height_ratios': [0.08, 1]}, sharex=True,
    )

    # Top bar: method color per region
    method_colors = {'zscore': '#E74C3C', 'mad': '#3498db'}
    bar_colors = [method_colors.get(m, '#999999') for m in methods]
    ax_bar.bar(range(len(methods)), [1]*len(methods), color=bar_colors, width=1.0)
    ax_bar.set_xlim(-0.5, len(methods) - 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_title('Feature x Region Activation Heatmap', fontsize=12,
                      fontweight='bold', pad=10)
    # Legend for methods
    from matplotlib.patches import Patch
    legend_patches = [Patch(color='#E74C3C', label='zscore'),
                      Patch(color='#3498db', label='MAD')]
    ax_bar.legend(handles=legend_patches, loc='upper right', fontsize=8,
                  ncol=2, framealpha=0.9)

    # Heatmap
    im = ax_heat.imshow(matrix, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
    ax_heat.set_yticks(range(len(feat_ids)))
    ax_heat.set_yticklabels([f'F{fid}' for fid in feat_ids], fontsize=7)
    ax_heat.set_xlabel('Regions (sorted by genomic position)', fontsize=10)
    ax_heat.set_ylabel('SAE Feature', fontsize=10)

    cbar = plt.colorbar(im, ax=ax_heat, pad=0.02, shrink=0.8)
    cbar.set_label('Activation at drop position', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_prevalence_by_method(
    signatures: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 50,
):
    """
    Stacked bar chart: per-feature region count split by detection method.

    For each top feature, shows how many z-score regions vs MAD regions
    it appears in.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not signatures:
        return

    sigs = signatures[:top_n]
    feat_ids = [f"F{s['feature_id']}" for s in sigs]
    zs_counts = [s.get('zscore_count', 0) for s in sigs]
    mad_counts = [s.get('mad_count', 0) for s in sigs]

    fig, ax = plt.subplots(figsize=(12, max(6, len(sigs) * 0.3)))

    y = range(len(sigs))
    ax.barh(y, zs_counts, color='#E74C3C', alpha=0.8, label='z-score regions')
    ax.barh(y, mad_counts, left=zs_counts, color='#3498db', alpha=0.8,
            label='MAD regions')

    ax.set_yticks(y)
    ax.set_yticklabels(feat_ids, fontsize=8)
    ax.set_xlabel('Number of regions where feature fires', fontsize=10)
    ax.set_title('SAE Feature Prevalence by Detection Method',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=9)

    # Annotate totals
    for i, (zc, mc) in enumerate(zip(zs_counts, mad_counts)):
        total = zc + mc
        if total > 0:
            ax.text(total + 0.3, i, str(total), va='center', fontsize=7,
                    color='#555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# INTERACTIVE NOTEBOOK GENERATION
# =============================================================================

def generate_exploration_notebook(
    results: List[Dict[str, Any]],
    signatures: List[Dict[str, Any]],
    entropy_file: str,
    output_dir: str,
    genbank_path: Optional[str] = None,
    logger: logging.Logger = None,
):
    """
    Generate a Jupyter notebook for interactive SAE exploration.

    Follows the same plotting style as the original SAE notebook:
    stacked per-feature subplots with GenBank annotation overlays.
    """
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    }

    cells = []
    abs_output_dir = os.path.abspath(output_dir)
    feature_matrices_path = os.path.join(abs_output_dir, 'data', 'feature_matrices.npz')

    # --- Cell 1: Title ---
    cells.append(nbformat.v4.new_markdown_cell(
        "# SAE Feature Exploration — Chromosome Drop Regions\n\n"
        "Interactive exploration of Sparse Autoencoder features activated\n"
        "at high-confidence entropy drop regions.\n\n"
        "Follows the visualization style from `evo2/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb`.\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"**Regions analyzed:** {len(results)}\n\n"
        f"**Signature features found:** {len(signatures)}"
    ))

    # --- Cell 2: Imports (same as notebook) ---
    cells.append(nbformat.v4.new_code_cell(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from Bio import SeqIO\n"
        "\n"
        "# Annotation colors — from the SAE notebook\n"
        "ANNOTATION_COLORS = {\n"
        "    'CDS': 'white',\n"
        "    'gene': 'gray',\n"
        "    'mobile_element': 'green',\n"
        "    'misc_feature': 'yellow',\n"
        "    'rRNA': '#7AC8AC',\n"
        "    'tRNA': '#662D91',\n"
        "    'ncRNA': 'white',\n"
        "    'Regulatory': 'red',\n"
        "    'tmRNA': 'red',\n"
        "}\n"
    ))

    # --- Cell 3: Load pre-computed data ---
    region_metadata = []
    for r in results:
        reg = r['region']
        region_metadata.append({
            'genomic_start': reg['genomic_start'],
            'genomic_end': reg['genomic_end'],
            'method': reg['method'],
            'start_confidence': reg['start_confidence'],
            'region_length': reg['region_length'],
            'drop_local_pos': reg['drop_local_pos'],
            'rise_local_pos': reg['rise_local_pos'],
            'padded_start': reg['padded_start'],
            'padded_end': reg['padded_end'],
        })

    top_features_per_region = [r['top_feature_idx'] for r in results]

    cells.append(nbformat.v4.new_code_cell(
        f"# Load pre-computed SAE results\n"
        f"data = np.load('{feature_matrices_path}', allow_pickle=True)\n"
        f"feature_matrices = [data[f'region_{{i}}'] for i in range({len(results)})]\n"
        f"\n"
        f"# Load entropy\n"
        f"entropy_data = np.load('{os.path.abspath(entropy_file)}')\n"
        f"entropy = entropy_data['entropy']\n"
        f"\n"
        f"# Region metadata\n"
        f"regions = {json.dumps(region_metadata, indent=2)}\n"
        f"\n"
        f"# Top feature IDs per region (auto-selected by total activation)\n"
        f"top_features = {json.dumps(top_features_per_region)}\n"
        f"\n"
        f"print(f'Loaded {{len(feature_matrices)}} regions')\n"
        f"print(f'Feature matrix shape: {{feature_matrices[0].shape}}')\n"
    ))

    # --- Cell 4: GenBank loading (if available) ---
    if genbank_path:
        cells.append(nbformat.v4.new_code_cell(
            f"# Load GenBank annotations\n"
            f"records = list(SeqIO.parse('{os.path.abspath(genbank_path)}', 'genbank'))\n"
            f"print(f'Loaded {{len(records)}} GenBank record(s)')\n"
        ))
    else:
        cells.append(nbformat.v4.new_code_cell(
            "# No GenBank file provided — set records to empty\n"
            "# To add annotations, load a GenBank file:\n"
            "#   records = list(SeqIO.parse('path/to/genome.gb', 'genbank'))\n"
            "records = []\n"
        ))

    # --- Cell 5: find_relevant_gb_annotations (from notebook) ---
    cells.append(nbformat.v4.new_code_cell(
        "def find_relevant_gb_annotations(records, window_start, window_size,\n"
        "                                valid_features={'CDS', 'gene', 'mobile_element', 'misc_feature',\n"
        "                                              'rRNA', 'tRNA', 'ncRNA', 'Regulatory', 'tmRNA'},\n"
        "                                valid_qualifiers={'gene', 'locus_id', 'product', 'mobile_element_type'}):\n"
        "    \"\"\"Extract annotations from GenBank records within a specified window.\"\"\"\n"
        "    window_end = window_start + window_size\n"
        "    annotations = []\n"
        "    for record in records:\n"
        "        for feature in record.features:\n"
        "            if feature.location.end < window_start or feature.location.start > window_end:\n"
        "                continue\n"
        "            if feature.type in valid_features:\n"
        "                start = max(0, feature.location.start - window_start)\n"
        "                end = min(window_size, feature.location.end - window_start)\n"
        "                qualifiers = {q: feature.qualifiers[q] for q in valid_qualifiers if q in feature.qualifiers}\n"
        "                annotations.append([start, end, feature.type, qualifiers])\n"
        "    return annotations\n"
    ))

    # --- Cell 6: Notebook-style feature plotting ---
    cells.append(nbformat.v4.new_markdown_cell(
        "## Feature Plots (Notebook Style)\n\n"
        "Each region shows its top features as stacked line plots,\n"
        "with GenBank annotations overlaid as colored bands."
    ))

    cells.append(nbformat.v4.new_code_cell(
        "def plot_region(region_idx):\n"
        "    \"\"\"Plot features for a region — same style as the SAE notebook.\"\"\"\n"
        "    feature_ts = feature_matrices[region_idx]\n"
        "    reg = regions[region_idx]\n"
        "    selected_features = top_features[region_idx]\n"
        "    \n"
        "    if not selected_features:\n"
        "        print('No active features in this region')\n"
        "        return\n"
        "    \n"
        "    # Get annotations if GenBank records are loaded\n"
        "    annotations = []\n"
        "    if records:\n"
        "        annotations = find_relevant_gb_annotations(\n"
        "            records, reg['padded_start'], reg['padded_end'] - reg['padded_start']\n"
        "        )\n"
        "    \n"
        "    # Plot — same pattern as notebook\n"
        "    fig, axes = plt.subplots(\n"
        "        len(selected_features), 1,\n"
        "        figsize=(30, 1 * len(selected_features)),\n"
        "        sharex=True,\n"
        "    )\n"
        "    if len(selected_features) == 1:\n"
        "        axes = [axes]\n"
        "    \n"
        "    for ind, feature_id in enumerate(selected_features):\n"
        "        axes[ind].plot(feature_ts[:, feature_id], lw=0.5,\n"
        "                       label=f'feature {feature_id}', alpha=0.9)\n"
        "        \n"
        "        # GenBank annotation overlay (from notebook)\n"
        "        for start, end, feature_type, _ in annotations:\n"
        "            axes[ind].axvspan(start, end,\n"
        "                            color=ANNOTATION_COLORS.get(feature_type, 'black'),\n"
        "                            alpha=0.2)\n"
        "        \n"
        "        # Drop boundary markers\n"
        "        axes[ind].axvline(reg['drop_local_pos'], color='red',\n"
        "                         linestyle='--', lw=0.8, alpha=0.7)\n"
        "        axes[ind].axvline(reg['rise_local_pos'], color='blue',\n"
        "                         linestyle='--', lw=0.8, alpha=0.7)\n"
        "        \n"
        "        axes[ind].set_xlim(0, feature_ts.shape[0])\n"
        "        axes[ind].set_ylim([0, 7])\n"
        "        axes[ind].set_yticks([0, 5])\n"
        "        axes[ind].legend()\n"
        "    \n"
        "    plt.suptitle(\n"
        "        f\"Region {region_idx+1}: chr {reg['genomic_start']:,}-{reg['genomic_end']:,} | \"\n"
        "        f\"method={reg['method']} | confidence={reg['start_confidence']:.2f}\",\n"
        "        fontsize=12, fontweight='bold',\n"
        "    )\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "\n"
        "# Plot first region\n"
        "plot_region(0)\n"
    ))

    # --- Cell 7: Browse all regions ---
    cells.append(nbformat.v4.new_code_cell(
        "# Plot all regions\n"
        "for i in range(len(regions)):\n"
        "    print(f'\\n--- Region {i+1}/{len(regions)} ---')\n"
        "    plot_region(i)\n"
    ))

    # --- Cell 8: Explore a single feature across all regions ---
    cells.append(nbformat.v4.new_markdown_cell(
        "## Single Feature Across All Regions\n\n"
        "Pick a feature ID and see how it activates across every detected region."
    ))

    sig_ids = [s['feature_id'] for s in signatures[:20]]
    cells.append(nbformat.v4.new_code_cell(
        f"# Signature feature IDs (most prevalent at drop boundaries)\n"
        f"signature_feature_ids = {sig_ids}\n"
        f"\n"
        f"# Pick one to explore\n"
        f"FEATURE_TO_EXPLORE = signature_feature_ids[0] if signature_feature_ids else 0\n"
        f"\n"
        f"fig, axes = plt.subplots(\n"
        f"    len(feature_matrices), 1,\n"
        f"    figsize=(30, 1 * len(feature_matrices)),\n"
        f"    sharex=False,\n"
        f")\n"
        f"if len(feature_matrices) == 1:\n"
        f"    axes = [axes]\n"
        f"\n"
        f"for i, (fm, reg) in enumerate(zip(feature_matrices, regions)):\n"
        f"    axes[i].plot(fm[:, FEATURE_TO_EXPLORE], lw=0.5,\n"
        f"                label=f'feature {{FEATURE_TO_EXPLORE}} (region {{i+1}})', alpha=0.9)\n"
        f"    axes[i].axvline(reg['drop_local_pos'], color='red', linestyle='--', lw=0.8, alpha=0.7)\n"
        f"    axes[i].axvline(reg['rise_local_pos'], color='blue', linestyle='--', lw=0.8, alpha=0.7)\n"
        f"    axes[i].set_xlim(0, fm.shape[0])\n"
        f"    axes[i].set_ylim([0, 7])\n"
        f"    axes[i].set_yticks([0, 5])\n"
        f"    axes[i].legend(fontsize=7)\n"
        f"\n"
        f"plt.suptitle(f'Feature {{FEATURE_TO_EXPLORE}} Across All Regions', fontsize=12, fontweight='bold')\n"
        f"plt.tight_layout()\n"
        f"plt.show()\n"
    ))

    # --- Cell 9: Signature summary ---
    cells.append(nbformat.v4.new_markdown_cell(
        "## Signature Features Summary"
    ))

    sig_data_str = json.dumps([{
        'feature_id': s['feature_id'],
        'prevalence': round(s['prevalence'], 3),
        'mean_activation': round(s['mean_activation'], 3),
        'drop_count': s['drop_count'],
        'rise_count': s['rise_count'],
    } for s in signatures[:30]], indent=2)

    cells.append(nbformat.v4.new_code_cell(
        f"signature_data = {sig_data_str}\n"
        "\n"
        "print(f'{{\"Feature ID\":>12}} {{\"Prevalence\":>12}} {{\"Mean Act\":>10}} {{\"Drops\":>7}} {{\"Rises\":>7}}')\n"
        "print('-' * 55)\n"
        "for s in signature_data:\n"
        "    print(f\"{s['feature_id']:>12} {s['prevalence']:>12.1%} {s['mean_activation']:>10.3f} {s['drop_count']:>7} {s['rise_count']:>7}\")\n"
    ))

    nb.cells = cells

    notebook_path = os.path.join(output_dir, 'sae_exploration.ipynb')
    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)

    if logger:
        logger.info(f"Generated interactive notebook: {notebook_path}")


# =============================================================================
# DATA OUTPUT
# =============================================================================

def save_results(
    results: List[Dict[str, Any]],
    signatures: List[Dict[str, Any]],
    output_dir: str,
    logger: logging.Logger = None,
):
    """Save TSV outputs and feature matrices."""
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # --- Per-region results TSV ---
    results_file = os.path.join(data_dir, 'sae_results.tsv')
    with open(results_file, 'w') as f:
        f.write("# SAE Feature Analysis of Chromosome Drop Regions\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Regions analyzed: {len(results)}\n")
        f.write("#\n")
        f.write("region_idx\tgenomic_start\tgenomic_end\tmethod\tconfidence\t"
                "top_features\tdrop_top_features\trise_top_features\n")

        for i, result in enumerate(results):
            reg = result['region']
            top_str = ','.join(str(f) for f in result['top_feature_idx'])
            drop_str = ','.join(f"{fid}:{act:.2f}" for fid, act in result['drop_features'][:10])
            rise_str = ','.join(f"{fid}:{act:.2f}" for fid, act in result['rise_features'][:10])

            f.write(f"{i}\t{reg['genomic_start']}\t{reg['genomic_end']}\t"
                    f"{reg['method']}\t{reg['start_confidence']:.4f}\t"
                    f"{top_str}\t{drop_str}\t{rise_str}\n")

    if logger:
        logger.info(f"Saved per-region results: {results_file}")

    # --- Signature features TSV ---
    sig_file = os.path.join(data_dir, 'signature_features.tsv')
    with open(sig_file, 'w') as f:
        f.write("# Signature SAE Features (recurring across drop regions)\n")
        f.write(f"# Total signature features: {len(signatures)}\n")
        f.write("#\n")
        f.write("feature_id\tprevalence\tmean_activation\tmax_activation\t"
                "drop_count\trise_count\ttotal_count\t"
                "zscore_count\tmad_count\tzscore_mean_activation\tmad_mean_activation\n")

        for sig in signatures:
            f.write(f"{sig['feature_id']}\t{sig['prevalence']:.4f}\t"
                    f"{sig['mean_activation']:.4f}\t{sig['max_activation']:.4f}\t"
                    f"{sig['drop_count']}\t{sig['rise_count']}\t{sig['total_count']}\t"
                    f"{sig.get('zscore_count', 0)}\t{sig.get('mad_count', 0)}\t"
                    f"{sig.get('zscore_mean_activation', 0.0):.4f}\t"
                    f"{sig.get('mad_mean_activation', 0.0):.4f}\n")

    if logger:
        logger.info(f"Saved signature features: {sig_file}")

    # --- Feature matrices (numpy compressed) ---
    matrices_file = os.path.join(data_dir, 'feature_matrices.npz')
    if results and results[0]['feature_ts'] is None:
        # extract_only mode: feature_ts was already written to _checkpoint.npz
        # during run_sae_on_regions; promote that file as the canonical output.
        chk_file = os.path.join(data_dir, '_checkpoint.npz')
        if os.path.exists(chk_file):
            import shutil
            shutil.copy2(chk_file, matrices_file)
            if logger:
                logger.info(f"Promoted checkpoint to feature matrices: {matrices_file}")
        else:
            if logger:
                logger.warning("extract_only: _checkpoint.npz not found; feature_matrices.npz will be empty")
    else:
        matrix_dict = {f'region_{i}': r['feature_ts'] for i, r in enumerate(results)}
        np.savez_compressed(matrices_file, **matrix_dict)
        if logger:
            logger.info(f"Saved feature matrices: {matrices_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Evo2 SAE analysis on chromosome drop regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (after running score_chromosome.py)
  python run_sae_on_chromosome_drops.py \\
      --boundaries chr21_test.drop_boundaries.tsv \\
      --entropy chr21_test.entropy.npz \\
      --fasta /path/to/genome.fna \\
      --chrom chr21

  # With GenBank annotations
  python run_sae_on_chromosome_drops.py \\
      --boundaries chr21_test.drop_boundaries.tsv \\
      --entropy chr21_test.entropy.npz \\
      --fasta /path/to/genome.fna \\
      --chrom chr21 \\
      --genbank /path/to/chromosome.gb

  # High-confidence only, limited to 30 regions
  python run_sae_on_chromosome_drops.py \\
      --boundaries chr21_test.drop_boundaries.tsv \\
      --entropy chr21_test.entropy.npz \\
      --fasta /path/to/genome.fna \\
      --chrom chr21 \\
      --min_confidence 3.0 \\
      --max_regions 30
        """
    )

    # Input paths (required unless --auto is used)
    parser.add_argument("--boundaries", default=None,
                        help="Path to .drop_boundaries.tsv from score_chromosome.py")
    parser.add_argument("--entropy", default=None,
                        help="Path to .entropy.npz from score_chromosome.py")
    parser.add_argument("--fasta", type=str, default=DEFAULT_FASTA,
                        help="Path to genome FASTA (default: human GRCh38)")
    parser.add_argument("--chrom", required=True,
                        help="Chromosome name (e.g., chr21, NC_000021.9)")

    # GenBank annotations (optional)
    parser.add_argument("--genbank", type=str, default=None,
                        help="Path to GenBank file for annotation overlays (optional)")
    parser.add_argument("--gtf", type=str, default=None,
                        help="Path to GTF file for annotation overlays (alternative to --genbank)")

    # Method filtering
    parser.add_argument("--method_filter", type=str, default=None,
                        choices=["zscore", "mad", "both"],
                        help="Filter regions by detection method: zscore, mad, "
                             "or both (overlapping MAD+zscore only)")
    parser.add_argument("--overlap_only", action="store_true",
                        help="Shorthand for --method_filter both (overlapping regions only)")

    # Filtering
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help=f"Minimum start_confidence to analyze (default: {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--max_regions", type=int, default=DEFAULT_MAX_REGIONS,
                        help=f"Maximum regions to analyze with SAE (default: {DEFAULT_MAX_REGIONS}). "
                             "All of these are used for latent analysis.")
    parser.add_argument("--max_plot_regions", type=int, default=50,
                        help="Maximum regions to generate per-region plots for (default: 50). "
                             "Set lower than --max_regions to run SAE on many regions for "
                             "latent analysis but only plot the top N.")

    # SAE parameters
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING,
                        help=f"Base pairs of padding around each region (default: {DEFAULT_PADDING})")
    parser.add_argument("--model_name", type=str, default="evo2_7b_262k",
                        help="Evo2 model name (default: evo2_7b_262k, same as SAE notebook)")
    parser.add_argument("--top_features", type=int, default=DEFAULT_TOP_FEATURES_PER_REGION,
                        help=f"Top features tracked per region (default: {DEFAULT_TOP_FEATURES_PER_REGION})")
    parser.add_argument("--n_plot_features", type=int, default=DEFAULT_N_PLOT_FEATURES,
                        help=f"Number of features to show in per-region plots (default: {DEFAULT_N_PLOT_FEATURES})")
    parser.add_argument("--signature_prevalence", type=float, default=DEFAULT_SIGNATURE_MIN_PREVALENCE,
                        help=f"Min prevalence for signature features (default: {DEFAULT_SIGNATURE_MIN_PREVALENCE})")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Root results directory (default: ./results)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover latest COMPLETED scoring run for --chrom. "
                             "Makes --boundaries and --entropy optional.")

    # Other
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device (default: cuda:0)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip_notebook", action="store_true",
                        help="Skip interactive notebook generation")
    parser.add_argument("--run_latent_analysis", action="store_true",
                        help="Run latent analysis (max-pool, cosine similarity, "
                             "Leiden clustering, t-SNE/UMAP) after SAE extraction")
    parser.add_argument("--leiden_resolution", type=float, default=1.0,
                        help="Leiden resolution for latent analysis (default: 1.0)")
    parser.add_argument("--stratified", action="store_true",
                        help="Use stratified sampling across confidence bins instead of "
                             "top-N selection. Produces more diverse region sets for t-SNE.")
    parser.add_argument("--latent_only", action="store_true",
                        help="Skip per-region plot generation (STEP 9) but still run SAE "
                             "extraction, save results, summary plots, and latent analysis. "
                             "Useful with --max_regions 3000 --stratified --run_latent_analysis.")
    parser.add_argument("--extract_only", action="store_true",
                        help="Save feature_matrices.npz and TSV summaries only — skip all "
                             "plots (per-region, summary, heatmap), notebook, and latent "
                             "analysis. Also releases feature_ts from RAM after each "
                             "checkpoint chunk so peak memory scales with "
                             "--checkpoint_interval rather than total regions. Recommended "
                             "when running large --max_regions (thousands) on GPU nodes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Sequences per GPU forward pass (default: 8). Regions within "
                             "each checkpoint chunk are sorted by length before batching to "
                             "minimise padding. Values of 4-8 typically give the best "
                             "throughput on H200 for drop regions of a few hundred bp.")
    parser.add_argument("--shard", type=str, default=None,
                        metavar="N/M",
                        help="Process only shard N of M (0-indexed). E.g. --shard 0/4 "
                             "processes the first quarter of qualifying regions. Submit M "
                             "jobs in parallel (each with a different N) to distribute "
                             "work across M GPUs. Shard tag is appended to the output dir "
                             "name so each shard writes to its own directory. After all "
                             "shards complete, run merge_sae_shards.py to combine them.")
    parser.add_argument("--checkpoint_interval", type=int, default=200,
                        help="Save a resume checkpoint every N regions (default: 200). "
                             "In --extract_only mode this also controls peak RAM usage.")

    args = parser.parse_args()

    # Parse --shard N/M
    shard_idx, n_shards = None, None
    if args.shard is not None:
        try:
            parts = args.shard.split('/')
            shard_idx = int(parts[0])
            n_shards  = int(parts[1])
            assert 0 <= shard_idx < n_shards, f"shard index {shard_idx} out of range [0, {n_shards})"
        except Exception as e:
            print(f"ERROR: --shard must be N/M (e.g. 0/4), got: {args.shard!r} ({e})")
            sys.exit(1)

    # Setup
    import time as _time
    _sae_wall_start = _time.time()
    logger = setup_logging(args.log_level)
    logger.info("=" * 70)
    logger.info("SAE ANALYSIS OF CHROMOSOME DROP REGIONS")
    logger.info("=" * 70)

    # Auto-discover upstream scoring run if --auto
    if args.auto:
        scoring_run = find_latest_completed(args.output_dir, args.chrom, "scoring")
        if scoring_run is None:
            logger.error(f"--auto: no COMPLETED scoring run found for {args.chrom} "
                         f"in {args.output_dir}/{args.chrom}/scoring/")
            sys.exit(1)
        logger.info(f"--auto: using scoring run {scoring_run}")
        if args.boundaries is None:
            args.boundaries = os.path.join(scoring_run, "data", "drop_boundaries.tsv")
        if args.entropy is None:
            args.entropy = os.path.join(scoring_run, "data", "entropy.npz")

    # Validate required inputs
    if args.boundaries is None:
        parser.error("--boundaries is required (or use --auto)")
    if args.entropy is None:
        parser.error("--entropy is required (or use --auto)")

    # Build organized output directory: results/{chrom}/sae/{timestamp}_{descriptor}/
    desc_parts = []
    if args.overlap_only or args.method_filter == 'both':
        desc_parts.append("overlap")
    if args.stratified:
        desc_parts.append("stratified")
    desc_parts.append(f"max{args.max_regions}" if args.max_regions > 0 else "all")
    desc_parts.append(f"conf{args.min_confidence}")
    if shard_idx is not None:
        desc_parts.append(f"shard{shard_idx}of{n_shards}")
    descriptor = "_".join(desc_parts)

    run_dir = build_run_dir(args.output_dir, args.chrom, "sae", descriptor)
    args.output_dir = run_dir
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)

    # Write source.json recording upstream inputs
    write_source(run_dir,
                 boundaries=os.path.abspath(args.boundaries),
                 entropy=os.path.abspath(args.entropy))

    # Resolve --overlap_only shorthand
    if args.overlap_only:
        args.method_filter = 'both'

    # -------------------------------------------------------------------------
    # STEP 1: Parse drop boundaries
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 1: Parsing drop boundaries")
    logger.info("-" * 70)

    from sae_utils import parse_chromosome_drops_tsv

    # Parse ALL regions first (no cap), apply confidence filter only
    regions = parse_chromosome_drops_tsv(
        args.boundaries,
        min_confidence=args.min_confidence,
        max_regions=0,  # no cap — we'll apply it after method filtering
    )

    if not regions:
        logger.error(
            f"No regions found with confidence >= {args.min_confidence}. "
            "Try lowering --min_confidence."
        )
        sys.exit(1)

    logger.info(f"Parsed {len(regions)} regions (confidence >= {args.min_confidence})")

    # Apply method filter if specified
    if args.method_filter:
        logger.info(f"Applying method filter: {args.method_filter}")
        regions = filter_regions_by_method(regions, args.method_filter, logger)

        if not regions:
            logger.error(
                f"No regions remain after method filter '{args.method_filter}'. "
                "Try a different filter or lower --min_confidence."
            )
            sys.exit(1)

    # Apply max_regions cap (default 0 = no cap in run_sae_fast.py)
    if args.max_regions > 0 and len(regions) > args.max_regions:
        if args.stratified:
            regions = stratified_sample_regions(
                regions, args.max_regions, n_bins=10, logger=logger,
            )
        else:
            logger.info(f"Capping to top {args.max_regions} regions (from {len(regions)})")
            regions = regions[:args.max_regions]

    # Apply shard slicing: split regions evenly across n_shards, take shard shard_idx
    if shard_idx is not None:
        total_before_shard = len(regions)
        # Contiguous slice: shard i gets indices [i*size, (i+1)*size)
        shard_size  = (total_before_shard + n_shards - 1) // n_shards  # ceil division
        shard_start = shard_idx * shard_size
        shard_end   = min(shard_start + shard_size, total_before_shard)
        regions = regions[shard_start:shard_end]
        logger.info(
            f"[shard {shard_idx}/{n_shards}] Processing regions "
            f"{shard_start}–{shard_end-1} of {total_before_shard} "
            f"({len(regions)} regions in this shard)"
        )

    logger.info(f"Using {len(regions)} regions")
    for i, r in enumerate(regions[:5]):
        logger.info(f"  Region {i+1}: {r['genomic_start']:,}-{r['genomic_end']:,} "
                     f"({r['region_length']} bp, {r['method']}, "
                     f"conf={r['start_confidence']:.2f})")
    if len(regions) > 5:
        logger.info(f"  ... and {len(regions)-5} more")

    # -------------------------------------------------------------------------
    # STEP 2: Load entropy data
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2: Loading entropy data")
    logger.info("-" * 70)

    entropy_data = np.load(args.entropy)
    entropy = entropy_data['entropy']
    logger.info(f"Loaded entropy array: {len(entropy):,} positions")

    # -------------------------------------------------------------------------
    # STEP 3: Load chromosome sequence
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3: Loading chromosome sequence")
    logger.info("-" * 70)

    chromosome_seq = load_chromosome_sequence(args.fasta, args.chrom, logger)

    # -------------------------------------------------------------------------
    # STEP 4: Extract region sequences
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4: Extracting region sequences")
    logger.info("-" * 70)

    regions = extract_region_sequences(regions, chromosome_seq, args.padding, logger)

    # -------------------------------------------------------------------------
    # STEP 5: Load annotations (GenBank or GTF, optional)
    # -------------------------------------------------------------------------
    gb_records = []
    gtf_features_all = None
    if args.genbank:
        logger.info("-" * 70)
        logger.info("STEP 5a: Loading GenBank annotations")
        logger.info("-" * 70)

        from Bio import SeqIO as GbSeqIO
        gb_records = list(GbSeqIO.parse(args.genbank, "genbank"))
        logger.info(f"Loaded {len(gb_records)} GenBank record(s)")
    elif args.gtf:
        logger.info("-" * 70)
        logger.info("STEP 5a: Loading GTF annotations")
        logger.info("-" * 70)

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
        from analyze_scoring_results import load_annotation_features

        chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
        # Load annotations spanning all regions with padding
        all_starts = [r['drop_start'] for r in regions]
        all_ends = [r['drop_end'] for r in regions]
        gtf_start = max(0, min(all_starts) - 10000)
        gtf_end = max(all_ends) + 10000
        gtf_features_all = load_annotation_features(
            args.gtf, chrom_id, gtf_start, gtf_end
        )
        logger.info(f"Loaded {len(gtf_features_all)} GTF features for {chrom_id} "
                     f"({gtf_start:,}-{gtf_end:,})")

    # -------------------------------------------------------------------------
    # STEP 5: Load model and SAE
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 5: Loading Evo2 model and SAE")
    logger.info("-" * 70)

    _import_torch()
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf

    logger.info(f"Loading ObservableEvo2 ({args.model_name})...")
    model = ObservableEvo2(args.model_name)
    logger.info(f"Model loaded. Device: {model.device}, Hidden dim: {model.d_hidden}")

    logger.info("Loading SAE from HuggingFace...")
    sae = load_topk_sae_from_hf(
        d_hidden=model.d_hidden,
        device=model.device,
        dtype=torch.bfloat16,
    )
    logger.info("SAE loaded (32,768 features, TopK=64)")

    # -------------------------------------------------------------------------
    # STEP 6: Run SAE analysis
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 6: Running SAE analysis on regions")
    logger.info("-" * 70)

    results = run_sae_on_regions(
        regions, model, sae,
        top_n=args.top_features,
        batch_size=args.batch_size,
        checkpoint_dir=os.path.join(args.output_dir, 'data'),
        checkpoint_interval=args.checkpoint_interval,
        extract_only=args.extract_only,
        logger=logger,
    )

    # -------------------------------------------------------------------------
    # STEP 6b: Compute per-feature z-score normalization stats
    # -------------------------------------------------------------------------
    # Use all processed regions as the normalization reference (chromosome-level proxy).
    # For each of the 32,768 SAE features: mean and std across all positions in all regions.
    # In --extract_only mode feature_ts has been cleared from memory; reload from
    # the checkpoint file that run_sae_on_regions wrote incrementally to disk.
    logger.info("-" * 70)
    logger.info("STEP 6b: Computing per-feature normalization stats across all regions")
    logger.info("-" * 70)

    if args.extract_only:
        # Reload feature_ts from per-chunk NPZ files (each chunk wrote its own file)
        import glob as _glob
        chunk_files = sorted(_glob.glob(
            os.path.join(args.output_dir, 'data', '_chunk_*.npz')
        ))
        if not chunk_files:
            logger.error("extract_only: no _chunk_*.npz files found — cannot compute stats")
            sys.exit(1)
        acts_list = []
        for cf in chunk_files:
            d = np.load(cf)
            for k in sorted(d.files, key=lambda x: int(x.split('_')[1])):
                acts_list.append(d[k])
        all_acts = np.concatenate(acts_list, axis=0)
    else:
        all_acts = np.concatenate([r['feature_ts'] for r in results], axis=0)

    feature_mean  = all_acts.mean(axis=0)
    feature_std   = np.maximum(all_acts.std(axis=0), 1e-6)
    feature_stats = {'mean': feature_mean, 'std': feature_std}
    logger.info(f"  Normalization stats computed from {all_acts.shape[0]:,} positions "
                f"across {len(results)} regions")
    del all_acts  # free memory

    # Save stats to disk for reuse
    stats_path = os.path.join(args.output_dir, 'data', 'feature_norm_stats.npz')
    np.savez_compressed(stats_path, mean=feature_mean, std=feature_std)
    logger.info(f"  Saved normalization stats to {stats_path}")

    # -------------------------------------------------------------------------
    # STEP 7: Find signature features
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 7: Finding signature features")
    logger.info("-" * 70)

    signatures = find_signature_features_across_regions(
        results,
        min_prevalence=args.signature_prevalence,
    )
    logger.info(f"Found {len(signatures)} signature features "
                f"(prevalence >= {args.signature_prevalence:.0%})")

    for sig in signatures[:10]:
        logger.info(f"  Feature {sig['feature_id']}: "
                     f"prevalence={sig['prevalence']:.1%}, "
                     f"mean_act={sig['mean_activation']:.3f}, "
                     f"drops={sig['drop_count']}, rises={sig['rise_count']}")

    # -------------------------------------------------------------------------
    # STEP 8: Save data outputs
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 8: Saving results")
    logger.info("-" * 70)

    save_results(results, signatures, args.output_dir, logger)

    # -------------------------------------------------------------------------
    # STEP 9: Generate plots (notebook-style)
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 9: Generating visualizations (notebook style)")
    logger.info("-" * 70)

    plots_dir = os.path.join(args.output_dir, 'plots')

    if args.extract_only:
        logger.info("--extract_only: skipping all visualizations (steps 9-11)")
    elif args.latent_only:
        logger.info("--latent_only: skipping per-region plots")
    else:
        n_plot = min(len(results), args.max_plot_regions)
        logger.info(f"Generating per-region plots for top {n_plot} of {len(results)} regions")

        for i, result in enumerate(results[:n_plot]):
            region = result['region']

            # Get annotations for this region (GenBank or GTF)
            annotations = []
            if gb_records:
                annotations = find_relevant_gb_annotations(
                    gb_records,
                    region['padded_start'],
                    region['padded_end'] - region['padded_start'],
                )
            elif gtf_features_all:
                annotations = gtf_to_genbank_annotations(
                    gtf_features_all,
                    region['padded_start'],
                    region['padded_end'] - region['padded_start'],
                )

            # Per-region GTF features for gene track panel
            region_gtf = None
            if gtf_features_all:
                region_gtf = [f for f in gtf_features_all
                              if f['end_exclusive'] > region['padded_start']
                              and f['start'] < region['padded_end']]

            # Feature plots (Figure 4g style: filled area + annotations + gene track)
            features_path = os.path.join(plots_dir, f'region_{i+1}_features.png')
            plot_region_figure4g(
                result, i, features_path, annotations,
                entropy=entropy, gtf_features=region_gtf,
                n_plot_features=args.n_plot_features,
                chrom=args.chrom,
                feature_stats=feature_stats,
            )

            # Standalone entropy curve (kept for quick viewing)
            entropy_path = os.path.join(plots_dir, f'region_{i+1}_entropy.png')
            plot_region_entropy(result, entropy, i, entropy_path)

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated plots for {i+1}/{n_plot} regions")

        logger.info(f"Generated {n_plot * 2} per-region plots ({len(results)} total regions for analysis)")

    if not args.extract_only:
        # Signature summary
        summary_path = os.path.join(plots_dir, 'signature_summary.png')
        plot_signature_summary(signatures, summary_path)
        logger.info("Generated signature summary plot")

        # Feature x Region heatmap
        heatmap_path = os.path.join(plots_dir, 'feature_heatmap.png')
        plot_feature_heatmap(results, signatures, heatmap_path)
        logger.info("Generated feature heatmap")

        # Method comparison
        method_path = os.path.join(plots_dir, 'feature_prevalence_by_method.png')
        plot_feature_prevalence_by_method(signatures, method_path)
        logger.info("Generated method comparison plot")

    # -------------------------------------------------------------------------
    # STEP 10: Generate interactive notebook
    # -------------------------------------------------------------------------
    if not args.extract_only and not args.skip_notebook:
        logger.info("-" * 70)
        logger.info("STEP 10: Generating interactive notebook")
        logger.info("-" * 70)

        generate_exploration_notebook(
            results, signatures, args.entropy, args.output_dir,
            genbank_path=args.genbank,
            logger=logger,
        )

    # -------------------------------------------------------------------------
    # STEP 11: Latent analysis (optional)
    # -------------------------------------------------------------------------
    if not args.extract_only and args.run_latent_analysis:
        logger.info("-" * 70)
        logger.info("STEP 11: Running latent analysis (max-pool, cosine sim, clustering)")
        logger.info("-" * 70)

        from analyze_sae_regions import (
            maxpool_regions, compute_cosine_similarity,
            compute_embedding_and_clusters, summarize_clusters,
            save_analysis_results,
            plot_cosine_similarity_heatmap, plot_embedding,
            plot_cluster_composition,
        )

        feature_matrices = [r['feature_ts'] for r in results]
        pooled = maxpool_regions(feature_matrices, pool_method="max", logger=logger)
        sim_matrix = compute_cosine_similarity(pooled, logger=logger)

        region_meta = []
        for i, r in enumerate(results):
            reg = r['region']
            region_meta.append({
                'region_idx': i,
                'genomic_start': reg['genomic_start'],
                'genomic_end': reg['genomic_end'],
                'method': reg.get('method', 'unknown'),
                'confidence': reg.get('start_confidence', 0.0),
                'region_length': reg.get('region_length', 0),
            })

        latent_dir = os.path.join(args.output_dir, 'latent_analysis')
        latent_plots = os.path.join(latent_dir, 'plots')
        os.makedirs(latent_plots, exist_ok=True)

        genomic_order = np.argsort([m['genomic_start'] for m in region_meta])
        plot_cosine_similarity_heatmap(
            sim_matrix, region_meta,
            os.path.join(latent_plots, 'cosine_similarity_heatmap.png'),
            order=genomic_order, title_suffix=" (genomic order)", logger=logger,
        )

        embedding_results = compute_embedding_and_clusters(
            pooled, region_meta, method="both",
            leiden_resolution=args.leiden_resolution,
            logger=logger,
        )
        clusters = embedding_results['cluster_assignments']

        if embedding_results['n_clusters'] > 1:
            cluster_order = np.argsort(clusters)
            plot_cosine_similarity_heatmap(
                sim_matrix, region_meta,
                os.path.join(latent_plots, 'cosine_similarity_clustered.png'),
                order=cluster_order, title_suffix=" (clustered)", logger=logger,
            )
            plot_cluster_composition(
                clusters, region_meta,
                os.path.join(latent_plots, 'cluster_composition.png'),
                logger=logger,
            )

        if embedding_results['embedding_umap'] is not None:
            plot_embedding(
                embedding_results['embedding_umap'], region_meta, clusters,
                os.path.join(latent_plots, 'umap_4panel.png'),
                embedding_name="UMAP", logger=logger,
            )
        if embedding_results['embedding_tsne'] is not None:
            plot_embedding(
                embedding_results['embedding_tsne'], region_meta, clusters,
                os.path.join(latent_plots, 'tsne_4panel.png'),
                embedding_name="t-SNE", logger=logger,
            )

        cluster_summaries = summarize_clusters(
            clusters, region_meta, pooled, sim_matrix, logger=logger,
        )
        save_analysis_results(
            pooled, sim_matrix, embedding_results, cluster_summaries,
            region_meta, latent_dir, logger=logger,
        )

        logger.info(f"Latent analysis complete: {embedding_results['n_clusters']} clusters")

    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"  data/sae_results.tsv              - Per-region top features")
    logger.info(f"  data/signature_features.tsv        - Recurring features (with method breakdown)")
    logger.info(f"  data/feature_matrices.npz          - Raw feature matrices")
    logger.info(f"  plots/region_*_features.png        - Figure 4g style (filled area + gene track)")
    logger.info(f"  plots/region_*_entropy.png         - Entropy curves with boundaries")
    logger.info(f"  plots/signature_summary.png        - Signature feature bar chart")
    logger.info(f"  plots/feature_heatmap.png          - Feature x region activation heatmap")
    logger.info(f"  plots/feature_prevalence_by_method.png - Feature prevalence by method")
    if not args.skip_notebook:
        logger.info(f"  sae_exploration.ipynb          - Interactive exploration notebook")
    if args.run_latent_analysis:
        logger.info(f"  latent_analysis/                   - Max-pool, cosine sim, clustering")
    logger.info("")
    logger.info(f"Regions analyzed: {len(results)}")
    logger.info(f"Signature features: {len(signatures)}")

    # Save run metadata
    metadata = {
        "script": "run_sae_on_chromosome_drops.py",
        "timestamp": datetime.now().isoformat(),
        "parameters": vars(args),
        "results": {
            "regions_analyzed": len(results),
            "signature_features": len(signatures),
        },
    }
    meta_path = os.path.join(args.output_dir, 'data', 'run_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Write COMPLETED sentinel (must be the very last action)
    _sae_wall_time = _time.time() - _sae_wall_start
    write_completed(run_dir, "run_sae_on_chromosome_drops.py", _sae_wall_time)
    logger.info(f"COMPLETED sentinel written to {run_dir}/COMPLETED")


if __name__ == "__main__":
    main()
