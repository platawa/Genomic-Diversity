#!/usr/bin/env python3
"""
genome_scoring_jan24.py (Updated Jan 24, 2026)

================================================================================
OVERVIEW
================================================================================
Multi-organism genome scoring pipeline using the Evo2 language model.
This script analyzes genomic regions (genes/transcripts) and computes per-position
entropy and perplexity scores to identify regions of biological interest,
particularly at exon/intron boundaries.

Key improvements from jan22 version:
- Organized output folder structure per gene/task
- Comprehensive annotations and docstrings
- Separated output types into subdirectories (data/, plots/, fasta/, metadata/)
- Improved logging and progress tracking

================================================================================
SUPPORTED ORGANISMS
================================================================================
- human    : Homo sapiens (GRCh38)
- bacillus : Bacillus subtilis (ASM904v1)
- ecoli    : Escherichia coli K-12 (ASM584v2)

================================================================================
FEATURES
================================================================================
- Robust logits extraction for tuple/nested model outputs
- Chunk scoring with overlap (context stability, fewer boundary artifacts)
- Entropy units: nats or bits
- Plot styles: plain or evodesigner-like fill
- Provenance metadata JSON
- FASTA exports: locus oriented + exon records oriented
- Plot suite: raw, smooth, boundaries, drops per method, zooms

================================================================================
OUTPUT FOLDER STRUCTURE
================================================================================
For each gene/transcript, outputs are organized as:

<out_dir>/<gene_id>/
    data/
        <gene_id>.tsv                 - Per-position scoring data
        <gene_id>.drops.txt           - Detected drop points
        <gene_id>.window_summary.tsv  - Sliding window entropy summary
    plots/
        <gene_id>.entropy_raw.png     - Raw entropy with exon shading
        <gene_id>.entropy_smooth.png  - Smoothed entropy with boundaries
        <gene_id>.entropy_boundaries.png - Boundary-focused view
        <gene_id>.drops_*.png         - Drop detection method plots
        <gene_id>.zoom_*.png          - Zoom plots around boundaries
    fasta/
        <gene_id>.locus_oriented.fa   - Full locus sequence (5'->3')
        <gene_id>.exons_oriented.fa   - Individual exon sequences
    metadata/
        <gene_id>.meta.json           - Complete run metadata/provenance

================================================================================
USAGE EXAMPLES
================================================================================
    # Score a human transcript
    python genome_scoring_jan24.py --organism human --transcript_id NM_000546.6

    # Score an E. coli gene (e.g., ffs - 4.5S RNA)
    python genome_scoring_jan24.py --organism ecoli --gene_id b0455

    # Score with custom output directory
    python genome_scoring_jan24.py --organism ecoli --gene_id b2911 --out_dir ./my_results

    # List available organisms and their configurations
    python genome_scoring_jan24.py --list_organisms

    # Pick 15 representative transcripts for analysis
    python genome_scoring_jan24.py --organism bacillus --pick15

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import math
import json
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq

from evo2 import Evo2


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the scoring pipeline.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("genome_scoring")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with formatting
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

    return logger


# =============================================================================
# ORGANISM CONFIGURATIONS
# =============================================================================
# Each organism has predefined paths to reference genomes, annotations,
# and default output directories. These can be overridden via CLI arguments.

ORGANISM_CONFIG: Dict[str, Dict[str, Any]] = {
    "human": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned/human",
        "buffer_bp": 5000,  # Larger buffer for complex human loci
        "description": "Homo sapiens (GRCh38)",
    },
    "bacillus": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned/bacillus",
        "buffer_bp": 1000,  # Smaller buffer for compact bacterial genome
        "description": "Bacillus subtilis (ASM904v1)",
    },
    "ecoli": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/execution_files/logs_gene_locus_exon_aligned/ecoli",
        "buffer_bp": 1000,  # Smaller buffer for compact bacterial genome
        "description": "Escherichia coli K-12 (ASM584v2)",
    },
}

DEFAULT_ORGANISM = "human"


# =============================================================================
# ALGORITHM PARAMETERS (DEFAULTS)
# =============================================================================

# --- Chunking parameters for long sequences ---
# Large sequences are split into overlapping chunks to fit GPU memory
MAX_CHUNK_LEN_DEFAULT = 15000   # Maximum bases per chunk
CHUNK_OVERLAP_DEFAULT = 1024    # Overlap between adjacent chunks to avoid edge artifacts

# --- Drop detection parameters ---
# These control sensitivity of entropy "drop" detection algorithms
DROP_SMOOTH_W = 51      # Window size for rolling mean smoothing
DROP_DERIV_Q = 0.01     # Quantile threshold for derivative-based detection
DROP_SHIFT_W = 200      # Window size for mean-shift detection
DROP_SHIFT_TOPK = 20    # Top K candidates for mean-shift method
DROP_CUSUM_H = 1.0      # CUSUM threshold parameter

# --- Plot parameters ---
ZOOM_BP_DEFAULT = 1000      # Base pairs to show in zoom plots (±zoom_bp around boundary)
MAX_ZOOM_PLOTS_DEFAULT = 60  # Maximum number of zoom plots to generate (safety limit)

# --- Numerical stability ---
EPS = 1e-12  # Small constant to prevent log(0)


# =============================================================================
# OUTPUT DIRECTORY MANAGEMENT
# =============================================================================

class OutputManager:
    """
    Manages organized output directory structure for each gene/transcript analysis.

    Creates and tracks subdirectories for different output types:
    - data/     : TSV files, drop points, window summaries
    - plots/    : All visualization PNG files
    - fasta/    : Sequence FASTA files
    - metadata/ : JSON metadata and provenance files

    Attributes:
        base_dir: Root directory for this gene's outputs
        data_dir: Directory for tabular data files
        plots_dir: Directory for plot images
        fasta_dir: Directory for sequence files
        meta_dir: Directory for metadata JSON files
    """

    def __init__(self, out_dir: str, gene_tag: str):
        """
        Initialize output directory structure.

        Args:
            out_dir: Parent output directory
            gene_tag: Gene/transcript identifier used for folder naming
        """
        # Clean the gene tag for use in filenames (remove special characters)
        safe_tag = gene_tag.replace(":", "_").replace("/", "_").replace("\\", "_")

        # Create base directory for this gene
        self.base_dir = os.path.join(out_dir, safe_tag)

        # Define subdirectories
        self.data_dir = os.path.join(self.base_dir, "data")
        self.plots_dir = os.path.join(self.base_dir, "plots")
        self.fasta_dir = os.path.join(self.base_dir, "fasta")
        self.meta_dir = os.path.join(self.base_dir, "metadata")

        # Create all directories
        for d in [self.data_dir, self.plots_dir, self.fasta_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)

    def data_path(self, filename: str) -> str:
        """Get full path for a data file."""
        return os.path.join(self.data_dir, filename)

    def plot_path(self, filename: str) -> str:
        """Get full path for a plot file."""
        return os.path.join(self.plots_dir, filename)

    def fasta_path(self, filename: str) -> str:
        """Get full path for a FASTA file."""
        return os.path.join(self.fasta_dir, filename)

    def meta_path(self, filename: str) -> str:
        """Get full path for a metadata file."""
        return os.path.join(self.meta_dir, filename)


# =============================================================================
# GTF PARSING FUNCTIONS
# =============================================================================

def parse_gtf_attributes(attr_str: str) -> Dict[str, str]:
    """
    Parse GTF attribute string into key-value dictionary.

    GTF attribute format: 'key1 "value1"; key2 "value2"; ...'

    Args:
        attr_str: The 9th column of a GTF line containing attributes

    Returns:
        Dictionary mapping attribute names to values (quotes stripped)

    Example:
        >>> parse_gtf_attributes('gene_id "b0001"; gene_name "thrL"')
        {'gene_id': 'b0001', 'gene_name': 'thrL'}
    """
    out = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(" ", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        val = parts[1].strip().strip('"')
        out[key] = val
    return out


def load_exons_from_gtf(
    gtf_path: str,
    gene_id: Optional[str] = None,
    transcript_id: Optional[str] = None,
) -> Tuple[str, str, List[Tuple[int, int]], Optional[str], Dict[str, str]]:
    """
    Load exon coordinates for a gene/transcript from GTF annotation file.

    This function parses the GTF file to find all exon records for the specified
    gene or transcript. For bacterial genomes (which lack exon annotations),
    it falls back to CDS records, then to gene records.

    Coordinate Convention:
        Input GTF uses 1-based, inclusive end coordinates.
        Output uses 1-based, end-exclusive coordinates: [start, end)
        This means a 100bp exon at position 1000 is represented as (1000, 1100)

    Args:
        gtf_path: Path to the GTF annotation file
        gene_id: Gene identifier (e.g., 'b0001' for E. coli)
        transcript_id: Transcript identifier (e.g., 'NM_000546.6')

    Returns:
        Tuple containing:
        - chrom: Chromosome/sequence name
        - strand: '+' or '-' indicating orientation
        - exons: List of (start, end_exclusive) tuples, merged and sorted
        - gene_name: Human-readable gene name if available
        - meta_attrs: Dictionary of additional GTF attributes for provenance

    Raises:
        AssertionError: If neither gene_id nor transcript_id provided
        ValueError: If no matching records found in GTF
        ValueError: If exons span multiple chromosomes or strands
    """
    assert (gene_id is not None) or (transcript_id is not None), \
        "Provide gene_id or transcript_id."

    # Try feature types in priority order: exon > CDS > gene
    # Bacteria typically use CDS or gene; eukaryotes use exon
    feature_priority = ["exon", "CDS", "gene"]

    for target_feature in feature_priority:
        chrom = None
        strand = None
        gene_name: Optional[str] = None
        exons: List[Tuple[int, int]] = []
        meta: Dict[str, str] = {}

        with open(gtf_path, "r") as f:
            for line in f:
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9:
                    continue

                seqname, source, feature, start, end, score, st, frame, attrs = fields

                # Filter for target feature type
                if feature != target_feature:
                    continue

                attrs_d = parse_gtf_attributes(attrs)

                # Match by transcript_id or gene_id
                if transcript_id is not None:
                    if attrs_d.get("transcript_id") != transcript_id:
                        continue
                else:
                    if attrs_d.get("gene_id") != gene_id:
                        continue

                # Extract gene name (try multiple attribute names)
                if gene_name is None:
                    gene_name = attrs_d.get("gene_name", attrs_d.get("gene", None))

                # Capture useful attributes for provenance tracking
                for k in ("gene_id", "transcript_id", "gene_name", "gene", "Name", "Dbxref"):
                    if k in attrs_d and k not in meta:
                        meta[k] = attrs_d[k]

                # Convert coordinates: GTF is 1-based inclusive -> 1-based end-exclusive
                s = int(start)          # 1-based inclusive start
                e_incl = int(end)       # 1-based inclusive end
                e = e_incl + 1          # Convert to end-exclusive

                # Validate chromosome/strand consistency
                if chrom is None:
                    chrom = seqname
                if strand is None:
                    strand = st

                if seqname != chrom:
                    raise ValueError(f"Multiple chromosomes for locus: {chrom} vs {seqname}")
                if st != strand:
                    raise ValueError(f"Multiple strands for locus: {strand} vs {st}")

                exons.append((s, e))

        # If we found records with this feature type, use them
        if chrom is not None and strand is not None and exons:
            print(f"[INFO] Found {len(exons)} region(s) using '{target_feature}' feature type")
            break

    if chrom is None or strand is None or not exons:
        raise ValueError("No exons/CDS/gene records found. Check IDs or the GTF.")

    # Sort and merge overlapping exons
    # This handles cases where annotation has overlapping or duplicate records
    exons.sort()
    merged = []
    for s, e in exons:
        if not merged or s > merged[-1][1]:
            # No overlap with previous - add new interval
            merged.append([s, e])
        else:
            # Overlap - extend previous interval
            merged[-1][1] = max(merged[-1][1], e)

    return chrom, strand, [(a, b) for a, b in merged], gene_name, meta


def exon_bounds(exons: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Get the genomic span of a list of exons.

    Args:
        exons: List of (start, end_exclusive) tuples

    Returns:
        Tuple of (min_start, max_end) defining the complete span
    """
    return min(s for s, _ in exons), max(e for _, e in exons)


# =============================================================================
# FASTA FILE OPERATIONS
# =============================================================================

def fetch_chrom_sequence(fasta_path: str, target_chrom: str) -> str:
    """
    Load a chromosome/contig sequence from a FASTA file.

    Iterates through all records in the FASTA file and returns the sequence
    for the matching chromosome. Sequences are converted to uppercase.

    Args:
        fasta_path: Path to the reference genome FASTA file
        target_chrom: Name of the chromosome/contig to fetch (must match exactly)

    Returns:
        The full chromosome sequence as an uppercase string

    Raises:
        ValueError: If the target chromosome is not found in the FASTA
    """
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == target_chrom:
            return str(record.seq).upper()
    raise ValueError(f"Chromosome {target_chrom} not found in FASTA.")


def slice_locus(seq_chr: str, start_1based: int, end_excl_1based: int) -> str:
    """
    Extract a subsequence from a chromosome using 1-based coordinates.

    Handles edge cases where requested coordinates extend beyond chromosome bounds.

    Args:
        seq_chr: Full chromosome sequence string
        start_1based: Start position (1-based, inclusive)
        end_excl_1based: End position (1-based, exclusive)

    Returns:
        Subsequence from the specified region

    Note:
        Coordinates are clamped to valid range [1, len(seq)+1]
    """
    if start_1based < 1:
        start_1based = 1
    if end_excl_1based > len(seq_chr) + 1:
        end_excl_1based = len(seq_chr) + 1
    # Convert to 0-based for Python slicing
    s0 = start_1based - 1
    e0 = end_excl_1based - 1
    return seq_chr[s0:e0]


def write_fasta(path: str, header: str, seq: str, wrap: int = 60) -> None:
    """
    Write a sequence to a FASTA file with proper line wrapping.

    Creates parent directories if they don't exist.

    Args:
        path: Output file path
        header: FASTA header line (without '>' prefix)
        seq: Sequence string to write
        wrap: Line width for sequence wrapping (default: 60)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), wrap):
            f.write(seq[i:i+wrap] + "\n")


# =============================================================================
# EXON LABELING AND ANNOTATION FUNCTIONS
# =============================================================================

def build_exon_labels_genomic_order(
    locus_start_1based: int,
    locus_len: int,
    exons_endexcl_1based: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create per-position exon labels for a genomic locus.

    For each position in the locus, determines:
    1. Whether it's within an exon (is_exon array)
    2. Which exon it belongs to (exon_id array)

    Exons are numbered 1, 2, 3, ... in genomic order.
    Intergenic/intronic positions get -1 for exon_id.

    Args:
        locus_start_1based: Genomic start position of the locus
        locus_len: Length of the locus in base pairs
        exons_endexcl_1based: List of (start, end_exclusive) exon coordinates

    Returns:
        Tuple of:
        - is_exon: np.ndarray of shape (locus_len,) with 0/1 values
        - exon_id: np.ndarray of shape (locus_len,) with exon numbers or -1
    """
    exon_id = np.full((locus_len,), -1, dtype=np.int32)

    for k, (s, e) in enumerate(exons_endexcl_1based, start=1):
        # Calculate overlap with locus
        lo = max(s, locus_start_1based)
        hi = min(e, locus_start_1based + locus_len)
        if hi <= lo:
            continue
        # Convert to locus-relative coordinates and assign exon ID
        exon_id[(lo - locus_start_1based):(hi - locus_start_1based)] = k

    is_exon = (exon_id != -1).astype(np.int32)
    return is_exon, exon_id


def build_boundary_distance_fields(is_exon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate distance from each position to nearest exon boundary.

    Creates two distance fields:
    1. Distance to nearest exon START (5' end of exon)
    2. Distance to nearest exon END (3' end of exon)

    These are useful for analyzing entropy changes at splice sites.

    Args:
        is_exon: Binary array indicating exon positions

    Returns:
        Tuple of:
        - dist_to_start: Distance to nearest exon start position
        - dist_to_end: Distance to nearest exon end position

    Note:
        Returns NaN for positions if no exon boundaries exist in the array.
    """
    L = len(is_exon)

    # Find exon start/end positions using shift comparison
    prev = np.concatenate(([0], is_exon[:-1]))  # Previous position's value
    nxt = np.concatenate((is_exon[1:], [0]))    # Next position's value

    # Exon starts: position where is_exon transitions from 0 to 1
    exon_starts = np.where((is_exon == 1) & (prev == 0))[0]
    # Exon ends: last position of exon (followed by non-exon)
    exon_ends = np.where((is_exon == 1) & (nxt == 0))[0]

    def dist_to_points(points: np.ndarray) -> np.ndarray:
        """Calculate minimum distance from each position to any point in array."""
        if points.size == 0:
            return np.full((L,), np.nan, dtype=np.float32)
        out = np.empty((L,), dtype=np.float32)
        for i in range(L):
            out[i] = float(np.min(np.abs(points - i)))
        return out

    return dist_to_points(exon_starts), dist_to_points(exon_ends)


def get_exon_intervals_oriented(
    is_exon: np.ndarray,
    exon_id: Optional[np.ndarray] = None
) -> List[Tuple[int, int, int]]:
    """
    Extract contiguous exon intervals from the is_exon array.

    Converts the per-position is_exon array into a list of interval tuples
    suitable for shading in plots or extracting subsequences.

    Args:
        is_exon: Binary array indicating exon positions
        exon_id: Optional array of exon IDs. If None, uses consecutive numbering.

    Returns:
        List of (start_idx, end_idx_exclusive, exon_id) tuples
    """
    L = len(is_exon)
    intervals: List[Tuple[int, int, int]] = []
    i = 0
    k = 0

    while i < L:
        if is_exon[i] == 1:
            # Found start of exon - find its end
            j = i + 1
            while j < L and is_exon[j] == 1:
                j += 1

            # Determine exon ID
            if exon_id is not None:
                eid = int(exon_id[i])
                if eid < 0:
                    k += 1
                    eid = k
            else:
                k += 1
                eid = k

            intervals.append((i, j, eid))
            i = j
        else:
            i += 1

    return intervals


def get_exon_boundaries_oriented(is_exon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find all exon start and end positions in oriented coordinates.

    Args:
        is_exon: Binary array indicating exon positions

    Returns:
        Tuple of:
        - exon_starts: Array of positions where exons begin
        - exon_ends: Array of positions where exons end (last position IN exon)
    """
    prev = np.concatenate(([0], is_exon[:-1]))
    nxt = np.concatenate((is_exon[1:], [0]))
    exon_starts = np.where((is_exon == 1) & (prev == 0))[0]
    exon_ends = np.where((is_exon == 1) & (nxt == 0))[0]
    return exon_starts, exon_ends


# =============================================================================
# EVO2 MODEL UTILITIES
# =============================================================================

def _bos_id(tok):
    """
    Get the Beginning-Of-Sequence token ID from the tokenizer.

    Different tokenizer implementations use different attribute names.
    This function tries common names and raises an error if none found.

    Args:
        tok: Evo2 tokenizer instance

    Returns:
        Integer token ID for BOS

    Raises:
        AssertionError: If no BOS/EOD token ID found
    """
    bid = getattr(tok, "bos_id", getattr(tok, "eod_id", None))
    if bid is None:
        raise AssertionError("Tokenizer must provide bos_id or eod_id.")
    return bid


def _extract_logits(model_out):
    """
    Robustly extract logits tensor from potentially nested model output.

    Evo2 and similar models may return outputs in various formats:
    - Direct tensor
    - NamedTuple with .logits attribute
    - Dictionary with 'logits' key
    - Nested combinations of the above

    This function recursively searches for a suitable logits tensor,
    preferring 3D tensors [Batch, Time, Vocab] when available.

    Args:
        model_out: Raw output from model forward pass

    Returns:
        Logits tensor, ideally shape [B, T, V]

    Raises:
        TypeError: If no logits tensor can be found
    """
    def walk(x):
        """Recursive generator that yields all tensor candidates."""
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

    # Prefer 3D tensors (batch, time, vocab)
    for t in candidates:
        if t.ndim == 3:
            return t
    return candidates[0]


def id_to_token_str(tok, idx: int) -> str:
    """
    Convert a token ID back to its string representation.

    Tries multiple tokenizer APIs for compatibility.

    Args:
        tok: Tokenizer instance
        idx: Token ID to convert

    Returns:
        String representation of the token, or str(idx) as fallback
    """
    for attr in ("id_to_token", "decode", "detokenize", "convert_ids_to_tokens"):
        fn = getattr(tok, attr, None)
        if callable(fn):
            try:
                out = fn([idx]) if attr in ("decode", "detokenize", "convert_ids_to_tokens") else fn(idx)
                return out[0] if isinstance(out, (list, tuple)) else str(out)
            except Exception:
                pass
    return str(idx)


# =============================================================================
# ENTROPY CALCULATION FUNCTIONS
# =============================================================================

@torch.inference_mode()
def entropy_like_reference_acgt(
    sequence: str,
    model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    prepend_bos: bool = True,
    reverse_complement: bool = False,
):
    """
    Calculate per-position entropy using only A/C/G/T tokens.

    This function computes the Shannon entropy of the model's predictions
    at each position, restricted to the four canonical nucleotides.
    The probabilities are renormalized after filtering to sum to 1.

    Entropy Interpretation:
    - Low entropy (~0): Model is confident about next nucleotide
    - High entropy (~1.39 nats = log(4)): Maximum uncertainty (uniform)

    Biological Significance:
    - Conserved regions often show lower entropy
    - Splice sites may show entropy "drops" or "spikes"
    - Repetitive regions may show high entropy

    Args:
        sequence: DNA sequence string (A, C, G, T only; no N's)
        model: Evo2 model instance
        ACGT_IDS: Tensor of token IDs for A, C, G, T
        device: Compute device ('cuda:0' or 'cpu')
        prepend_bos: Whether to prepend BOS token (recommended: True)
        reverse_complement: If True, also compute RC and average

    Returns:
        Tuple of:
        - entropy: Tensor of per-position entropy values (nats)
        - perplexity: Tensor of per-position perplexity (exp(entropy))
    """
    tok = model.tokenizer
    toks = tok.tokenize(sequence)
    if prepend_bos:
        toks = [_bos_id(tok)] + toks

    input_ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
    out = model(input_ids)
    logits = _extract_logits(out)  # [1, T, V]

    # Extract logits for only A/C/G/T tokens
    logits_sub = logits.index_select(-1, ACGT_IDS)  # [1, T, 4]

    # Renormalize to get proper probability distribution
    logZ = torch.logsumexp(logits_sub, dim=-1, keepdim=True)
    logp = logits_sub - logZ

    # Compute entropy: H = -sum(p * log(p))
    H_fwd = -(logp.exp() * logp).sum(dim=-1)  # [1, T]

    # Remove BOS position from output
    if prepend_bos:
        H_fwd = H_fwd[:, 1:]
    H_fwd = H_fwd.squeeze(0).detach().cpu()

    if not reverse_complement:
        H_final = H_fwd
    else:
        # Compute entropy on reverse complement and average
        seq_rc = str(Seq(sequence).reverse_complement())
        toks_rc = tok.tokenize(seq_rc)
        if prepend_bos:
            toks_rc = [_bos_id(tok)] + toks_rc
        input_ids_rc = torch.tensor(toks_rc, dtype=torch.long, device=device).unsqueeze(0)
        out_rc = model(input_ids_rc)
        logits_rc = _extract_logits(out_rc)

        logits_rc_sub = logits_rc.index_select(-1, ACGT_IDS)
        logZ_rc = torch.logsumexp(logits_rc_sub, dim=-1, keepdim=True)
        logp_rc = logits_rc_sub - logZ_rc
        H_rc = -(logp_rc.exp() * logp_rc).sum(dim=-1)

        if prepend_bos:
            H_rc = H_rc[:, 1:]
        H_rc = H_rc.squeeze(0).detach().cpu()

        # Flip RC entropy to align with forward strand
        H_rc = torch.flip(H_rc, dims=[0])
        # Average forward and RC entropy
        H_final = 0.5 * (H_fwd + H_rc)

    # Perplexity = exp(entropy)
    PPX = H_final.exp()
    return H_final, PPX


@torch.inference_mode()
def next_token_logprobs_and_targets_aligned(sequence: str, model: Evo2, device: str):
    """
    Get next-token log probabilities and actual token IDs for a sequence.

    This is useful for computing how well the model predicted each token,
    which can reveal regions that are "surprising" to the model.

    Args:
        sequence: DNA sequence string
        model: Evo2 model instance
        device: Compute device

    Returns:
        Tuple of:
        - logprobs_next: Log probabilities at each position [L, V]
        - target_next: Actual token IDs that appeared [L]
    """
    tok = model.tokenizer
    toks = [_bos_id(tok)] + tok.tokenize(sequence)
    input_ids = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)

    out = model(input_ids)
    logits = _extract_logits(out).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    # Shift: logprobs[t] predicts token[t+1]
    logprobs_next = logprobs[:, :-1, :]   # [1, L, V]
    target_next = input_ids[:, 1:]        # [1, L]
    return logprobs_next.squeeze(0), target_next.squeeze(0)


def next_token_probs_subset(logprobs_next: torch.Tensor, subset_ids: torch.Tensor) -> torch.Tensor:
    """
    Extract probabilities for a subset of tokens (e.g., A/C/G/T).

    Args:
        logprobs_next: Full vocabulary log probabilities [L, V]
        subset_ids: Token IDs to extract

    Returns:
        Probabilities (not log) for subset tokens [L, len(subset_ids)]
    """
    return logprobs_next.exp().index_select(-1, subset_ids)


# =============================================================================
# CHUNK-BASED LOCUS SCORING (FOR LONG SEQUENCES)
# =============================================================================

def score_locus_aligned_overlap(
    seq_oriented: str,
    evo2_model: Evo2,
    ACGT_IDS: torch.Tensor,
    device: str,
    max_chunk_len: int,
    chunk_overlap: int,
    compute_rcavg_entropy: bool = True,
):
    """
    Score a full locus using overlapping chunks for GPU memory efficiency.

    Long sequences are split into overlapping chunks. Each chunk is scored,
    and only the "core" region (excluding overlap zones) is used for the final
    result. This prevents edge artifacts where the model lacks context.

    Chunking Strategy:
    ```
    Chunk 1: [=====CORE=====|overlap]
    Chunk 2:          [overlap|=====CORE=====|overlap]
    Chunk 3:                           [overlap|=====CORE=====]
    ```

    N-handling:
    - Positions containing 'N' are skipped (returned as NaN)
    - Only contiguous non-N runs are scored

    Args:
        seq_oriented: DNA sequence in 5'->3' orientation
        evo2_model: Evo2 model instance
        ACGT_IDS: Tensor of token IDs for A, C, G, T
        device: Compute device
        max_chunk_len: Maximum bases per chunk (typically 15000)
        chunk_overlap: Overlap between chunks (typically 1024)
        compute_rcavg_entropy: Whether to compute RC-averaged entropy

    Returns:
        Tuple of:
        - entropy_fwd: Forward-strand entropy [L]
        - ppx_fwd: Forward-strand perplexity [L]
        - entropy_rc: RC-averaged entropy [L] (or NaN if not computed)
        - ppx_rc: RC-averaged perplexity [L]
        - p4: ACGT probabilities [L, 4]
        - true_tok: Token strings at each position [L]
        - ll_next: Log-likelihood of actual token [L]
    """
    L = len(seq_oriented)

    # Initialize output arrays with NaN (unscored positions remain NaN)
    entropy_fwd = np.full((L,), np.nan, dtype=np.float32)
    ppx_fwd = np.full((L,), np.nan, dtype=np.float32)
    entropy_rc = np.full((L,), np.nan, dtype=np.float32)
    ppx_rc = np.full((L,), np.nan, dtype=np.float32)

    p4 = np.full((L, 4), np.nan, dtype=np.float32)
    ll_next = np.full((L,), np.nan, dtype=np.float32)
    true_tok = np.array([""] * L, dtype=object)

    if L == 0:
        return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next

    # Calculate chunk start positions
    step = max(1, max_chunk_len - chunk_overlap)
    starts = list(range(0, L, step))

    for s in starts:
        e = min(L, s + max_chunk_len)

        # Define "core" region (excluding overlap edges)
        core_s = s if s == 0 else s + chunk_overlap // 2
        core_e = e if e == L else e - chunk_overlap // 2
        core_s = min(core_s, core_e)

        chunk_seq = seq_oriented[s:e]

        # Process contiguous non-N runs within this chunk
        i = 0
        while i < len(chunk_seq):
            if chunk_seq[i] == "N":
                i += 1
                continue

            # Find end of non-N run
            j = i + 1
            while j < len(chunk_seq) and chunk_seq[j] != "N":
                j += 1

            run_seq = chunk_seq[i:j]

            # Compute forward entropy
            Hf_t, Pf_t = entropy_like_reference_acgt(
                run_seq, evo2_model, ACGT_IDS, device,
                prepend_bos=True, reverse_complement=False
            )
            Hf = Hf_t.float().numpy().astype(np.float32)
            Pf = Pf_t.float().numpy().astype(np.float32)

            # Compute RC-averaged entropy if requested
            if compute_rcavg_entropy:
                Hr_t, Pr_t = entropy_like_reference_acgt(
                    run_seq, evo2_model, ACGT_IDS, device,
                    prepend_bos=True, reverse_complement=True
                )
                Hr = Hr_t.float().numpy().astype(np.float32)
                Pr = Pr_t.float().numpy().astype(np.float32)
            else:
                Hr = Pr = None

            # Compute next-token quantities (forward only)
            logprobs_next, target_next = next_token_logprobs_and_targets_aligned(
                run_seq, evo2_model, device
            )

            ll = (
                logprobs_next.float()
                .gather(-1, target_next.unsqueeze(-1))
                .squeeze(-1)
                .detach().cpu().numpy().astype(np.float32)
            )
            p4_run = (
                next_token_probs_subset(logprobs_next.float(), ACGT_IDS)
                .detach().cpu().numpy().astype(np.float32)
            )
            tok = evo2_model.tokenizer
            target_ids = target_next.detach().cpu().tolist()
            true_run = [id_to_token_str(tok, int(tid)) for tid in target_ids]

            # Write results back, but only for positions in the core region
            for k in range(i, j):
                g = s + k  # Global position in locus
                if g < core_s or g >= core_e:
                    continue  # Skip overlap zones
                rk = k - i  # Position within run
                entropy_fwd[g] = Hf[rk]
                ppx_fwd[g] = Pf[rk]
                if compute_rcavg_entropy:
                    entropy_rc[g] = Hr[rk]
                    ppx_rc[g] = Pr[rk]
                p4[g, :] = p4_run[rk, :]
                ll_next[g] = ll[rk]
                true_tok[g] = true_run[rk]

            i = j

    return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next


# =============================================================================
# DROP/CHANGE-POINT DETECTION METHODS
# =============================================================================

def _fill_nans_linear(x: np.ndarray) -> np.ndarray:
    """
    Fill NaN values using linear interpolation.

    For internal use in smoothing functions where NaN values would
    cause issues. Interpolates between known values; if only one value
    exists, fills with that value.

    Args:
        x: Array potentially containing NaN values

    Returns:
        Array with NaN values filled
    """
    y = x.astype(np.float32, copy=True)
    isn = np.isnan(y)
    if not np.any(isn):
        return y
    idx = np.arange(len(y))
    good = ~isn
    if good.sum() >= 2:
        y[isn] = np.interp(idx[isn], idx[good], y[good])
    elif good.sum() == 1:
        y[isn] = y[good][0]
    return y


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """
    Compute rolling mean (moving average) of an array.

    NaN values are filled before smoothing. Uses 'same' mode to
    maintain array length.

    Args:
        x: Input array
        w: Window size for rolling mean

    Returns:
        Smoothed array of same length
    """
    if w <= 1:
        return x.copy()
    y = _fill_nans_linear(x)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(y, kernel, mode="same")


def detect_drops_derivative(
    entropy: np.ndarray,
    smooth_w: int,
    thr_quantile: float
) -> List[int]:
    """
    Detect entropy drops using derivative (rate of change).

    Identifies positions where entropy decreases rapidly, which may
    indicate transitions into conserved regions or functional elements.

    Method:
    1. Smooth the entropy signal
    2. Compute first derivative (difference)
    3. Find positions where derivative is below a quantile threshold
    4. Merge nearby detections (within min_sep)

    Args:
        entropy: Per-position entropy values
        smooth_w: Window size for pre-smoothing
        thr_quantile: Quantile threshold (e.g., 0.01 = bottom 1%)

    Returns:
        List of positions where significant drops detected
    """
    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])
    thr = np.quantile(d, thr_quantile)
    candidates = np.where(d <= thr)[0].tolist()

    # Merge nearby candidates
    out = []
    last = -10**9
    min_sep = max(10, smooth_w // 2)
    for i in candidates:
        if i - last >= min_sep:
            out.append(i)
            last = i
    return out


def detect_drops_window_mean_shift(
    entropy: np.ndarray,
    w: int,
    top_k: int
) -> List[int]:
    """
    Detect entropy drops using windowed mean shift.

    Compares mean entropy in a window before vs after each position.
    Large negative shifts indicate potential drop points.

    Method:
    1. For each position, compute mean of window before and after
    2. Score = mean(after) - mean(before)
    3. Return top K positions with most negative scores

    Args:
        entropy: Per-position entropy values
        w: Window size on each side
        top_k: Maximum number of drop points to return

    Returns:
        List of positions with largest negative mean shifts
    """
    x = _fill_nans_linear(entropy)
    L = len(x)
    scores = np.full((L,), np.nan, dtype=np.float32)

    min_len = max(5, w // 10)
    for i in range(L):
        a0, a1 = max(0, i - w), i      # Window before
        b0, b1 = i, min(L, i + w)      # Window after
        if (a1 - a0) < min_len or (b1 - b0) < min_len:
            continue
        scores[i] = float(np.mean(x[b0:b1]) - np.mean(x[a0:a1]))

    good = ~np.isnan(scores)
    if good.sum() == 0:
        return []

    idx_good = np.where(good)[0]
    order = np.argsort(scores[good])  # Sort ascending (most negative first)
    picks = idx_good[order][:top_k].tolist()

    # Remove duplicates that are too close
    out = []
    for i in picks:
        if all(abs(i - j) > w // 2 for j in out):
            out.append(i)
    return out


def detect_drops_cusum(
    entropy: np.ndarray,
    smooth_w: int,
    h: float
) -> List[int]:
    """
    Detect entropy drops using CUSUM (Cumulative Sum) algorithm.

    CUSUM is a classic change-point detection method. It accumulates
    deviations from the mean; when accumulation exceeds threshold h,
    a change point is detected.

    Method:
    1. Smooth entropy and compute global mean
    2. Accumulate (mean - value) with reset-to-zero when negative
    3. Flag positions where accumulation exceeds h

    Args:
        entropy: Per-position entropy values
        smooth_w: Window size for pre-smoothing
        h: CUSUM threshold (higher = fewer, more confident detections)

    Returns:
        List of positions where CUSUM detected a drop
    """
    x = _rolling_mean(entropy, smooth_w)
    x = _fill_nans_linear(x)
    mu = float(np.mean(x))

    out = []
    s = 0.0
    last = -10**9
    min_sep = max(25, smooth_w)

    for i, xi in enumerate(x):
        s = max(0.0, s + (mu - float(xi)))  # Accumulate below-mean values
        if s > h and (i - last) > min_sep:
            out.append(i)
            last = i
            s = 0.0  # Reset after detection
    return out


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def shade_exons(
    ax: plt.Axes,
    exon_intervals: List[Tuple[int, int, int]],
    alpha: float = 0.12
) -> None:
    """
    Add vertical shading to mark exon regions on a plot.

    Args:
        ax: Matplotlib axes object
        exon_intervals: List of (start, end, exon_id) tuples
        alpha: Transparency of shading (0-1)
    """
    for (s, e, _) in exon_intervals:
        ax.axvspan(s, e, alpha=alpha)


def evodesigner_fill(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    low_quantile: float = 0.10
) -> None:
    """
    Apply EvoDesigner-style fill to highlight low-entropy regions.

    Creates a two-level fill:
    1. Light grey fill under entire curve
    2. Darker fill under positions below the low_quantile threshold

    This emphasizes conserved/functional regions with low entropy.

    Args:
        ax: Matplotlib axes object
        x: X-axis values (positions)
        y: Y-axis values (entropy)
        low_quantile: Threshold for "low" entropy (default: bottom 10%)
    """
    # Light fill everywhere
    ax.fill_between(x, y, 0, alpha=0.18)

    # Darker fill for low-entropy regions
    if np.any(~np.isnan(y)):
        thr = np.nanquantile(y, low_quantile)
        mask = y <= thr
        ax.fill_between(x, y, 0, where=mask, alpha=0.35)


def _save_fig(path: str, dpi: int = 200) -> None:
    """
    Save current matplotlib figure to file and close it.

    Args:
        path: Output file path
        dpi: Resolution in dots per inch
    """
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_suite(
    output_mgr: OutputManager,
    base_name: str,
    entropy_main: np.ndarray,
    is_exon: np.ndarray,
    drop_points: Dict[str, List[int]],
    title_prefix: str,
    smooth_w: int = 51,
    zoom_bp: int = 0,
    max_zoom_plots: int = 60,
    plot_style: str = "plain",
    unit: str = "nats",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Generate complete suite of entropy visualization plots.

    Creates multiple plot types:
    1. Raw entropy with exon shading
    2. Smoothed entropy with exon boundaries marked
    3. Boundary-focused view
    4. One plot per drop detection method
    5. Optional zoom plots around each exon boundary

    Args:
        output_mgr: OutputManager for file paths
        base_name: Base filename for outputs
        entropy_main: Per-position entropy values
        is_exon: Binary exon mask
        drop_points: Dict mapping method names to detected drop positions
        title_prefix: Title prefix for all plots
        smooth_w: Smoothing window size
        zoom_bp: If >0, create zoom plots with this radius
        max_zoom_plots: Maximum number of zoom plots (safety limit)
        plot_style: 'plain' or 'evodesigner'
        unit: 'nats' or 'bits' for axis label
        ylim: Optional (min, max) for Y-axis
    """
    x = np.arange(len(entropy_main))
    exon_intervals = get_exon_intervals_oriented(is_exon)
    exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)
    sm = _rolling_mean(entropy_main, smooth_w)

    def maybe_style(ax, xx, yy):
        """Apply evodesigner styling if configured."""
        if plot_style == "evodesigner":
            evodesigner_fill(ax, xx, yy, low_quantile=0.10)

    # --- Plot 1: Raw entropy with exon shading ---
    plt.figure(figsize=(16, 4))
    ax = plt.gca()
    shade_exons(ax, exon_intervals, alpha=0.12)
    ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
    maybe_style(ax, x, entropy_main)
    ax.set_title(f"{title_prefix} | raw")
    ax.set_xlabel("OrientedIdx (5'→3')")
    ax.set_ylabel(f"Entropy ({unit})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    _save_fig(output_mgr.plot_path(f"{base_name}.entropy_raw.png"))

    # --- Plot 2: Smoothed entropy with boundaries ---
    plt.figure(figsize=(16, 4))
    ax = plt.gca()
    shade_exons(ax, exon_intervals, alpha=0.12)
    ax.plot(x, sm, linewidth=1.2, label=f"Entropy(main) rolling_mean(w={smooth_w})")
    for s in exon_starts:
        ax.axvline(s, linestyle="--", linewidth=0.7, alpha=0.75)
    for e in exon_ends:
        ax.axvline(e, linestyle=":", linewidth=0.7, alpha=0.75)
    maybe_style(ax, x, sm)
    ax.set_title(f"{title_prefix} | smoothed + exon boundaries")
    ax.set_xlabel("OrientedIdx (5'→3')")
    ax.set_ylabel(f"Entropy ({unit})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    _save_fig(output_mgr.plot_path(f"{base_name}.entropy_smooth.png"))

    # --- Plot 3: Boundary-focused view ---
    plt.figure(figsize=(16, 4))
    ax = plt.gca()
    shade_exons(ax, exon_intervals, alpha=0.12)
    ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
    for s in exon_starts:
        ax.axvline(s, linestyle="--", linewidth=0.8, alpha=0.85)
    for e in exon_ends:
        ax.axvline(e, linestyle=":", linewidth=0.8, alpha=0.85)
    maybe_style(ax, x, entropy_main)
    ax.set_title(f"{title_prefix} | boundary-only")
    ax.set_xlabel("OrientedIdx (5'→3')")
    ax.set_ylabel(f"Entropy ({unit})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8)
    _save_fig(output_mgr.plot_path(f"{base_name}.entropy_boundaries.png"))

    # --- Plot 4: One plot per drop detection method ---
    for method, pts in drop_points.items():
        plt.figure(figsize=(16, 4))
        ax = plt.gca()
        shade_exons(ax, exon_intervals, alpha=0.12)
        ax.plot(x, sm, linewidth=1.2, label="Smoothed entropy")
        maybe_style(ax, x, sm)
        if pts:
            ys = sm[pts]
            ax.scatter(pts, ys, s=22, label=f"drops:{method}")
        ax.set_title(f"{title_prefix} | drops={method}")
        ax.set_xlabel("OrientedIdx (5'→3')")
        ax.set_ylabel(f"Entropy ({unit})")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=8)
        _save_fig(output_mgr.plot_path(f"{base_name}.drops_{method}.png"))

    # --- Plot 5: Zoom plots around boundaries (optional) ---
    if zoom_bp and zoom_bp > 0:
        boundaries = [(int(s), "start") for s in exon_starts] + \
                     [(int(e), "end") for e in exon_ends]
        boundaries.sort(key=lambda t: t[0])

        count = 0
        L = len(entropy_main)
        for idx, kind in boundaries:
            if count >= max_zoom_plots:
                break
            lo = max(0, idx - zoom_bp)
            hi = min(L, idx + zoom_bp)

            plt.figure(figsize=(14, 4))
            ax = plt.gca()

            # Shade exon portions visible in zoom window
            for (s, e, _) in exon_intervals:
                ss = max(s, lo)
                ee = min(e, hi)
                if ee > ss:
                    ax.axvspan(ss, ee, alpha=0.15)

            xx = np.arange(lo, hi)
            yy = sm[lo:hi]
            ax.plot(xx, yy, linewidth=1.3, label="Smoothed entropy")
            maybe_style(ax, xx, yy)

            ax.axvline(idx, linestyle="--" if kind == "start" else ":",
                      linewidth=1.2, alpha=0.9, label=f"exon_{kind}")

            ax.set_title(f"{title_prefix} | zoom {kind} @ {idx} (±{zoom_bp}bp)")
            ax.set_xlabel("OrientedIdx (5'→3')")
            ax.set_ylabel(f"Entropy ({unit})")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend(loc="best", fontsize=8)
            _save_fig(output_mgr.plot_path(f"{base_name}.zoom_{kind}_{idx}.png"))
            count += 1


# =============================================================================
# WINDOW SUMMARY ANALYSIS
# =============================================================================

def write_window_summary(
    out_path: str,
    entropy: np.ndarray,
    is_exon: np.ndarray,
    win: int = 200,
    step: int = 50
):
    """
    Compute sliding window summary statistics for entropy.

    Generates a TSV file with mean entropy in sliding windows,
    separated by exon vs intron positions.

    Columns:
    - WinStartOriented: Window start position (oriented coords)
    - WinEndOriented: Window end position
    - MeanEntropy: Mean entropy across all positions in window
    - MeanEntropyExon: Mean entropy for exon positions only
    - MeanEntropyIntron: Mean entropy for intron positions only
    - FracExon: Fraction of window positions that are exonic

    Args:
        out_path: Output TSV file path
        entropy: Per-position entropy values
        is_exon: Binary exon mask
        win: Window size in base pairs
        step: Step size between windows
    """
    with open(out_path, "w") as f:
        f.write("WinStartOriented\tWinEndOriented\tMeanEntropy\t"
                "MeanEntropyExon\tMeanEntropyIntron\tFracExon\n")
        L = len(entropy)
        for s in range(0, L - win + 1, step):
            e = s + win
            ent_w = entropy[s:e]
            ex = is_exon[s:e].astype(bool)

            mean_all = float(np.nanmean(ent_w)) if np.any(~np.isnan(ent_w)) else np.nan
            mean_ex = float(np.nanmean(ent_w[ex])) if np.any(ex) else np.nan
            mean_in = float(np.nanmean(ent_w[~ex])) if np.any(~ex) else np.nan
            frac_ex = float(np.mean(ex))

            f.write(f"{s}\t{e}\t{mean_all:.6f}\t{mean_ex:.6f}\t"
                    f"{mean_in:.6f}\t{frac_ex:.4f}\n")


# =============================================================================
# MAIN LOCUS PROCESSING FUNCTION
# =============================================================================

def run_one_locus(
    fasta_path: str,
    gtf_path: str,
    out_dir: str,
    gene_id: Optional[str],
    transcript_id: Optional[str],
    buffer_bp: int,
    max_chunk_len: int,
    chunk_overlap: int,
    drop_on: str,
    entropy_unit: str,
    plot_style: str,
    zoom_bp: int = 0,
    max_zoom_plots: int = 60,
):
    """
    Run complete scoring pipeline for a single gene/transcript locus.

    This is the main entry point for processing one genomic region.
    It performs all steps from loading annotations through generating outputs.

    Processing Steps:
    1. Load exon annotations from GTF
    2. Define locus with buffer around exons
    3. Fetch genomic sequence from FASTA
    4. Orient sequence 5'->3' (reverse complement if minus strand)
    5. Initialize Evo2 model
    6. Score sequence using overlapping chunks
    7. Run drop detection algorithms
    8. Generate all output files (TSV, FASTA, plots, metadata)

    Args:
        fasta_path: Path to reference genome FASTA
        gtf_path: Path to GTF annotation file
        out_dir: Base output directory
        gene_id: Gene identifier (e.g., 'b0455')
        transcript_id: Transcript identifier (alternative to gene_id)
        buffer_bp: Base pairs of flanking sequence to include
        max_chunk_len: Maximum chunk size for scoring
        chunk_overlap: Overlap between scoring chunks
        drop_on: 'rcavg' or 'fwd' - which entropy for drop detection
        entropy_unit: 'nats' or 'bits' for output
        plot_style: 'plain' or 'evodesigner'
        zoom_bp: Radius for zoom plots (0 to disable)
        max_zoom_plots: Maximum zoom plots to generate
    """
    # Determine the tag (identifier) for this analysis
    tag = transcript_id if transcript_id else gene_id
    if not tag:
        raise ValueError("Provide gene_id or transcript_id.")

    # Set up organized output directories
    output_mgr = OutputManager(out_dir, tag)
    print(f"[INFO] Output directory: {output_mgr.base_dir}")

    # --- Step 1: Load exon annotations ---
    print(f"[INFO] Loading annotations for {tag}...")
    chrom, strand, exons, gene_name, gtf_meta = load_exons_from_gtf(
        gtf_path, gene_id=gene_id, transcript_id=transcript_id
    )
    exon_start, exon_end_excl = exon_bounds(exons)

    # --- Step 2: Define locus with buffer ---
    locus_start = max(1, exon_start - buffer_bp)
    locus_end_excl = exon_end_excl + buffer_bp
    locus_len = locus_end_excl - locus_start

    print(f"[INFO] tag={tag}" + (f" | gene_name={gene_name}" if gene_name else ""))
    print(f"[INFO] chrom={chrom}, strand={strand}")
    print(f"[INFO] exon span  [{exon_start}, {exon_end_excl})")
    print(f"[INFO] locus span [{locus_start}, {locus_end_excl})  len={locus_len}  buffer={buffer_bp}")

    # --- Step 3: Fetch genomic sequence ---
    print("[INFO] Loading chromosome sequence from FASTA...")
    chr_seq = fetch_chrom_sequence(fasta_path, chrom)
    locus_seq_genomic = slice_locus(chr_seq, locus_start, locus_end_excl).upper()
    if len(locus_seq_genomic) != locus_len:
        raise RuntimeError("Locus slice length mismatch (check coordinate logic).")

    # --- Step 4: Build exon labels in genomic order ---
    is_exon_g, exon_id_g = build_exon_labels_genomic_order(locus_start, locus_len, exons)

    # --- Step 5: Orient sequence to 5'->3' ---
    if strand == "+":
        locus_seq = locus_seq_genomic
        is_exon = is_exon_g
        exon_id = exon_id_g
        pos = np.arange(locus_start, locus_end_excl, dtype=np.int64)
        map_str = "identity"
    elif strand == "-":
        # Reverse complement for minus strand
        locus_seq = str(Seq(locus_seq_genomic).reverse_complement())
        is_exon = is_exon_g[::-1].copy()
        exon_id = exon_id_g[::-1].copy()
        pos = np.arange(locus_end_excl - 1, locus_start - 1, -1, dtype=np.int64)
        map_str = "reverse_complement"
    else:
        raise ValueError(f"Unexpected strand: {strand}")

    dist_to_exon_start, dist_to_exon_end = build_boundary_distance_fields(is_exon)

    # Base name for output files
    base_name = f"{tag}_{chrom}_{locus_start}_{locus_end_excl}_strand{strand}"

    # --- Step 5b: Export FASTA files ---
    locus_hdr = f"{tag}|{gene_name or 'NA'}|{chrom}:{locus_start}-{locus_end_excl}|strand={strand}|oriented_5to3|map={map_str}"
    locus_fa_path = output_mgr.fasta_path(f"{base_name}.locus_oriented.fa")
    write_fasta(locus_fa_path, locus_hdr, locus_seq)
    print(f"[INFO] Wrote locus FASTA: {locus_fa_path}")

    exon_intervals = get_exon_intervals_oriented(is_exon, exon_id=exon_id)
    exons_fa_path = output_mgr.fasta_path(f"{base_name}.exons_oriented.fa")
    with open(exons_fa_path, "w") as f:
        for (s, e, eid) in exon_intervals:
            exon_seq = locus_seq[s:e]
            f.write(f">{tag}|{gene_name or 'NA'}|exon{eid}|orientedIdx={s}-{e}\n")
            for i in range(0, len(exon_seq), 60):
                f.write(exon_seq[i:i+60] + "\n")
    print(f"[INFO] Wrote exons FASTA: {exons_fa_path}")

    # --- Step 6: Initialize Evo2 model ---
    print("[INFO] Initializing Evo2 model...")
    evo2_model = Evo2("evo2_7b")
    device = str(getattr(evo2_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    if hasattr(evo2_model, "eval"):
        evo2_model.eval()
    print(f"[INFO] device: {device}")

    # Get token IDs for A/C/G/T nucleotides
    idx_A = evo2_model.tokenizer.tokenize("A")[0]
    idx_C = evo2_model.tokenizer.tokenize("C")[0]
    idx_G = evo2_model.tokenizer.tokenize("G")[0]
    idx_T = evo2_model.tokenizer.tokenize("T")[0]
    ACGT_IDS = torch.tensor([idx_A, idx_C, idx_G, idx_T], dtype=torch.long, device=device)

    # --- Step 7: Score the locus ---
    print("[INFO] Scoring locus (aligned arrays, overlap chunks)...")
    entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = score_locus_aligned_overlap(
        locus_seq,
        evo2_model,
        ACGT_IDS,
        device,
        max_chunk_len=max_chunk_len,
        chunk_overlap=chunk_overlap,
        compute_rcavg_entropy=True,
    )

    # Select entropy for drop detection
    if drop_on == "rcavg":
        entropy_main = entropy_rc
        name_main = "Entropy_RCavg"
    else:
        entropy_main = entropy_fwd
        name_main = "Entropy_fwd"

    # --- Step 8: Convert units if needed ---
    if entropy_unit == "bits":
        scale = 1.0 / math.log(2.0)
        entropy_fwd_u = entropy_fwd * scale
        entropy_rc_u = entropy_rc * scale
        entropy_main_u = entropy_main * scale
        unit = "bits"
        ylim = (0.0, 2.05)
    else:
        entropy_fwd_u = entropy_fwd
        entropy_rc_u = entropy_rc
        entropy_main_u = entropy_main
        unit = "nats"
        ylim = (0.0, 1.45)

    # --- Step 9: Run drop detection ---
    print(f"[INFO] Drop detection on: {name_main} ({unit})")
    drops = {
        "derivative": detect_drops_derivative(
            entropy_main_u, smooth_w=DROP_SMOOTH_W, thr_quantile=DROP_DERIV_Q
        ),
        "win_shift": detect_drops_window_mean_shift(
            entropy_main_u, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK
        ),
        "cusum": detect_drops_cusum(
            entropy_main_u, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H
        ),
    }

    # --- Step 10: Write metadata JSON ---
    meta_path = output_mgr.meta_path(f"{base_name}.meta.json")
    meta = {
        "script_version": "genome_scoring_jan24.py",
        "run_timestamp": datetime.now().isoformat(),
        "tag": tag,
        "gene_name": gene_name,
        "gtf_meta_attrs": gtf_meta,
        "chrom": chrom,
        "strand": strand,
        "exons_genomic_1based_endexcl": exons,
        "locus_genomic_1based_endexcl": [locus_start, locus_end_excl],
        "orientation_map": map_str,
        "buffer_bp": buffer_bp,
        "max_chunk_len": max_chunk_len,
        "chunk_overlap": chunk_overlap,
        "entropy_unit": unit,
        "drop_on": drop_on,
        "plot_style": plot_style,
        "exon_intervals_oriented_idx_endexcl": [(s, e, int(eid)) for (s, e, eid) in exon_intervals],
        "output_structure": {
            "base_dir": output_mgr.base_dir,
            "data_dir": output_mgr.data_dir,
            "plots_dir": output_mgr.plots_dir,
            "fasta_dir": output_mgr.fasta_dir,
            "metadata_dir": output_mgr.meta_dir,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Wrote metadata: {meta_path}")

    # --- Step 11: Write main TSV data file ---
    out_tsv = output_mgr.data_path(f"{base_name}.tsv")
    print(f"[INFO] Writing TSV: {out_tsv}")
    with open(out_tsv, "w") as f:
        # Header row
        f.write("Pos\tEntropy(nats)\tPerplexity(e)\t"
                "P(A)\tP(C)\tP(G)\tP(T)\tTrueToken\tLL_next(nats)")
        f.write("\tEntropy_RCavg(nats)\tPerplexity_RCavg(e)")
        f.write("\tBase\tOrientedIdx\tIsExon\tExonID\t"
                "DistToExonStart\tDistToExonEnd\n")

        for i in range(locus_len):
            ent = float(entropy_fwd[i]) if not np.isnan(entropy_fwd[i]) else np.nan
            px = float(ppx_fwd[i]) if not np.isnan(ppx_fwd[i]) else np.nan
            ll = float(ll_next[i]) if not np.isnan(ll_next[i]) else np.nan

            if not np.isnan(p4[i, 0]):
                a, c, g, t = p4[i, :].tolist()
            else:
                a = c = g = t = np.nan

            ent_rc = float(entropy_rc[i]) if not np.isnan(entropy_rc[i]) else np.nan
            px_rc = float(ppx_rc[i]) if not np.isnan(ppx_rc[i]) else np.nan

            f.write(
                f"{int(pos[i])}\t"
                f"{ent:.6f}\t{px:.6f}\t"
                f"{a:.6f}\t{c:.6f}\t{g:.6f}\t{t:.6f}\t"
                f"{true_tok[i]}\t{ll:.6f}\t"
                f"{ent_rc:.6f}\t{px_rc:.6f}\t"
                f"{locus_seq[i]}\t{i}\t{int(is_exon[i])}\t{int(exon_id[i])}\t"
                f"{dist_to_exon_start[i]:.1f}\t{dist_to_exon_end[i]:.1f}\n"
            )

    # --- Step 12: Generate plot suite ---
    title_prefix = f"{tag} {chrom}:{locus_start}-{locus_end_excl} strand {strand} (5'→3') | drop_on={drop_on}"
    print(f"[INFO] Writing plot suite to: {output_mgr.plots_dir}")
    plot_suite(
        output_mgr=output_mgr,
        base_name=base_name,
        entropy_main=entropy_main_u,
        is_exon=is_exon,
        drop_points=drops,
        title_prefix=title_prefix,
        smooth_w=DROP_SMOOTH_W,
        zoom_bp=zoom_bp,
        max_zoom_plots=max_zoom_plots,
        plot_style=plot_style,
        unit=unit,
        ylim=ylim,
    )

    # --- Step 13: Write drop detection results ---
    out_drops = output_mgr.data_path(f"{base_name}.drops.txt")
    print(f"[INFO] Writing drop points: {out_drops}")
    with open(out_drops, "w") as f:
        for k, pts in drops.items():
            f.write(f"{k}\t" + ",".join(map(str, pts)) + "\n")

    # --- Step 14: Write window summary ---
    out_summary = output_mgr.data_path(f"{base_name}.window_summary.tsv")
    print(f"[INFO] Writing window summary: {out_summary}")
    write_window_summary(out_summary, entropy_main_u, is_exon, win=200, step=50)

    print("[DONE] All outputs written successfully.")
    print(f"[DONE] Output directory: {output_mgr.base_dir}")


# =============================================================================
# TRANSCRIPT SELECTION HELPERS (PICK-15)
# =============================================================================

def parse_transcript_exons_from_gtf(gtf_path: str) -> Dict[str, Dict]:
    """
    Parse all transcripts from GTF and compute their properties.

    Used for selecting representative transcripts for analysis.

    Returns:
        Dictionary mapping transcript_id to properties:
        - chrom: Chromosome name
        - strand: '+' or '-'
        - exons: List of merged exon intervals
        - exon_count: Number of exons
        - exon_span: (min_start, max_end) tuple
        - length_bp: Total genomic span
    """
    tx_map: Dict[str, Dict] = {}

    with open(gtf_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqname, source, feature, start, end, score, st, frame, attrs = fields
            if feature != "exon":
                continue

            attrs_d = parse_gtf_attributes(attrs)
            tx_id = attrs_d.get("transcript_id")
            if tx_id is None:
                continue

            s = int(start)
            e = int(end) + 1

            rec = tx_map.get(tx_id)
            if rec is None:
                tx_map[tx_id] = {"chrom": seqname, "strand": st, "exons": [(s, e)]}
            else:
                if rec["chrom"] != seqname or rec["strand"] != st:
                    continue
                rec["exons"].append((s, e))

    # Merge overlapping exons and compute statistics
    for tx_id, rec in tx_map.items():
        exons = rec["exons"]
        exons.sort()
        merged = []
        for s, e in exons:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        exm = [(a, b) for a, b in merged]
        rec["exons"] = exm
        rec["exon_count"] = len(exm)
        mn = min(a for a, _ in exm)
        mx = max(b for _, b in exm)
        rec["exon_span"] = (mn, mx)
        rec["length_bp"] = mx - mn

    return tx_map


def pick_15_transcripts_recipe(gtf_path: str) -> List[str]:
    """
    Select 15 representative transcripts using a diversity recipe.

    Categories:
    1. Medium-sized multi-exon (10-50kb, 5-15 exons): 5 transcripts
    2. Large complex (>200kb): 4 transcripts
    3. Small simple (<5kb, ≤3 exons): 3 transcripts
    4. Long simple (>100kb, ≤3 exons): 2 transcripts
    5. Fill remaining from largest multi-exon

    Args:
        gtf_path: Path to GTF annotation file

    Returns:
        List of up to 15 transcript IDs
    """
    tx_map = parse_transcript_exons_from_gtf(gtf_path)
    items = [(tx, rec["chrom"], rec["strand"], rec["exon_count"], rec["length_bp"])
             for tx, rec in tx_map.items()]
    items_valid = [x for x in items if x[3] >= 2 and x[4] >= 500]

    # Category 1: Medium multi-exon (10-50kb, 5-15 exons)
    cat1 = [x for x in items_valid if (10000 <= x[4] <= 50000) and (5 <= x[3] <= 15)]
    cat1.sort(key=lambda z: (abs(z[4] - 25000), abs(z[3] - 10)))
    pick1 = [tx for tx, *_ in cat1[:5]]

    # Category 2: Large complex (>200kb)
    cat2 = [x for x in items_valid if x[4] >= 200000]
    cat2.sort(key=lambda z: (-z[3], -z[4]))
    pick2 = [tx for tx, *_ in cat2[:4]]

    # Category 3: Small simple (<5kb, ≤3 exons)
    cat3 = [x for x in items_valid if x[4] <= 5000 and x[3] <= 3]
    cat3.sort(key=lambda z: (z[4], z[3]))
    pick3 = [tx for tx, *_ in cat3[:3]]

    # Category 4: Long simple (>100kb, ≤3 exons)
    cat5 = [x for x in items_valid if x[4] >= 100000 and x[3] <= 3]
    cat5.sort(key=lambda z: (-z[4], z[3]))
    pick5 = [tx for tx, *_ in cat5[:2]]

    # Combine picks without duplicates
    picks: List[str] = []
    for group in (pick1, pick2, pick3, pick5):
        for tx in group:
            if tx not in picks:
                picks.append(tx)
            if len(picks) >= 15:
                return picks

    # Fill remaining slots from largest multi-exon
    remaining = [x for x in items_valid if x[0] not in picks]
    remaining.sort(key=lambda z: (-z[3], -z[4]))
    for tx, *_ in remaining:
        picks.append(tx)
        if len(picks) >= 15:
            break

    return picks[:15]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """
    Main entry point for the genome scoring CLI.

    Parses command-line arguments and dispatches to appropriate functions.
    """
    ap = argparse.ArgumentParser(
        description="Genome scoring with Evo2 for multiple organisms (Jan 24, 2026).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
================================================================================
SUPPORTED ORGANISMS
================================================================================
  human    - Homo sapiens (GRCh38)
  bacillus - Bacillus subtilis (ASM904v1)
  ecoli    - Escherichia coli K-12 (ASM584v2)

================================================================================
OUTPUT STRUCTURE
================================================================================
Outputs are organized per gene/transcript:
  <out_dir>/<gene_id>/
      data/      - TSV files, drop points, summaries
      plots/     - All visualization PNG files
      fasta/     - Sequence FASTA files
      metadata/  - JSON metadata files

================================================================================
EXAMPLES
================================================================================
  # Score a human transcript
  python genome_scoring_jan24.py --organism human --transcript_id NM_000546.6

  # Score an E. coli gene (ffs - 4.5S RNA)
  python genome_scoring_jan24.py --organism ecoli --gene_id b0455

  # Score ssrS (6S RNA) in E. coli
  python genome_scoring_jan24.py --organism ecoli --gene_id b2911

  # List available organisms
  python genome_scoring_jan24.py --list_organisms

  # Pick 15 representative transcripts for bacillus
  python genome_scoring_jan24.py --organism bacillus --pick15
        """
    )

    # --- Organism selection ---
    ap.add_argument("--organism", choices=list(ORGANISM_CONFIG.keys()),
                    default=DEFAULT_ORGANISM,
                    help=f"Organism to analyze. Default: {DEFAULT_ORGANISM}")
    ap.add_argument("--list_organisms", action="store_true",
                    help="List available organisms and their configurations.")

    # --- Path overrides ---
    ap.add_argument("--fasta", default=None,
                    help="Override FASTA path (default: use organism config)")
    ap.add_argument("--gtf", default=None,
                    help="Override GTF path (default: use organism config)")
    ap.add_argument("--out_dir", default=None,
                    help="Override output directory (default: use organism config)")

    # --- Gene/transcript selection ---
    ap.add_argument("--gene_id", default=None,
                    help="Gene ID to analyze (e.g., 'b0455' for E. coli ffs)")
    ap.add_argument("--transcript_id", default=None,
                    help="Transcript ID to analyze (e.g., 'NM_000546.6')")

    # --- Algorithm parameters ---
    ap.add_argument("--buffer_bp", type=int, default=None,
                    help="Buffer bp around locus (default: organism-specific)")
    ap.add_argument("--max_chunk_len", type=int, default=MAX_CHUNK_LEN_DEFAULT,
                    help=f"Max chunk length for scoring. Default: {MAX_CHUNK_LEN_DEFAULT}")
    ap.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP_DEFAULT,
                    help=f"Overlap between chunks. Default: {CHUNK_OVERLAP_DEFAULT}")
    ap.add_argument("--drop_on", choices=["rcavg", "fwd"], default="rcavg",
                    help="Entropy source for drop detection. Default: rcavg")

    # --- Output options ---
    ap.add_argument("--entropy_unit", choices=["nats", "bits"], default="bits",
                    help="Unit for entropy output. Default: bits")
    ap.add_argument("--plot_style", choices=["plain", "evodesigner"], default="evodesigner",
                    help="Plot style. Default: evodesigner")
    ap.add_argument("--zoom_bp", type=int, default=ZOOM_BP_DEFAULT,
                    help=f"Zoom plot radius in bp (0 to disable). Default: {ZOOM_BP_DEFAULT}")
    ap.add_argument("--max_zoom_plots", type=int, default=MAX_ZOOM_PLOTS_DEFAULT,
                    help=f"Maximum zoom plots. Default: {MAX_ZOOM_PLOTS_DEFAULT}")

    # --- Utility options ---
    ap.add_argument("--pick15", action="store_true",
                    help="Print 15 representative transcript_ids from GTF.")

    args = ap.parse_args()

    # --- Handle --list_organisms ---
    if args.list_organisms:
        print("\n" + "="*70)
        print("AVAILABLE ORGANISMS")
        print("="*70 + "\n")
        for org, cfg in ORGANISM_CONFIG.items():
            print(f"  {org}:")
            print(f"    Description: {cfg['description']}")
            print(f"    FASTA:       {cfg['fasta']}")
            print(f"    GTF:         {cfg['gtf']}")
            print(f"    Output dir:  {cfg['out_dir']}")
            print(f"    Buffer bp:   {cfg['buffer_bp']}")
            print()
        return

    # --- Get organism configuration ---
    org_cfg = ORGANISM_CONFIG[args.organism]
    print(f"[INFO] Organism: {args.organism} ({org_cfg['description']})")

    # Resolve paths: use CLI args if provided, otherwise organism defaults
    fasta_path = args.fasta if args.fasta else org_cfg["fasta"]
    gtf_path = args.gtf if args.gtf else org_cfg["gtf"]
    out_dir = args.out_dir if args.out_dir else org_cfg["out_dir"]
    buffer_bp = args.buffer_bp if args.buffer_bp is not None else org_cfg["buffer_bp"]

    # --- Handle --pick15 ---
    if args.pick15:
        txs = pick_15_transcripts_recipe(gtf_path)
        print(f"\n15 selected transcripts for {args.organism}:\n")
        print("\n".join(txs))
        return

    # --- Validate that we have a gene/transcript to analyze ---
    if (args.gene_id is None) and (args.transcript_id is None):
        raise SystemExit("Error: Provide --gene_id or --transcript_id, or use --pick15")

    # --- Run the main analysis ---
    run_one_locus(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        out_dir=out_dir,
        gene_id=args.gene_id,
        transcript_id=args.transcript_id,
        buffer_bp=buffer_bp,
        max_chunk_len=args.max_chunk_len,
        chunk_overlap=args.chunk_overlap,
        drop_on=args.drop_on,
        entropy_unit=args.entropy_unit,
        plot_style=args.plot_style,
        zoom_bp=args.zoom_bp,
        max_zoom_plots=args.max_zoom_plots,
    )


if __name__ == "__main__":
    main()
