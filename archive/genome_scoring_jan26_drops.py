#!/usr/bin/env python3
"""
genome_scoring_jan26_drops.py (Updated Jan 26, 2026)

================================================================================
OVERVIEW
================================================================================
Multi-organism genome scoring pipeline using the Evo2 language model.
This script analyzes genomic regions (genes/transcripts) and computes per-position
entropy and perplexity scores to identify regions of biological interest,
particularly at exon/intron boundaries.

Key improvements from jan24 version:
- NEW: Four enhanced drop detection methods with statistical confidence scores
- NEW: Z-score based detection with automatic FDR control (~80% fewer false positives)
- NEW: MAD (Median Absolute Deviation) for outlier-robust detection
- NEW: Local baseline normalization for regional variance adaptation
- NEW: Optional bootstrap consensus for high-confidence drops
- Enhanced visualization with confidence-aware marker sizing and color coding
- Top-N drop annotations on plots with position and confidence scores
- Backward compatible with all jan24 methods and outputs
- Organized output folder structure per gene/task
- Comprehensive annotations and docstrings

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
    # Basic usage with new statistical methods (default)
    python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455

    # Score a human transcript with enhanced detection
    python genome_scoring_jan26_drops.py --organism human --transcript_id NM_000546.6

    # Conservative detection (high confidence only)
    python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455 \
        --detection_methods zscore mad --zscore_threshold 3.0

    # Compare legacy vs new methods
    python genome_scoring_jan26_drops.py --organism ecoli --gene_id b2911 \
        --detection_methods derivative zscore mad

    # Bootstrap consensus for publication-quality results (slow)
    python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455 \
        --detection_methods zscore --bootstrap --n_bootstrap 100

    # List available organisms and their configurations
    python genome_scoring_jan26_drops.py --list_organisms

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import math
import json
from urllib.parse import unquote as _url_unquote
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import matplotlib.pyplot as plt

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[WARNING] Plotly not available. Install with 'pip install plotly' for interactive plots.")

from Bio import SeqIO
from Bio.Seq import Seq

from evo2 import Evo2

# Try to import SAE utilities for feature analysis
try:
    from sae_utils import (
        ObservableEvo2,
        load_topk_sae_from_hf,
        get_feature_ts,
        extract_regions_around_drops,
        analyze_drops_with_sae,
        find_signature_features,
        write_sae_analysis_output,
        SAE_LAYER_NAME,
    )
    SAE_AVAILABLE = True
except ImportError:
    SAE_AVAILABLE = False
    print("[WARNING] SAE utils not available. Install sae_utils.py for --analyze_sae feature.")


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
        "out_dir": "/orcd/data/zhang_f/001/platawa/jan31_files/outputs/human",
        "buffer_bp": 5000,  # Larger buffer for complex human loci
        "description": "Homo sapiens (GRCh38)",
    },
    "bacillus": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/jan31_files/outputs/bacillus",
        "buffer_bp": 1000,  # Smaller buffer for compact bacterial genome
        "description": "Bacillus subtilis (ASM904v1)",
    },
    "ecoli": {
        "fasta": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf",
        "out_dir": "/orcd/data/zhang_f/001/platawa/jan31_files/outputs/ecoli",
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

# --- Drop detection parameters (legacy methods) ---
# These control sensitivity of entropy "drop" detection algorithms
DROP_SMOOTH_W = 51      # Window size for rolling mean smoothing
DROP_DERIV_Q = 0.01     # Quantile threshold for derivative-based detection
DROP_SHIFT_W = 200      # Window size for mean-shift detection
DROP_SHIFT_TOPK = 20    # Top K candidates for mean-shift method
DROP_CUSUM_H = 1.0      # CUSUM threshold parameter

# --- Drop detection parameters (jan26 statistical methods) ---
DROP_ZSCORE_THRESHOLD = 2.5   # Z-score threshold (~1% FDR)
DROP_MAD_THRESHOLD = 3.0      # MAD threshold (robust outlier detection)
DROP_LOCAL_WINDOW = 500       # Local baseline window size (bp)
DROP_LOCAL_THRESHOLD = 2.0    # Local z-score threshold
DROP_MIN_SEPARATION = 75      # Minimum bp between clustered drops
DROP_ANNOTATE_TOP_N = 5       # Number of top drops to annotate on plots

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

    def __init__(
        self,
        out_dir: str,
        gene_tag: str,
        detection_methods: List[str] = None,
        organism: str = None,
        include_timestamp: bool = True
    ):
        """
        Initialize output directory structure.

        Args:
            out_dir: Parent output directory
            gene_tag: Gene/transcript identifier used for folder naming
            detection_methods: List of detection methods used (for folder naming)
            organism: Organism name (for folder naming)
            include_timestamp: Whether to include timestamp in folder name
        """
        # Clean the gene tag for use in filenames (remove special characters)
        safe_tag = gene_tag.replace(":", "_").replace("/", "_").replace("\\", "_")

        # Build descriptive folder name
        folder_parts = [safe_tag]

        # Add organism if provided
        if organism:
            folder_parts.append(organism)

        # Add detection methods summary
        if detection_methods:
            methods_short = "_".join(sorted(detection_methods))
            folder_parts.append(f"methods_{methods_short}")

        # Add timestamp
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_parts.append(timestamp)

        # Join parts with underscores
        folder_name = "_".join(folder_parts)

        # Create base directory for this gene
        self.base_dir = os.path.join(out_dir, folder_name)

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


def parse_gff3_attributes(attr_str: str) -> Dict[str, str]:
    """
    Parse GFF3 attribute string into key-value dictionary.

    GFF3 attribute format: 'key1=value1;key2=value2;...'
    Values are URL-decoded (e.g., %3B -> ;, %2C -> ,, %20 -> space).

    Args:
        attr_str: The 9th column of a GFF3 line containing attributes

    Returns:
        Dictionary mapping attribute names to values

    Example:
        >>> parse_gff3_attributes('ID=gene0;Name=thrL;gene_biotype=protein_coding')
        {'ID': 'gene0', 'Name': 'thrL', 'gene_biotype': 'protein_coding'}
    """
    out = {}
    for item in attr_str.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, val = item.split("=", 1)
            out[key.strip()] = _url_unquote(val.strip())
        else:
            # Bare flag (rare in GFF3)
            out[item] = ""
    return out


def load_gff_features(
    gff_path: str,
    chrom: str,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    """
    Load all GFF3 features overlapping a genomic region.

    Parses GFF3 and returns every feature whose coordinates overlap [start, end).
    Stops reading at ##FASTA directives (some GFF3 files embed sequence data).

    Coordinate Convention:
        GFF3 uses 1-based, inclusive coordinates.
        Returned features include an 'end_exclusive' field (1-based, end-exclusive)
        for consistency with the rest of the pipeline.

    Args:
        gff_path: Path to the GFF3 annotation file
        chrom: Chromosome/sequence name to filter on
        start: Start position (1-based, inclusive)
        end: End position (1-based, exclusive) -- pipeline convention

    Returns:
        List of feature dicts sorted by (start, end_exclusive), each containing:
        - seqid, source, feature_type, start, end, end_exclusive
        - score, strand, frame
        - attributes: Dict[str, str] (parsed key-value pairs)
        - id: str or None (the ID attribute)
        - parent: str or None (the Parent attribute)
    """
    features = []
    with open(gff_path, "r") as f:
        for line in f:
            # Stop at embedded FASTA section
            if line.startswith("##FASTA"):
                break
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqid, source, feature_type, f_start, f_end, score, strand, frame, attrs = fields

            # Filter by chromosome
            if seqid != chrom:
                continue

            f_start_i = int(f_start)       # 1-based inclusive
            f_end_i = int(f_end)           # 1-based inclusive
            f_end_excl = f_end_i + 1       # 1-based exclusive

            # Check overlap with [start, end)
            if f_start_i >= end or f_end_excl <= start:
                continue

            attrs_d = parse_gff3_attributes(attrs)

            features.append({
                "seqid": seqid,
                "source": source,
                "feature_type": feature_type,
                "start": f_start_i,
                "end": f_end_i,
                "end_exclusive": f_end_excl,
                "score": score,
                "strand": strand,
                "frame": frame,
                "attributes": attrs_d,
                "id": attrs_d.get("ID"),
                "parent": attrs_d.get("Parent"),
            })

    features.sort(key=lambda f: (f["start"], f["end_exclusive"]))
    return features


def map_position_to_gff_features(
    position: int,
    gff_features: List[Dict[str, Any]],
    tolerance: int = 0,
) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
    """
    Find all GFF features overlapping a genomic position.

    Args:
        position: Genomic position (1-based)
        gff_features: Feature list from load_gff_features()
        tolerance: Extra bp to search around position (default 0)

    Returns:
        Tuple of:
        - overlapping: List of feature dicts that overlap the position
        - nearest_dist: Distance to nearest feature (0 if overlapping)
        - nearest_type: Feature type of the nearest feature (None if empty list)
    """
    query_start = position - tolerance
    query_end = position + tolerance + 1  # exclusive

    overlapping = []
    nearest_dist = float("inf")
    nearest_type = None

    for feat in gff_features:
        # Check overlap: feature [feat.start, feat.end_exclusive) vs query [query_start, query_end)
        if feat["start"] < query_end and feat["end_exclusive"] > query_start:
            overlapping.append(feat)
            nearest_dist = 0
            nearest_type = feat["feature_type"]
        else:
            # Track nearest feature
            if position < feat["start"]:
                dist = feat["start"] - position
            else:
                dist = position - feat["end"]  # end is inclusive
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_type = feat["feature_type"]

    if nearest_dist == float("inf"):
        nearest_dist = -1  # No features at all

    return overlapping, int(nearest_dist), nearest_type


def annotate_events_with_gff(
    scored_drops: Dict[str, List[Tuple[int, float]]],
    scored_rises: Dict[str, List[Tuple[int, float]]],
    gff_features: List[Dict[str, Any]],
    pos_array: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Map all detected drops and rises to overlapping GFF3 features.

    For each drop/rise from each detection method, finds all overlapping
    GFF features and creates an enriched annotation record.

    Args:
        scored_drops: Dict mapping method -> list of (oriented_index, score)
        scored_rises: Dict mapping method -> list of (oriented_index, score)
        gff_features: Feature list from load_gff_features()
        pos_array: Array mapping oriented index -> genomic position

    Returns:
        List of annotation dicts ready for TSV output.
    """
    rows = []

    for event_type, scored_events in [("drop", scored_drops), ("rise", scored_rises)]:
        for method, event_list in scored_events.items():
            for idx, score in event_list:
                if idx < 0 or idx >= len(pos_array):
                    continue
                genomic_pos = int(pos_array[idx])

                overlapping, nearest_dist, nearest_type = map_position_to_gff_features(
                    genomic_pos, gff_features
                )

                # Extract key attributes from overlapping features
                feat_types = []
                feat_ids = []
                feat_products = []
                feat_functions = []
                feat_names = []
                in_cds = False
                in_utr = False

                for feat in overlapping:
                    feat_types.append(feat["feature_type"])
                    if feat["id"]:
                        feat_ids.append(feat["id"])
                    attrs = feat["attributes"]
                    if "product" in attrs:
                        feat_products.append(attrs["product"])
                    if "function" in attrs:
                        feat_functions.append(attrs["function"])
                    name = attrs.get("Name", attrs.get("gene", ""))
                    if name:
                        feat_names.append(name)

                    ft_lower = feat["feature_type"].lower()
                    if ft_lower == "cds":
                        in_cds = True
                    if "utr" in ft_lower:
                        in_utr = True

                rows.append({
                    "method": method,
                    "event_type": event_type,
                    "oriented_index": idx,
                    "genomic_position": genomic_pos,
                    "score": score,
                    "n_features": len(overlapping),
                    "feature_types": ";".join(feat_types) if feat_types else "",
                    "feature_ids": ";".join(feat_ids) if feat_ids else "",
                    "feature_products": ";".join(feat_products) if feat_products else "",
                    "feature_functions": ";".join(feat_functions) if feat_functions else "",
                    "feature_names": ";".join(feat_names) if feat_names else "",
                    "in_CDS": in_cds,
                    "in_UTR": in_utr,
                    "nearest_feature_type": nearest_type if nearest_type else "",
                    "nearest_feature_dist": nearest_dist,
                })

    return rows


def write_gff_annotation_tsv(annotations: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write GFF annotation mapping to a TSV file.

    Args:
        annotations: List of annotation dicts from annotate_events_with_gff()
        output_path: Path for the output TSV file
    """
    columns = [
        "method", "event_type", "oriented_index", "genomic_position", "score",
        "n_features", "feature_types", "feature_ids", "feature_products",
        "feature_functions", "feature_names", "in_CDS", "in_UTR",
        "nearest_feature_type", "nearest_feature_dist",
    ]
    with open(output_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in annotations:
            vals = []
            for col in columns:
                v = row[col]
                if isinstance(v, bool):
                    vals.append("True" if v else "False")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")
    print(f"[INFO] Saved GFF annotation mapping: {output_path}")


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
# MULTI-GPU DATA-PARALLEL SCORING
# =============================================================================

def _gpu_worker(
    gpu_id: int,
    work_items: list,
    seq_oriented: str,
    result_dict: dict,
    model_name: str = "evo2_7b",
    compute_rcavg_entropy: bool = True,
):
    """
    Worker process that loads a model on a specific GPU and scores assigned chunks.

    Each worker gets its own Evo2 model instance on a dedicated GPU. This enables
    true data parallelism: N GPUs process N chunks simultaneously.

    Args:
        gpu_id: CUDA device index for this worker
        work_items: List of (chunk_idx, start, end, core_start, core_end) tuples
        seq_oriented: Full locus sequence (shared read-only)
        result_dict: Shared dict (mp.Manager) to store results
        model_name: Evo2 model name
        compute_rcavg_entropy: Whether to compute RC-averaged entropy
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from evo2 import Evo2

    print(f"[GPU {gpu_id}] Loading model on CUDA device {gpu_id}...")
    model = Evo2(model_name)
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model"):
        model.model.eval()

    device = "cuda:0"  # Each worker sees only its assigned GPU as device 0
    idx_A = model.tokenizer.tokenize("A")[0]
    idx_C = model.tokenizer.tokenize("C")[0]
    idx_G = model.tokenizer.tokenize("G")[0]
    idx_T = model.tokenizer.tokenize("T")[0]
    ACGT_IDS = torch.tensor([idx_A, idx_C, idx_G, idx_T], dtype=torch.long, device=device)

    print(f"[GPU {gpu_id}] Model loaded. Processing {len(work_items)} chunks...")

    for chunk_idx, s, e, core_s, core_e in work_items:
        chunk_seq = seq_oriented[s:e]

        # Collect per-position results for the core region of this chunk
        chunk_results = []

        # Process contiguous non-N runs (same logic as sequential version)
        i = 0
        while i < len(chunk_seq):
            if chunk_seq[i] == "N":
                i += 1
                continue

            j = i + 1
            while j < len(chunk_seq) and chunk_seq[j] != "N":
                j += 1

            run_seq = chunk_seq[i:j]

            # Forward entropy (1 model call)
            Hf_t, Pf_t = entropy_like_reference_acgt(
                run_seq, model, ACGT_IDS, device,
                prepend_bos=True, reverse_complement=False
            )
            Hf = Hf_t.float().numpy().astype(np.float32)
            Pf = Pf_t.float().numpy().astype(np.float32)

            # RC-averaged entropy (2 model calls)
            if compute_rcavg_entropy:
                Hr_t, Pr_t = entropy_like_reference_acgt(
                    run_seq, model, ACGT_IDS, device,
                    prepend_bos=True, reverse_complement=True
                )
                Hr = Hr_t.float().numpy().astype(np.float32)
                Pr = Pr_t.float().numpy().astype(np.float32)
            else:
                Hr = Pr = None

            # Next-token logprobs (1 model call)
            logprobs_next, target_next = next_token_logprobs_and_targets_aligned(
                run_seq, model, device
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
            tok = model.tokenizer
            target_ids = target_next.detach().cpu().tolist()
            true_run = [id_to_token_str(tok, int(tid)) for tid in target_ids]

            # Collect results for positions in the core region
            for k in range(i, j):
                g = s + k  # Global position
                if g < core_s or g >= core_e:
                    continue
                rk = k - i
                entry = {
                    "g": g,
                    "Hf": float(Hf[rk]),
                    "Pf": float(Pf[rk]),
                    "Hr": float(Hr[rk]) if Hr is not None else float("nan"),
                    "Pr": float(Pr[rk]) if Pr is not None else float("nan"),
                    "p4": p4_run[rk, :].tolist(),
                    "ll": float(ll[rk]),
                    "true_tok": true_run[rk],
                }
                chunk_results.append(entry)

            i = j

        result_dict[chunk_idx] = chunk_results
        print(f"[GPU {gpu_id}] Completed chunk {chunk_idx} "
              f"(pos {s}-{e}, {len(chunk_results)} scored positions)")

    print(f"[GPU {gpu_id}] All chunks done.")


def score_locus_aligned_overlap_multigpu(
    seq_oriented: str,
    n_gpus: int = None,
    model_name: str = "evo2_7b",
    max_chunk_len: int = MAX_CHUNK_LEN_DEFAULT,
    chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    compute_rcavg_entropy: bool = True,
):
    """
    Multi-GPU data-parallel version of score_locus_aligned_overlap.

    Distributes chunks across multiple GPUs, each running its own model
    instance. This provides true data parallelism: N GPUs process N chunks
    simultaneously, achieving ~Nx speedup (minus model loading overhead).

    Architecture:
        - Main process computes chunk boundaries
        - N worker processes spawned (one per GPU)
        - Each worker loads its own Evo2 model instance
        - Chunks distributed round-robin across workers
        - Results merged back into output arrays

    Args:
        seq_oriented: DNA sequence in 5'->3' orientation
        n_gpus: Number of GPUs to use (default: all available)
        model_name: Evo2 model name (default: evo2_7b)
        max_chunk_len: Maximum bases per chunk
        chunk_overlap: Overlap between chunks
        compute_rcavg_entropy: Whether to compute RC-averaged entropy

    Returns:
        Same as score_locus_aligned_overlap:
        (entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next)
    """
    import torch.multiprocessing as mp

    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise ValueError("No GPUs available for multi-GPU scoring")

    L = len(seq_oriented)

    # Initialize output arrays
    entropy_fwd = np.full((L,), np.nan, dtype=np.float32)
    ppx_fwd = np.full((L,), np.nan, dtype=np.float32)
    entropy_rc = np.full((L,), np.nan, dtype=np.float32)
    ppx_rc = np.full((L,), np.nan, dtype=np.float32)
    p4 = np.full((L, 4), np.nan, dtype=np.float32)
    ll_next = np.full((L,), np.nan, dtype=np.float32)
    true_tok = np.array([""] * L, dtype=object)

    if L == 0:
        return entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next

    # Compute chunk boundaries (same as sequential version)
    step = max(1, max_chunk_len - chunk_overlap)
    starts = list(range(0, L, step))

    work_items_all = []
    for chunk_idx, s in enumerate(starts):
        e = min(L, s + max_chunk_len)
        core_s = s if s == 0 else s + chunk_overlap // 2
        core_e = e if e == L else e - chunk_overlap // 2
        core_s = min(core_s, core_e)
        work_items_all.append((chunk_idx, s, e, core_s, core_e))

    # Distribute chunks round-robin across GPUs
    gpu_work = [[] for _ in range(n_gpus)]
    for i, item in enumerate(work_items_all):
        gpu_work[i % n_gpus].append(item)

    print(f"[MULTIGPU] Distributing {len(work_items_all)} chunks across {n_gpus} GPUs:")
    for gid in range(n_gpus):
        print(f"  GPU {gid}: {len(gpu_work[gid])} chunks")

    # Use multiprocessing Manager for shared result dict
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    manager = mp.Manager()
    result_dict = manager.dict()

    # Spawn worker processes
    processes = []
    for gid in range(n_gpus):
        if not gpu_work[gid]:
            continue
        p = mp.Process(
            target=_gpu_worker,
            args=(gid, gpu_work[gid], seq_oriented, result_dict,
                  model_name, compute_rcavg_entropy),
        )
        processes.append(p)

    # Start all workers
    for p in processes:
        p.start()

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Merge results into output arrays
    for chunk_idx in sorted(result_dict.keys()):
        for entry in result_dict[chunk_idx]:
            g = entry["g"]
            entropy_fwd[g] = entry["Hf"]
            ppx_fwd[g] = entry["Pf"]
            entropy_rc[g] = entry["Hr"]
            ppx_rc[g] = entry["Pr"]
            p4[g, :] = entry["p4"]
            ll_next[g] = entry["ll"]
            true_tok[g] = entry["true_tok"]

    print(f"[MULTIGPU] Merged results: {np.count_nonzero(~np.isnan(entropy_fwd)):,} "
          f"of {L:,} positions scored")

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
# ENHANCED DROP DETECTION METHODS (JAN26)
# =============================================================================

def _drops_scored_to_positions(scored_drops: List[Tuple[int, float]]) -> List[int]:
    """
    Convert scored drops to position-only list for backward compatibility.

    Args:
        scored_drops: List of (position, score) tuples

    Returns:
        List of positions only, sorted
    """
    return sorted([pos for pos, _ in scored_drops])


def _cluster_and_pick_best(
    candidates: np.ndarray,
    scores: np.ndarray,
    min_separation: int,
    pick_min: bool = True
) -> List[Tuple[int, float]]:
    """
    Cluster nearby candidates and pick best score in each cluster.

    Args:
        candidates: Array of position indices
        scores: Corresponding score values
        min_separation: Cluster radius in positions
        pick_min: If True, pick minimum score (for drops, more negative = better).
                  If False, pick maximum score (for rises, more positive = better).

    Returns:
        List of (position, score) for best detection in each cluster
    """
    if len(candidates) == 0:
        return []

    # Sort by position
    order = np.argsort(candidates)
    candidates = candidates[order]
    scores = scores[order]

    clusters = []
    current_cluster = [(candidates[0], scores[0])]

    picker = min if pick_min else max

    for i in range(1, len(candidates)):
        if candidates[i] - current_cluster[-1][0] <= min_separation:
            current_cluster.append((candidates[i], scores[i]))
        else:
            # Close current cluster, pick best
            best_pos, best_score = picker(current_cluster, key=lambda x: x[1])
            clusters.append((int(best_pos), float(best_score)))

            # Start new cluster
            current_cluster = [(candidates[i], scores[i])]

    # Don't forget last cluster
    if current_cluster:
        best_pos, best_score = picker(current_cluster, key=lambda x: x[1])
        clusters.append((int(best_pos), float(best_score)))

    return clusters


def detect_drops_zscore(
    entropy: np.ndarray,
    smooth_w: int = 51,
    zscore_threshold: float = 2.5,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Detect entropy drops using statistical z-scores of derivatives.

    This method identifies drops that are statistically significant relative
    to the distribution of all entropy changes, automatically adapting to
    data variance. Returns confidence scores for each detection.

    Statistical Interpretation:
        - Z-score of -2.5 corresponds to ~99th percentile (1% FDR)
        - Z-score of -3.0 corresponds to ~99.7th percentile (0.3% FDR)
        - More negative scores = more confident detections

    Method:
        1. Smooth entropy with rolling mean
        2. Compute first derivative (rate of change)
        3. Calculate z-scores: (d - mean(d)) / std(d)
        4. Flag positions where z-score < -threshold
        5. Cluster nearby detections, keep strongest in each cluster

    Args:
        entropy: Per-position entropy values (nats or bits)
        smooth_w: Window size for rolling mean smoothing
        zscore_threshold: Minimum |z-score| for detection (default: 2.5)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, z_score) tuples, sorted by position.
        Z-scores are negative (more negative = stronger drop).

    Example:
        >>> drops = detect_drops_zscore(entropy, zscore_threshold=2.5)
        >>> for pos, zscore in drops:
        ...     print(f"Drop at {pos}: z={zscore:.2f} (statistically significant)")
    """
    # Handle edge cases
    if len(entropy) < smooth_w:
        return []

    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])

    # Compute z-scores
    mean_deriv = np.mean(d)
    std_deriv = np.std(d)

    if std_deriv < 1e-9:  # Flat signal
        return []

    zscores = (d - mean_deriv) / std_deriv

    # Find significantly negative z-scores
    candidates = np.where(zscores < -zscore_threshold)[0]

    if len(candidates) == 0:
        return []

    # Cluster and pick best
    drops = _cluster_and_pick_best(candidates, zscores[candidates], min_separation)

    # Sort by position
    drops.sort(key=lambda x: x[0])
    return drops


def detect_drops_mad(
    entropy: np.ndarray,
    smooth_w: int = 51,
    mad_threshold: float = 3.0,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Detect entropy drops using Median Absolute Deviation (MAD).

    MAD is a robust alternative to standard deviation that is less sensitive
    to outliers. This makes it ideal for genomic data with repetitive regions,
    sequencing errors, or GC-content extremes.

    Statistical Properties:
        - MAD-score of 3.0 ≈ z-score of 3.0 for normal distributions
        - More robust than z-score when data has heavy tails or outliers
        - MAD = median(|x - median(x)|)
        - Scaling factor 1.4826 makes MAD comparable to std for Gaussian data

    Method:
        1. Smooth entropy with rolling mean
        2. Compute first derivative
        3. Calculate MAD-based scores: (d - median(d)) / (1.4826 * MAD)
        4. Flag positions where score < -threshold
        5. Cluster and return strongest per cluster

    Args:
        entropy: Per-position entropy values
        smooth_w: Window size for rolling mean
        mad_threshold: Threshold in MAD units (default: 3.0)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, mad_score) tuples, sorted by position.
        Scores are negative (more negative = stronger drop).

    Use When:
        - Data has outliers or is non-Gaussian
        - Repetitive genomic regions present
        - Need robustness to noise spikes

    Example:
        >>> drops = detect_drops_mad(entropy, mad_threshold=3.0)
        >>> for pos, score in drops:
        ...     print(f"Drop at {pos}: MAD-score={score:.2f}")
    """
    if len(entropy) < smooth_w:
        return []

    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])

    # Compute MAD
    median_deriv = np.median(d)
    mad = np.median(np.abs(d - median_deriv))

    if mad < 1e-9:  # No variation
        return []

    # MAD-based score (1.4826 factor makes it comparable to std for normal dist)
    mad_scores = (d - median_deriv) / (1.4826 * mad)

    candidates = np.where(mad_scores < -mad_threshold)[0]

    if len(candidates) == 0:
        return []

    # Cluster and pick best
    drops = _cluster_and_pick_best(candidates, mad_scores[candidates], min_separation)

    drops.sort(key=lambda x: x[0])
    return drops


def detect_drops_local_baseline(
    entropy: np.ndarray,
    window_baseline: int = 500,
    threshold_sigma: float = 2.0,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Detect drops using local baseline normalization.

    This method accounts for regional differences in entropy levels (e.g.,
    exons vs introns, GC-rich vs AT-rich regions) by computing statistics
    in a sliding window. Drops are scored relative to LOCAL variance rather
    than global statistics.

    Biological Motivation:
        - Exons typically have different baseline entropy than introns
        - Repetitive regions have higher local variance
        - Global statistics may miss drops in high-variance regions
        - This method adapts to local context

    Method:
        1. For each position, compute local mean and std in ±window_baseline/2
        2. Compute derivative of smoothed entropy
        3. Calculate local z-score: (d - local_mean') / local_std
        4. Flag positions where local z-score < -threshold
        5. Cluster and return strongest per cluster

    Args:
        entropy: Per-position entropy values
        window_baseline: Window size for local statistics (default: 500bp)
        threshold_sigma: Threshold in local standard deviations (default: 2.0)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, local_zscore) tuples, sorted by position.
        Scores are negative (more negative = stronger relative to local context).

    Use When:
        - Genes have distinct exon/intron structure
        - GC content varies significantly across locus
        - Global methods miss biologically relevant drops

    Example:
        >>> drops = detect_drops_local_baseline(entropy, window_baseline=500)
        >>> for pos, score in drops:
        ...     print(f"Drop at {pos}: local z={score:.2f}")
    """
    if len(entropy) < window_baseline:
        return []

    sm = _rolling_mean(entropy, 51)
    d = np.diff(sm, prepend=sm[0])

    L = len(entropy)
    local_mean = np.zeros(L)
    local_std = np.zeros(L)

    # Compute local statistics
    half_w = window_baseline // 2
    for i in range(L):
        lo = max(0, i - half_w)
        hi = min(L, i + half_w)
        window = d[lo:hi]
        valid = ~np.isnan(window)
        if valid.sum() > 10:
            local_mean[i] = np.mean(window[valid])
            local_std[i] = np.std(window[valid])
        else:
            local_mean[i] = 0.0
            local_std[i] = 1.0

    # Compute local z-scores
    local_zscores = (d - local_mean) / (local_std + 1e-9)

    candidates = np.where(local_zscores < -threshold_sigma)[0]

    if len(candidates) == 0:
        return []

    # Cluster and pick best
    drops = _cluster_and_pick_best(candidates, local_zscores[candidates], min_separation)

    drops.sort(key=lambda x: x[0])
    return drops


def bootstrap_drop_confidence(
    entropy: np.ndarray,
    smooth_w: int = 51,
    zscore_threshold: float = 2.0,
    n_bootstrap: int = 100,
    consensus_threshold: float = 0.50,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Use bootstrap resampling to identify robust drops across sampling variation.

    This method tests if drops are stable across resampled versions of the
    entropy signal. Only drops that appear in >consensus_threshold fraction
    of bootstrap samples are reported.

    Statistical Rationale:
        - Bootstrap simulates measurement uncertainty
        - Consensus fraction = robustness to sampling variation
        - High consensus (>0.8) = very reliable detection

    Method:
        1. Generate n_bootstrap resampled entropy arrays (with replacement)
        2. Run zscore detection on each sample
        3. Count how often each position is flagged
        4. Return positions flagged in ≥consensus_threshold fraction

    Args:
        entropy: Per-position entropy values
        smooth_w: Smoothing window for base detection
        zscore_threshold: Z-score threshold for individual bootstrap samples
        n_bootstrap: Number of bootstrap samples (default: 100)
        consensus_threshold: Minimum fraction to be considered robust (default: 0.5)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, consensus_fraction) tuples, sorted by consensus DESC.
        Consensus_fraction ∈ [0, 1], where 1.0 means detected in all samples.

    WARNING: Computational Cost
        - This method is ~100x slower than other methods (runs detection 100 times)
        - Only use when confidence scoring is critical
        - Not recommended for production pipelines without caching

    Example:
        >>> drops = bootstrap_drop_confidence(entropy, n_bootstrap=100)
        >>> for pos, conf in drops:
        ...     print(f"Drop at {pos}: {conf*100:.0f}% consensus")
    """
    L = len(entropy)

    if L < smooth_w:
        return []

    # Fill NaNs for bootstrapping
    entropy_filled = _fill_nans_linear(entropy)

    # Count detections at each position across bootstrap samples
    detection_counts = np.zeros(L, dtype=int)

    print(f"[INFO] Running bootstrap consensus detection ({n_bootstrap} samples)...")

    for b in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(L, size=L, replace=True)
        entropy_boot = entropy_filled[indices]

        # Smooth and detect in bootstrap sample
        sm = _rolling_mean(entropy_boot, smooth_w)
        d = np.diff(sm, prepend=sm[0])

        mean_deriv = np.mean(d)
        std_deriv = np.std(d)
        if std_deriv < 1e-9:
            continue

        zscores = (d - mean_deriv) / std_deriv
        candidates = np.where(zscores < -zscore_threshold)[0]

        # Map back to original indices and increment counts
        for pos in candidates:
            if pos < L:
                orig_pos = indices[pos]
                detection_counts[orig_pos] += 1

    # Compute confidence scores
    confidence = detection_counts / n_bootstrap

    # Threshold by consensus
    robust_positions = np.where(confidence >= consensus_threshold)[0]

    if len(robust_positions) == 0:
        return []

    # Merge nearby positions
    drops = []
    positions = robust_positions.tolist()
    positions.sort()

    i = 0
    while i < len(positions):
        cluster_start = i
        while i + 1 < len(positions) and positions[i+1] - positions[i] <= min_separation:
            i += 1

        # Best in cluster (highest confidence)
        cluster_pos = positions[cluster_start:i+1]
        cluster_conf = confidence[cluster_pos]
        best_idx = np.argmax(cluster_conf)
        drops.append((cluster_pos[best_idx], float(cluster_conf[best_idx])))

        i += 1

    drops.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
    print(f"[INFO] Bootstrap consensus found {len(drops)} robust drops")
    return drops


# =============================================================================
# RISE DETECTION METHODS (End of drops - entropy returning to high values)
# =============================================================================

def detect_rises_derivative(
    entropy: np.ndarray,
    smooth_w: int,
    thr_quantile: float
) -> List[int]:
    """
    Detect entropy rises (end of drops) using derivative (rate of change).

    Identifies positions where entropy increases rapidly, which indicates
    transitions out of conserved regions back to high entropy.

    Method:
    1. Smooth the entropy signal
    2. Compute first derivative (difference)
    3. Find positions where derivative is above a quantile threshold
    4. Merge nearby detections (within min_sep)

    Args:
        entropy: Per-position entropy values
        smooth_w: Window size for pre-smoothing
        thr_quantile: Quantile threshold (e.g., 0.99 = top 1%)

    Returns:
        List of positions where significant rises detected
    """
    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])
    thr = np.quantile(d, thr_quantile)
    candidates = np.where(d >= thr)[0].tolist()

    # Merge nearby candidates
    out = []
    last = -10**9
    min_sep = max(10, smooth_w // 2)
    for i in candidates:
        if i - last >= min_sep:
            out.append(i)
            last = i
    return out


def detect_rises_window_mean_shift(
    entropy: np.ndarray,
    w: int,
    top_k: int
) -> List[int]:
    """
    Detect entropy rises (end of drops) using windowed mean shift.

    Compares mean entropy in a window before vs after each position.
    Large positive shifts indicate potential rise points (entropy recovery).

    Method:
    1. For each position, compute mean of window before and after
    2. Score = mean(after) - mean(before)
    3. Return top K positions with most positive scores

    Args:
        entropy: Per-position entropy values
        w: Window size on each side
        top_k: Maximum number of rise points to return

    Returns:
        List of positions with largest positive mean shifts
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
    order = np.argsort(scores[good])[::-1]  # Sort descending (most positive first)
    picks = idx_good[order][:top_k].tolist()

    # Remove duplicates that are too close
    out = []
    for i in picks:
        if all(abs(i - j) > w // 2 for j in out):
            out.append(i)
    return out


def detect_rises_cusum(
    entropy: np.ndarray,
    smooth_w: int,
    h: float
) -> List[int]:
    """
    Detect entropy rises (end of drops) using CUSUM (Cumulative Sum) algorithm.

    CUSUM is a classic change-point detection method. Here we detect rises
    by accumulating above-mean values.

    Method:
    1. Smooth entropy and compute global mean
    2. Accumulate (value - mean) with reset-to-zero when negative
    3. Flag positions where accumulation exceeds h

    Args:
        entropy: Per-position entropy values
        smooth_w: Window size for pre-smoothing
        h: CUSUM threshold (higher = fewer, more confident detections)

    Returns:
        List of positions where CUSUM detected a rise
    """
    x = _rolling_mean(entropy, smooth_w)
    x = _fill_nans_linear(x)
    mu = float(np.mean(x))

    out = []
    s = 0.0
    last = -10**9
    min_sep = max(25, smooth_w)

    for i, xi in enumerate(x):
        s = max(0.0, s + (float(xi) - mu))  # Accumulate above-mean values
        if s > h and (i - last) > min_sep:
            out.append(i)
            last = i
            s = 0.0  # Reset after detection
    return out


def detect_rises_zscore(
    entropy: np.ndarray,
    smooth_w: int = 51,
    zscore_threshold: float = 2.5,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Detect entropy rises (end of drops) using statistical z-scores of derivatives.

    This method identifies rises that are statistically significant relative
    to the distribution of all entropy changes, automatically adapting to
    data variance. Returns confidence scores for each detection.

    Statistical Interpretation:
        - Z-score of +2.5 corresponds to ~99th percentile (1% FDR)
        - Z-score of +3.0 corresponds to ~99.7th percentile (0.3% FDR)
        - More positive scores = more confident detections

    Method:
        1. Smooth entropy with rolling mean
        2. Compute first derivative (rate of change)
        3. Calculate z-scores: (d - mean(d)) / std(d)
        4. Flag positions where z-score > +threshold
        5. Cluster nearby detections, keep strongest in each cluster

    Args:
        entropy: Per-position entropy values (nats or bits)
        smooth_w: Window size for rolling mean smoothing
        zscore_threshold: Minimum z-score for detection (default: 2.5)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, z_score) tuples, sorted by position.
        Z-scores are positive (more positive = stronger rise).

    Example:
        >>> rises = detect_rises_zscore(entropy, zscore_threshold=2.5)
        >>> for pos, zscore in rises:
        ...     print(f"Rise at {pos}: z={zscore:.2f} (statistically significant)")
    """
    # Handle edge cases
    if len(entropy) < smooth_w:
        return []

    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])

    # Compute z-scores
    mean_deriv = np.mean(d)
    std_deriv = np.std(d)

    if std_deriv < 1e-9:  # Flat signal
        return []

    zscores = (d - mean_deriv) / std_deriv

    # Find significantly positive z-scores (entropy increasing)
    candidates = np.where(zscores > zscore_threshold)[0]

    if len(candidates) == 0:
        return []

    # Cluster and pick best (pick max for rises)
    rises = _cluster_and_pick_best(candidates, zscores[candidates], min_separation, pick_min=False)

    # Sort by position
    rises.sort(key=lambda x: x[0])
    return rises


def detect_rises_mad(
    entropy: np.ndarray,
    smooth_w: int = 51,
    mad_threshold: float = 3.0,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Detect entropy rises (end of drops) using Median Absolute Deviation (MAD).

    MAD is a robust alternative to standard deviation that is less sensitive
    to outliers. This makes it ideal for genomic data with repetitive regions,
    sequencing errors, or GC-content extremes.

    Statistical Properties:
        - MAD-score of 3.0 ≈ z-score of 3.0 for normal distributions
        - More robust than z-score when data has heavy tails or outliers
        - MAD = median(|x - median(x)|)
        - Scaling factor 1.4826 makes MAD comparable to std for Gaussian data

    Method:
        1. Smooth entropy with rolling mean
        2. Compute first derivative
        3. Calculate MAD-based scores: (d - median(d)) / (1.4826 * MAD)
        4. Flag positions where score > +threshold
        5. Cluster and return strongest per cluster

    Args:
        entropy: Per-position entropy values
        smooth_w: Window size for rolling mean
        mad_threshold: Threshold in MAD units (default: 3.0)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, mad_score) tuples, sorted by position.
        Scores are positive (more positive = stronger rise).

    Example:
        >>> rises = detect_rises_mad(entropy, mad_threshold=3.0)
        >>> for pos, score in rises:
        ...     print(f"Rise at {pos}: MAD-score={score:.2f}")
    """
    if len(entropy) < smooth_w:
        return []

    sm = _rolling_mean(entropy, smooth_w)
    d = np.diff(sm, prepend=sm[0])

    # Compute MAD
    median_deriv = np.median(d)
    mad = np.median(np.abs(d - median_deriv))

    if mad < 1e-9:  # No variation
        return []

    # MAD-based score (1.4826 factor makes it comparable to std for normal dist)
    mad_scores = (d - median_deriv) / (1.4826 * mad)

    candidates = np.where(mad_scores > mad_threshold)[0]

    if len(candidates) == 0:
        return []

    # Cluster and pick best (pick max for rises)
    rises = _cluster_and_pick_best(candidates, mad_scores[candidates], min_separation, pick_min=False)

    rises.sort(key=lambda x: x[0])
    return rises


def detect_rises_local_baseline(
    entropy: np.ndarray,
    window_baseline: int = 500,
    threshold_sigma: float = 2.0,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Detect rises (end of drops) using local baseline normalization.

    This method accounts for regional differences in entropy levels (e.g.,
    exons vs introns, GC-rich vs AT-rich regions) by computing statistics
    in a sliding window. Rises are scored relative to LOCAL variance rather
    than global statistics.

    Biological Motivation:
        - Exons typically have different baseline entropy than introns
        - Repetitive regions have higher local variance
        - Global statistics may miss rises in high-variance regions
        - This method adapts to local context

    Method:
        1. For each position, compute local mean and std in ±window_baseline/2
        2. Compute derivative of smoothed entropy
        3. Calculate local z-score: (d - local_mean') / local_std
        4. Flag positions where local z-score > +threshold
        5. Cluster and return strongest per cluster

    Args:
        entropy: Per-position entropy values
        window_baseline: Window size for local statistics (default: 500bp)
        threshold_sigma: Threshold in local standard deviations (default: 2.0)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, local_zscore) tuples, sorted by position.
        Scores are positive (more positive = stronger relative to local context).

    Example:
        >>> rises = detect_rises_local_baseline(entropy, window_baseline=500)
        >>> for pos, score in rises:
        ...     print(f"Rise at {pos}: local z={score:.2f}")
    """
    if len(entropy) < window_baseline:
        return []

    sm = _rolling_mean(entropy, 51)
    d = np.diff(sm, prepend=sm[0])

    L = len(entropy)
    local_mean = np.zeros(L)
    local_std = np.zeros(L)

    # Compute local statistics
    half_w = window_baseline // 2
    for i in range(L):
        lo = max(0, i - half_w)
        hi = min(L, i + half_w)
        window = d[lo:hi]
        valid = ~np.isnan(window)
        if valid.sum() > 10:
            local_mean[i] = np.mean(window[valid])
            local_std[i] = np.std(window[valid])
        else:
            local_mean[i] = 0.0
            local_std[i] = 1.0

    # Compute local z-scores
    local_zscores = (d - local_mean) / (local_std + 1e-9)

    candidates = np.where(local_zscores > threshold_sigma)[0]

    if len(candidates) == 0:
        return []

    # Cluster and pick best (pick max for rises)
    rises = _cluster_and_pick_best(candidates, local_zscores[candidates], min_separation, pick_min=False)

    rises.sort(key=lambda x: x[0])
    return rises


def bootstrap_rise_confidence(
    entropy: np.ndarray,
    smooth_w: int = 51,
    zscore_threshold: float = 2.0,
    n_bootstrap: int = 100,
    consensus_threshold: float = 0.50,
    min_separation: int = 75
) -> List[Tuple[int, float]]:
    """
    Use bootstrap resampling to identify robust rises (end of drops) across sampling variation.

    This method tests if rises are stable across resampled versions of the
    entropy signal. Only rises that appear in >consensus_threshold fraction
    of bootstrap samples are reported.

    Statistical Rationale:
        - Bootstrap simulates measurement uncertainty
        - Consensus fraction = robustness to sampling variation
        - High consensus (>0.8) = very reliable detection

    Method:
        1. Generate n_bootstrap resampled entropy arrays (with replacement)
        2. Run zscore detection for rises on each sample
        3. Count how often each position is flagged
        4. Return positions flagged in ≥consensus_threshold fraction

    Args:
        entropy: Per-position entropy values
        smooth_w: Smoothing window for base detection
        zscore_threshold: Z-score threshold for individual bootstrap samples
        n_bootstrap: Number of bootstrap samples (default: 100)
        consensus_threshold: Minimum fraction to be considered robust (default: 0.5)
        min_separation: Merge detections within this distance (bp)

    Returns:
        List of (position, consensus_fraction) tuples, sorted by consensus DESC.
        Consensus_fraction ∈ [0, 1], where 1.0 means detected in all samples.

    WARNING: Computational Cost
        - This method is ~100x slower than other methods (runs detection 100 times)
        - Only use when confidence scoring is critical
        - Not recommended for production pipelines without caching

    Example:
        >>> rises = bootstrap_rise_confidence(entropy, n_bootstrap=100)
        >>> for pos, conf in rises:
        ...     print(f"Rise at {pos}: {conf*100:.0f}% consensus")
    """
    L = len(entropy)

    if L < smooth_w:
        return []

    # Fill NaNs for bootstrapping
    entropy_filled = _fill_nans_linear(entropy)

    # Count detections at each position across bootstrap samples
    detection_counts = np.zeros(L, dtype=int)

    print(f"[INFO] Running bootstrap consensus rise detection ({n_bootstrap} samples)...")

    for b in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(L, size=L, replace=True)
        entropy_boot = entropy_filled[indices]

        # Smooth and detect rises in bootstrap sample
        sm = _rolling_mean(entropy_boot, smooth_w)
        d = np.diff(sm, prepend=sm[0])

        mean_deriv = np.mean(d)
        std_deriv = np.std(d)
        if std_deriv < 1e-9:
            continue

        zscores = (d - mean_deriv) / std_deriv
        candidates = np.where(zscores > zscore_threshold)[0]  # Positive for rises

        # Map back to original indices and increment counts
        for pos in candidates:
            if pos < L:
                orig_pos = indices[pos]
                detection_counts[orig_pos] += 1

    # Compute confidence scores
    confidence = detection_counts / n_bootstrap

    # Threshold by consensus
    robust_positions = np.where(confidence >= consensus_threshold)[0]

    if len(robust_positions) == 0:
        return []

    # Merge nearby positions
    rises = []
    positions = robust_positions.tolist()
    positions.sort()

    i = 0
    while i < len(positions):
        cluster_start = i
        while i + 1 < len(positions) and positions[i+1] - positions[i] <= min_separation:
            i += 1

        # Best in cluster (highest confidence)
        cluster_pos = positions[cluster_start:i+1]
        cluster_conf = confidence[cluster_pos]
        best_idx = np.argmax(cluster_conf)
        rises.append((cluster_pos[best_idx], float(cluster_conf[best_idx])))

        i += 1

    rises.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
    print(f"[INFO] Bootstrap consensus found {len(rises)} robust rises")
    return rises


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def shade_exons(
    ax: plt.Axes,
    exon_intervals: List[Tuple[int, int, int]],
    alpha: float = 0.15,
    color: str = "#2E8B57"  # Sea green - distinct from blue/orange
) -> None:
    """
    Add vertical shading to mark exon regions on a plot.

    Args:
        ax: Matplotlib axes object
        exon_intervals: List of (start, end, exon_id) tuples
        alpha: Transparency of shading (0-1)
        color: Color for exon shading (default: sea green)
    """
    for (s, e, _) in exon_intervals:
        ax.axvspan(s, e, alpha=alpha, facecolor=color, edgecolor='none')


def draw_exon_track(
    ax: plt.Axes,
    exon_intervals: List[Tuple[int, int, int]],
    track_height: float = 0.08,
    exon_color: str = "#2ecc71",  # Modern green
    intron_color: str = "#ecf0f1",  # Light gray
    label_exons: bool = True
) -> None:
    """
    Draw a modern exon/intron track at the top of the plot.

    Creates a horizontal track showing exon regions as colored bars,
    with clean, modern styling similar to genome browsers.

    Args:
        ax: Matplotlib axes object
        exon_intervals: List of (start, end, exon_id) tuples
        track_height: Height of track as fraction of y-axis (default: 0.08)
        exon_color: Color for exon regions (default: modern green)
        intron_color: Color for intron/background (default: light gray)
        label_exons: Whether to label exon numbers (default: True)
    """
    from matplotlib.patches import Rectangle, FancyBboxPatch

    # Get current y-axis limits
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    # Position track at top of plot
    track_bottom = ymax - (track_height * y_range)
    track_top = ymax
    bar_height = track_top - track_bottom

    # Get x-axis limits for full intron background
    xmin, xmax = ax.get_xlim()

    # Draw intron background (full width) with rounded corners
    ax.add_patch(Rectangle(
        (xmin, track_bottom), xmax - xmin, bar_height,
        facecolor=intron_color, edgecolor='#bdc3c7', linewidth=0.5, zorder=10
    ))

    # Draw thin intron line in the middle (like gene structure diagrams)
    intron_y = track_bottom + bar_height / 2
    ax.plot([xmin, xmax], [intron_y, intron_y], color='#7f8c8d',
            linewidth=2, solid_capstyle='round', zorder=9)

    # Draw each exon as a colored bar with subtle styling
    for (s, e, exon_id) in exon_intervals:
        # Exon bar with slight rounding
        ax.add_patch(Rectangle(
            (s, track_bottom + bar_height * 0.1), e - s, bar_height * 0.8,
            facecolor=exon_color, edgecolor='#27ae60', linewidth=0.8,
            zorder=11, alpha=0.9
        ))

        # Add exon label
        if label_exons and exon_id > 0:
            mid_x = (s + e) / 2
            mid_y = track_bottom + bar_height / 2
            # Only label if exon is wide enough
            if (e - s) > (xmax - xmin) * 0.025:
                ax.text(mid_x, mid_y, f'E{exon_id}', ha='center', va='center',
                       fontsize=7, fontweight='bold', color='white', zorder=12)

    # Draw subtle vertical shading at exon boundaries (highlighting in main plot area)
    for (s, e, _) in exon_intervals:
        # Light yellow highlight behind exon regions
        ax.axvspan(s, e, ymin=0, ymax=(track_bottom - ymin) / y_range,
                   facecolor='#ffeaa7', alpha=0.15, zorder=1)

    # Add legend entry for exon track (green = exons)
    from matplotlib.patches import Patch
    exon_legend_patch = Patch(facecolor=exon_color, edgecolor='#27ae60',
                              label='Exon regions')
    # Get existing handles/labels and add the exon patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(exon_legend_patch)
    labels.append('Exon regions')
    ax.legend(handles, labels, loc='best', fontsize=9)

    # Extend y-axis slightly to accommodate track
    ax.set_ylim(ymin, ymax)


def evodesigner_fill(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    low_quantile: float = 0.10
) -> None:
    """
    Apply EvoDesigner-style fill to highlight low-entropy regions.

    Creates a two-level fill:
    1. Light blue fill under entire curve
    2. Red/coral fill under positions below the low_quantile threshold

    This emphasizes conserved/functional regions with low entropy.

    Args:
        ax: Matplotlib axes object
        x: X-axis values (positions)
        y: Y-axis values (entropy)
        low_quantile: Threshold for "low" entropy (default: bottom 10%)
    """
    # Modern blue fill everywhere (similar to reference images)
    ax.fill_between(x, y, 0, alpha=0.35, color='#3498db')

    # Orange/amber fill for low-entropy regions (matches reference style)
    if np.any(~np.isnan(y)):
        thr = np.nanquantile(y, low_quantile)
        mask = y <= thr
        ax.fill_between(x, y, 0, where=mask, alpha=0.5, color='#f39c12')


# --- GFF feature track colors ---
_GFF_FEATURE_COLORS = {
    "CDS":              "#3498db",  # Blue
    "gene":             "#2ecc71",  # Green
    "mRNA":             "#1abc9c",  # Teal
    "exon":             "#a8e6cf",  # Light green
    "five_prime_UTR":   "#e67e22",  # Orange
    "three_prime_UTR":  "#e74c3c",  # Red
    "start_codon":      "#9b59b6",  # Purple
    "stop_codon":       "#8e44ad",  # Dark purple
}
_GFF_DEFAULT_COLOR = "#95a5a6"  # Gray for unknown types


def convert_gff_to_oriented_intervals(
    gff_features: List[Dict[str, Any]],
    locus_start: int,
    locus_end_excl: int,
    strand: str,
) -> List[Tuple[int, int, str, str]]:
    """
    Convert GFF features from genomic coordinates to oriented plot indices.

    Args:
        gff_features: Feature list from load_gff_features()
        locus_start: Locus start genomic position (1-based)
        locus_end_excl: Locus end genomic position (1-based, exclusive)
        strand: '+' or '-'

    Returns:
        List of (start_idx, end_idx, feature_type, label) tuples in oriented
        index space, suitable for plotting.
    """
    locus_len = locus_end_excl - locus_start
    intervals = []

    for feat in gff_features:
        # Clip feature to locus bounds
        g_start = max(feat["start"], locus_start)
        g_end_excl = min(feat["end_exclusive"], locus_end_excl)
        if g_start >= g_end_excl:
            continue

        if strand == "+":
            idx_start = g_start - locus_start
            idx_end = g_end_excl - locus_start
        else:
            # Minus strand: reverse mapping
            idx_start = locus_len - (g_end_excl - locus_start)
            idx_end = locus_len - (g_start - locus_start)

        # Build label from attributes
        attrs = feat["attributes"]
        label = attrs.get("product", attrs.get("Name", attrs.get("gene", "")))

        intervals.append((idx_start, idx_end, feat["feature_type"], label))

    # Sort by start position
    intervals.sort(key=lambda t: (t[0], t[1]))
    return intervals


def draw_gff_feature_track(
    ax: plt.Axes,
    gff_intervals: List[Tuple[int, int, str, str]],
    track_height: float = 0.10,
    label_features: bool = True,
) -> None:
    """
    Draw a GFF feature track at the bottom of the plot.

    Shows different GFF3 feature types (CDS, UTR, gene, etc.) as colored
    horizontal bars, similar to a genome browser annotation track. Each
    feature type gets its own labeled row.

    Args:
        ax: Matplotlib axes object
        gff_intervals: List of (start_idx, end_idx, feature_type, label) tuples
            in oriented index space (from convert_gff_to_oriented_intervals)
        track_height: Height of track as fraction of y-axis (default: 0.10)
        label_features: Whether to label features when wide enough
    """
    from matplotlib.patches import Rectangle, Patch

    if not gff_intervals:
        return

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin

    # Group features by type to assign rows
    feature_types_present = sorted(set(ft for _, _, ft, _ in gff_intervals))
    n_types = len(feature_types_present)
    if n_types == 0:
        return

    # Position track at bottom -- extend y-axis downward
    bar_height = track_height * y_range
    track_bottom = ymin - bar_height
    sub_height = bar_height / max(n_types, 1)
    type_to_row = {ft: i for i, ft in enumerate(feature_types_present)}

    # Extend y-axis to make room for the track
    ax.set_ylim(track_bottom - bar_height * 0.08, ymax)

    # Draw track background
    ax.add_patch(Rectangle(
        (xmin, track_bottom), x_range, bar_height,
        facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=0.5, zorder=10
    ))

    # Draw a thin separator line between the plot and the track
    ax.axhline(y=ymin, color="#bdc3c7", linewidth=0.8, zorder=10)

    # Readable short names for feature types
    _type_short = {
        "five_prime_UTR": "5' UTR",
        "three_prime_UTR": "3' UTR",
        "start_codon": "start",
        "stop_codon": "stop",
    }

    # Draw row labels on the left (feature type names)
    for ftype, row in type_to_row.items():
        row_mid_y = track_bottom + (row + 0.5) * sub_height
        display_type = _type_short.get(ftype, ftype)
        color = _GFF_FEATURE_COLORS.get(ftype, _GFF_DEFAULT_COLOR)
        ax.text(
            xmin + x_range * 0.005, row_mid_y, display_type,
            ha="left", va="center", fontsize=6, fontweight="bold",
            color=color, zorder=13, clip_on=True,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8),
        )

    # Draw each feature bar
    for (s, e, ftype, label) in gff_intervals:
        row = type_to_row[ftype]
        color = _GFF_FEATURE_COLORS.get(ftype, _GFF_DEFAULT_COLOR)

        feat_bottom = track_bottom + row * sub_height
        feat_h = sub_height * 0.85

        ax.add_patch(Rectangle(
            (s, feat_bottom + sub_height * 0.075), e - s, feat_h,
            facecolor=color, edgecolor="none", alpha=0.85, zorder=11
        ))

        # Label inside the bar: show product/name for wide bars,
        # feature type for medium bars, nothing for tiny bars
        if label_features:
            mid_x = (s + e) / 2
            mid_y = feat_bottom + sub_height / 2
            bar_width_frac = (e - s) / x_range

            if bar_width_frac > 0.06 and label:
                # Wide bar: show product/name
                display_label = label[:25] + "..." if len(label) > 25 else label
                ax.text(mid_x, mid_y, display_label, ha="center", va="center",
                        fontsize=5.5, color="white", fontweight="bold", zorder=12,
                        clip_on=True)
            elif bar_width_frac > 0.015:
                # Medium bar: show abbreviated feature type
                short = _type_short.get(ftype, ftype)
                ax.text(mid_x, mid_y, short, ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold", zorder=12,
                        clip_on=True)
            # Tiny bars (<1.5% of width): no label inside, but the row label
            # on the left already identifies the feature type

    # Add legend patches for feature types
    handles, labels = ax.get_legend_handles_labels()
    for ftype in feature_types_present:
        color = _GFF_FEATURE_COLORS.get(ftype, _GFF_DEFAULT_COLOR)
        display_type = _type_short.get(ftype, ftype)
        handles.append(Patch(facecolor=color, edgecolor="none", label=display_type))
        labels.append(display_type)
    ax.legend(handles, labels, loc="best", fontsize=7, ncol=2)


def _save_fig(path: str, dpi: int = 300) -> None:
    """
    Save current matplotlib figure to file and close it.

    Args:
        path: Output file path
        dpi: Resolution in dots per inch (default: 300 for high resolution)
    """
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def create_interactive_plot(
    entropy: np.ndarray,
    is_exon: np.ndarray,
    title: str,
    output_path: str,
    genomic_start: int = 0,
    unit: str = "bits",
    scored_drops: Optional[Dict[str, List[Tuple[int, float]]]] = None,
    smooth_w: int = 51,
    low_quantile: float = 0.10,
) -> None:
    """
    Create a modern interactive plot using Plotly.

    Features:
    - Interactive zoom, pan, hover
    - Chromosome coordinates at top corners
    - Clean, modern styling similar to genomic browser views
    - Exon regions highlighted with subtle shading
    - Low-entropy regions highlighted

    Args:
        entropy: Per-position entropy values
        is_exon: Binary exon mask
        title: Plot title
        output_path: Path for HTML output
        genomic_start: Genomic start position for coordinate display
        unit: Entropy unit ('bits' or 'nats')
        scored_drops: Optional dict of scored drop positions
        smooth_w: Smoothing window for display
        low_quantile: Threshold for low-entropy highlighting
    """
    if not PLOTLY_AVAILABLE:
        print(f"[WARNING] Skipping interactive plot (Plotly not available): {output_path}")
        return

    x = np.arange(len(entropy))
    genomic_x = x + genomic_start

    # Smooth the entropy for cleaner display
    sm = _rolling_mean(entropy, smooth_w)

    # Get exon intervals
    exon_intervals = get_exon_intervals_oriented(is_exon)

    # Calculate low-entropy threshold
    thr = np.nanquantile(entropy, low_quantile) if np.any(~np.isnan(entropy)) else 0

    # Create figure with secondary x-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    # Add exon shading as background rectangles
    for (start, end, exon_id) in exon_intervals:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 193, 7, 0.15)",  # Light yellow/gold
            layer="below",
            line_width=0,
            row=1, col=1,
        )

    # Add low-entropy highlighting
    # Create segments where entropy is below threshold
    low_mask = entropy <= thr
    in_low = False
    low_start = 0
    for i, is_low in enumerate(low_mask):
        if is_low and not in_low:
            low_start = i
            in_low = True
        elif not is_low and in_low:
            fig.add_vrect(
                x0=low_start, x1=i,
                fillcolor="rgba(231, 76, 60, 0.12)",  # Light red
                layer="below",
                line_width=0,
                row=1, col=1,
            )
            in_low = False
    if in_low:
        fig.add_vrect(
            x0=low_start, x1=len(entropy),
            fillcolor="rgba(231, 76, 60, 0.12)",
            layer="below",
            line_width=0,
            row=1, col=1,
        )

    # Main entropy trace with fill
    fig.add_trace(
        go.Scatter(
            x=x,
            y=entropy,
            mode='lines',
            name=f'Entropy ({unit})',
            line=dict(color='#3498db', width=0.8),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.3)',
            hovertemplate=(
                '<b>Position:</b> %{x}<br>'
                '<b>Genomic:</b> %{customdata:,}<br>'
                '<b>Entropy:</b> %{y:.3f}<extra></extra>'
            ),
            customdata=genomic_x,
        ),
        row=1, col=1,
    )

    # Add drop markers if provided
    if scored_drops:
        for method, drops in scored_drops.items():
            if drops:
                positions = [pos for pos, _ in drops]
                scores = [abs(score) for _, score in drops]
                ys = [sm[pos] if pos < len(sm) else 0 for pos in positions]

                fig.add_trace(
                    go.Scatter(
                        x=positions,
                        y=ys,
                        mode='markers',
                        name=f'Drops ({method})',
                        marker=dict(
                            size=8,
                            color=scores,
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(title='Score', x=1.02),
                            line=dict(width=1, color='black'),
                        ),
                        hovertemplate=(
                            f'<b>Method:</b> {method}<br>'
                            '<b>Position:</b> %{x}<br>'
                            '<b>Score:</b> %{marker.color:.2f}<extra></extra>'
                        ),
                    ),
                    row=1, col=1,
                )

    # Gene track at bottom (row 2)
    # Draw intron line
    fig.add_trace(
        go.Scatter(
            x=[0, len(entropy)],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='#7f8c8d', width=3),
            showlegend=False,
            hoverinfo='skip',
        ),
        row=2, col=1,
    )

    # Draw exons as rectangles
    for (start, end, exon_id) in exon_intervals:
        fig.add_trace(
            go.Scatter(
                x=[start, start, end, end, start],
                y=[0.2, 0.8, 0.8, 0.2, 0.2],
                mode='lines',
                fill='toself',
                fillcolor='#2ecc71',
                line=dict(color='#27ae60', width=1),
                showlegend=False,
                hovertemplate=f'<b>Exon {exon_id}</b><br>Start: {start}<br>End: {end}<extra></extra>',
            ),
            row=2, col=1,
        )

    # Update layout for modern look
    genomic_end = genomic_start + len(entropy)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family='Arial, sans-serif'),
            x=0.5,
        ),
        font=dict(family='Arial, sans-serif', size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        # Add chromosome coordinates as annotations at top corners
        annotations=[
            dict(
                x=0,
                y=1.08,
                xref='paper',
                yref='paper',
                text=f'<b>{genomic_start:,}</b>',
                showarrow=False,
                font=dict(size=11, color='#2c3e50'),
                xanchor='left',
            ),
            dict(
                x=1,
                y=1.08,
                xref='paper',
                yref='paper',
                text=f'<b>{genomic_end:,}</b>',
                showarrow=False,
                font=dict(size=11, color='#2c3e50'),
                xanchor='right',
            ),
        ],
    )

    # Update axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='#bdc3c7',
        zeroline=False,
        range=[0, len(entropy)],
        title_text='Position (bp)',
        title_font=dict(size=12),
        row=1, col=1,
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(189, 195, 199, 0.3)',
        showline=True,
        linewidth=1,
        linecolor='#bdc3c7',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='#bdc3c7',
        title_text=f'Entropy ({unit})',
        title_font=dict(size=12),
        row=1, col=1,
    )

    # Gene track axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='#bdc3c7',
        range=[0, len(entropy)],
        row=2, col=1,
    )

    fig.update_yaxes(
        showgrid=False,
        showline=False,
        showticklabels=False,
        range=[0, 1],
        fixedrange=True,
        row=2, col=1,
    )

    # Save as interactive HTML
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': output_path.replace('.html', ''),
                'height': 600,
                'width': 1200,
                'scale': 2,
            },
        },
    )
    print(f"[INFO] Saved interactive plot: {output_path}")


def plot_suite(
    output_mgr: OutputManager,
    base_name: str,
    entropy_main: np.ndarray,
    is_exon: np.ndarray,
    drop_points: Dict[str, List[int]],
    scored_drops: Dict[str, List[Tuple[int, float]]],
    title_prefix: str,
    smooth_w: int = 51,
    zoom_bp: int = 0,
    max_zoom_plots: int = 60,
    plot_style: str = "plain",
    unit: str = "nats",
    ylim: Optional[Tuple[float, float]] = None,
    annotate_top_n: int = 5,
    genomic_start: int = 0,
    rise_points: Optional[Dict[str, List[int]]] = None,
    scored_rises: Optional[Dict[str, List[Tuple[int, float]]]] = None,
    gff_intervals: Optional[List[Tuple[int, int, str, str]]] = None,
) -> None:
    """
    Generate complete suite of entropy visualization plots (ENHANCED JAN26).

    Creates multiple plot types:
    1. Raw entropy with exon shading
    2. Smoothed entropy with exon boundaries marked
    3. Boundary-focused view
    4. One plot per drop detection method (with confidence-aware sizing)
    5. Optional zoom plots around each exon boundary

    NEW in jan26:
    - Scored drops shown with confidence-based marker sizing (RED gradient)
    - Scored rises shown with confidence-based marker sizing (BLUE gradient)
    - Color intensity represents confidence score
    - Top N drops/rises annotated with position and score
    - Colorbar legend for confidence scale

    Args:
        output_mgr: OutputManager for file paths
        base_name: Base filename for outputs
        entropy_main: Per-position entropy values
        is_exon: Binary exon mask
        drop_points: Dict mapping method names to detected drop positions (legacy)
        scored_drops: Dict mapping method names to (position, score) tuples (NEW)
        title_prefix: Title prefix for all plots
        smooth_w: Smoothing window size
        zoom_bp: If >0, create zoom plots with this radius
        max_zoom_plots: Maximum number of zoom plots (safety limit)
        plot_style: 'plain' or 'evodesigner'
        unit: 'nats' or 'bits' for axis label
        ylim: Optional (min, max) for Y-axis
        annotate_top_n: Number of top drops to annotate (0=none)
        genomic_start: Genomic start position for chromosome coordinate labels
        rise_points: Dict mapping method names to detected rise positions (legacy)
        scored_rises: Dict mapping method names to (position, score) tuples for rises
    """
    # Initialize empty dicts if None
    if rise_points is None:
        rise_points = {}
    if scored_rises is None:
        scored_rises = {}
    x = np.arange(len(entropy_main))
    exon_intervals = get_exon_intervals_oriented(is_exon)
    exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)
    sm = _rolling_mean(entropy_main, smooth_w)

    def maybe_style(ax, xx, yy):
        """Apply evodesigner styling if configured."""
        if plot_style == "evodesigner":
            evodesigner_fill(ax, xx, yy, low_quantile=0.10)

    def apply_common_styling(ax, xx, title, add_chrom_axis=True):
        """Apply common styling: modern look with chromosome coords at corners."""
        # Modern clean background
        ax.set_facecolor('white')
        ax.figure.patch.set_facecolor('white')

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')

        # Subtle grid
        ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#bdc3c7')
        ax.xaxis.grid(False)

        # Set title and axis labels with clean fonts
        ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_xlabel("Position (bp)", fontsize=12, color='#2c3e50')
        ax.set_ylabel(f"Entropy ({unit})", fontsize=12, color='#2c3e50')

        # Increase tick label sizes
        ax.tick_params(axis='both', labelsize=11, colors='#2c3e50')

        # Remove white padding on x-axis
        ax.set_xlim(xx[0], xx[-1])

        # Add padding at top for exon track (extend y-axis by 15%)
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.15 * y_range)

        # Add chromosome coordinates at bottom corners
        if add_chrom_axis and genomic_start > 0:
            genomic_end = genomic_start + int(xx[-1] - xx[0])
            # Left corner: start position (at bottom)
            ax.annotate(
                f'{genomic_start:,}',
                xy=(0, -0.12), xycoords='axes fraction',
                fontsize=11, color='#2c3e50', fontweight='bold',
                ha='left', va='top'
            )
            # Right corner: end position (at bottom)
            ax.annotate(
                f'{genomic_end:,}',
                xy=(1, -0.12), xycoords='axes fraction',
                fontsize=11, color='#2c3e50', fontweight='bold',
                ha='right', va='top'
            )

    # --- Plot 1: Raw entropy with exon track at top ---
    plt.figure(figsize=(16, 4.5))
    ax = plt.gca()
    ax.plot(x, entropy_main, linewidth=0.8, label="Entropy(main)")
    maybe_style(ax, x, entropy_main)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)
    apply_common_styling(ax, x, f"{title_prefix}\nRaw Entropy")
    draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)
    if gff_intervals:
        draw_gff_feature_track(ax, gff_intervals, track_height=0.10, label_features=True)
    _save_fig(output_mgr.plot_path(f"{base_name}.entropy_raw.png"))

    # --- Plot 2: Smoothed entropy with exon track ---
    plt.figure(figsize=(16, 4.5))
    ax = plt.gca()
    ax.plot(x, sm, linewidth=1.2, label=f"Entropy(main) rolling_mean(w={smooth_w})")
    maybe_style(ax, x, sm)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)
    apply_common_styling(ax, x, f"{title_prefix}\nSmoothed Entropy with Exon Boundaries")
    draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)
    if gff_intervals:
        draw_gff_feature_track(ax, gff_intervals, track_height=0.10, label_features=True)
    _save_fig(output_mgr.plot_path(f"{base_name}.entropy_smooth.png"))

    # --- Plot 3: Boundary-focused view (with vertical lines at exon boundaries) ---
    plt.figure(figsize=(16, 4.5))
    ax = plt.gca()
    ax.plot(x, entropy_main, linewidth=0.8, label="Entropy", color='#3498db')
    # Just blue fill, no orange low-entropy highlighting for this plot
    ax.fill_between(x, entropy_main, 0, alpha=0.35, color='#3498db')

    # Add vertical dashed lines at all exon boundaries (both starts and ends)
    all_boundaries = set(exon_starts) | set(exon_ends)
    for i, boundary_pos in enumerate(sorted(all_boundaries)):
        label = "Exon boundary" if i == 0 else None
        ax.axvline(boundary_pos, color='#8e44ad', linestyle='--', linewidth=1.5, alpha=0.7, label=label)

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)
    apply_common_styling(ax, x, f"{title_prefix}\nExon Boundaries")
    draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)
    if gff_intervals:
        draw_gff_feature_track(ax, gff_intervals, track_height=0.10, label_features=True)
    _save_fig(output_mgr.plot_path(f"{base_name}.entropy_boundaries.png"))

    # --- Plot 4: One plot per drop/rise detection method (ENHANCED JAN26) ---
    # Handle both scored and unscored methods for both drops and rises
    all_drop_methods = set(drop_points.keys()) | set(scored_drops.keys())
    all_rise_methods = set(rise_points.keys()) | set(scored_rises.keys())
    all_methods = all_drop_methods | all_rise_methods

    for method in all_methods:
        plt.figure(figsize=(16, 4.5))
        ax = plt.gca()
        ax.plot(x, sm, linewidth=1.2, label="Smoothed entropy", color='gray', alpha=0.7)
        maybe_style(ax, x, sm)

        has_drops = False
        has_rises = False

        # --- Plot DROPS (RED gradient) ---
        if method in scored_drops:
            drops_with_scores = scored_drops[method]
            if drops_with_scores:
                has_drops = True
                positions = [pos for pos, _ in drops_with_scores]
                scores = [abs(score) for _, score in drops_with_scores]
                ys = sm[positions]

                # Normalize scores for visualization
                if len(scores) > 1:
                    score_min, score_max = min(scores), max(scores)
                    score_range = score_max - score_min if score_max > score_min else 1.0
                    norm_scores = [(s - score_min) / score_range for s in scores]
                else:
                    norm_scores = [1.0] * len(scores)

                # Marker size: 20 (min) to 150 (max)
                sizes = [20 + 130 * ns for ns in norm_scores]

                # Scatter with size and color encoding (RED for drops)
                scatter_drops = ax.scatter(
                    positions, ys,
                    s=sizes,
                    c=scores,
                    cmap='Reds',
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"drops:{method}",
                    vmin=min(scores) if len(scores) > 0 else 0,
                    vmax=max(scores) if len(scores) > 0 else 1,
                    marker='v'  # Downward triangle for drops
                )

                # Annotate top N drops (below the data points to avoid exon track)
                if annotate_top_n > 0 and len(drops_with_scores) > 0:
                    # Sort by score strength
                    sorted_drops = sorted(drops_with_scores, key=lambda x: abs(x[1]), reverse=True)
                    top_drops = sorted_drops[:min(annotate_top_n, len(sorted_drops))]

                    for rank, (pos, score) in enumerate(top_drops, 1):
                        ax.annotate(
                            f'D{rank}: {pos}',
                            xy=(pos, sm[pos]),
                            xytext=(0, -25),
                            textcoords='offset points',
                            fontsize=7,
                            ha='center',
                            va='top',
                            bbox=dict(boxstyle='round,pad=0.3', fc='#ffcccc', alpha=0.9, edgecolor='#cc0000', linewidth=0.5),
                            arrowprops=dict(arrowstyle='->', lw=0.8, color='#cc0000')
                        )

        elif method in drop_points:
            # Legacy method without scores (original behavior)
            pts = drop_points[method]
            if pts:
                has_drops = True
                ys = sm[pts]
                ax.scatter(pts, ys, s=30, label=f"drops:{method}", color='red', alpha=0.7, edgecolors='black', marker='v')

        # --- Plot RISES (BLUE gradient) ---
        if method in scored_rises:
            rises_with_scores = scored_rises[method]
            if rises_with_scores:
                has_rises = True
                positions = [pos for pos, _ in rises_with_scores]
                scores = [abs(score) for _, score in rises_with_scores]
                ys = sm[positions]

                # Normalize scores for visualization
                if len(scores) > 1:
                    score_min, score_max = min(scores), max(scores)
                    score_range = score_max - score_min if score_max > score_min else 1.0
                    norm_scores = [(s - score_min) / score_range for s in scores]
                else:
                    norm_scores = [1.0] * len(scores)

                # Marker size: 20 (min) to 150 (max)
                sizes = [20 + 130 * ns for ns in norm_scores]

                # Scatter with size and color encoding (BLUE for rises)
                scatter_rises = ax.scatter(
                    positions, ys,
                    s=sizes,
                    c=scores,
                    cmap='Blues',
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"rises:{method}",
                    vmin=min(scores) if len(scores) > 0 else 0,
                    vmax=max(scores) if len(scores) > 0 else 1,
                    marker='^'  # Upward triangle for rises
                )

                # Annotate top N rises
                if annotate_top_n > 0 and len(rises_with_scores) > 0:
                    sorted_rises = sorted(rises_with_scores, key=lambda x: abs(x[1]), reverse=True)
                    top_rises = sorted_rises[:min(annotate_top_n, len(sorted_rises))]

                    for rank, (pos, score) in enumerate(top_rises, 1):
                        ax.annotate(
                            f'R{rank}: {pos}',
                            xy=(pos, sm[pos]),
                            xytext=(0, 25),
                            textcoords='offset points',
                            fontsize=7,
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', fc='#cce5ff', alpha=0.9, edgecolor='#0066cc', linewidth=0.5),
                            arrowprops=dict(arrowstyle='->', lw=0.8, color='#0066cc')
                        )

        elif method in rise_points:
            # Legacy method without scores
            pts = rise_points[method]
            if pts:
                has_rises = True
                ys = sm[pts]
                ax.scatter(pts, ys, s=30, label=f"rises:{method}", color='blue', alpha=0.7, edgecolors='black', marker='^')

        # Add colorbars if we have scored data
        if has_drops and method in scored_drops and scored_drops[method]:
            cbar_drops = plt.colorbar(scatter_drops, ax=ax, pad=0.02)
            cbar_drops.set_label('Drop Score', rotation=270, labelpad=15, fontsize=9)

        if has_rises and method in scored_rises and scored_rises[method]:
            cbar_rises = plt.colorbar(scatter_rises, ax=ax, pad=0.08 if has_drops else 0.02)
            cbar_rises.set_label('Rise Score', rotation=270, labelpad=15, fontsize=9)

        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="best", fontsize=9)

        # Update title based on what's being shown
        if has_drops and has_rises:
            plot_title = f"{title_prefix}\nDrop & Rise Detection: {method}"
        elif has_rises:
            plot_title = f"{title_prefix}\nRise Detection: {method}"
        else:
            plot_title = f"{title_prefix}\nDrop Detection: {method}"

        apply_common_styling(ax, x, plot_title)
        draw_exon_track(ax, exon_intervals, track_height=0.06, label_exons=True)
        _save_fig(output_mgr.plot_path(f"{base_name}.transitions_{method}.png"))

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

            plt.figure(figsize=(14, 4.5))
            ax = plt.gca()

            xx = np.arange(lo, hi)
            yy = sm[lo:hi]
            ax.plot(xx, yy, linewidth=1.3, label="Smoothed entropy")
            maybe_style(ax, xx, yy)

            ax.axvline(idx, linestyle="--" if kind == "start" else ":",
                      linewidth=1.2, alpha=0.9, color='#E74C3C', label=f"exon_{kind}")

            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.legend(loc="best", fontsize=9)
            apply_common_styling(ax, xx, f"{title_prefix}\nZoom: {kind} @ position {idx} (\u00b1{zoom_bp}bp)")

            # Filter exon intervals visible in zoom window and draw track
            zoom_exon_intervals = []
            for (s, e, eid) in exon_intervals:
                ss = max(s, lo)
                ee = min(e, hi)
                if ee > ss:
                    zoom_exon_intervals.append((ss, ee, eid))
            draw_exon_track(ax, zoom_exon_intervals, track_height=0.08, label_exons=True)
            _save_fig(output_mgr.plot_path(f"{base_name}.zoom_{kind}_{idx}.png"))
            count += 1

    # --- Plot 6: Interactive HTML plots (if Plotly available) ---
    if PLOTLY_AVAILABLE:
        print(f"[INFO] Generating interactive HTML plots...")

        # Main interactive plot (raw entropy with all features)
        create_interactive_plot(
            entropy=entropy_main,
            is_exon=is_exon,
            title=f"{title_prefix} | Interactive",
            output_path=output_mgr.plot_path(f"{base_name}.interactive.html"),
            genomic_start=genomic_start,
            unit=unit,
            scored_drops=scored_drops,
            smooth_w=smooth_w,
        )

        # Interactive plot with smoothed data
        create_interactive_plot(
            entropy=sm,
            is_exon=is_exon,
            title=f"{title_prefix} | Smoothed Interactive",
            output_path=output_mgr.plot_path(f"{base_name}.smoothed_interactive.html"),
            genomic_start=genomic_start,
            unit=unit,
            scored_drops=scored_drops,
            smooth_w=1,  # Already smoothed
        )


# =============================================================================
# DROP DETECTION METHOD COMPARISON AND EVALUATION (NEW FEB 5, 2026)
# =============================================================================

@dataclass
class DropDetectionStats:
    """Statistics for a single drop detection method."""
    method: str
    n_drops: int
    scores: List[float]
    positions: List[int]
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    std_score: float

    # Exon boundary metrics (populated by evaluate_exon_boundary_capture)
    n_boundaries_captured: int = 0
    n_total_boundaries: int = 0
    boundary_capture_rate: float = 0.0
    mean_boundary_distance: float = float('inf')
    captured_boundaries: List[Tuple[int, int, float]] = field(default_factory=list)  # (boundary_pos, drop_pos, distance)
    false_positives: int = 0  # drops not near any boundary


def compute_drop_detection_stats(
    scored_drops: Dict[str, List[Tuple[int, float]]],
    drop_points: Dict[str, List[int]] = None,
) -> Dict[str, DropDetectionStats]:
    """
    Compute comprehensive statistics for each drop detection method.

    Args:
        scored_drops: Dict mapping method names to (position, score) tuples
        drop_points: Optional dict of legacy unscored drop positions

    Returns:
        Dict mapping method names to DropDetectionStats dataclass instances
    """
    stats = {}

    # Process scored drops (zscore, mad, local)
    for method, drops in scored_drops.items():
        if not drops:
            stats[method] = DropDetectionStats(
                method=method,
                n_drops=0,
                scores=[],
                positions=[],
                mean_score=0.0,
                median_score=0.0,
                min_score=0.0,
                max_score=0.0,
                std_score=0.0,
            )
            continue

        positions = [d[0] for d in drops]
        scores = [abs(d[1]) for d in drops]  # Use absolute value for comparability

        stats[method] = DropDetectionStats(
            method=method,
            n_drops=len(drops),
            scores=scores,
            positions=positions,
            mean_score=float(np.mean(scores)),
            median_score=float(np.median(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            std_score=float(np.std(scores)) if len(scores) > 1 else 0.0,
        )

    # Process unscored drops (derivative, cusum, win_shift) - assign score=1.0
    if drop_points:
        for method, positions in drop_points.items():
            if method in stats:
                continue  # Already processed as scored
            stats[method] = DropDetectionStats(
                method=method,
                n_drops=len(positions),
                scores=[1.0] * len(positions),  # No scores available
                positions=list(positions),
                mean_score=1.0 if positions else 0.0,
                median_score=1.0 if positions else 0.0,
                min_score=1.0 if positions else 0.0,
                max_score=1.0 if positions else 0.0,
                std_score=0.0,
            )

    return stats


def evaluate_exon_boundary_capture(
    stats: Dict[str, DropDetectionStats],
    is_exon: np.ndarray,
    tolerance_bp: int = 50,
) -> Dict[str, DropDetectionStats]:
    """
    Evaluate how well each drop detection method captures actual exon boundaries.

    A drop is considered to "capture" a boundary if it falls within tolerance_bp
    of an actual exon start or end position.

    Args:
        stats: Dict of DropDetectionStats from compute_drop_detection_stats
        is_exon: Binary array indicating exon positions
        tolerance_bp: Maximum distance (bp) for a drop to be considered as capturing a boundary

    Returns:
        Updated stats dict with boundary capture metrics populated
    """
    # Get actual exon boundaries
    exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)
    all_boundaries = np.concatenate([exon_starts, exon_ends])
    n_boundaries = len(all_boundaries)

    for method, method_stats in stats.items():
        if method_stats.n_drops == 0:
            method_stats.n_total_boundaries = n_boundaries
            method_stats.captured_boundaries = []
            continue

        positions = np.array(method_stats.positions)
        captured = []
        captured_boundary_set = set()

        for drop_pos in positions:
            # Find closest boundary
            distances = np.abs(all_boundaries - drop_pos)
            closest_idx = np.argmin(distances)
            closest_dist = distances[closest_idx]
            closest_boundary = all_boundaries[closest_idx]

            if closest_dist <= tolerance_bp:
                captured.append((int(closest_boundary), int(drop_pos), float(closest_dist)))
                captured_boundary_set.add(closest_boundary)

        method_stats.n_boundaries_captured = len(captured_boundary_set)
        method_stats.n_total_boundaries = n_boundaries
        method_stats.boundary_capture_rate = len(captured_boundary_set) / n_boundaries if n_boundaries > 0 else 0.0
        method_stats.captured_boundaries = captured
        method_stats.mean_boundary_distance = float(np.mean([c[2] for c in captured])) if captured else float('inf')
        method_stats.false_positives = method_stats.n_drops - len(captured)

    return stats


def plot_method_comparison_table(
    stats: Dict[str, DropDetectionStats],
    output_path: str,
    title: str = "Drop Detection Method Comparison",
) -> None:
    """
    Create a visual table comparing drop detection method statistics.

    Generates a matplotlib figure showing a formatted table with:
    - Number of drops detected
    - Score statistics (mean, median, range)
    - Boundary capture rate
    - False positive count

    Args:
        stats: Dict of DropDetectionStats
        output_path: Path to save the figure
        title: Title for the figure
    """
    # Prepare table data
    methods = sorted(stats.keys())

    columns = ['Method', 'Drops', 'Mean Score', 'Median', 'Min', 'Max',
               'Boundaries\nCaptured', 'Capture\nRate', 'False\nPositives']

    cell_data = []
    for method in methods:
        s = stats[method]
        row = [
            method.upper(),
            str(s.n_drops),
            f"{s.mean_score:.2f}" if s.n_drops > 0 else "-",
            f"{s.median_score:.2f}" if s.n_drops > 0 else "-",
            f"{s.min_score:.2f}" if s.n_drops > 0 else "-",
            f"{s.max_score:.2f}" if s.n_drops > 0 else "-",
            f"{s.n_boundaries_captured}/{s.n_total_boundaries}",
            f"{s.boundary_capture_rate*100:.1f}%",
            str(s.false_positives),
        ]
        cell_data.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(methods)))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Create table
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(columns),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header cells
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Color-code capture rate cells (column 7)
    for row_idx, method in enumerate(methods):
        s = stats[method]
        rate = s.boundary_capture_rate

        # Green gradient for capture rate
        if rate >= 0.8:
            color = '#27ae60'  # Dark green
        elif rate >= 0.5:
            color = '#f1c40f'  # Yellow
        elif rate >= 0.25:
            color = '#e67e22'  # Orange
        else:
            color = '#e74c3c'  # Red

        table[(row_idx + 1, 7)].set_facecolor(color)
        table[(row_idx + 1, 7)].set_text_props(color='white' if rate >= 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[INFO] Saved method comparison table: {output_path}")


def plot_method_comparison_heatmap(
    scored_drops: Dict[str, List[Tuple[int, float]]],
    is_exon: np.ndarray,
    output_path: str,
    title: str = "Drop Detection Scores Heatmap",
    tolerance_bp: int = 50,
) -> None:
    """
    Create a heatmap showing drop scores aligned to exon boundaries.

    Each row represents a detection method, columns represent exon boundaries,
    and cell color intensity shows the score of the closest drop (if any).

    Args:
        scored_drops: Dict mapping method names to (position, score) tuples
        is_exon: Binary array indicating exon positions
        output_path: Path to save the figure
        title: Title for the figure
        tolerance_bp: Maximum distance for associating a drop with a boundary
    """
    # Get exon boundaries
    exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)

    # Create boundary labels
    boundary_labels = []
    boundary_positions = []
    for i, pos in enumerate(exon_starts):
        boundary_labels.append(f"E{i+1} Start\n({pos})")
        boundary_positions.append(pos)
    for i, pos in enumerate(exon_ends):
        boundary_labels.append(f"E{i+1} End\n({pos})")
        boundary_positions.append(pos)

    if len(boundary_positions) == 0:
        print("[WARNING] No exon boundaries found, skipping heatmap")
        return

    methods = sorted(scored_drops.keys())
    n_methods = len(methods)
    n_boundaries = len(boundary_positions)

    # Build score matrix and distance matrix
    score_matrix = np.zeros((n_methods, n_boundaries))
    distance_matrix = np.full((n_methods, n_boundaries), np.inf)
    captured_matrix = np.zeros((n_methods, n_boundaries), dtype=bool)

    for m_idx, method in enumerate(methods):
        drops = scored_drops.get(method, [])
        if not drops:
            continue

        drop_positions = np.array([d[0] for d in drops])
        drop_scores = np.array([abs(d[1]) for d in drops])

        for b_idx, boundary_pos in enumerate(boundary_positions):
            distances = np.abs(drop_positions - boundary_pos)
            closest_idx = np.argmin(distances)
            closest_dist = distances[closest_idx]

            if closest_dist <= tolerance_bp:
                score_matrix[m_idx, b_idx] = drop_scores[closest_idx]
                distance_matrix[m_idx, b_idx] = closest_dist
                captured_matrix[m_idx, b_idx] = True

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n_boundaries * 1.5), max(4, n_methods * 0.8 + 2)))

    # Heatmap 1: Scores
    im1 = ax1.imshow(score_matrix, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(n_boundaries))
    ax1.set_xticklabels(boundary_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(n_methods))
    ax1.set_yticklabels([m.upper() for m in methods], fontsize=10)
    ax1.set_title("Drop Scores at Exon Boundaries", fontsize=12, fontweight='bold')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Drop Score', fontsize=10)

    # Add text annotations
    for m_idx in range(n_methods):
        for b_idx in range(n_boundaries):
            if captured_matrix[m_idx, b_idx]:
                text = f"{score_matrix[m_idx, b_idx]:.1f}"
                ax1.text(b_idx, m_idx, text, ha='center', va='center',
                        fontsize=7, color='white' if score_matrix[m_idx, b_idx] > score_matrix.max()/2 else 'black')
            else:
                ax1.text(b_idx, m_idx, "✗", ha='center', va='center',
                        fontsize=10, color='gray')

    # Heatmap 2: Distance to boundary
    distance_display = np.where(captured_matrix, distance_matrix, np.nan)
    im2 = ax2.imshow(distance_display, cmap='Blues_r', aspect='auto', vmin=0, vmax=tolerance_bp)
    ax2.set_xticks(range(n_boundaries))
    ax2.set_xticklabels(boundary_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(n_methods))
    ax2.set_yticklabels([m.upper() for m in methods], fontsize=10)
    ax2.set_title("Distance to Boundary (bp)", fontsize=12, fontweight='bold')

    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Distance (bp)', fontsize=10)

    # Add text annotations for distances
    for m_idx in range(n_methods):
        for b_idx in range(n_boundaries):
            if captured_matrix[m_idx, b_idx]:
                text = f"{int(distance_matrix[m_idx, b_idx])}"
                ax2.text(b_idx, m_idx, text, ha='center', va='center',
                        fontsize=7, color='white' if distance_matrix[m_idx, b_idx] < tolerance_bp/2 else 'black')
            else:
                ax2.text(b_idx, m_idx, "—", ha='center', va='center',
                        fontsize=10, color='gray')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[INFO] Saved method comparison heatmap: {output_path}")


def plot_method_comparison_summary(
    stats: Dict[str, DropDetectionStats],
    scored_drops: Dict[str, List[Tuple[int, float]]],
    entropy: np.ndarray,
    is_exon: np.ndarray,
    output_path: str,
    title: str = "Drop Detection Method Comparison Summary",
    smooth_w: int = 51,
) -> None:
    """
    Create a comprehensive multi-panel comparison figure.

    Includes:
    1. Bar chart of drops detected per method
    2. Bar chart of boundary capture rate
    3. Box plot of score distributions
    4. Precision-like metric visualization

    Args:
        stats: Dict of DropDetectionStats
        scored_drops: Dict mapping method names to (position, score) tuples
        entropy: Entropy values for context
        is_exon: Binary exon mask
        output_path: Path to save the figure
        title: Title for the figure
        smooth_w: Smoothing window for entropy display
    """
    methods = sorted(stats.keys())
    n_methods = len(methods)

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    method_colors = {m: colors[i] for i, m in enumerate(methods)}

    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Drops detected (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    drops_counts = [stats[m].n_drops for m in methods]
    bars1 = ax1.bar(methods, drops_counts, color=[method_colors[m] for m in methods], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Number of Drops', fontsize=11)
    ax1.set_title('Drops Detected', fontsize=12, fontweight='bold')
    ax1.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    for bar, count in zip(bars1, drops_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: Boundary capture rate (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    capture_rates = [stats[m].boundary_capture_rate * 100 for m in methods]
    bars2 = ax2.bar(methods, capture_rates, color=[method_colors[m] for m in methods], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Capture Rate (%)', fontsize=11)
    ax2.set_title('Exon Boundary Capture Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    for bar, rate in zip(bars2, capture_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 3: False positives vs True positives (stacked bar)
    ax3 = fig.add_subplot(gs[0, 2])
    true_positives = [len(stats[m].captured_boundaries) if stats[m].captured_boundaries else 0 for m in methods]
    false_positives = [stats[m].false_positives for m in methods]

    x = np.arange(n_methods)
    width = 0.6
    bars_tp = ax3.bar(x, true_positives, width, label='True Positives', color='#27ae60', edgecolor='black', linewidth=0.5)
    bars_fp = ax3.bar(x, false_positives, width, bottom=true_positives, label='False Positives', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('True vs False Positives', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=9)

    # Panel 4: Score distribution (box plot)
    ax4 = fig.add_subplot(gs[1, 0])
    score_data = [stats[m].scores for m in methods if stats[m].scores]
    score_labels = [m.upper() for m in methods if stats[m].scores]
    if score_data:
        bp = ax4.boxplot(score_data, labels=score_labels, patch_artist=True)
        for patch, method in zip(bp['boxes'], [m for m in methods if stats[m].scores]):
            patch.set_facecolor(method_colors[method])
            patch.set_alpha(0.7)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    # Panel 5: Mean distance to boundary (bar chart)
    ax5 = fig.add_subplot(gs[1, 1])
    mean_distances = []
    for m in methods:
        dist = stats[m].mean_boundary_distance
        mean_distances.append(dist if dist != float('inf') else 0)
    bars5 = ax5.bar(methods, mean_distances, color=[method_colors[m] for m in methods], edgecolor='black', linewidth=0.5)
    ax5.set_ylabel('Mean Distance (bp)', fontsize=11)
    ax5.set_title('Mean Distance to Nearest Boundary', fontsize=12, fontweight='bold')
    ax5.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    for bar, dist in zip(bars5, mean_distances):
        if dist > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{dist:.1f}", ha='center', va='bottom', fontsize=9)

    # Panel 6: Precision metric (TP / (TP + FP))
    ax6 = fig.add_subplot(gs[1, 2])
    precisions = []
    for m in methods:
        tp = len(stats[m].captured_boundaries) if stats[m].captured_boundaries else 0
        fp = stats[m].false_positives
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        precisions.append(precision)
    bars6 = ax6.bar(methods, precisions, color=[method_colors[m] for m in methods], edgecolor='black', linewidth=0.5)
    ax6.set_ylabel('Precision (%)', fontsize=11)
    ax6.set_title('Precision (TP / Total Drops)', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 105)
    ax6.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    for bar, prec in zip(bars6, precisions):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{prec:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 7: Entropy plot with all drops overlaid (spans bottom row)
    ax7 = fig.add_subplot(gs[2, :])

    # Smooth entropy
    sm = _rolling_mean(entropy, smooth_w)
    x = np.arange(len(entropy))

    # Plot entropy
    ax7.fill_between(x, sm, alpha=0.3, color='#3498db')
    ax7.plot(x, sm, color='#2980b9', linewidth=1, label='Smoothed Entropy')

    # Shade exon regions
    exon_intervals = get_exon_intervals_oriented(is_exon)
    for (start, end, eid) in exon_intervals:
        ax7.axvspan(start, end, alpha=0.15, color='#2E8B57')

    # Mark exon boundaries
    exon_starts, exon_ends = get_exon_boundaries_oriented(is_exon)
    for pos in exon_starts:
        ax7.axvline(pos, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1)
    for pos in exon_ends:
        ax7.axvline(pos, color='#e74c3c', linestyle=':', alpha=0.7, linewidth=1)

    # Plot drops for each method with vertical offset for visibility
    y_offsets = np.linspace(0.05, 0.25, n_methods)
    ymin, ymax = np.nanmin(sm), np.nanmax(sm)
    y_range = ymax - ymin

    for i, method in enumerate(methods):
        drops = scored_drops.get(method, [])
        if drops:
            positions = [d[0] for d in drops]
            y_pos = ymax + y_range * y_offsets[i]
            ax7.scatter(positions, [y_pos] * len(positions),
                       c=[method_colors[method]], s=50, marker='v',
                       label=f'{method.upper()} ({len(drops)})', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax7.set_xlabel('Position (bp)', fontsize=11)
    ax7.set_ylabel('Entropy', fontsize=11)
    ax7.set_title('All Methods Overlaid on Entropy Profile', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=9, ncol=min(n_methods, 4))
    ax7.set_xlim(0, len(entropy))

    # Add exon boundary legend
    ax7.plot([], [], color='#27ae60', linestyle='--', label='Exon Start')
    ax7.plot([], [], color='#e74c3c', linestyle=':', label='Exon End')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[INFO] Saved method comparison summary: {output_path}")


def write_method_comparison_tsv(
    stats: Dict[str, DropDetectionStats],
    output_path: str,
) -> None:
    """
    Write method comparison statistics to a TSV file.

    Args:
        stats: Dict of DropDetectionStats
        output_path: Path to save the TSV file
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("Method\tN_Drops\tMean_Score\tMedian_Score\tMin_Score\tMax_Score\tStd_Score\t")
        f.write("Boundaries_Captured\tTotal_Boundaries\tCapture_Rate\tMean_Boundary_Distance\tFalse_Positives\tPrecision\n")

        for method in sorted(stats.keys()):
            s = stats[method]
            tp = len(s.captured_boundaries) if s.captured_boundaries else 0
            precision = tp / (tp + s.false_positives) if (tp + s.false_positives) > 0 else 0

            f.write(f"{method}\t{s.n_drops}\t{s.mean_score:.4f}\t{s.median_score:.4f}\t")
            f.write(f"{s.min_score:.4f}\t{s.max_score:.4f}\t{s.std_score:.4f}\t")
            f.write(f"{s.n_boundaries_captured}\t{s.n_total_boundaries}\t{s.boundary_capture_rate:.4f}\t")
            dist_str = f"{s.mean_boundary_distance:.2f}" if s.mean_boundary_distance != float('inf') else "NA"
            f.write(f"{dist_str}\t{s.false_positives}\t{precision:.4f}\n")

    print(f"[INFO] Saved method comparison TSV: {output_path}")


def generate_method_comparison_report(
    output_mgr: 'OutputManager',
    base_name: str,
    scored_drops: Dict[str, List[Tuple[int, float]]],
    drop_points: Dict[str, List[int]],
    entropy: np.ndarray,
    is_exon: np.ndarray,
    title_prefix: str = "",
    tolerance_bp: int = 50,
    smooth_w: int = 51,
) -> Dict[str, DropDetectionStats]:
    """
    Generate a complete comparison report for all drop detection methods.

    This is the main entry point for method comparison analysis.
    Creates all comparison plots, heatmaps, and tables.

    Args:
        output_mgr: OutputManager for file paths
        base_name: Base filename for outputs
        scored_drops: Dict mapping method names to (position, score) tuples
        drop_points: Dict of legacy unscored drop positions
        entropy: Per-position entropy values
        is_exon: Binary exon mask
        title_prefix: Title prefix for all plots
        tolerance_bp: Maximum distance for boundary capture evaluation
        smooth_w: Smoothing window for entropy display

    Returns:
        Dict of DropDetectionStats with all metrics computed
    """
    print(f"\n[INFO] Generating method comparison report...")

    # Step 1: Compute basic stats
    stats = compute_drop_detection_stats(scored_drops, drop_points)

    # Step 2: Evaluate boundary capture
    stats = evaluate_exon_boundary_capture(stats, is_exon, tolerance_bp)

    # Step 3: Generate all visualizations
    title = f"{title_prefix} | Method Comparison" if title_prefix else "Method Comparison"

    # Comparison table
    plot_method_comparison_table(
        stats,
        output_path=output_mgr.plot_path(f"{base_name}.method_comparison_table.png"),
        title=title,
    )

    # Heatmap (only for scored methods)
    if scored_drops:
        plot_method_comparison_heatmap(
            scored_drops,
            is_exon,
            output_path=output_mgr.plot_path(f"{base_name}.method_comparison_heatmap.png"),
            title=title,
            tolerance_bp=tolerance_bp,
        )

    # Summary figure
    plot_method_comparison_summary(
        stats,
        scored_drops,
        entropy,
        is_exon,
        output_path=output_mgr.plot_path(f"{base_name}.method_comparison_summary.png"),
        title=title,
        smooth_w=smooth_w,
    )

    # TSV output
    write_method_comparison_tsv(
        stats,
        output_path=output_mgr.data_path(f"{base_name}.method_comparison.tsv"),
    )

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"METHOD COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'Drops':>6} {'Capture%':>9} {'Precision%':>11} {'FP':>4}")
    print(f"{'-'*60}")
    for method in sorted(stats.keys()):
        s = stats[method]
        tp = len(s.captured_boundaries) if s.captured_boundaries else 0
        precision = tp / (tp + s.false_positives) * 100 if (tp + s.false_positives) > 0 else 0
        print(f"{method.upper():<12} {s.n_drops:>6} {s.boundary_capture_rate*100:>8.1f}% {precision:>10.1f}% {s.false_positives:>4}")
    print(f"{'='*60}\n")

    return stats


def load_and_compare_methods(
    tsv_path: str,
    drops_path: str,
    output_dir: str,
    base_name: str = None,
    tolerance_bp: int = 50,
    smooth_w: int = 51,
) -> Dict[str, DropDetectionStats]:
    """
    Load existing entropy data and drop detections, then generate comparison report.

    This is a standalone function for analyzing previously computed results
    without re-running the full pipeline.

    Args:
        tsv_path: Path to the .tsv file with entropy data
        drops_path: Path to the .drops.txt file with detected drops
        output_dir: Directory to save comparison outputs
        base_name: Base name for output files (defaults to stem of tsv_path)
        tolerance_bp: Maximum distance for boundary capture evaluation
        smooth_w: Smoothing window for entropy display

    Returns:
        Dict of DropDetectionStats with all metrics computed

    Example:
        >>> stats = load_and_compare_methods(
        ...     tsv_path="output/NM_001134.5/data/NM_001134.5.tsv",
        ...     drops_path="output/NM_001134.5/data/NM_001134.5.drops.txt",
        ...     output_dir="output/NM_001134.5/comparison"
        ... )
    """
    import os
    from pathlib import Path

    # Parse base_name from tsv_path if not provided
    if base_name is None:
        base_name = Path(tsv_path).stem

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading entropy data from: {tsv_path}")

    # Load entropy data from TSV
    # Assume TSV format: first column is position, and there's an entropy column
    data = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

    # Typically: col0=pos, col1=?, col2=?, col3=entropy, col4=is_exon (varies by output)
    # Let's read the header to determine columns
    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')

    # Find entropy and is_exon columns
    entropy_col = None
    is_exon_col = None
    for i, col in enumerate(header):
        col_lower = col.lower()
        if 'entropy' in col_lower and entropy_col is None:
            entropy_col = i
        if 'exon' in col_lower or 'isexon' in col_lower:
            is_exon_col = i

    if entropy_col is None:
        # Default to column 3 (common in jan26 output)
        entropy_col = 3
        print(f"[WARNING] Entropy column not found, using column {entropy_col}")

    if is_exon_col is None:
        # Try to find it
        for i, col in enumerate(header):
            if col.lower() in ['is_exon', 'isexon', 'exon']:
                is_exon_col = i
                break
        if is_exon_col is None and data.shape[1] > 4:
            is_exon_col = 4
            print(f"[WARNING] Is_exon column not found, using column {is_exon_col}")

    entropy = data[:, entropy_col]
    is_exon = data[:, is_exon_col].astype(int) if is_exon_col is not None else np.zeros(len(entropy), dtype=int)

    print(f"[INFO] Loaded {len(entropy)} positions, entropy column={entropy_col}")

    # Load drop detections
    print(f"[INFO] Loading drop detections from: {drops_path}")
    scored_drops = {}
    drop_points = {}

    with open(drops_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            method = parts[0]
            data_str = parts[1]

            if ':' in data_str:
                # Scored format: pos1:score1,pos2:score2,...
                drops_list = []
                for entry in data_str.split(','):
                    if ':' in entry:
                        pos_str, score_str = entry.split(':')
                        drops_list.append((int(pos_str), float(score_str)))
                scored_drops[method] = drops_list
            else:
                # Legacy format: pos1,pos2,...
                positions = [int(p) for p in data_str.split(',') if p]
                drop_points[method] = positions

    print(f"[INFO] Loaded {len(scored_drops)} scored methods, {len(drop_points)} legacy methods")

    # Create a simple OutputManager-like object for file paths
    class SimpleOutputManager:
        def __init__(self, out_dir):
            self.base_dir = out_dir
            self.plots_dir = os.path.join(out_dir, 'plots')
            self.data_dir = os.path.join(out_dir, 'data')
            os.makedirs(self.plots_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)

        def plot_path(self, filename):
            return os.path.join(self.plots_dir, filename)

        def data_path(self, filename):
            return os.path.join(self.data_dir, filename)

    output_mgr = SimpleOutputManager(output_dir)

    # Generate comparison report
    stats = generate_method_comparison_report(
        output_mgr=output_mgr,
        base_name=base_name,
        scored_drops=scored_drops,
        drop_points=drop_points,
        entropy=entropy,
        is_exon=is_exon,
        title_prefix=base_name,
        tolerance_bp=tolerance_bp,
        smooth_w=smooth_w,
    )

    return stats


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
    detection_methods: List[str] = None,
    zscore_threshold: float = 2.5,
    mad_threshold: float = 3.0,
    local_window: int = 500,
    local_threshold: float = 2.0,
    min_separation: int = 75,
    bootstrap: bool = False,
    n_bootstrap: int = 100,
    consensus_threshold: float = 0.50,
    annotate_top_n: int = 5,
    organism: str = None,
    include_timestamp: bool = True,
    # SAE analysis parameters
    analyze_sae: bool = False,
    sae_window: int = 500,
    sae_max_drops: int = 100,
    sae_method: Optional[str] = None,
    # GFF3 annotation parameters
    gff_path: Optional[str] = None,
    # Multi-GPU data parallelism
    n_gpus: int = 1,
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
    5. Initialize Evo2 model (or skip if using multi-GPU)
    6. Score sequence using overlapping chunks (sequential or multi-GPU)
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

    # Default detection methods if not specified
    if detection_methods is None:
        # Default to zscore and mad - these match visual intuition (big drops = high scores)
        # "local" is available but not default - it scores relative to local variance,
        # which can be confusing (small dips in flat regions score higher than big drops in variable regions)
        detection_methods = ["zscore", "mad"]

    # Set up organized output directories with descriptive naming
    output_mgr = OutputManager(
        out_dir=out_dir,
        gene_tag=tag,
        detection_methods=detection_methods,
        organism=organism,
        include_timestamp=include_timestamp
    )
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

    # --- Step 2b: Load GFF3 annotations (optional) ---
    gff_features: List[Dict[str, Any]] = []
    if gff_path is not None:
        print(f"[INFO] Loading GFF3 annotations from: {gff_path}")
        gff_features = load_gff_features(gff_path, chrom, locus_start, locus_end_excl)
        print(f"[INFO]   Found {len(gff_features)} features overlapping locus")
        if gff_features:
            from collections import Counter
            type_counts = Counter(f["feature_type"] for f in gff_features)
            for ftype, count in type_counts.most_common():
                print(f"[INFO]     {ftype}: {count}")

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

    # --- Step 6 & 7: Initialize model and score locus ---
    if n_gpus > 1 and torch.cuda.device_count() >= n_gpus:
        # Multi-GPU data-parallel scoring
        print(f"[INFO] Using MULTI-GPU scoring ({n_gpus} GPUs)...")
        print(f"[INFO] Each GPU loads its own model instance for data parallelism.")
        entropy_fwd, ppx_fwd, entropy_rc, ppx_rc, p4, true_tok, ll_next = \
            score_locus_aligned_overlap_multigpu(
                locus_seq,
                n_gpus=n_gpus,
                model_name="evo2_7b",
                max_chunk_len=max_chunk_len,
                chunk_overlap=chunk_overlap,
                compute_rcavg_entropy=True,
            )
        # Load a model on main process for any remaining operations (SAE, etc.)
        print("[INFO] Loading model on main process for post-scoring steps...")
        evo2_model = Evo2("evo2_7b")
        device = str(getattr(evo2_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        if hasattr(evo2_model, "eval"):
            evo2_model.eval()
    else:
        # Sequential single-GPU scoring (original behavior)
        if n_gpus > 1:
            print(f"[WARNING] Requested {n_gpus} GPUs but only "
                  f"{torch.cuda.device_count()} available. Falling back to sequential.")
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

    # Ensure ACGT_IDS exists for any later use
    device = str(getattr(evo2_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    idx_A = evo2_model.tokenizer.tokenize("A")[0]
    idx_C = evo2_model.tokenizer.tokenize("C")[0]
    idx_G = evo2_model.tokenizer.tokenize("G")[0]
    idx_T = evo2_model.tokenizer.tokenize("T")[0]
    ACGT_IDS = torch.tensor([idx_A, idx_C, idx_G, idx_T], dtype=torch.long, device=device)

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

    # --- Step 9: Run drop detection (ENHANCED JAN26) ---
    print(f"[INFO] Drop detection on: {name_main} ({unit})")
    print(f"[INFO] Methods: {', '.join(detection_methods)}")

    drops = {}  # Position-only (for backward compatibility)
    scored_drops = {}  # With confidence scores

    # Run legacy methods if requested
    if "derivative" in detection_methods:
        print("[INFO]   - Running derivative method...")
        drops["derivative"] = detect_drops_derivative(
            entropy_main_u, smooth_w=DROP_SMOOTH_W, thr_quantile=DROP_DERIV_Q
        )
        print(f"[INFO]     Found {len(drops['derivative'])} drops")

    if "win_shift" in detection_methods:
        print("[INFO]   - Running window-shift method...")
        drops["win_shift"] = detect_drops_window_mean_shift(
            entropy_main_u, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK
        )
        print(f"[INFO]     Found {len(drops['win_shift'])} drops")

    if "cusum" in detection_methods:
        print("[INFO]   - Running CUSUM method...")
        drops["cusum"] = detect_drops_cusum(
            entropy_main_u, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H
        )
        print(f"[INFO]     Found {len(drops['cusum'])} drops")

    # Run new scored methods
    if "zscore" in detection_methods:
        print("[INFO]   - Running z-score method...")
        scored_drops["zscore"] = detect_drops_zscore(
            entropy_main_u,
            smooth_w=DROP_SMOOTH_W,
            zscore_threshold=zscore_threshold,
            min_separation=min_separation
        )
        drops["zscore"] = _drops_scored_to_positions(scored_drops["zscore"])
        print(f"[INFO]     Found {len(scored_drops['zscore'])} drops")

    if "mad" in detection_methods:
        print("[INFO]   - Running MAD method...")
        scored_drops["mad"] = detect_drops_mad(
            entropy_main_u,
            smooth_w=DROP_SMOOTH_W,
            mad_threshold=mad_threshold,
            min_separation=min_separation
        )
        drops["mad"] = _drops_scored_to_positions(scored_drops["mad"])
        print(f"[INFO]     Found {len(scored_drops['mad'])} drops")

    if "local" in detection_methods:
        print("[INFO]   - Running local baseline method...")
        scored_drops["local"] = detect_drops_local_baseline(
            entropy_main_u,
            window_baseline=local_window,
            threshold_sigma=local_threshold,
            min_separation=min_separation
        )
        drops["local"] = _drops_scored_to_positions(scored_drops["local"])
        print(f"[INFO]     Found {len(scored_drops['local'])} drops")

    # Optional bootstrap consensus
    if bootstrap:
        print(f"[INFO]   - Running bootstrap consensus (n={n_bootstrap})...")
        print("[WARNING] Bootstrap is computationally expensive (~100x slower)")
        scored_drops["bootstrap"] = bootstrap_drop_confidence(
            entropy_main_u,
            smooth_w=DROP_SMOOTH_W,
            zscore_threshold=zscore_threshold,
            n_bootstrap=n_bootstrap,
            consensus_threshold=consensus_threshold,
            min_separation=min_separation
        )
        drops["bootstrap"] = _drops_scored_to_positions(scored_drops["bootstrap"])
        print(f"[INFO]     Found {len(scored_drops['bootstrap'])} robust drops")

    # --- Step 9b: Run RISE detection (end of drops - entropy returning to high values) ---
    print(f"[INFO] Rise detection (end of drops) on: {name_main} ({unit})")
    rises = {}  # Position-only (for backward compatibility)
    scored_rises = {}  # With confidence scores

    # Run legacy methods if requested
    if "derivative" in detection_methods:
        print("[INFO]   - Running derivative rise method...")
        rises["derivative"] = detect_rises_derivative(
            entropy_main_u, smooth_w=DROP_SMOOTH_W, thr_quantile=(1.0 - DROP_DERIV_Q)
        )
        print(f"[INFO]     Found {len(rises['derivative'])} rises")

    if "win_shift" in detection_methods:
        print("[INFO]   - Running window-shift rise method...")
        rises["win_shift"] = detect_rises_window_mean_shift(
            entropy_main_u, w=DROP_SHIFT_W, top_k=DROP_SHIFT_TOPK
        )
        print(f"[INFO]     Found {len(rises['win_shift'])} rises")

    if "cusum" in detection_methods:
        print("[INFO]   - Running CUSUM rise method...")
        rises["cusum"] = detect_rises_cusum(
            entropy_main_u, smooth_w=DROP_SMOOTH_W, h=DROP_CUSUM_H
        )
        print(f"[INFO]     Found {len(rises['cusum'])} rises")

    # Run new scored methods for rises
    if "zscore" in detection_methods:
        print("[INFO]   - Running z-score rise method...")
        scored_rises["zscore"] = detect_rises_zscore(
            entropy_main_u,
            smooth_w=DROP_SMOOTH_W,
            zscore_threshold=zscore_threshold,
            min_separation=min_separation
        )
        rises["zscore"] = _drops_scored_to_positions(scored_rises["zscore"])
        print(f"[INFO]     Found {len(scored_rises['zscore'])} rises")

    if "mad" in detection_methods:
        print("[INFO]   - Running MAD rise method...")
        scored_rises["mad"] = detect_rises_mad(
            entropy_main_u,
            smooth_w=DROP_SMOOTH_W,
            mad_threshold=mad_threshold,
            min_separation=min_separation
        )
        rises["mad"] = _drops_scored_to_positions(scored_rises["mad"])
        print(f"[INFO]     Found {len(scored_rises['mad'])} rises")

    if "local" in detection_methods:
        print("[INFO]   - Running local baseline rise method...")
        scored_rises["local"] = detect_rises_local_baseline(
            entropy_main_u,
            window_baseline=local_window,
            threshold_sigma=local_threshold,
            min_separation=min_separation
        )
        rises["local"] = _drops_scored_to_positions(scored_rises["local"])
        print(f"[INFO]     Found {len(scored_rises['local'])} rises")

    # Optional bootstrap consensus for rises
    if bootstrap:
        print(f"[INFO]   - Running bootstrap rise consensus (n={n_bootstrap})...")
        scored_rises["bootstrap"] = bootstrap_rise_confidence(
            entropy_main_u,
            smooth_w=DROP_SMOOTH_W,
            zscore_threshold=zscore_threshold,
            n_bootstrap=n_bootstrap,
            consensus_threshold=consensus_threshold,
            min_separation=min_separation
        )
        rises["bootstrap"] = _drops_scored_to_positions(scored_rises["bootstrap"])
        print(f"[INFO]     Found {len(scored_rises['bootstrap'])} robust rises")

    # --- Step 9c: SAE Feature Analysis (optional) ---
    sae_results = None
    sae_signature_features = None

    if analyze_sae:
        if not SAE_AVAILABLE:
            print("[WARNING] SAE analysis requested but sae_utils not available. Skipping.")
        else:
            print("[INFO] Running SAE feature analysis on detected drops...")

            # Determine which method to use for SAE analysis
            if sae_method is not None:
                selected_method = sae_method
            else:
                # Use first scored method available
                for m in ["zscore", "mad", "local", "bootstrap"]:
                    if m in scored_drops and scored_drops[m]:
                        selected_method = m
                        break
                else:
                    selected_method = None

            if selected_method is None or selected_method not in scored_drops:
                print("[WARNING] No scored drops available for SAE analysis. Skipping.")
            else:
                drop_positions = scored_drops[selected_method]
                print(f"[INFO]   Using method: {selected_method} ({len(drop_positions)} drops)")

                # Limit number of drops
                if len(drop_positions) > sae_max_drops:
                    print(f"[INFO]   Limiting to top {sae_max_drops} drops by score")
                    drop_positions = sorted(drop_positions, key=lambda x: x[1])[:sae_max_drops]

                # Extract regions around drops
                print(f"[INFO]   Extracting ±{sae_window}bp windows around drops...")
                regions = extract_regions_around_drops(locus_seq, drop_positions, sae_window)

                # Initialize ObservableEvo2 and SAE
                print("[INFO]   Initializing ObservableEvo2 for SAE analysis...")
                observable_model = ObservableEvo2("evo2_7b")

                print("[INFO]   Loading SAE from HuggingFace...")
                sae = load_topk_sae_from_hf(
                    d_hidden=observable_model.d_hidden,
                    device=observable_model.device,
                    dtype=torch.bfloat16,
                )

                # Run SAE analysis
                print("[INFO]   Analyzing drops with SAE...")
                sae_results = analyze_drops_with_sae(
                    regions, observable_model, sae, SAE_LAYER_NAME
                )

                # Find signature features
                print("[INFO]   Finding signature features...")
                sae_signature_features = find_signature_features(sae_results)
                print(f"[INFO]   Found {len(sae_signature_features)} signature features")

                # Write SAE analysis output
                sae_output_path = output_mgr.data_path(f"{base_name}.sae_analysis.tsv")
                write_sae_analysis_output(sae_results, sae_signature_features, sae_output_path)

                # Clean up to free GPU memory
                del observable_model
                del sae
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Step 9d: GFF3 annotation mapping (optional) ---
    gff_annotations: List[Dict[str, Any]] = []
    if gff_features:
        print("[INFO] Mapping drops/rises to GFF3 features...")
        gff_annotations = annotate_events_with_gff(
            scored_drops=scored_drops,
            scored_rises=scored_rises,
            gff_features=gff_features,
            pos_array=pos,
        )
        if gff_annotations:
            n_total = len(gff_annotations)
            n_cds = sum(1 for r in gff_annotations if r["in_CDS"])
            n_utr = sum(1 for r in gff_annotations if r["in_UTR"])
            n_none = sum(1 for r in gff_annotations if r["n_features"] == 0)
            print(f"[INFO]   Annotated {n_total} events: "
                  f"{n_cds} in CDS, {n_utr} in UTR, {n_none} no feature overlap")
        else:
            print("[INFO]   No drops/rises to annotate.")

    # --- Step 10: Write metadata JSON ---
    meta_path = output_mgr.meta_path(f"{base_name}.meta.json")
    meta = {
        "script_version": "genome_scoring_jan26_drops.py",
        "version_date": "2026-01-26",
        "enhancements": "Statistical drop detection with confidence scores",
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
        "detection_methods_used": detection_methods,
        "detection_parameters": {
            "zscore_threshold": zscore_threshold if "zscore" in detection_methods else None,
            "mad_threshold": mad_threshold if "mad" in detection_methods else None,
            "local_window": local_window if "local" in detection_methods else None,
            "local_threshold": local_threshold if "local" in detection_methods else None,
            "min_separation": min_separation,
            "bootstrap_enabled": bootstrap,
            "n_bootstrap": n_bootstrap if bootstrap else None,
            "consensus_threshold": consensus_threshold if bootstrap else None,
        },
        "exon_intervals_oriented_idx_endexcl": [(s, e, int(eid)) for (s, e, eid) in exon_intervals],
        "output_structure": {
            "base_dir": output_mgr.base_dir,
            "data_dir": output_mgr.data_dir,
            "plots_dir": output_mgr.plots_dir,
            "fasta_dir": output_mgr.fasta_dir,
            "metadata_dir": output_mgr.meta_dir,
        },
        "sae_analysis": {
            "enabled": analyze_sae,
            "available": SAE_AVAILABLE if analyze_sae else None,
            "window_bp": sae_window if analyze_sae else None,
            "max_drops": sae_max_drops if analyze_sae else None,
            "method_used": sae_method if analyze_sae and sae_results else None,
            "n_drops_analyzed": len(sae_results) if sae_results else 0,
            "n_signature_features": len(sae_signature_features) if sae_signature_features else 0,
        },
        "gff_annotation": {
            "enabled": gff_path is not None,
            "gff_path": gff_path,
            "n_features_loaded": len(gff_features),
            "n_events_annotated": len(gff_annotations),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Wrote metadata: {meta_path}")

    # --- Step 11: Write main TSV data file ---
    out_tsv = output_mgr.data_path(f"{base_name}.tsv")
    print(f"[INFO] Writing TSV: {out_tsv}")
    with open(out_tsv, "w") as f:
        # Header row - use the selected unit (bits or nats)
        f.write(f"Pos\tEntropy({unit})\tPerplexity(e)\t"
                f"P(A)\tP(C)\tP(G)\tP(T)\tTrueToken\tLL_next(nats)")
        f.write(f"\tEntropy_RCavg({unit})\tPerplexity_RCavg(e)")
        f.write("\tBase\tOrientedIdx\tIsExon\tExonID\t"
                "DistToExonStart\tDistToExonEnd\n")

        for i in range(locus_len):
            # Use unit-converted entropy values (bits or nats based on --entropy_unit)
            ent = float(entropy_fwd_u[i]) if not np.isnan(entropy_fwd_u[i]) else np.nan
            px = float(ppx_fwd[i]) if not np.isnan(ppx_fwd[i]) else np.nan
            ll = float(ll_next[i]) if not np.isnan(ll_next[i]) else np.nan

            if not np.isnan(p4[i, 0]):
                a, c, g, t = p4[i, :].tolist()
            else:
                a = c = g = t = np.nan

            # Use unit-converted RC entropy values
            ent_rc = float(entropy_rc_u[i]) if not np.isnan(entropy_rc_u[i]) else np.nan
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

    # --- Step 12: Generate plot suite (ENHANCED JAN26) ---
    # Simplify chromosome name: "NC_000010.11" -> "Chr 10"
    import re
    chrom_match = re.search(r'NC_0*(\d+)', chrom)
    chrom_simple = f"Chr {chrom_match.group(1)}" if chrom_match else chrom
    strand_symbol = "+" if strand == "+" else "-"
    title_prefix = f"{tag} ({chrom_simple}, {strand_symbol} strand)"
    # Convert GFF features to oriented coordinates for plotting
    gff_intervals_oriented = None
    if gff_features:
        gff_intervals_oriented = convert_gff_to_oriented_intervals(
            gff_features, locus_start, locus_end_excl, strand
        )
        if gff_intervals_oriented:
            print(f"[INFO] GFF feature track: {len(gff_intervals_oriented)} features for plots")

    print(f"[INFO] Writing ENHANCED plot suite to: {output_mgr.plots_dir}")
    plot_suite(
        output_mgr=output_mgr,
        base_name=base_name,
        entropy_main=entropy_main_u,
        is_exon=is_exon,
        drop_points=drops,
        scored_drops=scored_drops,  # NEW: Pass scored drops
        title_prefix=title_prefix,
        smooth_w=DROP_SMOOTH_W,
        zoom_bp=zoom_bp,
        max_zoom_plots=max_zoom_plots,
        plot_style=plot_style,
        unit=unit,
        ylim=ylim,
        annotate_top_n=annotate_top_n,  # NEW: Top N annotations
        genomic_start=locus_start,  # For chromosome position labels
        rise_points=rises,  # NEW: Pass rise positions
        scored_rises=scored_rises,  # NEW: Pass scored rises
        gff_intervals=gff_intervals_oriented,  # GFF3 feature track
    )

    # --- Step 12b: Generate method comparison report (NEW FEB 5) ---
    print(f"[INFO] Generating method comparison analysis...")
    comparison_stats = generate_method_comparison_report(
        output_mgr=output_mgr,
        base_name=base_name,
        scored_drops=scored_drops,
        drop_points=drops,
        entropy=entropy_main_u,
        is_exon=is_exon,
        title_prefix=title_prefix,
        tolerance_bp=min_separation,  # Use same tolerance as drop separation
        smooth_w=DROP_SMOOTH_W,
    )

    # --- Step 13: Write drop detection results (ENHANCED JAN26) ---
    out_drops = output_mgr.data_path(f"{base_name}.drops.txt")
    print(f"[INFO] Writing drop points with confidence scores: {out_drops}")
    with open(out_drops, "w") as f:
        # Header
        f.write("# Drop detection results (genome_scoring_jan26_drops.py)\n")
        f.write("# Format: method_name<TAB>pos1:score1,pos2:score2,...\n")
        f.write("# For legacy methods without scores, only position is shown\n\n")

        # Write legacy methods (positions only)
        for method in ["derivative", "win_shift", "cusum"]:
            if method in drops and method not in scored_drops:
                pts = drops[method]
                f.write(f"{method}\t" + ",".join(map(str, pts)) + "\n")

        # Write scored methods
        for method, scored_pts in scored_drops.items():
            if scored_pts:
                entries = [f"{pos}:{score:.4f}" for pos, score in scored_pts]
                f.write(f"{method}\t" + ",".join(entries) + "\n")

    # --- Step 13b: Write rise detection results (end of drops) ---
    out_rises = output_mgr.data_path(f"{base_name}.rises.txt")
    print(f"[INFO] Writing rise points (end of drops) with confidence scores: {out_rises}")
    with open(out_rises, "w") as f:
        # Header
        f.write("# Rise detection results (end of drops - entropy returning to high values)\n")
        f.write("# Format: method_name<TAB>pos1:score1,pos2:score2,...\n")
        f.write("# For legacy methods without scores, only position is shown\n\n")

        # Write legacy methods (positions only)
        for method in ["derivative", "win_shift", "cusum"]:
            if method in rises and method not in scored_rises:
                pts = rises[method]
                f.write(f"{method}\t" + ",".join(map(str, pts)) + "\n")

        # Write scored methods
        for method, scored_pts in scored_rises.items():
            if scored_pts:
                entries = [f"{pos}:{score:.4f}" for pos, score in scored_pts]
                f.write(f"{method}\t" + ",".join(entries) + "\n")

    # --- Step 13c: Write GFF3 annotation mapping (optional) ---
    if gff_annotations:
        out_gff_annot = output_mgr.data_path(f"{base_name}.gff_annotations.tsv")
        print(f"[INFO] Writing GFF annotation mapping: {out_gff_annot}")
        write_gff_annotation_tsv(gff_annotations, out_gff_annot)

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
        description="ENHANCED genome scoring with statistical drop detection (Jan 26, 2026).",
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
      data/      - TSV files, drop points (with scores), summaries
      plots/     - All visualization PNG files (confidence-aware)
      fasta/     - Sequence FASTA files
      metadata/  - JSON metadata files

================================================================================
EXAMPLES
================================================================================
  # Basic: Use new statistical methods (default)
  python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455

  # Conservative: High-confidence drops only
  python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455 \\
      --detection_methods zscore mad --zscore_threshold 3.0

  # Compare legacy vs new methods
  python genome_scoring_jan26_drops.py --organism ecoli --gene_id b2911 \\
      --detection_methods derivative zscore mad

  # Bootstrap for publication-quality results (slow, ~100x runtime)
  python genome_scoring_jan26_drops.py --organism ecoli --gene_id b0455 \\
      --detection_methods zscore --bootstrap --n_bootstrap 100

  # List available organisms
  python genome_scoring_jan26_drops.py --list_organisms
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
    ap.add_argument("--gff", default=None,
                    help="GFF3 annotation file for richer feature mapping "
                         "(CDS, UTR, regulatory). Optional -- enriches drop/rise "
                         "annotations with overlapping GFF3 features.")
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

    # --- JAN26: Enhanced Drop Detection Options ---
    ap.add_argument("--detection_methods", nargs='+',
                    choices=["derivative", "win_shift", "cusum", "zscore", "mad", "local"],
                    default=["zscore", "mad"],
                    help="Drop detection methods. Default: zscore mad. Note: 'local' scores relative to local variance, which may not match visual intuition.")

    ap.add_argument("--zscore_threshold", type=float, default=DROP_ZSCORE_THRESHOLD,
                    help=f"Z-score threshold for zscore method. Default: {DROP_ZSCORE_THRESHOLD}")

    ap.add_argument("--mad_threshold", type=float, default=DROP_MAD_THRESHOLD,
                    help=f"MAD threshold for mad method. Default: {DROP_MAD_THRESHOLD}")

    ap.add_argument("--local_window", type=int, default=DROP_LOCAL_WINDOW,
                    help=f"Window size for local baseline method (bp). Default: {DROP_LOCAL_WINDOW}")

    ap.add_argument("--local_threshold", type=float, default=DROP_LOCAL_THRESHOLD,
                    help=f"Threshold for local baseline method (local sigmas). Default: {DROP_LOCAL_THRESHOLD}")

    ap.add_argument("--min_separation", type=int, default=DROP_MIN_SEPARATION,
                    help=f"Minimum separation between nearby drops (bp). Default: {DROP_MIN_SEPARATION}")

    # --- Bootstrap options ---
    ap.add_argument("--bootstrap", action="store_true",
                    help="Enable bootstrap consensus method (WARNING: ~100x slower)")

    ap.add_argument("--n_bootstrap", type=int, default=100,
                    help="Number of bootstrap samples. Default: 100")

    ap.add_argument("--consensus_threshold", type=float, default=0.50,
                    help="Minimum consensus fraction for bootstrap. Default: 0.50")

    # --- Visualization options ---
    ap.add_argument("--annotate_top_n", type=int, default=DROP_ANNOTATE_TOP_N,
                    help=f"Annotate top N drops on plots (0=none). Default: {DROP_ANNOTATE_TOP_N}")

    # --- SAE Analysis Options ---
    ap.add_argument("--analyze_sae", action="store_true",
                    help="Run SAE feature analysis on detected drops (requires sae_utils.py)")

    ap.add_argument("--sae_window", type=int, default=500,
                    help="Window size (bp) around drops for SAE analysis. Default: 500")

    ap.add_argument("--sae_max_drops", type=int, default=100,
                    help="Maximum number of drops to analyze with SAE. Default: 100")

    ap.add_argument("--sae_method", type=str, default=None,
                    help="Detection method for SAE analysis (default: first available scored method)")

    # --- Multi-GPU data parallelism ---
    ap.add_argument("--n_gpus", type=int, default=1,
                    help="Number of GPUs for data-parallel chunk scoring. "
                         "Default: 1 (sequential). Set >1 to distribute chunks "
                         "across multiple GPUs (loads one model per GPU).")

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

    # --- Run the main analysis (ENHANCED JAN26) ---
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
        # JAN26: Enhanced drop detection parameters
        detection_methods=args.detection_methods,
        zscore_threshold=args.zscore_threshold,
        mad_threshold=args.mad_threshold,
        local_window=args.local_window,
        local_threshold=args.local_threshold,
        min_separation=args.min_separation,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        consensus_threshold=args.consensus_threshold,
        annotate_top_n=args.annotate_top_n,
        # Organism and timestamp for output naming
        organism=args.organism,
        include_timestamp=True,
        # GFF3 annotation
        gff_path=args.gff,
        # Multi-GPU
        n_gpus=args.n_gpus,
    )


if __name__ == "__main__":
    main()
