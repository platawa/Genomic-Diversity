#!/usr/bin/env python3
"""
scan_feature_genome.py — Genome-wide SAE Feature Scanner

Scans an entire genome for activation of specific SAE feature(s).
Processes in 8192bp chunks, keeping only requested features per chunk
to stay memory-safe (~90MB vs ~600GB for the full feature matrix).

Usage:
    python tools/scan_feature_genome.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000913.3 \
        --features 19745 \
        --gtf /path/to/genomic.gtf \
        --output_dir results

    # Multiple features
    python tools/scan_feature_genome.py \
        --fasta /path/to/genome.fna \
        --chrom NC_000913.3 \
        --features 19745 15680 28339 \
        --gtf /path/to/genomic.gtf \
        --output_dir results
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed, write_source

# Defer torch import
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
    logger = logging.getLogger("scan_feature")
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
# CONSTANTS
# =============================================================================

CHUNK_SIZE = 8192
OVERLAP = 256
MERGE_GAP = 50  # merge contiguous regions if gap < 50bp

# Known biological features from Evo2 paper
KNOWN_BIO_FEATURES = {
    15680: ("CDS",        "coding regions"),
    28339: ("Intron",     "introns"),
    1050:  ("Exon start", "first base of exon following intron"),
    25666: ("Exon end",   "last base of exon followed by intron"),
    24278: ("Frameshift", "mutation-sensitive, frameshifts & premature stops"),
    19745: ("Prophage",   "prophage regions across prokaryotes"),
}


# =============================================================================
# GTF ANNOTATION
# =============================================================================

def load_gtf_genes(gtf_path: str, chrom: str) -> List[Dict[str, Any]]:
    """Load gene features from GTF for a chromosome."""
    from tools.analyze_scoring_results import parse_gtf_attributes

    genes = []
    with open(gtf_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            seqid, source, ftype, f_start, f_end, score, strand, frame, attrs = fields
            if seqid != chrom:
                continue
            if ftype not in ("gene", "CDS"):
                continue
            attrs_d = parse_gtf_attributes(attrs)
            name = (attrs_d.get("gene_name") or attrs_d.get("Name")
                    or attrs_d.get("gene") or attrs_d.get("gene_id", ""))
            genes.append({
                "start": int(f_start),
                "end": int(f_end),
                "feature_type": ftype,
                "strand": strand,
                "name": name,
            })
    genes.sort(key=lambda g: g["start"])
    return genes


def annotate_regions_with_genes(
    regions: List[Dict[str, Any]],
    genes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add overlapping gene names to each activation region."""
    for region in regions:
        r_start, r_end = region["start"], region["end"]
        overlapping = []
        for g in genes:
            if g["end"] < r_start:
                continue
            if g["start"] > r_end:
                break
            overlapping.append(g["name"])
        region["overlapping_genes"] = list(dict.fromkeys(overlapping))  # dedupe, preserve order
    return regions


# =============================================================================
# REGION DETECTION
# =============================================================================

def find_activation_regions(
    activation: np.ndarray,
    merge_gap: int = MERGE_GAP,
) -> List[Dict[str, Any]]:
    """Find contiguous stretches where activation > 0, merging close gaps."""
    active = activation > 0
    if not np.any(active):
        return []

    # Find runs of active positions
    diff = np.diff(active.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if active[0]:
        starts = np.concatenate([[0], starts])
    if active[-1]:
        ends = np.concatenate([ends, [len(activation)]])

    # Build initial regions
    raw_regions = list(zip(starts.tolist(), ends.tolist()))

    # Merge regions with gap < merge_gap
    if not raw_regions:
        return []

    merged = [list(raw_regions[0])]
    for s, e in raw_regions[1:]:
        if s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Build region dicts with stats
    regions = []
    for s, e in merged:
        chunk = activation[s:e]
        regions.append({
            "start": int(s),
            "end": int(e),
            "length": int(e - s),
            "mean_activation": float(np.mean(chunk)),
            "max_activation": float(np.max(chunk)),
            "total_activation": float(np.sum(chunk)),
            "active_positions": int(np.sum(chunk > 0)),
        })

    # Sort by total activation descending
    regions.sort(key=lambda r: -r["total_activation"])
    return regions


# =============================================================================
# GENOME SCANNING
# =============================================================================

def scan_genome(
    sequence: str,
    model,
    sae,
    feature_ids: List[int],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, np.ndarray]:
    """Scan genome in chunks, returning per-feature activation arrays.

    Processes the genome in overlapping chunks. For each chunk, runs
    get_feature_ts() but only keeps the requested feature columns.

    Returns:
        Dict mapping feature_id -> np.ndarray of shape (genome_len,)
    """
    from sae_utils import get_feature_ts

    genome_len = len(sequence)
    stride = chunk_size - overlap

    # Pre-allocate output arrays
    activations = {fid: np.zeros(genome_len, dtype=np.float32) for fid in feature_ids}
    # Count array for averaging overlapping regions
    counts = np.zeros(genome_len, dtype=np.float32)

    n_chunks = (genome_len - overlap + stride - 1) // stride
    if logger:
        logger.info(f"Scanning {genome_len:,} bp in {n_chunks} chunks "
                     f"({chunk_size} bp, {overlap} bp overlap)")

    t0 = time.time()
    for i in range(n_chunks):
        chunk_start = i * stride
        chunk_end = min(chunk_start + chunk_size, genome_len)
        chunk_seq = sequence[chunk_start:chunk_end]

        if len(chunk_seq) < 10:
            continue

        # Run SAE — returns (seq_len, 32768), we keep only requested features
        feature_ts = get_feature_ts(model, sae, chunk_seq)
        actual_len = feature_ts.shape[0]

        # For overlapping regions, accumulate and average later
        for fid in feature_ids:
            activations[fid][chunk_start:chunk_start + actual_len] += feature_ts[:, fid]
        counts[chunk_start:chunk_start + actual_len] += 1.0

        if logger and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_chunks - i - 1) / rate if rate > 0 else 0
            logger.info(f"  Chunk {i+1}/{n_chunks} ({elapsed:.0f}s elapsed, "
                         f"~{eta:.0f}s remaining)")

        # Free GPU memory
        del feature_ts

    # Average overlapping regions
    mask = counts > 0
    for fid in feature_ids:
        activations[fid][mask] /= counts[mask]

    elapsed = time.time() - t0
    if logger:
        logger.info(f"Scanning complete in {elapsed:.1f}s")

    return activations


# =============================================================================
# PLOTTING
# =============================================================================

def plot_genome_wide(
    activation: np.ndarray,
    feature_id: int,
    output_path: str,
    chrom: str,
    genes: Optional[List[Dict[str, Any]]] = None,
):
    """Plot genome-wide feature activation profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = KNOWN_BIO_FEATURES.get(feature_id, (f"f/{feature_id}", ""))[0]

    has_genes = genes is not None and len(genes) > 0
    n_panels = 2 if has_genes else 1
    height_ratios = [4, 1] if has_genes else [1]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(16, 5 if has_genes else 4),
        height_ratios=height_ratios,
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    x = np.arange(len(activation))
    ax.fill_between(x, activation, alpha=0.6, color="#3498db", linewidth=0)
    ax.set_ylabel(f"Feature f/{feature_id}\n({label})", fontsize=10)
    ax.set_title(f"Genome-wide activation of f/{feature_id} ({label}) — {chrom}", fontsize=12)
    ax.set_xlim(0, len(activation))

    if has_genes:
        gene_ax = axes[1]
        _draw_simple_gene_track(gene_ax, genes, 0, len(activation))
        gene_ax.set_xlabel("Genomic position (bp)", fontsize=10)
    else:
        ax.set_xlabel("Genomic position (bp)", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_regions(
    activation: np.ndarray,
    regions: List[Dict[str, Any]],
    feature_id: int,
    output_path: str,
    chrom: str,
    n_top: int = 6,
    padding: int = 2000,
    genes: Optional[List[Dict[str, Any]]] = None,
):
    """Plot zoomed views of top activation regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = KNOWN_BIO_FEATURES.get(feature_id, (f"f/{feature_id}", ""))[0]
    top = regions[:n_top]
    n = len(top)
    if n == 0:
        return

    has_genes = genes is not None and len(genes) > 0
    rows_per_region = 2 if has_genes else 1

    fig, axes = plt.subplots(
        n * rows_per_region, 1,
        figsize=(14, 3 * n * rows_per_region),
        height_ratios=([3, 1] if has_genes else [1]) * n,
    )
    if n * rows_per_region == 1:
        axes = [axes]

    for idx, region in enumerate(top):
        s = max(0, region["start"] - padding)
        e = min(len(activation), region["end"] + padding)
        chunk = activation[s:e]
        x = np.arange(s, e)

        ax_idx = idx * rows_per_region
        ax = axes[ax_idx]
        ax.fill_between(x, chunk, alpha=0.6, color="#3498db", linewidth=0)
        ax.axvspan(region["start"], region["end"], alpha=0.15, color="red")

        gene_str = ", ".join(region.get("overlapping_genes", [])[:5])
        ax.set_title(
            f"Region #{idx+1}: {region['start']:,}-{region['end']:,} "
            f"({region['length']:,} bp, max={region['max_activation']:.2f})"
            + (f" — {gene_str}" if gene_str else ""),
            fontsize=10,
        )
        ax.set_ylabel(f"f/{feature_id}")

        if has_genes:
            gene_ax = axes[ax_idx + 1]
            _draw_simple_gene_track(gene_ax, genes, s, e)

    fig.suptitle(
        f"Top {n} activation regions for f/{feature_id} ({label}) — {chrom}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_simple_gene_track(ax, genes, view_start, view_end):
    """Draw a simple gene track on an axes."""
    from matplotlib.patches import Rectangle

    vis = [g for g in genes if g["end"] >= view_start and g["start"] <= view_end]
    ax.set_xlim(view_start, view_end)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Genes", fontsize=8)

    color_map = {"CDS": "#3498db", "gene": "#2ecc71"}

    for g in vis:
        s = max(g["start"], view_start)
        e = min(g["end"], view_end)
        color = color_map.get(g["feature_type"], "#95a5a6")
        ax.add_patch(Rectangle(
            (s, 0.1), e - s, 0.8,
            facecolor=color, edgecolor="none", alpha=0.7,
        ))
        # Label if wide enough
        if (e - s) / (view_end - view_start) > 0.03 and g.get("name"):
            ax.text((s + e) / 2, 0.5, g["name"][:20],
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold", clip_on=True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# OUTPUT
# =============================================================================

def write_regions_tsv(
    regions: List[Dict[str, Any]],
    feature_id: int,
    output_path: str,
):
    """Write activation regions to TSV."""
    with open(output_path, "w") as f:
        f.write("# Activation regions for feature f/{}\n".format(feature_id))
        f.write("rank\tstart\tend\tlength\tmean_activation\tmax_activation\t"
                "total_activation\tactive_positions\toverlapping_genes\n")
        for i, r in enumerate(regions):
            genes_str = ";".join(r.get("overlapping_genes", []))
            f.write(f"{i+1}\t{r['start']}\t{r['end']}\t{r['length']}\t"
                    f"{r['mean_activation']:.4f}\t{r['max_activation']:.4f}\t"
                    f"{r['total_activation']:.2f}\t{r['active_positions']}\t"
                    f"{genes_str}\n")


def write_summary(
    activations: Dict[int, np.ndarray],
    all_regions: Dict[int, List[Dict[str, Any]]],
    output_path: str,
):
    """Write summary JSON with stats for each feature."""
    summary = {}
    for fid in activations:
        act = activations[fid]
        regions = all_regions.get(fid, [])
        active_mask = act > 0
        summary[f"f/{fid}"] = {
            "feature_id": fid,
            "label": KNOWN_BIO_FEATURES.get(fid, ("unknown", ""))[0],
            "genome_length": int(len(act)),
            "total_active_positions": int(np.sum(active_mask)),
            "fraction_active": float(np.mean(active_mask)),
            "mean_activation_when_active": float(np.mean(act[active_mask])) if np.any(active_mask) else 0.0,
            "max_activation": float(np.max(act)),
            "percentile_99": float(np.percentile(act[active_mask], 99)) if np.any(active_mask) else 0.0,
            "n_regions": len(regions),
            "top_5_regions": [
                {"start": r["start"], "end": r["end"], "length": r["length"],
                 "max_activation": r["max_activation"],
                 "genes": r.get("overlapping_genes", [])}
                for r in regions[:5]
            ],
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genome-wide SAE feature scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="Path to genome FASTA file")
    parser.add_argument("--chrom", required=True,
                        help="Chromosome/accession to scan (e.g. NC_000913.3)")
    parser.add_argument("--features", type=int, nargs="+", required=True,
                        help="SAE feature IDs to scan for (e.g. 19745)")
    parser.add_argument("--gtf", default=None,
                        help="Path to GTF annotation file (optional, for gene overlaps)")
    parser.add_argument("--output_dir", default="results",
                        help="Base output directory (default: results)")
    parser.add_argument("--chrom_name", default=None,
                        help="Friendly chromosome name for output dir (e.g. ecoli_K12)")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE,
                        help=f"Chunk size in bp (default: {CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"Overlap between chunks in bp (default: {OVERLAP})")
    parser.add_argument("--merge_gap", type=int, default=MERGE_GAP,
                        help=f"Merge activation regions closer than this (default: {MERGE_GAP})")
    parser.add_argument("--log_level", default="INFO",
                        help="Logging level (default: INFO)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t_start = time.time()

    # --- Load sequence ---
    logger.info(f"Loading chromosome {args.chrom} from {args.fasta}")
    from run_sae_on_chromosome_drops import load_chromosome_sequence, CHROM_MAP
    sequence = load_chromosome_sequence(args.fasta, args.chrom, logger)
    logger.info(f"Genome length: {len(sequence):,} bp")

    # --- Load GTF genes ---
    genes = []
    if args.gtf:
        logger.info(f"Loading GTF annotations from {args.gtf}")
        chrom_id = CHROM_MAP.get(args.chrom, args.chrom)
        genes = load_gtf_genes(args.gtf, chrom_id)
        logger.info(f"Loaded {len(genes)} gene/CDS features")

    # --- Build output directory ---
    chrom_name = args.chrom_name or args.chrom.replace(".", "_")
    feature_str = "_".join(f"f{fid}" for fid in args.features)
    run_dir = build_run_dir(args.output_dir, chrom_name, "sae_scan", feature_str)
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Initialize model and SAE ---
    logger.info("Initializing Evo2 model and SAE...")
    _import_torch()
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf
    model = ObservableEvo2("evo2_7b")
    sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)
    logger.info("Model and SAE loaded")

    # --- Scan genome ---
    activations = scan_genome(
        sequence, model, sae, args.features,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        logger=logger,
    )

    # --- Detect regions and annotate ---
    all_regions = {}
    for fid in args.features:
        regions = find_activation_regions(activations[fid], merge_gap=args.merge_gap)
        if genes:
            regions = annotate_regions_with_genes(regions, genes)
        all_regions[fid] = regions
        logger.info(f"Feature f/{fid}: {len(regions)} activation regions detected")

    # --- Save data ---
    # Save activations as .npz
    npz_data = {f"f{fid}": activations[fid] for fid in args.features}
    np.savez_compressed(os.path.join(data_dir, "feature_activations.npz"), **npz_data)
    logger.info("Saved feature_activations.npz")

    # Save regions TSV per feature
    for fid in args.features:
        tsv_path = os.path.join(data_dir, f"activation_regions_f{fid}.tsv")
        write_regions_tsv(all_regions[fid], fid, tsv_path)
        logger.info(f"Saved {tsv_path}")

    # Save summary
    summary_path = os.path.join(data_dir, "summary.json")
    write_summary(activations, all_regions, summary_path)
    logger.info("Saved summary.json")

    # --- Generate plots ---
    logger.info("Generating plots...")
    for fid in args.features:
        genome_plot = os.path.join(plots_dir, f"genome_wide_f{fid}.png")
        plot_genome_wide(activations[fid], fid, genome_plot, args.chrom,
                         genes=genes if genes else None)
        logger.info(f"Saved {genome_plot}")

        if all_regions[fid]:
            top_plot = os.path.join(plots_dir, f"top_regions_f{fid}.png")
            plot_top_regions(activations[fid], all_regions[fid], fid, top_plot,
                             args.chrom, genes=genes if genes else None)
            logger.info(f"Saved {top_plot}")

    # --- Write provenance ---
    write_source(run_dir, fasta=args.fasta, gtf=args.gtf)

    wall_time = time.time() - t_start
    write_completed(run_dir, "scan_feature_genome.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
