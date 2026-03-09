#!/usr/bin/env python3
"""
discover_region_features.py — Differential SAE Feature Discovery

Finds SAE features enriched in a target genomic region vs background.
Extracts a single window spanning target + background, runs SAE once,
then computes enrichment statistics for all 32,768 features.

Use case: Find features specific to CRISPR spacer regions, prophage
insertions, or any other target locus.

Usage:
    # CRISPR spacer region in E. coli (intergenic between iap and cas2)
    python tools/discover_region_features.py \
        --fasta /path/to/ecoli.fna \
        --chrom NC_000913.3 \
        --target_start 2877618 --target_end 2878569 \
        --bg_from_gtf --bg_flank 5000 \
        --gtf /path/to/genomic.gtf \
        --output_dir results

    # Manual background specification
    python tools/discover_region_features.py \
        --fasta /path/to/ecoli.fna \
        --chrom NC_000913.3 \
        --target_start 2877618 --target_end 2878569 \
        --bg_start 2872000 --bg_end 2884000 \
        --output_dir results
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

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
    logger = logging.getLogger("discover_features")
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

N_SAE_FEATURES = 32768
WINDOW_PADDING = 1000  # extra padding around the full window

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
# BACKGROUND REGION CONSTRUCTION
# =============================================================================

def build_background_from_gtf(
    gtf_path: str,
    chrom: str,
    target_start: int,
    target_end: int,
    flank: int = 5000,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Build background region from CDS features flanking the target.

    Returns:
        (window_start, window_end, bg_intervals)
        where bg_intervals is a list of (start, end) CDS regions in genomic coords.
    """
    from tools.analyze_scoring_results import load_annotation_features

    # Load features in the flanking region
    region_start = target_start - flank
    region_end = target_end + flank

    features = load_annotation_features(
        gtf_path, chrom, region_start, region_end, fmt="gtf"
    )

    # Collect CDS intervals as background
    bg_intervals = []
    for feat in features:
        if feat["feature_type"] != "CDS":
            continue
        # Exclude any CDS overlapping with target
        if feat["end_exclusive"] <= target_start or feat["start"] >= target_end:
            cds_start = max(feat["start"], region_start)
            cds_end = min(feat["end_exclusive"], region_end)
            bg_intervals.append((cds_start, cds_end))

    # If no CDS found, fall back to flanking non-target regions
    if not bg_intervals:
        bg_intervals = [
            (region_start, target_start),
            (target_end, region_end),
        ]

    window_start = region_start - WINDOW_PADDING
    window_end = region_end + WINDOW_PADDING
    return window_start, window_end, bg_intervals


# =============================================================================
# ENRICHMENT ANALYSIS
# =============================================================================

def compute_enrichment(
    feature_matrix: np.ndarray,
    target_mask: np.ndarray,
    bg_mask: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Compute enrichment statistics for all features.

    Args:
        feature_matrix: (n_positions, 32768) float32 array
        target_mask: boolean array, True for target positions
        bg_mask: boolean array, True for background positions

    Returns:
        List of feature dicts sorted by effect size, filtered to p_adj < 0.05.
    """
    from scipy.stats import mannwhitneyu

    n_target = int(np.sum(target_mask))
    n_bg = int(np.sum(bg_mask))
    if logger:
        logger.info(f"Target: {n_target} positions, Background: {n_bg} positions")

    target_data = feature_matrix[target_mask]  # (n_target, 32768)
    bg_data = feature_matrix[bg_mask]          # (n_bg, 32768)

    eps = 1e-10

    # Pre-filter: skip features with zero activation in target
    target_any_active = np.any(target_data > 0, axis=0)
    candidate_ids = np.where(target_any_active)[0]
    if logger:
        logger.info(f"Features active in target: {len(candidate_ids)} / {N_SAE_FEATURES}")

    results = []
    n_tests = len(candidate_ids)

    for idx, fid in enumerate(candidate_ids):
        t_vals = target_data[:, fid]
        b_vals = bg_data[:, fid]

        # Mean activation ratio
        t_mean = float(np.mean(t_vals))
        b_mean = float(np.mean(b_vals))
        mean_ratio = t_mean / (b_mean + eps)

        # Specificity score
        t_frac_active = float(np.mean(t_vals > 0))
        b_frac_active = float(np.mean(b_vals > 0))
        specificity = t_frac_active / (b_frac_active + eps)

        # Mann-Whitney U test
        # Only run if there's variance in at least one group
        if np.all(t_vals == 0) and np.all(b_vals == 0):
            continue

        try:
            u_stat, p_val = mannwhitneyu(t_vals, b_vals, alternative="greater")
        except ValueError:
            # All values identical
            continue

        # Bonferroni correction
        p_adj = min(p_val * n_tests, 1.0)

        # Effect size: rank-biserial correlation
        # r = 1 - (2U) / (n1 * n2)
        effect_size = 1.0 - (2.0 * u_stat) / (n_target * n_bg) if (n_target * n_bg) > 0 else 0.0
        # Note: mannwhitneyu with alternative="greater" gives U = sum of ranks in first sample - n1*(n1+1)/2
        # For rank-biserial: r = 2U/(n1*n2) - 1
        effect_size = 2.0 * u_stat / (n_target * n_bg) - 1.0 if (n_target * n_bg) > 0 else 0.0

        label = KNOWN_BIO_FEATURES.get(int(fid), ("", ""))

        results.append({
            "feature_id": int(fid),
            "label": label[0],
            "description": label[1],
            "mean_target": t_mean,
            "mean_background": b_mean,
            "mean_ratio": mean_ratio,
            "frac_target_active": t_frac_active,
            "frac_bg_active": b_frac_active,
            "specificity": specificity,
            "u_statistic": float(u_stat),
            "p_value": float(p_val),
            "p_adj": float(p_adj),
            "effect_size": effect_size,
        })

        if logger and (idx + 1) % 100 == 0:
            logger.debug(f"  Tested {idx+1}/{n_tests} features")

    # Filter to significant and sort by effect size
    significant = [r for r in results if r["p_adj"] < 0.05]
    significant.sort(key=lambda r: -r["effect_size"])

    if logger:
        logger.info(f"Significant features (p_adj < 0.05): {len(significant)} / {len(results)} tested")

    return significant


# =============================================================================
# PLOTTING
# =============================================================================

def plot_top_enriched(
    enriched: List[Dict[str, Any]],
    output_path: str,
    n_top: int = 20,
    chrom: str = "",
    target_start: int = 0,
    target_end: int = 0,
):
    """Plot top enriched features (Figure 4g-style horizontal bar chart)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = enriched[:n_top]
    if not top:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))

    y_pos = np.arange(len(top))
    effect_sizes = [r["effect_size"] for r in top]
    labels = []
    for r in top:
        lbl = f"f/{r['feature_id']}"
        if r["label"]:
            lbl += f" ({r['label']})"
        labels.append(lbl)

    colors = ["#e74c3c" if r["p_adj"] < 0.001 else "#f39c12" if r["p_adj"] < 0.01 else "#3498db"
              for r in top]

    ax.barh(y_pos, effect_sizes, color=colors, alpha=0.8, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Effect size (rank-biserial correlation)", fontsize=10)
    ax.set_title(
        f"Top enriched SAE features in target region\n"
        f"{chrom}:{target_start:,}-{target_end:,}",
        fontsize=12,
    )

    # Add p-value text
    for i, r in enumerate(top):
        ax.text(effect_sizes[i] + 0.01, i,
                f"p={r['p_adj']:.1e}", va="center", fontsize=7, color="#555")

    # Legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="p < 0.001"),
        Patch(facecolor="#f39c12", label="p < 0.01"),
        Patch(facecolor="#3498db", label="p < 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_volcano(
    all_results: List[Dict[str, Any]],
    output_path: str,
    chrom: str = "",
    target_start: int = 0,
    target_end: int = 0,
):
    """Volcano plot: effect size vs -log10(p_adj)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all_results:
        return

    effect_sizes = np.array([r["effect_size"] for r in all_results])
    p_adj = np.array([max(r["p_adj"], 1e-300) for r in all_results])
    neg_log_p = -np.log10(p_adj)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by significance
    sig_mask = p_adj < 0.05
    ax.scatter(effect_sizes[~sig_mask], neg_log_p[~sig_mask],
               c="#95a5a6", alpha=0.4, s=15, edgecolors="none", label="NS")
    ax.scatter(effect_sizes[sig_mask], neg_log_p[sig_mask],
               c="#e74c3c", alpha=0.7, s=25, edgecolors="none", label="p_adj < 0.05")

    # Label top hits
    top_by_effect = sorted(
        [r for r in all_results if r["p_adj"] < 0.05],
        key=lambda r: -r["effect_size"]
    )[:10]
    for r in top_by_effect:
        lbl = f"f/{r['feature_id']}"
        if r["label"]:
            lbl += f" ({r['label']})"
        ax.annotate(lbl, (r["effect_size"], -np.log10(max(r["p_adj"], 1e-300))),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Effect size (rank-biserial correlation)", fontsize=11)
    ax.set_ylabel("-log10(p_adj)", fontsize=11)
    ax.set_title(
        f"Enrichment volcano — {chrom}:{target_start:,}-{target_end:,}",
        fontsize=12,
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_vs_background(
    feature_matrix: np.ndarray,
    target_mask: np.ndarray,
    bg_mask: np.ndarray,
    enriched: List[Dict[str, Any]],
    output_path: str,
    n_top: int = 20,
    chrom: str = "",
    target_start: int = 0,
    target_end: int = 0,
):
    """Side-by-side heatmap of top features in target vs background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = enriched[:n_top]
    if not top:
        return

    fids = [r["feature_id"] for r in top]
    target_data = feature_matrix[target_mask][:, fids]  # (n_target, n_top)
    bg_data = feature_matrix[bg_mask][:, fids]           # (n_bg, n_top)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(fids) * 0.3)),
                                    sharey=True, width_ratios=[1, 1])

    labels = []
    for r in top:
        lbl = f"f/{r['feature_id']}"
        if r["label"]:
            lbl += f" ({r['label']})"
        labels.append(lbl)

    # Compute mean activation per feature for bar chart
    target_means = np.mean(target_data, axis=0)
    bg_means = np.mean(bg_data, axis=0)

    y = np.arange(len(fids))
    bar_height = 0.35

    ax1.barh(y - bar_height/2, target_means, bar_height, color="#e74c3c",
             alpha=0.8, label="Target")
    ax1.barh(y + bar_height/2, bg_means, bar_height, color="#3498db",
             alpha=0.8, label="Background")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean activation", fontsize=10)
    ax1.set_title("Mean activation", fontsize=11)
    ax1.legend(fontsize=8)

    # Fraction active
    target_frac = np.mean(target_data > 0, axis=0)
    bg_frac = np.mean(bg_data > 0, axis=0)

    ax2.barh(y - bar_height/2, target_frac, bar_height, color="#e74c3c",
             alpha=0.8, label="Target")
    ax2.barh(y + bar_height/2, bg_frac, bar_height, color="#3498db",
             alpha=0.8, label="Background")
    ax2.set_xlabel("Fraction active", fontsize=10)
    ax2.set_title("Fraction of positions active", fontsize=11)
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"Target vs Background — {chrom}:{target_start:,}-{target_end:,}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# OUTPUT
# =============================================================================

def write_enriched_tsv(
    enriched: List[Dict[str, Any]],
    output_path: str,
):
    """Write enriched features to TSV."""
    with open(output_path, "w") as f:
        f.write("rank\tfeature_id\tlabel\tdescription\tmean_target\tmean_background\t"
                "mean_ratio\tfrac_target_active\tfrac_bg_active\tspecificity\t"
                "u_statistic\tp_value\tp_adj\teffect_size\n")
        for i, r in enumerate(enriched):
            f.write(f"{i+1}\t{r['feature_id']}\t{r['label']}\t{r['description']}\t"
                    f"{r['mean_target']:.6f}\t{r['mean_background']:.6f}\t"
                    f"{r['mean_ratio']:.2f}\t{r['frac_target_active']:.4f}\t"
                    f"{r['frac_bg_active']:.4f}\t{r['specificity']:.2f}\t"
                    f"{r['u_statistic']:.1f}\t{r['p_value']:.2e}\t"
                    f"{r['p_adj']:.2e}\t{r['effect_size']:.4f}\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Differential SAE feature discovery for a target region",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="Path to genome FASTA file")
    parser.add_argument("--chrom", required=True,
                        help="Chromosome/accession (e.g. NC_000913.3)")
    parser.add_argument("--target_start", type=int, required=True,
                        help="Target region start (1-based)")
    parser.add_argument("--target_end", type=int, required=True,
                        help="Target region end (1-based)")

    # Background specification (mutually exclusive modes)
    bg_group = parser.add_argument_group("Background region")
    bg_group.add_argument("--bg_from_gtf", action="store_true",
                          help="Build background from CDS features flanking the target")
    bg_group.add_argument("--bg_flank", type=int, default=5000,
                          help="Flank size for GTF-based background (default: 5000)")
    bg_group.add_argument("--bg_start", type=int, default=None,
                          help="Manual background start (genomic coord)")
    bg_group.add_argument("--bg_end", type=int, default=None,
                          help="Manual background end (genomic coord)")
    bg_group.add_argument("--gtf", default=None,
                          help="Path to GTF file (required with --bg_from_gtf)")

    parser.add_argument("--output_dir", default="results",
                        help="Base output directory (default: results)")
    parser.add_argument("--chrom_name", default=None,
                        help="Friendly chromosome name for output dir (e.g. ecoli_K12)")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t_start = time.time()

    # --- Validate args ---
    if args.bg_from_gtf and not args.gtf:
        logger.error("--bg_from_gtf requires --gtf")
        sys.exit(1)
    if not args.bg_from_gtf and (args.bg_start is None or args.bg_end is None):
        # Default: use flanking regions as background
        args.bg_start = args.target_start - 5000
        args.bg_end = args.target_end + 5000
        logger.info(f"No background specified, using ±5000bp flanking regions")

    # --- Load sequence ---
    logger.info(f"Loading chromosome {args.chrom} from {args.fasta}")
    from run_sae_on_chromosome_drops import load_chromosome_sequence, CHROM_MAP
    sequence = load_chromosome_sequence(args.fasta, args.chrom, logger)
    genome_len = len(sequence)
    logger.info(f"Genome length: {genome_len:,} bp")

    # --- Determine window and background ---
    chrom_id = CHROM_MAP.get(args.chrom, args.chrom)

    if args.bg_from_gtf:
        logger.info("Building background from GTF CDS features...")
        window_start, window_end, bg_intervals = build_background_from_gtf(
            args.gtf, chrom_id, args.target_start, args.target_end, args.bg_flank,
        )
    else:
        window_start = max(0, min(args.bg_start, args.target_start) - WINDOW_PADDING)
        window_end = min(genome_len, max(args.bg_end, args.target_end) + WINDOW_PADDING)
        # Background = specified range minus target
        bg_intervals = []
        if args.bg_start < args.target_start:
            bg_intervals.append((max(args.bg_start, 0), args.target_start))
        if args.bg_end > args.target_end:
            bg_intervals.append((args.target_end, min(args.bg_end, genome_len)))

    window_start = max(0, window_start)
    window_end = min(genome_len, window_end)
    window_len = window_end - window_start

    logger.info(f"Window: {window_start:,}-{window_end:,} ({window_len:,} bp)")
    logger.info(f"Target: {args.target_start:,}-{args.target_end:,} "
                 f"({args.target_end - args.target_start:,} bp)")
    logger.info(f"Background intervals: {len(bg_intervals)}")
    for s, e in bg_intervals:
        logger.info(f"  {s:,}-{e:,} ({e-s:,} bp)")

    # --- Build output directory ---
    chrom_name = args.chrom_name or args.chrom.replace(".", "_")
    flags = f"target_{args.target_start}_{args.target_end}"
    run_dir = build_run_dir(args.output_dir, chrom_name, "sae_differential", flags)
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    # --- Initialize model and SAE ---
    logger.info("Initializing Evo2 model and SAE...")
    _import_torch()
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf, get_feature_ts
    model = ObservableEvo2("evo2_7b")
    sae = load_topk_sae_from_hf(model.d_hidden, model.device, model.dtype)
    logger.info("Model and SAE loaded")

    # --- Extract window and run SAE ---
    window_seq = sequence[window_start:window_end]
    logger.info(f"Running SAE on {len(window_seq):,} bp window...")
    t_sae = time.time()
    feature_matrix = get_feature_ts(model, sae, window_seq)
    sae_time = time.time() - t_sae
    logger.info(f"SAE extraction complete in {sae_time:.1f}s, "
                 f"matrix shape: {feature_matrix.shape}")

    # --- Build masks ---
    # Convert genomic coords to window-relative coords
    target_rel_start = args.target_start - window_start
    target_rel_end = args.target_end - window_start

    actual_len = feature_matrix.shape[0]
    target_mask = np.zeros(actual_len, dtype=bool)
    target_rel_start_clamp = max(0, min(target_rel_start, actual_len))
    target_rel_end_clamp = max(0, min(target_rel_end, actual_len))
    target_mask[target_rel_start_clamp:target_rel_end_clamp] = True

    bg_mask = np.zeros(actual_len, dtype=bool)
    for bg_s, bg_e in bg_intervals:
        rel_s = max(0, min(bg_s - window_start, actual_len))
        rel_e = max(0, min(bg_e - window_start, actual_len))
        bg_mask[rel_s:rel_e] = True

    # Ensure no overlap between target and background
    bg_mask &= ~target_mask

    logger.info(f"Target positions: {np.sum(target_mask):,}")
    logger.info(f"Background positions: {np.sum(bg_mask):,}")

    if np.sum(target_mask) == 0:
        logger.error("No target positions in window — check coordinates")
        sys.exit(1)
    if np.sum(bg_mask) == 0:
        logger.error("No background positions in window — check coordinates")
        sys.exit(1)

    # --- Compute enrichment ---
    logger.info("Computing enrichment statistics...")
    # Get all results (not just significant) for volcano plot
    enriched = compute_enrichment(feature_matrix, target_mask, bg_mask, logger)

    # Also compute all results for volcano (re-run without significance filter)
    all_results = compute_all_features(feature_matrix, target_mask, bg_mask, logger)

    # --- Save data ---
    # Enriched features TSV
    enriched_path = os.path.join(data_dir, "enriched_features.tsv")
    write_enriched_tsv(enriched, enriched_path)
    logger.info(f"Saved {len(enriched)} enriched features to {enriched_path}")

    # Feature matrices (for re-analysis)
    fids_to_save = [r["feature_id"] for r in enriched[:50]]
    if fids_to_save:
        np.savez_compressed(
            os.path.join(data_dir, "target_feature_matrix.npz"),
            features=feature_matrix[target_mask][:, fids_to_save],
            feature_ids=np.array(fids_to_save),
        )
        np.savez_compressed(
            os.path.join(data_dir, "background_feature_matrix.npz"),
            features=feature_matrix[bg_mask][:, fids_to_save],
            feature_ids=np.array(fids_to_save),
        )
        logger.info("Saved target/background feature matrices")

    # Region definitions
    region_defs = {
        "chrom": args.chrom,
        "chrom_id": chrom_id,
        "target_start": args.target_start,
        "target_end": args.target_end,
        "target_length": args.target_end - args.target_start,
        "window_start": window_start,
        "window_end": window_end,
        "background_intervals": bg_intervals,
        "n_target_positions": int(np.sum(target_mask)),
        "n_background_positions": int(np.sum(bg_mask)),
        "feature_matrix_shape": list(feature_matrix.shape),
        "sae_extraction_time_s": round(sae_time, 1),
    }
    with open(os.path.join(data_dir, "region_definitions.json"), "w") as f:
        json.dump(region_defs, f, indent=2)
        f.write("\n")

    # --- Generate plots ---
    logger.info("Generating plots...")

    plot_top_enriched(
        enriched,
        os.path.join(plots_dir, "top_enriched_features.png"),
        n_top=20, chrom=args.chrom,
        target_start=args.target_start, target_end=args.target_end,
    )

    plot_volcano(
        all_results,
        os.path.join(plots_dir, "enrichment_volcano.png"),
        chrom=args.chrom,
        target_start=args.target_start, target_end=args.target_end,
    )

    plot_target_vs_background(
        feature_matrix, target_mask, bg_mask, enriched,
        os.path.join(plots_dir, "target_vs_background.png"),
        n_top=20, chrom=args.chrom,
        target_start=args.target_start, target_end=args.target_end,
    )

    logger.info("Plots saved")

    # --- Write provenance ---
    write_source(run_dir, fasta=args.fasta, gtf=args.gtf)

    wall_time = time.time() - t_start
    write_completed(run_dir, "discover_region_features.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


def compute_all_features(
    feature_matrix: np.ndarray,
    target_mask: np.ndarray,
    bg_mask: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Compute enrichment for all active features (no significance filter).

    Used for volcano plot — returns all tested features regardless of p-value.
    """
    from scipy.stats import mannwhitneyu

    target_data = feature_matrix[target_mask]
    bg_data = feature_matrix[bg_mask]
    n_target = target_data.shape[0]
    n_bg = bg_data.shape[0]
    eps = 1e-10

    target_any_active = np.any(target_data > 0, axis=0)
    bg_any_active = np.any(bg_data > 0, axis=0)
    candidate_ids = np.where(target_any_active | bg_any_active)[0]

    if logger:
        logger.info(f"Computing all-features enrichment for volcano ({len(candidate_ids)} features)...")

    n_tests = len(candidate_ids)
    results = []

    for fid in candidate_ids:
        t_vals = target_data[:, fid]
        b_vals = bg_data[:, fid]

        if np.all(t_vals == 0) and np.all(b_vals == 0):
            continue

        t_mean = float(np.mean(t_vals))
        b_mean = float(np.mean(b_vals))

        try:
            u_stat, p_val = mannwhitneyu(t_vals, b_vals, alternative="greater")
        except ValueError:
            continue

        p_adj = min(p_val * n_tests, 1.0)
        effect_size = 2.0 * u_stat / (n_target * n_bg) - 1.0 if (n_target * n_bg) > 0 else 0.0

        label = KNOWN_BIO_FEATURES.get(int(fid), ("", ""))
        results.append({
            "feature_id": int(fid),
            "label": label[0],
            "effect_size": effect_size,
            "p_adj": float(p_adj),
            "mean_target": t_mean,
            "mean_background": b_mean,
        })

    return results


if __name__ == "__main__":
    main()
