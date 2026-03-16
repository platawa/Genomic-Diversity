#!/usr/bin/env python3
"""
discover_multi_locus_features.py — Multi-locus contrastive SAE feature discovery

Auto-detects CRISPR/prophage loci from GTF annotations, runs differential
SAE feature analysis at each locus, and aggregates results to find features
consistently enriched across multiple CRISPR/prophage sites.

Usage:
    python investigations/crispr_prophage/discover_multi_locus_features.py \
        --fasta /path/to/ecoli.fna \
        --chrom NC_000913.3 \
        --gtf /path/to/genomic.gtf \
        --chrom_name ecoli_K12
"""

import os
import sys
import json
import argparse
import logging
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tools'))

from results_utils import build_run_dir, write_completed, write_source
from sae_annotation_overlay import parse_gtf_genes
from discover_region_features import (
    compute_enrichment, compute_all_features, build_background_from_gtf,
    write_enriched_tsv, plot_top_enriched, plot_volcano, plot_target_vs_background,
    KNOWN_BIO_FEATURES, N_SAE_FEATURES, WINDOW_PADDING, setup_logging,
)

# Hardcoded fallback coordinates for E. coli K-12 MG1655 (NC_000913.3)
ECOLI_K12_KNOWN_LOCI = [
    ("CP4-6",     262246,  282260),
    ("DLP12",     556698,  582543),
    ("e14",      1195432, 1211059),
    ("Rac",      1408685, 1433217),
    ("Qin",      1629855, 1651856),
    ("CP4-44",   2064327, 2078613),
    ("CPS-53",   2161314, 2175866),
    ("CPZ-55",   2556942, 2563568),
    ("CP4-57",   2747020, 2773709),
    ("KpLE2",    3449036, 3467424),
    ("CRISPR-I", 2875441, 2876516),
    ("CRISPR-II",2877618, 2878569),
]

# Patterns to match prophage/CRISPR genes in GTF
PROPHAGE_GENE_PATTERNS = [
    r'(?i)\bCP4[-_]?6\b', r'(?i)\bDLP12\b', r'(?i)\be14\b',
    r'(?i)\bRac\b', r'(?i)\bQin\b', r'(?i)\bCP4[-_]?44\b',
    r'(?i)\bCPS[-_]?53\b', r'(?i)\bCPZ[-_]?55\b', r'(?i)\bCP4[-_]?57\b',
    r'(?i)\bKpLE2\b',
    r'(?i)\bint[A-Z]\b',   # integrases (intA, intB, etc.)
]
CRISPR_GENE_PATTERNS = [
    r'(?i)\bcas[0-9]\b', r'(?i)\bcsn[0-9]\b', r'(?i)\biap\b',
    r'(?i)\bCRISPR\b', r'(?i)\bcas[A-E]\b',
]
ALL_PATTERNS = [re.compile(p) for p in PROPHAGE_GENE_PATTERNS + CRISPR_GENE_PATTERNS]


def detect_target_loci_from_gtf(
    gtf_path: str,
    chrom: str,
    cluster_distance: int = 5000,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[str, int, int]]:
    """Detect prophage/CRISPR loci from GTF gene annotations.

    Parses GTF for genes matching prophage/CRISPR name patterns, clusters
    nearby matches into contiguous regions, and returns locus coordinates.
    Falls back to hardcoded coordinates if fewer than 3 loci are detected.
    """
    log = logger or logging.getLogger(__name__)
    features = parse_gtf_genes(gtf_path, chrom)

    # Find matching genes
    matches = []
    for feat in features:
        if feat['type'] != 'gene':
            continue
        gene_name = feat.get('gene_name', '')
        if not gene_name:
            continue
        for pat in ALL_PATTERNS:
            if pat.search(gene_name):
                matches.append((gene_name, feat['start'], feat['end']))
                break

    log.info(f"Found {len(matches)} prophage/CRISPR gene matches in GTF")

    if len(matches) < 3:
        log.warning(f"Only {len(matches)} GTF matches; supplementing with hardcoded loci")
        # Use hardcoded loci, avoiding duplicates by position
        existing_ranges = set()
        for _, s, e in matches:
            existing_ranges.add((s, e))
        loci = [(name, s, e) for name, s, e in ECOLI_K12_KNOWN_LOCI
                if (s, e) not in existing_ranges]
        return loci

    # Sort by position and cluster nearby matches
    matches.sort(key=lambda x: x[1])
    clusters = []
    current_names = [matches[0][0]]
    current_start = matches[0][1]
    current_end = matches[0][2]

    for name, start, end in matches[1:]:
        if start - current_end <= cluster_distance:
            current_names.append(name)
            current_end = max(current_end, end)
        else:
            cluster_name = current_names[0] if len(current_names) <= 2 else f"{current_names[0]}..{current_names[-1]}"
            clusters.append((cluster_name, current_start, current_end))
            current_names = [name]
            current_start = start
            current_end = end

    cluster_name = current_names[0] if len(current_names) <= 2 else f"{current_names[0]}..{current_names[-1]}"
    clusters.append((cluster_name, current_start, current_end))

    log.info(f"Clustered into {len(clusters)} target loci")

    # Supplement with hardcoded if still few
    if len(clusters) < 3:
        log.warning(f"Only {len(clusters)} clustered loci; supplementing with hardcoded")
        existing_starts = {s for _, s, _ in clusters}
        for name, s, e in ECOLI_K12_KNOWN_LOCI:
            if s not in existing_starts:
                clusters.append((name, s, e))
        clusters.sort(key=lambda x: x[1])

    return clusters


def aggregate_across_loci(
    locus_results: Dict[str, List[Dict[str, Any]]],
    p_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Aggregate enrichment results across multiple loci.

    For each feature, counts how many loci show significant enrichment
    and computes mean/max effect sizes.
    """
    feature_stats = defaultdict(lambda: {
        'n_enriched': 0,
        'effect_sizes': [],
        'loci': [],
    })

    n_total = len(locus_results)

    for locus_name, enriched_list in locus_results.items():
        for feat in enriched_list:
            fid = feat['feature_id']
            if feat.get('p_value', 1.0) < p_threshold:
                feature_stats[fid]['n_enriched'] += 1
                feature_stats[fid]['effect_sizes'].append(feat.get('effect_size', 0.0))
                feature_stats[fid]['loci'].append(locus_name)

    consensus = []
    for fid, stats in feature_stats.items():
        if stats['n_enriched'] < 1:
            continue
        label = KNOWN_BIO_FEATURES.get(fid, ("", ""))[0]
        consensus.append({
            'rank': 0,
            'feature_id': fid,
            'label': label,
            'n_loci_enriched': stats['n_enriched'],
            'n_loci_total': n_total,
            'mean_effect_size': float(np.mean(stats['effect_sizes'])),
            'max_effect_size': float(np.max(stats['effect_sizes'])),
            'loci': ','.join(stats['loci']),
        })

    consensus.sort(key=lambda x: (-x['n_loci_enriched'], -x['mean_effect_size']))
    for i, row in enumerate(consensus):
        row['rank'] = i + 1

    return consensus


def plot_consensus_heatmap(
    consensus: List[Dict[str, Any]],
    locus_results: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    n_top: int = 30,
    p_threshold: float = 0.05,
):
    """Plot heatmap of top consensus features across loci."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    top_features = consensus[:n_top]
    if not top_features:
        return

    locus_names = sorted(locus_results.keys(), key=lambda x: min(
        (f['feature_id'] for f in locus_results[x][:1]), default=0))
    feature_ids = [f['feature_id'] for f in top_features]

    # Build effect-size lookup per locus
    locus_effect = {}
    for lname, enriched_list in locus_results.items():
        locus_effect[lname] = {}
        for feat in enriched_list:
            if feat.get('p_value', 1.0) < p_threshold:
                locus_effect[lname][feat['feature_id']] = feat.get('effect_size', 0.0)

    # Build matrix
    matrix = np.zeros((len(feature_ids), len(locus_names)))
    for i, fid in enumerate(feature_ids):
        for j, lname in enumerate(locus_names):
            matrix[i, j] = locus_effect.get(lname, {}).get(fid, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(locus_names) * 0.8), max(6, len(feature_ids) * 0.35)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Effect size')

    # Labels
    y_labels = []
    for f in top_features:
        label = f"f/{f['feature_id']}"
        if f['label']:
            label += f" ({f['label']})"
        y_labels.append(label)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xticks(range(len(locus_names)))
    ax.set_xticklabels(locus_names, rotation=45, ha='right', fontsize=8)
    ax.set_title('Consensus enriched features across CRISPR/prophage loci')
    ax.set_xlabel('Locus')
    ax.set_ylabel('SAE Feature')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_targets_tsv(path: str) -> List[Tuple[str, int, int]]:
    """Load target loci from a TSV file (name, start, end)."""
    loci = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                loci.append((parts[0], int(parts[1]), int(parts[2])))
    return loci


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-locus contrastive SAE feature discovery for CRISPR/prophage regions")
    parser.add_argument('--fasta', required=True, help='Path to genome FASTA')
    parser.add_argument('--chrom', required=True, help='Chromosome/accession ID')
    parser.add_argument('--gtf', required=True, help='Path to GTF annotation file')
    parser.add_argument('--output_dir', default='results', help='Base output directory')
    parser.add_argument('--chrom_name', default='ecoli_K12', help='Human-readable chromosome name')
    parser.add_argument('--targets_tsv', default=None,
                        help='Optional TSV with target loci (name\\tstart\\tend). Overrides GTF detection.')
    parser.add_argument('--bg_flank', type=int, default=5000,
                        help='Flank size for background region construction (default: 5000)')
    parser.add_argument('--log_level', default='INFO', help='Logging level')
    return parser.parse_args()


def load_fasta_sequence(fasta_path: str, chrom: str) -> str:
    """Load a single chromosome sequence from a FASTA file."""
    seq_parts = []
    reading = False
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                if reading:
                    break
                if chrom in line.split()[0]:
                    reading = True
                continue
            if reading:
                seq_parts.append(line.strip())
    if not seq_parts:
        raise ValueError(f"Chromosome {chrom} not found in {fasta_path}")
    return ''.join(seq_parts)


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    t0 = time.time()

    # 1. Load chromosome sequence
    logger.info(f"Loading sequence for {args.chrom} from {args.fasta}")
    full_seq = load_fasta_sequence(args.fasta, args.chrom)
    logger.info(f"Loaded {len(full_seq):,} bp")

    # 2. Detect target loci
    if args.targets_tsv:
        target_loci = load_targets_tsv(args.targets_tsv)
        logger.info(f"Loaded {len(target_loci)} target loci from {args.targets_tsv}")
    else:
        target_loci = detect_target_loci_from_gtf(args.gtf, args.chrom, logger=logger)
        logger.info(f"Detected {len(target_loci)} target loci from GTF")

    for name, start, end in target_loci:
        logger.info(f"  {name}: {start:,}-{end:,} ({end - start:,} bp)")

    # 3. Init model + SAE
    logger.info("Loading Evo2 model and SAE...")
    import torch
    from sae_utils import ObservableEvo2, load_topk_sae_from_hf, get_feature_ts

    model = ObservableEvo2()
    sae = load_topk_sae_from_hf(device=model.device, dtype=model.dtype)
    logger.info("Model and SAE loaded")

    # 4. Build output directory
    run_dir = build_run_dir(args.output_dir, args.chrom_name,
                            'sae_multi_locus_differential',
                            f"{len(target_loci)}_loci")
    data_dir = os.path.join(run_dir, 'data')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Write target loci TSV
    loci_tsv_path = os.path.join(data_dir, 'target_loci.tsv')
    with open(loci_tsv_path, 'w') as f:
        f.write("name\tstart\tend\tlength\n")
        for name, start, end in target_loci:
            f.write(f"{name}\t{start}\t{end}\t{end - start}\n")

    # 5. Process each locus
    locus_results = {}
    for locus_idx, (locus_name, target_start, target_end) in enumerate(target_loci):
        logger.info(f"\n=== Locus {locus_idx + 1}/{len(target_loci)}: {locus_name} "
                     f"({target_start:,}-{target_end:,}) ===")

        # Build background from GTF
        try:
            bg_start, bg_end, bg_intervals = build_background_from_gtf(
                args.gtf, args.chrom, target_start, target_end, flank=args.bg_flank)
        except Exception as e:
            logger.warning(f"Failed to build background for {locus_name}: {e}, skipping")
            continue

        # Define extraction window
        window_start = max(0, bg_start - WINDOW_PADDING)
        window_end = min(len(full_seq), bg_end + WINDOW_PADDING)
        window_seq = full_seq[window_start:window_end]

        logger.info(f"Window: {window_start:,}-{window_end:,} ({len(window_seq):,} bp)")
        logger.info(f"Target: {target_start:,}-{target_end:,} within window")

        # Run SAE
        logger.info("Computing SAE features...")
        feature_matrix = get_feature_ts(model, sae, window_seq)
        logger.info(f"Feature matrix: {feature_matrix.shape}")

        # Build masks relative to window
        target_mask = np.zeros(feature_matrix.shape[0], dtype=bool)
        rel_target_start = target_start - window_start
        rel_target_end = target_end - window_start
        target_mask[max(0, rel_target_start):min(feature_matrix.shape[0], rel_target_end)] = True

        bg_mask = np.zeros(feature_matrix.shape[0], dtype=bool)
        for bs, be in bg_intervals:
            rel_bs = bs - window_start
            rel_be = be - window_start
            bg_mask[max(0, rel_bs):min(feature_matrix.shape[0], rel_be)] = True
        # Ensure no overlap
        bg_mask &= ~target_mask

        logger.info(f"Target positions: {target_mask.sum():,}, Background positions: {bg_mask.sum():,}")

        if target_mask.sum() < 10 or bg_mask.sum() < 10:
            logger.warning(f"Insufficient positions for {locus_name}, skipping")
            del feature_matrix
            import torch
            torch.cuda.empty_cache()
            continue

        # Compute enrichment
        enriched = compute_enrichment(feature_matrix, target_mask, bg_mask, logger=logger)
        all_results = compute_all_features(feature_matrix, target_mask, bg_mask, logger=logger)
        locus_results[locus_name] = all_results

        # Per-locus output
        per_locus_data = os.path.join(data_dir, 'per_locus', locus_name)
        per_locus_plots = os.path.join(plots_dir, 'per_locus', locus_name)
        os.makedirs(per_locus_data, exist_ok=True)
        os.makedirs(per_locus_plots, exist_ok=True)

        write_enriched_tsv(enriched, os.path.join(per_locus_data, 'enriched_features.tsv'))

        region_def = {
            'locus_name': locus_name,
            'target_start': target_start,
            'target_end': target_end,
            'bg_start': bg_start,
            'bg_end': bg_end,
            'window_start': window_start,
            'window_end': window_end,
            'n_target_positions': int(target_mask.sum()),
            'n_bg_positions': int(bg_mask.sum()),
            'n_enriched_features': len(enriched),
        }
        with open(os.path.join(per_locus_data, 'region_definitions.json'), 'w') as f:
            json.dump(region_def, f, indent=2)

        # Per-locus plots
        try:
            plot_top_enriched(enriched, os.path.join(per_locus_plots, 'top_enriched_features.png'),
                              chrom=args.chrom, target_start=target_start, target_end=target_end)
            plot_volcano(all_results, os.path.join(per_locus_plots, 'enrichment_volcano.png'),
                         chrom=args.chrom, target_start=target_start, target_end=target_end)
        except Exception as e:
            logger.warning(f"Plot generation failed for {locus_name}: {e}")

        logger.info(f"Locus {locus_name}: {len(enriched)} enriched features")

        # Free GPU memory
        del feature_matrix
        import torch
        torch.cuda.empty_cache()

    # 6. Aggregate across loci
    logger.info(f"\n=== Aggregating results across {len(locus_results)} loci ===")
    consensus = aggregate_across_loci(locus_results)
    logger.info(f"Found {len(consensus)} features enriched in at least one locus")

    # Write consensus TSV
    consensus_path = os.path.join(data_dir, 'consensus_features.tsv')
    with open(consensus_path, 'w') as f:
        if consensus:
            headers = list(consensus[0].keys())
            f.write('\t'.join(headers) + '\n')
            for row in consensus:
                f.write('\t'.join(str(row[h]) for h in headers) + '\n')
    logger.info(f"Wrote consensus features to {consensus_path}")

    # Consensus heatmap
    try:
        plot_consensus_heatmap(consensus, locus_results,
                               os.path.join(plots_dir, 'consensus_heatmap.png'))
        logger.info("Consensus heatmap saved")
    except Exception as e:
        logger.warning(f"Heatmap generation failed: {e}")

    # 7. Provenance and completion
    wall_time = time.time() - t0
    write_source(run_dir, fasta=args.fasta, gtf=args.gtf)
    write_completed(run_dir, os.path.basename(__file__), wall_time)

    logger.info(f"\nDone in {wall_time:.1f}s")
    logger.info(f"Output: {run_dir}")

    # Print top consensus features
    if consensus:
        logger.info(f"\nTop consensus features (enriched across multiple loci):")
        for feat in consensus[:10]:
            label = f" ({feat['label']})" if feat['label'] else ""
            logger.info(f"  f/{feat['feature_id']}{label}: "
                        f"{feat['n_loci_enriched']}/{feat['n_loci_total']} loci, "
                        f"mean effect={feat['mean_effect_size']:.3f}")


if __name__ == '__main__':
    main()
