#!/usr/bin/env python3
"""
replot.py — Regenerate plots from completed run directories (no GPU needed)

Auto-detects the pipeline stage from the run directory structure and
regenerates all plots from saved data files. Use this to tweak
visualizations without rerunning expensive GPU computations.

Supported stages:
  - sae/              SAE feature analysis (signature summary, heatmaps)
  - sae_differential/ Differential feature discovery (volcano, bar charts)
  - sae_feature_scan/ Genome-wide feature scanning (genome profile, zooms)
  - sae_multi_locus_differential/  Multi-locus consensus (heatmap, per-locus)
  - latent_analysis/  SAE region clustering (t-SNE, similarity heatmaps)

Usage:
    # Replot from a completed SAE run
    python tools/replot.py --run_dir results/chr22/sae/20260305_110000_.../

    # Replot into a different output directory
    python tools/replot.py --run_dir results/ecoli_K12/sae_differential/.../ \
        --output_dir my_new_plots/

    # Replot latent analysis (auto-detected from subdirectory)
    python tools/replot.py --run_dir results/chr22/sae/.../latent_analysis/
"""

import os
import sys
import csv
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

import numpy as np

# Add project root and tools to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("replot")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def load_tsv(path: str) -> List[Dict[str, str]]:
    """Load a TSV file as list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(dict(row))
    return rows


def detect_stage(run_dir: str) -> str:
    """Auto-detect pipeline stage from run directory contents."""
    data_dir = os.path.join(run_dir, 'data')

    # Check for latent_analysis (can be a subdirectory of sae/)
    if os.path.basename(run_dir) == 'latent_analysis':
        return 'latent_analysis'
    if os.path.isdir(os.path.join(run_dir, 'latent_analysis', 'data')):
        if os.path.exists(os.path.join(run_dir, 'latent_analysis', 'data', 'maxpooled_vectors.npy')):
            return 'sae_with_latent'

    if not os.path.isdir(data_dir):
        raise ValueError(f"No data/ directory found in {run_dir}")

    # Multi-locus differential
    if os.path.exists(os.path.join(data_dir, 'consensus_features.tsv')):
        return 'sae_multi_locus_differential'

    # Feature scan
    if os.path.exists(os.path.join(data_dir, 'summary.json')):
        npz_files = [f for f in os.listdir(data_dir) if f == 'feature_activations.npz']
        if npz_files:
            return 'sae_feature_scan'

    # Differential (single locus)
    if os.path.exists(os.path.join(data_dir, 'enriched_features.tsv')):
        return 'sae_differential'

    # SAE analysis
    if os.path.exists(os.path.join(data_dir, 'signature_features.tsv')):
        return 'sae'

    # Standalone latent analysis
    if os.path.exists(os.path.join(data_dir, 'maxpooled_vectors.npy')):
        return 'latent_analysis'

    raise ValueError(f"Cannot detect stage from files in {data_dir}")


# ---------------------------------------------------------------------------
# Stage: sae (run_sae_on_chromosome_drops.py)
# ---------------------------------------------------------------------------

def replot_sae(run_dir: str, output_dir: str, logger: logging.Logger,
               normalize: str = 'zscore', n_plot_regions: int = 50,
               n_plot_features: int = 8, entropy_path: str = None,
               gtf_path: str = None, chrom: str = None):
    """Replot SAE feature analysis from saved data.

    Args:
        normalize: Normalization for per-region feature plots.
                   'zscore'  — z-score per feature across all regions (default)
                   'minmax'  — scale each feature to [0, 1] across all regions
                   'both'    — generate both zscore and minmax subdirectories
                   'raw'     — no normalization (original behavior)
        n_plot_regions: Number of top-confidence regions to plot (0 = all).
        n_plot_features: Number of features to show per region plot.
        entropy_path: Path to entropy .npz file for entropy panel overlay.
        gtf_path: Path to GTF file for gene track panel.
        chrom: Chromosome name (needed for GTF loading).
    """
    from run_sae_on_chromosome_drops import (
        plot_signature_summary, plot_feature_heatmap,
        plot_feature_prevalence_by_method, plot_region_figure4g,
    )

    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Load signature features
    sig_path = os.path.join(data_dir, 'signature_features.tsv')
    sig_rows = load_tsv(sig_path)
    signatures = []
    for row in sig_rows:
        signatures.append({
            'feature_id': int(row.get('feature_id', row.get('feature', 0))),
            'prevalence': float(row.get('prevalence', 0)),
            'mean_activation': float(row.get('mean_activation', 0)),
            'drop_count': int(row.get('drop_count', 0)),
            'rise_count': int(row.get('rise_count', 0)),
            'zscore_count': int(row.get('zscore_count', 0)),
            'mad_count': int(row.get('mad_count', 0)),
        })
    logger.info(f"Loaded {len(signatures)} signature features")

    # Load SAE results (region metadata + top features)
    results_path = os.path.join(data_dir, 'sae_results.tsv')
    sae_rows = load_tsv(results_path)
    from collections import defaultdict
    region_meta = {}  # ridx -> row
    region_features_raw = defaultdict(list)
    for row in sae_rows:
        ridx = int(row.get('region_idx', row.get('region', 0)))
        region_meta[ridx] = row
        top_str = row.get('top_features', '')
        if top_str:
            region_features_raw[ridx] = [int(f) for f in top_str.split(',') if f.strip()]

    results = []
    for ridx in sorted(region_meta.keys()):
        row = region_meta[ridx]
        results.append({
            'region': {
                'region_idx': ridx,
                'genomic_start': int(row.get('genomic_start', 0)),
                'genomic_end': int(row.get('genomic_end', 0)),
                'method': row.get('method', 'unknown'),
                'start_confidence': float(row.get('confidence', 0)),
            },
            'top_feature_idx': region_features_raw.get(ridx, []),
            'drop_features': [],
        })
    logger.info(f"Loaded {len(results)} regions from SAE results")

    # Summary plots (no feature matrices needed)
    plot_signature_summary(signatures, os.path.join(output_dir, 'signature_summary.png'))
    logger.info("Plotted signature_summary.png")
    plot_feature_heatmap(results, signatures, os.path.join(output_dir, 'feature_heatmap.png'))
    logger.info("Plotted feature_heatmap.png")
    plot_feature_prevalence_by_method(signatures, os.path.join(output_dir, 'feature_prevalence_by_method.png'))
    logger.info("Plotted feature_prevalence_by_method.png")

    # ── Per-region normalized feature plots ──────────────────────────────────
    matrices_path = os.path.join(data_dir, 'feature_matrices.npz')
    if not os.path.exists(matrices_path):
        logger.warning(f"feature_matrices.npz not found at {matrices_path}, skipping per-region plots")
        return

    logger.info(f"Loading feature matrices from {matrices_path}")
    mat_data = np.load(matrices_path)
    n_regions = len(results)
    feature_matrices = []
    for i in range(n_regions):
        key = f'region_{i}'
        if key in mat_data:
            feature_matrices.append(mat_data[key])
        else:
            feature_matrices.append(None)

    # Compute normalization stats from ALL regions (chromosome-level proxy)
    valid_mats = [m for m in feature_matrices if m is not None]
    if not valid_mats:
        logger.warning("No valid feature matrices found")
        return

    all_acts = np.concatenate(valid_mats, axis=0)  # (total_positions, 32768)
    feat_mean = all_acts.mean(axis=0)
    feat_std = np.maximum(all_acts.std(axis=0), 1e-6)
    feat_min = all_acts.min(axis=0)
    feat_max = all_acts.max(axis=0)
    feat_span = np.maximum(feat_max - feat_min, 1e-6)
    del all_acts
    logger.info(f"Computed normalization stats from {len(valid_mats)} regions")

    # Save stats
    np.savez_compressed(os.path.join(data_dir, 'feature_norm_stats.npz'),
                        mean=feat_mean, std=feat_std,
                        min=feat_min, max=feat_max)
    logger.info("Saved feature_norm_stats.npz")

    # Optionally load entropy
    entropy = None
    if entropy_path and os.path.exists(entropy_path):
        entropy = np.load(entropy_path)['entropy']
        logger.info(f"Loaded entropy from {entropy_path}")

    # Optionally load GTF annotations
    gtf_features_all = None
    if gtf_path and chrom:
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from tools.analyze_scoring_results import load_annotation_features
            from run_sae_on_chromosome_drops import CHROM_MAP
            chrom_id = CHROM_MAP.get(chrom, chrom)
            all_starts = [r['region']['genomic_start'] for r in results]
            all_ends = [r['region']['genomic_end'] for r in results]
            gtf_features_all = load_annotation_features(
                gtf_path, chrom_id,
                max(0, min(all_starts) - 10000),
                max(all_ends) + 10000,
            )
            logger.info(f"Loaded {len(gtf_features_all)} GTF features")
        except Exception as e:
            logger.warning(f"Could not load GTF: {e}")

    # Determine which normalization modes to run
    modes = ['zscore', 'minmax'] if normalize == 'both' else [normalize]

    n_to_plot = n_plot_regions if n_plot_regions > 0 else n_regions
    n_to_plot = min(n_to_plot, n_regions)

    for mode in modes:
        if mode == 'raw':
            feature_stats = None
            subdir = os.path.join(output_dir, 'region_plots_raw')
        elif mode == 'zscore':
            feature_stats = {'mean': feat_mean, 'std': feat_std}
            subdir = os.path.join(output_dir, 'region_plots_zscore')
        elif mode == 'minmax':
            # Reuse the zscore dict interface but pass min/span as mean/std
            feature_stats = {'mean': feat_min, 'std': feat_span}
            subdir = os.path.join(output_dir, 'region_plots_minmax')
        else:
            continue

        os.makedirs(subdir, exist_ok=True)
        logger.info(f"Generating {n_to_plot} region plots ({mode} normalization) → {subdir}")

        for i in range(n_to_plot):
            fm = feature_matrices[i]
            if fm is None:
                continue
            reg = results[i]['region']
            result = {
                'region': {
                    **reg,
                    'padded_start': reg['genomic_start'],
                    'padded_end': reg['genomic_start'] + fm.shape[0],
                    'drop_local_pos': fm.shape[0] // 4,
                    'rise_local_pos': 3 * fm.shape[0] // 4,
                },
                'feature_ts': fm,
                'top_feature_idx': results[i]['top_feature_idx'],
                'drop_features': results[i]['drop_features'],
                'rise_features': [],
            }

            # Get GTF features for this region
            region_gtf = None
            if gtf_features_all:
                region_gtf = [f for f in gtf_features_all
                              if f['end_exclusive'] > reg['genomic_start']
                              and f['start'] < reg['genomic_end']]

            out_path = os.path.join(subdir, f'region_{i+1:05d}_features.png')
            try:
                plot_region_figure4g(
                    result, i, out_path, annotations=None,
                    entropy=entropy, gtf_features=region_gtf,
                    n_plot_features=n_plot_features,
                    chrom=chrom or '',
                    feature_stats=feature_stats,
                )
            except Exception as e:
                logger.warning(f"  Region {i+1}: plot failed — {e}")

            if (i + 1) % 20 == 0:
                logger.info(f"  {i+1}/{n_to_plot} regions plotted")

        logger.info(f"Done: {n_to_plot} plots in {subdir}")


# ---------------------------------------------------------------------------
# Stage: sae_differential (discover_region_features.py)
# ---------------------------------------------------------------------------

def replot_sae_differential(run_dir: str, output_dir: str, logger: logging.Logger):
    """Replot differential feature discovery from saved data."""
    from discover_region_features import plot_top_enriched, plot_volcano

    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Load region definitions for titles
    region_def_path = os.path.join(data_dir, 'region_definitions.json')
    chrom, target_start, target_end = "", 0, 0
    if os.path.exists(region_def_path):
        with open(region_def_path) as f:
            rdef = json.load(f)
        chrom = rdef.get('chrom', '')
        target_start = rdef.get('target_start', 0)
        target_end = rdef.get('target_end', 0)

    # Load enriched features
    enriched_path = os.path.join(data_dir, 'enriched_features.tsv')
    enriched = _load_enrichment_tsv(enriched_path)
    logger.info(f"Loaded {len(enriched)} enriched features")

    # Load all features (for volcano)
    all_path = os.path.join(data_dir, 'all_features.tsv')
    all_results = []
    if os.path.exists(all_path):
        all_results = _load_enrichment_tsv(all_path)
        logger.info(f"Loaded {len(all_results)} total features for volcano")

    # Plot
    plot_top_enriched(enriched, os.path.join(output_dir, 'top_enriched_features.png'),
                      chrom=chrom, target_start=target_start, target_end=target_end)
    logger.info("Plotted top_enriched_features.png")

    if all_results:
        plot_volcano(all_results, os.path.join(output_dir, 'enrichment_volcano.png'),
                     chrom=chrom, target_start=target_start, target_end=target_end)
        logger.info("Plotted enrichment_volcano.png")


def _load_enrichment_tsv(path: str) -> List[Dict[str, Any]]:
    """Load enrichment TSV with proper numeric types."""
    rows = load_tsv(path)
    results = []
    for row in rows:
        results.append({
            'feature_id': int(row.get('feature_id', 0)),
            'label': row.get('label', ''),
            'effect_size': float(row.get('effect_size', 0)),
            'p_value': float(row.get('p_value', 1.0)),
            'log10_p': float(row.get('log10_p', row.get('-log10_p', 0))),
            'target_mean': float(row.get('target_mean', 0)),
            'bg_mean': float(row.get('bg_mean', 0)),
            'u_statistic': float(row.get('u_statistic', 0)),
        })
    return results


# ---------------------------------------------------------------------------
# Stage: sae_feature_scan (scan_feature_genome.py)
# ---------------------------------------------------------------------------

def replot_sae_feature_scan(run_dir: str, output_dir: str, logger: logging.Logger,
                            gtf_path: Optional[str] = None, chrom: Optional[str] = None):
    """Replot genome-wide feature scan from saved data."""
    from scan_feature_genome import plot_genome_wide, plot_top_regions

    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Load summary for metadata
    summary_path = os.path.join(data_dir, 'summary.json')
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    chrom = chrom or summary.get('chrom', '')

    # Load gene annotations if GTF provided
    genes = None
    if gtf_path and chrom:
        from sae_annotation_overlay import parse_gtf_genes
        genes = parse_gtf_genes(gtf_path, chrom)
        logger.info(f"Loaded {len(genes)} gene features from GTF")

    # Load feature activations
    npz_path = os.path.join(data_dir, 'feature_activations.npz')
    npz = np.load(npz_path)
    logger.info(f"Loaded feature activations: {list(npz.keys())}")

    for key in npz.keys():
        # key format: "f0", "f1", etc. — index into feature list
        activation = npz[key]

        # Find corresponding feature ID from region TSV files
        # Try to find the feature_id from the activation_regions file
        fid = None
        region_files = [f for f in os.listdir(data_dir) if f.startswith('activation_regions_f') and f.endswith('.tsv')]
        # Match by index
        idx = int(key.replace('f', ''))
        if idx < len(region_files):
            # Extract feature_id from filename
            for rf in sorted(region_files):
                fid_str = rf.replace('activation_regions_f', '').replace('.tsv', '')
                try:
                    candidate_fid = int(fid_str)
                    if region_files.index(rf) == idx:
                        fid = candidate_fid
                        break
                except ValueError:
                    continue

        if fid is None:
            # Fallback: use the summary to find feature IDs
            feature_ids = summary.get('feature_ids', [])
            if idx < len(feature_ids):
                fid = feature_ids[idx]
            else:
                fid = idx
                logger.warning(f"Could not determine feature ID for {key}, using index {idx}")

        # Load regions for this feature
        regions = []
        region_tsv = os.path.join(data_dir, f'activation_regions_f{fid}.tsv')
        if os.path.exists(region_tsv):
            region_rows = load_tsv(region_tsv)
            for row in region_rows:
                regions.append({
                    'start': int(row.get('start', 0)),
                    'end': int(row.get('end', 0)),
                    'length': int(row.get('length', 0)),
                    'mean_activation': float(row.get('mean_activation', 0)),
                    'max_activation': float(row.get('max_activation', 0)),
                    'total_activation': float(row.get('total_activation', 0)),
                    'overlapping_genes': row.get('overlapping_genes', ''),
                })

        logger.info(f"Replotting feature f/{fid}: {len(activation):,} bp, {len(regions)} regions")

        plot_genome_wide(activation, fid, os.path.join(output_dir, f'genome_wide_f{fid}.png'),
                         chrom, genes=genes)
        logger.info(f"Plotted genome_wide_f{fid}.png")

        if regions:
            plot_top_regions(activation, regions, fid,
                             os.path.join(output_dir, f'top_regions_f{fid}.png'),
                             chrom, genes=genes)
            logger.info(f"Plotted top_regions_f{fid}.png")

    npz.close()


# ---------------------------------------------------------------------------
# Stage: latent_analysis (analyze_sae_regions.py)
# ---------------------------------------------------------------------------

def replot_latent_analysis(run_dir: str, output_dir: str, logger: logging.Logger,
                           annotation_tsv: Optional[str] = None,
                           use_normalized: bool = False):
    """Replot SAE region clustering from saved data."""
    from analyze_sae_regions import (
        plot_cosine_similarity_heatmap, plot_embedding,
        plot_cluster_composition, plot_cluster_feature_profiles,
        plot_genomic_position_by_cluster, plot_cluster_feature_heatmap,
        plot_region_size_distribution,
    )

    # Determine data directory
    if os.path.basename(run_dir) == 'latent_analysis':
        data_dir = os.path.join(run_dir, 'data')
    else:
        data_dir = os.path.join(run_dir, 'latent_analysis', 'data')

    # When using normalized vectors, save plots to a separate subdirectory
    if use_normalized:
        output_dir = os.path.join(output_dir, 'normalized')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if use_normalized:
        norm_path = os.path.join(data_dir, 'normalized_maxpooled_vectors.npy')
        if not os.path.isfile(norm_path):
            raise FileNotFoundError(
                f"normalized_maxpooled_vectors.npy not found in {data_dir}. "
                f"Run normalize_sae_features.py first."
            )
        pooled_vectors = np.load(norm_path)
        logger.info(f"Loaded normalized vectors: {pooled_vectors.shape}")
    else:
        pooled_vectors = np.load(os.path.join(data_dir, 'maxpooled_vectors.npy'))
    similarity_matrix = np.load(os.path.join(data_dir, 'cosine_similarity.npy'))
    logger.info(f"Loaded {pooled_vectors.shape[0]} regions, {pooled_vectors.shape[1]} features")

    # Load cluster assignments TSV for metadata + embeddings
    cluster_tsv_path = os.path.join(data_dir, 'cluster_assignments.tsv')
    cluster_rows = load_tsv(cluster_tsv_path)

    region_metadata = []
    clusters = []
    tsne_coords = []
    umap_coords = []
    for row in cluster_rows:
        region_metadata.append({
            'genomic_start': int(row.get('genomic_start', 0)),
            'genomic_end': int(row.get('genomic_end', 0)),
            'region_length': int(row.get('region_length', 0)),
            'method': row.get('method', ''),
            'confidence': float(row.get('confidence', row.get('start_confidence', 0))),
        })
        clusters.append(int(row.get('cluster', row.get('cluster_id', 0))))
        if 'tsne_x' in row and 'tsne_y' in row:
            tsne_coords.append([float(row['tsne_x']), float(row['tsne_y'])])
        if 'umap_x' in row and 'umap_y' in row:
            umap_coords.append([float(row['umap_x']), float(row['umap_y'])])

    clusters = np.array(clusters)
    logger.info(f"Loaded {len(region_metadata)} regions, {len(set(clusters))} clusters")

    # Genomic order for similarity heatmap
    genomic_order = np.argsort([m['genomic_start'] for m in region_metadata])

    # Cluster order for similarity heatmap
    cluster_order = np.argsort(clusters)

    # Cosine similarity heatmaps
    plot_cosine_similarity_heatmap(similarity_matrix, region_metadata,
                                  os.path.join(output_dir, 'cosine_similarity_heatmap.png'),
                                  order=genomic_order, title_suffix=" (genomic order)",
                                  logger=logger)
    logger.info("Plotted cosine_similarity_heatmap.png")

    plot_cosine_similarity_heatmap(similarity_matrix, region_metadata,
                                  os.path.join(output_dir, 'cosine_similarity_clustered.png'),
                                  order=cluster_order, title_suffix=" (clustered)",
                                  logger=logger)
    logger.info("Plotted cosine_similarity_clustered.png")

    # Embedding plots
    if tsne_coords:
        tsne_arr = np.array(tsne_coords)
        plot_embedding(tsne_arr, region_metadata, clusters,
                       os.path.join(output_dir, 'tsne_4panel.png'),
                       embedding_name="t-SNE", logger=logger)
        logger.info("Plotted tsne_4panel.png")

    if umap_coords:
        umap_arr = np.array(umap_coords)
        plot_embedding(umap_arr, region_metadata, clusters,
                       os.path.join(output_dir, 'umap_4panel.png'),
                       embedding_name="UMAP", logger=logger)
        logger.info("Plotted umap_4panel.png")

    # Cluster composition
    plot_cluster_composition(clusters, region_metadata,
                             os.path.join(output_dir, 'cluster_composition.png'),
                             logger=logger)
    logger.info("Plotted cluster_composition.png")

    # Cluster feature profiles
    plot_cluster_feature_profiles(clusters, pooled_vectors,
                                 os.path.join(output_dir, 'cluster_feature_profiles.png'),
                                 logger=logger)
    logger.info("Plotted cluster_feature_profiles.png")

    # Genomic position by cluster
    plot_genomic_position_by_cluster(clusters, region_metadata,
                                    os.path.join(output_dir, 'genomic_position_by_cluster.png'),
                                    logger=logger)
    logger.info("Plotted genomic_position_by_cluster.png")

    # Cluster feature heatmap
    plot_cluster_feature_heatmap(clusters, pooled_vectors,
                                os.path.join(output_dir, 'cluster_feature_heatmap.png'),
                                logger=logger)
    logger.info("Plotted cluster_feature_heatmap.png")

    # Region size distribution
    plot_region_size_distribution(clusters, region_metadata,
                                 os.path.join(output_dir, 'region_size_distribution.png'),
                                 logger=logger)
    logger.info("Plotted region_size_distribution.png")


# ---------------------------------------------------------------------------
# Stage: sae_multi_locus_differential
# ---------------------------------------------------------------------------

def replot_multi_locus(run_dir: str, output_dir: str, logger: logging.Logger):
    """Replot multi-locus consensus from saved data."""
    from discover_region_features import plot_top_enriched, plot_volcano

    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Load consensus features
    consensus_path = os.path.join(data_dir, 'consensus_features.tsv')
    consensus = load_tsv(consensus_path)
    for row in consensus:
        row['feature_id'] = int(row.get('feature_id', 0))
        row['n_loci_enriched'] = int(row.get('n_loci_enriched', 0))
        row['n_loci_total'] = int(row.get('n_loci_total', 0))
        row['mean_effect_size'] = float(row.get('mean_effect_size', 0))
        row['max_effect_size'] = float(row.get('max_effect_size', 0))
        row['label'] = row.get('label', '')
    logger.info(f"Loaded {len(consensus)} consensus features")

    # Load per-locus enrichment results
    per_locus_dir = os.path.join(data_dir, 'per_locus')
    locus_results = {}
    if os.path.isdir(per_locus_dir):
        for locus_name in sorted(os.listdir(per_locus_dir)):
            locus_data_dir = os.path.join(per_locus_dir, locus_name)
            enriched_path = os.path.join(locus_data_dir, 'enriched_features.tsv')
            if os.path.exists(enriched_path):
                locus_results[locus_name] = _load_enrichment_tsv(enriched_path)

                # Per-locus region definitions for titles
                rdef_path = os.path.join(locus_data_dir, 'region_definitions.json')
                chrom, ts, te = "", 0, 0
                if os.path.exists(rdef_path):
                    with open(rdef_path) as f:
                        rdef = json.load(f)
                    chrom = rdef.get('chrom', '')
                    ts = rdef.get('target_start', 0)
                    te = rdef.get('target_end', 0)

                # Replot per-locus
                per_locus_plots = os.path.join(output_dir, 'per_locus', locus_name)
                os.makedirs(per_locus_plots, exist_ok=True)
                enriched = locus_results[locus_name]
                plot_top_enriched(enriched, os.path.join(per_locus_plots, 'top_enriched_features.png'),
                                  chrom=chrom, target_start=ts, target_end=te)

                # Volcano if we have all features (enriched list has p_value)
                if enriched:
                    plot_volcano(enriched, os.path.join(per_locus_plots, 'enrichment_volcano.png'),
                                chrom=chrom, target_start=ts, target_end=te)

        logger.info(f"Replotted {len(locus_results)} per-locus results")

    # Consensus heatmap
    if consensus and locus_results:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'investigations', 'crispr_prophage'))
        from discover_multi_locus_features import plot_consensus_heatmap
        plot_consensus_heatmap(consensus, locus_results,
                               os.path.join(output_dir, 'consensus_heatmap.png'))
        logger.info("Plotted consensus_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate plots from completed run directories (no GPU needed)")
    parser.add_argument('--run_dir', required=True,
                        help='Path to completed run directory')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for plots (default: {run_dir}/plots/)')
    parser.add_argument('--stage', default=None,
                        choices=['sae', 'sae_differential', 'sae_feature_scan',
                                 'latent_analysis', 'sae_multi_locus_differential'],
                        help='Override auto-detected stage')
    parser.add_argument('--gtf', default=None,
                        help='GTF file for gene annotations (feature_scan stage)')
    parser.add_argument('--chrom', default=None,
                        help='Chromosome name (feature_scan stage)')
    parser.add_argument('--annotation_tsv', default=None,
                        help='Annotation TSV for colored embeddings (latent_analysis stage)')
    parser.add_argument('--use_normalized', action='store_true',
                        help='Load normalized_maxpooled_vectors.npy instead of raw vectors '
                             'for feature activation plots (latent_analysis stage). '
                             'Plots are saved to plots/normalized/.')
    # SAE per-region plot options
    parser.add_argument('--normalize', default='both',
                        choices=['zscore', 'minmax', 'both', 'raw'],
                        help='Normalization for per-region feature plots (sae stage). '
                             'zscore: z-score per feature; minmax: scale to [0,1]; '
                             'both: generate both; raw: no normalization. Default: both')
    parser.add_argument('--n_plot_regions', type=int, default=50,
                        help='Number of top-confidence regions to plot (0 = all). Default: 50')
    parser.add_argument('--n_plot_features', type=int, default=8,
                        help='Number of features to show per region plot. Default: 8')
    parser.add_argument('--entropy', default=None,
                        help='Path to entropy .npz file for entropy panel overlay (sae stage)')
    parser.add_argument('--log_level', default='INFO', help='Logging level')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Auto-detect stage
    stage = args.stage
    if stage is None:
        stage = detect_stage(run_dir)
    logger.info(f"Detected stage: {stage}")

    # Determine output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    elif stage == 'latent_analysis':
        output_dir = os.path.join(run_dir, 'plots') if os.path.basename(run_dir) == 'latent_analysis' \
            else os.path.join(run_dir, 'latent_analysis', 'plots')
    else:
        output_dir = os.path.join(run_dir, 'plots')

    logger.info(f"Output directory: {output_dir}")

    # Dispatch
    if stage == 'sae':
        replot_sae(run_dir, output_dir, logger,
                   normalize=args.normalize,
                   n_plot_regions=args.n_plot_regions,
                   n_plot_features=args.n_plot_features,
                   entropy_path=args.entropy,
                   gtf_path=args.gtf,
                   chrom=args.chrom)
    elif stage == 'sae_with_latent':
        replot_sae(run_dir, output_dir, logger,
                   normalize=args.normalize,
                   n_plot_regions=args.n_plot_regions,
                   n_plot_features=args.n_plot_features,
                   entropy_path=args.entropy,
                   gtf_path=args.gtf,
                   chrom=args.chrom)
        replot_latent_analysis(run_dir, output_dir, logger,
                               use_normalized=args.use_normalized)
    elif stage == 'sae_differential':
        replot_sae_differential(run_dir, output_dir, logger)
    elif stage == 'sae_feature_scan':
        replot_sae_feature_scan(run_dir, output_dir, logger, gtf_path=args.gtf, chrom=args.chrom)
    elif stage == 'latent_analysis':
        replot_latent_analysis(run_dir, output_dir, logger, annotation_tsv=args.annotation_tsv,
                               use_normalized=args.use_normalized)
    elif stage == 'sae_multi_locus_differential':
        replot_multi_locus(run_dir, output_dir, logger)
    else:
        logger.error(f"Unknown stage: {stage}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == '__main__':
    main()
