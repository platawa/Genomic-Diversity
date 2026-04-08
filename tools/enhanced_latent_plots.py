#!/usr/bin/env python3
"""
enhanced_latent_plots.py

Unified script for generating enhanced SAE latent analysis plots.
Adds new coloring modes and statistics on top of pre-computed tSNE/UMAP embeddings:

  1. distance_to_gene — Intergenic regions colored by distance to nearest upstream/downstream gene
  2. entropy_color   — All regions colored by average/minimum entropy drop
  3. length_stats    — Region length distribution + tSNE/UMAP colored by length
  4. top_features    — Extract and validate top-5 non-zero features per region
  5. firing_counts   — tSNE/UMAP colored by number of features fired per region
  6. firing_thresholds — Plots at 1%, 5%, 10% neuron firing thresholds

Works for: human chromosomes (per-chrom or genome-wide), E. coli, Bacillus.
Reads pre-computed embeddings from latent_analysis/data/ directories.
No GPU needed — runs on CPU with matplotlib.

Usage:
    # Single chromosome
    python tools/enhanced_latent_plots.py \\
        --scope chromosome --chrom chr22 --organism human \\
        --gtf /path/to/genomic.gtf \\
        --results_dir results/ \\
        --plots distance_to_gene,entropy_color,length_stats,top_features,firing_counts,firing_thresholds

    # E. coli
    python tools/enhanced_latent_plots.py \\
        --scope organism --organism ecoli \\
        --gtf /path/to/ecoli_genomic.gtf \\
        --results_dir results/ \\
        --plots distance_to_gene,entropy_color,length_stats,top_features,firing_counts,firing_thresholds

    # Genome-wide (all human chromosomes)
    python tools/enhanced_latent_plots.py \\
        --scope genome_wide --organism human \\
        --gtf /path/to/genomic.gtf \\
        --results_dir results/ \\
        --plots distance_to_gene,entropy_color,length_stats,top_features,firing_counts,firing_thresholds
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_utils import find_latest_completed
from tools.plot_tsne_by_annotation import (
    CHROM_MAP,
    classify_region,
    load_gtf_features,
)
from tools.latent_plot_utils import (
    compute_distance_to_nearest_gene,
    compute_feature_firing_stats,
    compute_firing_threshold_counts,
    compute_length_stats,
    compute_region_lengths,
    load_entropy_drop_stats,
    plot_continuous_scatter,
    plot_length_distribution,
    save_firing_stats_tsv,
    save_top_features_tsv,
)

logger = logging.getLogger(__name__)

# Organism → chromosome mapping
ORGANISM_CHROMS = {
    "ecoli": ["NC_000913.3"],
    "bacillus": ["NC_000964.3"],
    "human": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
}


def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_embeddings_and_metadata(latent_dir):
    """Load pre-computed embeddings and cluster assignments.

    Returns
    -------
    data : dict with keys:
        'cluster_assignments' : pd.DataFrame
        'embedding_tsne' : np.ndarray or None
        'embedding_umap' : np.ndarray or None
        'maxpooled_vectors' : np.ndarray or None
    """
    data_dir = os.path.join(latent_dir, "data")

    # cluster_assignments.tsv has coordinates + metadata
    ca_path = os.path.join(data_dir, "cluster_assignments.tsv")
    if not os.path.isfile(ca_path):
        logger.error(f"cluster_assignments.tsv not found: {ca_path}")
        return None

    ca = pd.read_csv(ca_path, sep="\t", comment="#")
    logger.info(f"Loaded {len(ca)} regions from {ca_path}")

    # Validate required columns
    required_cols = {"genomic_start", "genomic_end"}
    missing = required_cols - set(ca.columns)
    if missing:
        logger.error(f"cluster_assignments.tsv missing columns: {missing}")
        return None

    # Try loading numpy embeddings (more precise than TSV columns)
    tsne = None
    umap = None
    tsne_path = os.path.join(data_dir, "embedding_tsne.npy")
    umap_path = os.path.join(data_dir, "embedding_umap.npy")

    if os.path.isfile(tsne_path):
        tsne = np.load(tsne_path)
        logger.info(f"Loaded tSNE embedding: {tsne.shape}")
    elif "tsne_1" in ca.columns:
        tsne = ca[["tsne_1", "tsne_2"]].values
        logger.info(f"Using tSNE from TSV columns: {tsne.shape}")

    if os.path.isfile(umap_path):
        umap = np.load(umap_path)
        logger.info(f"Loaded UMAP embedding: {umap.shape}")
    elif "umap_1" in ca.columns:
        umap = ca[["umap_1", "umap_2"]].values
        logger.info(f"Using UMAP from TSV columns: {umap.shape}")

    # Maxpooled vectors
    mp_path = os.path.join(data_dir, "maxpooled_vectors.npy")
    mp = np.load(mp_path) if os.path.isfile(mp_path) else None
    if mp is not None:
        logger.info(f"Loaded maxpooled vectors: {mp.shape}")

    return {
        "cluster_assignments": ca,
        "embedding_tsne": tsne,
        "embedding_umap": umap,
        "maxpooled_vectors": mp,
    }


def find_scoring_boundaries(results_dir, chrom):
    """Find the drop_boundaries.tsv for a chromosome."""
    scoring_run = find_latest_completed(results_dir, chrom, "scoring")
    if scoring_run is None:
        # Try direct path
        scoring_dir = os.path.join(results_dir, chrom, "scoring")
        if os.path.isdir(scoring_dir):
            for entry in sorted(os.listdir(scoring_dir), reverse=True):
                path = os.path.join(scoring_dir, entry, "data", "drop_boundaries.tsv")
                if os.path.isfile(path):
                    return path
        return None
    return os.path.join(scoring_run, "data", "drop_boundaries.tsv")


def find_latent_dir(results_dir, chrom, latent_subdir="latent_analysis"):
    """Find the latent analysis directory for a chromosome.

    Tries:
      1. results/<chrom>/sae/<latent_subdir>/  (from_shards output)
      2. results/<chrom>/sae/<latest_completed>/<latent_subdir>/
    """
    def _has_valid_data(d):
        """Check that a latent dir has a non-empty cluster_assignments.tsv."""
        ca = os.path.join(d, "data", "cluster_assignments.tsv")
        if not os.path.isfile(ca):
            return False
        # Must have more than just comment/header lines
        return os.path.getsize(ca) > 500

    # Try under latest completed SAE run first (most reliable)
    sae_run = find_latest_completed(results_dir, chrom, "sae")
    if sae_run is not None:
        under_run = os.path.join(sae_run, latent_subdir)
        if _has_valid_data(under_run):
            return under_run

    # Fall back to from_shards output at top level
    direct = os.path.join(results_dir, chrom, "sae", latent_subdir)
    if _has_valid_data(direct):
        return direct

    return None


def resolve_chrom_id_for_gtf(chrom):
    """Get the chromosome ID as it appears in the GTF file."""
    return CHROM_MAP.get(chrom, chrom)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot generation functions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_distance_to_gene_plots(ca, embeddings, gtf_path, chrom, plots_dir):
    """Generate distance-to-gene plots for ALL regions.

    All points are plotted; genic regions have distance=0 (dark on colormap),
    intergenic regions are colored by their actual distance.
    """
    chrom_id = resolve_chrom_id_for_gtf(chrom)
    n_total = len(ca)

    logger.info(f"Computing distance to gene for {n_total} regions")

    upstream_dist, downstream_dist = compute_distance_to_nearest_gene(
        ca.genomic_start.values, ca.genomic_end.values,
        gtf_path, chrom_id
    )

    for emb_name, coords in embeddings.items():
        if coords is None:
            continue

        plot_continuous_scatter(
            coords, upstream_dist,
            cmap="plasma", colorbar_label="Distance to Upstream Gene (bp)",
            title=f"{emb_name.upper()} — All Regions\n"
                  f"Colored by Distance to Nearest Upstream Gene ({chrom}, N={n_total})",
            out_path=os.path.join(plots_dir, f"{emb_name}_upstream_gene_distance.png"),
            emb_name=emb_name, log_scale=True,
        )

        plot_continuous_scatter(
            coords, downstream_dist,
            cmap="plasma", colorbar_label="Distance to Downstream Gene (bp)",
            title=f"{emb_name.upper()} — All Regions\n"
                  f"Colored by Distance to Nearest Downstream Gene ({chrom}, N={n_total})",
            out_path=os.path.join(plots_dir, f"{emb_name}_downstream_gene_distance.png"),
            emb_name=emb_name, log_scale=True,
        )


def generate_entropy_color_plots(ca, embeddings, scoring_boundaries_path, chrom, plots_dir):
    """Generate entropy-colored plots."""
    if not os.path.isfile(scoring_boundaries_path):
        logger.warning(f"Scoring boundaries not found: {scoring_boundaries_path}")
        return

    avg_entropy, min_entropy = load_entropy_drop_stats(
        scoring_boundaries_path,
        ca.genomic_start.values, ca.genomic_end.values
    )

    for emb_name, coords in embeddings.items():
        if coords is None:
            continue

        plot_continuous_scatter(
            coords, avg_entropy,
            cmap="coolwarm_r", colorbar_label="Average Entropy Drop",
            title=f"{emb_name.upper()} — Colored by Average Entropy Drop\n({chrom}, N={len(ca)})",
            out_path=os.path.join(plots_dir, f"{emb_name}_by_avg_entropy.png"),
            emb_name=emb_name,
        )

        plot_continuous_scatter(
            coords, min_entropy,
            cmap="coolwarm_r", colorbar_label="Minimum Entropy Drop",
            title=f"{emb_name.upper()} — Colored by Minimum Entropy Drop\n({chrom}, N={len(ca)})",
            out_path=os.path.join(plots_dir, f"{emb_name}_by_min_entropy.png"),
            emb_name=emb_name,
        )


def generate_length_stats_plots(ca, embeddings, chrom, plots_dir, data_dir):
    """Generate region length distribution and length-colored plots."""
    lengths = compute_region_lengths(ca.genomic_start.values, ca.genomic_end.values)

    # Distribution plot
    stats = plot_length_distribution(
        lengths,
        os.path.join(plots_dir, "region_length_distribution.png"),
        title_prefix=chrom,
    )

    # Save stats JSON
    with open(os.path.join(data_dir, "region_length_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Length stats: mean={stats['mean']:.0f}, median={stats['median']:.0f}, "
                f"std={stats['std']:.1f}, uniform={stats['uniform']}")

    # Scatter colored by length
    for emb_name, coords in embeddings.items():
        if coords is None:
            continue

        plot_continuous_scatter(
            coords, lengths.astype(float),
            cmap="plasma", colorbar_label="Region Length (bp)",
            title=f"{emb_name.upper()} — Colored by Region Length\n({chrom}, N={len(ca)})",
            out_path=os.path.join(plots_dir, f"{emb_name}_by_region_length.png"),
            emb_name=emb_name,
        )


def generate_top_features(ca, maxpooled_vectors, chrom, data_dir):
    """Extract and save top-5 features per region."""
    if maxpooled_vectors is None:
        logger.warning("No maxpooled vectors available, skipping top features")
        return

    n_fired, top_k_features = compute_feature_firing_stats(maxpooled_vectors, top_k=5)

    save_top_features_tsv(
        top_k_features,
        ca.genomic_start.values, ca.genomic_end.values,
        os.path.join(data_dir, "top5_features_per_region.tsv"),
    )

    # Validation: check all top-5 have non-zero activations
    n_invalid = sum(1 for feats in top_k_features
                    for _, act in feats if act == 0)
    if n_invalid > 0:
        logger.warning(f"Found {n_invalid} zero-activation entries in top-5 features!")
    else:
        logger.info("Validation passed: all top-5 features have non-zero activations")


def generate_firing_count_plots(ca, embeddings, maxpooled_vectors, chrom, plots_dir, data_dir):
    """Generate plots colored by number of features fired."""
    if maxpooled_vectors is None:
        logger.warning("No maxpooled vectors available, skipping firing counts")
        return

    n_fired, _ = compute_feature_firing_stats(maxpooled_vectors, top_k=1)

    save_firing_stats_tsv(
        n_fired, ca.genomic_start.values, ca.genomic_end.values,
        os.path.join(data_dir, "firing_stats.tsv"),
    )

    for emb_name, coords in embeddings.items():
        if coords is None:
            continue

        plot_continuous_scatter(
            coords, n_fired.astype(float),
            cmap="inferno", colorbar_label="Number of Features Fired",
            title=f"{emb_name.upper()} — Colored by # Features Fired\n({chrom}, N={len(ca)})",
            out_path=os.path.join(plots_dir, f"{emb_name}_by_n_features_fired.png"),
            emb_name=emb_name,
        )


def generate_firing_threshold_plots(ca, embeddings, maxpooled_vectors, chrom, plots_dir):
    """Generate plots at different neuron firing thresholds."""
    if maxpooled_vectors is None:
        logger.warning("No maxpooled vectors available, skipping threshold plots")
        return

    thresholds = (0.01, 0.05, 0.10)
    threshold_results = compute_firing_threshold_counts(maxpooled_vectors, thresholds)

    for thresh, result in threshold_results.items():
        pct = int(thresh * 100)
        counts = result["per_region_counts"].astype(float)
        n_common = result["n_common_features"]

        for emb_name, coords in embeddings.items():
            if coords is None:
                continue

            plot_continuous_scatter(
                coords, counts,
                cmap="YlOrRd", colorbar_label=f"Active Common Features (of {n_common})",
                title=f"{emb_name.upper()} — Features Firing in >= {pct}% of Regions\n"
                      f"({chrom}, N={len(ca)}, {n_common} common features)",
                out_path=os.path.join(plots_dir, f"{emb_name}_firing_threshold_{pct}pct.png"),
                emb_name=emb_name,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Genome-wide mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_genome_wide(args, plot_types):
    """Run enhanced plots on genome-wide aggregated data."""
    results_dir = os.path.abspath(args.results_dir)
    cache_base = os.path.join(results_dir, "_genome_wide", "sae_tsne")

    # Look for cached genome-wide data — try _raw first, then normalized
    cache_dir = os.path.join(cache_base, "_cache")
    # Try raw first, fall back to normalized
    for suffix in ["_raw", "_normalized", ""]:
        candidate = os.path.join(cache_dir, f"combined_maxpooled{suffix}.npy")
        if os.path.isfile(candidate):
            combined_cache = candidate
            cache_suffix = suffix
            break
    else:
        combined_cache = os.path.join(cache_dir, "combined_maxpooled_raw.npy")
        cache_suffix = "_raw"
    metadata_cache = os.path.join(cache_dir, f"combined_metadata{cache_suffix}.json")

    if not os.path.isfile(metadata_cache):
        logger.error("No cached genome-wide metadata found. "
                     "Run genome_sae_tsne.py --all_human first.")
        sys.exit(1)

    # Load cached data
    with open(metadata_cache) as f:
        all_metadata = json.load(f)
    ca = pd.DataFrame(all_metadata)
    logger.info(f"Loaded genome-wide metadata: {len(ca)} regions")

    # Load embeddings
    embeddings = {}
    for name, fname in [("tsne", f"embedding_tsne{cache_suffix}.npy"), ("umap", f"embedding_umap{cache_suffix}.npy")]:
        path = os.path.join(cache_dir, fname)
        if os.path.isfile(path):
            embeddings[name] = np.load(path)
            logger.info(f"Loaded {name}: {embeddings[name].shape}")

    if not embeddings:
        logger.error("No cached embeddings found")
        sys.exit(1)

    # Load maxpooled vectors
    mp = np.load(combined_cache) if os.path.isfile(combined_cache) else None

    # Output directory
    output_dir = args.output_dir or os.path.join(
        cache_base,
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_enhanced_plots"
    )
    plots_dir = os.path.join(output_dir, "plots")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    label = f"Genome-Wide ({len(ca['chrom'].unique())} chroms)" if "chrom" in ca.columns else "Genome-Wide"

    # Run requested plots
    if "distance_to_gene" in plot_types:
        logger.info("=== Distance to Gene (genome-wide) ===")
        # Process per-chromosome then combine
        unique_chroms = ca["chrom"].unique() if "chrom" in ca.columns else []
        all_up = np.full(len(ca), np.nan)
        all_down = np.full(len(ca), np.nan)
        for chrom in unique_chroms:
            mask = ca["chrom"] == chrom
            chrom_id = resolve_chrom_id_for_gtf(chrom)
            up, down = compute_distance_to_nearest_gene(
                ca.loc[mask, "genomic_start"].values,
                ca.loc[mask, "genomic_end"].values,
                args.gtf, chrom_id
            )
            all_up[mask.values] = up
            all_down[mask.values] = down

        for emb_name, coords in embeddings.items():
            if coords is None:
                continue
            plot_continuous_scatter(
                coords, all_up,
                cmap="plasma", colorbar_label="Distance to Upstream Gene (bp)",
                title=f"{emb_name.upper()} — All Regions: Distance to Upstream Gene\n({label})",
                out_path=os.path.join(plots_dir, f"{emb_name}_upstream_gene_distance.png"),
                emb_name=emb_name, log_scale=True,
            )
            plot_continuous_scatter(
                coords, all_down,
                cmap="plasma", colorbar_label="Distance to Downstream Gene (bp)",
                title=f"{emb_name.upper()} — All Regions: Distance to Downstream Gene\n({label})",
                out_path=os.path.join(plots_dir, f"{emb_name}_downstream_gene_distance.png"),
                emb_name=emb_name, log_scale=True,
            )

    if "entropy_color" in plot_types:
        logger.info("=== Entropy Coloring (genome-wide) ===")
        all_avg = np.full(len(ca), np.nan)
        all_min = np.full(len(ca), np.nan)
        unique_chroms = ca["chrom"].unique() if "chrom" in ca.columns else []
        for chrom in unique_chroms:
            mask = ca["chrom"] == chrom
            bounds_path = find_scoring_boundaries(results_dir, chrom)
            if bounds_path and os.path.isfile(bounds_path):
                avg, mn = load_entropy_drop_stats(
                    bounds_path,
                    ca.loc[mask, "genomic_start"].values,
                    ca.loc[mask, "genomic_end"].values
                )
                all_avg[mask.values] = avg
                all_min[mask.values] = mn

        for emb_name, coords in embeddings.items():
            if coords is None:
                continue
            plot_continuous_scatter(
                coords, all_avg, cmap="coolwarm_r",
                colorbar_label="Average Entropy Drop",
                title=f"{emb_name.upper()} — Average Entropy Drop\n({label})",
                out_path=os.path.join(plots_dir, f"{emb_name}_by_avg_entropy.png"),
                emb_name=emb_name,
            )
            plot_continuous_scatter(
                coords, all_min, cmap="coolwarm_r",
                colorbar_label="Minimum Entropy Drop",
                title=f"{emb_name.upper()} — Minimum Entropy Drop\n({label})",
                out_path=os.path.join(plots_dir, f"{emb_name}_by_min_entropy.png"),
                emb_name=emb_name,
            )

    if "length_stats" in plot_types:
        generate_length_stats_plots(ca, embeddings, label, plots_dir, data_dir)

    if "top_features" in plot_types:
        generate_top_features(ca, mp, label, data_dir)

    if "firing_counts" in plot_types:
        generate_firing_count_plots(ca, embeddings, mp, label, plots_dir, data_dir)

    if "firing_thresholds" in plot_types:
        generate_firing_threshold_plots(ca, embeddings, mp, label, plots_dir)

    logger.info(f"\nGenome-wide enhanced plots saved to: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-chromosome / per-organism mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_chrom(args, chrom, plot_types):
    """Run enhanced plots for a single chromosome."""
    results_dir = os.path.abspath(args.results_dir)
    latent_subdir = args.latent_subdir

    # Find latent analysis directory
    latent_dir = find_latent_dir(results_dir, chrom, latent_subdir)
    if latent_dir is None:
        logger.error(f"No latent analysis found for {chrom}")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {chrom} — {latent_dir}")
    logger.info(f"{'='*60}")

    # Load data
    data = load_embeddings_and_metadata(latent_dir)
    if data is None:
        return False

    ca = data["cluster_assignments"]
    embeddings = {}
    if data["embedding_tsne"] is not None:
        embeddings["tsne"] = data["embedding_tsne"]
    if data["embedding_umap"] is not None:
        embeddings["umap"] = data["embedding_umap"]

    if not embeddings:
        logger.error(f"No embeddings found for {chrom}")
        return False

    # Output directories
    if args.output_dir:
        plots_dir = os.path.join(args.output_dir, chrom, "plots")
        data_dir_out = os.path.join(args.output_dir, chrom, "data")
    else:
        plots_dir = os.path.join(latent_dir, "plots")
        data_dir_out = os.path.join(latent_dir, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir_out, exist_ok=True)

    # Run requested plots
    if "distance_to_gene" in plot_types:
        logger.info("--- Distance to Gene ---")
        generate_distance_to_gene_plots(ca, embeddings, args.gtf, chrom, plots_dir)

    if "entropy_color" in plot_types:
        logger.info("--- Entropy Coloring ---")
        bounds_path = args.scoring_boundaries
        if not bounds_path:
            bounds_path = find_scoring_boundaries(results_dir, chrom)
        if bounds_path:
            generate_entropy_color_plots(ca, embeddings, bounds_path, chrom, plots_dir)
        else:
            logger.warning(f"No scoring boundaries found for {chrom}")

    if "length_stats" in plot_types:
        logger.info("--- Length Statistics ---")
        generate_length_stats_plots(ca, embeddings, chrom, plots_dir, data_dir_out)

    if "top_features" in plot_types:
        logger.info("--- Top Features ---")
        generate_top_features(ca, data["maxpooled_vectors"], chrom, data_dir_out)

    if "firing_counts" in plot_types:
        logger.info("--- Firing Counts ---")
        generate_firing_count_plots(ca, embeddings, data["maxpooled_vectors"],
                                    chrom, plots_dir, data_dir_out)

    if "firing_thresholds" in plot_types:
        logger.info("--- Firing Thresholds ---")
        generate_firing_threshold_plots(ca, embeddings, data["maxpooled_vectors"],
                                        chrom, plots_dir)

    logger.info(f"Done: {chrom} → {plots_dir}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced SAE latent analysis plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--scope", type=str, required=True,
                        choices=["chromosome", "genome_wide", "organism"],
                        help="Analysis scope")
    parser.add_argument("--chrom", type=str, default=None,
                        help="Chromosome name (required for scope=chromosome)")
    parser.add_argument("--organism", type=str, default="human",
                        choices=["human", "ecoli", "bacillus"],
                        help="Organism (default: human)")
    parser.add_argument("--gtf", type=str, required=True,
                        help="Path to GTF annotation file")
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="Root results directory")
    parser.add_argument("--scoring_boundaries", type=str, default=None,
                        help="Path to drop_boundaries.tsv (auto-discovered if omitted)")
    parser.add_argument("--plots", type=str,
                        default="distance_to_gene,entropy_color,length_stats,"
                                "top_features,firing_counts,firing_thresholds",
                        help="Comma-separated plot types to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--latent_subdir", type=str, default="latent_analysis",
                        help="Subdirectory name for latent analysis data")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    global logger
    logger = setup_logging(args.log_level)

    t0 = time.time()
    plot_types = [p.strip() for p in args.plots.split(",")]

    logger.info("Enhanced SAE Latent Analysis Plots")
    logger.info(f"Scope: {args.scope}, Organism: {args.organism}")
    logger.info(f"Plot types: {plot_types}")

    if args.scope == "genome_wide":
        run_genome_wide(args, plot_types)

    elif args.scope == "chromosome":
        if not args.chrom:
            parser.error("--chrom required for scope=chromosome")
        run_single_chrom(args, args.chrom, plot_types)

    elif args.scope == "organism":
        chroms = ORGANISM_CHROMS.get(args.organism, [])
        if not chroms:
            logger.error(f"Unknown organism: {args.organism}")
            sys.exit(1)
        n_ok = 0
        for chrom in chroms:
            if run_single_chrom(args, chrom, plot_types):
                n_ok += 1
        logger.info(f"\nCompleted {n_ok}/{len(chroms)} chromosomes")

    wall_time = time.time() - t0
    logger.info(f"Total wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
