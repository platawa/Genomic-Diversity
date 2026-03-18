#!/usr/bin/env python3
"""Intergenic feature specificity analysis for bacterial genomes.

Identifies SAE features that activate selectively in intergenic (non-gene)
low-entropy regions vs genic regions. Uses pre-computed SAE maxpooled vectors
and GTF annotations.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from results_utils import find_latest_completed, build_run_dir, write_completed
from tools.plot_tsne_by_annotation import load_gtf_features, classify_region
from tools.aggregate_genome_sae_stats import load_maxpooled_vectors
from tools.plot_intergenic_features import generate_all_plots

logger = logging.getLogger(__name__)


def load_region_coords(sae_run_dir):
    """Load region start/end coordinates from sae_results.tsv, skipping comment lines.

    Returns list of dicts with keys: region_idx, start, end, method, confidence.
    """
    tsv_path = os.path.join(sae_run_dir, "data", "sae_results.tsv")
    if not os.path.isfile(tsv_path):
        return None
    rows = []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            # Skip header row
            try:
                int(fields[0])
            except ValueError:
                continue
            rows.append({
                "region_idx": int(fields[0]),
                "start": int(fields[1]),
                "end": int(fields[2]),
                "method": fields[3],
                "confidence": float(fields[4]) if len(fields) > 4 else 0.0,
            })
    return rows


def classify_all_regions(metadata, gtf_path, chrom_id):
    """Classify each region as CDS/UTR-exon/Intron/Intergenic using GTF."""
    intervals = load_gtf_features(gtf_path, chrom_id)
    annotations = []
    for region in metadata:
        start = region["start"]
        end = region["end"]
        label = classify_region(start, end, intervals)
        annotations.append(label)
    logger.info("Region classification counts:")
    for label in sorted(set(annotations)):
        count = annotations.count(label)
        logger.info(f"  {label}: {count} ({100*count/len(annotations):.1f}%)")
    return annotations


def compute_feature_specificity(vectors, annotations, group_a="Intergenic", group_b=None):
    """Compute differential activation stats for each SAE feature.

    group_b=None means 'everything not in group_a'.
    Returns DataFrame with per-feature specificity metrics.
    """
    annotations = np.array(annotations)
    mask_a = annotations == group_a
    if group_b is None:
        mask_b = ~mask_a
    else:
        mask_b = annotations == group_b

    n_a, n_b = mask_a.sum(), mask_b.sum()
    logger.info(f"Group A ({group_a}): {n_a} regions, Group B: {n_b} regions")

    if n_a < 2 or n_b < 2:
        logger.error(f"Insufficient regions: {n_a} in group A, {n_b} in group B")
        return None

    vecs_a = vectors[mask_a]
    vecs_b = vectors[mask_b]
    n_features = vectors.shape[1]

    eps = 1e-10
    mean_a = vecs_a.mean(axis=0)
    mean_b = vecs_b.mean(axis=0)
    freq_a = (vecs_a > 0).mean(axis=0)
    freq_b = (vecs_b > 0).mean(axis=0)

    fold_change = (mean_a + eps) / (mean_b + eps)
    specificity_index = (mean_a - mean_b) / (mean_a + mean_b + eps)

    # Cohen's d
    pooled_std = np.sqrt(
        ((n_a - 1) * vecs_a.var(axis=0) + (n_b - 1) * vecs_b.var(axis=0))
        / (n_a + n_b - 2 + eps)
    )
    cohens_d = (mean_a - mean_b) / (pooled_std + eps)

    # Mann-Whitney U test per feature
    logger.info(f"Running Mann-Whitney U tests for {n_features} features...")
    pvalues = np.ones(n_features)
    u_stats = np.zeros(n_features)

    active_mask = (mean_a > 0) | (mean_b > 0)
    active_indices = np.where(active_mask)[0]
    logger.info(f"  Testing {len(active_indices)} active features (skipping {n_features - len(active_indices)} always-zero)")

    for i in active_indices:
        try:
            u, p = stats.mannwhitneyu(vecs_a[:, i], vecs_b[:, i], alternative="two-sided")
            u_stats[i] = u
            pvalues[i] = p
        except ValueError:
            pass

    # FDR correction (Benjamini-Hochberg) — manual implementation
    n_tests = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    qvalues = np.ones(n_tests)
    cummin = 1.0
    for i in range(n_tests - 1, -1, -1):
        q = sorted_pvals[i] * n_tests / (i + 1)
        cummin = min(cummin, q)
        qvalues[sorted_idx[i]] = min(cummin, 1.0)

    df = pd.DataFrame({
        "feature_idx": np.arange(n_features),
        "mean_intergenic": mean_a,
        "mean_genic": mean_b,
        "freq_intergenic": freq_a,
        "freq_genic": freq_b,
        "fold_change": fold_change,
        "log2_fold_change": np.log2(fold_change),
        "specificity_index": specificity_index,
        "cohens_d": cohens_d,
        "mann_whitney_U": u_stats,
        "pvalue": pvalues,
        "fdr_qvalue": qvalues,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Intergenic feature specificity analysis")
    parser.add_argument("--chrom", required=True, help="Chromosome ID (e.g., NC_000913.3)")
    parser.add_argument("--gtf", required=True, help="Path to GTF annotation file")
    parser.add_argument("--results_dir", default="results", help="Results base directory")
    parser.add_argument("--sae_run", default=None, help="Specific SAE run dir (default: auto-detect latest)")
    parser.add_argument("--fdr_threshold", type=float, default=0.05, help="FDR q-value threshold")
    parser.add_argument("--min_fold_change", type=float, default=2.0, help="Minimum fold change for significance")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top features to visualize")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    # 1. Find SAE run
    if args.sae_run:
        sae_run_dir = args.sae_run
    else:
        sae_run_dir = find_latest_completed(args.results_dir, args.chrom, "sae")
        if sae_run_dir is None:
            logger.error(f"No completed SAE run found for {args.chrom}")
            sys.exit(1)
    logger.info(f"Using SAE run: {sae_run_dir}")

    # 2. Load data
    vectors = load_maxpooled_vectors(sae_run_dir)
    if vectors is None:
        logger.error("Failed to load maxpooled vectors")
        sys.exit(1)
    logger.info(f"Loaded vectors: {vectors.shape}")

    metadata = load_region_coords(sae_run_dir)
    if metadata is None:
        logger.error("Failed to load region metadata")
        sys.exit(1)
    logger.info(f"Loaded metadata for {len(metadata)} regions")

    # Align vectors and metadata
    n = min(vectors.shape[0], len(metadata))
    vectors = vectors[:n]
    metadata = metadata[:n]

    # 3. Classify regions
    annotations = classify_all_regions(metadata, args.gtf, args.chrom)

    # 4. Compute feature specificity
    df = compute_feature_specificity(vectors, annotations)
    if df is None:
        sys.exit(1)

    # 5. Create output directory
    run_dir = build_run_dir(
        args.results_dir, args.chrom, "intergenic_analysis",
        f"fdr{args.fdr_threshold}_fc{args.min_fold_change}"
    )
    data_dir = os.path.join(run_dir, "data")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 6. Save full results
    df_sorted = df.sort_values("specificity_index", ascending=False)
    df_sorted.to_csv(os.path.join(data_dir, "feature_specificity.tsv"), sep="\t", index=False)

    # Filter significant intergenic-specific features
    sig = df_sorted[
        (df_sorted["fdr_qvalue"] < args.fdr_threshold)
        & (df_sorted["fold_change"] > args.min_fold_change)
    ]
    sig.to_csv(os.path.join(data_dir, "intergenic_specific_features.tsv"), sep="\t", index=False)
    logger.info(f"Found {len(sig)} intergenic-specific features (FDR<{args.fdr_threshold}, FC>{args.min_fold_change})")

    # Save region annotations
    annot_df = pd.DataFrame({
        "region_idx": range(n),
        "start": [m["start"] for m in metadata],
        "end": [m["end"] for m in metadata],
        "annotation": annotations,
    })
    annot_df.to_csv(os.path.join(data_dir, "region_annotations.tsv"), sep="\t", index=False)

    # 7. Summary
    top_features_sig = sig["feature_idx"].values[:args.top_n] if len(sig) > 0 else df_sorted["feature_idx"].values[:args.top_n]
    top_features_sig = top_features_sig.astype(int)

    summary = {
        "chrom": args.chrom,
        "sae_run": sae_run_dir,
        "gtf": args.gtf,
        "n_regions": n,
        "annotation_counts": {l: int(c) for l, c in zip(*np.unique(annotations, return_counts=True))},
        "fdr_threshold": args.fdr_threshold,
        "min_fold_change": args.min_fold_change,
        "n_significant_intergenic": len(sig),
        "top_features": top_features_sig.tolist(),
    }
    with open(os.path.join(data_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 8. Write source lineage
    source_path = os.path.join(run_dir, "source.json")
    with open(source_path, "w") as f:
        json.dump({"sae_run": os.path.relpath(sae_run_dir, run_dir)}, f, indent=2)

    # 9. Generate plots
    if not args.no_plots:
        generate_all_plots(run_dir, vectors=vectors, annotations=annotations,
                           fdr_threshold=args.fdr_threshold, min_fold_change=args.min_fold_change,
                           top_n=args.top_n)

    wall_time = time.time() - t0
    write_completed(run_dir, "intergenic_feature_analysis.py", wall_time)
    logger.info(f"Done in {wall_time:.1f}s. Output: {run_dir}")


if __name__ == "__main__":
    main()
