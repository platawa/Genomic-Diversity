#!/usr/bin/env python3
"""Cross-organism SAE feature comparison.

Compares SAE feature activation patterns across organisms (human, E. coli,
Bacillus) to determine whether Evo2 has learned universal biological concepts
or organism-specific representations.

Key analyses:
  1. Feature universality: Which features activate across all organisms?
  2. Feature specificity: Which features are organism-specific?
  3. Annotation-feature correlation: Do the same features mean CDS/intergenic
     in different organisms?
  4. Dimensionality reduction: Joint t-SNE/PCA of all organisms' regions

Usage:
    python tools/cross_organism_features.py \
        --results_dir results/ \
        --organisms ecoli:NC_000913.3:/path/to/ecoli.gtf \
                    bacillus:NC_000964.3:/path/to/bacillus.gtf \
                    human:chr22:/path/to/human.gtf

    # With predefined organism configs
    python tools/cross_organism_features.py \
        --results_dir results/ \
        --preset all
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from results_utils import find_latest_completed, build_run_dir, write_completed
from tools.plot_tsne_by_annotation import load_gtf_features, classify_region
from tools.aggregate_genome_sae_stats import load_maxpooled_vectors

logger = logging.getLogger(__name__)

# Predefined organism configurations
ORGANISM_PRESETS = {
    "ecoli": {
        "chroms": ["NC_000913.3"],
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_ecoli/ncbi_dataset/data/GCF_000005845.2/genomic.gtf",
    },
    "bacillus": {
        "chroms": ["NC_000964.3"],
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_bacillus/ncbi_dataset/data/GCF_000009045.1/genomic.gtf",
    },
    "human": {
        "chroms": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
        "gtf": "/orcd/data/zhang_f/001/platawa/data/MEng_Thesis/ncbi_dataset_all_2/ncbi_dataset/data/GCF_000001405.26/genomic.gtf",
    },
}

LABEL_ORDER = ["CDS", "UTR/exon", "Intron", "Intergenic"]


def load_organism_data(results_dir, organism_name, chroms, gtf_path, max_regions_per_chrom=None):
    """Load SAE vectors and annotations for all chromosomes of one organism.

    Returns dict with keys: vectors, labels, chroms, regions, organism.
    """
    all_vecs, all_labels, all_chroms = [], [], []

    for chrom in chroms:
        sae_run = find_latest_completed(results_dir, chrom, "sae")
        if sae_run is None:
            logger.info(f"  {organism_name}/{chrom}: no completed SAE run, skipping")
            continue

        vectors = load_maxpooled_vectors(sae_run)
        if vectors is None:
            logger.info(f"  {organism_name}/{chrom}: no vectors, skipping")
            continue

        # Load region coordinates
        tsv_path = os.path.join(sae_run, "data", "sae_results.tsv")
        if not os.path.isfile(tsv_path):
            logger.info(f"  {organism_name}/{chrom}: no sae_results.tsv, skipping")
            continue

        regions = []
        with open(tsv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    continue
                try:
                    int(fields[0])
                except ValueError:
                    continue
                regions.append({"start": int(fields[1]), "end": int(fields[2])})

        n = min(len(regions), vectors.shape[0])
        regions = regions[:n]
        vectors = vectors[:n]

        if max_regions_per_chrom and n > max_regions_per_chrom:
            idx = np.random.RandomState(42).choice(n, max_regions_per_chrom, replace=False)
            idx.sort()
            vectors = vectors[idx]
            regions = [regions[i] for i in idx]
            n = max_regions_per_chrom

        # Classify regions
        intervals = load_gtf_features(gtf_path, chrom)
        labels = [classify_region(r["start"], r["end"], intervals) for r in regions]

        all_vecs.append(vectors)
        all_labels.extend(labels)
        all_chroms.extend([chrom] * n)

        label_counts = {l: labels.count(l) for l in set(labels)}
        logger.info(f"  {organism_name}/{chrom}: {n} regions — {label_counts}")

    if not all_vecs:
        return None

    return {
        "organism": organism_name,
        "vectors": np.vstack(all_vecs),
        "labels": np.array(all_labels),
        "chroms": np.array(all_chroms),
    }


def compute_feature_activity_profile(vectors, labels):
    """Compute per-annotation mean activation for each SAE feature.

    Returns dict: {annotation: mean_vector (32768,)}.
    """
    profiles = {}
    for label in set(labels):
        mask = labels == label
        if mask.sum() == 0:
            continue
        profiles[label] = vectors[mask].mean(axis=0)
    return profiles


def compare_feature_profiles(profiles_a, profiles_b, org_a, org_b):
    """Compare feature profiles between two organisms.

    For each annotation type present in both organisms, compute:
      - Pearson correlation of per-feature mean activations
      - Top shared features (high in both)
      - Top divergent features (high in one, low in other)
    """
    from scipy.stats import pearsonr, spearmanr

    results = {}
    shared_labels = set(profiles_a.keys()) & set(profiles_b.keys())

    for label in sorted(shared_labels):
        va = profiles_a[label]
        vb = profiles_b[label]

        # Correlation on non-zero features
        active_mask = (va > 0) | (vb > 0)
        if active_mask.sum() < 10:
            continue

        va_active = va[active_mask]
        vb_active = vb[active_mask]

        pearson_r, pearson_p = pearsonr(va_active, vb_active)
        spearman_r, spearman_p = spearmanr(va_active, vb_active)

        # Top shared features (high mean activation in both)
        combined = np.minimum(va, vb)
        top_shared = np.argsort(combined)[::-1][:20]

        # Features specific to organism A
        diff_a = va - vb
        top_a_specific = np.argsort(diff_a)[::-1][:20]

        # Features specific to organism B
        top_b_specific = np.argsort(-diff_a)[::-1][:20]

        results[label] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n_active_features": int(active_mask.sum()),
            "top_shared_features": top_shared.tolist(),
            "top_shared_activations_a": va[top_shared].tolist(),
            "top_shared_activations_b": vb[top_shared].tolist(),
            f"top_{org_a}_specific": top_a_specific.tolist(),
            f"top_{org_b}_specific": top_b_specific.tolist(),
        }

    return results


def compute_feature_universality(all_profiles):
    """Identify universal vs organism-specific features.

    A feature is 'universal' if it has similar relative activation patterns
    across all organisms for the same annotation type.
    """
    organisms = list(all_profiles.keys())
    if len(organisms) < 2:
        return {}

    n_features = 32768
    # For each feature, count how many organisms show it as active (mean > 0.01)
    feature_organism_count = np.zeros(n_features, dtype=int)
    feature_organism_active = defaultdict(set)

    for org, profiles in all_profiles.items():
        for label, profile in profiles.items():
            active = np.where(profile > 0.01)[0]
            for f in active:
                feature_organism_active[int(f)].add(org)

    for f, orgs in feature_organism_active.items():
        feature_organism_count[f] = len(orgs)

    # Universal: active in all organisms
    universal = np.where(feature_organism_count == len(organisms))[0]
    # Organism-specific: active in exactly one
    specific = np.where(feature_organism_count == 1)[0]

    # For each universal feature, compute cross-organism consistency
    universal_details = []
    for f_idx in universal[:100]:  # top 100
        activations = {}
        for org, profiles in all_profiles.items():
            org_act = {label: float(profile[f_idx]) for label, profile in profiles.items()}
            activations[org] = org_act
        universal_details.append({
            "feature_idx": int(f_idx),
            "activations_by_organism": activations,
        })

    # For organism-specific, group by organism
    specific_by_org = defaultdict(list)
    for f_idx in specific:
        for org, orgs_set in feature_organism_active.items():
            if f_idx in [fi for fi, os in feature_organism_active.items() if os == {org}]:
                # Find which organism this feature belongs to
                pass
        for org in organisms:
            org_features = set()
            for fi, os in feature_organism_active.items():
                if os == {org}:
                    org_features.add(fi)
            if f_idx in org_features:
                specific_by_org[org].append(int(f_idx))
                break

    return {
        "n_universal": int(len(universal)),
        "n_organism_specific": int(len(specific)),
        "n_inactive": int(np.sum(feature_organism_count == 0)),
        "universal_features": universal[:200].tolist(),
        "universal_details": universal_details[:50],
        "organism_specific": {org: feats[:100] for org, feats in specific_by_org.items()},
    }


def plot_cross_organism(organism_data, all_profiles, comparisons, universality, output_dir):
    """Generate cross-organism comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    org_names = list(organism_data.keys())
    org_colors = {"human": "#2196F3", "ecoli": "#4CAF50", "bacillus": "#FF9800"}

    # --- 1. Feature correlation heatmap across organisms ---
    if len(comparisons) > 0:
        all_labels = set()
        for comp in comparisons.values():
            all_labels.update(comp.keys())
        all_labels = sorted(all_labels)

        fig, axes = plt.subplots(1, len(all_labels), figsize=(6 * len(all_labels), 5))
        if len(all_labels) == 1:
            axes = [axes]

        for ax, label in zip(axes, all_labels):
            pair_names = []
            correlations = []
            for pair_key, comp in comparisons.items():
                if label in comp:
                    pair_names.append(pair_key)
                    correlations.append(comp[label]["pearson_r"])

            if pair_names:
                bars = ax.barh(pair_names, correlations, color="#607D8B")
                ax.set_xlim(-0.2, 1.0)
                ax.set_title(f"{label} features")
                ax.set_xlabel("Pearson r")
                for bar, r in zip(bars, correlations):
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                            f"{r:.3f}", va="center", fontsize=10)

        fig.suptitle("Cross-Organism Feature Correlation by Annotation Type", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "feature_correlation_by_annotation.png"), dpi=150)
        plt.close(fig)

    # --- 2. Universality summary ---
    if universality:
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Universal\n(all organisms)", "Organism-\nspecific", "Inactive"]
        counts = [
            universality["n_universal"],
            universality["n_organism_specific"],
            universality["n_inactive"],
        ]
        colors = ["#4CAF50", "#FF9800", "#9E9E9E"]
        bars = ax.bar(categories, counts, color=colors)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{c:,}", ha="center", fontsize=12)
        ax.set_ylabel("Number of SAE features (out of 32,768)")
        ax.set_title("SAE Feature Universality Across Organisms")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "feature_universality.png"), dpi=150)
        plt.close(fig)

    # --- 3. Joint t-SNE ---
    all_vecs, all_org_labels, all_annot_labels = [], [], []
    max_per_org = 2000  # Subsample for t-SNE performance
    for org_name, data in organism_data.items():
        n = len(data["labels"])
        if n > max_per_org:
            idx = np.random.RandomState(42).choice(n, max_per_org, replace=False)
        else:
            idx = np.arange(n)
        all_vecs.append(data["vectors"][idx])
        all_org_labels.extend([org_name] * len(idx))
        all_annot_labels.extend(data["labels"][idx])

    if len(all_vecs) >= 2:
        X = np.vstack(all_vecs)
        org_arr = np.array(all_org_labels)
        annot_arr = np.array(all_annot_labels)

        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA

            # PCA to 50 dims first
            logger.info(f"Running PCA + t-SNE on {X.shape[0]} regions...")
            pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
            X_pca = pca.fit_transform(X)

            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
            X_tsne = tsne.fit_transform(X_pca)

            # Plot colored by organism
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            for org_name in org_names:
                mask = org_arr == org_name
                color = org_colors.get(org_name, "#999999")
                ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.4,
                            c=color, label=f"{org_name} (n={mask.sum()})")
            ax1.legend(markerscale=4)
            ax1.set_title("Colored by Organism")
            ax1.set_xlabel("t-SNE 1")
            ax1.set_ylabel("t-SNE 2")

            # Plot colored by annotation
            annot_colors = {
                "CDS": "#2196F3", "UTR/exon": "#4CAF50",
                "Intron": "#FF9800", "Intergenic": "#F44336",
            }
            for label in LABEL_ORDER:
                mask = annot_arr == label
                if mask.sum() == 0:
                    continue
                ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.4,
                            c=annot_colors.get(label, "#999"), label=f"{label} (n={mask.sum()})")
            ax2.legend(markerscale=4)
            ax2.set_title("Colored by Annotation")
            ax2.set_xlabel("t-SNE 1")
            ax2.set_ylabel("t-SNE 2")

            fig.suptitle("Joint t-SNE of SAE Features Across Organisms", fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "joint_tsne.png"), dpi=150)
            plt.close(fig)

            # Save t-SNE coordinates
            np.savez_compressed(
                os.path.join(output_dir, "data", "tsne_coords.npz"),
                coords=X_tsne,
                organisms=org_arr,
                annotations=annot_arr,
            )

        except ImportError:
            logger.warning("sklearn not available, skipping t-SNE")

    # --- 4. Per-organism annotation profile heatmaps ---
    fig, axes = plt.subplots(1, len(all_profiles), figsize=(6 * len(all_profiles), 6))
    if len(all_profiles) == 1:
        axes = [axes]

    for ax, (org_name, profiles) in zip(axes, all_profiles.items()):
        labels_present = [l for l in LABEL_ORDER if l in profiles]
        if not labels_present:
            continue
        # Top 30 most variable features across annotation types
        stacked = np.stack([profiles[l] for l in labels_present])
        feature_var = stacked.var(axis=0)
        top_feats = np.argsort(feature_var)[::-1][:30]

        mat = stacked[:, top_feats]  # (n_labels, 30)
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(labels_present)))
        ax.set_yticklabels(labels_present)
        ax.set_xticks(range(len(top_feats)))
        ax.set_xticklabels([str(f) for f in top_feats], rotation=90, fontsize=7)
        ax.set_xlabel("SAE Feature Index")
        ax.set_title(f"{org_name}")
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Top Annotation-Discriminative Features per Organism", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "annotation_profiles.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-organism SAE feature comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results/",
                        help="Root results directory")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory")
    parser.add_argument("--organisms", nargs="+", default=None,
                        help="Organism specs: name:chrom:gtf_path (repeatable)")
    parser.add_argument("--preset", choices=["all", "bacteria", "human_ecoli"],
                        default=None, help="Use predefined organism configs")
    parser.add_argument("--max_regions", type=int, default=None,
                        help="Max regions per chromosome (for speed)")
    parser.add_argument("--human_chroms", nargs="+", default=None,
                        help="Subset of human chroms to use (e.g., chr21 chr22)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t0 = time.time()

    # Build organism config
    org_configs = {}
    if args.preset:
        if args.preset == "all":
            org_configs = dict(ORGANISM_PRESETS)
        elif args.preset == "bacteria":
            org_configs = {k: v for k, v in ORGANISM_PRESETS.items() if k != "human"}
        elif args.preset == "human_ecoli":
            org_configs = {k: v for k, v in ORGANISM_PRESETS.items() if k != "bacillus"}

        if args.human_chroms and "human" in org_configs:
            org_configs["human"]["chroms"] = args.human_chroms

    elif args.organisms:
        for spec in args.organisms:
            parts = spec.split(":")
            if len(parts) != 3:
                logger.error(f"Invalid organism spec: {spec} (expected name:chrom:gtf)")
                sys.exit(1)
            name, chrom, gtf = parts
            if name not in org_configs:
                org_configs[name] = {"chroms": [], "gtf": gtf}
            org_configs[name]["chroms"].append(chrom)
    else:
        parser.error("Specify --preset or --organisms")

    # Load data for each organism
    organism_data = {}
    for org_name, config in org_configs.items():
        logger.info(f"Loading {org_name} ({len(config['chroms'])} chroms)...")
        data = load_organism_data(
            args.results_dir, org_name, config["chroms"], config["gtf"],
            max_regions_per_chrom=args.max_regions,
        )
        if data is not None:
            organism_data[org_name] = data
            logger.info(f"  {org_name}: {data['vectors'].shape[0]} total regions")
        else:
            logger.warning(f"  {org_name}: no data available")

    if len(organism_data) < 2:
        logger.error("Need at least 2 organisms with data for comparison")
        sys.exit(1)

    # Compute per-organism feature profiles
    all_profiles = {}
    for org_name, data in organism_data.items():
        all_profiles[org_name] = compute_feature_activity_profile(
            data["vectors"], data["labels"],
        )

    # Pairwise comparisons
    org_names = sorted(organism_data.keys())
    comparisons = {}
    for i in range(len(org_names)):
        for j in range(i + 1, len(org_names)):
            a, b = org_names[i], org_names[j]
            logger.info(f"Comparing {a} vs {b}...")
            comp = compare_feature_profiles(all_profiles[a], all_profiles[b], a, b)
            comparisons[f"{a}_vs_{b}"] = comp
            for label, stats in comp.items():
                logger.info(f"  {label}: pearson_r={stats['pearson_r']:.4f} "
                            f"(p={stats['pearson_p']:.2e}), "
                            f"n_active={stats['n_active_features']}")

    # Feature universality
    logger.info("Computing feature universality...")
    universality = compute_feature_universality(all_profiles)
    logger.info(f"  Universal: {universality.get('n_universal', 0)}, "
                f"Organism-specific: {universality.get('n_organism_specific', 0)}, "
                f"Inactive: {universality.get('n_inactive', 0)}")

    wall_time = time.time() - t0

    # Output
    desc = "_".join(org_names)
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = build_run_dir(
            args.results_dir, "_cross_organism_features", "analysis", desc,
        )

    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save results
    with open(os.path.join(data_dir, "pairwise_comparisons.json"), "w") as f:
        json.dump(comparisons, f, indent=2, default=str)
        f.write("\n")

    with open(os.path.join(data_dir, "feature_universality.json"), "w") as f:
        json.dump(universality, f, indent=2, default=str)
        f.write("\n")

    # Save per-organism profiles (compressed — 32768 floats per annotation)
    for org_name, profiles in all_profiles.items():
        arrays = {label: profile for label, profile in profiles.items()}
        np.savez_compressed(
            os.path.join(data_dir, f"feature_profile_{org_name}.npz"),
            **arrays,
        )

    # Generate plots
    plot_cross_organism(organism_data, all_profiles, comparisons, universality, out_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Cross-Organism Feature Comparison")
    logger.info(f"  Organisms: {', '.join(org_names)}")
    for pair, comp in comparisons.items():
        logger.info(f"  {pair}:")
        for label, stats in comp.items():
            logger.info(f"    {label}: r={stats['pearson_r']:.4f}")
    logger.info(f"  Universal features: {universality.get('n_universal', 0)}")
    logger.info(f"  Organism-specific: {universality.get('n_organism_specific', 0)}")
    logger.info(f"  Output: {out_dir}")
    logger.info("=" * 60)

    write_completed(out_dir, "cross_organism_features.py", wall_time)


if __name__ == "__main__":
    main()
