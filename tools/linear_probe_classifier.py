#!/usr/bin/env python3
"""Linear probe classifier on SAE max-pooled feature vectors.

Trains a simple logistic regression (linear probe) on SAE features to predict
genomic annotation type (CDS, UTR/exon, Intron, Intergenic). If a linear model
can decode annotations from SAE features, it proves Evo2 has learned a linearly
separable representation of genome structure.

Usage:
    # Single chromosome
    python tools/linear_probe_classifier.py \
        --chrom NC_000913.3 \
        --gtf /path/to/genomic.gtf \
        --results_dir results/

    # Cross-organism (train on E. coli, test on Bacillus)
    python tools/linear_probe_classifier.py \
        --train_chroms NC_000913.3 \
        --train_gtf /path/to/ecoli.gtf \
        --test_chroms NC_000964.3 \
        --test_gtf /path/to/bacillus.gtf \
        --results_dir results/

    # All human chromosomes with cross-validation
    python tools/linear_probe_classifier.py \
        --all_human \
        --gtf /path/to/human.gtf \
        --results_dir results/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from results_utils import find_latest_completed, build_run_dir, write_completed
from tools.plot_tsne_by_annotation import load_gtf_features, classify_region
from tools.aggregate_genome_sae_stats import load_maxpooled_vectors

logger = logging.getLogger(__name__)

ALL_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
    "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
    "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]

LABEL_ORDER = ["CDS", "UTR/exon", "Intron", "Intergenic"]


def load_regions_for_chrom(results_dir, chrom, gtf_path):
    """Load SAE vectors and annotations for one chromosome.

    Returns (vectors, labels, metadata) or (None, None, None) on failure.
    """
    sae_run = find_latest_completed(results_dir, chrom, "sae")
    if sae_run is None:
        logger.warning(f"No completed SAE run for {chrom}")
        return None, None, None

    vectors = load_maxpooled_vectors(sae_run)
    if vectors is None:
        logger.warning(f"No maxpooled vectors for {chrom}")
        return None, None, None

    # Load region coordinates
    tsv_path = os.path.join(sae_run, "data", "sae_results.tsv")
    if not os.path.isfile(tsv_path):
        logger.warning(f"No sae_results.tsv for {chrom}")
        return None, None, None

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
            regions.append({
                "region_idx": int(fields[0]),
                "start": int(fields[1]),
                "end": int(fields[2]),
            })

    if len(regions) != vectors.shape[0]:
        logger.warning(
            f"{chrom}: region count mismatch: {len(regions)} regions vs "
            f"{vectors.shape[0]} vectors. Using min of both."
        )
        n = min(len(regions), vectors.shape[0])
        regions = regions[:n]
        vectors = vectors[:n]

    # Classify regions using GTF
    intervals = load_gtf_features(gtf_path, chrom)
    labels = [classify_region(r["start"], r["end"], intervals) for r in regions]

    return vectors, labels, regions


def train_and_evaluate(X_train, y_train, X_test, y_test, label_names=None):
    """Train logistic regression and return metrics dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, balanced_accuracy_score,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_test, y_pred, labels=label_names, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_names)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names if label_names else sorted(set(y_test)),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "classifier": clf,
        "scaler": scaler,
    }


def top_features_per_class(clf, scaler, label_names, top_n=20):
    """Extract top SAE features per class from logistic regression weights."""
    coefs = clf.coef_  # shape: (n_classes, n_features)
    results = {}
    for i, label in enumerate(label_names):
        if i >= coefs.shape[0]:
            continue
        weights = coefs[i]
        top_idx = np.argsort(weights)[::-1][:top_n]
        results[label] = {
            "top_positive_features": top_idx.tolist(),
            "top_positive_weights": weights[top_idx].tolist(),
            "bottom_negative_features": np.argsort(weights)[:top_n].tolist(),
            "bottom_negative_weights": weights[np.argsort(weights)[:top_n]].tolist(),
        }
    return results


def plot_results(metrics, output_dir, title_prefix=""):
    """Generate confusion matrix and per-class F1 bar plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    label_names = metrics["label_names"]

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix}Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Per-class F1
    report = metrics["classification_report"]
    classes = [l for l in label_names if l in report]
    f1s = [report[c]["f1-score"] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, f1s, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{title_prefix}Per-Class F1 Score")
    ax.set_ylim(0, 1.05)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{f1:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_class_f1.png"), dpi=150)
    plt.close(fig)

    # Overall accuracy summary
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_names = ["Accuracy", "Balanced Acc", "F1 (macro)", "F1 (weighted)"]
    metric_vals = [
        metrics["accuracy"], metrics["balanced_accuracy"],
        metrics["f1_macro"], metrics["f1_weighted"],
    ]
    bars = ax.barh(metric_names, metric_vals, color="#607D8B")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11)
    ax.set_title(f"{title_prefix}Overall Metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "overall_metrics.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots saved to {plots_dir}")


def run_single_chrom(args):
    """Train/test with cross-validation on a single chromosome."""
    from sklearn.model_selection import StratifiedKFold

    vectors, labels, _ = load_regions_for_chrom(
        args.results_dir, args.chrom, args.gtf,
    )
    if vectors is None:
        logger.error(f"Failed to load data for {args.chrom}")
        return None

    labels = np.array(labels)
    present_labels = [l for l in LABEL_ORDER if l in set(labels)]
    logger.info(
        f"Loaded {len(labels)} regions for {args.chrom}: "
        + ", ".join(f"{l}={np.sum(labels == l)}" for l in present_labels)
    )

    # Stratified K-fold cross-validation
    n_splits = min(args.cv_folds, min(np.bincount(
        [present_labels.index(l) for l in labels]
    )))
    if n_splits < 2:
        logger.error("Not enough samples per class for cross-validation")
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    label_indices = np.array([present_labels.index(l) for l in labels])

    all_y_true, all_y_pred = [], []
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(vectors, label_indices)):
        X_train, X_test = vectors[train_idx], vectors[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        result = train_and_evaluate(X_train, y_train, X_test, y_test, present_labels)
        fold_accs.append(result["accuracy"])
        all_y_true.extend(y_test)
        all_y_pred.extend(result["classifier"].predict(
            result["scaler"].transform(X_test)
        ))
        logger.info(f"  Fold {fold+1}/{n_splits}: accuracy={result['accuracy']:.4f}")

    # Final evaluation on aggregated predictions
    final = train_and_evaluate(vectors, labels, vectors, labels, present_labels)
    final["cv_fold_accuracies"] = fold_accs
    final["cv_mean_accuracy"] = float(np.mean(fold_accs))
    final["cv_std_accuracy"] = float(np.std(fold_accs))

    return final, present_labels


def run_cross_organism(args):
    """Train on one set of chromosomes, test on another."""
    # Load training data
    train_vecs, train_labels = [], []
    for chrom in args.train_chroms:
        gtf = args.train_gtf or args.gtf
        v, l, _ = load_regions_for_chrom(args.results_dir, chrom, gtf)
        if v is not None:
            train_vecs.append(v)
            train_labels.extend(l)
            logger.info(f"  Train: {chrom} -> {len(l)} regions")

    # Load test data
    test_vecs, test_labels = [], []
    for chrom in args.test_chroms:
        gtf = args.test_gtf or args.gtf
        v, l, _ = load_regions_for_chrom(args.results_dir, chrom, gtf)
        if v is not None:
            test_vecs.append(v)
            test_labels.extend(l)
            logger.info(f"  Test:  {chrom} -> {len(l)} regions")

    if not train_vecs or not test_vecs:
        logger.error("Not enough data for cross-organism evaluation")
        return None

    X_train = np.vstack(train_vecs)
    y_train = np.array(train_labels)
    X_test = np.vstack(test_vecs)
    y_test = np.array(test_labels)

    present_labels = [l for l in LABEL_ORDER if l in set(y_train) | set(y_test)]
    logger.info(
        f"Train: {len(y_train)} regions from {len(args.train_chroms)} chroms, "
        f"Test: {len(y_test)} regions from {len(args.test_chroms)} chroms"
    )

    result = train_and_evaluate(X_train, y_train, X_test, y_test, present_labels)
    return result, present_labels


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe classifier on SAE features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results/",
                        help="Root results directory")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory")

    # Single-chromosome mode
    parser.add_argument("--chrom", default=None,
                        help="Single chromosome for cross-validated evaluation")
    parser.add_argument("--gtf", default=None,
                        help="GTF annotation file")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")

    # All human chromosomes
    parser.add_argument("--all_human", action="store_true",
                        help="Run on all human chromosomes with leave-one-chrom-out CV")

    # Cross-organism mode
    parser.add_argument("--train_chroms", nargs="+", default=None,
                        help="Chromosomes to train on")
    parser.add_argument("--train_gtf", default=None,
                        help="GTF for training chromosomes")
    parser.add_argument("--test_chroms", nargs="+", default=None,
                        help="Chromosomes to test on")
    parser.add_argument("--test_gtf", default=None,
                        help="GTF for test chromosomes")

    parser.add_argument("--top_features", type=int, default=20,
                        help="Number of top features to report per class")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t0 = time.time()

    # Determine mode and run
    if args.train_chroms and args.test_chroms:
        mode = "cross_organism"
        result, present_labels = run_cross_organism(args)
        desc = f"train_{'_'.join(args.train_chroms)}_test_{'_'.join(args.test_chroms)}"
    elif args.all_human:
        mode = "all_human"
        # Leave-one-chromosome-out: train on 23, test on 1
        if not args.gtf:
            parser.error("--gtf required with --all_human")
        all_vecs, all_labels, all_chrom_ids = [], [], []
        for chrom in ALL_HUMAN_CHROMS:
            v, l, _ = load_regions_for_chrom(args.results_dir, chrom, args.gtf)
            if v is not None:
                all_vecs.append(v)
                all_labels.extend(l)
                all_chrom_ids.extend([chrom] * len(l))
                logger.info(f"  {chrom}: {len(l)} regions")

        if not all_vecs:
            logger.error("No human chromosome data found")
            sys.exit(1)

        X = np.vstack(all_vecs)
        y = np.array(all_labels)
        chrom_ids = np.array(all_chrom_ids)
        present_labels = [l for l in LABEL_ORDER if l in set(y)]

        # Leave-one-chromosome-out CV
        unique_chroms = sorted(set(chrom_ids))
        fold_results = []
        for held_out in unique_chroms:
            train_mask = chrom_ids != held_out
            test_mask = chrom_ids == held_out
            r = train_and_evaluate(
                X[train_mask], y[train_mask],
                X[test_mask], y[test_mask],
                present_labels,
            )
            fold_results.append({
                "chrom": held_out,
                "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "f1_macro": r["f1_macro"],
                "n_test": r["n_test"],
            })
            logger.info(f"  LOCO {held_out}: acc={r['accuracy']:.4f} "
                        f"bal_acc={r['balanced_accuracy']:.4f}")

        # Full model for final metrics
        result = train_and_evaluate(X, y, X, y, present_labels)
        result["loco_results"] = fold_results
        result["loco_mean_accuracy"] = float(np.mean([r["accuracy"] for r in fold_results]))
        result["loco_std_accuracy"] = float(np.std([r["accuracy"] for r in fold_results]))
        desc = "all_human_loco"
    elif args.chrom:
        mode = "single_chrom"
        if not args.gtf:
            parser.error("--gtf required with --chrom")
        result, present_labels = run_single_chrom(args)
        desc = args.chrom
    else:
        parser.error("Specify --chrom, --all_human, or --train_chroms/--test_chroms")

    if result is None:
        logger.error("Analysis failed")
        sys.exit(1)

    wall_time = time.time() - t0
    metrics, clf, scaler = result, result.pop("classifier"), result.pop("scaler")

    # Output
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = build_run_dir(
            args.results_dir, "_linear_probe", "analysis", desc,
        )

    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(data_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        f.write("\n")

    # Save top features per class
    feat_info = top_features_per_class(clf, scaler, present_labels, args.top_features)
    with open(os.path.join(data_dir, "top_features_per_class.json"), "w") as f:
        json.dump(feat_info, f, indent=2)
        f.write("\n")

    # Save classifier weights
    np.savez_compressed(
        os.path.join(data_dir, "classifier_weights.npz"),
        coef=clf.coef_,
        intercept=clf.intercept_,
        classes=np.array(present_labels),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    # Generate plots
    plot_results(metrics, out_dir, title_prefix=f"{desc} — ")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Linear Probe Results ({mode})")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    if "cv_mean_accuracy" in metrics:
        logger.info(f"  CV accuracy:       {metrics['cv_mean_accuracy']:.4f} "
                     f"+/- {metrics['cv_std_accuracy']:.4f}")
    if "loco_mean_accuracy" in metrics:
        logger.info(f"  LOCO accuracy:     {metrics['loco_mean_accuracy']:.4f} "
                     f"+/- {metrics['loco_std_accuracy']:.4f}")
    logger.info(f"  Output: {out_dir}")
    logger.info("=" * 60)

    write_completed(out_dir, "linear_probe_classifier.py", wall_time)


if __name__ == "__main__":
    main()
