#!/usr/bin/env python3
"""
feature_linear_probes.py

For each SAE feature, fit a univariate logistic regression predicting an
annotation class (CDS vs intergenic, or any configurable label column)
from the feature's max-pooled activation. Reports per-feature AUC and the
top-N most discriminatory features per label.

Output:
  - feature_auc.tsv                — feature_idx × label × AUC × coef × n_pos × n_neg
  - top_discriminatory_features.tsv — top-N per label
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import write_completed

logger = logging.getLogger(__name__)


def load_bundle(scope_dir, label_column="annotation"):
    mp = np.load(os.path.join(scope_dir, "data", "maxpooled_vectors.npy"))
    ca = pd.read_csv(os.path.join(scope_dir, "data", "cluster_assignments.tsv"), sep="\t", comment="#")
    if label_column not in ca.columns:
        # Accept a merged file from cluster_top_genes.py outputs
        annot_path = os.path.join(scope_dir, "cluster_top_genes",
                                  "cluster_annotation_counts.tsv")
        if os.path.isfile(annot_path):
            raise RuntimeError(
                f"Region-level annotation column '{label_column}' not in cluster_assignments.tsv. "
                "Run cluster_top_genes.py first with --output_dir so annotation is saved per-region, "
                "or pass --labels_tsv to supply region-level labels."
            )
        raise ValueError(f"Column '{label_column}' not in {ca.columns.tolist()}")
    return mp, ca[label_column].values


def fit_per_feature(vectors, binary_labels):
    """Return AUC + logistic coefficient per feature (univariate)."""
    n_feat = vectors.shape[1]
    aucs = np.full(n_feat, np.nan)
    coefs = np.full(n_feat, np.nan)
    for f in range(n_feat):
        x = vectors[:, f]
        if x.std() == 0:
            continue
        try:
            auc = roc_auc_score(binary_labels, x)
        except ValueError:
            continue
        aucs[f] = auc
        # Fit logreg for coefficient sign
        try:
            lr = LogisticRegression(max_iter=200, solver="liblinear")
            lr.fit(x.reshape(-1, 1), binary_labels)
            coefs[f] = lr.coef_[0, 0]
        except Exception:
            pass
    return aucs, coefs


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scope_dir", required=True,
                   help="Dir containing data/{maxpooled_vectors.npy, cluster_assignments.tsv}")
    p.add_argument("--label_column", default="annotation",
                   help="Column in cluster_assignments.tsv to probe")
    p.add_argument("--labels_to_probe", nargs="+", default=None,
                   help="Subset of label values to create one-vs-rest probes for; "
                        "default: all label values")
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mp, labels = load_bundle(args.scope_dir, args.label_column)
    unique_labels = args.labels_to_probe or sorted(pd.unique(labels))
    out_dir = args.output_dir or os.path.join(args.scope_dir, "feature_linear_probes")
    os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    all_rows = []
    for label in unique_labels:
        y = (labels == label).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            logger.warning(f"label={label}: degenerate distribution; skipping")
            continue
        logger.info(f"probing label={label}: n_pos={int(y.sum())}, n_neg={int(len(y)-y.sum())}")
        aucs, coefs = fit_per_feature(mp, y)
        df = pd.DataFrame({
            "feature_idx": np.arange(mp.shape[1]),
            "label": label,
            "auc": aucs,
            "coef": coefs,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
        })
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    full.to_csv(os.path.join(out_dir, "feature_auc.tsv"), sep="\t", index=False)

    if not full.empty:
        top = (full.dropna(subset=["auc"])
               .assign(auc_abs=lambda d: (d["auc"] - 0.5).abs())
               .sort_values(["label", "auc_abs"], ascending=[True, False])
               .groupby("label").head(args.top_n))
        top.to_csv(os.path.join(out_dir, "top_discriminatory_features.tsv"),
                   sep="\t", index=False)

    write_completed(out_dir, "feature_linear_probes.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
