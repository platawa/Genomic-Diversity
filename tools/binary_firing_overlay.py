#!/usr/bin/env python3
"""
binary_firing_overlay.py

For a given scope (chrom or genome_wide):
  1. Load maxpooled_vectors.npy + embedding_tsne.npy + embedding_umap.npy
  2. Compute binary matrix[r, f] = (|v[r, f]| >= threshold * max_per_feature)
     at thresholds in {1%, 5%, 10%}
  3. Emit:
     - per-threshold binary matrix as .npz (regions × features, int8)
     - per-feature t-SNE+UMAP overlays for the top-K most-firing features
     - a summary heatmap: region × feature (sorted by firing count)

The binary matrices are what annotation-by-inference downstream tools consume.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10)


def load_bundle(scope_dir):
    paths = {
        "maxpool": os.path.join(scope_dir, "data", "maxpooled_vectors.npy"),
        "tsne": os.path.join(scope_dir, "data", "embedding_tsne.npy"),
        "umap": os.path.join(scope_dir, "data", "embedding_umap.npy"),
        "ca": os.path.join(scope_dir, "data", "cluster_assignments.tsv"),
    }
    bundle = {}
    for k, p in paths.items():
        if k == "ca":
            bundle[k] = pd.read_csv(p, sep="\t", comment="#") if os.path.isfile(p) else None
        else:
            bundle[k] = np.load(p) if os.path.isfile(p) else None
    return bundle


def compute_binary(vectors, thresholds):
    """Binarize per feature: threshold * (max activation for that feature)."""
    max_per_feat = np.abs(vectors).max(axis=0)
    max_per_feat[max_per_feat == 0] = 1.0  # avoid div-by-zero
    normalized = np.abs(vectors) / max_per_feat[None, :]
    binaries = {}
    for t in thresholds:
        binaries[t] = (normalized >= t).astype(np.int8)
    return binaries


def plot_feature_overlay(embedding, binary_col, feature_idx, out_path, emb_name="tsne"):
    fig, ax = plt.subplots(figsize=(8, 7))
    fired = binary_col.astype(bool)
    if embedding.shape[0] != binary_col.shape[0]:
        logger.warning(f"Embedding/binary shape mismatch: {embedding.shape[0]} vs {binary_col.shape[0]}")
        return
    ax.scatter(embedding[~fired, 0], embedding[~fired, 1], s=2, c="lightgray", alpha=0.4)
    ax.scatter(embedding[fired, 0], embedding[fired, 1], s=4, c="red", alpha=0.8)
    ax.set_title(f"Feature {feature_idx} firing ({fired.sum()} / {len(fired)} regions)")
    ax.set_xlabel(f"{emb_name}_1")
    ax.set_ylabel(f"{emb_name}_2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scope_dir", required=True,
                   help="Path containing data/{maxpooled_vectors.npy,embedding_*.npy}")
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--top_k_features", type=int, default=10,
                   help="Produce overlays for top-K features by firing count")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bundle = load_bundle(args.scope_dir)
    if bundle["maxpool"] is None:
        logger.error("maxpooled_vectors.npy missing")
        return 2

    out_dir = args.output_dir or os.path.join(args.scope_dir, "binary_firing")
    os.makedirs(out_dir, exist_ok=True)

    t0 = __import__("time").time()
    binaries = compute_binary(bundle["maxpool"], args.thresholds)
    for t, B in binaries.items():
        np.savez_compressed(os.path.join(out_dir, f"binary_t{int(t*100)}pct.npz"), binary=B)
        logger.info(f"threshold {t*100:.1f}% : {B.sum():,} firings across "
                    f"{B.shape[0]} regions × {B.shape[1]} features")

    # Per-feature overlays at median threshold (5%)
    t_mid = args.thresholds[len(args.thresholds) // 2]
    B_mid = binaries[t_mid]
    feature_firing = B_mid.sum(axis=0)
    top_feats = np.argsort(-feature_firing)[:args.top_k_features]

    for emb_name in ["tsne", "umap"]:
        emb = bundle[emb_name]
        if emb is None:
            continue
        emb_dir = os.path.join(out_dir, f"{emb_name}_overlays_t{int(t_mid*100)}pct")
        os.makedirs(emb_dir, exist_ok=True)
        for fi in top_feats:
            plot_feature_overlay(
                emb, B_mid[:, fi], int(fi),
                os.path.join(emb_dir, f"feat_{int(fi):05d}.png"),
                emb_name=emb_name,
            )
        logger.info(f"Wrote {len(top_feats)} {emb_name} overlays to {emb_dir}")

    # Summary TSV
    summary = pd.DataFrame({
        "feature_idx": np.arange(bundle["maxpool"].shape[1]),
        **{f"firing_count_t{int(t*100)}pct": binaries[t].sum(axis=0) for t in args.thresholds},
    })
    summary.to_csv(os.path.join(out_dir, "feature_firing_summary.tsv"),
                   sep="\t", index=False)
    write_completed(out_dir, "binary_firing_overlay.py",
                    __import__("time").time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
