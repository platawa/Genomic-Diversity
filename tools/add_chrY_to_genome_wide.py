#!/usr/bin/env python3
"""Add chrY to the existing genome-wide UMAP/t-SNE embeddings.

Appends chrY's 94 maxpooled vectors to the cached 740K combined array,
recomputes UMAP and t-SNE, and regenerates all plots via genome_replot_from_tsv.py.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True, help="Path to _cache/ directory")
    parser.add_argument("--chrY_latent", required=True, help="Path to chrY latent_analysis/data/")
    parser.add_argument("--output_dir", required=True, help="Output directory for new run")
    parser.add_argument("--n_pca", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()

    # Load existing cache
    logger.info("Loading cached genome-wide data...")
    combined = np.load(os.path.join(args.cache_dir, "combined_maxpooled_raw.npy"))
    with open(os.path.join(args.cache_dir, "combined_metadata_raw.json")) as f:
        metadata = json.load(f)
    logger.info(f"  Existing: {combined.shape[0]} regions, {len(set(m['chrom'] for m in metadata))} chroms")

    # Check if chrY already included
    existing_chroms = set(m["chrom"] for m in metadata)
    if "chrY" in existing_chroms:
        logger.info("chrY already in genome-wide data. Skipping append.")
    else:
        # Load chrY data
        chrY_vectors = np.load(os.path.join(args.chrY_latent, "maxpooled_vectors.npy"))
        chrY_ca = pd.read_csv(os.path.join(args.chrY_latent, "cluster_assignments.tsv"),
                              sep="\t", comment="#")
        logger.info(f"  chrY: {len(chrY_ca)} regions")

        # Build metadata for chrY
        chrY_meta = []
        for _, row in chrY_ca.iterrows():
            chrY_meta.append({
                "chrom": "chrY",
                "genomic_start": int(row["genomic_start"]),
                "genomic_end": int(row["genomic_end"]),
                "method": row.get("method", "unknown"),
                "confidence": float(row.get("confidence", 0)),
                "annotation": row.get("annotation", "unknown"),
            })

        # Append
        combined = np.vstack([combined, chrY_vectors])
        metadata = metadata + chrY_meta
        logger.info(f"  Combined: {combined.shape[0]} regions, {len(set(m['chrom'] for m in metadata))} chroms")

    # Compute new embeddings
    logger.info("Computing PCA...")
    try:
        import scanpy as sc
    except ImportError:
        logger.error("scanpy required. pip install scanpy")
        sys.exit(1)

    adata = sc.AnnData(X=combined.astype(np.float32))
    sc.pp.pca(adata, n_comps=args.n_pca)
    pca_vectors = adata.obsm["X_pca"]
    logger.info(f"  PCA: {pca_vectors.shape}")

    logger.info("Computing neighbors + UMAP...")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(adata, random_state=42)
    umap_coords = adata.obsm["X_umap"].copy()
    logger.info(f"  UMAP: {umap_coords.shape}")

    logger.info("Computing t-SNE...")
    sc.tl.tsne(adata, use_rep="X_pca", random_state=42)
    tsne_coords = adata.obsm["X_tsne"].copy()
    logger.info(f"  t-SNE: {tsne_coords.shape}")

    logger.info("Computing Leiden clustering...")
    sc.tl.leiden(adata, resolution=1.0, random_state=42)
    clusters = adata.obs["leiden"].astype(int).values

    # Save outputs
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # Save embeddings to cache for future use
    np.save(os.path.join(args.cache_dir, "embedding_umap_raw.npy"), umap_coords)
    np.save(os.path.join(args.cache_dir, "embedding_tsne_raw.npy"), tsne_coords)
    np.save(os.path.join(args.cache_dir, "combined_maxpooled_raw.npy"), combined)
    with open(os.path.join(args.cache_dir, "combined_metadata_raw.json"), "w") as f:
        json.dump(metadata, f)
    np.save(os.path.join(args.cache_dir, "cluster_assignments_raw.npy"), clusters)
    logger.info("Updated cache files.")

    # Build TSV for replotting
    df = pd.DataFrame(metadata)
    df["cluster"] = clusters
    df["tsne_1"] = tsne_coords[:, 0]
    df["tsne_2"] = tsne_coords[:, 1]
    df["umap_1"] = umap_coords[:, 0]
    df["umap_2"] = umap_coords[:, 1]
    tsv_path = os.path.join(args.output_dir, "data", "cluster_assignments.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info(f"Saved TSV: {tsv_path}")

    wall = time.time() - t0
    logger.info(f"Done in {wall:.1f}s. Now run genome_replot_from_tsv.py on {tsv_path}")


if __name__ == "__main__":
    main()
