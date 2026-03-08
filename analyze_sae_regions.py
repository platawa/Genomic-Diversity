#!/usr/bin/env python3
"""
analyze_sae_regions.py

================================================================================
OVERVIEW
================================================================================
Downstream analysis of SAE region fingerprints: max-pooling, cosine similarity,
Leiden clustering, and t-SNE/UMAP embedding visualization.

Takes the pre-computed feature matrices from run_sae_on_chromosome_drops.py
(feature_matrices.npz) and compares all entropy drop regions against each other
in SAE latent space. No GPU required — runs entirely on CPU.

Pipeline:
  1. Load per-region SAE feature matrices (seq_len x 32768 each)
  2. Max-pool across positions → one (32768,) fingerprint per region
  3. Compute pairwise cosine similarity matrix (N x N)
  4. Build kNN graph, run Leiden clustering
  5. Compute t-SNE and/or UMAP 2D embeddings
  6. Generate heatmaps, scatter plots, and cluster summaries

================================================================================
INPUTS
================================================================================
From run_sae_on_chromosome_drops.py:
  - <output_dir>/data/feature_matrices.npz   (per-region SAE feature time series)
  - <output_dir>/data/sae_results.tsv        (region metadata: coords, method, etc.)

================================================================================
OUTPUTS
================================================================================
<output_dir>/latent_analysis/
    data/
        maxpooled_vectors.npy           - (N, 32768) max-pooled fingerprints
        cosine_similarity.npy           - (N, N) similarity matrix
        cluster_assignments.tsv         - Per-region cluster + embedding coords
        cluster_summaries.tsv           - Per-cluster statistics
        analysis_metadata.json          - Run parameters
    plots/
        cosine_similarity_heatmap.png   - N x N heatmap (genomic order)
        cosine_similarity_clustered.png - N x N heatmap (cluster order)
        umap_4panel.png                 - UMAP colored by cluster/method/conf/len
        tsne_4panel.png                 - t-SNE colored by cluster/method/conf/len
        cluster_composition.png         - Method breakdown per cluster

================================================================================
USAGE
================================================================================
    # Basic usage (after running run_sae_on_chromosome_drops.py)
    python analyze_sae_regions.py \\
        --input_dir sae_chromosome_results

    # Custom Leiden resolution and both embeddings
    python analyze_sae_regions.py \\
        --input_dir sae_chromosome_results \\
        --embedding both \\
        --leiden_resolution 0.5

    # Skip clustering, only compute similarity matrix
    python analyze_sae_regions.py \\
        --input_dir sae_chromosome_results \\
        --skip_clustering

================================================================================
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np

# Optional imports — deferred to avoid slow startup
_SCANPY_AVAILABLE = None
_SKLEARN_AVAILABLE = None


def _check_scanpy():
    global _SCANPY_AVAILABLE
    if _SCANPY_AVAILABLE is None:
        try:
            import scanpy  # noqa: F401
            _SCANPY_AVAILABLE = True
        except ImportError:
            _SCANPY_AVAILABLE = False
    return _SCANPY_AVAILABLE


def _check_sklearn():
    global _SKLEARN_AVAILABLE
    if _SKLEARN_AVAILABLE is None:
        try:
            import sklearn  # noqa: F401
            _SKLEARN_AVAILABLE = True
        except ImportError:
            _SKLEARN_AVAILABLE = False
    return _SKLEARN_AVAILABLE


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("sae_latent_analysis")
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

# Detection method colors — matching run_sae_on_chromosome_drops.py
METHOD_COLORS = {'zscore': '#E74C3C', 'mad': '#3498db'}

# SAE dimensions
N_SAE_FEATURES = 32768


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_pool_feature_matrices(
    npz_path: str,
    pool_method: str = "max",
    logger: logging.Logger = None,
) -> Tuple[np.ndarray, int]:
    """
    Load feature matrices from NPZ and pool each region on-the-fly.

    Memory-efficient: loads one region at a time, pools it, then discards
    the full matrix before loading the next. This avoids holding all
    ~10GB of raw matrices in memory simultaneously.

    Args:
        npz_path: Path to feature_matrices.npz (keys: region_0, region_1, ...)
        pool_method: "max" for max-pooling, "mean" for mean-pooling
        logger: Optional logger

    Returns:
        Tuple of:
        - pooled_vectors: (N, 32768) array of pooled fingerprints
        - n_regions: Number of regions loaded
    """
    if logger:
        logger.info(f"Loading feature matrices from {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    keys = sorted(
        [k for k in data.files if k.startswith('region_')],
        key=lambda k: int(k.split('_')[1])
    )

    n_regions = len(keys)
    if n_regions == 0:
        raise ValueError(f"No region_* keys found in {npz_path}")

    pool_fn = np.max if pool_method == "max" else np.mean

    pooled_vectors = np.zeros((n_regions, N_SAE_FEATURES), dtype=np.float32)
    sparsity_stats = []

    for i, key in enumerate(keys):
        feature_ts = data[key]  # shape: (seq_len, 32768)
        pooled = pool_fn(feature_ts, axis=0).astype(np.float32)
        pooled_vectors[i] = pooled

        n_nonzero = np.count_nonzero(pooled)
        sparsity_stats.append(n_nonzero)

        if logger and (i < 3 or i == n_regions - 1):
            logger.debug(
                f"  Region {i}: {feature_ts.shape[0]} positions, "
                f"{n_nonzero}/{N_SAE_FEATURES} nonzero after {pool_method}-pool "
                f"({n_nonzero/N_SAE_FEATURES:.1%})"
            )

    data.close()

    if logger:
        mean_nnz = np.mean(sparsity_stats)
        logger.info(
            f"Loaded {n_regions} regions, {pool_method}-pooled to ({n_regions}, {N_SAE_FEATURES}). "
            f"Mean nonzero: {mean_nnz:.0f}/{N_SAE_FEATURES} ({mean_nnz/N_SAE_FEATURES:.1%})"
        )

    return pooled_vectors, n_regions


def load_region_metadata(
    results_tsv: str,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    Load region metadata from sae_results.tsv.

    Parses the TSV written by save_results() in run_sae_on_chromosome_drops.py.

    Args:
        results_tsv: Path to data/sae_results.tsv
        logger: Optional logger

    Returns:
        List of dicts with keys: region_idx, genomic_start, genomic_end,
        method, confidence
    """
    metadata = []

    with open(results_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Skip header
            if line.startswith('region_idx'):
                continue

            parts = line.split('\t')
            if len(parts) < 5:
                continue

            metadata.append({
                'region_idx': int(parts[0]),
                'genomic_start': int(parts[1]),
                'genomic_end': int(parts[2]),
                'method': parts[3],
                'confidence': float(parts[4]),
                'region_length': int(parts[2]) - int(parts[1]),
            })

    if logger:
        logger.info(f"Loaded metadata for {len(metadata)} regions from {results_tsv}")

    return metadata


# =============================================================================
# MAX-POOLING (standalone, for use with in-memory results)
# =============================================================================

def maxpool_regions(
    feature_matrices: List[np.ndarray],
    pool_method: str = "max",
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Max-pool (or mean-pool) each region's feature matrix across positions.

    For each region with feature_ts of shape (seq_len, 32768), takes
    the element-wise maximum across the seq_len axis, producing a
    single (32768,) fingerprint vector per region.

    Args:
        feature_matrices: List of N arrays, each (seq_len_i, 32768)
        pool_method: "max" or "mean"
        logger: Optional logger

    Returns:
        pooled: np.ndarray of shape (N, 32768), dtype float32
    """
    n_regions = len(feature_matrices)
    if n_regions == 0:
        raise ValueError("No feature matrices provided")

    n_features = feature_matrices[0].shape[1]
    pool_fn = np.max if pool_method == "max" else np.mean

    pooled = np.zeros((n_regions, n_features), dtype=np.float32)
    for i, fm in enumerate(feature_matrices):
        pooled[i] = pool_fn(fm, axis=0).astype(np.float32)

    if logger:
        mean_nnz = np.mean([np.count_nonzero(pooled[i]) for i in range(n_regions)])
        logger.info(
            f"{pool_method.capitalize()}-pooled {n_regions} regions to "
            f"({n_regions}, {n_features}). Mean nonzero: {mean_nnz:.0f}/{n_features}"
        )

    return pooled


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def compute_cosine_similarity(
    pooled_vectors: np.ndarray,
    logger: logging.Logger = None,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all max-pooled region vectors.

    Args:
        pooled_vectors: (N, 32768) array of pooled fingerprints

    Returns:
        similarity_matrix: (N, N) symmetric matrix with values in [0, 1].
                          Diagonal entries are 1.0.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(pooled_vectors).astype(np.float32)

    if logger:
        n = sim_matrix.shape[0]
        # Get upper triangle (excluding diagonal) for stats
        triu_idx = np.triu_indices(n, k=1)
        upper_vals = sim_matrix[triu_idx]
        logger.info(
            f"Cosine similarity matrix: ({n}, {n}). "
            f"Off-diagonal: mean={upper_vals.mean():.4f}, "
            f"std={upper_vals.std():.4f}, "
            f"min={upper_vals.min():.4f}, max={upper_vals.max():.4f}"
        )

    return sim_matrix


# =============================================================================
# COSINE SIMILARITY HEATMAP
# =============================================================================

def plot_cosine_similarity_heatmap(
    similarity_matrix: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    output_path: str,
    order: Optional[np.ndarray] = None,
    title_suffix: str = "",
    logger: logging.Logger = None,
):
    """
    Plot the N x N cosine similarity matrix as a heatmap.

    Args:
        similarity_matrix: (N, N) cosine similarity matrix
        region_metadata: List of region dicts (for method coloring)
        output_path: Path to save PNG
        order: Optional permutation array to reorder rows/columns
        title_suffix: Extra text for title (e.g., " (clustered)")
        logger: Optional logger
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = similarity_matrix.shape[0]

    if order is not None:
        sim = similarity_matrix[np.ix_(order, order)]
        meta_ordered = [region_metadata[i] for i in order]
    else:
        sim = similarity_matrix
        meta_ordered = region_metadata

    methods = [m.get('method', 'unknown') for m in meta_ordered]

    fig_size = max(8, n * 0.12 + 3)
    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(fig_size, fig_size + 0.8),
        gridspec_kw={'height_ratios': [0.04, 1]}, sharex=True,
    )

    # Top bar: method color per region
    bar_colors = [METHOD_COLORS.get(m, '#999999') for m in methods]
    ax_bar.bar(range(n), [1] * n, color=bar_colors, width=1.0)
    ax_bar.set_xlim(-0.5, n - 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_title(
        f'Pairwise Cosine Similarity{title_suffix}',
        fontsize=12, fontweight='bold', pad=10
    )
    legend_patches = [
        Patch(color='#E74C3C', label='zscore'),
        Patch(color='#3498db', label='MAD'),
    ]
    ax_bar.legend(handles=legend_patches, loc='upper right', fontsize=8,
                  ncol=2, framealpha=0.9)

    # Heatmap
    im = ax_heat.imshow(sim, aspect='auto', cmap='RdBu_r',
                        interpolation='nearest', vmin=0, vmax=1)
    ax_heat.set_xlabel('Region', fontsize=10)
    ax_heat.set_ylabel('Region', fontsize=10)

    cbar = plt.colorbar(im, ax=ax_heat, pad=0.02, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved cosine similarity heatmap: {output_path}")


# =============================================================================
# EMBEDDING AND CLUSTERING
# =============================================================================

def compute_embedding_and_clusters(
    pooled_vectors: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    method: str = "both",
    leiden_resolution: float = 1.0,
    n_neighbors: int = 15,
    random_state: int = 42,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """
    Compute 2D embedding and Leiden clusters from max-pooled SAE vectors.

    Uses scanpy if available (UMAP + Leiden + t-SNE), falls back to
    sklearn t-SNE only.

    Args:
        pooled_vectors: (N, 32768) max-pooled fingerprints
        region_metadata: List of region metadata dicts
        method: "umap", "tsne", or "both"
        leiden_resolution: Resolution for Leiden clustering (higher = more clusters)
        n_neighbors: Number of neighbors for kNN graph
        random_state: Random seed
        logger: Optional logger

    Returns:
        Dict with keys:
        - 'embedding_umap': (N, 2) array or None
        - 'embedding_tsne': (N, 2) array or None
        - 'cluster_assignments': (N,) integer array of cluster IDs
        - 'n_clusters': int
    """
    n = pooled_vectors.shape[0]

    if n < 5:
        if logger:
            logger.warning(
                f"Only {n} regions — too few for meaningful embedding/clustering. "
                f"Skipping (need >= 5)."
            )
        return {
            'embedding_umap': None,
            'embedding_tsne': None,
            'cluster_assignments': np.zeros(n, dtype=int),
            'n_clusters': 1,
        }

    # Clamp n_neighbors to at most N-1
    n_neighbors = min(n_neighbors, n - 1)

    if _check_scanpy():
        return _compute_embedding_scanpy(
            pooled_vectors, region_metadata, method,
            leiden_resolution, n_neighbors, random_state, logger
        )
    elif _check_sklearn():
        if logger:
            logger.warning(
                "scanpy not found. Install with: pip install scanpy. "
                "Falling back to sklearn t-SNE only (no UMAP, no Leiden clustering)."
            )
        return _compute_embedding_sklearn(
            pooled_vectors, region_metadata, random_state, logger
        )
    else:
        raise ImportError(
            "Neither scanpy nor scikit-learn found. "
            "Install at least one: pip install scanpy OR pip install scikit-learn"
        )


def _compute_embedding_scanpy(
    pooled_vectors: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    method: str,
    leiden_resolution: float,
    n_neighbors: int,
    random_state: int,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Compute embedding + clustering via scanpy."""
    import scanpy as sc
    import anndata

    if logger:
        logger.info(
            f"Running scanpy pipeline: n_neighbors={n_neighbors}, "
            f"leiden_resolution={leiden_resolution}, embedding={method}"
        )

    # Build AnnData object
    adata = anndata.AnnData(X=pooled_vectors.copy())
    adata.obs['method'] = [m.get('method', 'unknown') for m in region_metadata]
    adata.obs['confidence'] = [m.get('confidence', 0.0) for m in region_metadata]
    adata.obs['genomic_start'] = [m.get('genomic_start', 0) for m in region_metadata]
    adata.obs['genomic_end'] = [m.get('genomic_end', 0) for m in region_metadata]
    adata.obs['region_length'] = [m.get('region_length', 0) for m in region_metadata]

    # Neighbor graph with cosine metric on raw features (no PCA needed)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric='cosine',
                    use_rep='X', random_state=random_state)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=leiden_resolution, random_state=random_state)
    cluster_assignments = adata.obs['leiden'].astype(int).values

    if logger:
        n_clusters = len(np.unique(cluster_assignments))
        logger.info(f"Leiden clustering: {n_clusters} clusters found")

    # Embeddings
    embedding_umap = None
    embedding_tsne = None

    if method in ("umap", "both"):
        sc.tl.umap(adata, random_state=random_state)
        embedding_umap = adata.obsm['X_umap'].copy()
        if logger:
            logger.info("UMAP embedding computed")

    if method in ("tsne", "both"):
        sc.tl.tsne(adata, random_state=random_state, use_rep='X')
        embedding_tsne = adata.obsm['X_tsne'].copy()
        if logger:
            logger.info("t-SNE embedding computed")

    return {
        'embedding_umap': embedding_umap,
        'embedding_tsne': embedding_tsne,
        'cluster_assignments': cluster_assignments,
        'n_clusters': len(np.unique(cluster_assignments)),
    }


def _compute_embedding_sklearn(
    pooled_vectors: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    random_state: int,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Fallback: t-SNE only via sklearn, no Leiden clustering."""
    from sklearn.manifold import TSNE

    if logger:
        logger.info("Running sklearn t-SNE (fallback, no Leiden/UMAP)")

    tsne = TSNE(
        n_components=2,
        metric='cosine',
        random_state=random_state,
        perplexity=min(30, pooled_vectors.shape[0] - 1),
    )
    embedding_tsne = tsne.fit_transform(pooled_vectors)

    if logger:
        logger.info("t-SNE embedding computed")

    return {
        'embedding_umap': None,
        'embedding_tsne': embedding_tsne,
        'cluster_assignments': np.zeros(pooled_vectors.shape[0], dtype=int),
        'n_clusters': 1,
    }


# =============================================================================
# EMBEDDING VISUALIZATION
# =============================================================================

def plot_embedding(
    coordinates: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    cluster_assignments: np.ndarray,
    output_path: str,
    embedding_name: str = "UMAP",
    logger: logging.Logger = None,
):
    """
    4-panel scatter plot of 2D embedding colored by various metadata.

    Panel 1: Leiden cluster (discrete colors)
    Panel 2: Detection method (zscore=red, MAD=blue)
    Panel 3: start_confidence (viridis)
    Panel 4: region_length (plasma)

    Args:
        coordinates: (N, 2) embedding coordinates
        region_metadata: List of region metadata dicts
        cluster_assignments: (N,) cluster IDs
        output_path: Path to save PNG
        embedding_name: "UMAP" or "t-SNE" (for axis labels)
        logger: Optional logger
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(coordinates)
    x, y = coordinates[:, 0], coordinates[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f'{embedding_name} Embedding of Max-Pooled SAE Region Fingerprints (N={n})',
        fontsize=14, fontweight='bold', y=0.98
    )

    # --- Panel 1: Leiden cluster ---
    ax = axes[0, 0]
    unique_clusters = sorted(np.unique(cluster_assignments))
    cmap_discrete = plt.cm.tab10 if len(unique_clusters) <= 10 else plt.cm.tab20
    cluster_colors = {c: cmap_discrete(i % 20) for i, c in enumerate(unique_clusters)}

    for c in unique_clusters:
        mask = cluster_assignments == c
        ax.scatter(x[mask], y[mask], c=[cluster_colors[c]], label=f'Cluster {c}',
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_title('Leiden Cluster', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{embedding_name} 1', fontsize=9)
    ax.set_ylabel(f'{embedding_name} 2', fontsize=9)

    # --- Panel 2: Detection method ---
    ax = axes[0, 1]
    methods = [m.get('method', 'unknown') for m in region_metadata]
    unique_methods = sorted(set(methods))

    for meth in unique_methods:
        mask = np.array([m == meth for m in methods])
        color = METHOD_COLORS.get(meth, '#999999')
        ax.scatter(x[mask], y[mask], c=color, label=meth,
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_title('Detection Method', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{embedding_name} 1', fontsize=9)
    ax.set_ylabel(f'{embedding_name} 2', fontsize=9)

    # --- Panel 3: Confidence (continuous) ---
    ax = axes[1, 0]
    confidences = np.array([m.get('confidence', 0.0) for m in region_metadata])
    sc3 = ax.scatter(x, y, c=confidences, cmap='viridis',
                     s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    plt.colorbar(sc3, ax=ax, shrink=0.8, pad=0.02, label='Confidence')
    ax.set_title('Start Confidence', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{embedding_name} 1', fontsize=9)
    ax.set_ylabel(f'{embedding_name} 2', fontsize=9)

    # --- Panel 4: Region length (continuous) ---
    ax = axes[1, 1]
    lengths = np.array([m.get('region_length', 0) for m in region_metadata])
    sc4 = ax.scatter(x, y, c=lengths, cmap='plasma',
                     s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    plt.colorbar(sc4, ax=ax, shrink=0.8, pad=0.02, label='Region Length (bp)')
    ax.set_title('Region Length', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{embedding_name} 1', fontsize=9)
    ax.set_ylabel(f'{embedding_name} 2', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved {embedding_name} embedding plot: {output_path}")


def plot_cluster_composition(
    cluster_assignments: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    output_path: str,
    logger: logging.Logger = None,
):
    """
    Stacked bar chart showing method breakdown per Leiden cluster.

    Args:
        cluster_assignments: (N,) cluster IDs
        region_metadata: List of region metadata dicts
        output_path: Path to save PNG
        logger: Optional logger
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    unique_clusters = sorted(np.unique(cluster_assignments))
    methods_list = [m.get('method', 'unknown') for m in region_metadata]
    unique_methods = sorted(set(methods_list))

    # Count per cluster per method
    counts = {}
    for c in unique_clusters:
        mask = cluster_assignments == c
        cluster_methods = [methods_list[i] for i in range(len(methods_list)) if mask[i]]
        counts[c] = {meth: cluster_methods.count(meth) for meth in unique_methods}

    fig, ax = plt.subplots(figsize=(max(6, len(unique_clusters) * 1.5), 5))

    bottom = np.zeros(len(unique_clusters))
    for meth in unique_methods:
        vals = [counts[c].get(meth, 0) for c in unique_clusters]
        color = METHOD_COLORS.get(meth, '#999999')
        ax.bar(range(len(unique_clusters)), vals, bottom=bottom,
               color=color, label=meth, width=0.6)
        bottom += np.array(vals, dtype=float)

    ax.set_xticks(range(len(unique_clusters)))
    ax.set_xticklabels([f'Cluster {c}' for c in unique_clusters], fontsize=9)
    ax.set_ylabel('Number of Regions', fontsize=10)
    ax.set_title('Cluster Composition by Detection Method',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved cluster composition plot: {output_path}")


# =============================================================================
# ADDITIONAL LATENT ANALYSIS PLOTS
# =============================================================================

def plot_cluster_feature_profiles(
    cluster_assignments: np.ndarray,
    pooled_vectors: np.ndarray,
    output_path: str,
    top_n: int = 15,
    logger: logging.Logger = None,
):
    """
    Per-cluster bar charts of top SAE features (by mean activation).

    One subplot per cluster showing which features are most active,
    enabling comparison of what each cluster "cares about."
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    unique_clusters = sorted(np.unique(cluster_assignments))
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        if logger:
            logger.info("Skipping cluster feature profiles (only 1 cluster)")
        return

    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle('Top SAE Features per Leiden Cluster', fontsize=14,
                 fontweight='bold', y=1.01)

    for idx, c in enumerate(unique_clusters):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        mask = cluster_assignments == c
        cluster_mean = pooled_vectors[mask].mean(axis=0)
        top_idx = np.argsort(cluster_mean)[::-1][:top_n]

        feat_labels = [f'F{i}' for i in top_idx]
        vals = cluster_mean[top_idx]

        bars = ax.barh(range(len(feat_labels)), vals, color=plt.cm.tab10(idx % 10),
                       alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Activation', fontsize=9)
        ax.set_title(f'Cluster {c} (n={int(mask.sum())})', fontsize=10,
                     fontweight='bold')

    # Hide unused axes
    for idx in range(n_clusters, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved cluster feature profiles: {output_path}")


def plot_genomic_position_by_cluster(
    cluster_assignments: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    output_path: str,
    logger: logging.Logger = None,
):
    """
    Scatter plot showing genomic position (x) vs cluster assignment (y-jittered).

    Reveals whether clusters correspond to specific chromosomal neighborhoods
    or are scattered across the chromosome.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    unique_clusters = sorted(np.unique(cluster_assignments))
    if len(unique_clusters) < 2:
        if logger:
            logger.info("Skipping genomic position plot (only 1 cluster)")
        return

    positions = np.array([
        (m.get('genomic_start', 0) + m.get('genomic_end', 0)) / 2
        for m in region_metadata
    ])
    positions_mb = positions / 1e6

    fig, ax = plt.subplots(figsize=(14, 5))

    cmap = plt.cm.tab10 if len(unique_clusters) <= 10 else plt.cm.tab20
    rng = np.random.RandomState(42)

    for c in unique_clusters:
        mask = cluster_assignments == c
        jitter = rng.uniform(-0.2, 0.2, size=int(mask.sum()))
        ax.scatter(positions_mb[mask], c + jitter, c=[cmap(c % 20)],
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.5,
                   label=f'Cluster {c} (n={int(mask.sum())})')

    ax.set_yticks(unique_clusters)
    ax.set_yticklabels([f'Cluster {c}' for c in unique_clusters])
    ax.set_xlabel('Genomic Position (Mb)', fontsize=11)
    ax.set_title('Region Genomic Position by Leiden Cluster',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved genomic position by cluster: {output_path}")


def plot_cluster_feature_heatmap(
    cluster_assignments: np.ndarray,
    pooled_vectors: np.ndarray,
    output_path: str,
    top_n: int = 30,
    logger: logging.Logger = None,
):
    """
    Heatmap of mean activation per feature (columns) per cluster (rows).

    Uses the union of top features across all clusters to show
    which features distinguish clusters from each other.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    unique_clusters = sorted(np.unique(cluster_assignments))
    if len(unique_clusters) < 2:
        if logger:
            logger.info("Skipping cluster feature heatmap (only 1 cluster)")
        return

    # Collect top features per cluster, take union
    per_cluster_top = set()
    cluster_means = {}
    for c in unique_clusters:
        mask = cluster_assignments == c
        mean_act = pooled_vectors[mask].mean(axis=0)
        cluster_means[c] = mean_act
        top_idx = np.argsort(mean_act)[::-1][:top_n]
        per_cluster_top.update(top_idx.tolist())

    # Sort by overall mean activation
    overall_mean = pooled_vectors.mean(axis=0)
    feat_list = sorted(per_cluster_top, key=lambda f: overall_mean[f], reverse=True)
    # Limit to reasonable number
    feat_list = feat_list[:min(len(feat_list), 50)]

    # Build matrix: clusters × features
    matrix = np.zeros((len(unique_clusters), len(feat_list)), dtype=np.float32)
    for i, c in enumerate(unique_clusters):
        for j, f in enumerate(feat_list):
            matrix[i, j] = cluster_means[c][f]

    fig_width = max(12, len(feat_list) * 0.3 + 3)
    fig_height = max(4, len(unique_clusters) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(len(unique_clusters)))
    ax.set_yticklabels([f'Cluster {c}' for c in unique_clusters], fontsize=9)
    ax.set_xticks(range(len(feat_list)))
    ax.set_xticklabels([f'F{f}' for f in feat_list], fontsize=6, rotation=90)
    ax.set_xlabel('SAE Feature', fontsize=10)
    ax.set_title('Mean Feature Activation per Cluster',
                 fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Mean Activation', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved cluster feature heatmap: {output_path}")


def plot_region_size_distribution(
    cluster_assignments: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    output_path: str,
    logger: logging.Logger = None,
):
    """
    Box plot of region sizes per Leiden cluster.

    Shows whether certain clusters preferentially capture
    shorter or longer entropy-drop regions.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    unique_clusters = sorted(np.unique(cluster_assignments))
    if len(unique_clusters) < 2:
        if logger:
            logger.info("Skipping region size distribution (only 1 cluster)")
        return

    lengths_by_cluster = []
    labels = []
    for c in unique_clusters:
        mask = cluster_assignments == c
        cluster_lengths = [
            region_metadata[i].get('region_length', 0)
            for i in range(len(region_metadata)) if mask[i]
        ]
        lengths_by_cluster.append(cluster_lengths)
        labels.append(f'Cluster {c}\n(n={int(mask.sum())})')

    fig, ax = plt.subplots(figsize=(max(6, len(unique_clusters) * 1.5), 5))

    cmap = plt.cm.tab10 if len(unique_clusters) <= 10 else plt.cm.tab20
    bp = ax.boxplot(lengths_by_cluster, labels=labels, patch_artist=True,
                    widths=0.6, showfliers=True)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i % 20))
        patch.set_alpha(0.7)

    ax.set_ylabel('Region Length (bp)', fontsize=10)
    ax.set_title('Region Size Distribution by Leiden Cluster',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if logger:
        logger.info(f"Saved region size distribution: {output_path}")


# =============================================================================
# CLUSTER SUMMARY STATISTICS
# =============================================================================

def summarize_clusters(
    cluster_assignments: np.ndarray,
    region_metadata: List[Dict[str, Any]],
    pooled_vectors: np.ndarray,
    similarity_matrix: np.ndarray,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    Compute summary statistics for each Leiden cluster.

    For each cluster:
    - Number of regions
    - Method breakdown (zscore vs MAD counts)
    - Mean/std of start_confidence
    - Mean intra-cluster cosine similarity
    - Top distinguishing SAE features (highest mean activation in cluster)

    Args:
        cluster_assignments: (N,) cluster IDs
        region_metadata: List of region metadata dicts
        pooled_vectors: (N, 32768) max-pooled fingerprints
        similarity_matrix: (N, N) cosine similarity matrix
        logger: Optional logger

    Returns:
        List of cluster summary dicts, sorted by cluster ID
    """
    unique_clusters = sorted(np.unique(cluster_assignments))
    summaries = []

    for c in unique_clusters:
        mask = cluster_assignments == c
        indices = np.where(mask)[0]
        n_regions = len(indices)

        # Method breakdown
        methods = [region_metadata[i].get('method', 'unknown') for i in indices]
        zscore_count = methods.count('zscore')
        mad_count = methods.count('mad')

        # Confidence stats
        confs = np.array([region_metadata[i].get('confidence', 0.0) for i in indices])

        # Intra-cluster similarity
        if n_regions > 1:
            cluster_sim = similarity_matrix[np.ix_(indices, indices)]
            triu_idx = np.triu_indices(n_regions, k=1)
            intra_sim = float(np.mean(cluster_sim[triu_idx]))
        else:
            intra_sim = 1.0

        # Top distinguishing features: highest mean activation in this cluster
        cluster_pooled = pooled_vectors[indices]
        mean_activation = cluster_pooled.mean(axis=0)
        top_feat_idx = np.argsort(mean_activation)[::-1][:20]
        top_features = [
            (int(idx), float(mean_activation[idx]))
            for idx in top_feat_idx
            if mean_activation[idx] > 0
        ]

        summaries.append({
            'cluster_id': int(c),
            'n_regions': n_regions,
            'zscore_count': zscore_count,
            'mad_count': mad_count,
            'mean_confidence': float(np.mean(confs)) if n_regions > 0 else 0.0,
            'std_confidence': float(np.std(confs)) if n_regions > 1 else 0.0,
            'mean_intra_similarity': intra_sim,
            'top_features': top_features,
        })

    if logger:
        for s in summaries:
            logger.info(
                f"  Cluster {s['cluster_id']}: {s['n_regions']} regions "
                f"(zscore={s['zscore_count']}, MAD={s['mad_count']}), "
                f"intra-sim={s['mean_intra_similarity']:.4f}, "
                f"mean_conf={s['mean_confidence']:.2f}"
            )

    return summaries


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_analysis_results(
    pooled_vectors: np.ndarray,
    similarity_matrix: np.ndarray,
    embedding_results: Dict[str, Any],
    cluster_summaries: List[Dict[str, Any]],
    region_metadata: List[Dict[str, Any]],
    output_dir: str,
    args_dict: Dict[str, Any] = None,
    logger: logging.Logger = None,
):
    """
    Save all analysis outputs to the output directory.

    Args:
        pooled_vectors: (N, 32768) max-pooled fingerprints
        similarity_matrix: (N, N) cosine similarity matrix
        embedding_results: Dict from compute_embedding_and_clusters
        cluster_summaries: List from summarize_clusters
        region_metadata: Region metadata dicts
        output_dir: Directory for outputs
        args_dict: CLI arguments for metadata
        logger: Optional logger
    """
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # --- Max-pooled vectors ---
    np.save(os.path.join(data_dir, 'maxpooled_vectors.npy'), pooled_vectors)

    # --- Cosine similarity matrix ---
    np.save(os.path.join(data_dir, 'cosine_similarity.npy'), similarity_matrix)

    # --- Cluster assignments TSV ---
    clusters = embedding_results['cluster_assignments']
    umap_coords = embedding_results.get('embedding_umap')
    tsne_coords = embedding_results.get('embedding_tsne')

    assign_file = os.path.join(data_dir, 'cluster_assignments.tsv')
    with open(assign_file, 'w') as f:
        f.write("# Latent analysis: cluster assignments and embedding coordinates\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# N regions: {len(region_metadata)}\n")
        f.write(f"# N clusters: {embedding_results['n_clusters']}\n")
        f.write("#\n")

        header_parts = [
            "region_idx", "genomic_start", "genomic_end",
            "method", "confidence", "region_length", "cluster_id"
        ]
        if umap_coords is not None:
            header_parts.extend(["umap_1", "umap_2"])
        if tsne_coords is not None:
            header_parts.extend(["tsne_1", "tsne_2"])
        f.write('\t'.join(header_parts) + '\n')

        for i, meta in enumerate(region_metadata):
            row = [
                str(meta.get('region_idx', i)),
                str(meta.get('genomic_start', 0)),
                str(meta.get('genomic_end', 0)),
                meta.get('method', 'unknown'),
                f"{meta.get('confidence', 0.0):.4f}",
                str(meta.get('region_length', 0)),
                str(int(clusters[i])),
            ]
            if umap_coords is not None:
                row.extend([f"{umap_coords[i, 0]:.6f}", f"{umap_coords[i, 1]:.6f}"])
            if tsne_coords is not None:
                row.extend([f"{tsne_coords[i, 0]:.6f}", f"{tsne_coords[i, 1]:.6f}"])
            f.write('\t'.join(row) + '\n')

    # --- Cluster summaries TSV ---
    summary_file = os.path.join(data_dir, 'cluster_summaries.tsv')
    with open(summary_file, 'w') as f:
        f.write("# Cluster summary statistics\n")
        f.write("#\n")
        f.write("cluster_id\tn_regions\tzscore_count\tmad_count\t"
                "mean_confidence\tstd_confidence\tmean_intra_similarity\t"
                "top_features\n")

        for s in cluster_summaries:
            top_str = ','.join(
                f"{fid}:{act:.2f}" for fid, act in s['top_features'][:10]
            )
            f.write(
                f"{s['cluster_id']}\t{s['n_regions']}\t"
                f"{s['zscore_count']}\t{s['mad_count']}\t"
                f"{s['mean_confidence']:.4f}\t{s['std_confidence']:.4f}\t"
                f"{s['mean_intra_similarity']:.4f}\t{top_str}\n"
            )

    # --- Metadata JSON ---
    meta_json = {
        'generated': datetime.now().isoformat(),
        'n_regions': len(region_metadata),
        'n_clusters': embedding_results['n_clusters'],
        'has_umap': umap_coords is not None,
        'has_tsne': tsne_coords is not None,
        'pooled_shape': list(pooled_vectors.shape),
        'similarity_shape': list(similarity_matrix.shape),
    }
    if args_dict:
        meta_json['parameters'] = {
            k: v for k, v in args_dict.items()
            if isinstance(v, (str, int, float, bool))
        }

    with open(os.path.join(data_dir, 'analysis_metadata.json'), 'w') as f:
        json.dump(meta_json, f, indent=2)

    if logger:
        logger.info(f"Saved analysis outputs to {data_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE region fingerprints: max-pooling, cosine similarity, clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on pre-computed SAE results
  python analyze_sae_regions.py \\
      --input_dir sae_chromosome_results

  # Custom Leiden resolution and UMAP only
  python analyze_sae_regions.py \\
      --input_dir sae_chromosome_results \\
      --leiden_resolution 0.5 \\
      --embedding umap

  # Only compute similarity matrix (skip clustering)
  python analyze_sae_regions.py \\
      --input_dir sae_chromosome_results \\
      --skip_clustering

  # Use mean-pooling instead of max-pooling
  python analyze_sae_regions.py \\
      --input_dir sae_chromosome_results \\
      --pool_method mean
        """
    )

    parser.add_argument("--input_dir", required=True,
                        help="Path to output dir from run_sae_on_chromosome_drops.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: <input_dir>/latent_analysis/)")
    parser.add_argument("--embedding", type=str, default="both",
                        choices=["umap", "tsne", "both"],
                        help="Embedding method (default: both)")
    parser.add_argument("--leiden_resolution", type=float, default=1.0,
                        help="Leiden clustering resolution (default: 1.0)")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="Number of neighbors for kNN graph (default: 15)")
    parser.add_argument("--pool_method", type=str, default="max",
                        choices=["max", "mean"],
                        help="Pooling method across positions (default: max)")
    parser.add_argument("--skip_clustering", action="store_true",
                        help="Skip Leiden clustering and embedding")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or os.path.join(input_dir, 'latent_analysis')

    logger.info("=" * 70)
    logger.info("SAE Region Latent Analysis")
    logger.info("=" * 70)
    logger.info(f"Input dir:  {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    # Validate input files exist
    npz_path = os.path.join(input_dir, 'data', 'feature_matrices.npz')
    tsv_path = os.path.join(input_dir, 'data', 'sae_results.tsv')

    if not os.path.exists(npz_path):
        logger.error(f"Feature matrices not found: {npz_path}")
        sys.exit(1)
    if not os.path.exists(tsv_path):
        logger.error(f"SAE results not found: {tsv_path}")
        sys.exit(1)

    # Create output dirs
    data_dir = os.path.join(output_dir, 'data')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: Load and pool feature matrices
    # =========================================================================
    logger.info("")
    logger.info("STEP 1: Loading and pooling feature matrices")
    logger.info("-" * 50)

    pooled_vectors, n_regions = load_and_pool_feature_matrices(
        npz_path, pool_method=args.pool_method, logger=logger
    )

    # =========================================================================
    # STEP 2: Load region metadata
    # =========================================================================
    logger.info("")
    logger.info("STEP 2: Loading region metadata")
    logger.info("-" * 50)

    region_metadata = load_region_metadata(tsv_path, logger=logger)

    # Validate alignment
    if len(region_metadata) != n_regions:
        logger.warning(
            f"Metadata count ({len(region_metadata)}) != region count ({n_regions}). "
            f"Using min of both."
        )
        n = min(len(region_metadata), n_regions)
        region_metadata = region_metadata[:n]
        pooled_vectors = pooled_vectors[:n]

    # Check for zero vectors
    zero_mask = np.all(pooled_vectors == 0, axis=1)
    if np.any(zero_mask):
        n_zero = np.sum(zero_mask)
        logger.warning(
            f"{n_zero} regions have all-zero pooled vectors (no SAE activations). "
            f"These will be excluded from similarity/embedding analysis."
        )
        keep = ~zero_mask
        pooled_vectors = pooled_vectors[keep]
        region_metadata = [m for m, k in zip(region_metadata, keep) if k]
        n_regions = len(region_metadata)

    # =========================================================================
    # STEP 3: Compute cosine similarity
    # =========================================================================
    logger.info("")
    logger.info("STEP 3: Computing pairwise cosine similarity")
    logger.info("-" * 50)

    similarity_matrix = compute_cosine_similarity(pooled_vectors, logger=logger)

    # =========================================================================
    # STEP 4: Plot cosine similarity heatmap (genomic order)
    # =========================================================================
    logger.info("")
    logger.info("STEP 4: Plotting cosine similarity heatmap")
    logger.info("-" * 50)

    # Sort by genomic position for the default heatmap
    genomic_order = np.argsort([m.get('genomic_start', 0) for m in region_metadata])
    plot_cosine_similarity_heatmap(
        similarity_matrix, region_metadata,
        os.path.join(plots_dir, 'cosine_similarity_heatmap.png'),
        order=genomic_order,
        title_suffix=" (genomic order)",
        logger=logger,
    )

    # =========================================================================
    # STEP 5: Embedding and clustering
    # =========================================================================
    if not args.skip_clustering:
        logger.info("")
        logger.info("STEP 5: Computing embedding and Leiden clustering")
        logger.info("-" * 50)

        embedding_results = compute_embedding_and_clusters(
            pooled_vectors, region_metadata,
            method=args.embedding,
            leiden_resolution=args.leiden_resolution,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
            logger=logger,
        )

        clusters = embedding_results['cluster_assignments']

        # --- Clustered cosine similarity heatmap ---
        if embedding_results['n_clusters'] > 1:
            cluster_order = np.argsort(clusters)
            plot_cosine_similarity_heatmap(
                similarity_matrix, region_metadata,
                os.path.join(plots_dir, 'cosine_similarity_clustered.png'),
                order=cluster_order,
                title_suffix=" (clustered)",
                logger=logger,
            )

        # --- Embedding scatter plots ---
        if embedding_results['embedding_umap'] is not None:
            plot_embedding(
                embedding_results['embedding_umap'],
                region_metadata, clusters,
                os.path.join(plots_dir, 'umap_4panel.png'),
                embedding_name="UMAP",
                logger=logger,
            )

        if embedding_results['embedding_tsne'] is not None:
            plot_embedding(
                embedding_results['embedding_tsne'],
                region_metadata, clusters,
                os.path.join(plots_dir, 'tsne_4panel.png'),
                embedding_name="t-SNE",
                logger=logger,
            )

        # --- Cluster composition ---
        if embedding_results['n_clusters'] > 1:
            plot_cluster_composition(
                clusters, region_metadata,
                os.path.join(plots_dir, 'cluster_composition.png'),
                logger=logger,
            )

        # --- Additional latent analysis plots ---
        if embedding_results['n_clusters'] > 1:
            logger.info("")
            logger.info("Generating additional latent analysis plots")
            logger.info("-" * 50)

            plot_cluster_feature_profiles(
                clusters, pooled_vectors,
                os.path.join(plots_dir, 'cluster_feature_profiles.png'),
                logger=logger,
            )

            plot_genomic_position_by_cluster(
                clusters, region_metadata,
                os.path.join(plots_dir, 'genomic_position_by_cluster.png'),
                logger=logger,
            )

            plot_cluster_feature_heatmap(
                clusters, pooled_vectors,
                os.path.join(plots_dir, 'cluster_feature_heatmap.png'),
                logger=logger,
            )

            plot_region_size_distribution(
                clusters, region_metadata,
                os.path.join(plots_dir, 'region_size_distribution.png'),
                logger=logger,
            )

        # =====================================================================
        # STEP 6: Cluster summary statistics
        # =====================================================================
        logger.info("")
        logger.info("STEP 6: Computing cluster summaries")
        logger.info("-" * 50)

        cluster_summaries = summarize_clusters(
            clusters, region_metadata, pooled_vectors, similarity_matrix,
            logger=logger,
        )
    else:
        logger.info("")
        logger.info("STEP 5: Skipping clustering (--skip_clustering)")
        embedding_results = {
            'embedding_umap': None,
            'embedding_tsne': None,
            'cluster_assignments': np.zeros(n_regions, dtype=int),
            'n_clusters': 1,
        }
        cluster_summaries = []

    # =========================================================================
    # STEP 7: Save all outputs
    # =========================================================================
    logger.info("")
    logger.info("STEP 7: Saving outputs")
    logger.info("-" * 50)

    save_analysis_results(
        pooled_vectors, similarity_matrix, embedding_results,
        cluster_summaries, region_metadata, output_dir,
        args_dict=vars(args), logger=logger,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"Regions analyzed:  {n_regions}")
    logger.info(f"Clusters found:    {embedding_results['n_clusters']}")
    logger.info(f"Output directory:  {output_dir}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  data/maxpooled_vectors.npy           ({pooled_vectors.shape})")
    logger.info(f"  data/cosine_similarity.npy           ({similarity_matrix.shape})")
    logger.info(f"  data/cluster_assignments.tsv")
    logger.info(f"  data/cluster_summaries.tsv")
    logger.info(f"  data/analysis_metadata.json")
    logger.info(f"  plots/cosine_similarity_heatmap.png")
    if not args.skip_clustering and embedding_results['n_clusters'] > 1:
        logger.info(f"  plots/cosine_similarity_clustered.png")
    if embedding_results.get('embedding_umap') is not None:
        logger.info(f"  plots/umap_4panel.png")
    if embedding_results.get('embedding_tsne') is not None:
        logger.info(f"  plots/tsne_4panel.png")
    if not args.skip_clustering and embedding_results['n_clusters'] > 1:
        logger.info(f"  plots/cluster_composition.png")
        logger.info(f"  plots/cluster_feature_profiles.png")
        logger.info(f"  plots/genomic_position_by_cluster.png")
        logger.info(f"  plots/cluster_feature_heatmap.png")
        logger.info(f"  plots/region_size_distribution.png")


if __name__ == "__main__":
    main()
