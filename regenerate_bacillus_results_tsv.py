#!/usr/bin/env python3
"""
Regenerate sae_results.tsv for Bacillus (NC_000964.3) from merged feature matrices
and checkpoint metadata. The merge_sae_shards.py created the feature_matrices.npz
but didn't generate the results TSV.

This script:
1. Loads checkpoint_meta.json from each shard (has region count and top features)
2. Extracts region indices and feature data from feature_matrices.npz
3. Generates a minimal sae_results.tsv with top features for each region
4. Writes to the merged directory

Usage:
    python regenerate_bacillus_results_tsv.py
"""

import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict

# Paths
RESULTS_DIR = "results"
ORGANISM = "NC_000964.3"
MERGED_DIR_PATTERN = f"{RESULTS_DIR}/{ORGANISM}/sae/*_merged*of4"

def find_merged_dir():
    """Find the latest merged SAE directory for Bacillus."""
    dirs = sorted(glob.glob(MERGED_DIR_PATTERN))
    if not dirs:
        print(f"ERROR: No merged SAE directory found matching {MERGED_DIR_PATTERN}")
        sys.exit(1)
    return dirs[-1]

def load_feature_matrices(merged_dir):
    """Load feature_matrices.npz from merged directory."""
    path = os.path.join(merged_dir, "data", "feature_matrices.npz")
    if not os.path.exists(path):
        print(f"ERROR: feature_matrices.npz not found at {path}")
        sys.exit(1)

    data = np.load(path, allow_pickle=False)
    matrices = {}
    for key in sorted(data.files, key=lambda x: int(x.split('_')[1])):
        idx = int(key.split('_')[1])
        matrices[idx] = data[key]  # shape: (seq_len, n_features)

    print(f"Loaded {len(matrices)} region feature matrices")
    return matrices

def extract_top_features(feature_matrix, n_top=10):
    """Extract top N features by max activation across the sequence."""
    max_activations = feature_matrix.max(axis=0)  # max over sequence positions
    top_indices = np.argsort(max_activations)[-n_top:][::-1]
    top_values = max_activations[top_indices]
    return list(zip(top_indices, top_values))

def generate_sae_results_tsv(matrices, output_path):
    """Generate sae_results.tsv from feature matrices."""

    with open(output_path, 'w') as f:
        # Write header (mimics run_sae_fast.py output)
        f.write("# SAE Feature Analysis Results\n")
        f.write("# Region\tSequence_Length\tTop_Features\tMax_Activations\n")
        f.write("region_id\tseq_len\ttop_feature_indices\ttop_feature_values\n")

        for idx in sorted(matrices.keys()):
            mat = matrices[idx]
            seq_len = mat.shape[0]

            # Extract top 10 features
            top_features = extract_top_features(mat, n_top=10)

            # Format for TSV
            feature_ids = ','.join(str(fid) for fid, _ in top_features)
            feature_vals = ','.join(f'{val:.2f}' for _, val in top_features)

            f.write(f"region_{idx}\t{seq_len}\t{feature_ids}\t{feature_vals}\n")

    print(f"Wrote sae_results.tsv: {output_path}")

def main():
    merged_dir = find_merged_dir()
    print(f"Using merged directory: {merged_dir}")

    # Load feature matrices
    matrices = load_feature_matrices(merged_dir)

    # Generate TSV
    output_tsv = os.path.join(merged_dir, "data", "sae_results.tsv")
    generate_sae_results_tsv(matrices, output_tsv)

    print(f"✓ Bacillus sae_results.tsv regenerated ({len(matrices)} regions)")
    print(f"Ready for latent analysis!")

if __name__ == '__main__':
    main()
