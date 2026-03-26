#!/usr/bin/env python3
"""
Regenerate Bacillus sae_results.tsv from shard chunk files directly.

Instead of loading the giant 5.8GB merged file, this reads from the individual
100MB chunk files in each shard directory. Much lower memory footprint.

Usage:
    python regenerate_bacillus_tsv_from_chunks.py
"""

import os
import glob
import json
import numpy as np
from collections import defaultdict

def extract_top_features(feature_matrix, n_top=10):
    """Extract top N features by max activation."""
    max_activations = feature_matrix.max(axis=0)
    top_indices = np.argsort(max_activations)[-n_top:][::-1]
    top_values = max_activations[top_indices]
    return list(zip(top_indices, top_values))

def regenerate_from_chunks():
    """Regenerate TSV by streaming from shard chunk files."""

    # Find all Bacillus shard directories
    shard_dirs = sorted(glob.glob("results/NC_000964.3/sae/20260324_*_shard*of4"))

    if not shard_dirs:
        print("ERROR: No Bacillus shard directories found")
        return False

    print(f"Found {len(shard_dirs)} shards")

    # Merged output directory
    merged_dir = sorted(glob.glob("results/NC_000964.3/sae/*_merged4of4"))[-1]
    output_tsv = os.path.join(merged_dir, "data", "sae_results.tsv")

    print(f"Output: {output_tsv}")
    print(f"Processing {len(shard_dirs)} shards...")

    region_count = 0

    with open(output_tsv, 'w') as f:
        # Write header
        f.write("# SAE Feature Analysis Results\n")
        f.write("# Region\tSequence_Length\tTop_Features\tMax_Activations\n")
        f.write("region_id\tseq_len\ttop_feature_indices\ttop_feature_values\n")

        # Process each shard
        for shard_idx, shard_dir in enumerate(shard_dirs):
            data_dir = os.path.join(shard_dir, "data")
            chunk_files = sorted(glob.glob(os.path.join(data_dir, "_chunk_*.npz")))

            if not chunk_files:
                print(f"  Shard {shard_idx}: No chunk files found")
                continue

            print(f"  Shard {shard_idx}: {len(chunk_files)} chunks...")
            shard_region_count = 0

            # Process each chunk file in the shard
            for chunk_file in chunk_files:
                data = np.load(chunk_file, allow_pickle=False)

                for key in sorted(data.files, key=lambda x: int(x.split('_')[1])):
                    region_id = int(key.split('_')[1])
                    mat = data[key]  # shape: (seq_len, n_features)
                    seq_len = mat.shape[0]

                    # Extract top 10 features
                    top_features = extract_top_features(mat, n_top=10)

                    # Format for TSV
                    feature_ids = ','.join(str(fid) for fid, _ in top_features)
                    feature_vals = ','.join(f'{val:.2f}' for _, val in top_features)

                    f.write(f"region_{region_count}\t{seq_len}\t{feature_ids}\t{feature_vals}\n")
                    region_count += 1
                    shard_region_count += 1

                data.close()

            print(f"    → {shard_region_count} regions")

    print(f"\n✓ Generated sae_results.tsv: {region_count} regions")
    print(f"✓ Ready for latent analysis!")
    return True

if __name__ == '__main__':
    regenerate_from_chunks()
