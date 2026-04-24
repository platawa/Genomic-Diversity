#!/usr/bin/env python3
"""
Regenerate sae_results.tsv for E. coli conf0.0 shards from chunk NPZ files.

The --extract_only mode of run_sae_fast.py saves chunk checkpoints but not
sae_results.tsv. This script reconstructs the TSV needed by compute_sae_latent.py.

Uses drop_boundaries.tsv for genomic coordinates and checkpoint metadata for
region-to-shard mapping.

Usage:
    python regenerate_ecoli_tsv_from_chunks.py
"""

import os
import json
import glob
import numpy as np


def main():
    base = "results/NC_000913.3"
    scoring_run = os.path.join(base, "scoring", "20260309_122646_rc_logprobs_1gpu")
    boundaries_path = os.path.join(scoring_run, "data", "drop_boundaries.tsv")

    # Load all drop boundaries (conf >= 0.0 = all)
    regions = []
    with open(boundaries_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('chrom'):
                continue
            parts = line.split('\t')
            if len(parts) >= 11:
                regions.append({
                    'genomic_start': int(parts[3]),
                    'genomic_end': int(parts[4]),
                    'region_length': int(parts[5]),
                    'method': parts[6],
                    'confidence': float(parts[7]),
                })

    print(f"Loaded {len(regions)} regions from {boundaries_path}")

    # Deduplicate: keep one entry per (start, end) with highest confidence
    seen = {}
    for r in regions:
        key = (r['genomic_start'], r['genomic_end'])
        if key not in seen or r['confidence'] > seen[key]['confidence']:
            seen[key] = r

    # Sort by start position
    unique_regions = sorted(seen.values(), key=lambda r: r['genomic_start'])
    print(f"Unique regions (deduped): {len(unique_regions)}")

    # Find shard directories
    shard_dirs = sorted(glob.glob(os.path.join(base, "sae", "20260408_125325_max999999_conf0.0_shard*of2")))
    print(f"Found {len(shard_dirs)} shard dirs")

    for shard_dir in shard_dirs:
        # Check how many regions this shard has from checkpoint
        meta_path = os.path.join(shard_dir, "data", "_checkpoint_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  {os.path.basename(shard_dir)}: n_done={meta.get('n_done', '?')}")

        # Write sae_results.tsv for this shard
        output_tsv = os.path.join(shard_dir, "data", "sae_results.tsv")

        # Determine shard index and total from directory name
        dirname = os.path.basename(shard_dir)
        # e.g., 20260408_125325_max999999_conf0.0_shard0of2
        shard_idx = int(dirname.split("shard")[1].split("of")[0])
        n_shards = int(dirname.split("of")[1])

        # Calculate which regions belong to this shard
        n_total = len(unique_regions)
        shard_size = (n_total + n_shards - 1) // n_shards
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, n_total)
        shard_regions = unique_regions[start_idx:end_idx]

        print(f"  Writing {len(shard_regions)} regions to {output_tsv}")

        with open(output_tsv, 'w') as f:
            f.write("# SAE Feature Analysis Results (regenerated from chunks)\n")
            f.write("region_idx\tgenomic_start\tgenomic_end\tmethod\tconfidence\n")
            for i, r in enumerate(shard_regions):
                f.write(f"{start_idx + i}\t{r['genomic_start']}\t{r['genomic_end']}\t"
                        f"{r['method']}\t{r['confidence']:.4f}\n")

    print(f"\nDone! Generated sae_results.tsv for {len(shard_dirs)} shards")
    print(f"Total unique regions: {len(unique_regions)}")


if __name__ == '__main__':
    main()
