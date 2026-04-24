#!/usr/bin/env python3
"""
Regenerate sae_results.tsv for Bacillus conf0.0 shards from drop_boundaries.tsv.

Usage:
    python regenerate_bacillus_conf0_tsv.py
"""

import os
import glob


def main():
    base = "results/NC_000964.3"
    # Use the second scoring run (25K regions, not the empty first one)
    boundaries_path = os.path.join(base, "scoring",
        "20260318_155637_rc_logprobs_bf16_1gpu", "data", "drop_boundaries.tsv")

    # Load all drop boundaries
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

    unique_regions = sorted(seen.values(), key=lambda r: r['genomic_start'])
    print(f"Unique regions (deduped): {len(unique_regions)}")

    # Find completed conf0.0 shard directories
    shard_dirs = sorted(glob.glob(os.path.join(base, "sae", "*_conf0.0_shard*of4")))
    shard_dirs = [d for d in shard_dirs if os.path.isfile(os.path.join(d, "COMPLETED"))]
    print(f"Found {len(shard_dirs)} completed shard dirs")

    n_shards = 4
    n_total = len(unique_regions)

    for shard_dir in shard_dirs:
        dirname = os.path.basename(shard_dir)
        shard_idx = int(dirname.split("shard")[1].split("of")[0])

        shard_size = (n_total + n_shards - 1) // n_shards
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, n_total)
        shard_regions = unique_regions[start_idx:end_idx]

        output_tsv = os.path.join(shard_dir, "data", "sae_results.tsv")
        print(f"  {dirname}: writing {len(shard_regions)} regions to sae_results.tsv")

        with open(output_tsv, 'w') as f:
            f.write("# SAE Feature Analysis Results (regenerated)\n")
            f.write("region_idx\tgenomic_start\tgenomic_end\tmethod\tconfidence\n")
            for i, r in enumerate(shard_regions):
                f.write(f"{start_idx + i}\t{r['genomic_start']}\t{r['genomic_end']}\t"
                        f"{r['method']}\t{r['confidence']:.4f}\n")

    print(f"\nDone! Generated sae_results.tsv for {len(shard_dirs)} shards")


if __name__ == '__main__':
    main()
