#!/usr/bin/env python3
"""
merge_sae_shards.py

Merge per-shard SAE results produced by run_sae_fast.py --shard N/M into a
single unified output directory in the standard results layout.

Each shard writes to:
    results/<chrom>/sae/<timestamp>_all_conf8.0_shard<N>of<M>/

This script:
  1. Finds all completed shard directories for the chromosome
  2. Merges sae_results.tsv, signature_features.tsv, and feature_matrices.npz
  3. Re-runs signature-feature finding across merged regions
  4. Writes a new unified sae directory and marks it COMPLETED

Usage:
    python merge_sae_shards.py --chrom chr1 --n_shards 4 --output_dir results/
    python merge_sae_shards.py --chrom chr1 --output_dir results/  # auto-detect shards
"""

import os
import sys
import json
import argparse
import glob
import re
from datetime import datetime

import numpy as np


def find_shard_dirs(output_dir: str, chrom: str, n_shards: int | None = None) -> list[str]:
    """Find all completed shard run directories for a chromosome."""
    sae_root = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_root):
        print(f"ERROR: {sae_root} does not exist")
        sys.exit(1)

    pattern = re.compile(r"shard(\d+)of(\d+)")
    dirs = []
    for entry in sorted(os.listdir(sae_root)):
        full = os.path.join(sae_root, entry)
        completed = os.path.join(full, "COMPLETED")
        if not os.path.isfile(completed):
            continue
        m = pattern.search(entry)
        if not m:
            continue
        shard_idx = int(m.group(1))
        shard_total = int(m.group(2))
        if n_shards is not None and shard_total != n_shards:
            continue
        dirs.append((shard_idx, shard_total, full))

    if not dirs:
        print(f"ERROR: no COMPLETED shard directories found under {sae_root}")
        sys.exit(1)

    dirs.sort(key=lambda x: x[0])
    return dirs


def merge_tsv(paths: list[str], out_path: str):
    """Merge TSV files: write header once then all data rows."""
    header = None
    rows = []
    for path in paths:
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        rows.extend(lines[1:])
    if header is None:
        return
    with open(out_path, 'w') as f:
        f.write(header)
        f.writelines(rows)
    print(f"  Merged TSV → {out_path}  ({len(rows)} data rows)")


def merge_feature_matrices(shard_dirs: list[tuple], out_path: str) -> dict:
    """Merge _checkpoint.npz files from shards into one feature_matrices.npz.

    Returns a dict with shard-level offset info for downstream use.
    """
    all_arrays = {}
    region_offset = 0
    for shard_idx, _, shard_dir in shard_dirs:
        chk = os.path.join(shard_dir, 'data', '_checkpoint.npz')
        if not os.path.isfile(chk):
            print(f"  WARNING: checkpoint not found: {chk} (skipping shard {shard_idx})")
            continue
        data = np.load(chk)
        for key in sorted(data.files):
            # key is region_0, region_1, ...
            orig_idx = int(key.split('_')[1])
            new_key = f'region_{region_offset + orig_idx}'
            all_arrays[new_key] = data[key]
        # Count regions in this shard
        n_in_shard = sum(1 for k in data.files if k.startswith('region_'))
        region_offset += n_in_shard
        print(f"  Shard {shard_idx}: {n_in_shard} regions (offset now {region_offset})")

    np.savez_compressed(out_path, **all_arrays)
    print(f"  Merged feature matrices → {out_path}  ({region_offset} total regions)")
    return {'total_regions': region_offset}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", required=True,
                        help="Chromosome name (e.g. chr1)")
    parser.add_argument("--n_shards", type=int, default=None,
                        help="Number of shards to expect (auto-detect if omitted)")
    parser.add_argument("--output_dir", default="results/",
                        help="Root results directory (default: results/)")
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip('/')

    print(f"Finding completed shard directories for {args.chrom}...")
    shard_dirs = find_shard_dirs(output_dir, args.chrom, args.n_shards)
    n_shards = shard_dirs[0][1]
    found = [s[0] for s in shard_dirs]
    print(f"Found {len(shard_dirs)}/{n_shards} shards: {found}")

    missing = [i for i in range(n_shards) if i not in found]
    if missing:
        print(f"WARNING: missing shards {missing} — proceeding with available shards only")

    # Build merged output directory
    from results_utils import build_run_dir, write_completed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"all_conf8.0_merged{len(shard_dirs)}of{n_shards}"
    run_dir = build_run_dir(output_dir, args.chrom, "sae", descriptor)
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    print(f"\nMerging into: {run_dir}")

    # Merge sae_results.tsv
    tsv_paths = [os.path.join(d, 'data', 'sae_results.tsv') for _, _, d in shard_dirs
                 if os.path.isfile(os.path.join(d, 'data', 'sae_results.tsv'))]
    if tsv_paths:
        merge_tsv(tsv_paths, os.path.join(run_dir, 'data', 'sae_results.tsv'))

    # Merge feature matrices
    merge_feature_matrices(
        shard_dirs,
        os.path.join(run_dir, 'data', 'feature_matrices.npz')
    )

    # Copy/merge normalization stats (average across shards)
    stat_arrays = []
    for _, _, d in shard_dirs:
        p = os.path.join(d, 'data', 'feature_norm_stats.npz')
        if os.path.isfile(p):
            stat_arrays.append(np.load(p))
    if stat_arrays:
        merged_mean = np.mean([s['mean'] for s in stat_arrays], axis=0)
        merged_std  = np.mean([s['std']  for s in stat_arrays], axis=0)
        np.savez_compressed(
            os.path.join(run_dir, 'data', 'feature_norm_stats.npz'),
            mean=merged_mean, std=merged_std
        )
        print(f"  Merged normalization stats")

    # Write source.json pointing to all shard dirs
    source = {f"shard_{s[0]}": os.path.relpath(s[2], run_dir) for s in shard_dirs}
    with open(os.path.join(run_dir, 'source.json'), 'w') as f:
        json.dump(source, f, indent=2)

    # Write run_metadata.json
    meta = {
        'chrom':       args.chrom,
        'n_shards':    n_shards,
        'shards_used': found,
        'merged_at':   timestamp,
    }
    with open(os.path.join(run_dir, 'data', 'run_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Mark completed
    write_completed(run_dir, script='merge_sae_shards.py', wall_time_s=0)
    print(f"\nDone. Merged output: {run_dir}")


if __name__ == '__main__':
    main()
