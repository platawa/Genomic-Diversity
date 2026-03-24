#!/usr/bin/env python3
"""
finish_merges.py

Finish incomplete merge directories that have feature_matrices.npz
but are missing feature_norm_stats.npz and COMPLETED.

For chromosomes that never started merge, runs the full merge
(skipping finalize entirely — uses --include-partial).

Usage:
    python finish_merges.py --chrom chr2 --output_dir results/
    python finish_merges.py --all_human --output_dir results/
"""

import os
import sys
import json
import glob
import re
import argparse
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_utils import write_completed


def find_incomplete_merge(output_dir, chrom):
    """Find a merge directory that has feature_matrices.npz but no COMPLETED."""
    sae_dir = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_dir):
        return None
    for entry in sorted(os.listdir(sae_dir), reverse=True):
        if "merged" not in entry:
            continue
        d = os.path.join(sae_dir, entry)
        if not os.path.isdir(d):
            continue
        npz = os.path.join(d, "data", "feature_matrices.npz")
        completed = os.path.join(d, "COMPLETED")
        if os.path.isfile(npz) and not os.path.isfile(completed):
            return d
    return None


def compute_norm_stats_from_merged(merged_dir):
    """Compute feature_norm_stats.npz from the already-merged feature_matrices.npz.

    Streams through the npz file one region at a time using Welford's algorithm.
    """
    npz_path = os.path.join(merged_dir, "data", "feature_matrices.npz")
    print(f"  Computing norm stats from {npz_path}...")

    data = np.load(npz_path)
    keys = sorted(data.files, key=lambda x: int(x.replace("region_", "").replace(".npy", "")))

    n_features = None
    count = 0
    mean = None
    m2 = None

    for i, key in enumerate(keys):
        arr = data[key].astype(np.float64)
        if n_features is None:
            n_features = arr.shape[-1] if arr.ndim > 1 else arr.shape[0]
            mean = np.zeros(n_features, dtype=np.float64)
            m2 = np.zeros(n_features, dtype=np.float64)

        if arr.ndim == 1:
            # Single vector (max-pooled)
            n = 1
            batch_mean = arr
            batch_var = np.zeros(n_features, dtype=np.float64)
        else:
            # (seq_len, n_features)
            n = arr.shape[0]
            if n == 0:
                continue
            batch_mean = arr.mean(axis=0)
            batch_var = arr.var(axis=0)

        new_count = count + n
        delta = batch_mean - mean
        mean = (count * mean + n * batch_mean) / new_count
        m2 = m2 + n * batch_var + delta ** 2 * count * n / new_count
        count = new_count

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(keys)} regions...")

    data.close()

    if count == 0 or n_features is None:
        print("  WARNING: no data found")
        return False

    std = np.sqrt(m2 / count).astype(np.float32)
    std = np.maximum(std, 1e-6)
    mean_f32 = mean.astype(np.float32)

    out_path = os.path.join(merged_dir, "data", "feature_norm_stats.npz")
    np.savez_compressed(out_path, mean=mean_f32, std=std)
    print(f"  Norm stats: {count:,} positions, {n_features} features → {out_path}")
    return True


def finish_merge(merged_dir, chrom):
    """Finish an incomplete merge: compute norm stats, write metadata, mark COMPLETED."""
    wall_start = time.time()
    print(f"\n{'='*60}")
    print(f"Finishing merge for {chrom}: {merged_dir}")

    # Compute norm stats
    if not compute_norm_stats_from_merged(merged_dir):
        print(f"  FAILED to compute norm stats for {chrom}")
        return False

    # Write source.json if missing
    source_path = os.path.join(merged_dir, "source.json")
    if not os.path.isfile(source_path):
        with open(source_path, "w") as f:
            json.dump({"note": "finished by finish_merges.py"}, f, indent=2)

    # Write run_metadata.json if missing
    meta_path = os.path.join(merged_dir, "data", "run_metadata.json")
    if not os.path.isfile(meta_path):
        meta = {
            "chrom": chrom,
            "finished_by": "finish_merges.py",
            "timestamp": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Mark COMPLETED
    wall_time = time.time() - wall_start
    write_completed(merged_dir, "finish_merges.py", wall_time)
    print(f"  COMPLETED in {wall_time:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", type=str, default=None)
    parser.add_argument("--all_human", action="store_true")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    if args.all_human:
        chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    elif args.chrom:
        chroms = [args.chrom]
    else:
        print("Specify --chrom or --all_human")
        sys.exit(1)

    success = 0
    skipped = 0
    failed = 0
    need_full_merge = []

    for chrom in chroms:
        merged = find_incomplete_merge(args.output_dir, chrom)
        if merged:
            if finish_merge(merged, chrom):
                success += 1
            else:
                failed += 1
        else:
            # Check if already complete
            sae_dir = os.path.join(args.output_dir, chrom, "sae")
            already_done = False
            for entry in sorted(os.listdir(sae_dir) if os.path.isdir(sae_dir) else [], reverse=True):
                if "merged" in entry:
                    d = os.path.join(sae_dir, entry)
                    if os.path.isfile(os.path.join(d, "COMPLETED")):
                        print(f"\n{chrom}: already merged and COMPLETED, skipping")
                        skipped += 1
                        already_done = True
                        break
            if not already_done:
                need_full_merge.append(chrom)

    print(f"\n{'='*60}")
    print(f"Done: {success} finished, {skipped} already done, {failed} failed")
    if need_full_merge:
        print(f"Need full merge (no existing merge dir): {need_full_merge}")
        print("Run: python merge_sae_shards.py --chrom <chr> --n_shards 36 --output_dir results/ --include-partial")


if __name__ == "__main__":
    main()
