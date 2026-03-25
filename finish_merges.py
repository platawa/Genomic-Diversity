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
import glob as glob_mod
import re
import argparse
import shutil
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_utils import write_completed


def find_global_sae_stats(output_dir, chrom):
    """Find the latest global_sae_stats.npz for a chromosome.

    Looks in results/<chrom>/sae_global_stats/*/data/global_sae_stats.npz
    and returns the path to the most recent one, or None.
    """
    pattern = os.path.join(output_dir, chrom, "sae_global_stats", "*", "data", "global_sae_stats.npz")
    matches = sorted(glob_mod.glob(pattern))
    return matches[-1] if matches else None


def copy_global_stats_as_norm_stats(global_stats_path, merged_dir):
    """Copy global SAE stats (chunk_max_mean/std) as feature_norm_stats.npz (mean/std).

    The global stats contain chunk_max_mean and chunk_max_std which represent the
    mean and std of per-chunk maximum activations across the entire chromosome.
    These serve as reasonable normalization statistics for downstream z-score
    normalization (replot.py recomputes from scratch anyway).
    """
    out_path = os.path.join(merged_dir, "data", "feature_norm_stats.npz")
    gstats = np.load(global_stats_path)
    mean = gstats["chunk_max_mean"].astype(np.float32)
    std = np.maximum(gstats["chunk_max_std"].astype(np.float32), 1e-6)
    gstats.close()

    np.savez_compressed(out_path, mean=mean, std=std)
    print(f"  Copied global stats as norm stats → {out_path}")
    print(f"    Source: {global_stats_path}")
    return True


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

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"  ERROR: Cannot load {npz_path}: {e}")
        return False
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


def finish_merge(merged_dir, chrom, output_dir="results/", recompute=False):
    """Finish an incomplete merge: add norm stats, write metadata, mark COMPLETED.

    By default, uses existing global SAE stats (fast). Falls back to Welford's
    streaming recomputation only if global stats are not found or --recompute is set.
    """
    wall_start = time.time()
    print(f"\n{'='*60}")
    print(f"Finishing merge for {chrom}: {merged_dir}")

    norm_stats_path = os.path.join(merged_dir, "data", "feature_norm_stats.npz")
    have_norm_stats = os.path.isfile(norm_stats_path)

    if not have_norm_stats:
        # Try global stats first (fast path)
        if not recompute:
            global_stats = find_global_sae_stats(output_dir, chrom)
            if global_stats:
                print(f"  Using global SAE stats (fast path)")
                if not copy_global_stats_as_norm_stats(global_stats, merged_dir):
                    print(f"  FAILED to copy global stats for {chrom}")
                    return False
            else:
                print(f"  No global stats found, falling back to Welford recomputation...")
                if not compute_norm_stats_from_merged(merged_dir):
                    print(f"  FAILED to compute norm stats for {chrom}")
                    return False
        else:
            print(f"  --recompute: using Welford streaming recomputation")
            if not compute_norm_stats_from_merged(merged_dir):
                print(f"  FAILED to compute norm stats for {chrom}")
                return False
    else:
        print(f"  Norm stats already exist, skipping")

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
    parser.add_argument("--recompute", action="store_true",
                        help="Force recompute norm stats via Welford (slow) instead of using global stats")
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
            if finish_merge(merged, chrom, output_dir=args.output_dir, recompute=args.recompute):
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
