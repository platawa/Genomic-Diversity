#!/usr/bin/env python3
"""
merge_sae_shards_fast.py

Fast version of merge_sae_shards.py using ZIP_STORED (no compression).
Creates larger output file (~30-40GB uncompressed) but writes 10-15x faster.

Identical to merge_sae_shards.py except:
  - Uses zipfile.ZIP_STORED instead of ZIP_DEFLATED
  - Much faster write speed (~15-20 min vs 2+ hours)
  - Output file is uncompressed (larger but acceptable for read-only operations)

Usage:
    python merge_sae_shards_fast.py --chrom chr15 --n_shards 36 --output_dir results/
"""

import os
import sys
import json
import argparse
import glob
import re
from datetime import datetime

import numpy as np


def _shard_has_chunk_data(shard_dir: str) -> bool:
    """Check if a shard directory has usable chunk data (even without COMPLETED)."""
    data_dir = os.path.join(shard_dir, 'data')
    if not os.path.isdir(data_dir):
        return False
    chunk_files = glob.glob(os.path.join(data_dir, '_chunk_*.npz'))
    meta_file = os.path.join(data_dir, '_checkpoint_meta.json')
    return len(chunk_files) > 0 and os.path.isfile(meta_file)


def _extract_timestamp(dirname: str) -> str | None:
    """Extract YYYYMMDD_HHMMSS timestamp prefix from a run directory name."""
    m = re.match(r"(\d{8}_\d{6})", os.path.basename(dirname))
    return m.group(1) if m else None


def find_shard_dirs(output_dir: str, chrom: str, n_shards: int | None = None,
                    include_partial: bool = False,
                    after: str | None = None) -> list[str]:
    """Find shard run directories for a chromosome.

    If include_partial=True, also includes shards that have chunk data but no
    COMPLETED sentinel (e.g. step 6b OOM'd after SAE extraction finished).

    If after is set (e.g. "20260322_2135"), only includes shard dirs whose
    timestamp is >= the given value. This distinguishes current runs from
    previous failed attempts.
    """
    sae_root = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_root):
        print(f"ERROR: {sae_root} does not exist")
        sys.exit(1)

    pattern = re.compile(r"shard(\d+)of(\d+)")
    dirs = []
    for entry in sorted(os.listdir(sae_root)):
        full = os.path.join(sae_root, entry)
        m = pattern.search(entry)
        if not m:
            continue
        shard_idx = int(m.group(1))
        shard_total = int(m.group(2))
        if n_shards is not None and shard_total != n_shards:
            continue

        # Filter by timestamp if --after is set
        if after:
            ts = _extract_timestamp(entry)
            if ts and ts < after:
                continue

        completed = os.path.isfile(os.path.join(full, "COMPLETED"))
        has_chunks = _shard_has_chunk_data(full)

        if completed:
            dirs.append((shard_idx, shard_total, full, "complete"))
        elif include_partial and has_chunks:
            dirs.append((shard_idx, shard_total, full, "partial"))

    if not dirs:
        if include_partial:
            print(f"ERROR: no shard directories (complete or partial) found under {sae_root}")
        else:
            print(f"ERROR: no COMPLETED shard directories found under {sae_root}")
            print(f"  TIP: use --include-partial to also use shards with chunk data but no COMPLETED")
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


def _find_chunk_files(shard_dir: str) -> list[str]:
    """Find _chunk_*.npz files in a shard's data directory, sorted by chunk index."""
    data_dir = os.path.join(shard_dir, 'data')
    files = glob.glob(os.path.join(data_dir, '_chunk_*.npz'))
    # Sort by start index: _chunk_0_500.npz → 0
    def _sort_key(p):
        base = os.path.basename(p)  # _chunk_0_500.npz
        parts = base.replace('.npz', '').split('_')  # ['', 'chunk', '0', '500']
        return int(parts[2])
    return sorted(files, key=_sort_key)


def merge_feature_matrices(shard_dirs: list[tuple], out_path: str) -> dict:
    """Merge chunk NPZ files from shards into one feature_matrices.npz.

    Streams arrays one at a time into a zip file to avoid loading everything
    into memory at once.  Uses ZIP_STORED (no compression) for 10-15x faster writes.

    Works with both _checkpoint.npz (old format) and _chunk_*.npz (new format).
    Returns a dict with shard-level offset info for downstream use.
    """
    import zipfile
    import io
    from numpy.lib.format import write_array

    region_offset = 0
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for entry in shard_dirs:
            shard_idx, _, shard_dir = entry[0], entry[1], entry[2]
            chunk_files = _find_chunk_files(shard_dir)
            if chunk_files:
                n_in_shard = 0
                for cf in chunk_files:
                    data = np.load(cf)
                    for key in sorted(data.files, key=lambda x: int(x.split('_')[1])):
                        orig_idx = int(key.split('_')[1])
                        new_key = f'region_{region_offset + orig_idx}'
                        arr = data[key]
                        buf = io.BytesIO()
                        write_array(buf, arr)
                        zf.writestr(new_key + '.npy', buf.getvalue())
                        del arr, buf
                        n_in_shard = max(n_in_shard, orig_idx + 1)
                    data.close()
                region_offset += n_in_shard
                print(f"  Shard {shard_idx}: {n_in_shard} regions from {len(chunk_files)} chunk files (offset now {region_offset})")
            else:
                chk = os.path.join(shard_dir, 'data', '_checkpoint.npz')
                if not os.path.isfile(chk):
                    print(f"  WARNING: no chunk/checkpoint data found for shard {shard_idx} (skipping)")
                    continue
                data = np.load(chk)
                n_in_shard = 0
                for key in sorted(data.files):
                    orig_idx = int(key.split('_')[1])
                    new_key = f'region_{region_offset + orig_idx}'
                    arr = data[key]
                    buf = io.BytesIO()
                    write_array(buf, arr)
                    zf.writestr(new_key + '.npy', buf.getvalue())
                    del arr, buf
                    n_in_shard = max(n_in_shard, orig_idx + 1)
                data.close()
                region_offset += n_in_shard
                print(f"  Shard {shard_idx}: {n_in_shard} regions (offset now {region_offset})")

    print(f"  Merged feature matrices (uncompressed) → {out_path}  ({region_offset} total regions)")
    return {'total_regions': region_offset}


def compute_norm_stats_streaming(shard_dirs: list[tuple], out_path: str):
    """Compute per-feature normalization stats from chunk files in a streaming
    fashion — one chunk at a time, never loading all data into memory at once.

    Uses Welford's online algorithm for numerically stable mean/std.
    """
    n_features = None
    count = 0
    mean = None
    m2 = None  # sum of squared differences from the mean

    for entry in shard_dirs:
        shard_idx, _, shard_dir = entry[0], entry[1], entry[2]

        # Try existing norm stats first (from completed shards)
        stats_path = os.path.join(shard_dir, 'data', 'feature_norm_stats.npz')
        if os.path.isfile(stats_path):
            # Skip — we'll compute from raw chunks for consistency
            pass

        chunk_files = _find_chunk_files(shard_dir)
        if not chunk_files:
            chk = os.path.join(shard_dir, 'data', '_checkpoint.npz')
            if os.path.isfile(chk):
                chunk_files = [chk]

        for cf in chunk_files:
            data = np.load(cf)
            for key in sorted(data.files, key=lambda x: int(x.split('_')[1])):
                arr = data[key].astype(np.float64)  # shape: (seq_len, n_features)
                if n_features is None:
                    n_features = arr.shape[1]
                    mean = np.zeros(n_features, dtype=np.float64)
                    m2 = np.zeros(n_features, dtype=np.float64)

                # Vectorized batch Welford's: process entire region at once
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

    if count == 0 or n_features is None:
        print("  WARNING: no data found for normalization stats")
        return

    variance = m2 / count
    std = np.sqrt(variance).astype(np.float32)
    std = np.maximum(std, 1e-6)
    mean_f32 = mean.astype(np.float32)

    np.savez_compressed(out_path, mean=mean_f32, std=std)
    print(f"  Streaming norm stats computed from {count:,} positions across {n_features} features → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chrom", required=True,
                        help="Chromosome name (e.g. chr1)")
    parser.add_argument("--n_shards", type=int, default=None,
                        help="Number of shards to expect (auto-detect if omitted)")
    parser.add_argument("--output_dir", default="results/",
                        help="Root results directory (default: results/)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Include shards with chunk data but no COMPLETED sentinel "
                             "(e.g. shards that OOM'd during step 6b after SAE extraction)")
    parser.add_argument("--after", type=str, default=None,
                        help="Only use shard dirs with timestamp >= this value "
                             "(e.g. 20260322_2135). Filters out old failed runs.")
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip('/')
    include_partial = getattr(args, 'include_partial', False)

    print(f"Finding shard directories for {args.chrom}...")
    if args.after:
        print(f"  Filtering: only shards with timestamp >= {args.after}")
    shard_dirs = find_shard_dirs(output_dir, args.chrom, args.n_shards,
                                include_partial=include_partial,
                                after=args.after)
    n_shards = shard_dirs[0][1]
    found = [s[0] for s in shard_dirs]
    statuses = {s[0]: s[3] for s in shard_dirs}
    n_complete = sum(1 for s in shard_dirs if s[3] == "complete")
    n_partial = sum(1 for s in shard_dirs if s[3] == "partial")
    print(f"Found {len(shard_dirs)}/{n_shards} shards: {found}")
    print(f"  {n_complete} complete, {n_partial} partial (chunk data only)")

    missing = [i for i in range(n_shards) if i not in found]
    if missing:
        print(f"WARNING: missing shards {missing} — proceeding with available shards only")

    # Build merged output directory
    from results_utils import build_run_dir, write_completed
    import time as _time
    _wall_start = _time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"all_conf8.0_merged{len(shard_dirs)}of{n_shards}_fast"
    run_dir = build_run_dir(output_dir, args.chrom, "sae", descriptor)
    os.makedirs(os.path.join(run_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    print(f"\nMerging into: {run_dir}")
    print(f"  Using ZIP_STORED (no compression) for fast writes")

    # Merge sae_results.tsv (only from complete shards that have it)
    tsv_paths = [os.path.join(d, 'data', 'sae_results.tsv') for _, _, d, _ in shard_dirs
                 if os.path.isfile(os.path.join(d, 'data', 'sae_results.tsv'))]
    if tsv_paths:
        merge_tsv(tsv_paths, os.path.join(run_dir, 'data', 'sae_results.tsv'))
    else:
        print("  NOTE: No sae_results.tsv found in any shard (expected for partial shards)")

    # Merge feature matrices from chunk files
    print("\nMerging feature matrices (uncompressed, fast)...")
    merge_feature_matrices(
        shard_dirs,
        os.path.join(run_dir, 'data', 'feature_matrices.npz')
    )

    # Compute normalization stats using streaming (low memory)
    print("\nComputing normalization stats (streaming)...")
    compute_norm_stats_streaming(
        shard_dirs,
        os.path.join(run_dir, 'data', 'feature_norm_stats.npz')
    )

    # Write source.json pointing to all shard dirs
    source = {f"shard_{s[0]}": {
        "path": os.path.relpath(s[2], run_dir),
        "status": s[3],
    } for s in shard_dirs}
    with open(os.path.join(run_dir, 'source.json'), 'w') as f:
        json.dump(source, f, indent=2)

    # Write run_metadata.json
    meta = {
        'chrom':          args.chrom,
        'n_shards':       n_shards,
        'shards_used':    found,
        'shards_complete': [s[0] for s in shard_dirs if s[3] == "complete"],
        'shards_partial':  [s[0] for s in shard_dirs if s[3] == "partial"],
        'merged_at':      timestamp,
        'compression':    'ZIP_STORED (none)',
    }
    with open(os.path.join(run_dir, 'data', 'run_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Mark completed
    _wall_time = _time.time() - _wall_start
    write_completed(run_dir, script_name='merge_sae_shards_fast.py', wall_time_s=_wall_time)
    print(f"\nDone. Merged output: {run_dir}")


if __name__ == '__main__':
    main()
