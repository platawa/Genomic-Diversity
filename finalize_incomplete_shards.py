#!/usr/bin/env python3
"""
finalize_incomplete_shards.py

Post-hoc finalization of SAE shard directories that completed step 6
(SAE feature extraction with checkpoint files on disk) but were killed
during step 6b (per-feature normalization) due to system OOM.

For each incomplete shard directory this script:
  1. Verifies _chunk_*.npz and _checkpoint_meta.json exist
  2. Re-parses drop_boundaries.tsv to recover region metadata
  3. Computes normalization stats by streaming chunks (avoids OOM)
  4. Generates sae_results.tsv and signature_features.tsv
  5. Writes run_metadata.json and COMPLETED sentinel

No GPU or model needed — all expensive computation is already saved in chunks.

Usage:
    # Finalize all incomplete of36 shards for all chromosomes:
    python finalize_incomplete_shards.py --output_dir results/

    # Finalize a single chromosome:
    python finalize_incomplete_shards.py --output_dir results/ --chrom chr16

    # Dry run (show what would be finalized):
    python finalize_incomplete_shards.py --output_dir results/ --dry_run
"""

import os
import sys
import json
import re
import glob
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_utils import write_completed, find_latest_completed


# ---------------------------------------------------------------------------
# Region parsing (copied from sae_utils to avoid heavy imports)
# ---------------------------------------------------------------------------

def parse_chromosome_drops_tsv(boundaries_file, min_confidence=0.0):
    """Parse drop_boundaries.tsv — lightweight, no model needed."""
    regions = []
    with open(boundaries_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if parts[0] == 'chrom':
                continue
            if len(parts) < 11:
                continue
            region = {
                'chrom': parts[0],
                'drop_start': int(parts[1]),
                'drop_end': int(parts[2]),
                'genomic_start': int(parts[3]),
                'genomic_end': int(parts[4]),
                'region_length': int(parts[5]),
                'method': parts[6],
                'start_confidence': float(parts[7]),
                'end_confidence': float(parts[8]),
                'mean_entropy': float(parts[9]),
                'min_entropy': float(parts[10]),
            }
            if region['start_confidence'] >= min_confidence:
                regions.append(region)
    regions.sort(key=lambda r: -r['start_confidence'])
    return regions


def get_shard_regions(all_regions, shard_idx, n_shards):
    """Apply the same shard slicing logic as run_sae_fast.py."""
    total = len(all_regions)
    shard_size = (total + n_shards - 1) // n_shards
    shard_start = shard_idx * shard_size
    shard_end = min(shard_start + shard_size, total)
    return all_regions[shard_start:shard_end]


# ---------------------------------------------------------------------------
# Signature features (copied from run_sae_fast.py to avoid model imports)
# ---------------------------------------------------------------------------

def find_signature_features(results, min_prevalence=0.3):
    """Find SAE features recurring across drop regions."""
    feature_stats = defaultdict(lambda: {
        'drop_activations': [],
        'rise_activations': [],
        'positions': [],
        'zscore_activations': [],
        'mad_activations': [],
    })

    for result in results:
        region = result['region']
        method = region.get('method', 'unknown')

        for feat_id, activation in result['drop_features']:
            feature_stats[feat_id]['drop_activations'].append(activation)
            feature_stats[feat_id]['positions'].append(region['genomic_start'])
            if method == 'zscore':
                feature_stats[feat_id]['zscore_activations'].append(activation)
            elif method == 'mad':
                feature_stats[feat_id]['mad_activations'].append(activation)

        for feat_id, activation in result['rise_features']:
            feature_stats[feat_id]['rise_activations'].append(activation)
            if method == 'zscore':
                feature_stats[feat_id]['zscore_activations'].append(activation)
            elif method == 'mad':
                feature_stats[feat_id]['mad_activations'].append(activation)

    n_regions = len(results)
    min_count = max(1, int(n_regions * min_prevalence))

    signatures = []
    for feat_id, stats in feature_stats.items():
        total_appearances = len(stats['drop_activations']) + len(stats['rise_activations'])
        if total_appearances < min_count:
            continue
        all_acts = stats['drop_activations'] + stats['rise_activations']
        zs_acts = stats['zscore_activations']
        mad_acts = stats['mad_activations']
        signatures.append({
            'feature_id': feat_id,
            'total_count': total_appearances,
            'drop_count': len(stats['drop_activations']),
            'rise_count': len(stats['rise_activations']),
            'prevalence': total_appearances / n_regions,
            'mean_activation': float(np.mean(all_acts)) if all_acts else 0.0,
            'max_activation': float(np.max(all_acts)) if all_acts else 0.0,
            'positions': stats['positions'],
            'zscore_count': len(zs_acts),
            'mad_count': len(mad_acts),
            'zscore_mean_activation': float(np.mean(zs_acts)) if zs_acts else 0.0,
            'mad_mean_activation': float(np.mean(mad_acts)) if mad_acts else 0.0,
        })
    signatures.sort(key=lambda x: -x['mean_activation'])
    return signatures


# ---------------------------------------------------------------------------
# Streaming normalization stats (the fix for 6b OOM)
# ---------------------------------------------------------------------------

def compute_norm_stats_streaming(chunk_files):
    """Compute per-feature mean/std by streaming through chunk files.

    Uses vectorized batch Welford's algorithm — processes entire arrays at once
    instead of row-by-row, giving ~100x speedup over the naive Python loop.
    """
    n_features = None
    count = 0
    mean = None
    m2 = None  # sum of squared differences from the mean

    for cf in sorted(chunk_files):
        data = np.load(cf)
        for key in sorted(data.files, key=lambda x: int(x.split('_')[1])):
            arr = data[key].astype(np.float64)  # shape: (positions, features)
            if n_features is None:
                n_features = arr.shape[1]
                mean = np.zeros(n_features, dtype=np.float64)
                m2 = np.zeros(n_features, dtype=np.float64)

            # Vectorized batch Welford's: process entire region at once
            n = arr.shape[0]
            if n == 0:
                continue
            batch_mean = arr.mean(axis=0)
            batch_var = arr.var(axis=0)  # population variance
            new_count = count + n

            delta = batch_mean - mean
            mean = (count * mean + n * batch_mean) / new_count
            # Parallel variance merge formula
            m2 = m2 + n * batch_var + delta ** 2 * count * n / new_count

            count = new_count

        data.close()

    if count < 2:
        std = np.ones(n_features, dtype=np.float64) * 1e-6
    else:
        std = np.maximum(np.sqrt(m2 / count), 1e-6)

    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Find incomplete shard directories
# ---------------------------------------------------------------------------

def find_incomplete_shards(output_dir, chrom_filter=None, n_shards_filter=36):
    """Find shard dirs that have chunk data but no COMPLETED file."""
    pattern = re.compile(r'shard(\d+)of(\d+)')
    results = []

    chrom_dirs = sorted(glob.glob(os.path.join(output_dir, 'chr*')))
    for chrom_dir in chrom_dirs:
        chrom = os.path.basename(chrom_dir)
        if chrom_filter and chrom != chrom_filter:
            continue

        sae_dir = os.path.join(chrom_dir, 'sae')
        if not os.path.isdir(sae_dir):
            continue

        for entry in sorted(os.listdir(sae_dir)):
            full = os.path.join(sae_dir, entry)
            if not os.path.isdir(full):
                continue

            m = pattern.search(entry)
            if not m:
                continue

            shard_idx = int(m.group(1))
            n_shards = int(m.group(2))

            if n_shards_filter and n_shards != n_shards_filter:
                continue

            # Skip if already completed
            if os.path.isfile(os.path.join(full, 'COMPLETED')):
                continue

            # Check for chunk data
            data_dir = os.path.join(full, 'data')
            chunk_files = sorted(glob.glob(os.path.join(data_dir, '_chunk_*.npz')))
            meta_file = os.path.join(data_dir, '_checkpoint_meta.json')

            if not chunk_files or not os.path.isfile(meta_file):
                continue

            results.append({
                'dir': full,
                'chrom': chrom,
                'shard_idx': shard_idx,
                'n_shards': n_shards,
                'chunk_files': chunk_files,
                'meta_file': meta_file,
            })

    # For each (chrom, shard_idx), keep only the latest directory
    # (there may be multiple attempts; we want the one with the most data)
    best = {}
    for info in results:
        key = (info['chrom'], info['shard_idx'])
        if key not in best:
            best[key] = info
        else:
            # Prefer the one with more chunk files, then latest timestamp
            existing = best[key]
            if (len(info['chunk_files']) > len(existing['chunk_files']) or
                (len(info['chunk_files']) == len(existing['chunk_files']) and
                 info['dir'] > existing['dir'])):
                best[key] = info

    return sorted(best.values(), key=lambda x: (x['chrom'], x['shard_idx']))


# ---------------------------------------------------------------------------
# Finalize one shard
# ---------------------------------------------------------------------------

def finalize_shard(info, output_dir, min_confidence=8.0):
    """Finalize a single incomplete shard directory."""
    chrom = info['chrom']
    shard_idx = info['shard_idx']
    n_shards = info['n_shards']
    shard_dir = info['dir']
    data_dir = os.path.join(shard_dir, 'data')

    print(f"\n{'='*70}")
    print(f"Finalizing: {chrom} shard {shard_idx}/{n_shards}")
    print(f"  Dir: {shard_dir}")
    print(f"  Chunks: {len(info['chunk_files'])}")

    # Load checkpoint metadata
    with open(info['meta_file']) as f:
        meta = json.load(f)

    n_done = meta['n_done']
    print(f"  Regions in checkpoint: {n_done}")

    # Find the scoring run for this chromosome
    scoring_run = find_latest_completed(output_dir, chrom, 'scoring')
    if not scoring_run:
        print(f"  ERROR: No completed scoring run found for {chrom}, skipping")
        return False

    boundaries_file = os.path.join(scoring_run, 'data', 'drop_boundaries.tsv')
    if not os.path.isfile(boundaries_file):
        print(f"  ERROR: No drop_boundaries.tsv in {scoring_run}, skipping")
        return False

    print(f"  Scoring run: {scoring_run}")

    # Parse regions and apply shard slicing
    all_regions = parse_chromosome_drops_tsv(boundaries_file, min_confidence=min_confidence)
    if not all_regions:
        print(f"  WARNING: No regions at confidence >= {min_confidence}, skipping")
        return False

    shard_regions = get_shard_regions(all_regions, shard_idx, n_shards)
    print(f"  Total regions: {len(all_regions)}, shard regions: {len(shard_regions)}")

    if len(shard_regions) != n_done:
        print(f"  WARNING: Region count mismatch! Checkpoint has {n_done}, "
              f"shard slice has {len(shard_regions)}")
        # Use the smaller count to be safe
        n_done = min(n_done, len(shard_regions))

    # Build results list from checkpoint meta + region metadata
    results = []
    for i in range(n_done):
        results.append({
            'region': shard_regions[i],
            'feature_ts': None,  # not needed for finalization
            'top_feature_idx': meta['top_feature_idx'][i],
            'drop_features': [tuple(x) for x in meta['drop_features'][i]],
            'rise_features': [tuple(x) for x in meta['rise_features'][i]],
        })

    # Compute normalization stats by streaming (fixes the OOM)
    print(f"  Computing normalization stats (streaming)...")
    feature_mean, feature_std = compute_norm_stats_streaming(info['chunk_files'])
    stats_path = os.path.join(data_dir, 'feature_norm_stats.npz')
    np.savez_compressed(stats_path, mean=feature_mean, std=feature_std)
    print(f"  Saved: {stats_path}")

    # Find signature features
    signatures = find_signature_features(results)
    print(f"  Signature features: {len(signatures)}")

    # Write sae_results.tsv
    results_file = os.path.join(data_dir, 'sae_results.tsv')
    with open(results_file, 'w') as f:
        f.write("# SAE Feature Analysis of Chromosome Drop Regions\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Regions analyzed: {len(results)}\n")
        f.write(f"# Finalized by: finalize_incomplete_shards.py\n")
        f.write("#\n")
        f.write("region_idx\tgenomic_start\tgenomic_end\tmethod\tconfidence\t"
                "top_features\tdrop_top_features\trise_top_features\n")

        for i, result in enumerate(results):
            reg = result['region']
            top_str = ','.join(str(fid) for fid in result['top_feature_idx'])
            drop_str = ','.join(f"{fid}:{act:.2f}" for fid, act in result['drop_features'][:10])
            rise_str = ','.join(f"{fid}:{act:.2f}" for fid, act in result['rise_features'][:10])
            f.write(f"{i}\t{reg['genomic_start']}\t{reg['genomic_end']}\t"
                    f"{reg['method']}\t{reg['start_confidence']:.4f}\t"
                    f"{top_str}\t{drop_str}\t{rise_str}\n")
    print(f"  Saved: {results_file}")

    # Write signature_features.tsv
    sig_file = os.path.join(data_dir, 'signature_features.tsv')
    with open(sig_file, 'w') as f:
        f.write("# Signature SAE Features (recurring across drop regions)\n")
        f.write(f"# Total signature features: {len(signatures)}\n")
        f.write("#\n")
        f.write("feature_id\tprevalence\tmean_activation\tmax_activation\t"
                "drop_count\trise_count\ttotal_count\t"
                "zscore_count\tmad_count\tzscore_mean_activation\tmad_mean_activation\n")
        for sig in signatures:
            f.write(f"{sig['feature_id']}\t{sig['prevalence']:.4f}\t"
                    f"{sig['mean_activation']:.4f}\t{sig['max_activation']:.4f}\t"
                    f"{sig['drop_count']}\t{sig['rise_count']}\t{sig['total_count']}\t"
                    f"{sig.get('zscore_count', 0)}\t{sig.get('mad_count', 0)}\t"
                    f"{sig.get('zscore_mean_activation', 0.0):.4f}\t"
                    f"{sig.get('mad_mean_activation', 0.0):.4f}\n")
    print(f"  Saved: {sig_file}")

    # Write run_metadata.json
    run_meta = {
        "script": "finalize_incomplete_shards.py",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "chrom": chrom,
            "shard": f"{shard_idx}/{n_shards}",
            "min_confidence": min_confidence,
            "scoring_run": scoring_run,
            "original_dir": shard_dir,
        },
        "results": {
            "regions_analyzed": len(results),
            "signature_features": len(signatures),
        },
    }
    meta_path = os.path.join(data_dir, 'run_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(run_meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # Write COMPLETED sentinel
    write_completed(shard_dir, "finalize_incomplete_shards.py", wall_time_s=0)
    print(f"  COMPLETED written!")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", default="results/",
                        help="Root results directory (default: results/)")
    parser.add_argument("--chrom", default=None,
                        help="Only finalize this chromosome (e.g. chr16)")
    parser.add_argument("--n_shards", type=int, default=36,
                        help="Only process shards with this total (default: 36)")
    parser.add_argument("--min_confidence", type=float, default=8.0,
                        help="Minimum confidence threshold (default: 8.0)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be finalized without doing it")
    args = parser.parse_args()

    print(f"Scanning for incomplete shards in {args.output_dir}...")
    print(f"  Filter: n_shards={args.n_shards}, chrom={args.chrom or 'all'}")

    incomplete = find_incomplete_shards(
        args.output_dir,
        chrom_filter=args.chrom,
        n_shards_filter=args.n_shards,
    )

    if not incomplete:
        print("No incomplete shards found with chunk data. Nothing to do.")
        return

    print(f"\nFound {len(incomplete)} incomplete shards with chunk data:")
    by_chrom = defaultdict(list)
    for info in incomplete:
        by_chrom[info['chrom']].append(info['shard_idx'])
    for chrom in sorted(by_chrom.keys()):
        shards = by_chrom[chrom]
        print(f"  {chrom}: {len(shards)} shards — {shards}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Finalize each shard
    success = 0
    failed = 0
    for info in incomplete:
        try:
            if finalize_shard(info, args.output_dir, args.min_confidence):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR finalizing {info['chrom']} shard {info['shard_idx']}: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Done. Finalized: {success}, Failed: {failed}")


if __name__ == '__main__':
    main()
