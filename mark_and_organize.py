#!/usr/bin/env python3
"""
mark_and_organize.py

1. Mark fully-extracted shards as COMPLETED (where n_done matches expected)
2. Find best merge per chromosome (prefer fast/deduplicated)
3. Create results/conf8_ready/<chrom> symlinks to best clean merges

Usage:
    python mark_and_organize.py --output_dir results/
    python mark_and_organize.py --output_dir results/ --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys
import zipfile
from datetime import datetime

HUMAN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
N_SHARDS = 36


def mark_complete_shards(output_dir, dry_run=False):
    """Find shards with full extraction but no COMPLETED, and mark them."""
    marked = 0
    for chrom in HUMAN_CHROMS:
        sae_root = os.path.join(output_dir, chrom, "sae")
        if not os.path.isdir(sae_root):
            continue

        # First pass: find expected n_done from any COMPLETED shard
        expected_n_done = {}  # shard_total -> n_done
        pattern = re.compile(r"conf8\.0.*shard(\d+)of(\d+)")
        for entry in os.listdir(sae_root):
            m = pattern.search(entry)
            if not m:
                continue
            shard_total = int(m.group(2))
            full = os.path.join(sae_root, entry)
            if not os.path.isfile(os.path.join(full, "COMPLETED")):
                continue
            meta_path = os.path.join(full, "data", "_checkpoint_meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    n_done = meta.get("n_done", 0)
                    if shard_total not in expected_n_done or n_done > expected_n_done[shard_total]:
                        expected_n_done[shard_total] = n_done
                except (json.JSONDecodeError, KeyError):
                    pass

        if N_SHARDS not in expected_n_done:
            # Try to get expected from chunk file count comparison
            continue

        expected = expected_n_done[N_SHARDS]

        # Second pass: mark partial shards that match expected
        for entry in os.listdir(sae_root):
            m = pattern.search(entry)
            if not m:
                continue
            shard_total = int(m.group(2))
            if shard_total != N_SHARDS:
                continue
            full = os.path.join(sae_root, entry)
            if os.path.isfile(os.path.join(full, "COMPLETED")):
                continue

            # Check if this shard has chunk data
            data_dir = os.path.join(full, "data")
            if not os.path.isdir(data_dir):
                continue
            chunk_files = glob.glob(os.path.join(data_dir, "_chunk_*.npz"))
            meta_path = os.path.join(data_dir, "_checkpoint_meta.json")
            if not chunk_files or not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                n_done = meta.get("n_done", 0)
            except (json.JSONDecodeError, KeyError):
                continue

            if n_done >= expected:
                shard_idx = int(m.group(1))
                if dry_run:
                    print(f"  WOULD MARK {chrom} shard {shard_idx} ({n_done}/{expected} done)")
                else:
                    payload = {
                        "completed_at": datetime.now().isoformat(),
                        "script": "mark_and_organize.py (retroactive)",
                        "wall_time_s": 0,
                        "note": f"Marked retroactively: n_done={n_done} matches expected={expected}",
                    }
                    with open(os.path.join(full, "COMPLETED"), "w") as f:
                        json.dump(payload, f, indent=2)
                        f.write("\n")
                    print(f"  MARKED {chrom} shard {shard_idx} ({n_done}/{expected} done)")
                marked += 1

    return marked


def find_all_merges(output_dir, chrom):
    """Find ALL merge directories for a chromosome (not just latest)."""
    sae_root = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_root):
        return []

    merges = []
    for entry in sorted(os.listdir(sae_root)):
        if "merged" not in entry or "conf8.0" not in entry:
            continue
        full = os.path.join(sae_root, entry)
        if not os.path.isdir(full):
            continue

        info = {
            "dir_name": entry,
            "path": full,
            "has_completed": os.path.isfile(os.path.join(full, "COMPLETED")),
            "is_fast": "_fast" in entry,
            "npz_valid": False,
            "region_count": 0,
            "has_norm_stats": os.path.isfile(os.path.join(full, "data", "feature_norm_stats.npz")),
            "shards_used": [],
            "n_unique_indices": 0,
            "has_duplicates": False,
        }

        # Read run_metadata.json
        meta_path = os.path.join(full, "data", "run_metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                info["shards_used"] = sorted(meta.get("shards_used", []))
                info["n_unique_indices"] = len(set(info["shards_used"]))
            except (json.JSONDecodeError, KeyError):
                pass

        # Check NPZ
        npz_path = os.path.join(full, "data", "feature_matrices.npz")
        if os.path.isfile(npz_path):
            try:
                zf = zipfile.ZipFile(npz_path, 'r')
                names = zf.namelist()
                info["region_count"] = sum(1 for n in names if n.startswith("region_"))
                zf.close()
                info["npz_valid"] = True
            except (zipfile.BadZipFile, Exception):
                pass

        # Parse NofM from directory name
        m = re.search(r"merged(\d+)of(\d+)", entry)
        if m:
            info["n_dirs_in_name"] = int(m.group(1))
            info["n_expected_in_name"] = int(m.group(2))
            # Old script has no dedup: N > M means duplicate shard dirs were merged
            if info["n_dirs_in_name"] > info["n_expected_in_name"] and not info["is_fast"]:
                info["has_duplicates"] = True
            # For coverage: fast merges dedup, so n_dirs = unique indices
            # For old merges without metadata, n_dirs == n_expected means likely clean
            if info["is_fast"]:
                info["n_unique_indices"] = max(info["n_unique_indices"], info["n_dirs_in_name"])
            elif info["n_dirs_in_name"] == info["n_expected_in_name"] and info["n_unique_indices"] == 0:
                # Old merge with no metadata but N == M — assume clean (no dups)
                info["n_unique_indices"] = info["n_dirs_in_name"]

        merges.append(info)

    return merges


def pick_best_merge(merges):
    """Pick the best merge from candidates. Prefer: fast + valid + no dups + all 36 indices."""
    valid = [m for m in merges if m["npz_valid"]]
    if not valid:
        return None

    def score(m):
        s = 0
        if m["is_fast"]:
            s += 1000  # fast merges have dedup
        if not m["has_duplicates"]:
            s += 500
        if m["n_unique_indices"] >= N_SHARDS:
            s += 200
        if m["has_norm_stats"]:
            s += 100
        if m["has_completed"]:
            s += 50
        s += m["region_count"] / 100000  # tiebreak by region count
        return s

    valid.sort(key=score, reverse=True)
    return valid[0]


def organize(output_dir, dry_run=False):
    """Create results/conf8_ready/ with symlinks to best merges."""
    ready_dir = os.path.join(output_dir, "conf8_ready")
    if not dry_run:
        os.makedirs(ready_dir, exist_ok=True)

    results = []
    for chrom in HUMAN_CHROMS:
        # Count available shard indices (both COMPLETED and with-data)
        sae_root = os.path.join(output_dir, chrom, "sae")
        completed_indices = set()
        data_indices = set()
        if os.path.isdir(sae_root):
            pattern = re.compile(r"conf8\.0.*shard(\d+)of(\d+)")
            for entry in os.listdir(sae_root):
                m = pattern.search(entry)
                if not m:
                    continue
                if int(m.group(2)) != N_SHARDS:
                    continue
                idx = int(m.group(1))
                full = os.path.join(sae_root, entry)
                if os.path.isfile(os.path.join(full, "COMPLETED")):
                    completed_indices.add(idx)
                # Also check for chunk data (partial but usable shards)
                data_dir = os.path.join(full, "data")
                if os.path.isdir(data_dir):
                    chunks = glob.glob(os.path.join(data_dir, "_chunk_*.npz"))
                    if chunks:
                        data_indices.add(idx)
        data_indices |= completed_indices  # completed implies has data

        merges = find_all_merges(output_dir, chrom)
        best = pick_best_merge(merges)

        # Check if any merge already covers all 36 indices (even if shard sentinels are missing)
        has_complete_merge = (best is not None and best["npz_valid"] and
                             best["n_unique_indices"] >= N_SHARDS and
                             not best["has_duplicates"])

        label = "NO_DATA"
        if has_complete_merge:
            label = "COMPLETE"
        elif len(data_indices) == 0:
            label = "NO_DATA"
        elif len(data_indices) >= N_SHARDS:
            # All indices have data but no clean merge yet
            if best is None:
                label = "NEEDS_REMERGE (no valid merge)"
            elif best["has_duplicates"]:
                label = "NEEDS_REMERGE (duplicates)"
            elif best["n_unique_indices"] < N_SHARDS:
                label = f"NEEDS_REMERGE ({best['n_unique_indices']}/{N_SHARDS} indices in merge)"
            else:
                label = "COMPLETE"
        else:
            label = f"NEEDS_GPU ({len(data_indices)}/{N_SHARDS})"

        # Create symlink for COMPLETE chromosomes
        if label == "COMPLETE" and best:
            link_path = os.path.join(ready_dir, chrom)
            if dry_run:
                print(f"  WOULD LINK {chrom} -> {best['dir_name']}")
            else:
                if os.path.islink(link_path) or os.path.exists(link_path):
                    os.remove(link_path)
                os.symlink(os.path.abspath(best["path"]), link_path)

        merge_desc = ""
        if best:
            merge_desc = f"{best['dir_name']} ({best['region_count']} regions)"

        results.append({
            "chrom": chrom,
            "shards": len(completed_indices),
            "label": label,
            "merge": merge_desc,
        })

    # Print table
    print()
    print(f"{'Chrom':<7} {'Shards':<8} {'Label':<35} {'Best Merge'}")
    print("-" * 110)
    for r in results:
        print(f"{r['chrom']:<7} {r['shards']:<8} {r['label']:<35} {r['merge']}")

    # Summary
    from collections import Counter
    labels = [r["label"].split(" (")[0] for r in results]
    counts = Counter(labels)
    print()
    for lbl in ["COMPLETE", "NEEDS_REMERGE", "NEEDS_GPU", "NO_DATA"]:
        if counts.get(lbl, 0) > 0:
            chroms = [r["chrom"] for r in results if r["label"].startswith(lbl)]
            print(f"  {lbl}: {counts[lbl]} — {', '.join(chroms)}")
    print()

    if not dry_run:
        print(f"Symlinks created in: {ready_dir}/")
        # Write status JSON
        status_path = os.path.join(ready_dir, "status.json")
        with open(status_path, "w") as f:
            json.dump({"generated_at": datetime.now().isoformat(), "chromosomes": results}, f, indent=2)
        print(f"Status written to: {status_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default="results/",
                        help="Root results directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip("/")

    print("=" * 60)
    print("Step 1: Marking fully-extracted shards as COMPLETED")
    print("=" * 60)
    n_marked = mark_complete_shards(output_dir, dry_run=args.dry_run)
    print(f"\n{'Would mark' if args.dry_run else 'Marked'} {n_marked} shards\n")

    print("=" * 60)
    print("Step 2: Finding best merges & organizing into conf8_ready/")
    print("=" * 60)
    organize(output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
