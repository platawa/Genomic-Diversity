#!/usr/bin/env python3
"""
check_conf8_status.py

Check and label conf8.0 merge completeness for all human chromosomes.
Reports shard extraction coverage, merge status, and assigns labels:
  COMPLETE      — 36/36 shards, full merge, norm stats present
  NEEDS_REMERGE — 36/36 shards available but merge is stale/partial
  NEEDS_GPU     — <36 shards extracted
  NO_DATA       — no completed shards

Usage:
    python check_conf8_status.py --output_dir results/
    python check_conf8_status.py --output_dir results/ --json_out results/conf8_status.json
"""

import argparse
import json
import os
import re
import sys
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HUMAN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
N_SHARDS = 36


def find_completed_shard_indices(output_dir, chrom):
    """Find which shard indices (0-35) have at least one COMPLETED dir for conf8.0."""
    sae_root = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_root):
        return set()

    pattern = re.compile(r"conf8\.0.*shard(\d+)of(\d+)")
    completed = set()
    for entry in os.listdir(sae_root):
        m = pattern.search(entry)
        if not m:
            continue
        shard_idx = int(m.group(1))
        shard_total = int(m.group(2))
        if shard_total != N_SHARDS:
            continue
        full = os.path.join(sae_root, entry)
        if os.path.isfile(os.path.join(full, "COMPLETED")):
            completed.add(shard_idx)

    return completed


def find_latest_completed_merge(output_dir, chrom):
    """Find the latest COMPLETED merge dir for conf8.0."""
    sae_root = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_root):
        return None

    candidates = []
    for entry in sorted(os.listdir(sae_root), reverse=True):
        if "merged" not in entry or "conf8.0" not in entry:
            continue
        full = os.path.join(sae_root, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "COMPLETED")):
            candidates.append(full)
            break  # newest first due to reverse sort

    return candidates[0] if candidates else None


def inspect_merge(merge_dir):
    """Inspect a merge directory and return details."""
    info = {
        "dir_name": os.path.basename(merge_dir),
        "shards_used": [],
        "shards_complete": [],
        "shards_partial": [],
        "region_count": 0,
        "compression": "unknown",
        "has_norm_stats": False,
        "npz_valid": False,
        "is_fast": "_fast" in os.path.basename(merge_dir),
    }

    # Read run_metadata.json
    meta_path = os.path.join(merge_dir, "data", "run_metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            info["shards_used"] = sorted(meta.get("shards_used", []))
            info["shards_complete"] = sorted(meta.get("shards_complete", []))
            info["shards_partial"] = sorted(meta.get("shards_partial", []))
        except (json.JSONDecodeError, KeyError):
            pass

    # Check norm stats
    norm_path = os.path.join(merge_dir, "data", "feature_norm_stats.npz")
    info["has_norm_stats"] = os.path.isfile(norm_path)

    # Validate NPZ and count regions
    npz_path = os.path.join(merge_dir, "data", "feature_matrices.npz")
    if os.path.isfile(npz_path):
        try:
            zf = zipfile.ZipFile(npz_path, 'r')
            names = zf.namelist()
            info["region_count"] = sum(1 for n in names if n.startswith("region_"))
            # Check compression type from first entry
            if zf.infolist():
                ct = zf.infolist()[0].compress_type
                info["compression"] = "ZIP_STORED" if ct == 0 else "ZIP_DEFLATED"
            zf.close()
            info["npz_valid"] = True
        except (zipfile.BadZipFile, Exception) as e:
            info["npz_error"] = str(e)

    # File size
    if os.path.isfile(npz_path):
        size_bytes = os.path.getsize(npz_path)
        if size_bytes >= 1e12:
            info["size_str"] = f"{size_bytes / 1e12:.1f}T"
        elif size_bytes >= 1e9:
            info["size_str"] = f"{size_bytes / 1e9:.1f}G"
        else:
            info["size_str"] = f"{size_bytes / 1e6:.0f}M"
    else:
        info["size_str"] = "N/A"

    return info


def check_chromosome(output_dir, chrom):
    """Full status check for one chromosome. Returns a status dict."""
    result = {
        "chrom": chrom,
        "shards_complete": 0,
        "shards_missing": [],
        "merge_dir": None,
        "merge_info": None,
        "label": "NO_DATA",
    }

    # Step 1: Shard coverage
    completed_indices = find_completed_shard_indices(output_dir, chrom)
    result["shards_complete"] = len(completed_indices)
    all_indices = set(range(N_SHARDS))
    result["shards_missing"] = sorted(all_indices - completed_indices)

    # Step 2: Find latest completed merge
    merge_dir = find_latest_completed_merge(output_dir, chrom)

    if merge_dir:
        result["merge_dir"] = os.path.basename(merge_dir)
        result["merge_info"] = inspect_merge(merge_dir)

    # Step 3: Assign label
    if len(completed_indices) == 0:
        result["label"] = "NO_DATA"
    elif len(completed_indices) < N_SHARDS:
        result["label"] = "NEEDS_GPU"
    elif merge_dir is None:
        result["label"] = "NEEDS_REMERGE"
    else:
        mi = result["merge_info"]
        # Check if merge covers all 36 unique shard indices from complete shards
        merge_shard_set = set(mi["shards_used"])
        n_partial = len(mi["shards_partial"])
        has_all_shards_in_merge = (merge_shard_set == all_indices)
        is_clean = (n_partial == 0) and has_all_shards_in_merge

        if is_clean and mi["npz_valid"] and mi["has_norm_stats"]:
            result["label"] = "COMPLETE"
        else:
            result["label"] = "NEEDS_REMERGE"
            reasons = []
            if not has_all_shards_in_merge:
                reasons.append(f"merge has {len(merge_shard_set)}/36 indices")
            if n_partial > 0:
                reasons.append(f"{n_partial} partial shards in merge")
            if not mi["is_fast"]:
                reasons.append("compressed format (no fast merge)")
            if not mi["npz_valid"]:
                reasons.append("NPZ invalid")
            if not mi["has_norm_stats"]:
                reasons.append("missing norm stats")
            result["remerge_reason"] = "; ".join(reasons) if reasons else "unknown"

    return result


def print_table(results):
    """Print a formatted status table."""
    print()
    print(f"{'Chrom':<7} {'Label':<15} {'Shards':<10} {'Merge':<45} {'Regions':>8} {'Size':>7} {'Notes'}")
    print("-" * 120)

    for r in results:
        chrom = r["chrom"]
        label = r["label"]
        shards = f"{r['shards_complete']}/{N_SHARDS}"

        if r["merge_info"]:
            mi = r["merge_info"]
            merge = r["merge_dir"][:43] if r["merge_dir"] else "—"
            regions = str(mi["region_count"])
            size = mi["size_str"]
        else:
            merge = "—"
            regions = "—"
            size = "—"

        notes = ""
        if label == "NEEDS_REMERGE":
            notes = r.get("remerge_reason", "")
        elif label == "NEEDS_GPU":
            missing = r["shards_missing"]
            if len(missing) <= 6:
                notes = f"missing: {missing}"
            else:
                notes = f"missing: {missing[:3]}...+{len(missing)-3} more"

        print(f"{chrom:<7} {label:<15} {shards:<10} {merge:<45} {regions:>8} {size:>7} {notes}")

    print()

    # Summary
    from collections import Counter
    counts = Counter(r["label"] for r in results)
    print("Summary:")
    for label in ["COMPLETE", "NEEDS_REMERGE", "NEEDS_GPU", "NO_DATA"]:
        if counts[label] > 0:
            chroms = [r["chrom"] for r in results if r["label"] == label]
            print(f"  {label}: {counts[label]} — {', '.join(chroms)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check conf8.0 merge completeness for all human chromosomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", default="results/",
                        help="Results directory (default: results/)")
    parser.add_argument("--json_out", default=None,
                        help="Path for JSON output (default: {output_dir}/conf8_status.json)")
    args = parser.parse_args()

    json_path = args.json_out or os.path.join(args.output_dir, "conf8_status.json")

    results = []
    for chrom in HUMAN_CHROMS:
        results.append(check_chromosome(args.output_dir, chrom))

    print_table(results)

    # Write JSON
    output = {
        "generated_at": datetime.now().isoformat(),
        "chromosomes": {r["chrom"]: r for r in results},
        "summary": {},
    }
    from collections import Counter
    counts = Counter(r["label"] for r in results)
    output["summary"] = dict(counts)

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Status written to: {json_path}")


if __name__ == "__main__":
    main()
