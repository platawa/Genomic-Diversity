#!/usr/bin/env python3
"""
investigate_all.py — Deep investigation of ALL conf8.0 shards and merges.

For each chromosome:
  - Lists ALL shard directories (complete + partial), their n_done, chunk count
  - Identifies which shard indices are covered (0-35)
  - Lists ALL merge directories with full details
  - Reports whether any existing merge already covers all 36 indices cleanly
"""

import glob
import json
import os
import re
import sys
import zipfile

HUMAN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
N_SHARDS = 36


def investigate_shards(output_dir, chrom):
    """List ALL conf8.0 shard dirs for a chromosome with full details."""
    sae_root = os.path.join(output_dir, chrom, "sae")
    if not os.path.isdir(sae_root):
        return [], set()

    pattern = re.compile(r"conf8\.0.*shard(\d+)of(\d+)")
    shards = []
    for entry in sorted(os.listdir(sae_root)):
        m = pattern.search(entry)
        if not m:
            continue
        shard_idx = int(m.group(1))
        shard_total = int(m.group(2))
        if shard_total != N_SHARDS:
            continue

        full = os.path.join(sae_root, entry)
        has_completed = os.path.isfile(os.path.join(full, "COMPLETED"))
        data_dir = os.path.join(full, "data")
        chunk_files = glob.glob(os.path.join(data_dir, "_chunk_*.npz")) if os.path.isdir(data_dir) else []
        meta_path = os.path.join(data_dir, "_checkpoint_meta.json") if os.path.isdir(data_dir) else ""

        n_done = None
        n_total = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                n_done = meta.get("n_done")
                n_total = meta.get("n_total") or meta.get("n_regions")
            except:
                pass

        shards.append({
            "idx": shard_idx,
            "dir": entry,
            "completed": has_completed,
            "n_chunks": len(chunk_files),
            "n_done": n_done,
            "n_total": n_total,
            "has_data": len(chunk_files) > 0 or os.path.isfile(os.path.join(data_dir, "_checkpoint.npz")) if os.path.isdir(data_dir) else False,
        })

    # Which indices have usable data (completed OR has chunk data)?
    indices_with_data = set()
    indices_completed = set()
    for s in shards:
        if s["has_data"]:
            indices_with_data.add(s["idx"])
        if s["completed"]:
            indices_completed.add(s["idx"])

    return shards, indices_with_data, indices_completed


def investigate_merges(output_dir, chrom):
    """List ALL conf8.0 merge dirs for a chromosome with full details."""
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
            "has_completed": os.path.isfile(os.path.join(full, "COMPLETED")),
            "is_fast": "_fast" in entry,
            "npz_exists": False,
            "npz_valid": False,
            "region_count": 0,
            "has_norm_stats": os.path.isfile(os.path.join(full, "data", "feature_norm_stats.npz")),
            "shards_used": [],
            "shards_complete": [],
            "shards_partial": [],
            "n_unique_indices": 0,
        }

        # Parse NofM from name
        m = re.search(r"merged(\d+)of(\d+)", entry)
        if m:
            info["n_in_name"] = int(m.group(1))
            info["m_in_name"] = int(m.group(2))

        # Read run_metadata.json
        meta_path = os.path.join(full, "data", "run_metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                info["shards_used"] = sorted(meta.get("shards_used", []))
                info["shards_complete"] = sorted(meta.get("shards_complete", []))
                info["shards_partial"] = sorted(meta.get("shards_partial", []))
                info["n_unique_indices"] = len(set(info["shards_used"]))
            except:
                pass

        # Check NPZ
        npz_path = os.path.join(full, "data", "feature_matrices.npz")
        if os.path.isfile(npz_path):
            info["npz_exists"] = True
            info["npz_size_gb"] = os.path.getsize(npz_path) / 1e9
            try:
                zf = zipfile.ZipFile(npz_path, 'r')
                names = zf.namelist()
                info["region_count"] = sum(1 for n in names if n.startswith("region_"))
                if zf.infolist():
                    ct = zf.infolist()[0].compress_type
                    info["compression"] = "STORED" if ct == 0 else "DEFLATED"
                zf.close()
                info["npz_valid"] = True
            except Exception as e:
                info["npz_error"] = str(e)

        merges.append(info)

    return merges


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "results/"

    focus_chroms = sys.argv[2:] if len(sys.argv) > 2 else HUMAN_CHROMS

    for chrom in focus_chroms:
        print(f"\n{'='*80}")
        print(f"  {chrom}")
        print(f"{'='*80}")

        shards, indices_with_data, indices_completed = investigate_shards(output_dir, chrom)

        if not shards:
            print("  No conf8.0 shard directories found.")
            continue

        # Summarize shard coverage
        missing_data = sorted(set(range(N_SHARDS)) - indices_with_data)
        missing_completed = sorted(set(range(N_SHARDS)) - indices_completed)

        print(f"\n  SHARDS: {len(indices_completed)}/{N_SHARDS} COMPLETED, "
              f"{len(indices_with_data)}/{N_SHARDS} with data")

        if missing_data:
            print(f"  Missing data for indices: {missing_data}")
        if missing_completed and missing_completed != missing_data:
            only_missing_completed = sorted(indices_with_data - indices_completed)
            if only_missing_completed:
                print(f"  Have data but no COMPLETED: {only_missing_completed}")

        # Show per-index detail for non-trivial cases
        # Group shards by index
        by_idx = {}
        for s in shards:
            by_idx.setdefault(s["idx"], []).append(s)

        # Show detail for indices with multiple dirs or incomplete
        show_detail = (len(indices_with_data) < N_SHARDS or
                       len(indices_completed) < len(indices_with_data))
        if show_detail:
            print(f"\n  Per-shard detail (incomplete or multi-dir):")
            for idx in sorted(by_idx.keys()):
                entries = by_idx[idx]
                if len(entries) > 1 or not entries[0]["completed"]:
                    for s in entries:
                        status = "COMPLETED" if s["completed"] else ("HAS_DATA" if s["has_data"] else "EMPTY")
                        done_str = f"n_done={s['n_done']}/{s['n_total']}" if s["n_done"] is not None else ""
                        print(f"    shard {s['idx']:2d}: {status:<10} chunks={s['n_chunks']:<3} {done_str}  {s['dir']}")

        # Merges
        merges = investigate_merges(output_dir, chrom)
        if merges:
            print(f"\n  MERGES ({len(merges)} found):")
            for mi in merges:
                valid_str = "VALID" if mi["npz_valid"] else ("CORRUPT" if mi["npz_exists"] else "NO_NPZ")
                comp_str = mi.get("compression", "?")
                norm_str = "has_norm" if mi["has_norm_stats"] else "no_norm"
                compl_str = "COMPLETED" if mi["has_completed"] else "no_sentinel"
                size_str = f"{mi.get('npz_size_gb', 0):.1f}GB" if mi["npz_exists"] else ""

                # Coverage
                if mi["shards_used"]:
                    unique = set(mi["shards_used"])
                    cov_str = f"covers {len(unique)}/36 indices"
                    if len(mi["shards_used"]) > len(unique):
                        cov_str += f" (DUPS: {len(mi['shards_used'])}-{len(unique)}={len(mi['shards_used'])-len(unique)} extra)"
                elif mi.get("n_in_name"):
                    cov_str = f"name says {mi['n_in_name']}of{mi['m_in_name']}"
                    if not mi["is_fast"] and mi["n_in_name"] > mi["m_in_name"]:
                        cov_str += " (LIKELY DUPS — old script)"
                else:
                    cov_str = "no metadata"

                fast_str = "FAST" if mi["is_fast"] else "OLD"
                print(f"    {mi['dir_name']}")
                print(f"      {fast_str} {valid_str} {comp_str} {size_str} | {mi['region_count']} regions | {cov_str} | {norm_str} | {compl_str}")
                if mi.get("npz_error"):
                    print(f"      ERROR: {mi['npz_error']}")
                if mi["shards_partial"]:
                    print(f"      Partial shards in merge: {mi['shards_partial']}")

            # Verdict: does any merge already cover all 36 cleanly?
            best = None
            for mi in merges:
                if not mi["npz_valid"]:
                    continue
                if mi["shards_used"]:
                    unique = set(mi["shards_used"])
                    has_dups = len(mi["shards_used"]) > len(unique)
                    if len(unique) >= N_SHARDS and not has_dups:
                        best = mi
                        break
                elif mi["is_fast"] and mi.get("n_in_name", 0) >= N_SHARDS:
                    best = mi
                    break
                elif not mi["is_fast"] and mi.get("n_in_name") == mi.get("m_in_name"):
                    best = mi  # old merge with N==M, possibly clean
                    # don't break — prefer fast if available

            if best:
                print(f"\n  VERDICT: HAS CLEAN MERGE → {best['dir_name']} ({best['region_count']} regions)")
            elif len(indices_with_data) >= N_SHARDS:
                print(f"\n  VERDICT: ALL 36 INDICES HAVE DATA → needs fast re-merge")
            elif len(indices_with_data) > 0:
                print(f"\n  VERDICT: {len(indices_with_data)}/36 indices have data → needs GPU for {sorted(set(range(N_SHARDS)) - indices_with_data)}")
            else:
                print(f"\n  VERDICT: NO DATA")
        else:
            print(f"\n  No merges found.")
            if len(indices_with_data) >= N_SHARDS:
                print(f"  VERDICT: ALL 36 INDICES HAVE DATA → needs merge")
            elif len(indices_with_data) > 0:
                print(f"  VERDICT: {len(indices_with_data)}/36 → needs GPU")
            else:
                print(f"  VERDICT: NO DATA")


if __name__ == "__main__":
    main()
