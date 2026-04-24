#!/usr/bin/env python3
"""
regenerate_cluster_tsv.py

Rebuild a stale `cluster_assignments.tsv` so its row count matches the existing
`maxpooled_vectors.npy` in the same latent_analysis_* directory.

Root cause this addresses:
  Pool stage (`load_and_pool_from_shards` in analyze_sae_regions.py) walks
  COMPLETED shards and allocates rows using _checkpoint_meta.json["n_done"].
  Cluster stage does its OWN independent walk at cluster-stage time. If these
  two runs see different shard-COMPLETED snapshots, the two files diverge.
  Result: maxpooled_vectors.npy has more rows than cluster_assignments.tsv.

Vectors are correct (pool stage state is current). Only the metadata TSV is
stale. This script rebuilds the TSV directly from shard data in the same
order the pool stage used, so alignment is byte-exact.

Output TSV has the columns genome_sae_tsne.py reads via load_region_metadata:
  genomic_start, genomic_end, method, confidence, region_length
plus passthrough of any other columns present in the source sae_results.tsv.
It omits cluster_id and embedding_* columns (those are recomputed genome-wide
by genome_sae_tsne.py, so blanks are fine).

Usage:
    python tools/regenerate_cluster_tsv.py --chrom chr10 \\
        --latent_subdir latent_analysis_prenorm --dry-run

    python tools/regenerate_cluster_tsv.py --chrom chr1 \\
        --latent_subdir latent_analysis_prenorm \\
        --backup-dir results/chr1/sae/latent_analysis_prenorm/data/_stale_backup_20260423
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

SHARD_RE = re.compile(r"shard(\d+)of(\d+)")


def find_completed_shards(sae_root: Path, n_shards: int):
    """Return list of (shard_idx, shard_path) for COMPLETED shards with
    totals matching `n_shards`, deduped by index (newest dir wins) — matching
    load_and_pool_from_shards semantics exactly."""
    candidates = []
    for entry in sorted(os.listdir(sae_root)):
        if "merged" in entry:
            continue
        m = SHARD_RE.search(entry)
        if not m:
            continue
        if int(m.group(2)) != n_shards:
            continue
        full = sae_root / entry
        if (full / "COMPLETED").is_file():
            candidates.append((int(m.group(1)), full))
    seen = {}
    for idx, path in candidates:
        seen[idx] = path  # later (newer timestamp in sorted order) wins
    return sorted(seen.items())


def read_n_done(shard_path: Path) -> int:
    meta = shard_path / "data" / "_checkpoint_meta.json"
    with open(meta) as f:
        return int(json.load(f)["n_done"])


def find_boundaries_for_shard(shard_path: Path) -> Path:
    """Resolve drop_boundaries.tsv via shard's source.json."""
    source = shard_path / "source.json"
    if not source.is_file():
        raise FileNotFoundError(f"no source.json in {shard_path}")
    with open(source) as f:
        src = json.load(f)
    rel = src.get("boundaries")
    if not rel:
        raise KeyError(f"no 'boundaries' key in {source}")
    # Path in source.json is relative to the shard dir
    return (shard_path / rel).resolve()


def load_filtered_sorted_boundaries(boundaries_tsv: Path,
                                    min_confidence: float):
    """Read drop_boundaries.tsv, filter by start_confidence >= min_confidence,
    and sort by start_confidence DESCENDING — exactly matching
    `sae_utils.parse_chromosome_drops_tsv` + the sort in run_sae_fast.py.

    Notably does NOT deduplicate — the SAE pipeline accepts duplicates, and
    the shard math is applied to the filtered+sorted list directly."""
    regions = []
    with open(boundaries_tsv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "chrom":  # header row
                continue
            if len(parts) < 11:
                continue
            try:
                gs = int(parts[3]); ge = int(parts[4]); rl = int(parts[5])
                m = parts[6]
                start_conf = float(parts[7])
            except (ValueError, IndexError):
                continue
            if start_conf < min_confidence:
                continue
            regions.append({
                "genomic_start": gs, "genomic_end": ge,
                "region_length": rl, "method": m,
                "confidence": start_conf,
            })
    # Sort by start_confidence DESCENDING (matches run_sae_fast.py)
    regions.sort(key=lambda r: -r["confidence"])
    return regions


_CONF_RE = re.compile(r"conf(\d+\.?\d*)")


def parse_conf_threshold(shard_path: Path) -> float:
    """Extract the conf threshold from a shard dirname
    (e.g. '20260323_002921_all_conf8.0_shard0of36' → 8.0)."""
    m = _CONF_RE.search(shard_path.name)
    if not m:
        raise ValueError(f"cannot parse conf from {shard_path.name}")
    return float(m.group(1))


def boundaries_row_to_tsv_line(r: dict, region_idx: int,
                               header_cols: list) -> str:
    """Produce a TSV line matching the given header by filling available
    columns and blanks elsewhere. genome_sae_tsne.py only needs
    genomic_start/end/method/confidence/region_length; other columns can be
    empty strings."""
    vals = []
    for c in header_cols:
        if c == "region_idx":
            vals.append(str(region_idx))
        elif c == "genomic_start":
            vals.append(str(r["genomic_start"]))
        elif c == "genomic_end":
            vals.append(str(r["genomic_end"]))
        elif c == "region_length":
            vals.append(str(r["region_length"]))
        elif c == "method":
            vals.append(r["method"])
        elif c == "confidence":
            vals.append(f"{r['confidence']:.4f}")
        else:
            vals.append("")
    return "\t".join(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--latent_subdir", required=True,
                    help="e.g. latent_analysis_prenorm")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--n_shards", type=int, default=36,
                    help="Expected total shard count (matches pool stage "
                         "--n_shards arg). Default: 36 (human conf8.0).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts and a sample row; do not write.")
    ap.add_argument("--backup-dir", default=None,
                    help="Directory to move existing cluster_assignments.tsv "
                         "into before writing new one. Created if missing.")
    ap.add_argument("--compare-existing", action="store_true",
                    help="Dry-run mode: also compare regenerated coords "
                         "against the existing cluster_assignments.tsv "
                         "to verify alignment assumption.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    sae_root = results_dir / args.chrom / "sae"
    data_dir = sae_root / args.latent_subdir / "data"
    vectors_path = data_dir / "maxpooled_vectors.npy"
    tsv_path = data_dir / "cluster_assignments.tsv"

    if not vectors_path.is_file():
        sys.exit(f"ERROR: no maxpooled_vectors.npy at {vectors_path}")

    n_vectors = int(np.load(vectors_path, mmap_mode="r").shape[0])
    print(f"[{args.chrom}] maxpooled_vectors.npy rows: {n_vectors}")

    shards = find_completed_shards(sae_root, args.n_shards)
    if not shards:
        sys.exit(f"ERROR: no COMPLETED shards of{args.n_shards} under {sae_root}")
    print(f"[{args.chrom}] COMPLETED shards of{args.n_shards} found: {len(shards)}")

    # Pre-scan shards: read n_done for each; also find any shard that HAS
    # sae_results.tsv (to establish the canonical header) and load dedup'd
    # drop_boundaries lazily if needed.
    per_shard_info = []  # (shard_idx, shard_path, n_done, tsv_present)
    for shard_idx, shard_path in shards:
        n_done = read_n_done(shard_path)
        tsv_present = (shard_path / "data" / "sae_results.tsv").is_file()
        per_shard_info.append((shard_idx, shard_path, n_done, tsv_present))
    n_broken = sum(1 for *_, p in per_shard_info if not p)
    print(f"[{args.chrom}] shards with sae_results.tsv: "
          f"{len(per_shard_info) - n_broken}/{len(per_shard_info)} "
          f"({n_broken} broken — will derive from drop_boundaries.tsv)")

    # Lazy-load drop_boundaries if any shard is broken
    sorted_regions = None
    if n_broken > 0:
        # All shards for a given chrom point to the same scoring run
        boundaries = find_boundaries_for_shard(per_shard_info[0][1])
        if not boundaries.is_file():
            sys.exit(f"ERROR: drop_boundaries at {boundaries} does not exist")
        conf_thresh = parse_conf_threshold(per_shard_info[0][1])
        sorted_regions = load_filtered_sorted_boundaries(
            boundaries, min_confidence=conf_thresh
        )
        print(f"[{args.chrom}] drop_boundaries: {boundaries}")
        print(f"[{args.chrom}] conf threshold: {conf_thresh}")
        print(f"[{args.chrom}] regions after filter+sort: {len(sorted_regions)}")

    # Build metadata rows
    all_lines = []
    header = None
    total = 0
    cumulative = 0
    for shard_idx, shard_path, n_done, tsv_present in per_shard_info:
        if tsv_present:
            shard_tsv = shard_path / "data" / "sae_results.tsv"
            with open(shard_tsv) as f:
                # Skip comment lines; first non-comment line is header
                this_header = None
                for line in f:
                    stripped = line.rstrip("\n")
                    if stripped.startswith("#") or not stripped.strip():
                        continue
                    this_header = stripped
                    break
                if this_header is None:
                    sys.exit(f"ERROR: no header line in {shard_tsv}")
                if header is None:
                    header = this_header
                elif this_header != header:
                    sys.exit(f"ERROR: header mismatch in {shard_tsv}\n"
                             f"  got: {this_header}\n  expected: {header}")
                rows = []
                for line in f:
                    stripped = line.rstrip("\n")
                    if stripped.startswith("#") or not stripped.strip():
                        continue
                    rows.append(stripped)
                    if len(rows) >= n_done:
                        break
                if len(rows) < n_done:
                    sys.exit(f"ERROR: shard{shard_idx} sae_results.tsv has "
                             f"fewer rows ({len(rows)}) than n_done ({n_done})")
            all_lines.extend(rows)
            label = "tsv"
        else:
            # Derive from drop_boundaries via shard slicing math.
            # run_sae_fast.py uses: shard_size = ceil(N / n_shards),
            # slice = regions[shard_idx * shard_size : (shard_idx+1) * shard_size]
            # where N is len(regions) after confidence filter.
            assert sorted_regions is not None
            N = len(sorted_regions)
            shard_size = (N + args.n_shards - 1) // args.n_shards
            start = shard_idx * shard_size
            end = min(start + shard_size, N)
            expected_len = end - start
            if n_done > expected_len:
                sys.exit(
                    f"ERROR: shard{shard_idx} n_done={n_done} > expected "
                    f"slice len {expected_len}. N={N}, shard_size={shard_size}, "
                    f"start={start}, end={end}. Shard math inconsistent."
                )
            # Partial-and-broken shard: take first n_done regions from slice
            end = start + n_done
            if header is None:
                header = "region_idx\tgenomic_start\tgenomic_end\t" \
                         "method\tconfidence\tregion_length"
            header_cols = header.split("\t")
            slice_regions = sorted_regions[start:end]
            for i, r in enumerate(slice_regions):
                all_lines.append(
                    boundaries_row_to_tsv_line(r, start + i, header_cols)
                )
            label = "derived"

        total += n_done
        cumulative += n_done
        print(f"  shard{shard_idx:>2}: n_done={n_done} [{label}]  "
              f"({shard_path.name})")

    print(f"[{args.chrom}] regenerated rows: {total}")
    print(f"[{args.chrom}] vectors rows:     {n_vectors}")

    if total != n_vectors:
        sys.exit(f"FAIL: regenerated rows ({total}) != vectors rows "
                 f"({n_vectors}). Refusing to write. Investigate shard state.")
    print(f"[{args.chrom}] MATCH — row counts align.")

    # Optional cross-check against existing TSV (for verification chrom)
    if args.compare_existing and tsv_path.is_file():
        import csv
        print(f"[{args.chrom}] comparing regenerated coords to existing TSV...")
        existing_coords = []
        with open(tsv_path) as f:
            # Skip comment lines starting with '#'
            for line in f:
                if not line.startswith("#"):
                    hdr = line.rstrip("\n").split("\t")
                    break
            reader = csv.DictReader(f, fieldnames=hdr, delimiter="\t")
            for row in reader:
                existing_coords.append(
                    (row.get("genomic_start", ""),
                     row.get("genomic_end", ""))
                )
        # Build regenerated coord list
        regen_coords = []
        src_cols = header.split("\t")
        gs_i = src_cols.index("genomic_start") if "genomic_start" in src_cols else None
        ge_i = src_cols.index("genomic_end") if "genomic_end" in src_cols else None
        if gs_i is None or ge_i is None:
            print(f"  (skipping compare: source TSV lacks genomic_start/end cols)")
        else:
            for line in all_lines:
                parts = line.split("\t")
                regen_coords.append((parts[gs_i], parts[ge_i]))
            if len(regen_coords) != len(existing_coords):
                print(f"  len diff: regen={len(regen_coords)} existing={len(existing_coords)}")
            else:
                diffs = sum(1 for a, b in zip(regen_coords, existing_coords) if a != b)
                if diffs == 0:
                    print(f"  ✅ all {len(regen_coords)} coord rows match exactly.")
                else:
                    print(f"  ⚠ {diffs}/{len(regen_coords)} coord rows differ. "
                          f"First 3 divergences:")
                    shown = 0
                    for i, (a, b) in enumerate(zip(regen_coords, existing_coords)):
                        if a != b and shown < 3:
                            print(f"    row {i}: regen={a} existing={b}")
                            shown += 1

    if args.dry_run:
        print(f"[{args.chrom}] DRY RUN — not writing. Sample header:")
        print(f"  {header}")
        if all_lines:
            print(f"  first row: {all_lines[0][:200]}")
        return

    # Backup existing TSV if present
    if tsv_path.is_file():
        if args.backup_dir is None:
            sys.exit("ERROR: cluster_assignments.tsv exists; pass --backup-dir "
                     "to move it aside before rewriting.")
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        dst = backup_dir / "cluster_assignments.tsv"
        if dst.exists():
            sys.exit(f"ERROR: backup target already exists: {dst}")
        shutil.move(str(tsv_path), str(dst))
        print(f"[{args.chrom}] moved stale TSV → {dst}")
        # Also move companion embedding + analysis_metadata if present (they
        # reflect the same stale state)
        for name in ("embedding_tsne.npy", "embedding_umap.npy",
                     "analysis_metadata.json"):
            src = data_dir / name
            if src.is_file():
                shutil.move(str(src), str(backup_dir / name))
                print(f"  also moved {name}")

    # Write new TSV
    with open(tsv_path, "w") as f:
        f.write(f"# Regenerated by regenerate_cluster_tsv.py on "
                f"{__import__('datetime').datetime.now().isoformat()}\n")
        f.write(f"# Source: {len(shards)} COMPLETED shards under "
                f"{sae_root.relative_to(results_dir.parent)}\n")
        f.write(f"# Rows: {total} (matches maxpooled_vectors.npy)\n")
        f.write("#\n")
        f.write(header + "\n")
        for line in all_lines:
            f.write(line + "\n")
    print(f"[{args.chrom}] WROTE {tsv_path} ({total} rows)")


if __name__ == "__main__":
    main()
