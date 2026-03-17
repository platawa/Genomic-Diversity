#!/usr/bin/env python3
"""Compare two global_sae_stats.npz files to validate fused vs standalone equivalence.

Compares output from the standalone scan_sae_global_stats.py (flags: "minmax")
against the fused score_chromosome.py --collect_sae_stats (flags: "fused_minmax")
to confirm the fused approach produces equivalent results.

Usage:
    # Explicit paths
    python tools/compare_sae_stats.py \
        --standalone results/chr22/sae_global_stats/.../data/global_sae_stats.npz \
        --fused results/chr22/sae_global_stats/.../data/global_sae_stats.npz

    # Auto-discover from results directory
    python tools/compare_sae_stats.py --chrom chr22 --results_dir results

    # Save comparison JSON
    python tools/compare_sae_stats.py --chrom chr22 --results_dir results --output comparison.json
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

# Add project root to path for results_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import find_latest_completed


# ---------------------------------------------------------------------------
# Thresholds for PASS/FAIL
# ---------------------------------------------------------------------------
THRESHOLD_ACTIVE_OVERLAP = 0.99    # Jaccard overlap of active features
THRESHOLD_MAX_ABS_DIFF = 0.1      # Max absolute diff for global_max
THRESHOLD_RANK_CORR = 0.99        # Spearman rank correlation for top-100


def find_completed_by_flags(base_dir, chrom, stage, flags_substring):
    """Find the latest COMPLETED run whose directory name contains flags_substring.

    This mirrors find_latest_completed but filters to runs whose name
    (after the YYYYMMDD_HHMMSS_ prefix) contains the given substring.
    """
    stage_dir = os.path.join(base_dir, chrom, stage)
    if not os.path.isdir(stage_dir):
        return None

    matched = []
    for entry in sorted(os.listdir(stage_dir)):
        run_path = os.path.join(stage_dir, entry)
        if not os.path.isdir(run_path):
            continue
        if not os.path.isfile(os.path.join(run_path, "COMPLETED")):
            continue
        # entry format: YYYYMMDD_HHMMSS_<flags>
        parts = entry.split("_", 2)
        if len(parts) >= 3:
            flags = parts[2]
        else:
            flags = ""
        if flags_substring in flags:
            matched.append(entry)

    if not matched:
        return None
    matched.sort()
    return os.path.join(stage_dir, matched[-1])


def resolve_npz_path(path):
    """Accept either a direct .npz path or a run directory containing data/global_sae_stats.npz."""
    if path.endswith(".npz") and os.path.isfile(path):
        return path
    candidate = os.path.join(path, "data", "global_sae_stats.npz")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Cannot find global_sae_stats.npz at '{path}' or '{candidate}'"
    )


def load_stats(npz_path):
    """Load a global_sae_stats.npz and return a dict of arrays."""
    data = np.load(npz_path)
    return {
        "global_min": data["global_min"],
        "global_max": data["global_max"],
        "chunk_max_mean": data["chunk_max_mean"],
        "chunk_max_std": data["chunk_max_std"],
        "n_nonzero_chunks": data["n_nonzero_chunks"],
        "n_chunks": int(data["n_chunks"]),
        "genome_length": int(data["genome_length"]),
    }


def compare_stats(standalone, fused):
    """Run all comparisons and return a results dict."""
    results = {}

    n_features = len(standalone["global_max"])
    results["n_features"] = n_features

    # --- Metadata ---
    results["standalone_n_chunks"] = standalone["n_chunks"]
    results["fused_n_chunks"] = fused["n_chunks"]
    results["standalone_genome_length"] = standalone["genome_length"]
    results["fused_genome_length"] = fused["genome_length"]

    # --- Active features (global_max > 0) ---
    active_s = standalone["global_max"] > 0
    active_f = fused["global_max"] > 0
    n_active_s = int(active_s.sum())
    n_active_f = int(active_f.sum())
    n_both = int((active_s & active_f).sum())
    n_union = int((active_s | active_f).sum())
    overlap = n_both / n_union if n_union > 0 else 1.0

    results["active_features"] = {
        "standalone": n_active_s,
        "fused": n_active_f,
        "both": n_both,
        "union": n_union,
        "overlap_jaccard": round(overlap, 6),
    }

    # --- global_max agreement ---
    gmax_diff = np.abs(standalone["global_max"] - fused["global_max"])
    if np.std(standalone["global_max"]) > 0 and np.std(fused["global_max"]) > 0:
        gmax_corr = float(np.corrcoef(standalone["global_max"], fused["global_max"])[0, 1])
    else:
        gmax_corr = float("nan")

    results["global_max"] = {
        "max_abs_diff": round(float(gmax_diff.max()), 8),
        "mean_abs_diff": round(float(gmax_diff.mean()), 8),
        "correlation": round(gmax_corr, 8),
    }

    # --- global_min agreement ---
    gmin_diff = np.abs(standalone["global_min"] - fused["global_min"])
    if np.std(standalone["global_min"]) > 0 and np.std(fused["global_min"]) > 0:
        gmin_corr = float(np.corrcoef(standalone["global_min"], fused["global_min"])[0, 1])
    else:
        gmin_corr = float("nan")

    results["global_min"] = {
        "max_abs_diff": round(float(gmin_diff.max()), 8),
        "mean_abs_diff": round(float(gmin_diff.mean()), 8),
        "correlation": round(gmin_corr, 8),
    }

    # --- Top-100 features by global_max: rank correlation ---
    top_k = min(100, n_features)
    top_s = set(np.argsort(standalone["global_max"])[-top_k:])
    top_f = set(np.argsort(fused["global_max"])[-top_k:])
    top_overlap = len(top_s & top_f)

    union_top = sorted(top_s | top_f)
    if len(union_top) >= 2:
        rho, pval = spearmanr(
            standalone["global_max"][union_top],
            fused["global_max"][union_top],
        )
        rho = float(rho)
        pval = float(pval)
    else:
        rho = float("nan")
        pval = float("nan")

    results["top100_rank"] = {
        "top_k": top_k,
        "overlap_count": top_overlap,
        "spearman_rho": round(rho, 8),
        "spearman_pval": pval,
    }

    # --- chunk_max_mean agreement ---
    cmm_diff = np.abs(standalone["chunk_max_mean"] - fused["chunk_max_mean"])
    if np.std(standalone["chunk_max_mean"]) > 0 and np.std(fused["chunk_max_mean"]) > 0:
        cmm_corr = float(np.corrcoef(
            standalone["chunk_max_mean"], fused["chunk_max_mean"]
        )[0, 1])
    else:
        cmm_corr = float("nan")

    results["chunk_max_mean"] = {
        "max_abs_diff": round(float(cmm_diff.max()), 8),
        "mean_abs_diff": round(float(cmm_diff.mean()), 8),
        "correlation": round(cmm_corr, 8),
    }

    # --- PASS / FAIL ---
    checks = {}
    checks["active_overlap"] = overlap >= THRESHOLD_ACTIVE_OVERLAP
    checks["global_max_abs_diff"] = float(gmax_diff.max()) < THRESHOLD_MAX_ABS_DIFF
    checks["top100_rank_corr"] = (not np.isnan(rho)) and rho >= THRESHOLD_RANK_CORR
    overall = all(checks.values())

    results["checks"] = {
        f"active_overlap >= {THRESHOLD_ACTIVE_OVERLAP}": checks["active_overlap"],
        f"global_max max_abs_diff < {THRESHOLD_MAX_ABS_DIFF}": checks["global_max_abs_diff"],
        f"top100 spearman >= {THRESHOLD_RANK_CORR}": checks["top100_rank_corr"],
    }
    results["overall"] = "PASS" if overall else "FAIL"

    return results


def format_report(results, standalone_path, fused_path):
    """Format a human-readable text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("SAE Global Stats Comparison: Standalone vs Fused")
    lines.append("=" * 72)
    lines.append(f"  Standalone: {standalone_path}")
    lines.append(f"  Fused:      {fused_path}")
    lines.append(f"  Features:   {results['n_features']}")
    lines.append("")

    lines.append("--- Metadata ---")
    lines.append(f"  n_chunks:       standalone={results['standalone_n_chunks']}  "
                 f"fused={results['fused_n_chunks']}")
    lines.append(f"  genome_length:  standalone={results['standalone_genome_length']}  "
                 f"fused={results['fused_genome_length']}")
    lines.append("")

    af = results["active_features"]
    lines.append("--- Active Features (global_max > 0) ---")
    lines.append(f"  Standalone: {af['standalone']}")
    lines.append(f"  Fused:      {af['fused']}")
    lines.append(f"  Both:       {af['both']}")
    lines.append(f"  Union:      {af['union']}")
    lines.append(f"  Jaccard:    {af['overlap_jaccard']:.6f}")
    lines.append("")

    gm = results["global_max"]
    lines.append("--- global_max Agreement ---")
    lines.append(f"  Max abs diff:  {gm['max_abs_diff']:.8f}")
    lines.append(f"  Mean abs diff: {gm['mean_abs_diff']:.8f}")
    lines.append(f"  Correlation:   {gm['correlation']:.8f}")
    lines.append("")

    gn = results["global_min"]
    lines.append("--- global_min Agreement ---")
    lines.append(f"  Max abs diff:  {gn['max_abs_diff']:.8f}")
    lines.append(f"  Mean abs diff: {gn['mean_abs_diff']:.8f}")
    lines.append(f"  Correlation:   {gn['correlation']:.8f}")
    lines.append("")

    tr = results["top100_rank"]
    lines.append(f"--- Top-{tr['top_k']} Features by global_max ---")
    lines.append(f"  Overlap:      {tr['overlap_count']} / {tr['top_k']}")
    lines.append(f"  Spearman rho: {tr['spearman_rho']:.8f}")
    lines.append(f"  Spearman p:   {tr['spearman_pval']:.4e}")
    lines.append("")

    cm = results["chunk_max_mean"]
    lines.append("--- chunk_max_mean Agreement ---")
    lines.append(f"  Max abs diff:  {cm['max_abs_diff']:.8f}")
    lines.append(f"  Mean abs diff: {cm['mean_abs_diff']:.8f}")
    lines.append(f"  Correlation:   {cm['correlation']:.8f}")
    lines.append("")

    lines.append("--- Checks ---")
    for check_name, passed in results["checks"].items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"  [{status}] {check_name}")
    lines.append("")
    lines.append(f"Overall: {results['overall']}")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare standalone vs fused SAE global stats NPZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--standalone", type=str, default=None,
                        help="Path to standalone global_sae_stats.npz (or its run dir)")
    parser.add_argument("--fused", type=str, default=None,
                        help="Path to fused global_sae_stats.npz (or its run dir)")
    parser.add_argument("--chrom", type=str, default=None,
                        help="Chromosome name for auto-discovery (e.g. chr22)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison summary JSON to this path")

    args = parser.parse_args()

    # Resolve paths
    if args.standalone and args.fused:
        standalone_path = resolve_npz_path(args.standalone)
        fused_path = resolve_npz_path(args.fused)
    elif args.chrom:
        stage = "sae_global_stats"
        standalone_run = find_completed_by_flags(
            args.results_dir, args.chrom, stage, "minmax"
        )
        fused_run = find_completed_by_flags(
            args.results_dir, args.chrom, stage, "fused_minmax"
        )

        if standalone_run is None:
            print(f"ERROR: No completed standalone (minmax) run found for "
                  f"{args.chrom}/{stage} in {args.results_dir}", file=sys.stderr)
            sys.exit(1)
        if fused_run is None:
            print(f"ERROR: No completed fused (fused_minmax) run found for "
                  f"{args.chrom}/{stage} in {args.results_dir}", file=sys.stderr)
            sys.exit(1)

        # Disambiguate: "fused_minmax" also contains "minmax"
        if standalone_run == fused_run:
            stage_dir = os.path.join(args.results_dir, args.chrom, stage)
            candidates = []
            for entry in sorted(os.listdir(stage_dir)):
                run_path = os.path.join(stage_dir, entry)
                if not os.path.isdir(run_path):
                    continue
                if not os.path.isfile(os.path.join(run_path, "COMPLETED")):
                    continue
                parts = entry.split("_", 2)
                flags = parts[2] if len(parts) >= 3 else ""
                if flags == "minmax":
                    candidates.append(entry)
            if candidates:
                candidates.sort()
                standalone_run = os.path.join(stage_dir, candidates[-1])
            else:
                print("ERROR: Could not distinguish standalone from fused run. "
                      "Use --standalone and --fused explicitly.", file=sys.stderr)
                sys.exit(1)

        print(f"Standalone run: {standalone_run}")
        print(f"Fused run:      {fused_run}")
        print()

        standalone_path = resolve_npz_path(standalone_run)
        fused_path = resolve_npz_path(fused_run)
    else:
        parser.error("Provide either --standalone and --fused, or --chrom for auto-discovery.")

    # Load and compare
    standalone = load_stats(standalone_path)
    fused = load_stats(fused_path)

    if len(standalone["global_max"]) != len(fused["global_max"]):
        print(f"ERROR: Feature count mismatch: standalone has "
              f"{len(standalone['global_max'])} features, fused has "
              f"{len(fused['global_max'])} features.", file=sys.stderr)
        sys.exit(1)

    results = compare_stats(standalone, fused)
    report = format_report(results, standalone_path, fused_path)
    print(report)

    if args.output:
        output = {
            "standalone_path": standalone_path,
            "fused_path": fused_path,
            **results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        print(f"\nComparison JSON saved to: {args.output}")

    if results["overall"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
