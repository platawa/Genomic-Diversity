"""Shared utilities for the results/ directory structure.

All pipeline scripts use this module to build consistent output paths,
write COMPLETED sentinels, and discover upstream runs.
"""

import json
import os
from datetime import datetime


def build_run_dir(base_dir, chrom, stage, flags, timestamp=None):
    """Build and create results/{chrom}/{stage}/{YYYYMMDD_HHMMSS}_{flags}/.

    Args:
        base_dir: Root results directory (e.g. "./results").
        chrom: Chromosome name (e.g. "chr22", "ecoli_K12").
        stage: Pipeline stage ("scoring", "sae", "visualization").
        flags: Descriptor string (e.g. "rc_logprobs_4gpu").
        timestamp: Optional datetime; defaults to now.

    Returns:
        Path to the created run directory.
    """
    if timestamp is None:
        timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts_str}_{flags}"
    run_dir = os.path.join(base_dir, chrom, stage, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_completed(run_dir, script_name, wall_time_s):
    """Write COMPLETED JSON sentinel as the final action of a successful run."""
    payload = {
        "completed_at": datetime.now().isoformat(),
        "script": script_name,
        "wall_time_s": round(wall_time_s, 2),
    }
    path = os.path.join(run_dir, "COMPLETED")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_source(run_dir, **input_paths):
    """Write source.json with relative paths to upstream inputs.

    All paths are stored relative to run_dir so the dependency chain
    is portable across machines.
    """
    rel = {}
    for key, abspath in input_paths.items():
        if abspath is not None:
            try:
                rel[key] = os.path.relpath(abspath, run_dir)
            except ValueError:
                # On Windows, relpath can fail across drives
                rel[key] = abspath
    path = os.path.join(run_dir, "source.json")
    with open(path, "w") as f:
        json.dump(rel, f, indent=2)
        f.write("\n")


def find_latest_completed(base_dir, chrom, stage):
    """Find the most recent COMPLETED run dir for a given chrom+stage.

    Scans results/{chrom}/{stage}/ for subdirectories containing a
    COMPLETED file, returns the one with the latest timestamp prefix.

    Returns:
        Path to the run directory, or None if no completed run exists.
    """
    stage_dir = os.path.join(base_dir, chrom, stage)
    if not os.path.isdir(stage_dir):
        return None

    completed_runs = []
    for entry in os.listdir(stage_dir):
        run_path = os.path.join(stage_dir, entry)
        if os.path.isdir(run_path) and os.path.isfile(os.path.join(run_path, "COMPLETED")):
            completed_runs.append(entry)

    if not completed_runs:
        return None

    # Sort lexicographically — YYYYMMDD_HHMMSS prefix ensures chronological order
    completed_runs.sort()
    return os.path.join(stage_dir, completed_runs[-1])


def find_latest_completed_global(base_dir, subdir):
    """Find the most recent COMPLETED run in a global (non-chromosome) directory.

    Scans results/{subdir}/ for subdirectories containing a COMPLETED file.
    Used for genome-wide outputs like _genome_sae_stats/.

    Returns:
        Path to the run directory, or None if no completed run exists.
    """
    search_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(search_dir):
        return None

    completed_runs = []
    for entry in os.listdir(search_dir):
        run_path = os.path.join(search_dir, entry)
        if os.path.isdir(run_path) and os.path.isfile(os.path.join(run_path, "COMPLETED")):
            completed_runs.append(entry)

    if not completed_runs:
        return None

    completed_runs.sort()
    return os.path.join(search_dir, completed_runs[-1])


def load_global_stats(stats_path, prefer_nuc=True):
    """Load genome-wide SAE stats from either scan or aggregate format.

    Supports:
      - genome_wide_sae_stats_corrected.npz (from --aggregate_corrected):
        keys: nuc_mean, nuc_std (per-nucleotide), global_mean, global_std (chunk-max),
              genome_min, genome_max
      - genome_wide_sae_stats.npz (legacy --aggregate):
        keys: genome_min, genome_max, cross_chrom_mean, cross_chrom_std
      - global_feature_stats.npz (from aggregate_genome_sae_stats.py):
        keys: mean, std, min, max, valid_mask

    Args:
        stats_path: Path to the .npz stats file.
        prefer_nuc: If True (default), use nuc_mean/nuc_std when available.
            These are true per-nucleotide genome-wide stats.

    Returns dict with standardized keys: mean, std, min, max, valid_mask.
    Also includes 'nuc_stats' bool indicating whether nuc_mean/nuc_std was used.
    """
    import numpy as np
    raw = np.load(stats_path)

    # Prefer per-nucleotide stats (true genome-wide mean/std across all positions)
    if prefer_nuc and 'nuc_mean' in raw and 'nuc_std' in raw:
        mean = raw['nuc_mean'].astype(np.float32)
        std  = np.maximum(raw['nuc_std'].astype(np.float32), 1e-6)
        mn   = raw.get('genome_min', raw.get('min', np.zeros_like(mean)))
        mx   = raw.get('genome_max', raw.get('max', np.zeros_like(mean)))
        if hasattr(mn, 'astype'):
            mn = mn.astype(np.float32)
            mx = mx.astype(np.float32)
        valid_mask = std > 1e-6
        return {'mean': mean, 'std': std, 'min': mn, 'max': mx,
                'valid_mask': valid_mask, 'nuc_stats': True}

    # Fall back to chunk-max or legacy stats
    if 'cross_chrom_mean' in raw:
        mean = raw['cross_chrom_mean'].astype(np.float32)
        std  = np.maximum(raw['cross_chrom_std'].astype(np.float32), 1e-6)
        mn   = raw['genome_min'].astype(np.float32)
        mx   = raw['genome_max'].astype(np.float32)
    elif 'global_mean' in raw:
        mean = raw['global_mean'].astype(np.float32)
        std  = np.maximum(raw['global_std'].astype(np.float32), 1e-6)
        mn   = raw['genome_min'].astype(np.float32)
        mx   = raw['genome_max'].astype(np.float32)
    else:
        mean = raw['mean'].astype(np.float32)
        std  = np.maximum(raw['std'].astype(np.float32), 1e-6)
        mn   = raw['min'].astype(np.float32)
        mx   = raw['max'].astype(np.float32)
    valid_mask = std > 1e-6
    return {'mean': mean, 'std': std, 'min': mn, 'max': mx,
            'valid_mask': valid_mask, 'nuc_stats': False}


def find_all_completed(base_dir, chroms, stage):
    """Find completed runs for multiple chromosomes.

    Args:
        base_dir: Root results directory (e.g. "./results").
        chroms: List of chromosome names to check.
        stage: Pipeline stage ("scoring", "sae_global_stats", etc.).

    Returns:
        dict of {chrom: run_dir} for all chroms with a completed run.
    """
    result = {}
    for chrom in chroms:
        run_dir = find_latest_completed(base_dir, chrom, stage)
        if run_dir is not None:
            result[chrom] = run_dir
    return result
