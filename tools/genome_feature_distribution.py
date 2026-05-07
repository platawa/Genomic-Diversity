#!/usr/bin/env python3
"""
genome_feature_distribution.py

Per-feature SAE activation distribution stats across three normalization modes
(raw / prenorm / postnorm) for every human chromosome, the pooled human genome,
E. coli, and B. subtilis.

Outputs (under a single timestamped directory):
  stats.npz                canonical array output
    stat_names  (10,)
    entities    (n_entities,)
    modes       (3,)
    stats       (n_entities, n_modes, n_features, n_stats)  float32, NaN where missing
    n_regions   (n_entities, n_modes)                       int64, 0 where missing
  per_feature_stats.tsv    long-form: entity, mode, feature_idx, <10 stat columns>
  summary_table.tsv        small human-readable rollup: one row per entity x mode

Normalization mode availability (as of Apr 2026):
  - Human chr1..chr18, chr20..chr22, chrX, chrY: raw + prenorm + postnorm
  - chr19: absent in all modes -> NaN
  - NC_000913.3 (E. coli), NC_000964.3 (B. subtilis): raw + prenorm only -> postnorm NaN

The pooled "human_genome" entity accumulates moments exactly across chromosomes
using parallel-moment combination (Chan et al.) and approximates quantiles via a
uniform per-chromosome sub-sample concatenated into a pooled reservoir.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as scistats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results_utils import build_run_dir, write_completed

logger = logging.getLogger(__name__)

MODES = ["raw", "prenorm", "postnorm"]
SUBDIR = {
    "raw": "latent_analysis",
    "prenorm": "latent_analysis_prenorm",
    "postnorm": "latent_analysis_postnorm",
}
STAT_NAMES = [
    "mean", "median", "std",
    "q95", "q99", "q999", "max",
    "skew", "kurtosis", "frac_nonzero",
]
N_STATS = len(STAT_NAMES)

DEFAULT_HUMAN_CHROMS = [
    "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
    "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY",
]
DEFAULT_BACTERIA = ["NC_000913.3", "NC_000964.3"]
POOLED_ENTITY = "human_genome"


def vector_path(results_dir, entity, mode):
    return os.path.join(results_dir, entity, "sae", SUBDIR[mode], "data",
                        "maxpooled_vectors.npy")


def _per_feature_moments(v):
    """Compute (n, mean, m2, m3, m4) per feature where m_k = sum((x - mean)^k).

    Done in float64 for numerical stability on the skew/kurtosis sums.
    Returns per-feature arrays of shape (F,).
    """
    v64 = v.astype(np.float64, copy=False)
    n = v64.shape[0]
    mean = v64.mean(axis=0)
    d = v64 - mean
    m2 = np.einsum("ij,ij->j", d, d)
    m3 = np.einsum("ij,ij,ij->j", d, d, d)
    m4 = np.einsum("ij,ij,ij,ij->j", d, d, d, d)
    return n, mean, m2, m3, m4


def _combine_moments(n_a, mean_a, m2_a, m3_a, m4_a,
                     n_b, mean_b, m2_b, m3_b, m4_b):
    """Parallel-moment combination (Chan, Golub & LeVeque 1979)."""
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
    m3 = (m3_a + m3_b
          + delta**3 * n_a * n_b * (n_a - n_b) / n**2
          + 3 * delta * (n_a * m2_b - n_b * m2_a) / n)
    m4 = (m4_a + m4_b
          + delta**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / n**3
          + 6 * delta**2 * (n_a**2 * m2_b + n_b**2 * m2_a) / n**2
          + 4 * delta * (n_a * m3_b - n_b * m3_a) / n)
    return n, mean, m2, m3, m4


def _finalize_from_moments(n, mean, m2, m3, m4):
    """Return (std, skew_biased, kurt_fisher_biased) from aggregated central moments.

    Matches scipy.stats.skew(bias=True) and kurtosis(fisher=True, bias=True).
    """
    var = m2 / n
    std = np.sqrt(var)
    # Guard against zero variance: yield 0 skew and kurt to match scipy behavior
    safe_var = np.where(var > 0, var, 1.0)
    skew = (m3 / n) / np.power(safe_var, 1.5)
    kurt = (m4 / n) / (safe_var ** 2) - 3.0
    skew = np.where(var > 0, skew, 0.0)
    kurt = np.where(var > 0, kurt, -3.0)  # match scipy: kurt-3 when var=0 -> -3
    # Actually scipy returns 0 for kurtosis when variance is zero; override:
    kurt = np.where(var > 0, kurt, 0.0)
    return std.astype(np.float32), skew.astype(np.float32), kurt.astype(np.float32)


def compute_stats_direct(v):
    """Compute all 10 per-feature stats directly from a fully-loaded vector array.

    v: (N, F) float32. Used for single-entity (non-pooled) modes where N fits in RAM.
    """
    n, f = v.shape
    mean = v.mean(axis=0).astype(np.float32)
    std = v.std(axis=0).astype(np.float32)
    frac_nonzero = (v != 0).mean(axis=0).astype(np.float32)
    vmax = v.max(axis=0).astype(np.float32)
    skew = scistats.skew(v, axis=0, bias=True).astype(np.float32)
    kurt = scistats.kurtosis(v, axis=0, fisher=True, bias=True).astype(np.float32)
    qs = np.quantile(v, [0.5, 0.95, 0.99, 0.999], axis=0).astype(np.float32)
    out = np.full((f, N_STATS), np.nan, dtype=np.float32)
    out[:, STAT_NAMES.index("mean")] = mean
    out[:, STAT_NAMES.index("median")] = qs[0]
    out[:, STAT_NAMES.index("std")] = std
    out[:, STAT_NAMES.index("q95")] = qs[1]
    out[:, STAT_NAMES.index("q99")] = qs[2]
    out[:, STAT_NAMES.index("q999")] = qs[3]
    out[:, STAT_NAMES.index("max")] = vmax
    out[:, STAT_NAMES.index("skew")] = skew
    out[:, STAT_NAMES.index("kurtosis")] = kurt
    out[:, STAT_NAMES.index("frac_nonzero")] = frac_nonzero
    return out, n


def _uniform_rowsample(v, k, seed):
    """Uniform row sample of size <= k from v without replacement (per-feature columns kept intact)."""
    n = v.shape[0]
    if n <= k:
        return v
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return v[idx]


def compute_pooled_stats(chrom_paths, mode, quantile_sample_per_chrom, seed=0):
    """Pool across chrom_paths for a given mode.

    Returns (stats_matrix, total_n) with stats_matrix shape (F, N_STATS).
    Missing chromosomes are skipped. Moments are exact; quantiles use a
    concatenated uniform per-chromosome sample.
    """
    present = [(c, p) for c, p in chrom_paths if os.path.isfile(p)]
    if not present:
        return None, 0

    acc = None            # (n, mean, m2, m3, m4)
    max_arr = None        # per-feature running max (float32)
    nz_count = None       # per-feature nonzero count (int64)
    samples = []          # list of (k_c, F) arrays for pooled quantile estimation

    for chrom, path in present:
        logger.info(f"    loading {chrom} {mode}: {path}")
        v = np.load(path)
        if v.ndim != 2:
            raise ValueError(f"{path}: expected 2D (N, F), got {v.shape}")

        # Moments
        n_c, mean_c, m2_c, m3_c, m4_c = _per_feature_moments(v)
        if acc is None:
            acc = (n_c, mean_c, m2_c, m3_c, m4_c)
            max_arr = v.max(axis=0).astype(np.float32)
            nz_count = (v != 0).sum(axis=0).astype(np.int64)
        else:
            acc = _combine_moments(*acc, n_c, mean_c, m2_c, m3_c, m4_c)
            max_arr = np.maximum(max_arr, v.max(axis=0).astype(np.float32))
            nz_count = nz_count + (v != 0).sum(axis=0).astype(np.int64)

        # Pooled quantile sample
        samples.append(_uniform_rowsample(v, quantile_sample_per_chrom,
                                         seed + hash(chrom) % (2**32 - 1)))
        del v

    n_total, mean, m2, m3, m4 = acc
    std, skew, kurt = _finalize_from_moments(n_total, mean, m2, m3, m4)
    frac_nz = (nz_count / n_total).astype(np.float32)

    # Concatenate per-chrom samples for pooled quantiles
    pooled = np.vstack(samples)
    logger.info(f"    pooled quantile sample size: {pooled.shape}")
    qs = np.quantile(pooled, [0.5, 0.95, 0.99, 0.999], axis=0).astype(np.float32)

    f = mean.shape[0]
    out = np.full((f, N_STATS), np.nan, dtype=np.float32)
    out[:, STAT_NAMES.index("mean")] = mean.astype(np.float32)
    out[:, STAT_NAMES.index("median")] = qs[0]
    out[:, STAT_NAMES.index("std")] = std
    out[:, STAT_NAMES.index("q95")] = qs[1]
    out[:, STAT_NAMES.index("q99")] = qs[2]
    out[:, STAT_NAMES.index("q999")] = qs[3]
    out[:, STAT_NAMES.index("max")] = max_arr
    out[:, STAT_NAMES.index("skew")] = skew
    out[:, STAT_NAMES.index("kurtosis")] = kurt
    out[:, STAT_NAMES.index("frac_nonzero")] = frac_nz
    return out, n_total


def build_summary_rollup(stats, entities, modes, n_regions):
    """Collapse per-feature stats into a small human-readable table.

    For each (entity, mode), report across all 32 768 features:
      - n_regions
      - median(feature_mean)       — typical per-feature mean
      - median(feature_q99)        — typical per-feature tail
      - median(feature_frac_nonzero)
      - mean(feature_max)          — typical per-feature extreme
    """
    rows = []
    mean_idx = STAT_NAMES.index("mean")
    q99_idx = STAT_NAMES.index("q99")
    max_idx = STAT_NAMES.index("max")
    nz_idx = STAT_NAMES.index("frac_nonzero")
    for ei, e in enumerate(entities):
        for mi, m in enumerate(modes):
            n = int(n_regions[ei, mi])
            if n == 0:
                rows.append({
                    "entity": e, "mode": m,
                    "n_regions": 0,
                    "feature_mean_median": np.nan,
                    "feature_q99_median": np.nan,
                    "feature_frac_nonzero_median": np.nan,
                    "feature_max_mean": np.nan,
                })
                continue
            s = stats[ei, mi]
            rows.append({
                "entity": e, "mode": m,
                "n_regions": n,
                "feature_mean_median": float(np.nanmedian(s[:, mean_idx])),
                "feature_q99_median": float(np.nanmedian(s[:, q99_idx])),
                "feature_frac_nonzero_median": float(np.nanmedian(s[:, nz_idx])),
                "feature_max_mean": float(np.nanmean(s[:, max_idx])),
            })
    return pd.DataFrame(rows)


def stats_to_longform(stats, entities, modes, n_regions):
    f_ = stats.shape[2]
    chunks = []
    feat_idx = np.arange(f_, dtype=np.int32)
    for ei, e in enumerate(entities):
        for mi, m in enumerate(modes):
            if int(n_regions[ei, mi]) == 0:
                continue
            s = stats[ei, mi]
            df = pd.DataFrame(s, columns=STAT_NAMES)
            df.insert(0, "feature_idx", feat_idx)
            df.insert(0, "mode", m)
            df.insert(0, "entity", e)
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", default=None,
                   help="Default: results/_genome_wide/feature_distribution/<TS>_all_entities/")
    p.add_argument("--entities", nargs="+", default=None,
                   help="Restrict to a subset; default = pooled human_genome + 24 human chroms + 2 bacteria.")
    p.add_argument("--modes", nargs="+", default=MODES,
                   choices=MODES, help="Subset of modes to compute.")
    p.add_argument("--quantile_sample_per_chrom", type=int, default=2000,
                   help="Rows per chrom pooled into human_genome quantile sample.")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.entities is None:
        entities = [POOLED_ENTITY] + DEFAULT_HUMAN_CHROMS + DEFAULT_BACTERIA
    else:
        entities = list(args.entities)
    modes = list(args.modes)

    if args.output_dir is None:
        out_dir = build_run_dir(args.results_dir, "_genome_wide",
                                "feature_distribution", "all_entities")
    else:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # Probe F from the first available vector
    f_features = None
    for e in entities:
        if e == POOLED_ENTITY:
            continue
        for m in modes:
            p_ = vector_path(args.results_dir, e, m)
            if os.path.isfile(p_):
                f_features = int(np.load(p_, mmap_mode="r").shape[1])
                break
        if f_features is not None:
            break
    if f_features is None:
        raise RuntimeError("No maxpooled_vectors found for any (entity, mode). "
                           "Check --results_dir / --entities.")
    logger.info(f"Detected F = {f_features} features")

    stats = np.full((len(entities), len(modes), f_features, N_STATS),
                    np.nan, dtype=np.float32)
    n_regions = np.zeros((len(entities), len(modes)), dtype=np.int64)

    t0 = time.time()
    for ei, e in enumerate(entities):
        for mi, m in enumerate(modes):
            logger.info(f"[{ei+1}/{len(entities)}] {e} x {m}")
            if e == POOLED_ENTITY:
                chrom_paths = [(c, vector_path(args.results_dir, c, m))
                               for c in DEFAULT_HUMAN_CHROMS]
                s, n = compute_pooled_stats(chrom_paths, m,
                                            args.quantile_sample_per_chrom)
            else:
                path = vector_path(args.results_dir, e, m)
                if not os.path.isfile(path):
                    logger.warning(f"  MISSING {path} -> NaN")
                    continue
                logger.info(f"  loading {path}")
                v = np.load(path)
                s, n = compute_stats_direct(v)
                del v
            if s is None:
                logger.warning(f"  {e} x {m}: no data")
                continue
            stats[ei, mi] = s
            n_regions[ei, mi] = n
            logger.info(f"  n_regions={n}")

    # Write outputs
    npz_path = os.path.join(out_dir, "stats.npz")
    np.savez_compressed(
        npz_path,
        stat_names=np.array(STAT_NAMES),
        entities=np.array(entities),
        modes=np.array(modes),
        stats=stats,
        n_regions=n_regions,
    )
    logger.info(f"Wrote {npz_path}")

    longform = stats_to_longform(stats, entities, modes, n_regions)
    long_path = os.path.join(out_dir, "per_feature_stats.tsv")
    longform.to_csv(long_path, sep="\t", index=False, float_format="%.6g")
    logger.info(f"Wrote {long_path} ({len(longform):,} rows)")

    summary = build_summary_rollup(stats, entities, modes, n_regions)
    sum_path = os.path.join(out_dir, "summary_table.tsv")
    summary.to_csv(sum_path, sep="\t", index=False, float_format="%.6g")
    logger.info(f"Wrote {sum_path}")

    # Dump run metadata
    meta = {
        "results_dir": os.path.abspath(args.results_dir),
        "entities": entities,
        "modes": modes,
        "quantile_sample_per_chrom": args.quantile_sample_per_chrom,
        "n_features": f_features,
        "stat_names": STAT_NAMES,
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    write_completed(out_dir, "genome_feature_distribution.py", time.time() - t0)
    logger.info(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
