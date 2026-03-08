#!/usr/bin/env python3
"""
detection_methods.py

Unified module for all entropy drop/rise detection methods.
Consolidates detection logic from score_chromosome.py and
archive/genome_scoring_jan26_drops.py into a single importable module.

Six detection methods:
  1. zscore      — z-score of derivative (statistical, parametric)
  2. mad         — MAD-based derivative (robust, non-parametric)
  3. derivative  — quantile-thresholded derivative
  4. window_mean_shift — windowed mean comparison
  5. cusum       — CUSUM change-point detection
  6. local_baseline — locally-normalized z-score

All methods return List[Tuple[int, float]] = [(position, score), ...].
A METHODS registry dict maps method names to callables.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable

# ─── Default constants ───────────────────────────────────────────────────────
SMOOTH_W = 51
ZSCORE_THRESHOLD = 2.5
MAD_THRESHOLD = 3.0
MIN_SEPARATION = 75


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _fill_nans_linear(x: np.ndarray) -> np.ndarray:
    """Fill NaN values using linear interpolation."""
    y = x.astype(np.float32, copy=True)
    isn = np.isnan(y)
    if not np.any(isn):
        return y
    idx = np.arange(len(y))
    good = ~isn
    if good.sum() >= 2:
        y[isn] = np.interp(idx[isn], idx[good], y[good])
    elif good.sum() == 1:
        y[isn] = y[good][0]
    return y


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Compute rolling mean with edge handling."""
    if len(arr) < w or w <= 1:
        return arr.copy()
    kernel = np.ones(w, dtype=np.float32) / float(w)
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    result = np.convolve(padded, kernel, mode='valid')
    return result[:len(arr)]


def _cluster_and_pick_best(
    candidates: np.ndarray,
    scores: np.ndarray,
    min_separation: int,
    pick_min: bool = True
) -> List[Tuple[int, float]]:
    """Cluster nearby candidates and pick best score in each cluster.

    Args:
        pick_min: If True, pick minimum score (for drops). If False, pick max (for rises).
    """
    if len(candidates) == 0:
        return []

    order = np.argsort(candidates)
    candidates = candidates[order]
    scores = scores[order]

    clusters = []
    current_cluster = [(candidates[0], scores[0])]
    picker = min if pick_min else max

    for i in range(1, len(candidates)):
        if candidates[i] - current_cluster[-1][0] <= min_separation:
            current_cluster.append((candidates[i], scores[i]))
        else:
            best_pos, best_score = picker(current_cluster, key=lambda x: x[1])
            clusters.append((int(best_pos), float(best_score)))
            current_cluster = [(candidates[i], scores[i])]

    if current_cluster:
        best_pos, best_score = picker(current_cluster, key=lambda x: x[1])
        clusters.append((int(best_pos), float(best_score)))

    return clusters


# =============================================================================
# DROP DETECTION METHODS
# =============================================================================

def detect_drops_zscore(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect entropy drops using z-scores of derivatives.

    Returns list of (position, z_score) tuples. Z-scores are negative.
    """
    if len(entropy) < smooth_w:
        return []

    valid_mask = ~np.isnan(entropy)
    if valid_mask.sum() < smooth_w:
        return []

    sm = _rolling_mean(np.nan_to_num(entropy, nan=np.nanmean(entropy)), smooth_w)
    d = np.diff(sm, prepend=sm[0])

    mean_deriv = np.mean(d)
    std_deriv = np.std(d)
    if std_deriv < 1e-9:
        return []

    zscores = (d - mean_deriv) / std_deriv
    candidates = np.where(zscores < -zscore_threshold)[0]
    if len(candidates) == 0:
        return []

    drops = _cluster_and_pick_best(candidates, zscores[candidates], min_separation)
    drops.sort(key=lambda x: x[0])
    return drops


def detect_drops_mad(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    mad_threshold: float = MAD_THRESHOLD,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect entropy drops using MAD-based scores of derivatives.

    Returns list of (position, mad_score) tuples. Scores are negative.
    """
    if len(entropy) < smooth_w:
        return []

    valid_mask = ~np.isnan(entropy)
    if valid_mask.sum() < smooth_w:
        return []

    sm = _rolling_mean(np.nan_to_num(entropy, nan=np.nanmean(entropy)), smooth_w)
    d = np.diff(sm, prepend=sm[0])

    median_deriv = np.nanmedian(d)
    mad = np.nanmedian(np.abs(d - median_deriv))
    if mad < 1e-9:
        return []

    mad_scores = (d - median_deriv) / (1.4826 * mad)
    candidates = np.where(mad_scores < -mad_threshold)[0]
    if len(candidates) == 0:
        return []

    drops = _cluster_and_pick_best(candidates, mad_scores[candidates], min_separation)
    drops.sort(key=lambda x: x[0])
    return drops


def detect_drops_derivative(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    thr_quantile: float = 0.01,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect entropy drops using quantile-thresholded derivative.

    Returns list of (position, derivative_value) tuples.
    """
    sm = _rolling_mean(_fill_nans_linear(entropy), smooth_w)
    d = np.diff(sm, prepend=sm[0])
    thr = np.quantile(d, thr_quantile)
    candidates = np.where(d <= thr)[0]

    if len(candidates) == 0:
        return []

    # Cluster nearby candidates
    out = []
    last = -10**9
    min_sep = max(min_separation, smooth_w // 2)
    for i in candidates:
        if i - last >= min_sep:
            out.append((int(i), float(d[i])))
            last = i
    return out


def detect_drops_window_mean_shift(
    entropy: np.ndarray,
    w: int = 200,
    top_k: int = 200,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect drops by comparing mean entropy before/after each position.

    Returns list of (position, shift_score) tuples. Scores are negative.
    Uses vectorized cumsum for O(n) performance on large arrays.
    """
    x = _fill_nans_linear(entropy)
    L = len(x)
    min_len = max(5, w // 10)

    if L < 2 * min_len:
        return []

    # Vectorized windowed mean using cumsum
    cs = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))

    # mean_before[i] = mean(x[max(0,i-w):i])
    # mean_after[i]  = mean(x[i:min(L,i+w)])
    lo_before = np.maximum(np.arange(L) - w, 0)
    hi_before = np.arange(L)
    lo_after = np.arange(L)
    hi_after = np.minimum(np.arange(L) + w, L)

    len_before = hi_before - lo_before
    len_after = hi_after - lo_after

    # Compute means via cumsum differences
    sum_before = cs[hi_before] - cs[lo_before]
    sum_after = cs[hi_after] - cs[lo_after]

    # Mask positions without enough data on either side
    valid = (len_before >= min_len) & (len_after >= min_len)
    scores = np.full(L, np.nan, dtype=np.float32)
    scores[valid] = ((sum_after[valid] / len_after[valid]) -
                     (sum_before[valid] / len_before[valid])).astype(np.float32)

    good = ~np.isnan(scores)
    if good.sum() == 0:
        return []

    idx_good = np.where(good)[0]
    order = np.argsort(scores[good])  # ascending (most negative first)
    picks = idx_good[order][:top_k]

    out = []
    half_w = max(min_separation, w // 2)
    for i in picks:
        if all(abs(i - j) > half_w for j, _ in out):
            out.append((int(i), float(scores[i])))
    return out


def detect_drops_cusum(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    h: float = 5.0,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect drops using CUSUM (Cumulative Sum) change-point detection.

    Returns list of (position, cusum_value) tuples.
    """
    x = _rolling_mean(_fill_nans_linear(entropy), smooth_w)
    mu = float(np.mean(x))

    out = []
    s = 0.0
    last = -10**9
    min_sep = max(min_separation, smooth_w)

    for i, xi in enumerate(x):
        s = max(0.0, s + (mu - float(xi)))
        if s > h and (i - last) > min_sep:
            out.append((int(i), -float(s)))  # negative so larger magnitude = stronger
            last = i
            s = 0.0
    return out


def detect_drops_local_baseline(
    entropy: np.ndarray,
    window_baseline: int = 500,
    threshold_sigma: float = 2.0,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect drops using local baseline normalization.

    Computes local statistics in a sliding window, making it adaptive
    to regional entropy differences (e.g., exons vs introns).

    Returns list of (position, local_zscore) tuples. Scores are negative.
    """
    if len(entropy) < window_baseline:
        return []

    sm = _rolling_mean(_fill_nans_linear(entropy), 51)
    d = np.diff(sm, prepend=sm[0])

    # Vectorized local mean/std using uniform_filter1d (O(n), not O(n*w))
    from scipy.ndimage import uniform_filter1d
    d_clean = np.nan_to_num(d, nan=0.0).astype(np.float64)
    local_mean = uniform_filter1d(d_clean, size=window_baseline, mode='nearest').astype(np.float32)
    local_sq_mean = uniform_filter1d(d_clean**2, size=window_baseline, mode='nearest')
    local_var = local_sq_mean - local_mean.astype(np.float64)**2
    local_var = np.maximum(local_var, 0.0)  # numerical safety
    local_std = np.sqrt(local_var).astype(np.float32)
    # Floor std to avoid division by zero in low-variation regions
    local_std = np.maximum(local_std, 1e-6)

    local_zscores = (d - local_mean) / (local_std + 1e-9)
    candidates = np.where(local_zscores < -threshold_sigma)[0]
    if len(candidates) == 0:
        return []

    drops = _cluster_and_pick_best(candidates, local_zscores[candidates], min_separation)
    drops.sort(key=lambda x: x[0])
    return drops


# =============================================================================
# RISE DETECTION METHODS
# =============================================================================

def detect_rises_zscore(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect entropy rises (end of drops) using z-scores."""
    if len(entropy) < smooth_w:
        return []

    valid_mask = ~np.isnan(entropy)
    if valid_mask.sum() < smooth_w:
        return []

    sm = _rolling_mean(np.nan_to_num(entropy, nan=np.nanmean(entropy)), smooth_w)
    d = np.diff(sm, prepend=sm[0])

    mean_deriv = np.nanmean(d)
    std_deriv = np.nanstd(d)
    if std_deriv < 1e-9:
        return []

    zscores = (d - mean_deriv) / std_deriv
    candidates = np.where(zscores > zscore_threshold)[0]
    if len(candidates) == 0:
        return []

    rises = _cluster_and_pick_best(candidates, zscores[candidates], min_separation, pick_min=False)
    rises.sort(key=lambda x: x[0])
    return rises


def detect_rises_mad(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    mad_threshold: float = MAD_THRESHOLD,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Detect entropy rises (end of drops) using MAD."""
    if len(entropy) < smooth_w:
        return []

    valid_mask = ~np.isnan(entropy)
    if valid_mask.sum() < smooth_w:
        return []

    sm = _rolling_mean(np.nan_to_num(entropy, nan=np.nanmean(entropy)), smooth_w)
    d = np.diff(sm, prepend=sm[0])

    median_deriv = np.nanmedian(d)
    mad = np.nanmedian(np.abs(d - median_deriv))
    if mad < 1e-9:
        return []

    mad_scores = (d - median_deriv) / (1.4826 * mad)
    candidates = np.where(mad_scores > mad_threshold)[0]
    if len(candidates) == 0:
        return []

    rises = _cluster_and_pick_best(candidates, mad_scores[candidates], min_separation, pick_min=False)
    rises.sort(key=lambda x: x[0])
    return rises


# =============================================================================
# BOOTSTRAP CONFIDENCE (expensive — use sparingly)
# =============================================================================

def bootstrap_drop_confidence(
    entropy: np.ndarray,
    smooth_w: int = SMOOTH_W,
    zscore_threshold: float = 2.0,
    n_bootstrap: int = 100,
    consensus_threshold: float = 0.50,
    min_separation: int = MIN_SEPARATION
) -> List[Tuple[int, float]]:
    """Bootstrap resampling to identify robust drops.

    Returns list of (position, consensus_fraction) tuples, sorted by
    consensus descending. Consensus 1.0 = detected in all bootstrap samples.

    WARNING: ~100x slower than other methods.
    """
    L = len(entropy)
    if L < smooth_w:
        return []

    entropy_filled = _fill_nans_linear(entropy)
    detection_counts = np.zeros(L, dtype=int)

    for _ in range(n_bootstrap):
        indices = np.random.choice(L, size=L, replace=True)
        entropy_boot = entropy_filled[indices]

        sm = _rolling_mean(entropy_boot, smooth_w)
        d = np.diff(sm, prepend=sm[0])

        mean_d = np.mean(d)
        std_d = np.std(d)
        if std_d < 1e-9:
            continue

        zscores = (d - mean_d) / std_d
        candidates = np.where(zscores < -zscore_threshold)[0]

        for pos in candidates:
            if pos < L:
                orig_pos = indices[pos]
                detection_counts[orig_pos] += 1

    confidence = detection_counts / n_bootstrap
    robust_positions = np.where(confidence >= consensus_threshold)[0]
    if len(robust_positions) == 0:
        return []

    positions = sorted(robust_positions.tolist())
    drops = []
    i = 0
    while i < len(positions):
        cluster_start = i
        while i + 1 < len(positions) and positions[i+1] - positions[i] <= min_separation:
            i += 1
        cluster_pos = positions[cluster_start:i+1]
        cluster_conf = confidence[cluster_pos]
        best_idx = np.argmax(cluster_conf)
        drops.append((cluster_pos[best_idx], float(cluster_conf[best_idx])))
        i += 1

    drops.sort(key=lambda x: x[1], reverse=True)
    return drops


# =============================================================================
# METHODS REGISTRY
# =============================================================================

METHODS: Dict[str, Callable[..., List[Tuple[int, float]]]] = {
    "zscore": detect_drops_zscore,
    "mad": detect_drops_mad,
    "derivative": detect_drops_derivative,
    "window_mean_shift": detect_drops_window_mean_shift,
    "cusum": detect_drops_cusum,
    "local_baseline": detect_drops_local_baseline,
}

RISE_METHODS: Dict[str, Callable[..., List[Tuple[int, float]]]] = {
    "zscore": detect_rises_zscore,
    "mad": detect_rises_mad,
}

# Method-specific default parameter overrides (beyond the function defaults)
METHOD_DEFAULTS: Dict[str, Dict] = {
    "zscore": {"smooth_w": 51, "zscore_threshold": 2.5, "min_separation": 75},
    "mad": {"smooth_w": 51, "mad_threshold": 3.0, "min_separation": 75},
    "derivative": {"smooth_w": 51, "thr_quantile": 0.01, "min_separation": 75},
    "window_mean_shift": {"w": 200, "top_k": 200, "min_separation": 75},
    "cusum": {"smooth_w": 51, "h": 5.0, "min_separation": 75},
    "local_baseline": {"window_baseline": 500, "threshold_sigma": 2.0, "min_separation": 75},
}


def run_method(name: str, entropy: np.ndarray, **kwargs) -> List[Tuple[int, float]]:
    """Run a named detection method with optional parameter overrides.

    Args:
        name: Method name (key in METHODS dict)
        entropy: Per-position entropy array
        **kwargs: Override any default parameters

    Returns:
        List of (position, score) tuples
    """
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {list(METHODS.keys())}")

    defaults = METHOD_DEFAULTS.get(name, {}).copy()
    defaults.update(kwargs)
    return METHODS[name](entropy, **defaults)
