"""metrics.py — shared verification statistics, computed directly from the
eval dataset (no separate pre-computed stats.json pipeline needed: with a
real ensemble stored for every sample, everything here runs in seconds over
the whole eval set)."""
import numpy as np


def rmse(pred, truth):
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))


def mae(pred, truth):
    return float(np.nanmean(np.abs(pred - truth)))


def fair_crps(truth, ens, ens_axis=0):
    """truth: array; ens: array with an ensemble axis (default axis 0),
    remaining axes matching truth. Fair (unbiased) CRPS estimator:
    mean|ens-truth| - 0.5 * mean pairwise |ens_i-ens_j|.

    The mean-pairwise-difference term is computed via a sort-based identity
    (mean_{i<j}|x_i-x_j| = (2/(E(E-1))) * sum_k (2k-E+1)*x_sorted[k]) in
    O(E log E), fully vectorized -- NOT the naive O(E^2) double loop over
    ensemble pairs, which is too slow to run over a full evaluation set."""
    ens = np.moveaxis(ens, ens_axis, 0)
    E = ens.shape[0]
    mae_term = np.mean(np.abs(ens - truth[None]), axis=0)
    if E == 1:
        return mae_term
    sorted_ens = np.sort(ens, axis=0)
    weights = (2.0 * np.arange(E) - E + 1)
    shape = (E,) + (1,) * (sorted_ens.ndim - 1)
    spread_term = (weights.reshape(shape) * sorted_ens).sum(axis=0) * (2.0 / (E * (E - 1)))
    return mae_term - 0.5 * spread_term


def rank_histogram(truth, ens, mask=None):
    """truth: (N,H,W), ens: (N,E,H,W). Returns counts over E+1 bins: the
    rank of each (masked) truth pixel among its own ensemble at that
    pixel/sample. A real per-pixel rank histogram (no Gaussian synthesis --
    every sample here carries its full, real ensemble)."""
    E = ens.shape[1]
    counts = np.zeros(E + 1, dtype=np.int64)
    for i in range(truth.shape[0]):
        t, e = truth[i], ens[i]
        m = mask[i] if mask is not None else np.ones_like(t, dtype=bool)
        if not m.any():
            continue
        tv = t[m]
        ev = e[:, m]  # (E, n_pix)
        ranks = (ev < tv[None]).sum(axis=0)
        for r in range(E + 1):
            counts[r] += int((ranks == r).sum())
    return counts


def csi_ets_freqbias(pred, truth, threshold):
    p = pred > threshold
    t = truth > threshold
    hits = float((p & t).sum())
    misses = float((~p & t).sum())
    false_alarms = float((p & ~t).sum())
    correct_neg = float((~p & ~t).sum())
    n = hits + misses + false_alarms + correct_neg
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    hits_random = (hits + misses) * (hits + false_alarms) / n if n > 0 else 0
    denom = hits + misses + false_alarms - hits_random
    ets = (hits - hits_random) / denom if denom > 0 else np.nan
    freq_bias = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan
    return csi, ets, freq_bias


def fss(pred_binary, truth_binary, half_width):
    """Fractions Skill Score at a given neighborhood half-width (px), for
    stacks of binary fields (N,H,W)."""
    from scipy.ndimage import uniform_filter
    k = 2 * half_width + 1
    ft = uniform_filter(truth_binary.astype(float), size=(0, k, k))
    fp = uniform_filter(pred_binary.astype(float), size=(0, k, k))
    num = np.mean((ft - fp) ** 2)
    den = np.mean(ft ** 2) + np.mean(fp ** 2)
    return 1 - num / (den + 1e-10)
