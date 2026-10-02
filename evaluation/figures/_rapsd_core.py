"""_rapsd_core.py — correct radially-averaged power spectral density.

Fixes the "sudden drop" artifact via:
  1. Planar detrend (remove mean + linear ramp) to reduce edge discontinuity
  2. 2-D Hann window (removes spectral leakage / Gibbs cliff)
  3. Proper window power normalization
  4. Radial averaging with log-spaced wavenumber bins
  5. Clip to the reliable wavenumber range (drop the lowest-2 and Nyquist bins)

`mean_rapsd` reads through eval_io.EvalDataset (a possibly multi-shard evaluation dataset).
"""
import numpy as np


def _planar_detrend(field):
    H, W = field.shape
    y, x = np.mgrid[0:H, 0:W]
    A = np.column_stack([x.ravel(), y.ravel(), np.ones(H * W)])
    coef, *_ = np.linalg.lstsq(A, field.ravel(), rcond=None)
    plane = (A @ coef).reshape(H, W)
    return field - plane


def _hann2d(H, W):
    return np.outer(np.hanning(H), np.hanning(W))


_GEOM_CACHE = {}


def _geometry(H, W):
    """Per-(H,W) precomputed pieces: planar-detrend projector, Hann window,
    radial bin index for the fftshifted spectrum. Same maths as the original
    per-call implementation, just computed once."""
    key = (H, W)
    if key not in _GEOM_CACHE:
        y, x = np.mgrid[0:H, 0:W]
        A = np.column_stack([x.ravel(), y.ravel(), np.ones(H * W)])
        pinv = np.linalg.pinv(A)
        win = _hann2d(H, W)
        cy, cx = H // 2, W // 2
        yy, xx = np.indices((H, W))
        kr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
        k_max = min(cx, cy)
        valid = (kr >= 1) & (kr < k_max)
        bin_idx = np.floor(kr[valid]).astype(int) - 1
        _GEOM_CACHE[key] = (A, pinv, win, (win ** 2).mean(), valid, bin_idx, k_max)
    return _GEOM_CACHE[key]


def rapsd(field, pixel_km=3.0, n_bins=40):
    """2-D field -> (wavelength_km, psd), ascending wavenumber order."""
    field = np.asarray(field, dtype=float)
    H, W = field.shape
    A, pinv, win, win_power, valid, bin_idx, k_max = _geometry(H, W)

    f = field - (A @ (pinv @ field.ravel())).reshape(H, W)
    f = f * win
    F = np.fft.fftshift(np.fft.fft2(f))
    psd_flat = ((np.abs(F) ** 2) / (H * W * win_power)).ravel()

    nb = k_max - 1
    sums = np.bincount(bin_idx, weights=psd_flat[valid], minlength=nb)
    cnts = np.bincount(bin_idx, minlength=nb)
    psd_radial = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    _b = np.arange(1, k_max + 1)
    k_centers = 0.5 * (_b[:-1] + _b[1:])

    wavenumber = k_centers / H
    wavelength = pixel_km / wavenumber
    ps = psd_radial.copy()
    ps[ps <= 0] = np.nan
    wl = wavelength[1:-1]
    ps = ps[1:-1]
    order = np.argsort(wl)
    return wl[order], ps[order]


def mean_rapsd(eval_ds, var_key, group, n_samples=200, seed=42, wind=False, pixel_km=3.0):
    """Average RAPSD over n_samples patches read from `eval_ds`
    (eval_io.EvalDataset). `group` is one of "target", "regression",
    "low_res_input_upsampled", "prediction_mean" (ensemble mean),
    "prediction_member" (one random member per sample). Uses nanmean to
    tolerate empty wavenumber bins (common for sparse precipitation
    fields)."""
    indices = eval_ds.random_indices(n_samples, seed=seed)

    if group == "prediction_mean":
        fields = eval_ds.wind_speed("prediction", indices).mean(axis=1) if wind \
            else eval_ds.get("prediction", var_key, indices).mean(axis=1)
    elif group == "prediction_member":
        rng = np.random.default_rng(seed + 1)
        if wind:
            ens = eval_ds.wind_speed("prediction", indices)
        else:
            ens = eval_ds.get("prediction", var_key, indices)
        member_idx = rng.integers(0, ens.shape[1], size=ens.shape[0])
        fields = ens[np.arange(ens.shape[0]), member_idx]
    else:
        fields = eval_ds.wind_speed(group, indices) if wind else eval_ds.get(group, var_key, indices)

    all_psd, wl_ref = [], None
    for field in fields:
        wl, ps = rapsd(field, pixel_km=pixel_km)
        all_psd.append(ps)
        wl_ref = wl
    return wl_ref, np.nanmean(np.array(all_psd), axis=0)
