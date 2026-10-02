#!/usr/bin/env python3
"""
rain_filter.py — thresholded UNet+CorrDiff blend for precipitation (tp)
=========================================================================
CorrDiff's diffusion residual restores realistic small-scale rain texture,
but it perturbs the field everywhere, including light/no-rain areas where
that texture is just noise. The default threshold of 0.02 mm replaces only a
small fraction of pixels (the ones the UNet itself predicts to be raining).

    blended = where(unet_tp_mm > threshold_mm, corrdiff, unet)

applied per ensemble member (the mask is 2D, broadcast over any leading
ensemble axis), using UNet's OWN predicted rain to decide where diffusion
detail is trusted. This is the single source of truth for the rain-filter
threshold -- the evaluation and inference scripts
both import `apply_rain_filter` / `DEFAULT_RAIN_FILTER_MM` from here rather
than reimplementing it, so the threshold can't silently drift between them.
"""
import numpy as np

import collect_hrrr as collect

DEFAULT_RAIN_FILTER_MM = 0.02


def apply_rain_filter(corrdiff_tp: np.ndarray, unet_tp: np.ndarray,
                       threshold_mm: float = DEFAULT_RAIN_FILTER_MM) -> np.ndarray:
    """corrdiff_tp: (H,W) or (E,H,W) log10(1+mm) CorrDiff tp prediction(s).
    unet_tp: (H,W) log10(1+mm) UNet (regression-only) tp prediction, used ONLY
    to build the threshold mask (never itself modified). Returns an array the
    same shape as corrdiff_tp: UNet's value kept wherever UNet's own predicted
    rain rate is at/below `threshold_mm`, CorrDiff's value used above it."""
    mask = collect.precip_inv(unet_tp) > threshold_mm  # (H,W) bool
    return np.where(mask, corrdiff_tp, unet_tp)
