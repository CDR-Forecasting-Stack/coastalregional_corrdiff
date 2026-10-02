#!/usr/bin/env python3
"""
coarsen.py -- Gaussian blur followed by xESMF conservative regridding
=====================================================================
Turns a native ~3 km HRRR patch (C,256,256) into a ~24 km coarse patch (C,32,32) that mimics a
reanalysis-scale (ERA5-like) driver. This is the only coarse-input construction used by the
data pipeline; no quantile mapping to ERA5 is applied (HRRR-vs-ERA5 differences, e.g. in shortwave
radiation from cloud microphysics and aerosols, are physical and are deliberately not removed).

  1. ``gaussian_blur``          -- 2-D Gaussian filter (NaN-aware, reflect padding at the patch
                                   edges) removing structure finer than the coarse grid can carry.
  2. ``ConservativeCoarsener``  -- area-weighted *conservative* regridding with xESMF of the blurred
                                   field onto the 32x32 coarse cells of the patch (8x8 native cells
                                   per coarse cell, ~24 km ~ 0.25 deg).

Design choices:
  * Gaussian width ``sigma_px`` (native 3 km pixels). Default 5 (15 km): chosen with
    ``tools/sigma_sweep.py`` as the width whose 32x32 spectra and errors are closest to ERA5 over
    ocean patches (see docs/DATASET.md).
  * Target grid: the 32x32 aggregation of each patch, so the coarse input stays exactly
    co-registered with the 256x256 target (a global 0.25-deg lattice would not give 32x32 cells
    per patch).
  * All patches are congruent in the HRRR projection, so the regridding weights are built once
    from one reference patch and reused.

Because every coarse cell is exactly a union of 8x8 native cells, conservative regridding reduces to
area-weighted block averaging up to the (small) curvature of the grid on the sphere; the blur is what
distinguishes this method from a plain block mean (see tests/test_coarsen.py).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# 1. Gaussian blur
# --------------------------------------------------------------------------
def gaussian_blur(stack: np.ndarray, sigma_px: float = 5.0, truncate: float = 4.0) -> np.ndarray:
    """2-D Gaussian filter of every channel of ``stack`` (C,H,W) -> (C,H,W), float32.

    NaN-aware (normalized convolution: blur(x*mask)/blur(mask)); NaNs stay NaN.
    Patch edges use reflect padding (per-patch approximation of a full-domain blur).
    """
    from scipy.ndimage import gaussian_filter
    stack = np.asarray(stack)
    if sigma_px <= 0:
        return stack.astype(np.float32, copy=True)
    out = np.empty(stack.shape, dtype=np.float32)
    for c in range(stack.shape[0]):
        x = stack[c].astype(np.float64)
        ok = np.isfinite(x)
        if ok.all():
            out[c] = gaussian_filter(x, sigma_px, mode="reflect", truncate=truncate)
        else:
            num = gaussian_filter(np.where(ok, x, 0.0), sigma_px, mode="reflect", truncate=truncate)
            den = gaussian_filter(ok.astype(np.float64), sigma_px, mode="reflect", truncate=truncate)
            y = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
            y[~ok] = np.nan
            out[c] = y
    return out


# --------------------------------------------------------------------------
# 2. Conservative regridding (xESMF)
# --------------------------------------------------------------------------
def _cell_corners(center: np.ndarray) -> np.ndarray:
    """(H,W) cell-centre coordinates -> (H+1,W+1) cell-corner coordinates
    (interior corners = mean of the 4 neighbouring centres, edges by linear extrapolation)."""
    p = np.pad(center.astype(np.float64), 1, mode="reflect", reflect_type="odd")
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


class ConservativeCoarsener:
    """xESMF conservative regridder from a (256,256) native patch grid onto its
    (32,32) coarse grid (``factor`` x ``factor`` native cells per coarse cell).

    All patches are congruent in the HRRR projection (same 8x8 aggregation of native
    cells), so the regridding weights are built ONCE from a reference patch's
    lat/lon and reused for every patch (``exact_per_patch=False``); the error of
    doing so is the difference in spherical-cell-area geometry between patch
    locations (well below the model's noise level). Build a coarsener per patch
    with ``from_latlon`` if you need exact weights.
    """

    def __init__(self, regridder, factor: int, lat_c: np.ndarray, lon_c: np.ndarray):
        self.regridder = regridder
        self.factor = factor
        self.lat_coarse = lat_c
        self.lon_coarse = lon_c

    @classmethod
    def from_latlon(cls, lat: np.ndarray, lon: np.ndarray, factor: int = 8) -> "ConservativeCoarsener":
        import xesmf as xe
        H, W = lat.shape
        assert H % factor == 0 and W % factor == 0, "patch size must be a multiple of the factor"
        lat_b, lon_b = _cell_corners(lat), _cell_corners(lon)
        lat_bc, lon_bc = lat_b[::factor, ::factor], lon_b[::factor, ::factor]        # (H/f+1, W/f+1)
        lat_c = 0.25 * (lat_bc[:-1, :-1] + lat_bc[1:, :-1] + lat_bc[:-1, 1:] + lat_bc[1:, 1:])
        lon_c = 0.25 * (lon_bc[:-1, :-1] + lon_bc[1:, :-1] + lon_bc[:-1, 1:] + lon_bc[1:, 1:])
        grid_in = {"lat": lat.astype(np.float64), "lon": lon.astype(np.float64), "lat_b": lat_b, "lon_b": lon_b}
        grid_out = {"lat": lat_c, "lon": lon_c, "lat_b": lat_bc, "lon_b": lon_bc}
        regridder = xe.Regridder(grid_in, grid_out, "conservative")
        return cls(regridder, factor, lat_c, lon_c)

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """(C,256,256) -> (C,32,32) float32."""
        return np.asarray(self.regridder(np.asarray(stack, dtype=np.float64))).astype(np.float32)


def coarsen_patch(stack: np.ndarray, coarsener: ConservativeCoarsener, sigma_px: float = 5.0) -> np.ndarray:
    """Gaussian blur, then conservative regridding. (C,256,256) -> (C,32,32)."""
    return coarsener(gaussian_blur(stack, sigma_px))
