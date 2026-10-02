"""Sanity tests for coarsen.py (need xesmf; skipped if it is not installed).
Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
import coarsen  # noqa: E402

xesmf = pytest.importorskip("xesmf")


def _patch_grid(n=256):
    """Slightly rotated curvilinear ~3 km grid (like a projected model grid)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    th = np.deg2rad(12.0)
    xr, yr = x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)
    lat = 35.0 + yr * 3.0 / 111.0
    lon = -125.0 + xr * 3.0 / (111.0 * np.cos(np.deg2rad(35.0)))
    return lat, lon


@pytest.fixture(scope="module")
def coarsener():
    lat, lon = _patch_grid()
    return coarsen.ConservativeCoarsener.from_latlon(lat, lon, 8)


def test_conservative_regrid_matches_block_mean_without_blur(coarsener):
    rng = np.random.default_rng(0)
    f = rng.normal(size=(3, 256, 256)).cumsum(axis=1).cumsum(axis=2) * 0.01
    ref = f.reshape(3, 32, 8, 32, 8).mean(axis=(2, 4))
    got = coarsener(f)
    assert got.shape == (3, 32, 32)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())


def test_regrid_preserves_mean(coarsener):
    rng = np.random.default_rng(1)
    f = rng.uniform(size=(1, 256, 256))
    assert abs(coarsener(f).mean() - f.mean()) < 1e-3


def test_blur_smooths_and_preserves_mean():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(2, 256, 256)).astype(np.float32)
    b = coarsen.gaussian_blur(f, 4.0)
    assert b.std() < 0.2 * f.std()
    assert abs(b.mean() - f.mean()) < 5e-3


def test_coarsen_patch_end_to_end(coarsener):
    rng = np.random.default_rng(6)
    f = rng.normal(size=(4, 256, 256)).astype(np.float32)
    out = coarsen.coarsen_patch(f, coarsener, 5.0)
    assert out.shape == (4, 32, 32) and np.isfinite(out).all()
    assert abs(out.mean() - f.mean()) < 5e-3
    assert out.std() < f.std()


def test_blur_nan_aware():
    f = np.ones((1, 64, 64), np.float32); f[0, 10:12, 10:12] = np.nan
    b = coarsen.gaussian_blur(f, 2.0)
    assert np.isnan(b[0, 10, 10]) and np.isfinite(b[0, 30, 30]) and abs(b[0, 12, 12] - 1) < 1e-5
