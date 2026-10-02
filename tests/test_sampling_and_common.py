"""Offline tests of the dataset sampling rules and shared utilities (no network, no GPU)."""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data_pipeline"), str(ROOT / "common")]
import collect_hrrr as c  # noqa: E402
import channels  # noqa: E402
import rain_filter  # noqa: E402


def test_timestamp_quotas_by_year_and_month():
    sel, reserves = c.sample_timestamps(c.TRAIN_START, c.TRAIN_END, c.TRAIN_N_TS, seed=1, reserve_frac=0.3)
    assert len(sel) == len(set(sel)) == 1200
    per_year = collections.Counter(d.year for d in sel)
    assert set(per_year.values()) == {200} and sorted(per_year) == list(range(2018, 2024))
    per_month = collections.Counter((d.year, d.month) for d in sel)
    assert set(per_month.values()) <= {16, 17}
    assert not set(sel) & {d for v in reserves.values() for d in v}          # reserves are disjoint
    assert all(d.year == y and d.month == m for (y, m), v in reserves.items() for d in v)


def test_test_split_quotas():
    sel, _ = c.sample_timestamps(c.TEST_START, c.TEST_END, c.TEST_N_TS, seed=1)
    assert sorted(collections.Counter(d.year for d in sel).items()) == [(2024, 250), (2025, 250)]


def test_dataset_constants():
    assert c.TRAIN_N_TS * c.TRAIN_PER_TS == c.TRAIN_N == 120_000
    assert c.TEST_N_TS * c.TEST_PER_TS == c.TEST_N == 10_000
    assert c.OCEAN_MIN_FRAC == 0.70 and len(c.IN_CH) == 34 and len(c.OUT_CH) == 8


def test_ocean_centres_from_shipped_grid():
    inv = c.load_hrrr_inv(c.HRRR_REF)
    centres = c.find_ocean_centers(inv)
    r, k = centres[len(centres) // 2]
    lsm = inv["lsm_mean"]
    assert lsm[r, k] < c.OCEAN_THRESH
    assert (lsm[r - 128:r + 128, k - 128:k + 128] < c.OCEAN_THRESH).mean() >= c.OCEAN_MIN_FRAC
    by_band = c.band_centers(centres, inv["longitude"])
    assert all(len(by_band[i]) > 100 for i in range(len(c.LON_BANDS)))


def test_channel_metadata_complete():
    assert set(channels.INPUT_MATCH) == set(channels.OUT_CH)
    assert all(v in channels.IN_CH for v in channels.INPUT_MATCH.values())
    assert all(ch in channels.UNITS for ch in channels.IN_CH + channels.OUT_CH)


def test_rain_filter_blend():
    unet = np.log10(1 + np.array([[0.0, 0.01], [0.5, 2.0]]))
    cd = unet + 1.0
    out = rain_filter.apply_rain_filter(cd, unet, 0.02)
    np.testing.assert_allclose(out[0], unet[0])                                # at/below threshold: UNet kept
    np.testing.assert_allclose(out[1], cd[1])                                  # above: CorrDiff used


def test_physical_constraints():
    import physical_constraints as pc
    oc = channels.OUT_CH
    f = np.full((2, len(oc), 3, 3), 5.0)                      # (E, C, H, W)
    for v in ("ssrd", "strd", "tp"):
        f[0, oc.index(v), 0, 0] = -4.0                        # negative values are clipped
    f[:, oc.index("ssrd"), 2, 2] = np.nan                     # NaN is preserved
    f[:, oc.index("2t"), 0, 0] = -4.0                         # other variables untouched
    cos = np.ones((3, 3)); cos[1, :] = 0.0                    # row 1 is night
    out = pc.apply_physical_constraints(f, cos)
    assert out[0, oc.index("ssrd"), 0, 0] == 0 and out[0, oc.index("strd"), 0, 0] == 0 and out[0, oc.index("tp"), 0, 0] == 0
    assert (out[:, oc.index("ssrd"), 1, :] == 0).all()        # ssrd zero at night
    assert (out[:, oc.index("strd"), 1, :] == 5.0).all()      # strd is not zeroed at night
    assert (out[:, oc.index("tp"), 1, :] == 5.0).all()
    assert np.isnan(out[:, oc.index("ssrd"), 2, 2]).all()
    assert out[0, oc.index("2t"), 0, 0] == -4.0
    assert f[0, oc.index("ssrd"), 0, 0] == -4.0               # input not modified
