#!/usr/bin/env python3
"""
era5_input.py — reusable ERA5 (ARCO Zarr) input-channel builder for coastal regional CorrDiff
================================================================================
Builds the same 34 physical input channels the HRRR pipeline builds
(collect_hrrr.IN_CH), for an arbitrary timestamp and an arbitrary target
lat/lon grid, by fetching the public ARCO ERA5 Zarr store and bilinearly
regridding onto that grid. ERA5 is natively ~0.25deg (~25km) -- the same
coarse resolution the 32x32 model input is meant to represent -- so unlike
the HRRR path there is no separate downsampling step: just fetch once and
regrid onto whatever target lat/lon grid the caller wants (a single
inference patch's 32x32 grid, a whole-region low-res grid, etc).

This module is deliberately generic (no HRRR patch/tiling assumptions) so it
can be reused wherever ERA5-sourced conditioning is needed:
  - inference/infer_region.py --input_source era5|both (live ERA5-driven inference,
    for comparing against HRRR-driven inference and a bilinear baseline)
  - era5_hrrr_validate.py (channel-by-channel ERA5-vs-HRRR statistics) could
    be refactored onto this module too; it currently has its own copy of
    the same fetch/regrid logic, predating this module.

Derived channels (q_sfc, tp_in, ssrd/strd unit conversion, cos_sza) are
computed AFTER regridding the raw ERA5 fields onto the target grid, not
before: q_sfc's Magnus-formula dependence on d2m/sp is nonlinear, so
"regrid then derive" is the physically consistent order (an average of a
nonlinear function is not the same as the nonlinear function of an average).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

import collect_hrrr as collect

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
G = 9.80665  # standard gravity, m/s^2 -- geopotential -> geopotential height

# collect short name -> ARCO ERA5 variable name. Excludes "cape" (not one of the
# 34 IN_CH channels) and excludes
# "q_sfc" (derived from d2m+sp, not fetched directly).
ERA5_SFC_VARS = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "d2m": "2m_dewpoint_temperature",
    "tcwv": "total_column_water_vapour",
    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "tp": "total_precipitation",
    "tcc": "total_cloud_cover",
    "lsm": "land_sea_mask",
}
ERA5_PL_VARS = {
    "u": "u_component_of_wind", "v": "v_component_of_wind",
    "z": "geopotential", "t": "temperature", "q": "specific_humidity",
}
PL_LEVS = collect.PL_LEVS  # [1000, 850, 500, 250]

_EXPECTED_CHANNELS = (
    set(ERA5_SFC_VARS) - {"tp"} | {"tp_in", "q_sfc"}
    | {f"{v}{lev}" for v in ERA5_PL_VARS for lev in PL_LEVS}
    | set(collect.DERIV_CH)
)
assert _EXPECTED_CHANNELS == set(collect.IN_CH), (
    "era5_input.py's channel set has drifted from collect_hrrr.IN_CH -- "
    f"missing={set(collect.IN_CH) - _EXPECTED_CHANNELS} "
    f"extra={_EXPECTED_CHANNELS - set(collect.IN_CH)}"
)


def open_arco_store():
    """Anonymous read-only handle to the public ARCO ERA5 Zarr store."""
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    import gcsfs
    import zarr
    fs = gcsfs.GCSFileSystem(token="anon")
    return zarr.open(fs.get_mapper(ARCO_STORE), mode="r")


def era5_time_index(store, dt_utc: datetime) -> int:
    epoch = datetime(1900, 1, 1, tzinfo=timezone.utc)
    target_h = int((dt_utc - epoch).total_seconds() // 3600)
    return target_h - int(store["time"][0])


def era5_latlon_box(store, lat_min, lat_max, lon_min, lon_max, pad=2):
    """Returns (lat_slice, lon_slice, era5_lat_1d, era5_lon_1d) covering the
    requested box plus `pad` extra ERA5 grid cells on each side, so bilinear
    regridding near the box edge never needs to extrapolate."""
    lats = np.array(store["latitude"])   # 90 -> -90
    lons = np.array(store["longitude"])  # 0 -> 359.75
    lon_min_360, lon_max_360 = lon_min % 360, lon_max % 360
    lat_i_hi = max(0, int(np.searchsorted(-lats, -lat_max, side="left")) - pad)
    lat_i_lo = min(len(lats), int(np.searchsorted(-lats, -lat_min, side="right")) + pad)
    lon_i_lo = max(0, int(np.searchsorted(lons, lon_min_360, side="left")) - pad)
    lon_i_hi = min(len(lons), int(np.searchsorted(lons, lon_max_360, side="right")) + pad)
    return (slice(lat_i_hi, lat_i_lo), slice(lon_i_lo, lon_i_hi),
            lats[lat_i_hi:lat_i_lo], lons[lon_i_lo:lon_i_hi])


def regrid_linear(src_lat, src_lon, values2d, tgt_lat2d, tgt_lon2d):
    """src_lat descending (N->S); flipped to ascending for
    RegularGridInterpolator. Out-of-bounds -> NaN (no extrapolation)."""
    from scipy.interpolate import RegularGridInterpolator
    src_lat_asc = src_lat[::-1]
    values_asc = values2d[::-1]
    tgt_lon_360 = tgt_lon2d % 360
    interp = RegularGridInterpolator(
        (src_lat_asc.astype(np.float64), src_lon.astype(np.float64)),
        values_asc.astype(np.float64), method="linear", bounds_error=False, fill_value=np.nan,
    )
    pts = np.stack([tgt_lat2d.ravel(), tgt_lon_360.ravel()], axis=-1)
    return interp(pts).reshape(tgt_lat2d.shape).astype(np.float32)


class Era5RawRegion:
    """Raw (un-regridded) ERA5 fields for one timestamp over one lat/lon box,
    at ERA5's own native ~0.25deg grid. Fetch once per inference run (the
    expensive part is the GCS round-trip) and reuse for every patch's regrid
    (cheap -- a few thousand interpolated points each)."""

    def __init__(self, dt: datetime, store, lat_min, lat_max, lon_min, lon_max, pad=3):
        self.dt = dt
        t_idx = era5_time_index(store, dt)
        lat_sl, lon_sl, self.era5_lat, self.era5_lon = era5_latlon_box(
            store, lat_min, lat_max, lon_min, lon_max, pad=pad
        )
        self.raw: dict[str, np.ndarray] = {}
        for short, era5_name in ERA5_SFC_VARS.items():
            self.raw[short] = np.array(store[era5_name][t_idx, lat_sl, lon_sl], dtype=np.float32)
        levels = np.array(store["level"])
        for v_short, era5_name in ERA5_PL_VARS.items():
            for lev in PL_LEVS:
                lev_idx = int(np.where(levels == lev)[0][0])
                raw = np.array(store[era5_name][t_idx, lev_idx, lat_sl, lon_sl], dtype=np.float32)
                if v_short == "z":
                    raw = raw / G  # geopotential (m^2/s^2) -> geopotential height (m)
                self.raw[f"{v_short}{lev}"] = raw


def build_era5_input_stack(region: Era5RawRegion, tgt_lat2d: np.ndarray, tgt_lon2d: np.ndarray) -> np.ndarray:
    """Regrids `region`'s raw fields onto (tgt_lat2d, tgt_lon2d) and assembles
    the 34-channel input stack in collect.IN_CH order -- an ERA5-sourced drop-in
    for collect_hrrr.downsample_256_to_32(build_hrrr_input_stack(...))."""
    def rg(short):
        return regrid_linear(region.era5_lat, region.era5_lon, region.raw[short], tgt_lat2d, tgt_lon2d)

    d2m = rg("d2m")
    sp = rg("sp")
    out = {
        "u10": rg("u10"), "v10": rg("v10"), "t2m": rg("t2m"),
        "sp": sp, "msl": rg("msl"), "d2m": d2m,
        "q_sfc": collect.derive_q_surface(d2m, sp),
        "tcwv": rg("tcwv"),
        "ssrd": np.clip(rg("ssrd") / 3600.0, 0, None),
        "strd": np.clip(rg("strd") / 3600.0, 0, None),
        "tp_in": collect.precip_log(np.clip(rg("tp") * 1000.0, 0, None)),
        "tcc": rg("tcc"),
        "lsm": rg("lsm"),
    }
    for v_short in ERA5_PL_VARS:
        for lev in PL_LEVS:
            out[f"{v_short}{lev}"] = rg(f"{v_short}{lev}")
    out["cos_sza"] = collect.compute_cos_sza(tgt_lat2d, tgt_lon2d, region.dt)

    return np.stack([out[ch] for ch in collect.IN_CH], axis=0).astype(np.float32)
