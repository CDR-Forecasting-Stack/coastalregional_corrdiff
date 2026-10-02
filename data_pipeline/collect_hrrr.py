#!/usr/bin/env python3
"""
collect_hrrr.py -- coarse-HRRR -> high-resolution-HRRR training/test dataset builder
====================================================================================

Pure HRRR: no ERA5 is used to build the dataset.
  - Input  (34 ch, 32x32):  HRRR f00 surface + wrfprs f00 pressure-level fields, coarsened from
                            the native 256x256 patch by a 2-D Gaussian blur followed by
                            xESMF conservative regridding onto the 32x32 cells (see coarsen.py).
  - Output (8 ch, 256x256): HRRR f00 (all variables) + f01 (precipitation only; f00 tp = 0)

Patch selection (see README / docs/DATASET.md):
  - 256x256 windows (~768 km) whose centre pixel is ocean and which are >= 70 % ocean
  - equal number of patches from each of four longitude bands per timestamp
  - distinct centres within a timestamp, timestamps random by year then month

Per-timestamp parallelism: one worker thread per timestamp (downloads f00 + f01 + prs in
parallel); all patches of a timestamp are cut from in-memory arrays.
Checkpointing: progress saved after every batch; re-running the same command resumes.

Usage:
  python collect_hrrr.py --split train --workers 8 --out-base /path/to/data
  python collect_hrrr.py --split test  --workers 8 --out-base /path/to/data
  python collect_hrrr.py --split stats    --out-base /path/to/data     # normalization stats from train
  python collect_hrrr.py --split validate --out-base /path/to/data
  # small test run:  --split test --timestamp "2024-08-03 18:00" --per-ts 100 --out-base /tmp/ccd_test
"""

from __future__ import annotations
import argparse, calendar, json, logging, math, os
import socket, tempfile, time, urllib.request, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import ssl
import xarray as xr

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

GCS_HRRR = "https://storage.googleapis.com/high-resolution-rapid-refresh"
# HRRR invariant fields (latitude, longitude, terrain height, land-sea mask) on the CONUS grid.
# Shipped with the repository; override with CCD_HRRR_INVARIANTS.
HRRR_REF = os.environ.get("CCD_HRRR_INVARIANTS",
                          str(Path(__file__).resolve().parents[1] / "assets" / "hrrr_invariants.nc"))
# Default output directory for the dataset (override with CCD_DATA_DIR or --out-base).
OUT_BASE = os.environ.get("CCD_DATA_DIR", "./data")

# Dataset sizes
TRAIN_N_TS  = 1200;  TRAIN_PER_TS = 100; TRAIN_N = 120_000  # 200 ts/yr × 6 yr
TRAIN_START = "2018-01-01"; TRAIN_END = "2023-12-31"
TEST_N_TS   = 500;   TEST_PER_TS  = 20;  TEST_N  =  10_000   # 250 ts/yr × 2 yr
TEST_START  = "2024-01-01"; TEST_END  = "2025-12-31"

# Patch geometry
LR_SIZE   = 32    # input patch (downsampled from 256)
HR_SIZE   = 256   # output patch (HRRR native)
HRRR_NY   = 1059; HRRR_NX = 1799

# Ocean filter: center pixel must be ocean + ≥70% of patch
OCEAN_THRESH    = 0.5   # HRRR lsm: 0=ocean,1=land (binary)
OCEAN_MIN_FRAC  = 0.70  # ≥70% of 256×256 patch must be ocean
REQUIRE_CENTER  = True  # center pixel must be ocean

# Pressure levels to extract
PL_LEVS = [1000, 850, 500, 250]

# ── HRRR surface input channels ────────────────────────────────────────────
# Format: output_channel_name -> [cfgrib_var_candidates]
HRRR_SFC_IN = {
    "u10":   ["u10"],
    "v10":   ["v10"],
    "t2m":   ["t2m"],
    "sp":    ["sp"],
    "msl":   ["mslma","msl"],
    "d2m":   ["d2m"],
    "tcwv":  ["pwat"],           # precipitable water ≈ tcwv
    "tclw":  ["tcolw"],          # total column liquid water
    "ssrd":  ["sdswrf","dswrf"], # W/m² already — no conversion needed
    "strd":  ["sdlwrf","dlwrf"], # W/m² already — no conversion needed
    "tp_in": ["tp","prate"],     # f01 precip as INPUT (log-transformed)
    "tcc":   ["tcc"],            # total cloud cover 0-100 → /100 → 0-1
    "lsm":   ["lsm"],            # 0=ocean,1=land (binary in HRRR)
    # "cape" is deliberately not used: HRRR's wrfsfc file holds several different "cape" fields (different
    # typeOfLevel) and cfgrib does not guarantee which one is returned.
}

# HRRR pressure-level input channels from wrfprsf00
# Available: t,u,v,q,gh,r,w,dpt,absv,clwmr,rwmr,snmr,grle at 40 levels
HRRR_PL_IN = {
    "u": "u",    # m/s
    "v": "v",    # m/s
    "z": "gh",   # geopotential height m (note: ERA5 uses m2/s2, HRRR uses m)
    "t": "t",    # K
    "q": "q",    # kg/kg
}

# ── HRRR output channels ───────────────────────────────────────────────────
HRRR_OUT_MAP = {
    "2t":   ["t2m"],
    "10u":  ["u10"],
    "10v":  ["v10"],
    "sp":   ["sp"],
    "q":    ["sh2","q2m","q"],
    "ssrd": ["sdswrf","dswrf"],
    "strd": ["sdlwrf","dlwrf"],
    "tp":   ["tp","prate"],   # f01 only
}

# ── Channel ordering ───────────────────────────────────────────────────────
# Surface input channels (13) -- "cape" removed, see HRRR_SFC_IN comment above
SFC_CH = ["u10","v10","t2m","sp","msl","d2m","q_sfc",
           "tcwv","ssrd","strd","tp_in","tcc","lsm"]
# q_sfc = derived from d2m+sp
# Pressure-level channels: u,v,z,t,q at 4 levels = 20
PL_CH = [f"{v}{lev}" for v in ["u","v","z","t","q"]
         for lev in PL_LEVS]
# Derived
DERIV_CH = ["cos_sza"]
IN_CH  = SFC_CH + PL_CH + DERIV_CH   # 13 + 20 + 1 = 34
OUT_CH = ["2t","10u","10v","tp","ssrd","strd","sp","q"]

assert len(IN_CH)  == 34, f"Expected 34 input channels, got {len(IN_CH)}"
assert len(OUT_CH) == 8,  f"Expected 8 output channels, got {len(OUT_CH)}"

# Regional stratification
LON_BANDS = [(-135,-115),(-115,-95),(-95,-80),(-80,-60)]

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_hrrr")


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════

def precip_log(x: np.ndarray) -> np.ndarray:
    """HRRR tp (mm) → log10(1+P). Clips negatives."""
    return np.log10(1.0 + np.maximum(0.0, x)).astype(np.float32)

def precip_inv(x: np.ndarray) -> np.ndarray:
    """Inverse: log10(1+P) → mm/hr."""
    return np.maximum(0.0, 10.0**x - 1.0).astype(np.float32)

def derive_q_surface(d2m: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """2m specific humidity from dewpoint + surface pressure (Magnus formula).
    d2m in K, sp in Pa. Returns kg/kg. Accuracy <0.1%."""
    e_s = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
    return np.maximum(0.0, 0.622 * e_s / (sp - 0.378 * e_s)).astype(np.float32)

def compute_cos_sza(lat_2d: np.ndarray, lon_2d: np.ndarray,
                    dt: datetime) -> np.ndarray:
    """Cosine solar zenith angle ∈ [0,1]."""
    doy  = dt.timetuple().tm_yday
    hr   = dt.hour + dt.minute / 60.0
    decl = np.radians(23.45 * np.sin(np.radians(360.0/365.0*(doy-81))))
    ha   = np.radians(15.0 * (hr + lon_2d / 15.0 - 12.0))
    latr = np.radians(lat_2d)
    return np.maximum(0.0,
        np.sin(latr)*np.sin(decl) +
        np.cos(latr)*np.cos(decl)*np.cos(ha)
    ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# BILINEAR DOWNSAMPLING (numba-accelerated)
# ═══════════════════════════════════════════════════════════════════════════

# Coarsening: 2-D Gaussian blur -> xESMF conservative regridding 256 -> 32 (see coarsen.py).
# No quantile mapping to ERA5 is applied: HRRR-vs-ERA5 differences (e.g. in shortwave radiation) are
# physical model differences, and forcing the distributions to match would inject real bias.
import threading
_COARSEN = {"sigma_px": 5.0, "coarsener": None}
_COARSEN_LOCK = threading.Lock()


def configure_coarsening(sigma_px: float = 5.0):
    """Set the Gaussian width (native HRRR pixels, 3 km each) and (re)build the regridder."""
    with _COARSEN_LOCK:
        _COARSEN.update(sigma_px=float(sigma_px), coarsener=None)
    _get_coarsener()


def _get_coarsener():
    """Regridding weights are identical for every 256x256 patch in the HRRR projection, so they are
    built once from a reference patch of the HRRR grid (see coarsen.py)."""
    with _COARSEN_LOCK:
        if _COARSEN["coarsener"] is None:
            import coarsen
            inv = load_hrrr_inv(HRRR_REF)
            r0, c0 = 400, 800
            _COARSEN["coarsener"] = coarsen.ConservativeCoarsener.from_latlon(
                inv["latitude"][r0:r0 + HR_SIZE, c0:c0 + HR_SIZE].astype(np.float64),
                inv["longitude"][r0:r0 + HR_SIZE, c0:c0 + HR_SIZE].astype(np.float64), HR_SIZE // LR_SIZE)
        return _COARSEN["coarsener"]


def coarsening_description() -> str:
    return (f"2-D Gaussian blur (sigma={_COARSEN['sigma_px']:g} native px) -> xESMF conservative "
            "regridding 256->32; no quantile mapping")


def downsample_256_to_32(stack: np.ndarray) -> np.ndarray:
    """(C, 256, 256) native-resolution patch -> (C, 32, 32) coarse input.

    2-D Gaussian blur (NaN-aware, reflect padding) followed by xESMF conservative regridding onto
    the 32x32 coarse cells (~24 km). Equivalent to what a ~25 km model such as ERA5 resolves."""
    import coarsen
    return coarsen.coarsen_patch(stack, _get_coarsener(), _COARSEN["sigma_px"])


# ═══════════════════════════════════════════════════════════════════════════
# HRRR INVARIANTS + OCEAN CENTERS
# ═══════════════════════════════════════════════════════════════════════════

def load_hrrr_inv(path: str) -> dict:
    ds  = nc.Dataset(path, "r")
    inv = ds.groups["invariant"]
    r   = {k: np.array(inv.variables[k][:], dtype=np.float32)
           for k in ["latitude","longitude","elev_mean","lsm_mean"]}
    ds.close()
    log.info(f"HRRR inv: {r['latitude'].shape}  "
             f"lat [{r['latitude'].min():.1f},{r['latitude'].max():.1f}]  "
             f"lon [{r['longitude'].min():.1f},{r['longitude'].max():.1f}]")
    return r

def find_ocean_centers(inv: dict) -> list[tuple[int,int]]:
    """Find HRRR (row,col) where center is ocean AND ≥OCEAN_MIN_FRAC of the 256×256 patch is ocean."""
    log.info("Finding ocean centers...")
    lsm  = inv["lsm_mean"]
    ny, nx = lsm.shape; half = HR_SIZE // 2
    centers = []
    for r in range(half, ny-half):
        for c in range(half, nx-half):
            if REQUIRE_CENTER and lsm[r,c] >= OCEAN_THRESH:
                continue
            patch = lsm[r-half:r+half, c-half:c+half]
            if (patch < OCEAN_THRESH).mean() >= OCEAN_MIN_FRAC:
                centers.append((r,c))
    log.info(f"  {len(centers):,} valid ocean centers")
    return centers

def band_centers(centers: list, lon_grid: np.ndarray) -> dict:
    by_band: dict[int,list] = defaultdict(list)
    for r,c in centers:
        lon = float(lon_grid[r,c])
        for i,(lo,hi) in enumerate(LON_BANDS):
            if lo <= lon < hi: by_band[i].append((r,c)); break
    for i,(lo,hi) in enumerate(LON_BANDS):
        log.info(f"  Band {lo}–{hi}: {len(by_band[i]):,}")
    return by_band


# ═══════════════════════════════════════════════════════════════════════════
# TIMESTAMP SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def sample_timestamps(start: str, end: str, n: int, seed: int=42,
                      reserve_frac: float=0.0) -> tuple[list[datetime], dict]:
    """Random hourly timestamps, stratified by year then month.

    n is split evenly over the years in [start, end]; within a year the quota is split
    evenly over the 12 months (the remainder goes to randomly chosen months). Every hour of
    the pool is equally likely. Returns (selected, reserves) where reserves maps (year, month)
    to extra random timestamps (disjoint from `selected`) that replace timestamps whose
    HRRR download fails, so the final count is exact.
    """
    rng = np.random.default_rng(seed)
    d0  = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    d1  = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    pools: dict[tuple[int,int], list] = defaultdict(list)
    for h in range(int((d1-d0).total_seconds()//3600)+1):
        dt = d0+timedelta(hours=h); pools[(dt.year, dt.month)].append(dt)
    years = sorted({y for y, _ in pools})
    sel: list = []; reserves: dict = {}
    for yi, y in enumerate(years):
        q_year = n//len(years) + (1 if yi < n % len(years) else 0)
        months = sorted(m for yy, m in pools if yy == y)
        base, rem = divmod(q_year, len(months))
        extra = set(rng.choice(len(months), rem, replace=False).tolist()) if rem else set()
        for mi, m in enumerate(months):
            pool = pools[(y, m)]
            q = base + (1 if mi in extra else 0)
            nres = int(math.ceil(q*reserve_frac))
            idx = rng.permutation(len(pool))[:q+nres]
            sel.extend(pool[k] for k in idx[:q])
            reserves[(y, m)] = [pool[k] for k in idx[q:]]
    sel.sort()
    log.info(f"Sampled {len(sel)} timestamps ({start}→{end}); per year: "
             f"{ {y: sum(1 for d in sel if d.year==y) for y in years} }")
    return sel, reserves


# ═══════════════════════════════════════════════════════════════════════════
# HRRR DOWNLOADERS
# ═══════════════════════════════════════════════════════════════════════════

def _hrrr_url(dt: datetime, fxx: int=0, product: str="wrfsfc") -> str:
    return (f"{GCS_HRRR}/hrrr.{dt.strftime('%Y%m%d')}/conus/"
            f"hrrr.t{dt.hour:02d}z.{product}f{fxx:02d}.grib2")

def _ssl_opener():
    """Build a urllib opener that skips SSL cert verification (needed on some HPC clusters)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

_OPENER = _ssl_opener()   # module-level singleton


def _download_parse(url: str, retries: int=3) -> dict | None:
    """Download a HRRR GRIB2 file and parse variables.

    Returns flat dict of numpy arrays:
      2D vars:  {varname: array(1059,1799)}
      3D isobaric vars: {varname_LEVEL: array(1059,1799)} for each level
        e.g. "u_1000", "t_850", "gh_500"
    Uses SSL verification bypass required on some HPC clusters.
    Uses cfgrib directly (no xarray wrapper) for compatibility with xarray>=2024.
    """
    try:
        import cfgrib, urllib.request
    except ImportError:
        raise ImportError("pip install cfgrib eccodes")

    for attempt in range(1, retries+1):
        tmp = None
        try:
            old = socket.getdefaulttimeout(); socket.setdefaulttimeout(90)
            try:
                with _OPENER.open(url, timeout=90) as r:
                    data = r.read()
            finally:
                socket.setdefaulttimeout(old)

            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                f.write(data); tmp = f.name
            del data

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dsets = cfgrib.open_datasets(tmp)

            out: dict = {}
            for ds in dsets:
                # lat/lon coordinates
                if "latitude" in ds.coords:
                    a = np.array(ds.latitude)
                    if a.shape == (HRRR_NY, HRRR_NX) and "latitude" not in out:
                        out["latitude"] = a.astype(np.float32)
                if "longitude" in ds.coords:
                    a = np.array(ds.longitude)
                    if a.shape == (HRRR_NY, HRRR_NX) and "longitude" not in out:
                        out["longitude"] = a.astype(np.float32)

                for vname in ds.data_vars:
                    k   = vname.lower()
                    arr = np.array(ds[vname])

                    if arr.ndim == 2 and arr.shape == (HRRR_NY, HRRR_NX):
                        if k not in out:
                            out[k] = arr.astype(np.float32)

                    elif arr.ndim == 3:
                        # Check for isobaricInhPa dimension
                        if "isobaricInhPa" in ds[vname].dims:
                            levels = np.array(ds["isobaricInhPa"])
                            for li, lev in enumerate(levels):
                                if arr[li].shape == (HRRR_NY, HRRR_NX):
                                    key = f"{k}_{int(lev)}"
                                    if key not in out:
                                        out[key] = arr[li].astype(np.float32)
                ds.close()
            return out

        except Exception as e:
            log.warning(f"  Download attempt {attempt}: {url[-40:]} → {e}")
            time.sleep(2**attempt)
        finally:
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except: pass
    return None

def fetch_hrrr_timestamp(dt: datetime) -> tuple[dict,dict,dict] | None:
    """
    Download all three HRRR files for one timestamp in parallel:
      f00 surface (wrfsfcf00)  — all surface vars
      f01 surface (wrfsfcf01)  — tp only (f00 tp=0)
      f00 pressure (wrfprsf00) — pressure-level vars

    Returns (sfc_f00, sfc_f01, prs) dicts or None on failure.
    """
    urls = {
        "sfc_f00": _hrrr_url(dt, fxx=0, product="wrfsfc"),
        "sfc_f01": _hrrr_url(dt, fxx=1, product="wrfsfc"),
        "prs_f00": _hrrr_url(dt, fxx=0, product="wrfprs"),
    }

    results: dict[str, dict | None] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_download_parse, url): key
                for key, url in urls.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception as e:
                log.warning(f"  {key} download failed: {e}")
                results[key] = None

    if results.get("sfc_f00") is None:
        log.warning(f"  sfc_f00 missing for {dt} — skipping timestamp")
        return None

    sfc_f00 = results["sfc_f00"]
    sfc_f01 = results.get("sfc_f01")
    prs_f00 = results.get("prs_f00")

    # Merge pressure-level keys from prs_f00 into sfc_f00.
    # prs_f00 has all 40 levels; sfc_f00 has a subset.
    # Merged dict used for both surface + pressure-level lookup.
    if prs_f00 is not None:
        merged = dict(sfc_f00)
        for k, v in prs_f00.items():
            if "_" in k and k not in merged:   # pressure-level key e.g. "u_1000"
                merged[k] = v
    else:
        log.warning(f"  prs_f00 missing for {dt} — pressure levels from sfc only")
        merged = sfc_f00

    return merged, sfc_f01, merged   # pass merged as both sfc and prs


# ═══════════════════════════════════════════════════════════════════════════
# BUILD FULL-RESOLUTION INPUT STACK FROM HRRR
# ═══════════════════════════════════════════════════════════════════════════

def build_hrrr_input_stack(
    sfc: dict,
    prs: dict,
    inv_lats: np.ndarray,  # (1059,1799) HRRR lat grid
    inv_lons: np.ndarray,  # (1059,1799) HRRR lon grid
    dt: datetime,
) -> np.ndarray | None:
    """
    Build a (34, 1059, 1799) float32 input stack from HRRR surface + pressure files.
    All variables at full HRRR resolution. Will be sliced + downsampled per patch.

    Channel order matches IN_CH exactly.
    Returns None if any critical channel is missing.
    """
    stack_list: list[np.ndarray | None] = []

    # ── Surface channels (15) ──────────────────────────────────────────────
    for ch in SFC_CH:
        if ch == "q_sfc":
            # Derive from d2m + sp
            d2m = sfc.get("d2m"); sp = sfc.get("sp")
            if d2m is not None and sp is not None:
                arr = derive_q_surface(d2m, sp)
            else:
                arr = None
        elif ch == "tp_in":
            # Will be filled from f01 later — placeholder zeros here
            arr = np.zeros((HRRR_NY, HRRR_NX), dtype=np.float32)
        elif ch == "tcc":
            # HRRR tcc is 0-100 → divide by 100 → 0-1
            raw = None
            for cand in ["tcc"]:
                if cand in sfc: raw = sfc[cand]; break
            arr = (np.clip(raw, 0.0, 100.0) / 100.0).astype(np.float32) \
                  if raw is not None else None
        elif ch == "lsm":
            arr = sfc.get("lsm")
            if arr is not None:
                arr = arr.astype(np.float32)
        else:
            # Look up by candidate names
            cands = HRRR_SFC_IN.get(ch, [ch])
            arr = None
            for cand in cands:
                if cand.lower() in sfc:
                    arr = sfc[cand.lower()].astype(np.float32)
                    break

        if arr is not None and arr.shape == (HRRR_NY, HRRR_NX):
            stack_list.append(arr)
        else:
            if ch not in ("tp_in",):
                log.debug(f"  Missing surface channel: {ch}")
            stack_list.append(np.full((HRRR_NY, HRRR_NX), np.nan, dtype=np.float32))

    # ── Pressure-level channels (20) ──────────────────────────────────────
    # Keys in prs dict: "u_1000", "t_850", "gh_500" etc.
    for v_short, v_grib in HRRR_PL_IN.items():
        for lev in PL_LEVS:
            key = f"{v_grib}_{lev}"
            if key in prs:
                stack_list.append(prs[key].astype(np.float32))
            else:
                log.debug(f"  Missing PL key: {key}")
                stack_list.append(np.full((HRRR_NY, HRRR_NX), np.nan, dtype=np.float32))

    # ── cos_sza (1) ────────────────────────────────────────────────────────
    csza = compute_cos_sza(inv_lats, inv_lons, dt)
    stack_list.append(csza)

    if len(stack_list) != 34:
        log.error(f"  Stack has {len(stack_list)} channels, expected 34")
        return None

    return np.stack(stack_list, axis=0)   # (34, 1059, 1799)


def get_hrrr_output_stack(sfc_f00: dict, sfc_f01: dict | None) -> np.ndarray | None:
    """
    Build (8, 1059, 1799) output stack at full HRRR resolution.
    tp comes from f01; all others from f00.
    Returns None if critical channels missing.
    """
    # Merge f01 tp into f00
    sfc = dict(sfc_f00)
    if sfc_f01 is not None:
        for k in ("tp","prate"):
            if k in sfc_f01:
                sfc[k] = sfc_f01[k]

    stack_list: list[np.ndarray | None] = []
    for ch in OUT_CH:
        cands = HRRR_OUT_MAP.get(ch, [ch])
        arr = None
        for cand in cands:
            if cand.lower() in sfc:
                arr = sfc[cand.lower()].astype(np.float32)
                break
        if arr is not None and arr.shape == (HRRR_NY, HRRR_NX):
            # Apply log transform to tp
            if ch == "tp":
                arr = precip_log(arr)
            stack_list.append(arr)
        else:
            stack_list.append(np.full((HRRR_NY, HRRR_NX), np.nan, dtype=np.float32))

    if not all(stack_list[i] is not None for i in [0,1,2]):
        return None   # 2t, 10u, 10v must be present

    return np.stack(stack_list, axis=0)   # (8, 1059, 1799)


# ═══════════════════════════════════════════════════════════════════════════
# PATCH SLICING + DOWNSAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def slice_and_downsample_patch(
    in_stack:  np.ndarray,   # (34, 1059, 1799) full res input
    out_stack: np.ndarray,   # (8,  1059, 1799) full res output
    tp_in_sfc_f01: np.ndarray | None,  # (1059,1799) f01 tp for input channel
    row: int, col: int,
) -> tuple[dict[str,np.ndarray], dict[str,np.ndarray]] | None:
    """
    Extract one 256×256 patch, downsample input to 32×32.
    Returns (era_patch_dict, hrrr_patch_dict) or None.
    """
    half = HR_SIZE // 2
    r0, r1 = row-half, row+half
    c0, c1 = col-half, col+half
    if r0 < 0 or r1 > HRRR_NY or c0 < 0 or c1 > HRRR_NX:
        return None

    # Output patch: (8, 256, 256) at full HRRR resolution
    out_patch = out_stack[:, r0:r1, c0:c1]   # (8,256,256)

    # Input patch: (34, 256, 256) then downsample → (34, 32, 32)
    in_patch_256 = in_stack[:, r0:r1, c0:c1].copy()   # (34,256,256)

    # Fill tp_in channel (index 11 in SFC_CH) from f01
    if tp_in_sfc_f01 is not None:
        tp_patch = tp_in_sfc_f01[r0:r1, c0:c1].astype(np.float32)
        tp_log   = precip_log(tp_patch)
        in_patch_256[SFC_CH.index("tp_in")] = tp_log

    # Downsample to 32×32
    in_patch_32 = downsample_256_to_32(in_patch_256)   # (34,32,32)

    # Build dicts
    ep: dict[str,np.ndarray] = {}
    for i, ch in enumerate(IN_CH):
        ep[ch] = in_patch_32[i]

    hp: dict[str,np.ndarray] = {}
    for i, ch in enumerate(OUT_CH):
        hp[ch] = out_patch[i]

    return ep, hp


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS ONE TIMESTAMP
# ═══════════════════════════════════════════════════════════════════════════

def process_timestamp(
    dt: datetime,
    inv: dict,
    by_band: dict,
    ocean_centers: list,
    per_ts: int,
    rng_seed: int,
) -> tuple[datetime, list[dict] | None]:
    """
    Full pipeline for one timestamp:
      1. Download f00 surface + f01 surface + f00 pressure in parallel (3 threads)
      2. Build full-resolution input (34,1059,1799) and output (8,1059,1799) stacks
      3. Sample per_ts patch centers with regional stratification
      4. Slice and downsample each patch (in parallel within timestamp)
    """
    rng = np.random.default_rng(rng_seed)

    # Download all three HRRR files in parallel
    result = fetch_hrrr_timestamp(dt)
    if result is None: return dt, None
    sfc_f00, sfc_f01, prs_f00 = result

    # Build full-resolution stacks
    in_stack = build_hrrr_input_stack(
        sfc_f00, prs_f00,
        inv["latitude"], inv["longitude"], dt
    )
    if in_stack is None: return dt, None

    out_stack = get_hrrr_output_stack(sfc_f00, sfc_f01)
    if out_stack is None: return dt, None

    # f01 tp for input channel
    tp_f01 = None
    if sfc_f01 is not None:
        for k in ("tp","prate"):
            if k in sfc_f01:
                tp_f01 = sfc_f01[k].astype(np.float32)
                break

    # Sample patch centers with regional stratification
    n_per = max(1, per_ts // len(LON_BANDS))
    selected: list[tuple[int,int]] = []
    for i in range(len(LON_BANDS)):
        band = by_band.get(i,[])
        if band:
            n = min(n_per, len(band))
            selected.extend([band[j] for j in rng.choice(len(band), n, replace=False)])
    deficit = per_ts - len(selected)
    if deficit > 0:                       # top up with random centres, never repeating one
        chosen = set(selected)
        pool_idx = rng.permutation(len(ocean_centers))
        for j in pool_idx:
            if deficit == 0: break
            rc = ocean_centers[j]
            if rc not in chosen:
                selected.append(rc); chosen.add(rc); deficit -= 1
    assert len(set(selected)) == len(selected), "duplicate patch centres within a timestamp"

    # Slice all patches in parallel within this timestamp
    samples = []
    def _slice(rc):
        r, c = rc
        return slice_and_downsample_patch(in_stack, out_stack, tp_f01, r, c)

    with ThreadPoolExecutor(max_workers=min(8, len(selected))) as pool:
        futs = {pool.submit(_slice, rc): rc for rc in selected}
        for fut in as_completed(futs):
            rc = futs[fut]
            r, c = rc
            try:
                res = fut.result()
            except Exception as e:
                log.debug(f"  Patch ({r},{c}) failed: {e}")
                continue
            if res is None: continue
            ep, hp = res
            samples.append({
                "dt": dt, "row": r, "col": c,
                "lat": float(inv["latitude"][r,c]),
                "lon": float(inv["longitude"][r,c]),
                "era5": ep, "hrrr": hp,
            })

    return dt, samples


# ═══════════════════════════════════════════════════════════════════════════
# NETCDF WRITER
# ═══════════════════════════════════════════════════════════════════════════

def create_nc(path: Path, inv: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("sample", None)
    ds.createDimension("y_lr",   LR_SIZE)
    ds.createDimension("x_lr",   LR_SIZE)
    ds.createDimension("y_hr",   HR_SIZE)
    ds.createDimension("x_hr",   HR_SIZE)
    ds.createDimension("y_grid", HRRR_NY)
    ds.createDimension("x_grid", HRRR_NX)
    ds.createDimension("coord",  2)

    v = ds.createVariable("time","i8",("sample",),chunksizes=(1000,))
    v.units = "seconds since 1970-01-01 00:00:00 UTC"
    v = ds.createVariable("coord","u2",("sample","coord"),chunksizes=(1000,2))
    v.long_name = "HRRR grid [row,col] of patch center"
    v = ds.createVariable("lat","f4",("sample",),chunksizes=(1000,))
    v.long_name = "Patch center latitude (deg N)"
    v = ds.createVariable("lon","f4",("sample",),chunksizes=(1000,))
    v.long_name = "Patch center longitude (deg E)"

    gi = ds.createGroup("input")
    for ch in IN_CH:
        v = gi.createVariable(ch,"f4",("sample","y_lr","x_lr"),
                              zlib=True, complevel=4,
                              chunksizes=(1,LR_SIZE,LR_SIZE),
                              fill_value=np.float32(np.nan))
        v.long_name = ch

    go = ds.createGroup("output")
    for ch in OUT_CH:
        v = go.createVariable(ch,"f4",("sample","y_hr","x_hr"),
                              zlib=True, complevel=4,
                              chunksizes=(1,HR_SIZE,HR_SIZE),
                              fill_value=np.float32(np.nan))
        v.long_name = ch

    ginv = ds.createGroup("invariant")
    for field in ["latitude","longitude","elev_mean","lsm_mean"]:
        v = ginv.createVariable(field,"f4",("y_grid","x_grid"),
                                zlib=True,complevel=4)
        v.long_name = field
        v[:] = inv[field]

    ds.title      = "Coastal regional CorrDiff dataset: coarse HRRR -> native HRRR (100% HRRR-based)"
    ds.hrrr_src   = GCS_HRRR
    ds.coarsening = coarsening_description()   # how the 32x32 input was built from the 256x256 patch
    ds.in_ch      = f"{len(IN_CH)} channels (HRRR 256→32, see the `coarsening` attribute)"
    ds.out_ch     = f"{len(OUT_CH)} channels (HRRR native 256)"
    ds.transforms = (f"Input: HRRR 256x256 coarsened to 32x32 ({coarsening_description()}). "
                     "q_sfc derived from d2m+sp (Magnus). "
                     "tcc /100 -> 0-1. tp log10(1+P) from f01. "
                     "Output: tp log10(1+P) from f01; others f00.")
    ds.ocean_frac = f">={OCEAN_MIN_FRAC*100:.0f}% ocean + center ocean"
    ds.created    = datetime.utcnow().isoformat()
    ds.close()
    log.info(f"Created: {path}")

def write_sample(ds, idx, dt, row, col, lat, lon, ep, hp):
    ds["time"][idx]    = int(calendar.timegm(dt.timetuple()))
    ds["coord"][idx,:] = [row, col]
    ds["lat"][idx]     = lat
    ds["lon"][idx]     = lon
    for ch in IN_CH:
        if ch in ep: ds["input"][ch][idx] = ep[ch]
    for ch in OUT_CH:
        if ch in hp: ds["output"][ch][idx] = hp[ch]


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════

def ckpt_save(path: Path, n: int, ts_done: list[str]):
    tmp = path.with_suffix(".tmp")
    with open(tmp,"w") as f:
        json.dump({"samples_done":n,"timestamps_done":ts_done},f,indent=2)
    tmp.rename(path)

def ckpt_load(path: Path) -> tuple[int, set[str]]:
    if not path.exists(): return 0, set()
    with open(path) as f: s = json.load(f)
    n  = s.get("samples_done",0)
    ts = set(s.get("timestamps_done",[]))
    log.info(f"  Resume: {n:,} samples, {len(ts)} timestamps done")
    return n, ts


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BUILD LOOP
# ═══════════════════════════════════════════════════════════════════════════

def build(split: str, workers: int=8, seed: int=42, out_base: str | None=None,
          n_ts_override: int | None=None, per_ts_override: int | None=None,
          timestamp: str | None=None):
    is_train = (split == "train")
    out_dir  = Path(out_base or OUT_BASE); out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"regional_{split}.nc"
    ckpt_path = out_dir / f"checkpoint_{split}.json"

    fh = logging.FileHandler(out_dir / f"collection_{split}.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s"))
    log.addHandler(fh)

    n_ts, per_ts, n_total = (
        (TRAIN_N_TS, TRAIN_PER_TS, TRAIN_N) if is_train
        else (TEST_N_TS, TEST_PER_TS, TEST_N)
    )
    date_start, date_end = (
        (TRAIN_START, TRAIN_END) if is_train
        else (TEST_START, TEST_END)
    )

    if timestamp:                                  # single-timestamp test run
        n_ts = 1
    if n_ts_override: n_ts = n_ts_override
    if per_ts_override: per_ts = per_ts_override
    if timestamp or n_ts_override or per_ts_override:
        n_total = n_ts * per_ts

    log.info("=" * 60)
    log.info(f"Coastal regional CorrDiff dataset — {split.upper()} (100% HRRR)")
    log.info(f"  Output  : {out_path}")
    log.info(f"  Target  : {n_total:,} ({n_ts} ts x {per_ts} patches)")
    log.info(f"  Dates   : {date_start} -> {date_end}")
    log.info(f"  Workers : {workers} timestamps in parallel")
    log.info(f"  Input   : {len(IN_CH)} ch x {LR_SIZE}x{LR_SIZE} "
             f"({coarsening_description()})")
    log.info(f"  Output  : {len(OUT_CH)} ch x {HR_SIZE}x{HR_SIZE} (HRRR native)")
    log.info(f"  Files   : f00 surface + f01 surface (tp) + f00 pressure")
    log.info("=" * 60)

    inv           = load_hrrr_inv(HRRR_REF)
    ocean_centers = find_ocean_centers(inv)
    by_band       = band_centers(ocean_centers, inv["longitude"])
    if timestamp:
        all_ts = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)]
        reserves: dict = {}
    else:
        all_ts, reserves = sample_timestamps(date_start, date_end, n_ts, seed, reserve_frac=0.3)
    n_done, ts_done = ckpt_load(ckpt_path)
    remaining     = [dt for dt in all_ts if dt.isoformat() not in ts_done]
    log.info(f"Timestamps: {len(all_ts)} total, {len(ts_done)} done, "
             f"{len(remaining)} remaining")

    if not out_path.exists() or n_done == 0:
        create_nc(out_path, inv)

    ts_done_list = list(ts_done)
    t_start      = time.time()
    ds_out       = nc.Dataset(str(out_path), "a")

    def run_pass(pending: list) -> list:
        """Collect the timestamps in `pending`; return those that produced no samples."""
        nonlocal n_done
        failed = []
        for b_start in range(0, len(pending), workers):
            if n_done >= n_total: break
            batch = pending[b_start: b_start+workers]
            log.info(f"[BATCH {b_start//workers+1}] {len(batch)} timestamps "
                     f"(each downloads 3 HRRR files in parallel)...")

            futures_map: dict = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for dt in batch:
                    f = pool.submit(
                        process_timestamp,
                        dt, inv, by_band, ocean_centers, per_ts,
                        seed + int(dt.timestamp())//3600,      # per-timestamp seed, independent of order
                    )
                    futures_map[f] = dt

                for future in as_completed(futures_map):
                    dt = futures_map[future]
                    try:
                        _, samples = future.result()
                    except Exception as e:
                        log.warning(f"  [{dt}] {e}")
                        samples = None

                    if not samples:
                        log.warning(f"  Skipping {dt}")
                        failed.append(dt)
                    else:
                        written = 0
                        for s in samples:
                            if n_done >= n_total: break
                            write_sample(ds_out, n_done,
                                         s["dt"],s["row"],s["col"],
                                         s["lat"],s["lon"],
                                         s["era5"],s["hrrr"])
                            n_done  += 1; written += 1
                        log.info(f"  {dt.strftime('%Y-%m-%dT%HZ')}: "
                                 f"{written}/{len(samples)} patches. "
                                 f"Total: {n_done:,}/{n_total:,}")
                    ts_done_list.append(dt.isoformat())

            ds_out.sync()
            ckpt_save(ckpt_path, n_done, ts_done_list)
            elapsed = time.time()-t_start
            rate    = n_done/elapsed*3600 if elapsed>30 else 0
            eta_h   = (n_total-n_done)/rate if rate>0 else 0
            log.info(f"  Progress: {n_done:,}/{n_total:,}  "
                     f"rate={rate:.0f}/hr  ETA={eta_h:.1f}h")
        return failed

    try:
        # timestamps finished in an earlier (interrupted) run that left no samples in the file
        have = set(int(x) for x in np.unique(np.array(ds_out["time"][:n_done]))) if n_done else set()
        failed = [dt for dt in all_ts if dt.isoformat() in ts_done and int(dt.timestamp()) not in have]
        pending = remaining
        while True:
            failed += run_pass(pending)
            # replace each failed timestamp by an unused random one from the same year-month
            pending, unreplaced = [], []
            for dt in failed:
                pool = reserves.get((dt.year, dt.month), [])
                while pool and pool[0].isoformat() in ts_done_list:
                    pool.pop(0)
                if pool:
                    pending.append(pool.pop(0))
                    log.info(f"  replacing {dt:%Y-%m-%dT%HZ} with {pending[-1]:%Y-%m-%dT%HZ}")
                else:
                    unreplaced.append(dt)
            failed = unreplaced
            if not pending or n_done >= n_total: break
        if n_done < n_total:
            log.warning(f"Finished short: {n_done:,}/{n_total:,} (no reserve timestamps left for "
                        f"{len(failed)} failures)")

    finally:
        ds_out.close()
        ckpt_save(ckpt_path, n_done, ts_done_list)

    log.info(f"Done: {n_done:,}/{n_total:,} -> {out_path}")
    return n_done


# ═══════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(train_path: Path, out_json: Path, chunk: int=1000):
    log.info(f"Computing stats from {train_path}...")
    t0 = time.time()
    ds = nc.Dataset(str(train_path),"r")
    n  = ds.dimensions["sample"].size
    stats: dict = {"input":{},"output":{},"invariant":{}}
    for grp in ["input","output"]:
        g = ds.groups[grp]
        log.info(f"  [{grp}] {len(g.variables)} vars...")
        for vname,v in g.variables.items():
            cnt=mean=M2=0.0
            for s in range(0,n,chunk):
                b  = np.array(v[s:s+chunk],dtype=np.float64).ravel()
                b  = b[np.isfinite(b)]; nb=len(b)
                if nb==0: continue
                mb=b.mean(); vb=b.var(); d=mb-mean; nc_=cnt+nb
                mean=mean+d*nb/nc_; M2=M2+vb*nb+d**2*cnt*nb/nc_; cnt=nc_
            std=float(np.sqrt(M2/max(cnt,1)))
            stats[grp][vname]={"mean":float(mean),"std":max(std,1e-6)}
            log.info(f"    {vname}: mean={mean:.4e}  std={std:.4e}")
    for vname,v in ds.groups["invariant"].variables.items():
        d=np.array(v[:],dtype=np.float64).ravel()
        stats["invariant"][vname]={"mean":float(d.mean()),
                                   "std":float(max(d.std(),1e-6))}
    ds.close()
    with open(out_json,"w") as f: json.dump(stats,f,indent=2)
    log.info(f"Stats -> {out_json} in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATE
# ═══════════════════════════════════════════════════════════════════════════

def validate(path: Path, stats_path: Path | None=None):
    log.info(f"Validating: {path}")
    ds = nc.Dataset(str(path),"r")
    n  = ds.dimensions["sample"].size
    if n==0: log.warning("Empty file"); ds.close(); return
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(200,n), replace=False)

    assert ds.dimensions["y_lr"].size==LR_SIZE
    assert ds.dimensions["x_lr"].size==LR_SIZE
    log.info(f"  OK dims y_lr={LR_SIZE} x_lr={LR_SIZE}")

    ivars = list(ds.groups["input"].variables)
    assert len(ivars)==34, f"Expected 34 input, got {len(ivars)}"
    assert "cos_sza" in ivars and "tcwv" in ivars
    log.info(f"  OK 34 input channels")

    ovars = list(ds.groups["output"].variables)
    assert len(ovars)==8
    log.info(f"  OK 8 output channels")

    t = ds.variables["time"][:]
    assert t.dtype==np.int64
    dt0 = datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(seconds=int(t[0]))
    log.info(f"  OK time dtype=int64, first={dt0.date()}")

    tp = ds.groups["output"]["tp"][idx]
    fin = tp[np.isfinite(tp)]
    assert fin.max()<5.0, f"tp max={fin.max():.2f} not log-transformed?"
    log.info(f"  OK output/tp max={fin.max():.3f} log-transformed, "
             f"nonzero={( fin>0).mean()*100:.1f}%")

    lsm = ds.groups["input"]["lsm"][idx]
    fin = lsm[np.isfinite(lsm)]
    log.info(f"  OK lsm range [{fin.min():.3f},{fin.max():.3f}]")

    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    pac  = (lons<-110).sum(); atl=(lons>-85).sum()
    log.info(f"  OK samples:{n}  ts:{len(np.unique(t))}")
    log.info(f"  OK Pacific:{pac}({100*pac/len(lons):.0f}%)  "
             f"Atlantic:{atl}({100*atl/len(lons):.0f}%)")

    if stats_path and stats_path.exists():
        with open(stats_path) as f: st=json.load(f)
        assert len(st["input"])==34 and len(st["output"])==8
        log.info(f"  OK stats.json: 34+8 vars")

    ds.close()
    log.info("ALL CHECKS PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# SBATCH SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Coastal regional CorrDiff dataset builder (coarse HRRR -> native HRRR)")
    p.add_argument("--split", required=True,
                   choices=["train","test","stats","validate"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--gaussian-sigma-px", type=float, default=5.0,
                   help="Gaussian blur width in native (3 km) HRRR pixels before conservative regridding")
    p.add_argument("--out-base", default=None, help="output directory (default: OUT_BASE); use for test runs")
    p.add_argument("--n-ts", type=int, default=None, help="override number of timestamps")
    p.add_argument("--per-ts", type=int, default=None, help="override patches per timestamp")
    p.add_argument("--timestamp", default=None, help='single timestamp "YYYY-MM-DD HH:MM" (UTC)')
    a = p.parse_args()
    configure_coarsening(a.gaussian_sigma_px)
    d = Path(a.out_base or OUT_BASE)

    if a.split in ("train","test"):
        build(a.split, a.workers, a.seed, a.out_base, a.n_ts, a.per_ts, a.timestamp)
    elif a.split=="stats":
        tp=d/"regional_train.nc"
        if not tp.exists(): raise FileNotFoundError(tp)
        compute_stats(tp, d/"stats.json")
    elif a.split=="validate":
        sp=d/"stats.json"
        for s in ["train","test"]:
            pp=d/f"regional_{s}.nc"
            if pp.exists(): validate(pp, sp if sp.exists() else None)

if __name__=="__main__":
    main()
