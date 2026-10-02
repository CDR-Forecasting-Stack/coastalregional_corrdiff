#!/usr/bin/env python3
"""
infer_region.py -- large-area regional inference (HRRR-input or ERA5-input)
===========================================================================
Given a timestamp and a lat/lon bounding box, this script:

  1. Fetches HRRR (f00 surface + f01 surface + f00 pressure) straight from Google's public HRRR
     bucket, exactly like data_pipeline/collect_hrrr.py does when building the training data
     (same 34 input / 8 output channels). If --input_source includes era5, also fetches ERA5 from
     the public ARCO Zarr store (data_pipeline/era5_input.py) over the same region/timestamp.
  2. Tiles the requested area into overlapping 256x256 (HRRR-native) patches. Builds each patch's
     34-channel coarse (32x32) conditioning input from HRRR (2-D Gaussian blur + xESMF conservative
     regridding -- the same coarsening used for training, data_pipeline/coarsen.py) and/or from ERA5
     (direct bilinear regrid onto that patch's own 32x32 grid; ERA5 is already ~25 km native).
  3. Runs the trained regression (UNet) model and, unless --mode regression is given, the CorrDiff
     diffusion model (all ensemble members) on every patch, once per requested input source.
  4. Stitches all patches back into one contiguous map per source with linear feathering across the
     overlap band (see --overlap), and applies the precipitation rain filter (--rain_filter_mm).
  5. Fills NaN wherever the requested area falls outside HRRR's domain, or where a patch had any
     missing/invalid input data for that source -- an entire patch is NaN'd out (for that source) if
     *any* of its 34 input channels has *any* NaN pixel.

Requires NVIDIA PhysicsNeMo (``pip install nvidia-physicsnemo``) and a GPU (CPU works, slowly).

    python inference/infer_region.py \\
        --timestamp "2025-07-15 00:00" \\
        --lat_min 25 --lat_max 31 --lon_min -98 --lon_max -88 \\
        --reg_ckpt_dir $CCD_CKPT_DIR/regression --res_ckpt_dir $CCD_CKPT_DIR/diffusion \\
        --stats_path $CCD_DATA_DIR/stats.json --overlap 3 --num_ensembles 4 --input_source both

--overlap is in pixels of the 256-native output grid (3 = blend adjacent patches over a 3 px feather
band; 0 = hard edges). --input_source hrrr (default) conditions the model on coarsened HRRR; era5|both
conditions it on live ERA5 and adds '_era5'-suffixed groups. A bilinear-upsample (no-model) baseline
for either source can be reconstructed downstream from its 'input'/'input_era5' group.

Output is one NetCDF file with groups:
  input[_era5]      (34 ch, y_lr, x_lr)    the model's coarse conditioning
                                            field, stitched at low res (see
                                            ds.notes for exactly how)
  target            (8 ch,  y, x)          HRRR ground truth at native res
                                           (tp stays log10(1+mm), matching
                                           how it's stored during training)
  unet[_era5]       (8 ch,  y, x)          regression-only (deterministic
                                           mean) prediction
  corrdiff[_era5]   (8 ch,  ensemble, y, x) regression + diffusion residual,
                                            one map per ensemble member
                                            (omitted if --mode regression)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "data_pipeline"), str(REPO / "common")]
import collect_hrrr as collect  # noqa: E402
import era5_input  # noqa: E402
from ccd_model import (Normalizer, build_sampler_fn, find_latest_checkpoint, init_single_process,  # noqa: E402
                       load_networks, run_patch, upsample_32_to_256)  # noqa: E402,F401
from physical_constraints import apply_physical_constraints  # noqa: E402
from rain_filter import DEFAULT_RAIN_FILTER_MM, apply_rain_filter  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (mirrors collect_hrrr.py / regional_hrrr.py exactly — reused, not
# redefined, so the two pipelines can never silently drift apart)
# ═══════════════════════════════════════════════════════════════════════════

IN_CH = collect.IN_CH                      # 34 input channel names, exact order
OUT_CH = collect.OUT_CH                    # 8 output channel names, exact order
INVARIANT_CH = ["elev_mean", "lsm_mean"]  # matches RegionalHRRRDataset default
HR_SIZE = collect.HR_SIZE                  # 256
LR_SIZE = collect.LR_SIZE                  # 32
UPSAMPLE_FACTOR = HR_SIZE // LR_SIZE   # 8
HRRR_NY, HRRR_NX = collect.HRRR_NY, collect.HRRR_NX
DEFAULT_LAT_MIN, DEFAULT_LAT_MAX = 7.851, 52.252
DEFAULT_LON_MIN, DEFAULT_LON_MAX = -151.853, -103.783

DEFAULT_HRRR_REF = collect.HRRR_REF
DEFAULT_OUT_DIR = "./inference_output"
ERA5_PAD = 3  # extra ARCO grid cells fetched on each side of the region, for safe bilinear regridding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("infer_region")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Coastal regional CorrDiff: large-area inference from HRRR or ERA5 input")
    p.add_argument("--timestamp", required=True,
                    help='HRRR init time (UTC, hourly), e.g. "2025-07-15 00:00"')
    p.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    p.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    p.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    p.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    p.add_argument("--overlap", type=int, default=3,
                    help="pixel overlap (256-native grid) between adjacent patches "
                         "for seam blending; 0 disables blending entirely (default: 3)")
    p.add_argument("--num_ensembles", type=int, default=4)
    p.add_argument("--seed_batch_size", type=int, default=1,
                    help="how many ensemble members the diffusion model samples per forward pass")
    p.add_argument("--sampler", choices=["stochastic", "deterministic"], default="stochastic")
    p.add_argument("--sampler_steps", type=int, default=None,
                    help="default 18 for stochastic, 9 for deterministic")
    p.add_argument("--solver", default="euler", help="ODE solver for deterministic sampler")
    p.add_argument("--mode", choices=["all", "regression"], default="all",
                    help="'regression' skips the diffusion model for a fast UNet-only preview")
    p.add_argument("--input_source", choices=["hrrr", "era5", "both"], default="hrrr",
                    help="what to feed the model as coarse conditioning input: HRRR "
                         "downsampled the same way as training (default, matches the "
                         "original single-source behavior), ERA5 fetched live from the "
                         "public ARCO Zarr store and regridded, or both (writes extra "
                         "'_era5'-suffixed groups alongside the usual hrrr-named ones, "
                         "for a direct HRRR-input vs ERA5-input comparison). 'target' is "
                         "always real HRRR ground truth regardless of this setting.")
    p.add_argument("--reg_ckpt", default=None, help="regression checkpoint (.mdlus); default: latest under --reg_ckpt_dir")
    p.add_argument("--res_ckpt", default=None, help="diffusion checkpoint (.mdlus); default: latest under --res_ckpt_dir")
    p.add_argument("--reg_ckpt_dir", default=None, help="directory searched for CorrDiffRegressionUNet.*.mdlus")
    p.add_argument("--res_ckpt_dir", default=None, help="directory searched for EDMPrecondSuperResolution.*.mdlus")
    p.add_argument("--stats_path", required=True, help="normalization stats.json of the training data (stats of the checkpoints)")
    p.add_argument("--rain_filter_mm", type=float, default=DEFAULT_RAIN_FILTER_MM,
                    help="precipitation rain filter threshold in mm (blend: UNet tp wherever UNet rain <= "
                         "threshold, CorrDiff above it); negative disables (default %(default)s)")
    p.add_argument("--no_physical_constraints", action="store_true",
                    help="skip the physical-consistency constraints (ssrd/strd/tp >= 0, ssrd = 0 at night)")
    p.add_argument("--hrrr_ref", default=DEFAULT_HRRR_REF, help="HRRR invariant file (default: assets/hrrr_invariants.nc)")
    p.add_argument("--output", default=None,
                    help=f"default: {DEFAULT_OUT_DIR}/region_<timestamp>.nc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    return p.parse_args()


def parse_timestamp(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Could not parse --timestamp {s!r}; try e.g. '2025-07-15 00:00'")


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY: bbox -> HRRR row/col rectangle, patch tiling, seam blending
# ═══════════════════════════════════════════════════════════════════════════

def load_hrrr_ref(path: str) -> dict:
    ds = nc.Dataset(path, "r")
    inv = ds.groups["invariant"]
    out = {k: np.array(inv.variables[k][:], dtype=np.float32)
           for k in ("latitude", "longitude", "elev_mean", "lsm_mean")}
    ds.close()
    return out


def bbox_to_rowcol(inv: dict, lat_min, lat_max, lon_min, lon_max):
    """HRRR is a Lambert conformal projection, so a lat/lon box is not a rectangle
    in (row,col) space. Returns the (row,col) bounding rectangle that contains the
    box, plus the exact per-pixel boolean mask (used later to NaN out the corners
    of that rectangle that fall outside the true requested box)."""
    lat, lon = inv["latitude"], inv["longitude"]
    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    if not mask.any():
        raise ValueError(
            "Requested lat/lon bbox does not intersect the HRRR domain "
            f"(HRRR covers roughly lat [{lat.min():.2f},{lat.max():.2f}], "
            f"lon [{lon.min():.2f},{lon.max():.2f}])"
        )
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1, mask


def tile_starts(a0: int, a1: int, size: int, stride: int) -> list[int]:
    """Start indices of `size`-length tiles (in the same coordinate space as
    a0/a1) that fully cover [a0, a1), spaced by `stride`, with a final tile
    snapped to end exactly at a1."""
    if a1 - a0 <= size:
        return [a0]
    starts = list(range(a0, a1 - size, stride))
    if starts[-1] != a1 - size:
        starts.append(a1 - size)
    return starts


def clamp_start(s: int, size: int, n: int) -> int:
    return max(0, min(s, n - size))


def edge_weight_1d(size: int, overlap: int) -> np.ndarray:
    """Trapezoidal feather window: ramps 1..overlap+1 over the first overlap+1
    pixels, flat at overlap+1 in the middle, ramps back down at the end. Never
    hits zero, so `weighted_sum / weight_sum` stays correct even where a pixel
    is only covered by a single patch (e.g. the true edge of the mosaic).
    overlap=0 collapses this to a uniform weight of 1 everywhere (no blending)."""
    o = np.arange(size)
    w = np.minimum(np.minimum(o + 1, overlap + 1), size - o)
    return w.astype(np.float32)


def patch_weight_2d(size: int, overlap: int) -> np.ndarray:
    w1 = edge_weight_1d(size, overlap)
    return w1[:, None] * w1[None, :]


def block_nanmean(x: np.ndarray, factor: int) -> np.ndarray:
    """(C,H,W) -> (C, ceil(H/factor), ceil(W/factor)) via NaN-aware block averaging.
    Pads with NaN to a multiple of `factor` first so blocks stay aligned to the
    mosaic's own coordinate origin regardless of where individual patches started."""
    C, H, W = x.shape
    Hp, Wp = int(np.ceil(H / factor)) * factor, int(np.ceil(W / factor)) * factor
    xp = np.full((C, Hp, Wp), np.nan, dtype=np.float32)
    xp[:, :H, :W] = x
    xp = xp.reshape(C, Hp // factor, factor, Wp // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN block -> NaN, expected
        return np.nanmean(xp, axis=(2, 4)).astype(np.float32)


def build_lr_hrrr(r0, c0, in_stack):
    """(34,256,256) native-res physical input patch -> (34,32,32) via the same Gaussian blur +
    xESMF conservative regridding that collect_hrrr.py uses when building the training data.
    Returns (in_patch_native, lr)."""
    in_patch = in_stack[:, r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]  # (34,256,256) physical
    return in_patch, collect.downsample_256_to_32(in_patch)        # (34,32,32) physical


def build_lr_era5(r0, c0, inv, era5_region):
    """ERA5 has no native-res HRRR-aligned grid to downsample -- instead,
    regrid the pre-fetched ERA5 region directly onto this patch's own local
    32x32 lat/lon grid (the same grid build_lr_hrrr's coarsening
    defines for this specific patch), giving a like-for-like
    ERA5-sourced replacement for the HRRR-derived (34,32,32) coarse input."""
    lat_patch = inv["latitude"][r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]
    lon_patch = inv["longitude"][r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]
    latlon_lr = np.stack([lat_patch, lon_patch], axis=0).reshape(2, LR_SIZE, UPSAMPLE_FACTOR, LR_SIZE, UPSAMPLE_FACTOR).mean(axis=(2, 4))
    return era5_input.build_era5_input_stack(era5_region, latlon_lr[0], latlon_lr[1])


def run_patch_source(lr, inv_patch, nets, normalizer, sampler_fn, device, args):
    """Regression (+ diffusion ensemble) on one (34,32,32) physical-unit coarse input; None if it has NaNs."""
    return run_patch(lr, inv_patch, nets[0], nets[1], sampler_fn, normalizer, device,
                     len(OUT_CH), args.num_ensembles, args.seed_batch_size)


# ═══════════════════════════════════════════════════════════════════════════
# NETCDF WRITER
# ═══════════════════════════════════════════════════════════════════════════

def _write_input_group(ds, name, input_lr):
    if input_lr is None:
        return
    g = ds.createGroup(name)
    for i, ch in enumerate(IN_CH):
        v = g.createVariable(ch, "f4", ("y_lr", "x_lr"), zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))
        v[:] = input_lr[i]


def _write_output_group(ds, name, data, dims):
    if data is None:
        return
    g = ds.createGroup(name)
    for i, ch in enumerate(OUT_CH):
        v = g.createVariable(ch, "f4", dims, zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))
        v[:] = data[i] if data.ndim == 3 else data[:, i]


def write_output(path, args, dt, lat, lon, lat_lr, lon_lr, target, reg_ckpt, res_ckpt,
                  input_hrrr=None, unet_hrrr=None, corrdiff_hrrr=None,
                  input_era5=None, unet_era5=None, corrdiff_era5=None):
    """Writes one NetCDF file. Backward-compatible plain-named groups
    ('input'/'unet'/'corrdiff') always mirror the HRRR-sourced results when
    present, so existing tools keep
    working unchanged; '_era5'-suffixed groups are added on top when
    --input_source era5|both requested an ERA5-conditioned run."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ny, nx = lat.shape
    ny_lr, nx_lr = lat_lr.shape
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)
    ds.createDimension("y_lr", ny_lr)
    ds.createDimension("x_lr", nx_lr)
    if corrdiff_hrrr is not None or corrdiff_era5 is not None:
        ds.createDimension("ensemble", args.num_ensembles)

    v = ds.createVariable("lat", "f4", ("y", "x"), zlib=True); v[:] = lat
    v.long_name = "latitude"; v.units = "degrees_north"
    v = ds.createVariable("lon", "f4", ("y", "x"), zlib=True); v[:] = lon
    v.long_name = "longitude"; v.units = "degrees_east"
    v = ds.createVariable("lat_lr", "f4", ("y_lr", "x_lr"), zlib=True); v[:] = lat_lr
    v = ds.createVariable("lon_lr", "f4", ("y_lr", "x_lr"), zlib=True); v[:] = lon_lr

    gt = ds.createGroup("target")
    for i, ch in enumerate(OUT_CH):
        v = gt.createVariable(ch, "f4", ("y", "x"), zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        v[:] = target[i]

    # Plain names = HRRR-sourced results (back-compat with existing plotting scripts).
    _write_input_group(ds, "input", input_hrrr)
    _write_output_group(ds, "unet", unet_hrrr, ("y", "x"))
    _write_output_group(ds, "corrdiff", corrdiff_hrrr, ("ensemble", "y", "x"))
    # '_era5' suffix = ERA5-sourced results, added alongside when requested.
    _write_input_group(ds, "input_era5", input_era5)
    _write_output_group(ds, "unet_era5", unet_era5, ("y", "x"))
    _write_output_group(ds, "corrdiff_era5", corrdiff_era5, ("ensemble", "y", "x"))

    ds.title = "Coastal regional CorrDiff: large-area inference"
    ds.timestamp = dt.isoformat()
    ds.lat_min, ds.lat_max = args.lat_min, args.lat_max
    ds.lon_min, ds.lon_max = args.lon_min, args.lon_max
    ds.overlap_px = args.overlap
    ds.input_source = args.input_source
    ds.num_ensembles = args.num_ensembles if (corrdiff_hrrr is not None or corrdiff_era5 is not None) else 0
    ds.regression_checkpoint = str(reg_ckpt)
    ds.diffusion_checkpoint = str(res_ckpt) if res_ckpt else "none"
    ds.notes = (
        "target=HRRR ground truth (tp is log10(1+mm), matching training); "
        "unet[_era5]=regression-only mean prediction conditioned on HRRR- or "
        "ERA5-sourced coarse input; corrdiff[_era5]=regression+diffusion residual "
        "per ensemble member; input[_era5]=the model's actual (34ch, 32x32-per-patch-"
        "equivalent) coarse conditioning field, stitched at low resolution -- for HRRR "
        "this is an 8x8 block-average of the stitched native-res input (pixel-aligned "
        "with lat_lr/lon_lr); for ERA5 it is a direct bilinear regrid of the public ARCO "
        "ERA5 Zarr store onto the same lat_lr/lon_lr grid, no separate downsampling since "
        "ERA5 is already ~0.25deg/~25km native resolution. A naive bilinear-upsample "
        "baseline (no model) can be reconstructed downstream directly from either input "
        "group via the same 8x zoom-extrapolate the model's own conditioning uses. "
        "NaN = outside the requested bbox, outside the HRRR domain, or the covering "
        "patch(es) had missing/invalid input data for that source."
    )
    ds.created = datetime.utcnow().isoformat()
    ds.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not (0 <= args.overlap < HR_SIZE):
        raise ValueError("--overlap must be in [0, 256)")

    # Single standalone process: pin the single-process path so PhysicsNeMo's DistributedManager does not
    # try to rendezvous under SLURM (see ccd_model.init_single_process).
    device = init_single_process()
    if args.device:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    dt = parse_timestamp(args.timestamp)
    if dt.minute != 0 or dt.second != 0:
        log.warning(f"HRRR is hourly — rounding {dt.isoformat()} down to the hour")
        dt = dt.replace(minute=0, second=0, microsecond=0)

    run_hrrr = args.input_source in ("hrrr", "both")
    run_era5 = args.input_source in ("era5", "both")

    log.info(f"Loading HRRR reference grid from {args.hrrr_ref}")
    inv = load_hrrr_ref(args.hrrr_ref)
    row0, row1, col0, col1, bbox_mask = bbox_to_rowcol(
        inv, args.lat_min, args.lat_max, args.lon_min, args.lon_max
    )
    canvas_h, canvas_w = row1 - row0, col1 - col0
    log.info(f"Target canvas: rows[{row0}:{row1}) cols[{col0}:{col1}) -> "
             f"{canvas_h}x{canvas_w} px (HRRR grid is ~3km)")

    # ── tiling geometry (needed early: bounds the ERA5 fetch box too) ──────
    stride = HR_SIZE - args.overlap
    row_starts = sorted({clamp_start(s, HR_SIZE, HRRR_NY)
                          for s in tile_starts(row0, row1, HR_SIZE, stride)})
    col_starts = sorted({clamp_start(s, HR_SIZE, HRRR_NX)
                          for s in tile_starts(col0, col1, HR_SIZE, stride)})
    n_patches = len(row_starts) * len(col_starts)
    log.info(f"Tiling into {len(row_starts)} x {len(col_starts)} = {n_patches} patches "
             f"(256px, overlap={args.overlap}px, stride={stride}px, input_source={args.input_source})")

    era5_region = None
    if run_era5:
        # Fetch once, over the exact pixel rectangle every patch can touch
        # (patches can extend slightly beyond [row0:row1)/[col0:col1) at the
        # edges — see tile_starts/clamp_start) — not per-patch: the GCS
        # round-trip is the expensive part, the later per-patch regrid is cheap.
        all_r0, all_r1 = min(row_starts), max(row_starts) + HR_SIZE
        all_c0, all_c1 = min(col_starts), max(col_starts) + HR_SIZE
        lat_box = inv["latitude"][all_r0:all_r1, all_c0:all_c1]
        lon_box = inv["longitude"][all_r0:all_r1, all_c0:all_c1]
        log.info(f"Fetching ERA5 (ARCO) for {dt.isoformat()} over lat "
                 f"[{lat_box.min():.2f},{lat_box.max():.2f}] lon [{lon_box.min():.2f},{lon_box.max():.2f}] ...")
        era5_store = era5_input.open_arco_store()
        era5_region = era5_input.Era5RawRegion(
            dt, era5_store, float(lat_box.min()), float(lat_box.max()),
            float(lon_box.min()), float(lon_box.max()), pad=ERA5_PAD,
        )

    log.info(f"Fetching HRRR for {dt.isoformat()} ...")
    result = collect.fetch_hrrr_timestamp(dt)
    if result is None:
        raise RuntimeError(f"Could not download HRRR data for {dt.isoformat()} (sfc f00 missing)")
    sfc_f00, sfc_f01, prs_f00 = result

    in_stack = collect.build_hrrr_input_stack(sfc_f00, prs_f00, inv["latitude"], inv["longitude"], dt)
    if in_stack is None:
        raise RuntimeError("Failed to build the 34-channel HRRR input stack")
    out_stack = collect.get_hrrr_output_stack(sfc_f00, sfc_f01)
    if out_stack is None:
        raise RuntimeError("Failed to build the HRRR output stack (missing 2t/10u/10v)")

    # build_hrrr_input_stack leaves tp_in as a zero placeholder — collect_hrrr.py normally
    # fills it per-patch from f01 inside slice_and_downsample_patch; since we work on the
    # whole grid at once here, fill it once, over the whole grid, the same way.
    tp_f01 = None
    if sfc_f01 is not None:
        for k in ("tp", "prate"):
            if k in sfc_f01:
                tp_f01 = sfc_f01[k].astype(np.float32)
                break
    if tp_f01 is not None:
        in_stack[collect.SFC_CH.index("tp_in")] = collect.precip_log(tp_f01)
    else:
        log.warning("f01 precip missing — tp_in input channel will be all zeros (log10(1)=0)")

    pl_start = len(collect.SFC_CH)
    pl_end = pl_start + len(collect.PL_CH)
    if np.isnan(in_stack[pl_start:pl_end]).all():
        log.warning("All pressure-level input channels are NaN for this timestamp — "
                     "wrfprsf00 may be unavailable; every patch will come out NaN.")

    inv2 = np.stack([inv[k] for k in INVARIANT_CH], axis=0)  # (2, HRRR_NY, HRRR_NX)

    # ── checkpoints, networks, normalization ───────────────────────────────
    if not args.reg_ckpt and not args.reg_ckpt_dir:
        raise SystemExit("give --reg_ckpt or --reg_ckpt_dir")
    reg_ckpt = args.reg_ckpt or find_latest_checkpoint(args.reg_ckpt_dir, "CorrDiffRegressionUNet")
    res_ckpt = None
    if args.mode == "all":
        if not args.res_ckpt and not args.res_ckpt_dir:
            raise SystemExit("give --res_ckpt or --res_ckpt_dir (or use --mode regression)")
        res_ckpt = args.res_ckpt or find_latest_checkpoint(args.res_ckpt_dir, "EDMPrecondSuperResolution")
    log.info(f"Regression checkpoint: {reg_ckpt}")
    if res_ckpt:
        log.info(f"Diffusion checkpoint:  {res_ckpt}")
    nets = load_networks(reg_ckpt, res_ckpt, device)
    sampler_fn = build_sampler_fn(args.sampler, args.sampler_steps, args.solver)
    normalizer = Normalizer(args.stats_path, IN_CH, OUT_CH, INVARIANT_CH)

    # ── accumulators ─────────────────────────────────────────────────────
    # hrrr-path accumulators (also carry target + the back-compat "input"
    # group, since those depend on real HRRR data regardless of which
    # source is fed to the model) share one weight/validity mask.
    accum_w = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    accum_in = np.zeros((len(IN_CH), canvas_h, canvas_w), dtype=np.float32)
    accum_target = np.zeros((len(OUT_CH), canvas_h, canvas_w), dtype=np.float32)
    accum_unet = np.zeros((len(OUT_CH), canvas_h, canvas_w), dtype=np.float32) if run_hrrr else None
    accum_ens = (np.zeros((args.num_ensembles, len(OUT_CH), canvas_h, canvas_w), dtype=np.float32)
                 if run_hrrr and args.mode == "all" else None)

    accum_w_era5 = np.zeros((canvas_h, canvas_w), dtype=np.float32) if run_era5 else None
    accum_unet_era5 = np.zeros((len(OUT_CH), canvas_h, canvas_w), dtype=np.float32) if run_era5 else None
    accum_ens_era5 = (np.zeros((args.num_ensembles, len(OUT_CH), canvas_h, canvas_w), dtype=np.float32)
                      if run_era5 and args.mode == "all" else None)

    weight_patch = patch_weight_2d(HR_SIZE, args.overlap)  # (256,256), identical for every patch

    t0 = time.time()
    n_valid_hrrr = n_skipped_hrrr = n_valid_era5 = n_skipped_era5 = 0
    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            idx += 1
            out_patch = out_stack[:, r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]
            inv_patch = inv2[:, r0:r0 + HR_SIZE, c0:c0 + HR_SIZE]

            # write-region = intersection of this patch with the canvas (shared geometry)
            wr0, wr1 = max(r0, row0), min(r0 + HR_SIZE, row1)
            wc0, wc1 = max(c0, col0), min(c0 + HR_SIZE, col1)
            pr0, pc0 = wr0 - r0, wc0 - c0                 # offset into the 256x256 patch
            pr1, pc1 = pr0 + (wr1 - wr0), pc0 + (wc1 - wc0)
            cr0, cc0 = wr0 - row0, wc0 - col0              # offset into the canvas
            cr1, cc1 = cr0 + (wr1 - wr0), cc0 + (wc1 - wc0)
            w = weight_patch[pr0:pr1, pc0:pc1]

            in_patch, lr_hrrr = build_lr_hrrr(r0, c0, in_stack)
            if np.isfinite(lr_hrrr).all():
                n_valid_hrrr += 1
                accum_w[cr0:cr1, cc0:cc1] += w
                accum_in[:, cr0:cr1, cc0:cc1] += in_patch[:, pr0:pr1, pc0:pc1] * w
                accum_target[:, cr0:cr1, cc0:cc1] += out_patch[:, pr0:pr1, pc0:pc1] * w
                if run_hrrr:
                    res = run_patch_source(lr_hrrr, inv_patch, nets, normalizer, sampler_fn, device, args)
                    if res is not None:
                        unet_phys, ens_phys = res
                        accum_unet[:, cr0:cr1, cc0:cc1] += unet_phys[:, pr0:pr1, pc0:pc1] * w
                        if ens_phys is not None:
                            accum_ens[:, :, cr0:cr1, cc0:cc1] += ens_phys[:, :, pr0:pr1, pc0:pc1] * w[None, None]
            else:
                n_skipped_hrrr += 1

            if run_era5:
                lr_era5 = build_lr_era5(r0, c0, inv, era5_region)
                if np.isfinite(lr_era5).all():
                    n_valid_era5 += 1
                    accum_w_era5[cr0:cr1, cc0:cc1] += w
                    res = run_patch_source(lr_era5, inv_patch, nets, normalizer, sampler_fn, device, args)
                    if res is not None:
                        unet_phys, ens_phys = res
                        accum_unet_era5[:, cr0:cr1, cc0:cc1] += unet_phys[:, pr0:pr1, pc0:pc1] * w
                        if ens_phys is not None:
                            accum_ens_era5[:, :, cr0:cr1, cc0:cc1] += ens_phys[:, :, pr0:pr1, pc0:pc1] * w[None, None]
                else:
                    n_skipped_era5 += 1

            if idx % 10 == 0 or idx == n_patches:
                era5_note = f" era5(valid={n_valid_era5} skipped={n_skipped_era5})" if run_era5 else ""
                log.info(f"  [{idx}/{n_patches}] hrrr(valid={n_valid_hrrr} skipped={n_skipped_hrrr}){era5_note} "
                         f"elapsed={time.time()-t0:.0f}s")

    # ── finalize: weighted average, NaN wherever nothing valid contributed ──
    def finalize(accum, accum_w_):
        if accum is None:
            return None
        w_ = accum_w_[None] if accum.ndim == 3 else accum_w_[None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w_ > 0, accum / np.maximum(w_, 1e-12), np.nan).astype(np.float32)

    input_stitched = finalize(accum_in, accum_w)
    target_stitched = finalize(accum_target, accum_w)
    unet_stitched = finalize(accum_unet, accum_w)
    corrdiff_stitched = finalize(accum_ens, accum_w)
    unet_era5_stitched = finalize(accum_unet_era5, accum_w_era5)
    corrdiff_era5_stitched = finalize(accum_ens_era5, accum_w_era5)

    # rain filter: use the UNet's own tp to decide where the diffusion detail is trusted
    if args.rain_filter_mm is not None and args.rain_filter_mm >= 0:
        tp_i = OUT_CH.index("tp")
        if corrdiff_stitched is not None:
            corrdiff_stitched[:, tp_i] = apply_rain_filter(corrdiff_stitched[:, tp_i], unet_stitched[tp_i], args.rain_filter_mm)
        if corrdiff_era5_stitched is not None:
            corrdiff_era5_stitched[:, tp_i] = apply_rain_filter(corrdiff_era5_stitched[:, tp_i], unet_era5_stitched[tp_i], args.rain_filter_mm)

    # physical constraints on the model output: ssrd/strd/tp >= 0, ssrd = 0 where the sun is below the horizon
    if not args.no_physical_constraints:
        cos_sza = collect.compute_cos_sza(inv["latitude"][row0:row1, col0:col1],
                                      inv["longitude"][row0:row1, col0:col1], dt)
        if unet_stitched is not None:
            unet_stitched = apply_physical_constraints(unet_stitched, cos_sza)
        if corrdiff_stitched is not None:
            corrdiff_stitched = apply_physical_constraints(corrdiff_stitched, cos_sza)
        if unet_era5_stitched is not None:
            unet_era5_stitched = apply_physical_constraints(unet_era5_stitched, cos_sza)
        if corrdiff_era5_stitched is not None:
            corrdiff_era5_stitched = apply_physical_constraints(corrdiff_era5_stitched, cos_sza)

    # precise lat/lon polygon mask — cleans up the corners of the row/col
    # rectangle that fall outside the requested box (Lambert projection is not
    # axis-aligned in lat/lon)
    outside = ~bbox_mask[row0:row1, col0:col1]
    for arr in (input_stitched, target_stitched, unet_stitched, unet_era5_stitched):
        if arr is not None:
            arr[:, outside] = np.nan
    for arr in (corrdiff_stitched, corrdiff_era5_stitched):
        if arr is not None:
            arr[:, :, outside] = np.nan

    lat_c = inv["latitude"][row0:row1, col0:col1]
    lon_c = inv["longitude"][row0:row1, col0:col1]
    input_lr = block_nanmean(input_stitched, UPSAMPLE_FACTOR)
    lat_lr = block_nanmean(lat_c[None], UPSAMPLE_FACTOR)[0]
    lon_lr = block_nanmean(lon_c[None], UPSAMPLE_FACTOR)[0]

    # Precise low-res lat/lon bbox mask: lat_lr/lon_lr span the whole row/col
    # rectangle bbox_to_rowcol returned, which (Lambert projection) extends
    # beyond the true requested lat/lon box at the corners. input_lr inherits
    # an approximate version of this masking from block-averaging the
    # already-outside-masked native `input_stitched`; input_lr_era5 has no
    # such native-res masking step (ERA5 has no "native res" to inherit it
    # from -- it's sampled fresh, directly on this grid), so without this it
    # would show real ERA5 values in the corners no other group displays.
    outside_lr = ~((lat_lr >= args.lat_min) & (lat_lr <= args.lat_max)
                   & (lon_lr >= args.lon_min) & (lon_lr <= args.lon_max))
    input_lr[:, outside_lr] = np.nan

    input_lr_era5 = None
    if run_era5:
        # Sampled directly at the (lat_lr, lon_lr) grid in one shot -- ERA5 is a
        # smooth continuous field, not a discrete patch-aligned grid, so there's
        # no stitching/feathering to do here the way there is for HRRR's native-
        # res mosaic; this is exactly the low-res grid HRRR's own "input" group
        # is pixel-aligned with, so the two are directly comparable side by side.
        input_lr_era5 = era5_input.build_era5_input_stack(era5_region, lat_lr, lon_lr)
        input_lr_era5[:, outside_lr] = np.nan

    out_path = args.output or f"{DEFAULT_OUT_DIR}/region_{dt.strftime('%Y%m%dT%H%M')}.nc"
    write_output(out_path, args, dt, lat_c, lon_c, lat_lr, lon_lr, target_stitched, reg_ckpt, res_ckpt,
                 input_hrrr=input_lr, unet_hrrr=unet_stitched, corrdiff_hrrr=corrdiff_stitched,
                 input_era5=input_lr_era5, unet_era5=unet_era5_stitched, corrdiff_era5=corrdiff_era5_stitched)
    era5_summary = f", era5 {n_valid_era5}/{n_patches} valid" if run_era5 else ""
    log.info(f"Done: hrrr {n_valid_hrrr}/{n_patches} valid{era5_summary} -> {out_path}")


if __name__ == "__main__":
    main()
