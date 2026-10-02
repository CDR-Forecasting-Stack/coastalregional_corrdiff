#!/usr/bin/env python3
"""
generate_eval_dataset.py -- run the trained model over a random subsample of the held-out test set
and write the evaluation NetCDF shards read by every script in evaluation/figures/.

Uses NVIDIA PhysicsNeMo through common/ccd_model.py -- the same code path as inference/infer_region.py --
so evaluation and deployment cannot drift apart. The coarse input is read from the test file
(regional_test.nc), i.e. it is exactly the Gaussian-blur + xESMF coarsened HRRR the model was trained on.

Data source: the test file's `input`/`output`/`invariant` groups, read by direct index (no eager
full-dataset load).

Output: one NetCDF "shard" per GPU worker under <output-dir>/shard_NN.nc,
group schema:
  low_res_input            (34ch, 32x32)   -- the model's actual coarse input
  low_res_input_upsampled  (8ch, 256x256)  -- bilinear-upsampled Low-res HRRR,
                                               one channel per output variable
                                               (naive-baseline column in figures)
  regression                (8ch, 256x256)  -- UNet Regression (pre-diffusion) stage
  prediction                (E,8,256,256)   -- final ensemble output
  target                    (8ch, 256x256)  -- HRRR ground truth
All tp/tp_in fields are converted from the model's internal log10(1+mm)
representation to plain mm, and the rain filter is applied to `prediction`'s
tp channel, and the physical constraints (ssrd/strd/tp >= 0, ssrd = 0 at night) are applied -- exactly matching what inference/infer_region.py does at inference
time, so these figures represent the model's actual deployed behavior.

Usage:
    python evaluation/generate_eval_dataset.py --test-nc $CCD_DATA_DIR/regional_test.nc \
        --reg-ckpt-dir $CCD_CKPT_DIR/regression --res-ckpt-dir $CCD_CKPT_DIR/diffusion \
        --stats $CCD_DATA_DIR/stats.json --n-samples 1000 --ensemble 16 --gpus 4 --output-dir ./eval_data
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("generate_eval_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "common"), str(REPO_ROOT / "data_pipeline")]

UPSAMPLE_FACTOR = 8
HR_SIZE = 256


def select_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(n_samples, n_total), replace=False))


def chunk(items: np.ndarray, n: int) -> list[np.ndarray]:
    n = max(1, min(n, len(items)))
    return [c for c in np.array_split(items, n) if len(c)]


def read_sample(ds, idx: int, in_ch, out_ch, invariant_ch):
    import netCDF4 as nc  # noqa: F401  (import kept local to worker process)
    gi, go, ginv = ds.groups["input"], ds.groups["output"], ds.groups["invariant"]
    x_lr = np.stack([np.array(gi.variables[ch][idx]) for ch in in_ch], axis=0).astype(np.float32)
    y = np.stack([np.array(go.variables[ch][idx]) for ch in out_ch], axis=0).astype(np.float32)
    row, col = [int(v) for v in ds.variables["coord"][idx]]
    half = HR_SIZE // 2
    inv = np.stack(
        [np.array(ginv.variables[ch][row - half:row + half, col - half:col + half]) for ch in invariant_ch],
        axis=0,
    ).astype(np.float32)
    lat_grid = np.array(ginv.variables["latitude"][row - half:row + half, col - half:col + half])
    lon_grid = np.array(ginv.variables["longitude"][row - half:row + half, col - half:col + half])
    lat = float(ds.variables["lat"][idx])
    lon = float(ds.variables["lon"][idx])
    t = int(ds.variables["time"][idx])
    return x_lr, y, inv, lat, lon, t, lat_grid, lon_grid


def create_shard(path: Path, n: int, ensemble: int, in_ch, out_ch):
    import netCDF4 as nc
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("sample", n)
    ds.createDimension("y_lr", 32)
    ds.createDimension("x_lr", 32)
    ds.createDimension("y_hr", HR_SIZE)
    ds.createDimension("x_hr", HR_SIZE)
    ds.createDimension("ensemble", ensemble)

    ds.createVariable("sample_idx", "i8", ("sample",))
    ds.createVariable("lat", "f4", ("sample",))
    ds.createVariable("lon", "f4", ("sample",))
    v = ds.createVariable("time", "i8", ("sample",))
    v.units = "seconds since 1970-01-01 00:00:00 UTC"

    g_lr = ds.createGroup("low_res_input")
    for ch in in_ch:
        g_lr.createVariable(ch, "f4", ("sample", "y_lr", "x_lr"), zlib=True, complevel=4,
                             fill_value=np.float32(np.nan))

    g_lru = ds.createGroup("low_res_input_upsampled")
    for ch in out_ch:
        g_lru.createVariable(ch, "f4", ("sample", "y_hr", "x_hr"), zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))

    g_reg = ds.createGroup("regression")
    for ch in out_ch:
        g_reg.createVariable(ch, "f4", ("sample", "y_hr", "x_hr"), zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))

    g_pred = ds.createGroup("prediction")
    for ch in out_ch:
        g_pred.createVariable(ch, "f4", ("sample", "ensemble", "y_hr", "x_hr"), zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))

    g_tgt = ds.createGroup("target")
    for ch in out_ch:
        g_tgt.createVariable(ch, "f4", ("sample", "y_hr", "x_hr"), zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))
    return ds


def process_shard(worker_id: int, indices: np.ndarray, args, out_path: Path):
    """Runs in a spawned subprocess (or the main process for --gpus<=1):
    loads the model once, iterates its assigned sample indices, and writes
    (or resumes) its own shard file."""
    if args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    import netCDF4 as nc
    import torch

    from channels import IN_CH, INVARIANT_CH, OUT_CH, INPUT_MATCH
    from ccd_model import (Normalizer, build_sampler_fn, find_latest_checkpoint, init_single_process,
                           load_networks, run_patch, upsample_32_to_256)
    from collect_hrrr import precip_inv
    from collect_hrrr import compute_cos_sza
    from physical_constraints import apply_physical_constraints
    from rain_filter import apply_rain_filter

    TP_IDX = OUT_CH.index("tp")
    TP_IN_IDX = IN_CH.index("tp_in")

    device = init_single_process() if args.device.startswith("cuda") else torch.device("cpu")
    np.random.seed(args.seed + worker_id)
    torch.manual_seed(args.seed + worker_id)

    n = len(indices)
    resume = out_path.exists()
    ds_out = nc.Dataset(out_path, "r+") if resume else create_shard(out_path, n, args.ensemble, IN_CH, OUT_CH)
    if resume:
        log.info(f"[worker {worker_id}] resuming existing shard {out_path}")

    reg_ckpt = find_latest_checkpoint(args.reg_ckpt_dir, "CorrDiffRegressionUNet")
    res_ckpt = find_latest_checkpoint(args.res_ckpt_dir, "EDMPrecondSuperResolution")
    nets = load_networks(reg_ckpt, res_ckpt, device)
    sampler_fn = build_sampler_fn("stochastic", args.sampler_steps)
    normalizer = Normalizer(args.stats, IN_CH, OUT_CH, INVARIANT_CH)

    g_lr, g_lru = ds_out.groups["low_res_input"], ds_out.groups["low_res_input_upsampled"]
    g_reg, g_pred, g_tgt = ds_out.groups["regression"], ds_out.groups["prediction"], ds_out.groups["target"]

    ds_test = nc.Dataset(args.test_nc, "r")
    t0 = time.time()
    n_done = n_skipped = 0
    for i, idx in enumerate(indices):
        # resumability: a row already has a finite 2t target value -> done
        existing = np.array(g_tgt.variables[OUT_CH[0]][i])
        if np.isfinite(existing).all() and existing.size:
            n_skipped += 1
            continue

        x_lr, y, inv, lat, lon, t, lat_grid, lon_grid = read_sample(ds_test, int(idx), IN_CH, OUT_CH, INVARIANT_CH)
        x_up = upsample_32_to_256(x_lr)                          # (34,256,256) physical, for the baseline column
        # the ensemble is sampled in batches of seed_batch_size, which must divide the ensemble size
        # (otherwise the last, short batch would mismatch) -- fall back to 1 if it does not.
        effective_batch = min(args.seed_batch_size, args.ensemble)
        if args.ensemble % effective_batch != 0:
            effective_batch = 1
        unet_phys, ens_phys = run_patch(x_lr, inv, nets[0], nets[1], sampler_fn, normalizer, device,
                                        len(OUT_CH), args.ensemble, effective_batch)

        if args.rain_filter_mm is not None and args.rain_filter_mm >= 0:
            ens_phys[:, TP_IDX] = apply_rain_filter(ens_phys[:, TP_IDX], unet_phys[TP_IDX], args.rain_filter_mm)

        if not args.no_physical_constraints:       # ssrd/strd/tp >= 0, ssrd = 0 at night (cos SZA <= 0)
            cos_sza = compute_cos_sza(lat_grid, lon_grid, datetime.fromtimestamp(t, tz=timezone.utc))
            unet_phys = apply_physical_constraints(unet_phys, cos_sza)
            ens_phys = apply_physical_constraints(ens_phys, cos_sza)

        unet_out, pred_out, target_out = unet_phys.copy(), ens_phys.copy(), y.copy()
        unet_out[TP_IDX] = precip_inv(unet_out[TP_IDX])
        pred_out[:, TP_IDX] = precip_inv(pred_out[:, TP_IDX])
        target_out[TP_IDX] = precip_inv(target_out[TP_IDX])

        low_res_up = np.stack([
            precip_inv(x_up[IN_CH.index(INPUT_MATCH[v])]) if v == "tp" else x_up[IN_CH.index(INPUT_MATCH[v])]
            for v in OUT_CH
        ], axis=0)

        x_lr_write = x_lr.copy()
        x_lr_write[TP_IN_IDX] = precip_inv(x_lr_write[TP_IN_IDX])

        ds_out.variables["sample_idx"][i] = idx
        ds_out.variables["lat"][i] = lat
        ds_out.variables["lon"][i] = lon
        ds_out.variables["time"][i] = t
        for ci, ch in enumerate(IN_CH):
            g_lr.variables[ch][i] = x_lr_write[ci]
        for ci, ch in enumerate(OUT_CH):
            g_lru.variables[ch][i] = low_res_up[ci]
            g_reg.variables[ch][i] = unet_out[ci]
            g_tgt.variables[ch][i] = target_out[ci]
            g_pred.variables[ch][i] = pred_out[:, ci]

        n_done += 1
        if n_done % 25 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            log.info(f"[worker {worker_id}] {i+1}/{n} ({n_done} computed, {n_skipped} resumed) "
                     f"{elapsed:.0f}s elapsed, ETA {eta/60:.1f}min")
            ds_out.sync()

    ds_test.close()
    ds_out.complete = 1
    ds_out.regression_checkpoint = reg_ckpt
    ds_out.diffusion_checkpoint = res_ckpt
    ds_out.rain_filter_mm = args.rain_filter_mm if args.rain_filter_mm is not None else -1.0
    ds_out.close()
    log.info(f"[worker {worker_id}] done: {out_path} ({n_done} computed, {n_skipped} resumed)")


def _worker_entry(worker_id, indices, args, gpu_id, out_path):
    args.gpu_id = gpu_id
    args.device = "cuda"
    process_shard(worker_id, indices, args, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-nc", default=os.environ.get("CCD_DATA_DIR", ".") + "/regional_test.nc", help="held-out test file")
    p.add_argument("--reg-ckpt-dir", required=True, help="directory with CorrDiffRegressionUNet.*.mdlus")
    p.add_argument("--res-ckpt-dir", required=True, help="directory with EDMPrecondSuperResolution.*.mdlus")
    p.add_argument("--stats", required=True, help="stats.json of the training data")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--ensemble", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=int, default=1, help="-1 = all visible")
    p.add_argument("--seed-batch-size", type=int, default=8)
    p.add_argument("--sampler-steps", type=int, default=None)
    p.add_argument("--rain-filter-mm", type=float, default=0.02)
    p.add_argument("--no-rain-filter", action="store_true")
    p.add_argument("--no-physical-constraints", action="store_true",
                   help="skip the constraints ssrd/strd/tp >= 0 and ssrd = 0 at night")
    p.add_argument("--output-dir", default="./eval_data")
    p.add_argument("--cpu", action="store_true", help="force CPU (debugging only)")
    args = p.parse_args()
    if args.no_rain_filter:
        args.rain_filter_mm = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import netCDF4 as nc
    with nc.Dataset(args.test_nc, "r") as ds:
        n_total = ds.dimensions["sample"].size
    indices = select_indices(n_total, args.n_samples, args.seed)
    log.info(f"Selected {len(indices)}/{n_total} test-set samples (seed={args.seed})")

    if args.cpu:
        args.device = "cpu"
        process_shard(0, indices, args, out_dir / "shard_00.nc")
        return

    import torch
    n_visible = torch.cuda.device_count()
    n_workers = n_visible if args.gpus < 0 else max(1, min(args.gpus, n_visible or 1))
    if n_visible == 0:
        log.warning("No GPU visible -- running on CPU")
        args.device = "cpu"
        process_shard(0, indices, args, out_dir / "shard_00.nc")
        return

    shards = chunk(indices, n_workers)
    log.info(f"{n_workers} GPU worker(s), {[len(s) for s in shards]} samples/shard")

    if n_workers == 1:
        args.gpu_id = 0
        args.device = "cuda"
        process_shard(0, shards[0], args, out_dir / "shard_00.nc")
        return

    ctx = multiprocessing.get_context("spawn")
    procs = []
    for wid, shard_indices in enumerate(shards):
        out_path = out_dir / f"shard_{wid:02d}.nc"
        proc = ctx.Process(target=_worker_entry, args=(wid, shard_indices, args, wid, out_path))
        proc.start()
        procs.append(proc)
    ok = True
    for proc in procs:
        proc.join()
        ok = ok and proc.exitcode == 0
    if not ok:
        log.error("One or more workers failed -- check per-worker logs above")
        sys.exit(1)
    log.info(f"All shards written to {out_dir}")


if __name__ == "__main__":
    main()
