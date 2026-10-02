"""
ccd_model.py -- shared model-side helpers for inference and evaluation (PhysicsNeMo CorrDiff)
==============================================================================================
Loads the two trained networks (regression UNet + EDM diffusion residual model) with NVIDIA
PhysicsNeMo, normalizes inputs exactly like the training dataset class (training/physicsnemo/
regional_hrrr.py), and runs one 256x256 patch through regression (+ diffusion ensemble).

Used by inference/infer_region.py and evaluation/generate_eval_dataset.py so the two can never
drift apart.  Requires PhysicsNeMo (``pip install nvidia-physicsnemo``).
"""
from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from numba import jit, prange

from physicsnemo import Module
from physicsnemo.diffusion.generate import diffusion_step, regression_step
from physicsnemo.diffusion.samplers import deterministic_sampler, stochastic_sampler
from physicsnemo.distributed import DistributedManager

HR_SIZE = 256
UPSAMPLE_FACTOR = 8
HR_MEAN_CONDITIONING = True   # property of the trained diffusion checkpoint (conf/base/model/diffusion.yaml)


@jit(nopython=True, cache=True)
def _zoom_extrapolate(x, y, factor):
    """Bilinear zoom with extrapolation (identical to training/physicsnemo/regional_hrrr.py)."""
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1
                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)


def upsample_32_to_256(x: np.ndarray) -> np.ndarray:
    """(C,32,32) coarse input -> (C,256,256) by bilinear interpolation (as in training)."""
    y = np.empty((x.shape[0], HR_SIZE, HR_SIZE), dtype=np.float32)
    _zoom_extrapolate(x.astype(np.float32), y, UPSAMPLE_FACTOR)
    return y


def find_latest_checkpoint(ckpt_dir: str, prefix: str) -> str:
    """Highest-iteration ``<prefix>.<rank>.<iter>.mdlus`` under ckpt_dir (searched recursively)."""
    paths = list(Path(ckpt_dir).rglob(f"{prefix}.*.mdlus"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}*.mdlus checkpoints found under {ckpt_dir}")

    def iter_num(p: Path) -> int:
        try:
            return int(p.stem.split(".")[-1])
        except ValueError:
            return -1

    return str(max(paths, key=iter_num))


def load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(np.float32)
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(np.float32)
    return mean, std


class Normalizer:
    """z-score normalization of the 36-channel model input (34 coarse channels + 2 invariants) and
    de-normalization of the 8-channel output, reproducing RegionalHRRRDataset exactly."""

    def __init__(self, stats_path, in_ch, out_ch, invariant_ch=("elev_mean", "lsm_mean")):
        with open(stats_path) as f:
            stats = json.load(f)
        im, istd = load_stats(stats, in_ch, "input")
        vm, vstd = load_stats(stats, list(invariant_ch), "invariant")
        self.input_mean = np.concatenate([im, vm], axis=0)
        self.input_std = np.concatenate([istd, vstd], axis=0)
        self.output_mean, self.output_std = load_stats(stats, out_ch, "output")

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def denormalize_output(self, x):
        return x * self.output_std + self.output_mean


def init_single_process():
    """Return the compute device, initialising PhysicsNeMo's DistributedManager for single-process
    use when a GPU is present. (Pins RANK/WORLD_SIZE so it does not try to rendezvous under SLURM.)
    On a CPU-only machine the manager cannot initialise, so it is skipped and the CPU is returned."""
    import os
    if not torch.cuda.is_available():
        return torch.device("cpu")
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0"); os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    return DistributedManager().device


def load_networks(reg_ckpt: str, res_ckpt: str | None, device):
    """Load the regression network and (optionally) the diffusion residual network."""
    def _load(path):
        net = Module.from_checkpoint(path, override_args={"use_apex_gn": False})
        net = net.eval().to(device).to(memory_format=torch.channels_last)
        if hasattr(net, "amp_mode"):
            net.amp_mode = False
        return net
    return _load(reg_ckpt), (_load(res_ckpt) if res_ckpt else None)


def build_sampler_fn(sampler: str = "stochastic", steps: int | None = None, solver: str = "euler"):
    if sampler == "stochastic":
        return partial(stochastic_sampler, num_steps=steps or 18, patching=None)
    return partial(deterministic_sampler, num_steps=steps or 9, solver=solver, patching=None)


def run_patch(lr, inv_patch, net_reg, net_res, sampler_fn, normalizer, device,
              n_out: int, num_ensembles: int, seed_batch_size: int):
    """One patch: (34,32,32) physical coarse input + (2,256,256) invariants -> regression prediction
    (n_out,256,256) and ensemble (E,n_out,256,256), both in physical units (tp in log10(1+mm)).
    Returns None if `lr` has any non-finite value; ensemble is None when net_res is None."""
    if not np.isfinite(lr).all():
        return None
    x = np.concatenate([upsample_32_to_256(lr), inv_patch], axis=0)          # (36,256,256)
    x_norm = normalizer.normalize_input(x).astype(np.float32)
    img_lr = torch.from_numpy(x_norm[None]).to(device=device, dtype=torch.float32)
    img_lr = img_lr.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        image_reg = regression_step(net=net_reg, img_lr=img_lr, latents_shape=(1, n_out, HR_SIZE, HR_SIZE))
        unet_phys = normalizer.denormalize_output(image_reg[0].float().cpu().numpy())
        ens_phys = None
        if net_res is not None:
            mean_hr = image_reg[0:1] if HR_MEAN_CONDITIONING else None
            seeds = torch.arange(num_ensembles)
            rank_batches = list(seeds.split(seed_batch_size))
            image_res = diffusion_step(
                net=net_res, sampler_fn=sampler_fn, img_shape=(HR_SIZE, HR_SIZE), img_out_channels=n_out,
                rank_batches=rank_batches,
                img_lr=img_lr.expand(seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                rank=0, device=device, mean_hr=mean_hr,
            )
            image_out = image_reg.expand(num_ensembles, -1, -1, -1) + image_res
            ens_phys = normalizer.denormalize_output(image_out.float().cpu().numpy())
    return unet_phys, ens_phys
