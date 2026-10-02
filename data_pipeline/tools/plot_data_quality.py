#!/usr/bin/env python3
"""
plot_data_quality.py -- quality / distribution checks of a collected NetCDF (train or test).
  scatter_patchmean.png     density scatter, patch-mean(coarse input) vs patch-mean(HRRR target), --n-stat random samples
  scatter_pixel.png         density scatter, bilinear-upsampled coarse input vs HRRR target, pixel-based, --n-maps random samples
  maps_sample<k>.png        coarse input (32x32) vs HRRR target (256x256), the --n-maps samples
  hist_targets.png / hist_inputs.png / hist_coarse_vs_target.png   pixel distributions over --n-stat random samples
"""
import argparse, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as Fn
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # data_pipeline/
import collect_hrrr as collect

IN_NAME = {"2t": "t2m", "10u": "u10", "10v": "v10", "tp": "tp_in", "ssrd": "ssrd", "strd": "strd", "sp": "sp", "q": "q_sfc"}
UNIT = {"2t": "K", "10u": "m/s", "10v": "m/s", "tp": "log10(1+mm)", "ssrd": "W/m²", "strd": "W/m²", "sp": "Pa", "q": "kg/kg"}


def nn_up(a, f=8):
    return np.repeat(np.repeat(a, f, -2), f, -1)


def bilinear_up(a, f=8):
    return Fn.interpolate(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None], scale_factor=f,
                          mode="bilinear", align_corners=False)[0, 0].numpy()


def dens(ax, x, y, label_x, label_y, title):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    lo, hi = np.percentile(np.concatenate([x, y]), [0.1, 99.9]); pad = .03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    hb = ax.hexbin(x, y, gridsize=90, extent=(lo, hi, lo, hi), bins="log", cmap="viridis", mincnt=1)
    ax.plot([lo, hi], [lo, hi], "r--", lw=.8)
    r = np.corrcoef(x, y)[0, 1]; rmse = np.sqrt(np.mean((y - x) ** 2)); bias = np.mean(x - y)
    ax.set_title(f"{title}\nr={r:.3f} RMSE={rmse:.3g} bias(in−tgt)={bias:.3g}", fontsize=8)
    ax.set_xlabel(label_x, fontsize=8); ax.set_ylabel(label_y, fontsize=8); ax.tick_params(labelsize=7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    return hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default="./data_quality")
    ap.add_argument("--n-maps", type=int, default=10)
    ap.add_argument("--n-stat", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(a.dataset); n = ds.dimensions["sample"].size
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    map_idx = np.sort(perm[:a.n_maps]); stat_idx = np.sort(perm[:a.n_stat])        # map samples are a subset of the stat samples
    coords = np.array(ds["coord"][:]); lat = np.array(ds.groups["invariant"]["latitude"][:]); lon = np.array(ds.groups["invariant"]["longitude"][:])
    times = np.array(ds["time"][:])
    gi, go = ds.groups["input"], ds.groups["output"]

    # ---- read the statistics subset (inputs full, targets full-res per sample)
    I = {c: np.stack([np.array(gi[c][i]) for i in stat_idx]) for c in collect.IN_CH}      # (S,32,32)
    O_mean = {v: np.zeros(len(stat_idx), np.float32) for v in collect.OUT_CH}
    O_pix = {v: [] for v in collect.OUT_CH}                                                # every 8th pixel each way
    O_map = {}
    for k, i in enumerate(stat_idx):
        for v in collect.OUT_CH:
            x = np.array(go[v][i])
            O_mean[v][k] = x.mean(); O_pix[v].append(x[::8, ::8].ravel())
            if i in map_idx: O_map.setdefault(int(i), {})[v] = x
        if k % 500 == 0: print(f"  read {k}/{len(stat_idx)}", flush=True)
    O_pix = {v: np.concatenate(x) for v, x in O_pix.items()}
    ds.close()
    pos = {int(i): k for k, i in enumerate(stat_idx)}

    # ---- patch-mean density scatter
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.5))
    for x_, v in zip(ax.ravel(), collect.OUT_CH):
        hb = dens(x_, I[IN_NAME[v]].mean((1, 2)), O_mean[v], "coarse-input patch mean", "HRRR-target patch mean", f"{v} [{UNIT[v]}]")
    fig.colorbar(hb, ax=ax, fraction=.015, label="count"); fig.suptitle(f"Patch-mean density scatter ({len(stat_idx)} random samples)", fontsize=12)
    fig.savefig(out / "scatter_patchmean.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- pixel density scatter on the map samples (bilinear upsampled input)
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.5))
    for x_, v in zip(ax.ravel(), collect.OUT_CH):
        xs = np.concatenate([bilinear_up(I[IN_NAME[v]][pos[int(i)]]).ravel() for i in map_idx])
        ys = np.concatenate([O_map[int(i)][v].ravel() for i in map_idx])
        hb = dens(x_, xs, ys, "bilinear-upsampled coarse input", "HRRR target", f"{v} [{UNIT[v]}]")
    fig.colorbar(hb, ax=ax, fraction=.015, label="count"); fig.suptitle(f"Pixel-based density scatter ({len(map_idx)} random samples x 65,536 px)", fontsize=12)
    fig.savefig(out / "scatter_pixel.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- maps
    for k, i in enumerate(map_idx):
        fig, ax = plt.subplots(2, 8, figsize=(26, 6.6), gridspec_kw={"wspace": .08, "hspace": .12})
        for j, v in enumerate(collect.OUT_CH):
            lr = I[IN_NAME[v]][pos[int(i)]]; hr = O_map[int(i)][v]
            lo, hi = np.nanpercentile(np.concatenate([nn_up(lr).ravel(), hr.ravel()]), [1, 99])
            cmap = "RdBu_r" if v in ("10u", "10v") else "viridis"
            if v in ("10u", "10v"): m = max(abs(lo), abs(hi)); lo, hi = -m, m
            ax[0, j].imshow(nn_up(lr), vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
            im = ax[1, j].imshow(hr, vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
            ax[0, j].set_title(f"{v} [{UNIT[v]}]", fontsize=9)
            fig.colorbar(im, ax=ax[:, j], orientation="horizontal", fraction=.04, pad=.03).ax.tick_params(labelsize=6)
        for x_ in ax.ravel(): x_.set_xticks([]); x_.set_yticks([])
        ax[0, 0].set_ylabel("coarse input 32x32"); ax[1, 0].set_ylabel("HRRR target 256x256")
        r, c = coords[i]
        import datetime as dt
        fig.suptitle(f"sample {i}  {dt.datetime.utcfromtimestamp(int(times[i])):%Y-%m-%d %H:%M} UTC  centre lat {lat[r, c]:.1f} lon {lon[r, c]:.1f}", fontsize=11)
        fig.savefig(out / f"maps_sample{k + 1}.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- distributions over the whole statistics subset
    def hist(ax, x, color="C0", label=None):
        x = x[np.isfinite(x)]; lo, hi = np.percentile(x, [0.1, 99.9])
        if hi <= lo: hi = lo + 1e-6
        ax.hist(x, bins=100, range=(lo, hi), density=True, histtype="step", color=color, label=label)
    fig, ax = plt.subplots(6, 6, figsize=(20, 16)); ax = ax.ravel()
    for x_, c in zip(ax, collect.IN_CH): hist(x_, I[c].ravel()); x_.set_title(c, fontsize=9); x_.set_yscale("log"); x_.tick_params(labelsize=7)
    for x_ in ax[len(collect.IN_CH):]: x_.axis("off")
    fig.suptitle(f"Input channels, pixel histograms ({len(stat_idx)} samples x 32x32)", fontsize=12)
    fig.savefig(out / "hist_inputs.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel()
    for x_, v in zip(ax, collect.OUT_CH): hist(x_, O_pix[v], "C1"); x_.set_title(f"{v} [{UNIT[v]}]"); x_.set_yscale("log")
    fig.suptitle(f"Target channels, pixel histograms ({len(stat_idx)} samples, every 8th pixel)", fontsize=12)
    fig.savefig(out / "hist_targets.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel()
    for x_, v in zip(ax, collect.OUT_CH):
        hist(x_, I[IN_NAME[v]].ravel(), "C0", "coarse input"); hist(x_, O_pix[v], "C1", "HRRR target")
        x_.set_title(f"{v} [{UNIT[v]}]"); x_.set_yscale("log"); x_.legend(fontsize=8)
    fig.suptitle("Coarse input vs high-res target, pixel distributions", fontsize=12)
    fig.savefig(out / "hist_coarse_vs_target.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("figures ->", out)


if __name__ == "__main__":
    main()
