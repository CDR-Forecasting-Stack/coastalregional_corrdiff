#!/usr/bin/env python3
"""
plot_coarsen_test.py -- sanity figures for a small test collection (HRRR only).
  maps_patch<k>.png   for 5 random patches: coarse input (32x32) row vs high-res target (256x256) row, 8 target variables
  hist_inputs.png     pixel histograms of all input channels (all patches)
  hist_targets.png    pixel histograms of the 8 target channels
  hist_coarse_vs_target.png  coarse input vs high-res target, for the 8 shared variables (same units)
"""
import argparse, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # data_pipeline/
import collect_hrrr as collect

IN_NAME = {"2t": "t2m", "10u": "u10", "10v": "v10", "tp": "tp_in", "ssrd": "ssrd", "strd": "strd", "sp": "sp", "q": "q_sfc"}
UNIT = {"2t": "K", "10u": "m/s", "10v": "m/s", "tp": "log10(1+mm)", "ssrd": "W/m²", "strd": "W/m²", "sp": "Pa", "q": "kg/kg"}


def nn_up(a, f=8):
    return np.repeat(np.repeat(a, f, -2), f, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default="./coarsen_test_figs")
    ap.add_argument("--n-maps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(a.dataset)
    n = ds.dimensions["sample"].size
    print("coarsening attr:", getattr(ds, "coarsening", "?"), "| samples:", n)
    I = {c: np.array(ds.groups["input"][c][:]) for c in collect.IN_CH}      # (N,32,32)
    O = {c: np.array(ds.groups["output"][c][:]) for c in collect.OUT_CH}    # (N,256,256)
    coords = np.array(ds["coord"][:])
    lat = np.array(ds.groups["invariant"]["latitude"][:]); lon = np.array(ds.groups["invariant"]["longitude"][:])
    ds.close()

    rng = np.random.default_rng(a.seed)
    for k, i in enumerate(np.sort(rng.choice(n, min(a.n_maps, n), replace=False))):
        fig, ax = plt.subplots(2, 8, figsize=(26, 6.6), gridspec_kw={"wspace": .08, "hspace": .12})
        for j, v in enumerate(collect.OUT_CH):
            lr = I[IN_NAME[v]][i]; hr = O[v][i]
            lo, hi = np.nanpercentile(np.concatenate([nn_up(lr).ravel(), hr.ravel()]), [1, 99])
            cmap = "RdBu_r" if v in ("10u", "10v") else "viridis"
            if v in ("10u", "10v"):
                m = max(abs(lo), abs(hi)); lo, hi = -m, m
            ax[0, j].imshow(nn_up(lr), vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
            im = ax[1, j].imshow(hr, vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
            ax[0, j].set_title(f"{v} [{UNIT[v]}]", fontsize=9)
            fig.colorbar(im, ax=ax[:, j], orientation="horizontal", fraction=.04, pad=.03).ax.tick_params(labelsize=6)
        for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
        ax[0, 0].set_ylabel("coarse input 32x32\n(Gauss σ=5 + xESMF)"); ax[1, 0].set_ylabel("HRRR target 256x256")
        r, c = coords[i]
        fig.suptitle(f"sample {i}: patch centre lat {lat[r, c]:.1f} lon {lon[r, c]:.1f}  (row {r}, col {c})", fontsize=11)
        fig.savefig(out / f"maps_patch{k + 1}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    def hist(ax, x, color="C0", label=None):
        x = x[np.isfinite(x)]
        lo, hi = np.percentile(x, [0.1, 99.9])
        if hi <= lo: hi = lo + 1e-6
        ax.hist(x, bins=100, range=(lo, hi), density=True, histtype="step", color=color, label=label)

    names = collect.IN_CH
    fig, ax = plt.subplots(6, 6, figsize=(20, 16)); ax = ax.ravel()
    for x, c in zip(ax, names):
        hist(x, I[c].ravel()); x.set_title(c, fontsize=9); x.set_yscale("log"); x.tick_params(labelsize=7)
    for x in ax[len(names):]: x.axis("off")
    fig.suptitle(f"Input channels, pixel histograms ({n} patches x 32x32 px)", fontsize=12)
    fig.savefig(out / "hist_inputs.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel()
    for x, v in zip(ax, collect.OUT_CH):
        hist(x, O[v].ravel(), "C1"); x.set_title(f"{v} [{UNIT[v]}]"); x.set_yscale("log")
    fig.suptitle(f"Target channels, pixel histograms ({n} patches x 256x256 px)", fontsize=12)
    fig.savefig(out / "hist_targets.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel()
    for x, v in zip(ax, collect.OUT_CH):
        hist(x, I[IN_NAME[v]].ravel(), "C0", "coarse input"); hist(x, O[v].ravel(), "C1", "HRRR target")
        x.set_title(f"{v} [{UNIT[v]}]"); x.set_yscale("log"); x.legend(fontsize=8)
    fig.suptitle("Coarse input vs high-res target, pixel distributions", fontsize=12)
    fig.savefig(out / "hist_coarse_vs_target.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    print("nan frac inputs:", {c: float(np.isnan(I[c]).mean()) for c in names if np.isnan(I[c]).any()} or "none")
    print("nan frac targets:", {c: float(np.isnan(O[c]).mean()) for c in O if np.isnan(O[c]).any()} or "none")
    print("figures ->", out)


if __name__ == "__main__":
    main()
