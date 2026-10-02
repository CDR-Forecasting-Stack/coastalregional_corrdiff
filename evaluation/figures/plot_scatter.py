#!/usr/bin/env python3
"""
plot_scatter.py — CorrDiff (mean)
vs HRRR truth scatter, 8 variables. (a) one point per sample (spatial mean),
(b) all-pixel (subsampled).

Run from this directory:
    python plot_scatter.py --eval-dir ../data/eval --outdir ../output
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import ALL_VARS, CB, CMAP, DC, apply_style, open_eval, panel_label, save_fig

N_SAMPLES_MEAN = 1000
N_SAMPLES_PIXEL = 400
N_PIX = 400_000


def pairs_mean(eval_ds, key, idx):
    pm = eval_ds.get("prediction", key, idx).mean(axis=1).reshape(len(idx), -1).mean(axis=1)
    tr = eval_ds.get("target", key, idx).reshape(len(idx), -1).mean(axis=1)
    return pm, tr


def pairs_pixel(eval_ds, key, idx, seed=42):
    pm = eval_ds.get("prediction", key, idx).mean(axis=1).ravel()
    tr = eval_ds.get("target", key, idx).ravel()
    rng = np.random.default_rng(seed)
    if len(pm) > N_PIX:
        sel = rng.choice(len(pm), N_PIX, replace=False)
        pm, tr = pm[sel], tr[sel]
    return pm, tr


def render(eval_ds, pairs_fn, idx, title, out_path):
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(DC, DC * 0.62),
                              gridspec_kw={"hspace": 0.5, "wspace": 0.40})
    for i, vd in enumerate(ALL_VARS):
        ax = axes.ravel()[i]
        k = vd["key"]
        try:
            pm, tr = pairs_fn(eval_ds, k, idx)
        except Exception as e:
            print(f"  S1 {k}: {e}")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            continue
        log = k == "tp"
        if log:
            pm = np.log10(np.clip(pm, 1e-3, None)); tr = np.log10(np.clip(tr, 1e-3, None))
        ax.hexbin(tr, pm, gridsize=40, cmap=CMAP["density"], mincnt=1, linewidths=0.1, bins="log")
        lims = [min(pm.min(), tr.min()), max(pm.max(), tr.max())]
        ax.plot(lims, lims, "k--", lw=0.8, zorder=5)
        r2 = 1 - np.sum((tr - pm) ** 2) / np.sum((tr - tr.mean()) ** 2)
        rm = float(np.sqrt(np.mean((pm - tr) ** 2)))
        ax.text(0.05, 0.95, f"$R^2$={r2:.3f}\nRMSE={rm:.3g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6, color=CB["prediction"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
        pre = "log " if log else ""
        ax.set_xlabel(f"{pre}HRRR truth ({vd['unit']})", fontsize=6.5)
        ax.set_ylabel(f"{pre}CorrDiff ({vd['unit']})", fontsize=6.5)
        ax.set_title(vd["label"], fontsize=8, pad=3, color=CB["prediction"], fontweight="bold")
        panel_label(ax, chr(97 + i), x=-0.22, y=1.10)
    fig.suptitle(title, fontsize=9, y=1.0)
    save_fig(fig, out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    eval_ds = open_eval(a.eval_dir)
    idx_mean = eval_ds.random_indices(N_SAMPLES_MEAN, seed=42)
    render(eval_ds, pairs_mean, idx_mean,
           f"CorrDiff (mean) vs HRRR — sample spatial means (n={len(idx_mean)})",
           f"{a.outdir}/figure_S1a_scatter.pdf")
    print("S1a (sample-mean) done.")

    idx_pix = eval_ds.random_indices(N_SAMPLES_PIXEL, seed=42)
    render(eval_ds, pairs_pixel, idx_pix,
           "CorrDiff (mean) vs HRRR — all pixels (subsampled)",
           f"{a.outdir}/figure_S1b_scatter.pdf")
    print("S1b (all-pixel) done.")
    eval_ds.close()


if __name__ == "__main__":
    main()
