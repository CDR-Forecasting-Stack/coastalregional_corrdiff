#!/usr/bin/env python3
"""
plot_rapsd.py — radially-averaged power spectral density,
all 8 output variables. Shows whether CorrDiff restores
truth's high-wavenumber (small-scale) power that the UNet Regression stage
blurs away, and that a single ensemble member (not the mean, which
averages away high-frequency detail the same way regression does) is the
correct comparison for that claim.

Run from this directory:
    python plot_rapsd.py --eval-dir ../data/eval --outdir ../output
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import ALL_VARS, CB, DC, apply_style, open_eval, panel_label, save_fig
from _rapsd_core import mean_rapsd

N_SAMPLES = 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)

    fig, axes = plt.subplots(2, 4, figsize=(DC, 5.2),
                              gridspec_kw={"hspace": 0.5, "wspace": 0.34})

    for pi, vd in enumerate(ALL_VARS):
        ax = axes.ravel()[pi]
        k = vd["key"]
        wind = k == "10u"  # plot wind speed spectrum once (u10 panel), not u/v separately
        print(f"  RAPSD {k}...")
        try:
            wl, ps_truth = mean_rapsd(eval_ds, k, "target", n_samples=a.n_samples, wind=wind)
            _, ps_lr = mean_rapsd(eval_ds, k, "low_res_input_upsampled", n_samples=a.n_samples, wind=wind)
            _, ps_reg = mean_rapsd(eval_ds, k, "regression", n_samples=a.n_samples, wind=wind)
            _, ps_mean = mean_rapsd(eval_ds, k, "prediction_mean", n_samples=a.n_samples, wind=wind)
            _, ps_mem = mean_rapsd(eval_ds, k, "prediction_member", n_samples=a.n_samples, wind=wind)

            # Layering: HRRR truth is bottommost (it's black -- everything
            # else should draw over it), CorrDiff (member) is topmost, in
            # red, and drawn narrower so it reads as an overlay on top of
            # every other line rather than competing with them.
            ax.axvline(25, color=CB["grey"], lw=0.7, ls=":", zorder=0)
            ax.loglog(wl, ps_truth, color=CB["truth"], lw=2.0, alpha=0.9,
                      solid_capstyle="round", label="HRRR truth", zorder=1)
            ax.loglog(wl, ps_lr, color=CB["low_res"], lw=1.1, ls=":", label="Low-res HRRR", zorder=2)
            ax.loglog(wl, ps_reg, color=CB["regression"], lw=1.1, ls="--", label="UNet Regression", zorder=3)
            ax.loglog(wl, ps_mean, color=CB["prediction"], lw=1.4, ls="-.",
                      label="CorrDiff (mean)", zorder=4)
            ax.loglog(wl, ps_mem, color="red", lw=0.8, alpha=0.9,
                      label="CorrDiff (member)", zorder=5)
            ax.invert_xaxis()
        except Exception as e:
            print(f"  RAPSD {k} failed: {e}")
            ax.text(0.5, 0.5, "data error", transform=ax.transAxes, ha="center", va="center")

        ax.set_xlabel("Wavelength (km)", fontsize=7.5)
        if pi in (0, 4):
            ax.set_ylabel("PSD", fontsize=7.5)
        ax.set_title(vd["label"], fontsize=9, pad=4)
        panel_label(ax, chr(97 + pi), x=-0.2, y=1.08)
        if pi == 0:
            ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9)

    save_fig(fig, f"{a.outdir}/plot_rapsd.pdf")
    plt.close(fig)
    eval_ds.close()
    print("done.")


if __name__ == "__main__":
    main()
