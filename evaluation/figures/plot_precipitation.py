#!/usr/bin/env python3
"""
plot_precipitation.py — precipitation verification.
(a) wet fraction (b) intensity PDF (c) CSI/ETS/FreqBias (d) FSS

Uses CorrDiff's ensemble mean for tp, which already has the
rain filter applied at eval-generation time (matching the model's actual
deployed/production behavior).

Run from this directory:
    python plot_precipitation.py --eval-dir ../data/eval --outdir ../output
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import CB, DC, apply_style, open_eval, panel_label, save_fig
from metrics import csi_ets_freqbias, fss

WET_THRESH = 0.1  # mm
N_SAMP = 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-samp", type=int, default=N_SAMP)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_samp, seed=11)
    truth = eval_ds.get("target", "tp", idx)
    pred = eval_ds.get("prediction", "tp", idx).mean(axis=1)  # ensemble mean

    fig, axes = plt.subplots(1, 4, figsize=(DC * 1.1, 2.8),
                              gridspec_kw={"wspace": 0.50}, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.20)

    # (a) wet fraction
    ax = axes[0]
    wt = float((truth > WET_THRESH).mean())
    wp = float((pred > WET_THRESH).mean())
    bars = ax.bar(["HRRR", "CorrDiff"], [wt * 100, wp * 100],
                   color=[CB["truth"], CB["prediction"]], width=0.55)
    for b, v in zip(bars, [wt * 100, wp * 100]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Wet fraction (%)"); ax.set_title("(a) Wet fraction", fontsize=8)
    ax.set_ylim(0, max(wt, wp) * 100 * 1.3); panel_label(ax, "a")

    # (b) intensity PDF
    ax = axes[1]
    rng = np.random.default_rng(42)
    td = truth.ravel(); pd_ = pred.ravel()
    n_pix = 150_000
    if len(td) > n_pix:
        sel = rng.choice(len(td), n_pix, replace=False)
        td, pd_ = td[sel], pd_[sel]
    bins = np.logspace(-2, 2, 55)
    for data, col, lbl, ls in [(td, CB["truth"], "HRRR", "-"),
                                (pd_, CB["prediction"], "CorrDiff", "--")]:
        w = data[data > 0.01]
        h, e = np.histogram(w, bins=bins, density=True); c = np.sqrt(e[:-1] * e[1:])
        ax.loglog(c, h, color=col, lw=1.2, ls=ls, label=lbl)
    ax.set_xlabel("Rain (mm)"); ax.set_ylabel("Density")
    ax.set_title("(b) Intensity PDF", fontsize=8); ax.legend(fontsize=6)
    panel_label(ax, "b")

    # (c) CSI / ETS / FreqBias
    ax = axes[2]
    tk = [0.1, 1.0, 5.0]; x = np.arange(3); bw = 0.25
    csi_v, ets_v, fb_v = zip(*[csi_ets_freqbias(pred, truth, t) for t in tk])
    ax.bar(x - bw, csi_v, bw, label="CSI", color=CB["low_res"])
    ax.bar(x, ets_v, bw, label="ETS", color=CB["regression"])
    ax.bar(x + bw, fb_v, bw, label="FBias", color=CB["prediction"])
    ax.axhline(1, color=CB["accent"], lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(["0.1", "1.0", "5.0"])
    ax.set_xlabel("Threshold (mm)")
    ax.set_title("(c) Categorical", fontsize=8); ax.legend(fontsize=5.5)
    panel_label(ax, "c")

    # (d) FSS
    ax = axes[3]
    scales_px = np.array([1, 3, 5, 10, 20, 30])
    scales_km = scales_px * 3.0 * 2 + 3.0
    for thresh, color, marker in [(1.0, CB["prediction"], "o"), (3.0, CB["regression"], "s"),
                                   (5.0, CB["accent"], "^")]:
        ot = (truth > thresh); op = (pred > thresh)
        fss_vals = [fss(op, ot, s) for s in scales_px]
        ax.plot(scales_km, fss_vals, color=color, lw=1.3, marker=marker, ms=3,
                label=f"CorrDiff @ {thresh:g} mm")
    ax.axhline(0.5, color=CB["grey"], lw=0.7, ls=":", label="useful (0.5)")
    ax.set_xlabel("Neighborhood width (km)"); ax.set_ylabel("FSS")
    ax.set_title("(d) FSS", fontsize=8); ax.set_ylim(0, 1)
    ax.legend(fontsize=5.5, loc="lower right"); panel_label(ax, "d")

    save_fig(fig, f"{a.outdir}/plot_precipitation.pdf")
    plt.close(fig)
    eval_ds.close()
    print("done.")


if __name__ == "__main__":
    main()
