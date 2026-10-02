#!/usr/bin/env python3
"""
plot_qq.py — Q-Q plots, CorrDiff
(mean) vs HRRR truth, for u10 (wind speed), T2m, surface pressure, and Rain.

Run from this directory:
    python plot_qq.py --eval-dir ../data/eval --outdir ../output
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

N_SAMPLES = 300
N_PIX = 200_000


def flat_field(eval_ds, group, key, idx, wind=False, seed=42, n=N_PIX):
    arr = eval_ds.wind_speed(group, idx) if wind else eval_ds.get(group, key, idx)
    if group == "prediction":
        arr = arr.mean(axis=1)
    a = arr.ravel()
    rng = np.random.default_rng(seed)
    return rng.choice(a, min(n, len(a)), replace=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(N_SAMPLES, seed=42)

    panels = [("(a) u10", r"m s$^{-1}$", False, "10u", True),
              ("(b) T2m", "K", False, "2t", False),
              ("(c) $p_s$", "Pa", False, "sp", False),
              ("(d) Rain", "mm", True, "tp", False)]

    fig, axes = plt.subplots(1, 4, figsize=(DC * 1.1, 2.8),
                              gridspec_kw={"wspace": 0.42}, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.18)
    ql = np.linspace(0, 100, 500)

    for pi, (title, unit, log, key, wind) in enumerate(panels):
        ax = axes[pi]
        try:
            td = flat_field(eval_ds, "target", key, idx, wind=wind, seed=42)
            md = flat_field(eval_ds, "prediction", key, idx, wind=wind, seed=42)
            tq, mq = np.percentile(td, ql), np.percentile(md, ql)
            if log:
                m = (tq > 0) & (mq > 0)
                ax.loglog(tq[m], mq[m], color=CB["prediction"], lw=1.3)
                ax.loglog([tq[m].min(), tq[m].max()], [tq[m].min(), tq[m].max()], "k:", lw=0.8)
            else:
                ax.plot(tq, mq, color=CB["prediction"], lw=1.3, label="CorrDiff")
                ax.plot([tq[0], tq[-1]], [tq[0], tq[-1]], "k:", lw=0.8, label="y=x")
            for pct in ([90, 95, 99] if log else [75, 90, 99]):
                j = np.argmin(np.abs(ql - pct))
                ax.axvline(tq[j], color=CB["grey_lt"], lw=0.5, ls=":")
        except Exception as e:
            print(f"  s2 {key}: {e}")
        ax.set_xlabel(f"HRRR truth ({unit})", fontsize=7)
        if pi == 0:
            ax.set_ylabel(f"CorrDiff ({unit})", fontsize=7)
            ax.legend(fontsize=6)
        ax.set_title(title, fontsize=8, pad=3)
        panel_label(ax, chr(97 + pi))

    save_fig(fig, f"{a.outdir}/plot_qq.pdf")
    plt.close(fig)
    eval_ds.close()
    print("S2 done.")


if __name__ == "__main__":
    main()
