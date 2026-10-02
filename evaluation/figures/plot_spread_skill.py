#!/usr/bin/env python3
"""
plot_spread_skill.py — spread-skill relationship, 8 variables
(heatmap + scatter variants).

Run from this directory:
    python plot_spread_skill.py --eval-dir ../data/eval --outdir ../output
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

N_SAMP = 1000


def spread_skill(eval_ds, key, idx):
    truth = eval_ds.get("target", key, idx)
    ens = eval_ds.get("prediction", key, idx)  # (N,E,H,W)
    mean = ens.mean(axis=1)
    spread = ens.std(axis=1).reshape(len(idx), -1).mean(axis=1)
    skill = np.sqrt(((mean - truth) ** 2).reshape(len(idx), -1).mean(axis=1))
    return spread, skill


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-samp", type=int, default=N_SAMP)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_samp, seed=42)

    for variant in ("heatmap", "scatter"):
        fig, axes = plt.subplots(2, 4, figsize=(DC, DC * 0.55),
                                  gridspec_kw={"wspace": 0.48, "hspace": 0.5})
        for pi, vd in enumerate(ALL_VARS):
            ax = axes.ravel()[pi]
            k = vd["key"]
            try:
                s, r = spread_skill(eval_ds, k, idx)
                ssr = float(s.mean() / r.mean())
                if variant == "heatmap":
                    ax.hexbin(s, r, gridsize=28, cmap=CMAP["density"], mincnt=1, linewidths=0.1)
                else:
                    ax.scatter(s, r, s=4, alpha=0.35, color=CB["prediction"],
                               edgecolors="none", rasterized=True)
                m, b = np.polyfit(s, r, 1)
                xf = np.linspace(s.min(), s.max(), 50)
                ax.plot(xf, m * xf + b, color=CB["accent"], lw=1.2, zorder=5, label="fit")
                lo = min(float(s.min()), float(r.min()))
                hi = max(float(s.max()), float(r.max()))
                ax.plot([lo, hi], [lo, hi], color="#111", lw=0.8, ls="--", zorder=6, label="1:1")
                ax.text(0.05, 0.95, f"SSR={ssr:.3f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=6.5, fontweight="bold", color=CB["prediction"])
                ax.set_xlim(0, float(s.max()) * 1.08)
                ax.set_ylim(0, float(r.max()) * 1.08)
            except Exception as e:
                print(f"  fig4 {k}: {e}")
            ax.set_xlabel(rf"Spread $\sigma$ ({vd['unit']})", fontsize=7)
            if pi % 4 == 0:
                ax.set_ylabel(f"RMSE ({vd['unit']})", fontsize=7)
            ax.set_title(vd["label"], fontsize=8, pad=3)
            panel_label(ax, chr(97 + pi), x=-0.22, y=1.08)
            if pi == 0:
                ax.legend(fontsize=5.5, loc="lower right")
        suffix = "a_heatmap" if variant == "heatmap" else "b_scatter"
        save_fig(fig, f"{a.outdir}/figure_04{suffix}_spread_skill.pdf")
        plt.close(fig)
    eval_ds.close()
    print("done.")


if __name__ == "__main__":
    main()
