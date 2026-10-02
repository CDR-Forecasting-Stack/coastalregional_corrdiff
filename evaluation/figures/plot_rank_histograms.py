#!/usr/bin/env python3
"""
plot_rank_histograms.py — rank histograms, 8 variables.

Real per-pixel rank histograms against the actual stored ensemble (no
Gaussian-synthesis approximation needed, since every sample carries its
full, real 16-member ensemble). For tp/ssrd, pixels are filtered to
truth > threshold first to remove zero-inflation (dry pixels, night
radiation) so the histogram reflects calibration on real events only.

Run from this directory:
    python plot_rank_histograms.py --eval-dir ../data/eval --outdir ../output
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
from metrics import rank_histogram

N_SAMP = 1000
FILTER_THRESH = {"tp": 0.1, "ssrd": 1.0}
MAX_PIX_PER_SAMPLE = 1500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-samp", type=int, default=N_SAMP)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_samp, seed=7)
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(2, 4, figsize=(DC, 3.8),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.34})

    for i, vd in enumerate(ALL_VARS):
        ax = axes.ravel()[i]
        k = vd["key"]
        print(f"  Fig3 {k}...")
        truth = eval_ds.get("target", k, idx)
        ens = eval_ds.get("prediction", k, idx)  # (N,E,H,W)

        mask = None
        if k in FILTER_THRESH:
            mask = truth > FILTER_THRESH[k]
            # subsample masked pixels per sample for tractability
            for s in range(mask.shape[0]):
                ys, xs = np.where(mask[s])
                if len(ys) > MAX_PIX_PER_SAMPLE:
                    keep = rng.choice(len(ys), MAX_PIX_PER_SAMPLE, replace=False)
                    drop_mask = np.ones(len(ys), dtype=bool)
                    drop_mask[keep] = False
                    mask[s, ys[drop_mask], xs[drop_mask]] = False

        counts = rank_histogram(truth, ens, mask=mask)
        nb = len(counts)
        a_ = counts.astype(float)
        total = a_.sum()
        a_ = a_ / total if total > 0 else a_

        ax.bar(np.arange(nb), a_, width=1.0, align="edge",
               color=CB["prediction"], edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(1 / nb, color=CB["accent"], lw=1.0, ls="--", zorder=4)
        ax.set_xlim(0, nb); ax.set_ylim(bottom=0)
        ax.set_title(vd["label"], fontsize=8, pad=3)
        ax.set_xlabel("Rank", fontsize=7)
        if i % 4 == 0:
            ax.set_ylabel("Rel. frequency", fontsize=7)
        panel_label(ax, chr(97 + i), x=-0.20, y=1.10)

    for j in range(len(ALL_VARS), axes.size):
        axes.ravel()[j].axis("off")

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=CB["accent"], lw=1, ls="--",
                                label="Uniform (ideal)")],
               loc="upper right", bbox_to_anchor=(0.99, 0.99))
    fig.text(0.01, 0.005, "tp/ssrd: rank histograms computed on real events only "
             "(truth>threshold) to remove zero-inflation/noise.",
             fontsize=5, color=CB["grey"], style="italic")
    save_fig(fig, f"{a.outdir}/plot_rank_histograms.pdf")
    plt.close(fig)
    eval_ds.close()
    print("done.")


if __name__ == "__main__":
    main()
