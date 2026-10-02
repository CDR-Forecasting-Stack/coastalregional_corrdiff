#!/usr/bin/env python3
"""
plot_ensemble_sensitivity.py — CRPS
sensitivity to ensemble size, computed directly from the real 16-member
ensemble already stored in the evaluation dataset (the single evaluation
dataset carries the full ensemble for every sample).

Method: for each target size N in {1,2,4,8,16}, draw n_subsets random
N-member subsets (without replacement) from the 16 stored members per
sample, compute fair-CRPS for each subset, and report mean +/- std across
subsets and samples (N=16 uses all members directly, no subsampling).

Run from this directory:
    python plot_ensemble_sensitivity.py --eval-dir ../data/eval --outdir ../output
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
from metrics import fair_crps

ENSEMBLE_SIZES = [1, 2, 4, 8, 16]
N_SAMP = 200
N_SUBSETS = 20
PIXEL_STRIDE = 4


def crps_vs_n(truth_flat, ens_flat, n_subsets=N_SUBSETS, seed=0):
    rng = np.random.default_rng(seed)
    E = ens_flat.shape[0]
    out = {}
    for n in ENSEMBLE_SIZES:
        if n > E:
            continue
        if n == E:
            out[n] = [float(np.mean(fair_crps(truth_flat, ens_flat)))]
            continue
        vals = []
        for _ in range(n_subsets):
            members = rng.choice(E, size=n, replace=False)
            vals.append(float(np.mean(fair_crps(truth_flat, ens_flat[members]))))
        out[n] = vals
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-samp", type=int, default=N_SAMP)
    p.add_argument("--n-subsets", type=int, default=N_SUBSETS)
    p.add_argument("--pixel-stride", type=int, default=PIXEL_STRIDE)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_samp, seed=0)

    targets = [("T2m", "2t", False, r"T2m CRPS (K)"),
               ("u10", "10u", True, r"u10 CRPS (m s$^{-1}$)"),
               ("Rain", "tp", False, r"Rain CRPS (mm)")]

    results = {}
    for label, key, wind, _ in targets:
        print(f"  [S5] {label}...")
        truth = eval_ds.wind_speed("target", idx) if wind else eval_ds.get("target", key, idx)
        ens = eval_ds.wind_speed("prediction", idx) if wind else eval_ds.get("prediction", key, idx)
        truth = truth[:, ::a.pixel_stride, ::a.pixel_stride].reshape(-1)
        ens = ens[:, :, ::a.pixel_stride, ::a.pixel_stride].transpose(1, 0, 2, 3).reshape(ens.shape[1], -1)
        per_size = crps_vs_n(truth, ens, n_subsets=a.n_subsets, seed=1)
        results[label] = {n: {"mean": float(np.mean(v)), "std": float(np.std(v))} for n, v in per_size.items()}
        print(f"  [S5] {label}: " + ", ".join(f"N={n}:{v['mean']:.4f}" for n, v in sorted(results[label].items())))

    fig, axes = plt.subplots(1, 3, figsize=(DC * 0.95, 2.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18, wspace=0.35)
    for pi, (label, key, wind, ylabel) in enumerate(targets):
        ax = axes[pi]
        d = results[label]
        ns = sorted(d.keys())
        means = [d[n]["mean"] for n in ns]
        stds = [d[n]["std"] for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", ms=4, lw=1.3,
                    color=CB["prediction"], capsize=3, ecolor=CB["grey"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ENSEMBLE_SIZES); ax.set_xticklabels([str(n) for n in ENSEMBLE_SIZES])
        ax.set_xlabel("Ensemble size N", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(label, fontsize=9, pad=4)
        ax.tick_params(labelsize=7)
        panel_label(ax, chr(97 + pi))

    fig.suptitle("CRPS Sensitivity to Ensemble Size (real 16-member subsampling)",
                  fontsize=10, fontweight="bold", y=0.99)
    save_fig(fig, f"{a.outdir}/plot_ensemble_sensitivity.pdf")
    plt.close(fig)
    eval_ds.close()
    print("done.")


if __name__ == "__main__":
    main()
