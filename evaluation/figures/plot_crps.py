#!/usr/bin/env python3
"""
plot_crps.py — normalized CRPS across all 8 variables, plus
% improvement of CorrDiff over the Low-res HRRR baseline.

Low-res HRRR and UNet Regression are deterministic (CRPS reduces to MAE);
CorrDiff's CRPS uses the real 16-member ensemble stored in the
eval dataset (fair-CRPS estimator, no synthetic/approximated ensemble).

Run from this directory:
    python plot_crps.py --eval-dir ../data/eval --outdir ../output
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
from metrics import fair_crps, mae

N_SUB = 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-sub", type=int, default=N_SUB)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_sub, seed=42)

    labels, low_res, regr, pred, imps = [], [], [], [], []
    for vd in ALL_VARS:
        k = vd["key"]
        print(f"  Fig2 {k}...")
        truth = eval_ds.get("target", k, idx)
        lr = eval_ds.get("low_res_input_upsampled", k, idx)
        rg = eval_ds.get("regression", k, idx)
        ens = eval_ds.get("prediction", k, idx)  # (N,E,H,W)

        truth_std = float(truth.std())
        truth_std = truth_std if truth_std > 0 else 1.0

        lr_mae = mae(lr, truth) / truth_std
        rg_mae = mae(rg, truth) / truth_std
        pred_crps = float(fair_crps(truth, ens, ens_axis=1).mean()) / truth_std

        labels.append(vd["label"])
        low_res.append(lr_mae); regr.append(rg_mae); pred.append(pred_crps)
        imps.append(100 * (lr_mae - pred_crps) / lr_mae if lr_mae > 0 else 0)

    n = len(labels); x = np.arange(n); bw = 0.26
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(DC, 4.6),
                                    gridspec_kw={"height_ratios": [3, 1.7], "hspace": 0.42})

    for oi, (vals, col, lbl) in enumerate([
        (low_res, CB["low_res"], "Low-res HRRR"),
        (regr, CB["regression"], "UNet Regression"),
        (pred, CB["prediction"], "CorrDiff"),
    ]):
        axa.bar(x + [-bw, 0, bw][oi], vals, width=bw, color=col, edgecolor="white",
                linewidth=0.4, label=lbl, zorder=3)
    axa.set_xticks(x); axa.set_xticklabels(labels)
    axa.set_ylabel("Normalized CRPS\n(CRPS / $\\sigma_{truth}$)")
    axa.set_xlim(-0.55, n - 0.45); axa.set_ylim(bottom=0)
    axa.legend(loc="upper right", ncol=3)
    panel_label(axa, "a")

    cols = [CB["prediction"] if v > 0 else CB["accent"] for v in imps]
    axb.bar(x, imps, width=0.6, color=cols, zorder=3)
    axb.axhline(0, color="#333", lw=0.6)
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set_ylabel("CRPS improvement\nvs Low-res HRRR (%)")
    axb.set_xlim(-0.55, n - 0.45)
    panel_label(axb, "b")

    save_fig(fig, f"{a.outdir}/plot_crps.pdf")
    plt.close(fig)
    eval_ds.close()
    print("done.")


if __name__ == "__main__":
    main()
