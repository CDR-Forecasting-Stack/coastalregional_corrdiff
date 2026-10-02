#!/usr/bin/env python3
"""
tables.py — skill tables for the coastal regional CorrDiff model (8 variables), computed directly
from the evaluation shards (runs in seconds).

  deterministic : RMSE, MAE -- Low-res HRRR | UNet Regression | CorrDiff
  probabilistic : CRPS, CRPS improvement, spread-skill ratio (SSR)
  precipitation : detailed precipitation metrics (CSI, ETS, frequency bias, ...)

Run from this directory:
    python tables.py --eval-dir ../data/eval --outdir ../output
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import ALL_VARS, open_eval
from metrics import csi_ets_freqbias, fair_crps, mae, rmse

N_SUB = 1000


def bold_best(vals, fmt=".4f", lower=True):
    num = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not num:
        return ["--"] * len(vals)
    bi = (min if lower else max)(num, key=lambda x: x[1])[0]
    return [("--" if v is None else (f"\\textbf{{{v:{fmt}}}}" if i == bi else f"{v:{fmt}}"))
            for i, v in enumerate(vals)]


def write_latex(rows, headers, caption, label, path):
    cf = "l" + "r" * (len(headers) - 1)
    L = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{cf}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    L += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(L)); print(f"  LaTeX -> {path}")


def write_csv(rows, headers, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    print(f"  CSV   -> {path}")


def per_var_stats(eval_ds, idx, key):
    truth = eval_ds.get("target", key, idx)
    low_res = eval_ds.get("low_res_input_upsampled", key, idx)
    regr = eval_ds.get("regression", key, idx)
    ens = eval_ds.get("prediction", key, idx)
    pred_mean = ens.mean(axis=1)

    lr_rmse, lr_mae = rmse(low_res, truth), mae(low_res, truth)
    rg_rmse, rg_mae = rmse(regr, truth), mae(regr, truth)
    pd_rmse, pd_mae = rmse(pred_mean, truth), mae(pred_mean, truth)
    crps = float(fair_crps(truth, ens, ens_axis=1).mean())
    spread = float(ens.std(axis=1).mean())
    ssr = spread / pd_rmse if pd_rmse > 0 else None
    return dict(lr_rmse=lr_rmse, lr_mae=lr_mae, rg_rmse=rg_rmse, rg_mae=rg_mae,
                pd_rmse=pd_rmse, pd_mae=pd_mae, crps=crps, spread=spread, ssr=ssr)


def table_1(eval_ds, idx, outdir):
    print("Deterministic table...")
    headers = ["Variable", "Unit", "Low-res RMSE", "Regr. RMSE", "Pred. RMSE",
               "Low-res MAE", "Regr. MAE", "Pred. MAE"]
    rl, rc = [], []
    for vd in ALL_VARS:
        k = vd["key"]; print(f"  {k}...")
        s = per_var_stats(eval_ds, idx, k)
        rmse_f = bold_best([s["lr_rmse"], s["rg_rmse"], s["pd_rmse"]])
        mae_f = bold_best([s["lr_mae"], s["rg_mae"], s["pd_mae"]])
        unit = vd["unit"].replace("$", "").replace("^", "").replace("{", "").replace("}", "")
        rl.append([vd["label"], unit] + rmse_f + mae_f)
        rc.append([vd["label"], unit, s["lr_rmse"], s["rg_rmse"], s["pd_rmse"],
                    s["lr_mae"], s["rg_mae"], s["pd_mae"]])
    cap = (f"Deterministic skill (RMSE, MAE) on {len(idx)} independent held-out test samples. "
           "Best per metric in bold.")
    write_latex(rl, headers, cap, "tab:det", f"{outdir}/table_deterministic.tex")
    write_csv(rc, [h.replace("\\", "") for h in headers], f"{outdir}/table_deterministic.csv")


def table_2(eval_ds, idx, outdir):
    print("Probabilistic table...")
    headers = ["Variable", "Unit", "Low-res CRPS", "Regr. CRPS", "Pred. CRPS", "Improv. (%)", "SSR"]
    rl, rc = [], []
    for vd in ALL_VARS:
        k = vd["key"]; print(f"  {k}...")
        s = per_var_stats(eval_ds, idx, k)
        imp = 100 * (s["lr_mae"] - s["crps"]) / s["lr_mae"] if s["lr_mae"] else None
        crps_f = bold_best([s["lr_mae"], s["rg_mae"], s["crps"]])
        unit = vd["unit"].replace("$", "").replace("^", "").replace("{", "").replace("}", "")
        rl.append([vd["label"], unit] + crps_f +
                  [f"{imp:.1f}" if imp is not None else "--", f"{s['ssr']:.3f}" if s["ssr"] else "--"])
        rc.append([vd["label"], unit, s["lr_mae"], s["rg_mae"], s["crps"], imp, s["ssr"]])
    cap = ("Probabilistic skill. CRPS reduces to MAE for the deterministic low-res-input and "
           "UNet Regression baselines; CorrDiff's CRPS uses the real 16-member "
           "ensemble (fair-CRPS estimator). Improvement is CorrDiff vs Low-res HRRR. "
           "SSR = spread/RMSE ($\\approx1$ ideal). Best CRPS in bold.")
    write_latex(rl, headers, cap, "tab:prob", f"{outdir}/table_probabilistic.tex")
    write_csv(rc, [h.replace("\\", "") for h in headers], f"{outdir}/table_probabilistic.csv")


def table_s1(eval_ds, idx, outdir):
    print("Precipitation table...")
    truth = eval_ds.get("target", "tp", idx)
    pred = eval_ds.get("prediction", "tp", idx).mean(axis=1)
    headers = ["Metric", "0.1 mm", "1.0 mm", "5.0 mm"]
    tk = [0.1, 1.0, 5.0]
    csi_v, ets_v, fb_v = zip(*[csi_ets_freqbias(pred, truth, t) for t in tk])
    rows = [
        ["CSI"] + [f"{v:.3f}" for v in csi_v],
        ["ETS"] + [f"{v:.3f}" for v in ets_v],
        ["Freq. bias"] + [f"{v:.3f}" for v in fb_v],
        ["Wet frac (pred)", f"{float((pred > 0.1).mean()):.3f}", "", ""],
        ["Wet frac (truth)", f"{float((truth > 0.1).mean()):.3f}", "", ""],
        ["MAE (wet px)", f"{mae(pred[truth > 0.1], truth[truth > 0.1]):.4f}", "", ""],
        ["RMSE (wet px)", f"{rmse(pred[truth > 0.1], truth[truth > 0.1]):.4f}", "", ""],
    ]
    cap = (f"Detailed precipitation verification metrics for CorrDiff (tp) on "
           f"{len(idx)} test samples. CSI = Critical Success Index; ETS = Equitable Threat Score.")
    write_latex(rows, headers, cap, "tab:s1_precip", f"{outdir}/table_precip.tex")
    write_csv(rows, headers, f"{outdir}/table_precip.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-sub", type=int, default=N_SUB)
    p.add_argument("--skip", nargs="*", default=[])
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    sk = set(a.skip)

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_sub, seed=42)

    if "t1" not in sk: table_1(eval_ds, idx, a.outdir)
    if "t2" not in sk: table_2(eval_ds, idx, a.outdir)
    if "ts1" not in sk: table_s1(eval_ds, idx, a.outdir)
    eval_ds.close()
    print(f"Tables saved to {a.outdir}/")


if __name__ == "__main__":
    main()
