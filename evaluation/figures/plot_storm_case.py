#!/usr/bin/env python3
"""
plot_storm_case.py — storm case study.
Rows: 10-m wind speed | Rainfall rate
Cols: Low-res HRRR | UNet Regression | CorrDiff (mean) |
      CorrDiff (member) | CorrDiff (std) | HRRR truth

Storm case(s) auto-selected as high-wind-speed quantile samples unless
--idx/--idxs is given explicitly.

Run from this directory:
    python plot_storm_case.py --eval-dir ../data/eval --outdir ../output
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import CB, CMAP, DC, FS_ANNOT, apply_style, neat_ticks, open_eval, save_fig


def pick_storm_index(eval_ds, quantile=97, seed=3):
    idx = eval_ds.random_indices(min(800, eval_ds.n_total), seed=seed)
    ws = eval_ds.wind_speed("target", idx).reshape(len(idx), -1).max(axis=1)
    target = np.percentile(ws, quantile)
    return int(idx[np.argmin(np.abs(ws - target))])


def make_figure(eval_ds, storm_idx, out_path, suffix=""):
    apply_style()
    COLS = [
        ("Low-res HRRR",   "low_res_input_upsampled", CB["low_res"]),
        ("UNet Regression", "regression",              CB["regression"]),
        ("CorrDiff (mean)", "prediction_mean",   CB["prediction"]),
        ("CorrDiff (member)", "prediction_member", CB["member"]),
        ("CorrDiff (std)", "prediction_std",      CB["std"]),
        ("HRRR truth",       "target",                 CB["truth"]),
    ]

    def load(group, key, wind):
        if group == "prediction_mean":
            arr = eval_ds.wind_speed("prediction", [storm_idx]) if wind else eval_ds.get("prediction", key, [storm_idx])
            return np.nanmean(arr[0], axis=0)
        if group == "prediction_member":
            arr = eval_ds.wind_speed("prediction", [storm_idx]) if wind else eval_ds.get("prediction", key, [storm_idx])
            rng = np.random.default_rng(storm_idx)
            return arr[0, rng.integers(0, arr.shape[1])]
        if group == "prediction_std":
            arr = eval_ds.wind_speed("prediction", [storm_idx]) if wind else eval_ds.get("prediction", key, [storm_idx])
            return np.nanstd(arr[0], axis=0)
        arr = eval_ds.wind_speed(group, [storm_idx]) if wind else eval_ds.get(group, key, [storm_idx])
        return arr[0]

    rows = [("10-m Wind Speed", None, "YlOrRd", r"m s$^{-1}$", True),
            ("Rainfall Rate", "tp", "cividis", "mm", False)]

    N_C = len(COLS)
    fig = plt.figure(figsize=(DC * 1.15, 4.0), constrained_layout=False)
    gs = gridspec.GridSpec(2, N_C, figure=fig, hspace=0.10, wspace=0.06,
                            left=0.07, right=0.90, top=0.90, bottom=0.05)
    for ri, (rt, key, cmap, unit, wind) in enumerate(rows):
        tf = load("target", key, wind)
        log = not wind
        if log:
            norm = mcolors.SymLogNorm(linthresh=0.1, linscale=0.5, vmin=0,
                                       vmax=max(float(np.percentile(tf, 99)), 1))
        else:
            norm = mcolors.Normalize(0, float(np.percentile(tf, 99)))
        std_f = load("prediction_std", key, wind)
        norm_s = mcolors.Normalize(0, max(float(std_f.max()), 1e-6))
        field_im = None
        for ci, (cl, group, border) in enumerate(COLS):
            ax = fig.add_subplot(gs[ri, ci])
            is_std = "std" in group
            f = load(group, key, wind)
            cmap_use = CMAP["std"] if is_std else cmap
            norm_use = norm_s if is_std else norm
            im = ax.imshow(f, origin="lower", cmap=cmap_use, norm=norm_use,
                            interpolation="nearest", aspect="equal")
            if not is_std:
                field_im = im
            ax.set_xticks([]); ax.set_yticks([])
            lw = 1.6 if cl == "HRRR truth" else 0.6
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(lw)
            if is_std:
                ax.text(0.96, 0.05, f"$\\bar\\sigma$={f.mean():.2g}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=FS_ANNOT - 0.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.55, lw=0))
            elif cl != "HRRR truth":
                ax.text(0.96, 0.05, f"RMSE {float(np.sqrt(((f - tf) ** 2).mean())):.2g}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=FS_ANNOT - 0.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.55, lw=0))
            if ri == 0:
                ax.set_title(cl, fontsize=7, pad=4, color=border,
                              fontweight="bold" if cl == "HRRR truth" else "normal")
            if ci == 0:
                ax.set_ylabel(rt, fontsize=8, labelpad=4)

        vmin_cb, vmax_cb = field_im.get_clim()
        sm_field = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin_cb, vmax_cb) if not log else
            mcolors.SymLogNorm(linthresh=0.1, linscale=0.5, vmin=0, vmax=vmax_cb))
        sm_field.set_array([])
        cax = fig.add_axes([0.915, 0.55 - ri * 0.45, 0.013, 0.33])
        cb = fig.colorbar(sm_field, cax=cax)
        cb.set_label(unit, fontsize=6.5); cb.ax.tick_params(labelsize=6)
        if log:
            cb.set_ticks([0, 0.1, 1, 5, 10]); cb.set_ticklabels(["0", "0.1", "1", "5", "10"])
        else:
            tks = neat_ticks(vmin_cb, vmax_cb, n=5)
            cb.set_ticks(tks); cb.set_ticklabels([f"{t:g}" for t in tks])
        cax2 = fig.add_axes([0.96, 0.55 - ri * 0.45, 0.013, 0.33])
        sm2 = plt.cm.ScalarMappable(cmap=CMAP["std"], norm=norm_s)
        cb2 = fig.colorbar(sm2, cax=cax2); cb2.set_label("$\\sigma$", fontsize=6.5)
        cb2.ax.tick_params(labelsize=6)
        tks2 = neat_ticks(0, max(float(std_f.max()), 1e-6), n=5)
        cb2.set_ticks(tks2); cb2.set_ticklabels([f"{t:g}" for t in tks2])

    fig.suptitle(f"Storm case study (test-set sample {storm_idx})", fontsize=9,
                  fontweight="bold", y=0.96)
    save_fig(fig, out_path)
    plt.close(fig)
    print("done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--idx", type=int, default=None)
    p.add_argument("--idxs", type=int, nargs="*", default=None)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    eval_ds = open_eval(a.eval_dir)
    if a.idxs:
        import string
        for letter, idx in zip(string.ascii_lowercase, a.idxs):
            make_figure(eval_ds, idx, f"{a.outdir}/figure_07{letter}_storm_case.pdf", suffix=letter)
    else:
        idx = a.idx if a.idx is not None else pick_storm_index(eval_ds)
        make_figure(eval_ds, idx, f"{a.outdir}/plot_storm_case.pdf")
    eval_ds.close()


if __name__ == "__main__":
    main()
