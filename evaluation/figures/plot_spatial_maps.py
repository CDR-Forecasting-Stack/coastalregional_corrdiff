#!/usr/bin/env python3
"""
plot_spatial_maps.py — spatial comparison, one PDF per variable (8).

Columns: Low-res HRRR | UNet Regression | CorrDiff (mean) |
         CorrDiff (member) | CorrDiff (std) | HRRR truth
Rows:    storm / moderate / calm case (quantile-selected on truth's spatial mean)

Run from this directory:
    python plot_spatial_maps.py --eval-dir ../data/eval --outdir ../output
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
from fig_style import ALL_VARS, CB, CMAP, DC, FS_ANNOT, apply_style, neat_ticks, open_eval, panel_label


def pick_cases(target_means, quantiles):
    out = []
    labels = {20: "Low (Q20)", 50: "Median (Q50)", 90: "High (Q90)",
              65: "Q65", 80: "Q80", 95: "High (Q95)"}
    for q in quantiles:
        target = np.percentile(target_means, q)
        idx = int(np.argmin(np.abs(target_means - target)))
        out.append((labels.get(q, f"Q{q}"), idx))
    return out


def rmse(p, t):
    return float(np.sqrt(np.nanmean((p - t) ** 2)))


def stamp(ax, txt):
    ax.text(0.96, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_ANNOT - 0.5, color="white",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.55, lw=0))


def make_figure(var, eval_ds, out_path, n_pool=None):
    apply_style()
    key, unit, cmap_field, log = var["key"], var["unit"], var["cmap"], var["log"]
    # Wind SPEED is used only to pick which cases (storm/moderate/calm) to
    # show for the 10u/10v panels -- the displayed field is always the
    # actual signed component (see load() below), never |wind|.
    select_by_wind_speed = key in ("10u", "10v")

    pool_idx = np.arange(eval_ds.n_total) if n_pool is None else eval_ds.random_indices(n_pool, seed=1)
    truth_pool = eval_ds.wind_speed("target", pool_idx) if select_by_wind_speed else eval_ds.get("target", key, pool_idx)
    means = truth_pool.reshape(truth_pool.shape[0], -1).mean(axis=1)

    q_sel = (95, 80, 65) if key in ("ssrd", "strd", "tp") else (90, 50, 20)
    local_cases = pick_cases(means, q_sel)
    cases = [(lbl, int(pool_idx[i])) for lbl, i in local_cases]

    def load(group, idx, member=None):
        # NOTE: always fetch the actual named scalar field (key), never wind
        # SPEED, here -- `wind` above is only for case-selection pooling.
        # 10u and 10v must show their own signed component, not |wind|.
        if group == "prediction_mean":
            arr = eval_ds.get("prediction", key, [idx])
            return np.nanmean(arr[0], axis=0)
        if group == "prediction_member":
            arr = eval_ds.get("prediction", key, [idx])
            rng = np.random.default_rng(idx)
            return arr[0, rng.integers(0, arr.shape[1])]
        if group == "prediction_std":
            arr = eval_ds.get("prediction", key, [idx])
            return np.nanstd(arr[0], axis=0)
        arr = eval_ds.get(group, key, [idx])
        return arr[0]

    COLS = [
        ("Low-res HRRR",              lambda i: load("low_res_input_upsampled", i), CB["low_res"],    False),
        ("UNet Regression",            lambda i: load("regression", i),              CB["regression"], False),
        ("CorrDiff (mean)", lambda i: load("prediction_mean", i),         CB["prediction"], False),
        ("CorrDiff (member)", lambda i: load("prediction_member", i),     CB["member"],     False),
        ("HRRR truth",                 lambda i: load("target", i),                  CB["truth"],      False),
        ("CorrDiff (std)",  lambda i: load("prediction_std", i),          CB["std"],        True),
    ]

    truth_fields = [load("target", ci) for _, ci in cases]
    allt = np.concatenate([f.ravel() for f in truth_fields])

    if log:
        vmin, vmax = 0, max(float(np.nanpercentile(allt, 99)), 1.0)
        norm_field = mcolors.SymLogNorm(linthresh=0.1, linscale=0.5, vmin=0, vmax=vmax)
    elif cmap_field == "RdBu_r":
        vlim = float(np.nanpercentile(np.abs(allt), 98))
        vmin, vmax = -vlim, vlim
        norm_field = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        p2, p98 = np.nanpercentile(allt, 2), np.nanpercentile(allt, 98)
        c = np.nanmedian(allt); h = max(abs(p98 - c), abs(c - p2))
        vmin, vmax = c - h, c + h
        norm_field = mcolors.Normalize(vmin=vmin, vmax=vmax)

    std_fields = [load("prediction_std", ci) for _, ci in cases]
    std_max = float(np.nanpercentile(np.concatenate([f.ravel() for f in std_fields]), 95))
    norm_std = mcolors.Normalize(0, max(std_max, 1e-6))

    N_COL, N_ROW = len(COLS), 3
    fig = plt.figure(figsize=(DC * 1.15, 4.5), constrained_layout=False)
    gs = gridspec.GridSpec(N_ROW, N_COL, figure=fig, hspace=0.10, wspace=0.06,
                            left=0.075, right=0.97, top=0.90, bottom=0.22)

    field_ims = []
    for ri, (case_lbl, ci) in enumerate(cases):
        tf = truth_fields[ri]
        for mi, (col_lbl, loader, border, is_std) in enumerate(COLS):
            ax = fig.add_subplot(gs[ri, mi])
            field = loader(ci)
            cmap = CMAP["std"] if is_std else cmap_field
            norm = norm_std if is_std else norm_field
            im = ax.imshow(field, origin="lower", cmap=cmap, norm=norm,
                            interpolation="nearest", aspect="equal")
            if not is_std:
                field_ims.append(im)
            ax.set_xticks([]); ax.set_yticks([])
            lw = 1.6 if col_lbl == "HRRR truth" else 0.6
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(lw)
            if is_std:
                stamp(ax, f"$\\bar\\sigma$={field.mean():.2g}")
            elif col_lbl != "HRRR truth":
                stamp(ax, f"RMSE {rmse(field, tf):.2g}")
            if ri == 0:
                ax.set_title(col_lbl, fontsize=7, pad=4, color=border,
                              fontweight="bold" if col_lbl == "HRRR truth" else "normal")
            if mi == 0:
                ax.set_ylabel(case_lbl, fontsize=8, labelpad=4)

    cax1 = fig.add_axes([0.075, 0.12, 0.55, 0.028])
    cb1 = fig.colorbar(field_ims[-1], cax=cax1, orientation="horizontal")
    cb1.set_label(unit, fontsize=7, labelpad=3)
    cb1.ax.tick_params(labelsize=6, direction="in")
    if log:
        cb1.set_ticks([0, 0.1, 1, 5, 10]); cb1.set_ticklabels(["0", "0.1", "1", "5", "10"])
    else:
        tks = neat_ticks(vmin, vmax, n=6)
        cb1.set_ticks(tks); cb1.set_ticklabels([f"{t:g}" for t in tks])

    cax2 = fig.add_axes([0.68, 0.12, 0.27, 0.028])
    sm = plt.cm.ScalarMappable(cmap=CMAP["std"], norm=norm_std)
    cb2 = fig.colorbar(sm, cax=cax2, orientation="horizontal")
    cb2.set_label(f"$\\sigma$ ({unit})", fontsize=7, labelpad=3)
    cb2.ax.tick_params(labelsize=6, direction="in")
    tks2 = neat_ticks(0, max(std_max, 1e-6), n=5)
    cb2.set_ticks(tks2); cb2.set_ticklabels([f"{t:g}" for t in tks2])

    fig.suptitle(var["long"], fontsize=9, fontweight="bold", y=0.965)
    save_fig_local(fig, out_path)
    plt.close(fig)


def save_fig_local(fig, path):
    from fig_style import save_fig
    save_fig(fig, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    eval_ds = open_eval(a.eval_dir)
    for v in ALL_VARS:
        print(f"  Fig1 {v['key']}...")
        make_figure(v, eval_ds, f"{a.outdir}/figure_01_{v['key']}.pdf")
    eval_ds.close()
    print(f"done -> {a.outdir}/")


if __name__ == "__main__":
    main()
