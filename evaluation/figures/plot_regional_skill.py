#!/usr/bin/env python3
"""
plot_regional_skill.py — regional skill maps.
Top row: continuous per-sample RMSE scatter map (lon/lat, colored by RMSE).
Bottom row: RMSE summarized into 4 coastal regions (simple lon/lat split --
Pacific vs Atlantic/Gulf, north vs south -- since regional_test.nc stores real
per-sample lat/lon directly, no external basin-lookup file is needed).

Run from this directory:
    python plot_regional_skill.py --eval-dir ../data/eval --outdir ../output
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import CB, CMAP, DC, apply_style, open_eval, panel_label, save_fig

LON_SPLIT = -105.0  # west of this = Pacific-facing, east = Atlantic/Gulf-facing
LAT_SPLIT = 35.0


def classify_basin(lat, lon):
    pacific = lon < LON_SPLIT
    north = lat >= LAT_SPLIT
    out = np.empty(lat.shape, dtype=object)
    out[pacific & north] = "North Pacific"
    out[pacific & ~north] = "South Pacific"
    out[~pacific & north] = "North Atlantic"
    out[~pacific & ~north] = "South Atlantic/Gulf"
    return out


def per_sample_rmse(eval_ds, key, idx, wind=False):
    truth = eval_ds.wind_speed("target", idx) if wind else eval_ds.get("target", key, idx)
    pred = eval_ds.wind_speed("prediction", idx).mean(axis=1) if wind else \
        eval_ds.get("prediction", key, idx).mean(axis=1)
    return np.sqrt(((pred - truth) ** 2).mean(axis=(1, 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--n-samp", type=int, default=1000)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    apply_style()

    eval_ds = open_eval(a.eval_dir)
    idx = eval_ds.random_indices(a.n_samp, seed=7)
    lat = eval_ds.root("lat", idx)
    lon = eval_ds.root("lon", idx)
    basin = classify_basin(lat, lon)

    fig, axes = plt.subplots(2, 2, figsize=(DC, 6.4), gridspec_kw={"hspace": 0.55, "wspace": 0.55})
    panels = [("T2m", "2t", False, CMAP["temp"], "K"),
              ("Rain (tp)", "tp", False, CMAP["precip"], "mm")]

    for col, (label, key, wind, cmap, unit) in enumerate(panels):
        print(f"  S3 {label}...")
        err = per_sample_rmse(eval_ds, key, idx, wind=wind)

        axm = axes[0, col]
        sc = axm.scatter(lon, lat, c=err, s=6, cmap=cmap, alpha=0.75, linewidths=0,
                          vmin=np.percentile(err, 2), vmax=np.percentile(err, 98))
        cb = fig.colorbar(sc, ax=axm, fraction=0.05, pad=0.03)
        cb.set_label(f"RMSE ({unit})", fontsize=6.5); cb.ax.tick_params(labelsize=6)
        axm.set_xlabel("Longitude", fontsize=7); axm.set_ylabel("Latitude", fontsize=7)
        axm.set_title(f"({chr(97 + col)}) {label} — sample RMSE map", fontsize=8, pad=4)
        axm.tick_params(labelsize=6)
        panel_label(axm, chr(97 + col))

        axb = axes[1, col]
        basins_order = ["North Pacific", "South Pacific", "North Atlantic", "South Atlantic/Gulf"]
        meds, lo, hi, ns = [], [], [], []
        for b in basins_order:
            m = basin == b
            q25, q50, q75 = np.percentile(err[m], [25, 50, 75]) if m.sum() else (0, 0, 0)
            meds.append(q50); lo.append(q50 - q25); hi.append(q75 - q50)
            ns.append(int(m.sum()))
        x = np.arange(len(basins_order))
        axb.bar(x, meds, yerr=[lo, hi], capsize=3,
                color=CB["low_res"] if col == 0 else CB["regression"], edgecolor="white", linewidth=0.4)
        axb.set_xticks(x)
        axb.set_xticklabels([b.replace(" ", "\n", 1).replace("/", "/\n") + f"\n(n={n_b})" for b, n_b in zip(basins_order, ns)], fontsize=6)
        axb.set_ylabel(f"Median RMSE ({unit})", fontsize=7)
        axb.set_title(f"({chr(99 + col)}) {label} — RMSE by region", fontsize=8, pad=4)
        panel_label(axb, chr(99 + col))

    fig.suptitle("Regional skill maps", fontsize=9, y=1.0)
    save_fig(fig, f"{a.outdir}/plot_regional_skill.pdf")
    plt.close(fig)
    eval_ds.close()
    print("S3 done.")


if __name__ == "__main__":
    main()
