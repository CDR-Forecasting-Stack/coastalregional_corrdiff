#!/usr/bin/env python3
"""
plot_training_data.py — training data
coverage (STANDALONE -- reads regional_train.nc directly, not the eval dataset).

Top panel: map of ALL training patch centers (real lat/lon, read directly
from the root group -- no index-lookup/grid-mismatch risk).
Bottom panels: Low-res HRRR vs HRRR truth distributions (per-sample
spatial means) for all 8 output variables.

Usage:
    python plot_training_data.py --train /path/to/data/regional_train.nc \
        --n-hist-samples 3000 --outdir ../output
"""
import os
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import ALL_VARS, CB, DC, apply_style, panel_label, save_fig

ZERO_TRIM_VARS = {"tp", "ssrd"}
DEFAULT_TRAIN_NC = os.environ.get("CCD_DATA_DIR", ".") + "/regional_train.nc"


def _us_coastline_lines():
    try:
        import geopandas as gpd
        import geodatasets
        path = geodatasets.get_path("naturalearth.land")
        gdf = gpd.read_file(path).cx[-140:-55, 15:60]
        boundary = gdf.boundary
        lines = []
        for geom in boundary:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                lines.append((np.array(xs), np.array(ys)))
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    xs, ys = part.xy
                    lines.append((np.array(xs), np.array(ys)))
        if not lines:
            raise RuntimeError("no geometry returned")
        print(f"  [S9] coastline: geopandas/Natural Earth ({len(lines)} segments)")
        return lines
    except Exception as e:
        print(f"  [S9] geopandas/Natural Earth unavailable ({e}); using bundled coarse outline")
        coarse = [
            ([-117.1, -117.3, -118.2, -119.7, -120.9, -121.8, -122.4, -122.8, -123.8, -124.2, -124.1],
             [32.5, 33.0, 34.0, 34.4, 34.5, 36.6, 37.2, 38.0, 39.4, 40.8, 46.2]),
            ([-97.1, -95.3, -93.8, -91.9, -89.9, -88.0, -85.0, -83.0, -82.6, -81.8],
             [25.9, 28.8, 29.7, 29.2, 29.1, 30.2, 29.7, 29.8, 27.9, 24.6]),
            ([-80.1, -78.9, -77.9, -76.0, -75.5, -74.0, -71.0, -70.2, -67.0],
             [25.8, 33.8, 34.2, 36.9, 38.9, 40.6, 41.4, 43.7, 44.8]),
            ([-124.7, -117.0, -104.0, -95.2, -82.4], [49.0, 49.0, 49.0, 49.0, 42.2]),
            ([-117.1, -108.2, -106.5, -104.9, -97.1], [32.5, 31.3, 31.8, 29.7, 25.9]),
        ]
        return [(np.array(x), np.array(y)) for x, y in coarse]


def _draw_basemap(ax):
    for xs, ys in _us_coastline_lines():
        ax.plot(xs, ys, color="#888888", lw=0.6, zorder=1)
    ax.set_xlim(-140, -55); ax.set_ylim(18, 56)
    ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
    ax.set_aspect("equal")


def load_all_patch_centers(nc_path):
    import netCDF4 as nc4
    ds = nc4.Dataset(nc_path, "r")
    lats = np.array(ds.variables["lat"][:], dtype=np.float64)
    lons = np.array(ds.variables["lon"][:], dtype=np.float64)
    n = len(lats)
    ds.close()
    return lats, lons, n


def load_histogram_subset(train_path, n_samples, seed=42):
    import netCDF4 as nc4
    ds = nc4.Dataset(train_path, "r")
    n_total = ds.dimensions["sample"].size
    rng = np.random.default_rng(seed)
    sl = np.sort(rng.choice(n_total, min(n_samples, n_total), replace=False)) if n_samples < n_total \
        else np.arange(n_total)

    low_res_means, hrrr_means = {}, {}
    t0 = time.time()
    for vd in ALL_VARS:
        k = vd["key"]
        in_key = vd["input_match"]
        if k in ds.groups["output"].variables:
            arr = np.array(ds.groups["output"].variables[k][sl])
            hrrr_means[k] = arr.reshape(arr.shape[0], -1).mean(axis=1)
        if in_key in ds.groups["input"].variables:
            arr = np.array(ds.groups["input"].variables[in_key][sl])
            low_res_means[k] = arr.reshape(arr.shape[0], -1).mean(axis=1)
        print(f"  [S9] histogram data for {k} (input key={in_key}): {time.time()-t0:.1f}s elapsed")
    ds.close()
    return low_res_means, hrrr_means, len(sl)


def _hist_panel(ax, vd, low_res_vals, hrrr_vals):
    k = vd["key"]
    e = np.asarray(low_res_vals, dtype=float) if low_res_vals is not None else None
    h = np.asarray(hrrr_vals, dtype=float) if hrrr_vals is not None else None
    if e is not None:
        e = e[np.isfinite(e)]
    if h is not None:
        h = h[np.isfinite(h)]
    if k in ZERO_TRIM_VARS:
        if e is not None: e = e[e > 1e-6]
        if h is not None: h = h[h > 1e-6]

    log = vd.get("log", False) and k in ZERO_TRIM_VARS
    if log:
        lo = max(min(h.min() if h is not None and h.size else 1e-3, 1e-3), 1e-4)
        hi = max(h.max() if h is not None and h.size else 1.0, 1.0)
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.set_xscale("log")
    else:
        all_v = np.concatenate([v for v in (e, h) if v is not None and v.size])
        if all_v.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            return
        lo, hi = np.nanpercentile(all_v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)

    if e is not None and e.size:
        ax.hist(e, bins=bins, density=True, histtype="stepfilled",
                 color=CB["low_res"], alpha=0.35, label="Low-res HRRR", zorder=2)
    if h is not None and h.size:
        ax.hist(h, bins=bins, density=True, histtype="step", color=CB["truth"], lw=1.3,
                 label="HRRR truth", zorder=3)

    ax.set_xlabel(f"{vd['label']} ({vd['unit']})", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)
    ax.set_title(vd["long"], fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    if k in ZERO_TRIM_VARS:
        ax.text(0.97, 0.93, "zeros removed", transform=ax.transAxes, ha="right", va="top",
                 fontsize=5.5, style="italic", color="#666")


def make_figure(train_path, n_hist_samples, outdir):
    apply_style()
    t0 = time.time()
    lats, lons, n_total = load_all_patch_centers(train_path)
    print(f"  [S9] map: loaded all {n_total} patch centers in {time.time()-t0:.2f}s")

    t1 = time.time()
    low_res_data, hrrr_data, n_used = load_histogram_subset(train_path, n_hist_samples)
    print(f"  [S9] histograms: read {n_used} sample subset in {time.time()-t1:.1f}s")

    fig = plt.figure(figsize=(DC, DC * 1.15), constrained_layout=False)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.3, 1, 1], hspace=0.55, wspace=0.35,
                            left=0.08, right=0.97, top=0.91, bottom=0.05)

    ax_map = fig.add_subplot(gs[0, :])
    _draw_basemap(ax_map)
    ax_map.scatter(lons, lats, s=1.0, alpha=0.15, color=CB["prediction"], edgecolors="none",
                    rasterized=True, zorder=2)
    ax_map.set_title(f"Training patch centers (N={n_total:,})", fontsize=9, pad=10)
    panel_label(ax_map, "a")

    for pi, vd in enumerate(ALL_VARS):
        row = 1 + pi // 4
        col = pi % 4
        ax = fig.add_subplot(gs[row, col])
        _hist_panel(ax, vd, low_res_data.get(vd["key"]), hrrr_data.get(vd["key"]))
        panel_label(ax, chr(98 + pi))

    handles, labels = [], []
    for ax in fig.axes[1:]:
        h_, l_ = ax.get_legend_handles_labels()
        for hh, ll in zip(h_, l_):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.93),
                    fontsize=7, framealpha=0.9)

    fig.suptitle(f"Training Dataset: Geographic Coverage (N={n_total:,}) and "
                 f"Per-Sample Spatial-Mean Variable Distributions (N={n_used:,})",
                 fontsize=9.5, fontweight="bold", y=0.995)
    save_fig(fig, f"{outdir}/plot_training_data.pdf")
    plt.close(fig)
    print("done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=DEFAULT_TRAIN_NC)
    p.add_argument("--n-hist-samples", type=int, default=3000)
    p.add_argument("--outdir", default="../output")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    make_figure(a.train, a.n_hist_samples, a.outdir)


if __name__ == "__main__":
    main()
