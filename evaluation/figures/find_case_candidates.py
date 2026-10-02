#!/usr/bin/env python3
"""
find_case_candidates.py — screens the eval dataset for storm and
frontal/gradient case-study candidates, and renders truth-only quick-look
panels for human review BEFORE plot_storm_case.py / figure_frontal_
case.py are run with fixed --idx/--idxs values.

Storm candidates: ranked by a combined intensity score (rank-normalized sum
of max wind speed + surface-pressure depression + max rain rate), all from
HRRR truth.

Frontal candidates, three categories (per request):
  (a) fully-ocean patch, high wind-speed std (the closest proxy available
      in this coastal-patch dataset to a coherent, intense, all-ocean
      vortex/typhoon-like feature)
  (b) any all-ocean patch (no further ranking -- just a representative case)
  (c) a mixed land/ocean (near-coast) patch where plot_frontal_case.py's
      fixed NW-SE diagonal transect crosses only ocean pixels

Land/ocean is NOT stored in the eval dataset itself (generate_eval_dataset.py
feeds the invariant lsm patch to the model but never writes it out) -- this
script re-derives it from the ORIGINAL regional_test.nc's native-res invariant
group via each eval sample's stored `sample_idx` -> `coord`. No GPU needed.

Usage:
    python find_case_candidates.py --eval-dir ../data/eval --outdir ../output/candidates
"""
import os
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import CB, DC, apply_style, open_eval, save_fig

TEST_NC = os.environ.get("CCD_DATA_DIR", ".") + "/regional_test.nc"
HR_SIZE = 256
OCEAN_THRESH = 0.5  # lsm_mean < 0.5 => ocean, matches collect_hrrr.py's convention
TRANSECT_P0 = (int(0.20 * HR_SIZE), int(0.20 * HR_SIZE))
TRANSECT_P1 = (int(0.80 * HR_SIZE), int(0.80 * HR_SIZE))


def load_ocean_fraction_and_mask(sample_indices):
    import netCDF4 as nc4
    ds_test = nc4.Dataset(TEST_NC, "r")
    lsm_full = np.array(ds_test.groups["invariant"].variables["lsm_mean"][:])
    coords = np.array(ds_test.variables["coord"][:])
    half = HR_SIZE // 2
    lsm_patches = np.empty((len(sample_indices), HR_SIZE, HR_SIZE), dtype=np.float32)
    for i, si in enumerate(sample_indices):
        row, col = coords[int(si)]
        lsm_patches[i] = lsm_full[row - half:row + half, col - half:col + half]
    ds_test.close()
    ocean_frac = (lsm_patches < OCEAN_THRESH).mean(axis=(1, 2))
    return ocean_frac, lsm_patches


def transect_ocean_fraction(lsm_patch, n=100):
    rr = np.linspace(TRANSECT_P0[0], TRANSECT_P1[0], n)
    cc = np.linspace(TRANSECT_P0[1], TRANSECT_P1[1], n)
    vals = map_coordinates(lsm_patch, np.vstack([rr, cc]), order=1, mode="nearest")
    return float((vals < OCEAN_THRESH).mean())


def rankscore(x):
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def quicklook_panel(ax, ws, sp, tp, lsm, title):
    im = ax.imshow(ws, origin="lower", cmap="YlOrRd", vmin=0, vmax=max(float(np.percentile(ws, 99)), 5))
    ax.contour(lsm, levels=[OCEAN_THRESH], colors="cyan", linewidths=0.8)
    ax.plot([TRANSECT_P0[1], TRANSECT_P1[1]], [TRANSECT_P0[0], TRANSECT_P1[0]], "k--", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.5)
    return im


def render_candidates(eval_ds, indices, scores_label, score_vals, ws, sp, tp, lsm_patches,
                       lat, lon, out_path, title):
    apply_style()
    n = len(indices)
    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False)
    for j, i in enumerate(indices):
        ax = axes[j // ncols, j % ncols]
        subtitle = (f"idx={i}\nlat={lat[i]:.1f} lon={lon[i]:.1f}\n"
                    f"{scores_label}={score_vals[i]:.3g}")
        quicklook_panel(ax, ws[i], sp[i], tp[i], lsm_patches[i], subtitle)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title, fontsize=10, fontweight="bold")
    save_fig(fig, out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output/candidates")
    p.add_argument("--n-storm", type=int, default=10)
    p.add_argument("--n-frontal", type=int, default=6)
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    eval_ds = open_eval(a.eval_dir)
    sample_idx_all = eval_ds.root("sample_idx").astype(int)
    lat = eval_ds.root("lat")
    lon = eval_ds.root("lon")
    print(f"Screening {eval_ds.n_total} eval samples...")

    ocean_frac, lsm_patches = load_ocean_fraction_and_mask(sample_idx_all)

    ws = eval_ds.wind_speed("target", None)   # (N,H,W)
    sp = eval_ds.get("target", "sp", None)
    tp = eval_ds.get("target", "tp", None)
    n = ws.shape[0]

    max_ws = ws.reshape(n, -1).max(axis=1)
    sp_depression = sp.reshape(n, -1).mean(axis=1) - sp.reshape(n, -1).min(axis=1)
    max_tp = tp.reshape(n, -1).max(axis=1)
    ws_std = ws.reshape(n, -1).std(axis=1)

    # ── Storm candidates ────────────────────────────────────────────────
    storm_score = rankscore(max_ws) + rankscore(sp_depression) + rankscore(max_tp)
    top_storm = np.argsort(-storm_score)[:a.n_storm]
    print("\nTop storm candidates (idx, lat, lon, max_wind[m/s], sp_depression[Pa], max_tp[mm], ocean_frac):")
    for i in top_storm:
        print(f"  {i:4d}  {lat[i]:7.2f} {lon[i]:8.2f}  {max_ws[i]:6.1f}  "
              f"{sp_depression[i]:7.1f}  {max_tp[i]:6.1f}  {ocean_frac[i]:.2f}")
    render_candidates(eval_ds, top_storm, "score", storm_score, ws, sp, tp, lsm_patches, lat, lon,
                       f"{a.outdir}/storm_candidates.pdf",
                       "Storm candidates (ranked by wind + pressure depression + rain) -- HRRR truth wind speed, cyan=coastline, dashed=transect")

    # ── Frontal candidates ──────────────────────────────────────────────
    all_ocean = ocean_frac > 0.98
    near_coast = (ocean_frac > 0.3) & (ocean_frac < 0.9)

    pool_a = np.where(all_ocean)[0]
    cand_a = pool_a[np.argsort(-ws_std[pool_a])[:a.n_frontal]]
    print(f"\n(a) Fully-ocean, high wind-speed-std candidates (pool size {len(pool_a)}):")
    for i in cand_a:
        print(f"  {i:4d}  {lat[i]:7.2f} {lon[i]:8.2f}  ws_std={ws_std[i]:.2f}  ocean_frac={ocean_frac[i]:.3f}")
    render_candidates(eval_ds, cand_a, "ws_std", ws_std, ws, sp, tp, lsm_patches, lat, lon,
                       f"{a.outdir}/frontal_candidates_a_ocean_high_std.pdf",
                       "(a) Fully-ocean, high wind-speed-std candidates")

    med = np.median(ws_std[pool_a])
    cand_b = pool_a[np.argsort(np.abs(ws_std[pool_a] - med))[:a.n_frontal]]
    print(f"\n(b) Any all-ocean candidates (near-median wind-speed-std, pool size {len(pool_a)}):")
    for i in cand_b:
        print(f"  {i:4d}  {lat[i]:7.2f} {lon[i]:8.2f}  ws_std={ws_std[i]:.2f}  ocean_frac={ocean_frac[i]:.3f}")
    render_candidates(eval_ds, cand_b, "ws_std", ws_std, ws, sp, tp, lsm_patches, lat, lon,
                       f"{a.outdir}/frontal_candidates_b_ocean_any.pdf",
                       "(b) Any all-ocean candidates (near-median wind-speed-std)")

    pool_c = np.where(near_coast)[0]
    transect_frac = np.full(n, np.nan)
    for i in pool_c:
        transect_frac[i] = transect_ocean_fraction(lsm_patches[i])
    cand_c_pool = pool_c[transect_frac[pool_c] > 0.98]
    cand_c = cand_c_pool[np.argsort(-ws_std[cand_c_pool])[:a.n_frontal]] if len(cand_c_pool) else np.array([], dtype=int)
    print(f"\n(c) Near-coast, transect-fully-over-ocean candidates (pool size {len(pool_c)}, "
          f"qualifying {len(cand_c_pool)}):")
    for i in cand_c:
        print(f"  {i:4d}  {lat[i]:7.2f} {lon[i]:8.2f}  ws_std={ws_std[i]:.2f}  "
              f"ocean_frac={ocean_frac[i]:.3f}  transect_ocean_frac={transect_frac[i]:.3f}")
    if len(cand_c):
        render_candidates(eval_ds, cand_c, "ws_std", ws_std, ws, sp, tp, lsm_patches, lat, lon,
                           f"{a.outdir}/frontal_candidates_c_near_coast.pdf",
                           "(c) Near-coast, transect-over-ocean-only candidates")
    else:
        print("  WARNING: no near-coast candidate found with a fully-ocean transect in this eval subset.")

    eval_ds.close()
    print(f"\nCandidate review PDFs written to {a.outdir}/")


if __name__ == "__main__":
    main()
