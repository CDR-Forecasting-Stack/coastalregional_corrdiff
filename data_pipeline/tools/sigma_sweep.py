#!/usr/bin/env python3
"""
sigma_sweep.py -- pick the Gaussian width for coarsen.py by comparing against ERA5.
================================================================================
For one HRRR timestamp (patches taken from regional_test.nc at that time) builds the 32x32 coarse input with
  * plain 8x8 block mean (no blur, as a baseline),
  * Gaussian blur (sigma in --sigmas) + xESMF conservative regridding (no quantile mapping),
and compares each with ERA5 on the same 32x32 grid.

Outputs (in --outdir):
  maps_<channel>.png     native HRRR | ERA5 | area | sigma... for one patch, plus (method - ERA5) row
  rapsd_<channel>.png    radially averaged PSD of the 32x32 fields (mean over patches) + ratio to ERA5
  metrics.csv            per channel/method: RMSE & bias vs ERA5, high-k power ratio to ERA5
Quantile mapping is not applied: it is a per-channel distribution shift fit once and is
separate from the blur width.   No GPU needed.
"""
import argparse, csv, sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # data_pipeline/
import collect_hrrr as collect
import coarsen
import era5_input as e5

CH = ["t2m", "u10", "v10", "tp_in", "ssrd", "strd", "sp", "q_sfc"]
HR = collect.HR_SIZE
F = 8


def rapsd(a):
    n = a.shape[-1]
    w = np.hanning(n)[:, None] * np.hanning(n)[None]
    x = (a - a.mean()) * w
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    y, xx = np.indices(p.shape); c = n // 2
    r = np.hypot(y - c, xx - c).astype(int)
    k = np.arange(1, n // 2)
    return k, np.array([p[r == i].mean() for i in k])


def nn_up(a):
    return np.repeat(np.repeat(a, F, -2), F, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", default="2024-08-03 18:00")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[3, 4, 5])
    ap.add_argument("--n-patches", type=int, default=20)
    ap.add_argument("--map-patch", type=int, default=0, help="index (into the patch list) shown in the map figures")
    ap.add_argument("--outdir", default="./sigma_sweep")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    dt = datetime.strptime(a.timestamp, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

    ds = nc.Dataset(a.dataset)
    t = np.array(ds["time"][:]); coords = np.array(ds["coord"][:])
    sel = np.where(t == int(dt.timestamp()))[0][: a.n_patches]
    if len(sel) == 0:
        raise SystemExit("no patches in dataset at that timestamp")
    coords = coords[sel]; ds.close()
    print(f"{len(sel)} patches at {dt}", flush=True)

    inv = collect.load_hrrr_inv(collect.HRRR_REF)
    sfc0, sfc1, prs0 = collect.fetch_hrrr_timestamp(dt)
    stack = collect.build_hrrr_input_stack(sfc0, prs0, inv["latitude"], inv["longitude"], dt)
    for k in ("tp", "prate"):
        if sfc1 is not None and k in sfc1:
            stack[collect.SFC_CH.index("tp_in")] = collect.precip_log(sfc1[k].astype(np.float32)); break
    store = e5.open_arco_store()

    methods = ["area"] + [f"sigma{s:g}" for s in a.sigmas]
    data = {m: [] for m in methods + ["era5", "native"]}
    for (r, c) in coords:
        sl = (slice(r - 128, r + 128), slice(c - 128, c + 128))
        patch = stack[(slice(None),) + sl].astype(np.float32)
        lat = inv["latitude"][sl].astype(np.float64); lon = inv["longitude"][sl].astype(np.float64)
        co = coarsen.ConservativeCoarsener.from_latlon(lat, lon, F)
        ll = collect.downsample_256_to_32(np.stack([lat, lon]).astype(np.float32))
        region = e5.Era5RawRegion(dt, store, lat.min() - .5, lat.max() + .5, lon.min() - .5, lon.max() + .5, pad=3)
        data["era5"].append(e5.build_era5_input_stack(region, ll[0], ll[1]))
        data["area"].append(collect.downsample_256_to_32(patch))
        for s in a.sigmas:
            data[f"sigma{s:g}"].append(coarsen.coarsen_patch(patch, co, s))
        data["native"].append(patch)
    data = {k: np.stack(v) for k, v in data.items()}          # (P, 34, ...)
    np.savez_compressed(out / "sweep_data.npz", **data, coords=coords)
    chi = [collect.IN_CH.index(c) for c in CH]
    colors = {"era5": "k", "area": "#E69F00", **dict(zip(methods[1:], ["#0072B2", "#009E73", "#CC79A7", "#D55E00"]))}

    rows = []
    for name, ci in zip(CH, chi):
        # ---- maps
        p = min(a.map_patch, len(sel) - 1)
        cols = [("HRRR native 3 km", data["native"][p, ci]), ("ERA5", nn_up(data["era5"][p, ci])),
                ("8x8 area avg", nn_up(data["area"][p, ci]))] + \
               [(f"Gauss σ={s:g}px + xESMF", nn_up(data[f"sigma{s:g}"][p, ci])) for s in a.sigmas]
        vals = np.concatenate([v.ravel() for _, v in cols]); lo, hi = np.nanpercentile(vals, [1, 99])
        fig, ax = plt.subplots(2, len(cols), figsize=(3.1 * len(cols), 6.4), gridspec_kw={"wspace": .05, "hspace": .12})
        era = nn_up(data["era5"][p, ci]); res = []
        for j, (ttl, arr) in enumerate(cols):
            im = ax[0, j].imshow(arr, vmin=lo, vmax=hi, cmap="viridis", interpolation="nearest"); ax[0, j].set_title(ttl, fontsize=8)
            if j >= 2:
                d = arr - era; res.append(np.nanpercentile(np.abs(d), 99))
        m = max(res)
        for j, (ttl, arr) in enumerate(cols):
            if j < 2: ax[1, j].axis("off"); continue
            dm = ax[1, j].imshow(arr - era, vmin=-m, vmax=m, cmap="RdBu_r", interpolation="nearest")
            ax[1, j].set_title(f"{ttl} − ERA5", fontsize=8)
        for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
        fig.colorbar(im, ax=ax[0], fraction=.012, pad=.01); fig.colorbar(dm, ax=ax[1], fraction=.012, pad=.01)
        la, lo_ = inv["latitude"][coords[p][0], coords[p][1]], inv["longitude"][coords[p][0], coords[p][1]]
        fig.suptitle(f"{name}  {dt:%Y-%m-%d %H:%M} UTC  patch centre lat {la:.1f} lon {lo_:.1f}", fontsize=10)
        fig.savefig(out / f"maps_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        # ---- RAPSD on the 32x32 fields
        spec = {}
        for m_ in ["era5"] + methods:
            ps = []
            for x in data[m_][:, ci]:
                if np.isfinite(x).all():
                    k, pk = rapsd(x); ps.append(pk)
            spec[m_] = np.mean(ps, 0)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for m_ in ["era5"] + methods:
            ax[0].loglog(k, spec[m_], color=colors[m_], lw=2 if m_ == "era5" else 1.4, label=m_)
            ax[1].semilogx(k, spec[m_] / spec["era5"], color=colors[m_], lw=1.4, label=m_)
        ax[0].set_ylabel("PSD"); ax[1].set_ylabel("PSD / ERA5 PSD"); ax[1].axhline(1, c="gray", ls=":")
        for x in ax: x.set_xlabel("wavenumber (cycles / 32 px)"); x.legend(fontsize=8); x.grid(alpha=.3, which="both")
        fig.suptitle(f"RAPSD of 32x32 coarse input — {name} ({len(data['era5'])} patches)", fontsize=10)
        fig.savefig(out / f"rapsd_{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        # ---- metrics vs ERA5
        for m_ in methods:
            d = data[m_][:, ci] - data["era5"][:, ci]
            hk = k >= 8
            rows.append(dict(channel=name, method=m_, rmse_vs_era5=float(np.sqrt(np.nanmean(d ** 2))),
                             bias_vs_era5=float(np.nanmean(d)), highk_power_ratio=float(spec[m_][hk].sum() / spec["era5"][hk].sum())))
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("done ->", out)


if __name__ == "__main__":
    main()
