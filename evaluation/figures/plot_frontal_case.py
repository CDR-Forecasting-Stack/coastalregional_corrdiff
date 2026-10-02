#!/usr/bin/env python3
"""
plot_frontal_case.py — cross-section case study.

Rows (3): 2-m Temperature | Along-front wind | Across-front wind
          (wind rotated into components along / across a front oriented at
          --angle degrees, default 45)
Cols (5): Low-res HRRR | UNet Regression | CorrDiff (mean) |
          CorrDiff (member) | HRRR truth   + one cross-section profile plot
          per row.

Each row has its OWN cross-section (red line on every map panel), chosen where
the methods differ most so the differences are visible in the profile plot.
For samples listed in HAND_TRANSECTS the lines were drawn by hand; for any
other sample they are selected automatically (see auto_transect).

Run from this directory:
    python plot_frontal_case.py --eval-dir ../data/eval --outdir ../output --idxs 151
"""
import argparse
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import CB, DC, apply_style, open_eval, save_fig

PX_KM = 3.0
LINE_COLOR = "red"

# (row, col) endpoints in the 256x256 patch (origin lower-left, as plotted),
# one transect per figure row: [temperature, along-front wind, across-front wind]
HAND_TRANSECTS = {
    151: [((212, 10), (203, 82)),
          ((170, 113), (131, 223)),
          ((188, 125), (43, 187))],
}


def rotate_wind(u, v, angle_deg):
    a = np.deg2rad(angle_deg)
    along = u * np.cos(a) + v * np.sin(a)
    across = -u * np.sin(a) + v * np.cos(a)
    return along, across


def sample_line(field, p0, p1):
    n = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) + 1
    rr = np.linspace(p0[0], p1[0], n)
    cc = np.linspace(p0[1], p1[1], n)
    return map_coordinates(field, np.vstack([rr, cc]), order=1, mode="nearest")


def auto_transect(f_truth, f_lowres, f_reg, length=90, margin=6, step=12, angle_step=15):
    """Straight line of `length` px where low-res and regression-only deviate
    most from truth (RMS error normalised by the field's std)."""
    H, W = f_truth.shape
    sd = float(np.std(f_truth)) + 1e-9
    best, best_score = None, -1.0
    for ang in np.arange(0, 180, angle_step):
        dr, dc = np.sin(np.deg2rad(ang)) * length, np.cos(np.deg2rad(ang)) * length
        for r in np.arange(margin, H - margin, step):
            for c in np.arange(margin, W - margin, step):
                p0, p1 = (r, c), (r + dr, c + dc)
                if not (margin <= p1[0] <= H - margin and margin <= p1[1] <= W - margin):
                    continue
                t = sample_line(f_truth, p0, p1)
                e1 = np.sqrt(np.mean((sample_line(f_lowres, p0, p1) - t) ** 2))
                e2 = np.sqrt(np.mean((sample_line(f_reg, p0, p1) - t) ** 2))
                score = (e1 + e2) / sd
                if score > best_score:
                    best, best_score = (p0, p1), score
    return best


def pick_frontal_index(eval_ds, seed=3):
    idx = eval_ds.random_indices(min(500, eval_ds.n_total), seed=seed)
    t2m = eval_ds.get("target", "2t", idx)
    grad = np.sqrt(np.gradient(t2m, axis=1) ** 2 + np.gradient(t2m, axis=2) ** 2)
    score = grad.reshape(len(idx), -1).max(axis=1)
    return int(idx[np.argmax(score)])


def components(eval_ds, group, idx):
    if group == "prediction_mean":
        t = eval_ds.get("prediction", "2t", [idx])[0].mean(axis=0)
        u = eval_ds.get("prediction", "10u", [idx])[0].mean(axis=0)
        v = eval_ds.get("prediction", "10v", [idx])[0].mean(axis=0)
    elif group == "prediction_member":
        rng = np.random.default_rng(idx)
        e_t = eval_ds.get("prediction", "2t", [idx])[0]
        m = rng.integers(0, e_t.shape[0])
        t = e_t[m]
        u = eval_ds.get("prediction", "10u", [idx])[0][m]
        v = eval_ds.get("prediction", "10v", [idx])[0][m]
    else:
        t = eval_ds.get(group, "2t", [idx])[0]
        u = eval_ds.get(group, "10u", [idx])[0]
        v = eval_ds.get(group, "10v", [idx])[0]
    return t, u, v


def make_figure(eval_ds, idx, out_path, front_angle=45):
    apply_style()
    t_truth, u_truth, v_truth = components(eval_ds, "target", idx)
    t_reg, u_reg, v_reg = components(eval_ds, "regression", idx)
    t_lr, u_lr, v_lr = components(eval_ds, "low_res_input_upsampled", idx)
    t_mean, u_mean, v_mean = components(eval_ds, "prediction_mean", idx)
    t_mem, u_mem, v_mem = components(eval_ds, "prediction_member", idx)

    ens_t = eval_ds.get("prediction", "2t", [idx])[0]
    ens_u = eval_ds.get("prediction", "10u", [idx])[0]
    ens_v = eval_ds.get("prediction", "10v", [idx])[0]

    rot = {k: rotate_wind(u, v, front_angle) for k, (u, v) in {
        "truth": (u_truth, v_truth), "reg": (u_reg, v_reg), "lr": (u_lr, v_lr),
        "mean": (u_mean, v_mean), "mem": (u_mem, v_mem)}.items()}
    ens_rot = [rotate_wind(ens_u[k], ens_v[k], front_angle) for k in range(ens_u.shape[0])]

    H, W = t_truth.shape
    rows = [
        dict(title="2-m Temperature", unit="K", cmap="RdYlBu_r",
             f=dict(t=t_truth, c=t_mean, m=t_mem, l=t_lr, r=t_reg), ens=ens_t,
             w=dict(t=(u_truth, v_truth), c=(u_mean, v_mean), m=(u_mem, v_mem),
                    l=(u_lr, v_lr), r=(u_reg, v_reg))),
        dict(title="Along-front wind", unit=r"m s$^{-1}$", cmap="RdBu_r",
             f=dict(t=rot["truth"][0], c=rot["mean"][0], m=rot["mem"][0], l=rot["lr"][0], r=rot["reg"][0]),
             ens=np.stack([e[0] for e in ens_rot]),
             w={k: (rot[n][0], np.zeros_like(rot[n][0])) for k, n in
                dict(t="truth", c="mean", m="mem", l="lr", r="reg").items()}),
        dict(title="Across-front wind", unit=r"m s$^{-1}$", cmap="RdBu_r",
             f=dict(t=rot["truth"][1], c=rot["mean"][1], m=rot["mem"][1], l=rot["lr"][1], r=rot["reg"][1]),
             ens=np.stack([e[1] for e in ens_rot]),
             w={k: (np.zeros_like(rot[n][1]), rot[n][1]) for k, n in
                dict(t="truth", c="mean", m="mem", l="lr", r="reg").items()}),
    ]

    if idx in HAND_TRANSECTS:
        transects = HAND_TRANSECTS[idx]
    else:
        transects = [auto_transect(R["f"]["t"], R["f"]["l"], R["f"]["r"]) for R in rows]

    COL_LABELS = ["Low-res HRRR", "UNet Regression", "CorrDiff (mean)", "CorrDiff (member)", "HRRR truth"]
    col_keys = ["l", "r", "c", "m", "t"]
    border_colors = [CB["low_res"], CB["regression"], CB["prediction"], CB["member"], CB["truth"]]

    fig = plt.figure(figsize=(DC * 1.75, 6.6), constrained_layout=False)
    gs = gridspec.GridSpec(3, 8, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.06, 0.55, 1.25],
                            hspace=0.26, wspace=0.10, left=0.045, right=0.97, top=0.88, bottom=0.07)
    skip = 16
    yy, xx = np.mgrid[0:H:skip, 0:W:skip]

    print(f"\n[sample {idx}] per-row transect statistics (RMSE vs HRRR truth along the line)")
    for ri, (R, (p0, p1)) in enumerate(zip(rows, transects)):
        if R["cmap"] == "RdBu_r":
            vlim = float(np.nanpercentile(np.abs(R["f"]["t"]), 98))
            norm = mcolors.Normalize(-vlim, vlim)
        else:
            p2, p98 = np.nanpercentile(R["f"]["t"], [2, 98])
            norm = mcolors.Normalize(p2, p98)

        im = None
        for ci, (ck, border) in enumerate(zip(col_keys, border_colors)):
            ax = fig.add_subplot(gs[ri, ci])
            im = ax.imshow(R["f"][ck], origin="lower", cmap=R["cmap"], norm=norm,
                            interpolation="nearest", aspect="equal")
            au, av = R["w"][ck]
            ax.quiver(xx, yy, au[::skip, ::skip], av[::skip, ::skip],
                      scale=300, width=0.004, color="#222", alpha=0.7)
            ax.plot([p0[1], p1[1]], [p0[0], p1[0]], color=LINE_COLOR, lw=1.4, solid_capstyle="round")
            ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(border)
                sp.set_linewidth(1.4 if ck == "t" else 0.7)
            if ri == 0:
                ax.set_title(COL_LABELS[ci], fontsize=10, color=border,
                              fontweight="bold" if ck == "t" else "normal", pad=6)
            if ci == 0:
                ax.set_ylabel(R["title"], fontsize=10, labelpad=4)

        cax = fig.add_subplot(gs[ri, 5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(R["unit"], fontsize=8.5, labelpad=2)
        cb.ax.tick_params(labelsize=6, pad=1)

        axc = fig.add_subplot(gs[ri, 7])
        prof = {k: sample_line(R["f"][k], p0, p1) for k in col_keys}
        ens_prof = np.stack([sample_line(R["ens"][k], p0, p1) for k in range(R["ens"].shape[0])])
        prof_mean, prof_std = ens_prof.mean(0), ens_prof.std(0)
        dist = np.arange(len(prof["t"])) * PX_KM

        for name, k in [("Low-res HRRR", "l"), ("UNet Regression", "r"), ("CorrDiff (mean)", "c"), ("CorrDiff (member)", "m")]:
            e = np.sqrt(np.mean((prof[k] - prof["t"]) ** 2))
            print(f"   row {ri} ({R['title']}): {name:16s} RMSE {e:.3f}  max|diff| {np.max(np.abs(prof[k]-prof['t'])):.3f}")

        axc.plot(dist, prof["t"], color="black", lw=1.3, label="HRRR truth", zorder=1)
        axc.plot(dist, prof["l"], color=CB["low_res"], lw=1.1, ls=":", label="Low-res HRRR", zorder=2)
        axc.plot(dist, prof["r"], color=CB["regression"], lw=1.1, ls="--", label="UNet Regression", zorder=3)
        axc.plot(dist, prof["c"], color=CB["prediction"], lw=1.3, label="CorrDiff (mean)", zorder=4)
        axc.fill_between(dist, prof_mean - prof_std, prof_mean + prof_std,
                          color=CB["prediction"], alpha=0.25, lw=0, zorder=4)
        axc.plot(dist, prof["m"], color="red", lw=0.8, label="CorrDiff (member)", zorder=5)
        axc.set_ylabel(R["unit"], fontsize=9, labelpad=2)
        if ri == 2:
            axc.set_xlabel("Distance along transect (km)", fontsize=9)
        if ri == 0:
            axc.legend(fontsize=7.5, loc="best")
            axc.set_title("Cross section", fontsize=10, pad=4)
        axc.tick_params(labelsize=8)

    fig.suptitle(f"Cross-section case study (test-set sample {idx})", fontsize=12, fontweight="bold", y=0.995)
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"Frontal-case figure done (sample {idx}).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--outdir", default="../output")
    p.add_argument("--idx", type=int, default=None)
    p.add_argument("--idxs", type=int, nargs="*", default=None)
    p.add_argument("--angle", type=float, default=45, help="front orientation (deg) for the wind rotation")
    p.add_argument("--names", nargs="*", default=None,
                   help="output file stems, one per idx (default figure_frontal_a, _b, ...)")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    eval_ds = open_eval(a.eval_dir)
    if a.idxs:
        stems = a.names or [f"figure_frontal_{l}" for l in string.ascii_lowercase[:len(a.idxs)]]
        for stem, idx in zip(stems, a.idxs):
            make_figure(eval_ds, idx, f"{a.outdir}/{stem}.pdf", a.angle)
    else:
        idx = a.idx if a.idx is not None else pick_frontal_index(eval_ds)
        make_figure(eval_ds, idx, f"{a.outdir}/plot_frontal_case.pdf", a.angle)
    eval_ds.close()


if __name__ == "__main__":
    main()
