"""fig_style.py — shared publication style + variable metadata for the coastal regional
CorrDiff evaluation figures (single unified model, 8 output variables).

Terminology (used consistently across every figure in this repo):
  "Low-res HRRR"    — the model's coarse conditioning field (HRRR
                       Gaussian-blurred and conservatively regridded to ~24 km, standing in for
                       what an ERA5-like coarse driver would see -- not real ERA5), or its
                       naive bilinear-upsampled baseline for comparison.
  "UNet Regression" — the deterministic first stage (pre-diffusion).
  "CorrDiff"        — the final model output (regression + diffusion
                       residual), mean or a single ensemble member.
  "HRRR truth"      — ground truth.

Variable metadata (channel names, units, input matching) comes from common/channels.py.
"""
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "common"))
from channels import OUT_CH, UNITS, INPUT_MATCH  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_io import EvalDataset  # noqa: E402

DEFAULT_EVAL_DIR = os.environ.get("CCD_EVAL_DIR", "./eval_data")

# ── Page geometry (single / double column, publication style) ──────────────
SC, DC, DPI = 3.5, 7.2, 600
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT, FS_PANEL, FS_LEGEND = 9, 8, 7, 6.5, 9, 7

# ── Colorblind-safe palette (Wong/Tol), one colour per role ────────────────
CB = {
    "low_res":     "#0072B2",
    "regression":  "#E69F00",
    "prediction":  "#009E73",
    "member":      "#56B4E9",
    "truth":       "#000000",
    "std":         "#CC79A7",
    "accent":      "#D55E00",
    "grey":        "#999999",
    "grey_lt":     "#CCCCCC",
}

CMAP = {
    "temp": "viridis", "wind": "cividis", "wind_div": "RdBu_r", "pressure": "viridis",
    "humidity": "viridis", "precip": "cividis", "radiation": "viridis",
    "std": "magma", "diff": "RdBu_r", "corr": "RdBu_r", "density": "viridis",
}

# ── Variable metadata: one flat list, 8 vars, no grouped-model split ───────
_META = {
    "2t":   {"label": "T2m",             "long": "2-m Temperature",        "cmap": "viridis", "log": False},
    "10u":  {"label": "u10",             "long": "10-m Zonal Wind",        "cmap": "RdBu_r",  "log": False},
    "10v":  {"label": "v10",             "long": "10-m Meridional Wind",   "cmap": "RdBu_r",  "log": False},
    "sp":   {"label": "$p_s$",           "long": "Surface Pressure",       "cmap": "viridis", "log": False},
    "q":    {"label": "q",               "long": "Specific Humidity",      "cmap": "viridis", "log": False},
    "tp":   {"label": "Rain",            "long": "Precipitation",          "cmap": "cividis", "log": True},
    "ssrd": {"label": r"SW$\downarrow$", "long": "Shortwave Radiation",    "cmap": "viridis", "log": False},
    "strd": {"label": r"LW$\downarrow$", "long": "Longwave Radiation",     "cmap": "viridis", "log": False},
}
_UNIT_TEX = {"m/s": r"m s$^{-1}$", "kg/kg": r"kg kg$^{-1}$", "W/m2": r"W m$^{-2}$"}

ALL_VARS = [
    {"key": k, "unit": _UNIT_TEX.get(UNITS[k], UNITS[k]), "input_match": INPUT_MATCH[k], **_META[k]}
    for k in OUT_CH
]
VAR_BY_KEY = {v["key"]: v for v in ALL_VARS}


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": FS_LABEL, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "figure.titlesize": FS_TITLE,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "figure.dpi": 150, "savefig.dpi": DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": CB["grey_lt"], "legend.handlelength": 1.5,
        "legend.borderpad": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
    })


def panel_label(ax, letter, x=-0.16, y=1.02, **kw):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left", **kw)


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    png = str(path).rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"  Saved -> {path}  (+ .png preview)")


def neat_ticks(vmin, vmax, n=5):
    """n evenly-spaced 'neat' tick values (1,2,2.5,5 x 10^k steps) between
    vmin and vmax, so colorbars show e.g. 0,5,10,15,20 not 1.24,7.83,..."""
    if vmax <= vmin:
        return np.array([vmin, vmax])
    raw_step = (vmax - vmin) / max(n - 1, 1)
    mag = 10.0 ** np.floor(np.log10(raw_step))
    norm = raw_step / mag
    for mult in [1, 2, 2.5, 5, 10]:
        if mult >= norm:
            step = mult * mag
            break
    else:
        step = 10 * mag
    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + step * 0.01, step)
    ticks = ticks[(ticks >= vmin - step * 0.01) & (ticks <= vmax + step * 0.01)]
    return np.unique(np.round(ticks / step) * step)


def open_eval(eval_dir: str | None = None) -> EvalDataset:
    return EvalDataset(eval_dir or DEFAULT_EVAL_DIR)
