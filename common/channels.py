"""Channel names, units and the input<->output variable matching used by evaluation and plotting.

Channel order is defined once, in data_pipeline/collect_hrrr.py (IN_CH: 34 coarse input channels,
OUT_CH: 8 target channels); this module only adds metadata. Evaluation files store precipitation in mm
(the model works in log10(1+mm))."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
import collect_hrrr as _c  # noqa: E402

IN_CH = list(_c.IN_CH)
OUT_CH = list(_c.OUT_CH)
INVARIANT_CH = ["elev_mean", "lsm_mean"]

UNITS = {
    "u10": "m/s", "v10": "m/s", "t2m": "K", "sp": "Pa", "msl": "Pa", "d2m": "K",
    "q_sfc": "kg/kg", "tcwv": "kg/m2", "ssrd": "W/m2", "strd": "W/m2",
    "tp_in": "mm", "tcc": "0-1", "lsm": "0-1 (land fraction)", "cos_sza": "1",
    "2t": "K", "10u": "m/s", "10v": "m/s", "tp": "mm", "q": "kg/kg",
}
for _v, _u in {"u": "m/s", "v": "m/s", "z": "m", "t": "K", "q": "kg/kg"}.items():
    for _lev in (1000, 850, 500, 250):
        UNITS[f"{_v}{_lev}"] = _u

# which coarse input channel corresponds to each target variable
INPUT_MATCH = {"2t": "t2m", "10u": "u10", "10v": "v10", "tp": "tp_in",
               "ssrd": "ssrd", "strd": "strd", "sp": "sp", "q": "q_sfc"}
