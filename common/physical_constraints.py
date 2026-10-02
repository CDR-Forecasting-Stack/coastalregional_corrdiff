"""Physical-consistency constraints applied to the high-resolution model output (regression and ensemble)
in physical units, just before it is written:

- ``ssrd`` (shortwave), ``strd`` (longwave) and ``tp`` (precipitation) are never negative.
- ``ssrd`` is zero wherever the sun is at or below the horizon (cos of the solar zenith angle <= 0).
  ``strd`` is *not* zeroed at night: downwelling longwave radiation is emitted by the atmosphere day and night.

NaN (outside the requested region / missing data) is left as NaN. Precipitation may be given either in mm or in
the model's log10(1 + mm) representation: both are non-negative exactly when the rain is, so the same clip applies.
"""
from __future__ import annotations

import numpy as np

from channels import OUT_CH

SSRD_IDX = OUT_CH.index("ssrd")
STRD_IDX = OUT_CH.index("strd")
TP_IDX = OUT_CH.index("tp")


def apply_physical_constraints(field: np.ndarray, cos_sza: np.ndarray) -> np.ndarray:
    """field: (..., C, H, W) with channels in OUT_CH order; cos_sza: (H, W) cosine of the solar zenith angle
    per pixel (``collect_hrrr.compute_cos_sza``). Returns a new array; the input is not modified."""
    out = field.copy()
    for idx in (SSRD_IDX, STRD_IDX, TP_IDX):
        ch = out[..., idx, :, :]
        out[..., idx, :, :] = np.where(ch < 0.0, 0.0, ch)          # NaN < 0 is False -> NaN preserved
    night = cos_sza <= 0.0
    ssrd = out[..., SSRD_IDX, :, :]
    out[..., SSRD_IDX, :, :] = np.where(night & ~np.isnan(ssrd), 0.0, ssrd)
    return out
