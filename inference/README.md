# inference

`infer_region.py` downscales any lat/lon box at any time from a coarse driver:

* **HRRR input** (`--input_source hrrr`): fetches HRRR for the time and coarsens it exactly like the training data (Gaussian blur + xESMF conservative regridding);
  HRRR at 3 km also serves as the verification target.
* **ERA5 input** (`--input_source era5`): fetches ERA5 from the public ARCO store and regrids it bilinearly onto each patch's 32 x 32 grid (ERA5 is already ~25 km, so no coarsening
  is needed). The target group is still HRRR where HRRR exists.
* `--input_source both` runs both and writes `_era5`-suffixed groups next to the HRRR ones.

The box is tiled into 256 x 256 patches (HRRR-native grid) with `--overlap` pixels of linear feathering, each patch goes through the regression UNet and the diffusion ensemble
(`common/ccd_model.py`), and the patches are stitched into one map. Patches with any missing input value, and everything outside the HRRR domain, are NaN.

```bash
python inference/infer_region.py \
    --timestamp "2025-07-15 00:00" \
    --lat_min 25 --lat_max 31 --lon_min -98 --lon_max -88 \
    --reg_ckpt_dir $CCD_CKPT_DIR/regression --res_ckpt_dir $CCD_CKPT_DIR/diffusion \
    --stats_path $CCD_DATA_DIR/stats.json \
    --input_source both --num_ensembles 8 --overlap 3 --output out.nc
```

Useful options: `--mode regression` (UNet only, fast preview), `--sampler stochastic|deterministic`, `--sampler_steps`, `--seed_batch_size` (ensemble members per diffusion pass),
`--rain_filter_mm` (default 0.02, negative disables), `--reg_ckpt` / `--res_ckpt` (explicit files instead of the latest in a directory), `--device cpu`.
The stats file must be the `stats.json` the checkpoints were trained with.

## Output (NetCDF)

| Group | Content |
|---|---|
| `target` | HRRR truth, 8 channels, native resolution (`tp` in log10(1+mm)) |
| `input`, `input_era5` | the coarse 34-channel conditioning field the model actually saw, on the stitched 32 x 32-block grid |
| `unet`, `unet_era5` | regression (deterministic) prediction |
| `corrdiff`, `corrdiff_era5` | ensemble prediction (members, 8 channels, y, x), rain-filtered `tp` |

## Post-processing of the output

* **Rain filter** (`--rain_filter_mm`, default **0.02 mm**, negative disables). The diffusion residual restores small-scale rain texture but perturbs light/no-rain areas.
  `common/rain_filter.py` keeps the UNet's `tp` wherever the UNet predicts at most the threshold (`blended = where(unet_tp > threshold, corrdiff, unet)`), per ensemble member.
* **Physical constraints** (`--no_physical_constraints` disables), `common/physical_constraints.py`: `ssrd`, `strd` and `tp` are clipped at zero, and `ssrd` is set to zero wherever
  the sun is at or below the horizon (cos of the solar zenith angle <= 0, computed from latitude, longitude and time). `strd` is not zeroed at night. NaN is preserved.
  Both steps are applied to the UNet and the ensemble outputs, for HRRR- and ERA5-driven runs.

## Notes

* Needs PhysicsNeMo and, practically, a GPU (about 1 s per patch and ensemble member on a modern GPU; minutes on a CPU). On CPU-only machines PhysicsNeMo's distributed manager cannot start and is skipped automatically.
* HRRR and ERA5 are downloaded live from public buckets; there are no credentials.
* ERA5 differs physically from HRRR (clouds, radiation), so ERA5-driven output is a different experiment from HRRR-driven output, not a bug; compare the two with `--input_source both`.
