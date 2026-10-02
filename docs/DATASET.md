# Dataset

The training and test sets are built entirely from HRRR (no ERA5 enters the dataset). Each sample pairs a
**coarse 32 x 32 input** (34 channels, ~24 km) with the **native 256 x 256 HRRR target** (8 channels, 3 km) of the
same 768 km window and the same HRRR analysis time.

`data_pipeline/collect_hrrr.py` builds them; the numbers below describe what its default configuration produces.

## Contents

| | train | test |
|---|---|---|
| Period | 2018-01-01 .. 2023-12-31 | 2024-01-01 .. 2025-12-31 |
| Timestamps | 1,200 (200 per year) | 500 (250 per year) |
| Patches per timestamp | 100 | 20 |
| Samples | 120,000 | 10,000 |
| File | `regional_train.nc` (~65 GB, compressed) | `regional_test.nc` (~5.8 GB) |

Plus `stats.json` (per-channel mean / std of the **training** file, used to normalize the model input and output).

## Patch selection

1. **Window:** 256 x 256 native pixels (~768 km), fully inside the HRRR grid.
2. **Ocean rule:** the centre pixel is ocean (HRRR land-sea mask `lsm` < 0.5) **and** at least **70 %** of the window is ocean
   (`OCEAN_MIN_FRAC`). Land pixels inside the window are kept in input and target.
3. **Geographic balance:** the valid centres (153,171 at 70 %) are split into four longitude bands
   (-135..-115, -115..-95, -95..-80, -80..-60 deg). Each timestamp takes an equal number of centres per band, at random,
   without repeating a centre inside a timestamp (centres may repeat across timestamps; windows may overlap).
4. **Timestamps:** hourly, drawn uniformly at random with a fixed quota per year and, within a year, per month
   (the remainder goes to randomly chosen months). All 24 hours and all seasons occur. A random reserve per year-month
   replaces any timestamp whose HRRR files cannot be downloaded, so the final counts are exact.
5. **Reproducibility:** `--seed` (default 42) fixes the timestamp draw; every timestamp gets its own derived seed for its patch draw.

## Coarse input construction

For each 256 x 256 native patch, `data_pipeline/coarsen.py`:

1. **Gaussian blur** with sigma = 5 native pixels (15 km); NaN-aware, reflect padding at the patch edge;
2. **xESMF conservative regridding** onto the patch's own 32 x 32 coarse cells (8 x 8 native pixels each, ~24 km);

**No quantile mapping to ERA5 is applied.** HRRR-versus-ERA5 differences, for example in shortwave radiation (HRRR's explicit
microphysics gives optically thicker clouds than ERA5's parameterised clouds; aerosol treatment also differs), are physical, and
forcing the distributions to agree would inject real bias.

**Choice of sigma** (`data_pipeline/tools/sigma_sweep.py`, 20 ocean test patches at one summer timestamp, 32 x 32 fields compared with ERA5
on the same grid). High-wavenumber power relative to ERA5 (1 = ERA5):

| channel | no blur (8 x 8 block mean) | sigma 4 | **sigma 5** | sigma 6 |
|---|---|---|---|---|
| t2m | 5.0 | 1.9 | **1.2** | 0.7 |
| strd | 7.9 | 2.7 | **1.7** | 0.9 |
| sp | 6.0 | 2.1 | **1.3** | 0.75 |
| q_sfc | 2.0 | 0.75 | 0.46 | 0.27 |

Without the blur, a block mean aliases small-scale structure that a ~25 km reanalysis does not contain; sigma = 5 comes closest to ERA5's spectrum
on most channels without over-smoothing the temperature and pressure fields. RMSE against ERA5 decreases monotonically with sigma on every
channel, so the spectra, not the RMSE, decide the choice. The sweep used a single timestamp; re-run the tool before changing sigma.

## NetCDF schema

```
regional_train.nc / regional_test.nc
├── Dimensions: sample, y_lr=32, x_lr=32, y_hr=256, x_hr=256, y_grid=1059, x_grid=1799, coord=2
├── Root variables
│   time   (int64)    Unix seconds, HRRR f00 analysis time
│   coord  (uint16)   [row, col] of the patch centre in the 1059 x 1799 HRRR grid
│   lat, lon (float32) patch-centre latitude / longitude
├── Group input      34 channels x (sample, 32, 32) float32
├── Group output      8 channels x (sample, 256, 256) float32
└── Group invariant  latitude, longitude, elev_mean, lsm_mean x (1059, 1799)
Global attribute `coarsening` records how the 32 x 32 input was built.
```

## Input channels (34) -- coarsened HRRR

HRRR sources: `wrfsfcf00` (surface), `wrfsfcf01` (precipitation), `wrfprsf00` (pressure levels), from the public
`gs://high-resolution-rapid-refresh` archive. All channels have ERA5 counterparts, which is how the same model is driven from ERA5 at inference
(`data_pipeline/era5_input.py`).

**Surface (13)**

| Channel | HRRR variable | Transform | Unit | ERA5 equivalent |
|---|---|---|---|---|
| `u10`, `v10` | u10, v10 | none | m/s | 10m u/v wind |
| `t2m` | t2m | none | K | 2m temperature |
| `sp` | sp | none | Pa | surface pressure |
| `msl` | mslma | none | Pa | mean sea level pressure |
| `d2m` | d2m | none | K | 2m dewpoint |
| `q_sfc` | derived from d2m, sp | Magnus formula | kg/kg | derived the same way |
| `tcwv` | pwat | none | kg/m2 | total column water vapour |
| `ssrd` | sdswrf | none | W/m2 | ssrd / 3600 (accumulated J/m2 -> W/m2) |
| `strd` | sdlwrf | none | W/m2 | strd / 3600 |
| `tp_in` | tp (f01) | log10(1 + mm) | log10(1+mm/h) | total precipitation x 1000, then log |
| `tcc` | tcc | / 100 | 0-1 | total cloud cover |
| `lsm` | lsm | none (continuous after coarsening) | 0-1 | land-sea mask; use ERA5's continuous field directly, do not binarise |

`tclw` (NaN in HRRR) and `cape` (inconsistent field selection across archive versions) are not used.

**Pressure levels (20):** `u`, `v`, `z` (geopotential height, m = ERA5 geopotential / 9.80665), `t`, `q` at 1000, 850, 500, 250 hPa
(`u1000 ... q250`).

**Derived (1):** `cos_sza`, cosine of the solar zenith angle from latitude, longitude and time.

At model-input time the 34 coarse channels are bilinearly upsampled to 256 x 256 and two invariant channels (terrain height `elev_mean`
and land-sea mask `lsm_mean`) are appended, giving 36 input channels.

## Output channels (8) -- native HRRR 256 x 256

| Channel | HRRR variable | Source | Transform | Unit |
|---|---|---|---|---|
| `2t` | t2m | f00 | none | K |
| `10u`, `10v` | u10, v10 | f00 | none | m/s |
| `tp` | tp | **f01** | log10(1 + mm) | log10(1+mm/h) |
| `ssrd` | sdswrf | f00 | none | W/m2 |
| `strd` | sdlwrf | f00 | none | W/m2 |
| `sp` | sp | f00 | none | Pa |
| `q` | sh2 | f00 | none | kg/kg |

HRRR f00 has zero precipitation accumulation at initialisation, so precipitation (input `tp_in` and target `tp`) comes from the 1-hour forecast f01.

## Normalization statistics

`stats.json` is computed on the training file only (`collect_hrrr.py --split stats`). Because the patches are at least 70 % ocean, spatial
variability is smaller than it would be for land-containing patches; for example the 2-m temperature standard deviation is 5.9 K and the surface pressure
standard deviation 1,980 Pa. A model must always be run with the `stats.json` of the data it was trained on (`assets/stats.json` is the statistics file of the
dataset described here).

## Data sources and limitations

* HRRR archive: public bucket `gs://high-resolution-rapid-refresh` (surface ~113 MB, pressure ~413 MB per timestamp); a few timestamps are missing and are replaced from the reserve.
* ERA5 (inference only): public ARCO ERA5 Zarr store.
* HRRR is the ground truth: errors of HRRR itself (e.g. radiation under broken low clouds) are inherited by the model.
* The 256 x 256 windows cover coastal ocean; the model is not trained for inland or high-latitude conditions beyond the HRRR CONUS domain.
