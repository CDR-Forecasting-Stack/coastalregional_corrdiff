# coastalregional_corrdiff

Kilometer-scale atmospheric downscaling for **coastal ocean areas** with **CorrDiff** (residual corrective diffusion).
A coarse ~24 km field is downscaled to the 3 km resolution of NOAA's HRRR for eight surface variables (2-m temperature, 10-m winds,
precipitation, downwelling shortwave and longwave radiation, surface pressure, specific humidity) over the ocean and coasts of the contiguous
United States, as an ensemble.

The model is trained **self-supervised on HRRR**: the coarse input is HRRR itself, blurred and conservatively regridded to reanalysis
resolution; the target is the native 3 km HRRR field. At inference the coarse driver can be coarsened **HRRR** or **ERA5**.

## The pipeline

```
 HRRR archive (GCS)                                                             ERA5 (ARCO Zarr)
   │ wrfsfc f00/f01, wrfprs f00                                                      │
   ▼                                                                                 │
 [1] data_pipeline/collect_hrrr.py ── 256x256 ocean-centred patches (3 km)           │
   │   coarse input 32x32 = Gaussian blur (sigma 5 px) + xESMF conservative regrid   │
   │   → regional_train.nc (120,000)  regional_test.nc (10,000)  stats.json          │
   ▼                                                                                 │
 [2] training/  (NVIDIA PhysicsNeMo CorrDiff)                                        │
   │   regression UNet 5,000,000 samples → EDM diffusion residual 50,000,000 samples │
   ▼                                                                                 ▼
 [3] evaluation/  16-member ensemble on the test set → tables, figures   [4] inference/infer_region.py
                                                                             HRRR- or ERA5-driven, any lat/lon box and time
```

### 1. Data collection (`data_pipeline/`)

* **Source:** HRRR analyses (surface f00, 1-hour precipitation from f01, pressure levels f00) from the public `gs://high-resolution-rapid-refresh` archive; no ERA5 is
  used to build the dataset.
* **Patches:** 256 x 256 windows (~768 km) of the 3 km grid whose centre pixel is ocean and which are **at least 70 % ocean**; the 153,171 valid centres are
  split into four longitude bands and every timestamp takes the same number of centres from each band, at random, never repeating a centre inside a timestamp.
* **Timestamps:** hourly, random, with a fixed quota per year and month. **Train: 1,200 timestamps (200 per year, 2018-2023) x 100 patches = 120,000 samples.
  Test: 500 timestamps (250 per year, 2024-2025) x 20 patches = 10,000 samples.** The split is chronological. A random reserve replaces timestamps whose files
  cannot be downloaded, so the counts are exact.
* **Target:** 8 channels at 256 x 256 (`2t`, `10u`, `10v`, `tp`, `ssrd`, `strd`, `sp`, `q`).
* **Input:** 34 channels (13 surface, 20 pressure-level at 1000/850/500/250 hPa, `cos_sza`) at 32 x 32, made from the native patch by a **2-D Gaussian blur
  (sigma = 5 native pixels, 15 km) followed by xESMF conservative regridding** onto 32 x 32 cells of ~24 km. No quantile mapping to ERA5 is applied: HRRR-ERA5
  differences (for example shortwave radiation, from different cloud and aerosol treatment) are physical and are not removed. Sigma was chosen with
  `data_pipeline/tools/sigma_sweep.py` (see [`docs/DATASET.md`](docs/DATASET.md)).
* **Statistics:** `stats.json` holds the per-channel mean and standard deviation of the training file, used to normalize the model input and output.
* Everything is resumable, runs under SLURM (templates provided), and takes about 10 h (train) and 3 h (test) with 8 workers.

### 2. Training (`training/`)

Two-stage CorrDiff with [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo): a **regression UNet** (conditional mean) trained for 5,000,000 samples, then an **EDM
diffusion** model for the residual trained for 50,000,000 samples and conditioned on the final regression checkpoint. Per-GPU batch 40 on 4 GPUs (total 160),
Adam learning rate 2e-4, attention at 16 x 16. The model input is the 34 coarse channels bilinearly upsampled to 256 x 256 plus terrain height and land-sea mask
(36 channels). A launcher script runs the unfinished stage and can be chained for the multi-day diffusion run.

### 3. Evaluation (`evaluation/`)

Runs regression + a 16-member diffusion ensemble over a random, seeded subset (1,000 samples) of the held-out test set and produces deterministic and
probabilistic skill tables (RMSE, MAE, CRPS, spread-skill ratio, precipitation scores) and figures (maps, scatter, rank histograms, spectra, quantile-quantile, ...).

### 4. Inference (`inference/`)

`infer_region.py` downscales any lat/lon box at any time. The driver is coarsened HRRR (`--input_source hrrr`), ERA5 (`era5`), or both. The box is tiled into
256 x 256 patches with feathered overlap, each patch goes through regression + diffusion, and the result is stitched into one map.

**Post-processing of the output** (applied by inference and evaluation, can be disabled):
* **Rain filter** (`--rain_filter_mm`, default **0.02 mm**): the UNet's precipitation is kept wherever the UNet predicts at most the threshold and CorrDiff's is used above it
  (the diffusion residual adds realistic rain texture but only adds noise where it is dry).
* **Physical constraints:** `ssrd`, `strd` and `tp` are never negative; `ssrd` is set to zero wherever the sun is at or below the horizon (cos of the solar zenith angle <= 0);
  `strd` is not zeroed at night (the atmosphere emits longwave radiation day and night). NaN is preserved.

## Repository layout

```
coastalregional_corrdiff/
├── data_pipeline/   collect_hrrr.py, coarsen.py, era5_input.py, tools/ (sigma sweep, data-quality plots), scripts/ (SLURM templates)
├── training/        configs/, physicsnemo/ (dataset class + PhysicsNeMo patch), scripts/ (train.sh, build_cache.py)
├── evaluation/      generate_eval_dataset.py, figures/ (tables and plots), scripts/
├── inference/       infer_region.py
├── common/          ccd_model.py (PhysicsNeMo model, normalization, patch inference), channels.py, rain_filter.py, physical_constraints.py
├── assets/          hrrr_invariants.nc (HRRR grid, terrain, land-sea mask), stats.json (normalization statistics of the training data)
├── docs/            DATASET.md (channels, NetCDF schema, sampling rules, coarsening)
└── tests/           unit tests
```

## Install

```bash
conda env create -f environment.yml          # Python 3.11, xESMF/ESMF, cfgrib/eccodes, PyTorch, PhysicsNeMo
conda activate coastalregional
```

PhysicsNeMo (`pip install nvidia-physicsnemo`) is required for training, evaluation and inference; training additionally needs a clone of the PhysicsNeMo
repository (see [`training/README.md`](training/README.md)). The data pipeline needs only the HRRR/ERA5 libraries and xESMF. A GPU is needed for training and
practical for inference; the scripts also run on CPU (slowly) for smoke tests.

Environment variables used throughout (every script also takes explicit command-line arguments):

| Variable | Meaning |
|---|---|
| `CCD_DATA_DIR` | directory with `regional_train.nc`, `regional_test.nc`, `stats.json` |
| `CCD_CKPT_DIR` | checkpoint directory (`regression/` and `diffusion/` below it) |
| `PHYSICSNEMO_CORRDIFF_DIR` | `<physicsnemo>/examples/weather/corrdiff` (training) |

## Quick start

```bash
export CCD_DATA_DIR=/path/to/data CCD_CKPT_DIR=/path/to/checkpoints

# 1. data (SLURM; sizes and runtimes in data_pipeline/README.md)
sbatch --job-name=train data_pipeline/scripts/collect_data.sbatch train
sbatch --job-name=test  data_pipeline/scripts/collect_data.sbatch test
sbatch --job-name=stats data_pipeline/scripts/collect_data.sbatch stats

# 2. training: cache the data once, then regression -> diffusion (resumable; chain several submissions)
sbatch data_pipeline/scripts/build_cache.sbatch
sbatch training/scripts/train.sh

# 3. evaluation
sbatch evaluation/scripts/run_generate_eval.sbatch
bash   evaluation/scripts/run_figures.sh

# 4. inference from HRRR and from ERA5 over a box
python inference/infer_region.py --timestamp "2025-07-15 00:00" \
    --lat_min 25 --lat_max 31 --lon_min -98 --lon_max -88 --input_source both \
    --reg_ckpt_dir $CCD_CKPT_DIR/regression --res_ckpt_dir $CCD_CKPT_DIR/diffusion \
    --stats_path $CCD_DATA_DIR/stats.json --num_ensembles 8
```

## Checkpoints

The trained weights (regression and diffusion, about 300 MB each) are not stored in this repository. They are archived on Zenodo:
**<https://doi.org/10.5281/zenodo.22942778>** 

Download them into `$CCD_CKPT_DIR/regression/` and `$CCD_CKPT_DIR/diffusion/` (the `.mdlus` files; the highest-numbered file in each directory is used) and use the
`stats.json` that was trained with them (`assets/stats.json` is the statistics file of the training set described above).

## Tests

```bash
python -m pytest tests -q      # coarsening, sampling rules, rain filter (the coarsening tests need xesmf)
```

## Citation and license

Please cite this software using [`CITATION.cff`](CITATION.cff).

Licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)). The CorrDiff model, EDM preconditioning and training recipe come from
[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (Apache-2.0); see [`NOTICE`](NOTICE). This project is not affiliated with or endorsed by NVIDIA.
HRRR data are provided by NOAA through the public cloud archive; ERA5 by ECMWF/Copernicus through the public ARCO ERA5 store.

Authors: Sadegh Ranjbar, Lucas Gloege, Noah Planavsky, Elizabeth Yankovsky (Yale University).
