# data_pipeline

Builds the paired **coarse HRRR -> native HRRR** dataset. See [`../docs/DATASET.md`](../docs/DATASET.md) for the channel list, NetCDF schema,
patch-selection rules and the choice of the Gaussian width.

| File | Purpose |
|---|---|
| `collect_hrrr.py` | Downloads HRRR GRIB2 (surface f00/f01, pressure f00), builds the 34-channel input and 8-channel output stacks on the native 3 km CONUS grid, cuts 256 x 256 ocean-centred patches, coarsens them to 32 x 32, writes NetCDF. Also computes normalization statistics (`--split stats`) and validates a file (`--split validate`). Resumable. |
| `coarsen.py` | `gaussian_blur` (NaN-aware) and `ConservativeCoarsener` (xESMF conservative regridding of each patch onto its own 32 x 32 cells). |
| `era5_input.py` | Builds the same 34 channels from ERA5 (public ARCO Zarr store) on an arbitrary grid; used by inference and by `tools/sigma_sweep.py`. |
| `tools/sigma_sweep.py` | Compares a plain block mean (no blur) and several Gaussian widths with ERA5 on test patches (maps, spectra, error table). |
| `tools/plot_coarsen_test.py` | Maps and histograms for a small test collection. |
| `tools/plot_data_quality.py` | Quality checks of a finished file: patch-mean and pixel density scatter, maps, input/target histograms. |
| `tests/test_coarsen.py` | Unit tests of the coarsening. |
| `scripts/` | SLURM templates (`collect_data.sbatch`, `build_cache.sbatch`). |

## Usage

```bash
export CCD_DATA_DIR=/path/to/data
cd data_pipeline

# small test run first: one timestamp, 100 patches (about a minute)
python collect_hrrr.py --split test --timestamp "2024-08-03 18:00" --per-ts 100 --out-base /tmp/ccd_test
python tools/plot_coarsen_test.py --dataset /tmp/ccd_test/regional_test.nc

# full collection (SLURM templates in scripts/): resumable, re-submit to continue
python collect_hrrr.py --split train --workers 8 --gaussian-sigma-px 5 --out-base $CCD_DATA_DIR
python collect_hrrr.py --split test  --workers 8 --gaussian-sigma-px 5 --out-base $CCD_DATA_DIR
python collect_hrrr.py --split stats --out-base $CCD_DATA_DIR      # stats.json from the training file
python tools/plot_data_quality.py --dataset $CCD_DATA_DIR/regional_train.nc
```

Defaults: train 1,200 timestamps (200 per year) x 100 patches = 120,000 samples; test 500 timestamps (250 per year) x 20 patches = 10,000 samples;
at least 70 % ocean per patch. Edit the constants near the top of `collect_hrrr.py` (`TRAIN_*`, `TEST_*`, `OCEAN_MIN_FRAC`) or use `--n-ts`, `--per-ts`.

## Resources

* **Time:** about 10,000 patches per hour per job with 8 workers (the train set took ~9.5 h, the test set ~2.7 h). Most of it is downloading GRIB files.
* **Memory:** each timestamp being processed peaks at about 15 GB (full-grid GRIB fields); with `--workers 8` request at least ~130 GB, 400 GB is comfortable.
  Too little memory ends in an out-of-memory kill; the checkpoint makes the re-run cheap.
* **Disk:** `regional_train.nc` is ~65 GB (compressed). The training dataset class additionally builds ~270 GB of `.npy` caches next to it (see `../training`).
* **Network:** each timestamp reads ~640 MB from the public HRRR bucket. Some HPC clusters need the SSL workaround already built into the downloader.

## Notes

* The HRRR grid invariants (`assets/hrrr_invariants.nc`: latitude, longitude, terrain height, land-sea mask) ship with the repository; point `CCD_HRRR_INVARIANTS` at another file to override.
* Install `xesmf` and `esmpy` from conda-forge; the unit tests skip themselves if xESMF is missing.
* No ERA5 enters the dataset. ERA5 is used only for the optional sigma comparison and at inference.
