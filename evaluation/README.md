# evaluation

1. `generate_eval_dataset.py` runs the trained model (regression + diffusion ensemble, rain filter) over a random, seeded subsample of the held-out test set and writes NetCDF shards
   (one per GPU) with the groups `low_res_input`, `low_res_input_upsampled` (bilinear baseline), `regression`, `prediction` (ensemble) and `target`.
   Precipitation is converted to mm. The same post-processing as inference is applied: rain filter (default 0.02 mm; `--rain-filter-mm`, `--no-rain-filter`) and physical constraints (`ssrd`/`strd`/`tp` >= 0, `ssrd` = 0 at night; `--no-physical-constraints`). It uses the same PhysicsNeMo code path as `inference/infer_region.py` (`common/ccd_model.py`).
2. `figures/` reads those shards and produces tables and figures; `scripts/run_figures.sh` runs them all.

```bash
export CCD_DATA_DIR=/path/to/data CCD_CKPT_DIR=/path/to/checkpoints
sbatch evaluation/scripts/run_generate_eval.sbatch                       # 1000 samples, 16 members, 4 GPUs (resumable)
EVAL_DIR=./eval_data OUT_DIR=./figure_output bash evaluation/scripts/run_figures.sh
```

Smoke test on CPU: `python evaluation/generate_eval_dataset.py ... --n-samples 2 --ensemble 2 --sampler-steps 3 --cpu`.

| Script | Output |
|---|---|
| `tables.py` | deterministic and probabilistic skill tables (CRPS, spread-skill ratio) and precipitation scores (CSV) |
| `plot_crps.py` | CRPS and skill scores against the bilinear and regression baselines |
| `plot_scatter.py` | density scatter against HRRR truth |
| `plot_spatial_maps.py` | spatial maps per variable |
| `plot_rank_histograms.py`, `plot_spread_skill.py` | ensemble calibration |
| `plot_rapsd.py` | radially averaged power spectra |
| `plot_precipitation.py`, `plot_qq.py` | precipitation statistics, quantile-quantile |
| `plot_regional_skill.py` | skill maps by region |
| `plot_ensemble_sensitivity.py` | sensitivity to ensemble size |
| `plot_training_data.py` | training-data coverage |
| `plot_frontal_case.py`, `plot_storm_case.py`, `find_case_candidates.py` | case studies (sample indices given with `--idxs`) |

Common utilities: `eval_io.py` (shard reader), `fig_style.py` (style and variable metadata), `metrics.py`, `_rapsd_core.py`. Terminology used in all figures: *Low-res HRRR* is the model's
coarse conditioning field (or its bilinear upsampling), *UNet Regression* the deterministic first stage, *CorrDiff* the final output, *HRRR truth* the target.
