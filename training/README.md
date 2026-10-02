# training

Two-stage CorrDiff training with [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
(`examples/weather/corrdiff`, commit `3cd5ffc450b610f8b671649047426a8801f49462`):

1. **Regression** UNet (`config_regression.yaml`), 5,000,000 training samples;
2. **Diffusion** EDM residual model (`config_diffusion.yaml`), 50,000,000 samples, conditioned on the final regression checkpoint.

Per-GPU batch 40 on 4 GPUs (total batch 160), Adam learning rate 2e-4, attention at 16 x 16 resolution, last three checkpoints kept.

| File | Purpose |
|---|---|
| `configs/config_regression.yaml`, `configs/config_diffusion.yaml` | Hydra configs. Data/checkpoint paths come from `CCD_DATA_DIR` / `CCD_CKPT_DIR`. |
| `configs/dataset_regional_hrrr.yaml` | Hydra dataset group (goes to `conf/base/dataset/regional_hrrr.yaml`). |
| `physicsnemo/regional_hrrr.py` | Dataset class: reads the stored 32 x 32 input, bilinearly upsamples it to 256 x 256, appends the two invariant channels (36 input channels), z-score normalizes with `stats.json`; memory-maps `.npy` caches of the NetCDF arrays. |
| `physicsnemo/physicsnemo_regional.patch` | Small PhysicsNeMo edits: registers the `regional_hrrr` dataset (needed), minor `generate.py` changes. |
| `scripts/build_cache.py` | Builds the `.npy` caches once on a CPU node. |
| `scripts/train.sh` | SLURM launcher: runs the stage that is not yet finished; resumable; safe to chain. |

## Setup

```bash
git clone https://github.com/NVIDIA/physicsnemo && cd physicsnemo
git checkout 3cd5ffc450b610f8b671649047426a8801f49462
git apply /path/to/coastalregional_corrdiff/training/physicsnemo/physicsnemo_regional.patch
pip install -e .
cd examples/weather/corrdiff
R=/path/to/coastalregional_corrdiff/training
cp $R/physicsnemo/regional_hrrr.py                 datasets/
cp $R/configs/dataset_regional_hrrr.yaml          conf/base/dataset/regional_hrrr.yaml
cp $R/configs/config_regression.yaml $R/configs/config_diffusion.yaml conf/
export PHYSICSNEMO_CORRDIFF_DIR=$PWD CCD_DATA_DIR=/path/to/data CCD_CKPT_DIR=/path/to/checkpoints
```

## Run

```bash
# 1) once, on a CPU node: ~1.2 h for 120,000 samples, needs ~900 GB RAM (the output group alone is 252 GB, stacked in memory)
sbatch data_pipeline/scripts/build_cache.sbatch
# 2) training: regression first, then diffusion; resubmitting resumes from the latest checkpoint
j=$(sbatch --parsable training/scripts/train.sh)
for i in 2 3 4 5 6; do j=$(sbatch --parsable --dependency=afterany:$j training/scripts/train.sh); done
```

`train.sh` decides which stage to run from the highest checkpoint number in `$CCD_CKPT_DIR/{regression,diffusion}`, starts diffusion only after regression
reached `REG_TARGET`, passes the final regression checkpoint to it automatically, and exits immediately when both stages are complete.
Edit the `#SBATCH` header (add your partition/account) and use `REG_TARGET` / `DIF_TARGET` to change the lengths. The batch script is copied by SLURM at submission time,
so change these values before submitting.

## What to expect (4 x B200 GPUs)

* Regression: ~150 samples/s, so 5,000,000 samples take ~9 h; the loss falls from ~15,000 to ~3,900 (flattening after ~4M samples).
* Diffusion: ~110 samples/s, so 50,000,000 samples take ~5 days (several 24 h jobs); the loss falls from ~22,000 to ~14,000 within the first 2M samples and then decreases slowly.
* GPU memory is close to full (~170 of 191 GB); PyTorch may print transient allocator warnings at start-up, which are harmless.
* Memory in the job: 1800 GB was requested because all ranks memory-map the 252 GB output cache; the process itself stays near 8 GB.

Checkpoints appear as `$CCD_CKPT_DIR/regression/checkpoints_regression/CorrDiffRegressionUNet.0.<samples>.mdlus` and
`$CCD_CKPT_DIR/diffusion/checkpoints_diffusion/EDMPrecondSuperResolution.0.<samples>.mdlus`.
