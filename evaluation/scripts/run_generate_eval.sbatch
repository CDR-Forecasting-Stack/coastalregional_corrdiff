#!/bin/bash
#SBATCH --job-name=ccd_eval
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=ccd_eval_%j.log
# Add your cluster's partition/account, e.g.  #SBATCH --partition=<gpu-partition>

# Runs the trained model (regression + 16-member diffusion ensemble, rain filter 0.02 mm) over a random
# subsample of the held-out test set. Resubmitting resumes any incomplete shard.
#   export CCD_DATA_DIR=/path/to/data CCD_CKPT_DIR=/path/to/checkpoints
#   sbatch evaluation/scripts/run_generate_eval.sbatch                   # 1000 samples, 16 members, 4 GPUs
#   sbatch evaluation/scripts/run_generate_eval.sbatch --n-samples 300 --ensemble 8
: "${CCD_DATA_DIR:?}"; : "${CCD_CKPT_DIR:?}"
[ -n "$CONDA_ENV" ] && { source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; }
cd "${SLURM_SUBMIT_DIR:-.}"
python evaluation/generate_eval_dataset.py \
    --test-nc "$CCD_DATA_DIR/regional_test.nc" --stats "$CCD_DATA_DIR/stats.json" \
    --reg-ckpt-dir "$CCD_CKPT_DIR/regression" --res-ckpt-dir "$CCD_CKPT_DIR/diffusion" \
    --n-samples 1000 --ensemble 16 --gpus 4 --seed-batch-size 8 --rain-filter-mm 0.02 \
    --output-dir ./eval_data "$@"
