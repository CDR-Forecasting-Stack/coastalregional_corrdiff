#!/bin/bash
#SBATCH --job-name=ccd_collect
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=ccd_collect_%x_%j.log
# Add your cluster's partition/account, e.g.  #SBATCH --partition=<cpu-partition>

# Build the training or test set (resumable: re-submit the same command to continue from the checkpoint).
#   export CCD_DATA_DIR=/path/to/data
#   sbatch --job-name=train data_pipeline/scripts/collect_data.sbatch train     # ~10 h, 120,000 samples
#   sbatch --job-name=test  data_pipeline/scripts/collect_data.sbatch test      # ~3 h,   10,000 samples
#   sbatch --job-name=stats data_pipeline/scripts/collect_data.sbatch stats     # after train finishes (~30 min)
# Memory: each of the 8 parallel timestamps peaks near 15 GB (full-grid GRIB fields), so ~120 GB are needed;
# 400 GB leaves headroom for the file cache.
SPLIT=${1:?usage: collect_data.sbatch train|test|stats|validate}
: "${CCD_DATA_DIR:?set CCD_DATA_DIR}"
[ -n "$CONDA_ENV" ] && { source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; }
export MALLOC_ARENA_MAX=4      # limit glibc per-thread arena growth
cd "${SLURM_SUBMIT_DIR:-.}/data_pipeline"
python collect_hrrr.py --split "$SPLIT" --workers 8 --gaussian-sigma-px 5 --out-base "$CCD_DATA_DIR"
