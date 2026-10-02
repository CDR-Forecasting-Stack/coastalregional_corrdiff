#!/bin/bash
#SBATCH --job-name=ccd_cache
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=06:00:00
#SBATCH --output=ccd_cache_%j.log
# Add your cluster's partition/account, e.g.  #SBATCH --partition=<cpu-partition>

# Build the memory-mapped .npy caches used by the training dataset class (see training/scripts/build_cache.py).
#   export CCD_DATA_DIR=/path/to/data PHYSICSNEMO_CORRDIFF_DIR=/path/to/physicsnemo/examples/weather/corrdiff
#   sbatch data_pipeline/scripts/build_cache.sbatch
# The 252 GB output group is stacked in memory (copy included) -> 900 GB requested; ~1.2 h for 120,000 samples.
: "${CCD_DATA_DIR:?}"; : "${PHYSICSNEMO_CORRDIFF_DIR:?}"
[ -n "$CONDA_ENV" ] && { source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; }
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"
python training/scripts/build_cache.py --data "$CCD_DATA_DIR/regional_train.nc"
