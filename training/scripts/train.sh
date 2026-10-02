#!/bin/bash
#SBATCH --job-name=ccd_train
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --mem=1800G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=ccd_train_%j.log
# Add your cluster's partition/account, e.g.  #SBATCH --partition=<gpu-partition>

# Two-stage CorrDiff training (regression UNet, then EDM diffusion residual model).
# Resumable: resubmitting continues the current stage from its latest checkpoint; once both stages
# are complete the job exits immediately, so a dependency chain of submissions is safe, e.g.
#     j=$(sbatch --parsable training/scripts/train.sh)
#     for i in 2 3 4 5 6; do j=$(sbatch --parsable --dependency=afterany:$j training/scripts/train.sh); done
#
# Required environment (export before sbatch, or edit below):
#   CCD_DATA_DIR            directory containing regional_train.nc, stats.json (+ the .npy caches, see build_cache.py)
#   CCD_CKPT_DIR            checkpoint output directory
#   PHYSICSNEMO_CORRDIFF_DIR  <physicsnemo>/examples/weather/corrdiff, prepared as in training/README.md
# Optional: REG_TARGET / DIF_TARGET (training samples per stage), CONDA_ENV.

REG_TARGET=${REG_TARGET:-5000000}
DIF_TARGET=${DIF_TARGET:-50000000}
: "${CCD_DATA_DIR:?set CCD_DATA_DIR}"; : "${CCD_CKPT_DIR:?set CCD_CKPT_DIR}"
: "${PHYSICSNEMO_CORRDIFF_DIR:?set PHYSICSNEMO_CORRDIFF_DIR}"
export CCD_DATA_DIR CCD_CKPT_DIR

[ -n "$CONDA_ENV" ] && { source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; }
cd "$PHYSICSNEMO_CORRDIFF_DIR" || exit 1

latest() {  # latest(dir, prefix) -> path of the highest-numbered checkpoint, empty if none
    ls "$1"/"$2".*.mdlus 2>/dev/null | awk -F. '{print $(NF-1), $0}' | sort -n | tail -1 | cut -d' ' -f2
}
num() { basename "$1" .mdlus | awk -F. '{print $NF}'; }

REG_CKPT=$(latest "$CCD_CKPT_DIR/regression/checkpoints_regression" CorrDiffRegressionUNet)
REG_N=0; [ -n "$REG_CKPT" ] && REG_N=$(num "$REG_CKPT")

# per-GPU batch 40 x 4 GPUs = total batch 160; attention at 16x16 resolution
COMMON="++model.model_args.attn_resolutions=[16] ++training.hp.batch_size_per_gpu=40 ++training.hp.total_batch_size=160 ++training.io.save_n_recent_checkpoints=3"

if [ "$REG_N" -lt "$REG_TARGET" ]; then
    echo "=== REGRESSION: at $REG_N / $REG_TARGET ==="
    torchrun --standalone --nproc_per_node=4 train.py --config-name=config_regression.yaml \
        ++training.hp.training_duration=$REG_TARGET $COMMON
else
    DIF_CKPT=$(latest "$CCD_CKPT_DIR/diffusion/checkpoints_diffusion" EDMPrecondSuperResolution)
    DIF_N=0; [ -n "$DIF_CKPT" ] && DIF_N=$(num "$DIF_CKPT")
    if [ "$DIF_N" -ge "$DIF_TARGET" ]; then echo "=== training complete (diffusion at $DIF_N) ==="; exit 0; fi
    echo "=== DIFFUSION: at $DIF_N / $DIF_TARGET, regression checkpoint $REG_CKPT ==="
    torchrun --standalone --nproc_per_node=4 train.py --config-name=config_diffusion.yaml \
        ++training.hp.training_duration=$DIF_TARGET $COMMON \
        ++training.io.regression_checkpoint_path=$REG_CKPT
fi
