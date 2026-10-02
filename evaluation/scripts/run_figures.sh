#!/bin/bash
# Regenerates the evaluation figures and tables from the evaluation shards written by
# evaluation/scripts/run_generate_eval.sbatch. CPU only (~1-2 h for 1,000 samples).
# Run from the repository root:
#   EVAL_DIR=./eval_data OUT_DIR=./figure_output TRAIN_NC=$CCD_DATA_DIR/regional_train.nc \
#       bash evaluation/scripts/run_figures.sh
# Optional case-study figures: FRONTAL_IDXS="151 407" STORM_IDXS="461 563" (sample indices into the
# evaluation set; evaluation/figures/find_case_candidates.py helps pick them).
#SBATCH --job-name=ccd_figs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
E="${EVAL_DIR:-$ROOT/eval_data}"
O="${OUT_DIR:-$ROOT/figure_output}"
TRAIN_NC="${TRAIN_NC:-${CCD_DATA_DIR:-.}/regional_train.nc}"
mkdir -p "$O"
cd "$ROOT/evaluation/figures"

python3 tables.py                    --eval-dir "$E" --outdir "$O"   # skill tables
python3 plot_crps.py            --eval-dir "$E" --outdir "$O"   # CRPS / skill scores
python3 plot_scatter.py         --eval-dir "$E" --outdir "$O"   # density scatter vs HRRR truth
python3 plot_spatial_maps.py         --eval-dir "$E" --outdir "$O"   # spatial maps (one file per variable)
python3 plot_rank_histograms.py --eval-dir "$E" --outdir "$O"   # ensemble rank histograms
python3 plot_spread_skill.py    --eval-dir "$E" --outdir "$O"   # spread-skill
python3 plot_rapsd.py           --eval-dir "$E" --outdir "$O"   # radially averaged power spectra
python3 plot_precipitation.py          --eval-dir "$E" --outdir "$O"   # precipitation statistics
python3 plot_qq.py              --eval-dir "$E" --outdir "$O"   # quantile-quantile
python3 plot_regional_skill.py        --eval-dir "$E" --outdir "$O"   # regional skill maps
python3 plot_ensemble_sensitivity.py --eval-dir "$E" --outdir "$O"  # ensemble-size sensitivity
python3 plot_training_data.py   --train "$TRAIN_NC" --outdir "$O"   # training-data coverage
if [ -n "${FRONTAL_IDXS:-}" ]; then
    python3 plot_frontal_case.py --eval-dir "$E" --outdir "$O" --idxs $FRONTAL_IDXS
fi
if [ -n "${STORM_IDXS:-}" ]; then
    python3 plot_storm_case.py --eval-dir "$E" --outdir "$O" --idxs $STORM_IDXS
fi
echo "done -> $O"
