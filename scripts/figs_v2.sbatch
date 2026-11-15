#!/bin/bash
#SBATCH -J akhat-figs2
#SBATCH -p dev
#SBATCH -c 16
#SBATCH -t 00:59:00
#SBATCH -o /path/to/akhat/logs/figs_%j.out
# Figures 1 and 5 on CPU (no GPU slot needed). Usage:
#   sbatch figs.sbatch <checkpoint> <out_dir>
set -euo pipefail
ROOT=/path/to/akhat
CKPT=${1:?checkpoint}
OUT=${2:?out_dir}
module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
export TORCH_HOME=$ROOT/torch_home PYTHONPATH=$ROOT/repro_v2:${PYTHONPATH:-} OMP_NUM_THREADS=16
cd "$ROOT/repro_v2"
python tools/make_figs.py configs/c4_swinl_ffta_dehaze_v2.py "$CKPT" "$OUT" \
    $ROOT/figs/photos/commons_06.jpg $ROOT/figs/photos/commons_04.jpg \
    $ROOT/figs/photos/commons_02.jpg $ROOT/figs/photos/commons_19_plates_blurred.jpg \
    $ROOT/figs/photos/commons_11.jpg --device cpu 2>&1 | grep -vE "The value is the same|^ *$"
echo FIGS_DONE
