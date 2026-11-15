#!/bin/bash
#SBATCH -J akhat-final
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH -c 10
#SBATCH -t 26:00:00
#SBATCH -o /path/to/akhat/logs/final_%j.out
# Final runs of the third round: full training split, 30 epochs, seed 0, COCO initialisation.
#   sbatch [-t hh:mm:ss] final.sbatch <config name> <weights file in weights/>
# e.g. sbatch -t 14:00:00 final.sbatch r50_a_v2 ddq_r50_coco.pth
set -euo pipefail
ROOT=/path/to/akhat
ARM=${1:?config}
WEIGHTS=$ROOT/weights/${2:?weights}
WORK=$ROOT/work_dirs/${ARM}_seed0
[ -f "$WORK/.done" ] && { echo "already done: $ARM"; exit 0; }
module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
export TORCH_HOME=$ROOT/torch_home PYTHONPATH=$ROOT/repro_v2:${PYTHONPATH:-}
mkdir -p "$WORK"
RESUME=()
[ -f "$WORK/last_checkpoint" ] && RESUME=(--resume)
echo "START $ARM $(hostname) $(date -Is)"
nvidia-smi -L
cd "$ROOT/mmdetection"
python tools/train.py "$ROOT/repro_v2/configs/$ARM.py" --work-dir "$WORK" "${RESUME[@]}" \
    --cfg-options "randomness.seed=0" "load_from=$WEIGHTS"
touch "$WORK/.done"
echo "DONE $ARM $(date -Is)"
