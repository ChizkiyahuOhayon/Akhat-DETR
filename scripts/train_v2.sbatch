#!/bin/bash
#SBATCH -J akhat-v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH -c 10
#SBATCH -t 26:00:00
#SBATCH --array=0-2%3
#SBATCH -o /path/to/akhat/logs/v2_%A_%a.out
# Revised-code round (one round only, rule fixed before the first round finished):
#   c4  full model: identity-initialised dehazing module + FFTA with untied fusion weights
#   c2  no dehazing module, FFTA with untied fusion weights
#   c3  identity-initialised dehazing module + standard FPN
# Same split, schedule, COCO initialisation and seed as the first round. Imports from
# repro_v2 so that the first-round code in repro/ stays as it was.
set -euo pipefail
ROOT=/path/to/akhat
JOBS=(c4_swinl_ffta_dehaze_v2:0 c2_swinl_ffta_v2:0 c3_swinl_fpn_dehaze_v2:0)
JOB=${JOBS[$SLURM_ARRAY_TASK_ID]}
ARM=${JOB%%:*}
SEED=${JOB##*:}
WEIGHTS=$ROOT/weights/ddq_swinl_coco.pth
WORK=$ROOT/work_dirs/${ARM}_seed${SEED}
[ -f "$WORK/.done" ] && { echo "already done: $ARM seed $SEED"; exit 0; }

module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
export TORCH_HOME=$ROOT/torch_home PYTHONPATH=$ROOT/repro_v2:${PYTHONPATH:-}
mkdir -p "$WORK"
RESUME=()
[ -f "$WORK/last_checkpoint" ] && RESUME=(--resume)

echo "START $ARM seed=$SEED $(hostname) $(date -Is)"
nvidia-smi -L
cd "$ROOT/mmdetection"
python tools/train.py "$ROOT/repro_v2/configs/$ARM.py" --work-dir "$WORK" "${RESUME[@]}" \
    --cfg-options "randomness.seed=$SEED" "load_from=$WEIGHTS"
touch "$WORK/.done"
echo "DONE $ARM seed=$SEED $(date -Is)"
