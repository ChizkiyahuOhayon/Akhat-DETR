#!/bin/bash
#SBATCH -J akhat-screen
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH -c 10
#SBATCH -t 14:00:00
#SBATCH -o /path/to/akhat/logs/screen_%A_%a.out
# Design screening on the validation split held out from the training images.
# Usage: sbatch --array=<i,j,...> screen.sbatch   (indices into RUNS)
set -euo pipefail
ROOT=/path/to/akhat
RUNS=(s0_nodh_ffta s1_preft_ffta s2_prefrz_ffta s3_nodh_fpn s4_nodh_fftaattn)
ARM=${RUNS[$SLURM_ARRAY_TASK_ID]}
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
    --cfg-options "randomness.seed=0" "load_from=$ROOT/weights/ddq_swinl_coco.pth"
touch "$WORK/.done"
echo "DONE $ARM $(date -Is)"
