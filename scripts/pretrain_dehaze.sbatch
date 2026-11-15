#!/bin/bash
#SBATCH -J akhat-pretrain
#SBATCH -p dev
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 00:55:00
#SBATCH -o /path/to/akhat/logs/pretrain_%j.out
# Pretrain the dehazing module on RESIDE OTS pairs (short job, dev partition).
set -euo pipefail
ROOT=/path/to/akhat
module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
export PYTHONPATH=$ROOT/repro_v2:${PYTHONPATH:-}
nvidia-smi -L
cd "$ROOT/repro_v2"
python tools/pretrain_dehaze.py $ROOT/ots/clear $ROOT/ots/part2 $ROOT/weights/dehaze_ots_pretrained.pth --iters 20000
