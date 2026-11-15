#!/bin/bash
#SBATCH -J akhat-smoke2
#SBATCH -p dev
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -o /path/to/akhat/logs/smoke2_%j.out
# CPU checks of the revised-code arms. Runs from repro_v2 so the first-round runs, which
# import from repro/, are not affected.
set -euo pipefail
ROOT=/path/to/akhat
module load anaconda3/2023.09-0-gcc-11.5.0-mxpgp2g
eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
export TORCH_HOME=$ROOT/torch_home PYTHONPATH=$ROOT/repro_v2:${PYTHONPATH:-} OMP_NUM_THREADS=16
cd "$ROOT/repro_v2"
python tools/smoke_v2.py "$ROOT/work_dirs/b4_swinl_ffta_dehaze_seed0/epoch_14.pth" 2>&1 \
    | grep -vE "The value is the same|^ *$"
