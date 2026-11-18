#!/bin/bash -l
set -euo pipefail

repo=/mnt/home/noneill/Programs/symmetrix-dev
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${LSC_TEST_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$repo/.lsc-reference-logs"
cd "$repo"
hostname
python -m pip check
python -m pytest symmetrix/test/lsc symmetrix/test/test_cubic_spline.py \
    -q --durations=5 \
    --junitxml=".lsc-reference-logs/pytest-login-${run_id}.xml" "$@" \
    2>&1 | tee ".lsc-reference-logs/pytest-login-${run_id}.log"
