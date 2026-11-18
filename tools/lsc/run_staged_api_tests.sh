#!/bin/bash -l
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
backend="${1:-cuda}"
if [[ "$backend" == cuda ]]; then
    source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
    export LSC_REQUIRE_CUDA=1
    selection=()
elif [[ "$backend" == cpu ]]; then
    module load gcc/13.3.0
    source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
    selection=(-m 'not lsc_cuda')
else
    echo 'Expected cpu or cuda' >&2
    exit 2
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$repo"
python -m pytest -q -s -o junit_family=legacy symmetrix/test/lsc/test_staged_api.py \
    "${selection[@]}" --junitxml=".lsc-reference-logs/staged-api-$backend.xml"
