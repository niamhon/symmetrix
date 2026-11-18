#!/bin/bash -l
set -euo pipefail

repo=/mnt/home/noneill/Programs/symmetrix-dev
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export LSC_REQUIRE_CUDA=1
cd "$repo"
run_id="${SLURM_JOB_ID:-manual}"
hostname
python -m pip check
python -m pytest symmetrix/test/lsc/test_cuda_loader.py symmetrix/test/lsc/test_cuda_geometry.py symmetrix/test/lsc/test_cuda_local.py symmetrix/test/lsc/test_cuda_sources.py symmetrix/test/lsc/test_cuda_energy.py symmetrix/test/lsc/test_cuda_derivatives.py symmetrix/test/lsc/test_native_calculator.py -q -s --tb=short \
    -o junit_family=xunit1 --junitxml=".lsc-reference-logs/cuda-integration-gpu-${run_id}.xml" "$@" \
    2>&1 | tee ".lsc-reference-logs/cuda-integration-gpu-${run_id}.log"
