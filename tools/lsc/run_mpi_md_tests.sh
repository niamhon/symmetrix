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
export LSC_REQUIRE_LAMMPS=1 LSC_SKIP_HELP=1
export LSC_LAMMPS_BINARY="/mnt/home/noneill/Programs/lammps-lsc-dev/build-$backend/lmp"
export LSC_LAMMPS_LAUNCHER='/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none'
cd "$repo"
if [[ "${LSC_REUSE_BUILD:-0}" != 1 ]]; then
    bash -l tools/lsc/build_lammps.sh "$backend"
fi
python -m pytest -q -s -o junit_family=legacy "${selection[@]}" \
    symmetrix/test/lsc/test_lammps_lsc_mpi_md.py \
    --junitxml=".lsc-reference-logs/mpi-md-focused-$backend.xml"
