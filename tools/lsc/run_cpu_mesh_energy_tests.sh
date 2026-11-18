#!/bin/bash -l
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LSC_REQUIRE_CPU_MESH=1
cd "$repo"
/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none \
    python -m pytest symmetrix/test/lsc/test_cpu_mesh_energy.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=.lsc-reference-logs/cpu-mesh-mpi1.xml 2>&1 | tee .lsc-reference-logs/cpu-mesh-mpi1.log
