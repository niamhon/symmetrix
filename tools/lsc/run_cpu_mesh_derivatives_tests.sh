#!/bin/bash -l
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LSC_REQUIRE_CPU_MESH=1 LSC_REQUIRE_LAMMPS=1
export LSC_LAMMPS_BINARY=/mnt/home/noneill/Programs/lammps-lsc-dev/build-cpu/lmp
export LSC_LAMMPS_LAUNCHER="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none"
cd "$repo"
# Native mesh checks inside one MPI process; LAMMPS launches separately below.
$LSC_LAMMPS_LAUNCHER python -m pytest symmetrix/test/lsc/test_cpu_mesh_energy.py symmetrix/test/lsc/test_cpu_mesh_derivatives.py \
    -q -s --tb=short -o junit_family=xunit1 --junitxml=.lsc-reference-logs/cpu-mesh-derivatives-mpi1.xml \
    2>&1 | tee .lsc-reference-logs/cpu-mesh-derivatives-mpi1.log
python -m pytest symmetrix/test/lsc/test_lammps_lsc_mesh.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=.lsc-reference-logs/cpu-mesh-derivatives-lammps.xml 2>&1 | tee .lsc-reference-logs/cpu-mesh-derivatives-lammps.log
