#!/bin/bash -l
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
lsc_lammps=/mnt/home/noneill/Programs/lammps-lsc-dev
backend="${1:-cuda}"
if [[ "$backend" == cuda ]]; then
    source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
    definitions=(-DSYMMETRIX_CUDA)
else
    module load gcc/13.3.0
    source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
    definitions=()
fi
mpi=/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8
"$mpi/bin/mpicxx" -std=c++20 -O2 "${definitions[@]}" \
    -I"$lsc_lammps/src" -I"$repo/libsymmetrix/source" \
    "$repo/tools/lsc/forward_mpi_driver.cpp" \
    -L"$lsc_lammps/build-$backend" -Wl,-rpath,"$lsc_lammps/build-$backend" -llammps \
    -o "$repo/.lsc-reference-logs/lsc-forward-mpi-$backend"
