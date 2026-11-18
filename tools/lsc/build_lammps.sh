#!/bin/bash -l
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
lsc_lammps=/mnt/home/noneill/Programs/lammps-lsc-dev
backend="${1:-cpu}"
if [[ "$backend" == cuda ]]; then
    source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
    lsc_native="$repo/symmetrix/build-lsc-cuda"
    lsc_enable_cuda=ON
else
    module load gcc/13.3.0
    source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
    lsc_native="$repo/symmetrix/build"
    lsc_enable_cuda=OFF
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cmake -S "$lsc_lammps/cmake" -B "$lsc_lammps/build-$backend" \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$(command -v g++)" \
    -D CMAKE_CXX_FLAGS=-march=x86-64-v3 -D CMAKE_CXX_STANDARD=20 \
    -D PKG_KSPACE=ON -D FFT=FFTW3 -D FFT_SINGLE=OFF \
    -D BUILD_MPI=ON -D BUILD_OMP=ON -D BUILD_SHARED_LIBS=ON -D DOWNLOAD_POTENTIALS=OFF \
    -D MPI_CXX_COMPILER=/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpicxx \
    -D LSC_SYMMETRIX_ROOT="$repo" \
    -D LSC_SYMMETRIX_LIBRARY="$lsc_native/libsymmetrix/libsymmetrix.a" \
    -D LSC_SPHERICART_LIBRARY="$lsc_native/libsymmetrix/external/sphericart/sphericart/libsphericart.a" \
    -D LSC_ENABLE_CUDA="$lsc_enable_cuda"
cmake --build "$lsc_lammps/build-$backend" --parallel "${LSC_BUILD_JOBS:-4}"
if [[ "${LSC_SKIP_HELP:-0}" != 1 ]]; then
    "$lsc_lammps/build-$backend/lmp" -h | sed -n '/symmetrix/p'
fi
