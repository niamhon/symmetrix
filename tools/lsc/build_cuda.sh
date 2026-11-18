#!/bin/bash -l
set -euo pipefail

repo=/mnt/home/noneill/Programs/symmetrix-dev
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export PYTHONDONTWRITEBYTECODE=1
export CMAKE_BUILD_PARALLEL_LEVEL="${LSC_BUILD_JOBS:-4}"
export OMP_NUM_THREADS="$CMAKE_BUILD_PARALLEL_LEVEL"
export OPENBLAS_NUM_THREADS=1
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
lsc_cuda_root="$(dirname "$(dirname "$(command -v nvcc)")")"
cd "$repo"
# Clear deprecated KokkosKernels cache aliases on repeat configurations.
python -m pip install --verbose --no-deps --no-build-isolation --editable ./symmetrix \
    -Cbuild-dir="$repo/symmetrix/build-lsc-cuda" \
    "-Ccmake.args=-UKOKKOSKERNELS_*" \
    -Ccmake.define.CMAKE_CXX_COMPILER="$repo/libsymmetrix/external/kokkos/bin/nvcc_wrapper" \
    -Ccmake.define.CMAKE_CUDA_COMPILER="$lsc_cuda_root/bin/nvcc" \
    -Ccmake.define.CUDAToolkit_ROOT="$lsc_cuda_root" \
    -Ccmake.define.SYMMETRIX_KOKKOS=ON \
    -Ccmake.define.SYMMETRIX_NATIVE_ARCH=OFF \
    -Ccmake.define.SPHERICART_ARCH_NATIVE=OFF \
    -Ccmake.define.CMAKE_CXX_FLAGS=-march=x86-64-v3 \
    -Ccmake.define.Kokkos_ENABLE_CUDA=ON \
    -Ccmake.define.Kokkos_ARCH_NATIVE=OFF \
    -Ccmake.define.Kokkos_ARCH_AMPERE80=ON \
    -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80 \
    -Ccmake.define.SYMMETRIX_LSC_MESH=ON \
    -Ccmake.define.SYMMETRIX_SPHERICART_CUDA=OFF
python -c 'import symmetrix,sys; print("LSC_CUDA_BUILD",sys.prefix,symmetrix.lsc_cuda_compiled()); assert symmetrix.lsc_cuda_compiled()'
