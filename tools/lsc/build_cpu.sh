#!/bin/bash -l
set -euo pipefail

module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
export PYTHONDONTWRITEBYTECODE=1
export CMAKE_BUILD_PARALLEL_LEVEL="${LSC_BUILD_JOBS:-4}"
export OMP_NUM_THREADS="$CMAKE_BUILD_PARALLEL_LEVEL"
export OPENBLAS_NUM_THREADS=1
cd /mnt/home/noneill/Programs/symmetrix-dev
python -m pip install --no-deps --no-build-isolation --editable ./symmetrix \
    -Ccmake.define.SYMMETRIX_KOKKOS=OFF \
    -Ccmake.define.SYMMETRIX_LSC_MESH=ON \
    -Ccmake.define.SYMMETRIX_SPHERICART_CUDA=OFF \
    -Ccmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
python -c 'import symmetrix; from importlib.metadata import version; print("SYMMETRIX_BUILD", version("symmetrix"), symmetrix.__file__)'
