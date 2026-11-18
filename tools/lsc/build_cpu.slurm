#!/bin/bash -l
#SBATCH --job-name=lsc-dev-build
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/build-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/build-%j.err

set -euo pipefail
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
export PYTHONDONTWRITEBYTECODE=1
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$CMAKE_BUILD_PARALLEL_LEVEL"
export OPENBLAS_NUM_THREADS=1
cd /mnt/home/noneill/Programs/symmetrix-dev
python -m pip install --no-deps --no-build-isolation --editable ./symmetrix \
    -Ccmake.define.SYMMETRIX_KOKKOS=OFF \
    -Ccmake.define.SYMMETRIX_SPHERICART_CUDA=OFF \
    -Ccmake.define.CMAKE_CUDA_COMPILER=NOTFOUND
python -c 'import symmetrix; from importlib.metadata import version; print("SYMMETRIX_BUILD", version("symmetrix"), symmetrix.__file__)'
