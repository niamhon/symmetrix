#!/bin/bash -l
#SBATCH --job-name=lsc-reference-tests
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/tests-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/tests-%j.err

set -euo pipefail
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
cd /mnt/home/noneill/Programs/symmetrix-dev
python -m pip check
python -m pytest symmetrix/test/lsc symmetrix/test/test_cubic_spline.py \
    -q --durations=5 \
    --junitxml=.lsc-reference-logs/pytest-${SLURM_JOB_ID}.xml
