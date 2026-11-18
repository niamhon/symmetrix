#!/bin/bash -l
#SBATCH --job-name=lsc-save-reference
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/reference-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/reference-%j.err

set -euo pipefail
source /mnt/home/noneill/Programs/venvs/mace-scf/activate-reference.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
cd /mnt/home/noneill/Programs/symmetrix-dev
python -u -B tools/lsc/generate_reference.py --threads "$OMP_NUM_THREADS"
