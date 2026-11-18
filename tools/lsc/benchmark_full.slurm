#!/bin/bash -l
#SBATCH --job-name=lsc-full-benchmark
#SBATCH --account=icc
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --constraint=rocky9
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/benchmark-full-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/benchmark-full-%j.err
set -euo pipefail
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /mnt/home/noneill/Programs/symmetrix-dev
python tools/lsc/benchmark_full.py --repeats 5 --output ".lsc-reference-logs/benchmark-full-${SLURM_JOB_ID}.json"
