#!/bin/bash -l
#SBATCH --job-name=lsc-cuda-sources
#SBATCH --account=icc
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --constraint=rocky9
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cuda-sources-gpu-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cuda-sources-gpu-%j.err

set -euo pipefail
cd /mnt/home/noneill/Programs/symmetrix-dev
if bash -l tools/lsc/build_cuda.sh >.lsc-reference-logs/cuda-sources-gpu-build-${SLURM_JOB_ID}.log 2>&1; then
    tail -n 12 .lsc-reference-logs/cuda-sources-gpu-build-${SLURM_JOB_ID}.log
else
    tail -n 100 .lsc-reference-logs/cuda-sources-gpu-build-${SLURM_JOB_ID}.log
    exit 1
fi
bash -l tools/lsc/run_cuda_sources_tests.sh
