#!/bin/bash -l
#SBATCH --job-name=lsc-cuda-derivatives
#SBATCH --account=icc
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --constraint=rocky9
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cuda-derivatives-gpu-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cuda-derivatives-gpu-%j.err

set -euo pipefail
cd /mnt/home/noneill/Programs/symmetrix-dev
bash -l tools/lsc/run_cuda_derivatives_tests.sh
