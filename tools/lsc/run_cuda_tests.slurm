#!/bin/bash -l
#SBATCH --job-name=lsc-cuda-loader
#SBATCH --account=icc
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --constraint=rocky9
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cuda-loader-gpu-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cuda-loader-gpu-%j.err

set -euo pipefail
bash -l /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/run_cuda_tests.sh
