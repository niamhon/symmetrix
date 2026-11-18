#!/bin/bash -l
#SBATCH --job-name=lsc-lammps-timing
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
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/benchmark-lammps-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/benchmark-lammps-%j.err
set -euo pipefail
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LSC_LAMMPS_LAUNCHER="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none"
cd /mnt/home/noneill/Programs/symmetrix-dev
python tools/lsc/benchmark_lammps.py --output ".lsc-reference-logs/benchmark-lammps-${SLURM_JOB_ID}.json"
