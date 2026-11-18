#!/bin/bash -l
#SBATCH --job-name=lsc-lammps-build
#SBATCH --account=icc
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --constraint=rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/lammps-lsc-build-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/lammps-lsc-build-%j.err
set -euo pipefail
cd /mnt/home/noneill/Programs/symmetrix-dev
export LSC_SKIP_HELP=1
bash -l tools/lsc/build_lammps.sh cpu >.lsc-reference-logs/lammps-lsc-cpu-resume-${SLURM_JOB_ID}.log 2>&1
bash -l tools/lsc/build_lammps.sh cuda >.lsc-reference-logs/lammps-lsc-cuda-build-${SLURM_JOB_ID}.log 2>&1
