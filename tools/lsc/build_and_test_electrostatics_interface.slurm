#!/bin/bash -l
#SBATCH --job-name=lsc-electrostatics-hook
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
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/interface-gpu-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/interface-gpu-%j.err
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LSC_REQUIRE_CUDA=1 LSC_REQUIRE_LAMMPS=1
export LSC_LAMMPS_BINARY=/mnt/home/noneill/Programs/lammps-lsc-dev/build-cuda/lmp
export LSC_LAMMPS_LAUNCHER="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none"
cd "$repo"
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv
bash -l tools/lsc/build_cuda.sh > ".lsc-reference-logs/interface-gpu-build-${SLURM_JOB_ID}.log" 2>&1
python -m pytest symmetrix/test/lsc/test_electrostatics_interface.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=".lsc-reference-logs/interface-gpu-focused-${SLURM_JOB_ID}.xml" 2>&1 | tee ".lsc-reference-logs/interface-gpu-focused-${SLURM_JOB_ID}.log"
cmake --build /mnt/home/noneill/Programs/lammps-lsc-dev/build-cuda --parallel 4 > ".lsc-reference-logs/interface-lammps-gpu-${SLURM_JOB_ID}.log" 2>&1
python -m pip check
python -m pytest symmetrix/test/lsc symmetrix/test/test_cubic_spline.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=".lsc-reference-logs/interface-gpu-full-${SLURM_JOB_ID}.xml" 2>&1 | tee ".lsc-reference-logs/interface-gpu-full-${SLURM_JOB_ID}.log"
