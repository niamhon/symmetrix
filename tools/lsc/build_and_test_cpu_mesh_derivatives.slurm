#!/bin/bash -l
#SBATCH --job-name=lsc-cpu-mesh-derivatives-regress
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
#SBATCH --output=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cpu-mesh-derivatives-gpu-%j.out
#SBATCH --error=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/cpu-mesh-derivatives-gpu-%j.err
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LSC_REQUIRE_CPU_MESH=1 LSC_REQUIRE_CUDA=1 LSC_REQUIRE_LAMMPS=1
export LSC_LAMMPS_BINARY=/mnt/home/noneill/Programs/lammps-lsc-dev/build-cuda/lmp
export LSC_LAMMPS_LAUNCHER="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none"
cd "$repo"
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ "${LSC_REUSE_BUILD:-0}" != 1 ]]; then
bash -l tools/lsc/build_cuda.sh > ".lsc-reference-logs/cpu-mesh-derivatives-gpu-build-${SLURM_JOB_ID}.log" 2>&1
fi
python -m pytest symmetrix/test/lsc/test_cpu_mesh_energy.py symmetrix/test/lsc/test_cpu_mesh_derivatives.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=".lsc-reference-logs/cpu-mesh-derivatives-gpu-focused-${SLURM_JOB_ID}.xml" 2>&1 | tee ".lsc-reference-logs/cpu-mesh-derivatives-gpu-focused-${SLURM_JOB_ID}.log"
LSC_SKIP_HELP=1 bash -l tools/lsc/build_lammps.sh cuda > ".lsc-reference-logs/cpu-mesh-derivatives-lammps-gpu-${SLURM_JOB_ID}.log" 2>&1
python -m pip check
python -m pytest symmetrix/test/lsc symmetrix/test/test_cubic_spline.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=".lsc-reference-logs/cpu-mesh-derivatives-gpu-full-${SLURM_JOB_ID}.xml" 2>&1 | tee ".lsc-reference-logs/cpu-mesh-derivatives-gpu-full-${SLURM_JOB_ID}.log"
