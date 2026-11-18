#!/bin/bash -l
# Run inside an allocated GPU step. On Rusty use the user's active H100 allocation:
# srun --jobid=JOBID --overlap -N1 -n1 -c4 --exact bash -l tools/lsc/run_gpu_mesh_tests.sh
set -euo pipefail
repo=/mnt/home/noneill/Programs/symmetrix-dev
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export LSC_REQUIRE_CUDA=1 LSC_REQUIRE_CPU_MESH=1 LSC_REQUIRE_LAMMPS=1
export LSC_LAMMPS_BINARY=/mnt/home/noneill/Programs/lammps-lsc-dev/build-cuda/lmp
export LSC_LAMMPS_LAUNCHER="/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun -np 1 --bind-to none"
cd "$repo"
# Hardware-matched old-code control, captured independently of the mesh solver.
# Original ASE and A100 reference files remain untouched.
if python -c 'import sys,torch; sys.exit(0 if "H100" in torch.cuda.get_device_name() else 1)'; then
    export LSC_CUDA_DIRECT_BASELINE="${LSC_CUDA_DIRECT_BASELINE:-$repo/.lsc-reference-logs/gpu-mesh-old-h100-baseline.npz}"
    export LSC_CUDA_DIRECT_REPEATABILITY="${LSC_CUDA_DIRECT_REPEATABILITY:-$repo/.lsc-reference-logs/gpu-mesh-old-h100-repeatability.json}"
fi
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv
if [[ "${LSC_REUSE_BUILD:-0}" != 1 ]]; then
    bash -l tools/lsc/build_cuda.sh > .lsc-reference-logs/gpu-mesh-final-build.log 2>&1
fi
LSC_SKIP_HELP=1 bash -l tools/lsc/build_lammps.sh cuda > .lsc-reference-logs/gpu-mesh-lammps-build.log 2>&1
python -m pip check
python -m pytest symmetrix/test/lsc/test_cuda_mesh_energy.py symmetrix/test/lsc/test_cuda_mesh_derivatives.py symmetrix/test/lsc/test_lammps_lsc_cuda_mesh.py symmetrix/test/lsc/test_cuda_mesh_api.py \
    -q -s --tb=short -o junit_family=xunit1 --junitxml=.lsc-reference-logs/gpu-mesh-focused.xml 2>&1 | tee .lsc-reference-logs/gpu-mesh-focused.log
python -m pytest symmetrix/test/lsc symmetrix/test/test_cubic_spline.py -q -s --tb=short -o junit_family=xunit1 \
    --junitxml=.lsc-reference-logs/gpu-mesh-full.xml 2>&1 | tee .lsc-reference-logs/gpu-mesh-full.log
