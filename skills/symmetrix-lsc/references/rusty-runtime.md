# Runtime and installation

## Rusty setup

Use `ssh -x rusty` and `scp -o ForwardX11=no`. The authoritative checkout is
`/mnt/home/noneill/Programs/symmetrix-dev`, fork
`https://github.com/niamhon/symmetrix`, branch `symmetrix/lsc-mace`; upstream is
`https://github.com/wcwitt/symmetrix`. LAMMPS is in
`/mnt/home/noneill/Programs/lammps-lsc-dev`. Inspect actual revisions and status.

Reference MACE-SCF: `/mnt/home/noneill/Programs/mace-scf-dev-niamh`, branch
`niamh-dev`, baseline `4d1fe8a40481be0c73507f0761f5eaf2b1d32468`. Ordinary model
export/run tasks do not need edits to this reference source.

CPU activation:

```bash
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
```

CUDA activation (also suitable for CPU-only export):

```bash
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
```

Use the activated Python; check `symmetrix.__file__` and `mace_scf.__file__` if
imports are unexpected. Do not upgrade dependencies/CUDA for an ordinary run.

Binaries are `/mnt/home/noneill/Programs/lammps-lsc-dev/build-cpu/lmp` and
`/mnt/home/noneill/Programs/lammps-lsc-dev/build-cuda/lmp`. The CUDA binary includes
MOLECULE, EXTRA-FIX and EXTRA-DUMP for full atom style, CSVR and DCD. Check other
required packages. The validated launcher is
`/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun`.
Use `--bind-to none -np 1` inside a one-GPU job. Verify paths and MPI compatibility
if the installation changes. Python MPI subprocesses need `stdin=DEVNULL` so
they cannot consume an enclosing SSH heredoc.

## One-H100 shell example

Choose the project path, input and wall time for the actual task. Prepare and
validate the model separately before using this execution shell.

```bash
#!/bin/bash -l
#SBATCH --job-name=lsc-project
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/absolute/project/run
#SBATCH --output=lsc-%j.out
#SBATCH --error=lsc-%j.err
set -euo pipefail
module purge
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cache="/tmp/noneill-lsc-${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$cache/cuda" PYTORCH_KERNEL_CACHE_PATH="$cache/kernels"
export TMPDIR="$cache/tmp" MPLCONFIGDIR="$cache/matplotlib"
mkdir -p "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR"
ulimit -c 0
EXEC=/mnt/home/noneill/Programs/lammps-lsc-dev/build-cuda/lmp
MPI=/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
"$MPI" --bind-to none -np 1 "$EXEC" -in in.lammps </dev/null
```

Submit when authorized. Inspect current allocations instead of reusing historical
job IDs; leave unrelated jobs alone. Use allocated GPUs for GPU checks and login
CPUs for small exports/analysis. Avoid competing GPU workloads during benchmarks.

## Rebuilding or moving hosts

Read the checkout's `tools/lsc/README.md` for current build commands and
`pair_symmetrix/pair_symmetrix_mace_lsc.rst` for capabilities. Rusty helpers are
`tools/lsc/build_cpu.sh`, `build_cuda.sh` and `build_lammps.sh cpu|cuda`. The custom
LAMMPS development build links the native library through
`tools/lsc/lammps_prebuilt.cmake`.

For a standard source install use `pair_symmetrix/install.sh /path/to/fresh/lammps`.
Enable MPI, KSPACE, FFTW3 double-precision FFTs and `SYMMETRIX_LSC_MESH=ON`.
CPU-only builds can use `SYMMETRIX_KOKKOS=OFF`; CUDA builds follow the repository's
Kokkos/CUDA recipe and link cudart/cuBLAS/cuFFT. Use a fresh LAMMPS tree rather
than layering this installer over the custom prebuilt integration. Record the
LAMMPS revision: the validated baseline is
`582a54bcaaec568889b5960793d318c1b863ffbc`.

Single-H100 execution is validated. Existing builds contain SM80 code/PTX;
H100 use does not establish A100/H200 measurements. On another host check the
driver/toolkit/target architecture and run model validation. Multi-rank MPI is
implemented; multi-GPU scaling remains a separate validation task. Start from
one GPU/one MPI process unless the request requires otherwise.

Development regression commands are in `tools/lsc/README.md` and
`symmetrix/test/lsc/README.md`. Their frozen stage-2 model is external to Git;
fixtures carry its checksum. Do not replace it with new-project weights. Keep
large checkpoints, exports, trajectories, caches and build directories outside
source commits; record their hashes and locations in run manifests.
