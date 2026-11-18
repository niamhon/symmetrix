# Stable runtime and development

## Stable default for new projects

Use `ssh -x rusty` and `scp -o ForwardX11=no`. The stable release is:

`/mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f`

It is pinned to Symmetrix commit
`2762b9fc10791220eef38edbe8e3fd70a9a1a124`, MACE-SCF commit
`4d1fe8a40481be0c73507f0761f5eaf2b1d32468`, and LAMMPS revision
`582a54bcaaec568889b5960793d318c1b863ffbc`. The native libraries are independent
copies of the tested artifacts, packaged with non-editable Python code. The
LAMMPS executable searches the release's own library directory. It still uses
the installed immutable Nix/system compiler, MPI and CUDA runtime dependencies.

Discover the promoted release with:

```bash
readlink -f /mnt/home/noneill/Programs/symmetrix-lsc-stable/current
```

Resolve that path once and write it literally into each project's scripts and
manifest. A queued project should not silently follow a later promotion.

For this release, activate:

```bash
source /mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f/activate.sh
```

The convenience entry point is
`/mnt/home/noneill/Programs/venvs/symmetrix-lsc-stable/activate-lsc-stable.sh`.
Use this wrapper or the release activation script, not just `venv/bin/activate`,
because the wrapper also selects the matching compiler/CUDA modules and library
paths. Activation removes development `PYTHONPATH` settings and selects:

- Python: `releases/2762b9f/venv/bin/python`, relative to the stable root above.
- LAMMPS: `releases/2762b9f/bin/lmp`.
- `LSC_RELEASE_ROOT`, `LSC_LAMMPS_BINARY`, and `LSC_LAMMPS_LAUNCHER`, all pinned to
  this physical release. `python_bin` also points to its interpreter.

The same environment supports CPU FP64 and CUDA FP32/mixed evaluation. The
LAMMPS binary supports both `symmetrix/mace/lsc` and `symmetrix/mace/lsc/cuda`;
select the pair for the requested hardware. It includes MOLECULE, EXTRA-FIX and
EXTRA-DUMP for full atom style, CSVR and DCD. Check other packages when needed.

The matching MPI executable is
`/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun`.
Use `--bind-to none -np 1` for the basic one-GPU setup. Python MPI subprocesses
need `stdin=DEVNULL` so they cannot consume an enclosing SSH heredoc.

Do not upgrade dependencies, replace CUDA, edit source files, or rebuild this
release in place for a new project. If another dependency is genuinely needed,
prepare an appropriate separate project environment while retaining the pinned
runtime, and verify actual module/library origins. Unexpected imports should be
diagnosed through `sys.prefix`, `symmetrix.__file__`, `mace_scf.__file__`, the
release manifest and `ldd`, rather than by adding development paths.

## One-H100 shell example

Adapt the project directory, input, requested runtime and resources. Export and
validate that project's model before production; this is only the launch shell.

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
source /mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f/activate.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS=1
cache="/tmp/noneill-lsc-${SLURM_JOB_ID}"
export CUDA_CACHE_PATH="$cache/cuda" PYTORCH_KERNEL_CACHE_PATH="$cache/kernels"
export TMPDIR="$cache/tmp" MPLCONFIGDIR="$cache/matplotlib"
mkdir -p "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR"
ulimit -c 0
MPI=/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
"$MPI" --bind-to none -np 1 "$LSC_LAMMPS_BINARY" -in in.lammps </dev/null
```

Submit when authorized. Inspect current allocations rather than reusing historic
job IDs, and leave unrelated jobs alone. Use login CPUs for small exports and
analysis. Run GPU checks on an allocated GPU; avoid competing workloads during
performance measurements.

## Ongoing development and future releases

Development remains in `/mnt/home/noneill/Programs/symmetrix-dev`, branch
`symmetrix/lsc-mace`, fork `https://github.com/niamhon/symmetrix`; upstream is
`https://github.com/wcwitt/symmetrix`. Development LAMMPS and its editable CPU/CUDA
environments remain under `Programs/lammps-lsc-dev` and
`Programs/venvs/symmetrix-lsc-dev` / `symmetrix-lsc-cuda-dev`. Use them for code
changes or explicit development tests, not default scientific runs.

Before promoting an optimization, preserve its source commit, run numerical
regressions and short MD, and compare performance against the previous stable
version. Assemble a new independent release directory, verify its imports and
ELF links, and validate it before changing `current`. Existing projects keep
their recorded physical release paths. No PR is needed for this local release
workflow.

For source rebuilding or another host, read `tools/lsc/README.md` and
`pair_symmetrix/pair_symmetrix_mace_lsc.rst` at the chosen commit. The Rusty
development helpers hardcode development paths; do not run them expecting to
build into a stable directory. The standard source installer uses a fresh
LAMMPS tree, MPI, KSPACE, FFTW3 double-precision FFTs and
`SYMMETRIX_LSC_MESH=ON`. CUDA follows the documented Kokkos/CUDA recipe and links
cudart, cuBLAS and cuFFT. Do not overlay it on a custom prebuilt integration.

Single-H100 execution is validated. SM80 code/PTX availability is not a measured
A100/H200 validation result. On a new host check driver/toolkit/architecture and
run model validation. Multi-rank communication is implemented; multi-GPU scaling
is a separate validation task.

The frozen stage-2 model is a regression fixture, not a default model for other
projects. Keep large models, JSON exports, trajectories, caches and binaries out
of source commits and retain their hashes and locations in run manifests.
