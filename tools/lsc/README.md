# LSC development and validation on Rusty

The working repositories are `~/Programs/symmetrix-dev` and the separate
`~/Programs/lammps-lsc-dev`. Original MACE-SCF and Polar source trees are references.
The test checkpoint is the frozen stage-2 water model in `.lsc-reference-data/`;
its export is `.lsc-reference-logs/water-stage2-energy.json`. Do not regenerate the
frozen ASE outputs while testing an implementation change.

## Existing development environments

CPU (login node is sufficient):

```bash
module load gcc/13.3.0
source ~/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
cd ~/Programs/symmetrix-dev
bash -l tools/lsc/build_cpu.sh
LSC_SKIP_HELP=1 bash -l tools/lsc/build_lammps.sh cpu
bash -l tools/lsc/build_forward_mpi_driver.sh cpu
```

CUDA (inside an allocated GPU job):

```bash
source ~/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
cd ~/Programs/symmetrix-dev
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUDA_CACHE_PATH=/tmp/$USER-lsc-${SLURM_JOB_ID}/cuda
export PYTORCH_KERNEL_CACHE_PATH=/tmp/$USER-lsc-${SLURM_JOB_ID}/kernels
export TMPDIR=/tmp/$USER-lsc-${SLURM_JOB_ID}/tmp
mkdir -p "$CUDA_CACHE_PATH" "$PYTORCH_KERNEL_CACHE_PATH" "$TMPDIR"
LSC_BUILD_JOBS=8 bash -l tools/lsc/build_cuda.sh
LSC_SKIP_HELP=1 LSC_BUILD_JOBS=8 bash -l tools/lsc/build_lammps.sh cuda
bash -l tools/lsc/build_forward_mpi_driver.sh cuda
```

The CUDA build uses the existing CUDA 12.8 toolkit, portable x86-64-v3 host code,
SM80 code/PTX and FP32 or mixed neural layers. Physical electrostatics and
reductions use FP64. H100 runtime validation does not establish validation on
A100/H200 or multiple GPUs.

The standard CUDA/Kokkos source build also provides
`symmetrix/mace/lsc/cuda/kk` for one MPI process, following Polar's device graph
and physical-output path. Launch with `-k on g 1 -sf kk -pk kokkos newton on
neigh half` and use `atom_modify map array`. It accepts the same direct/mesh,
FP32/mixed and external-field settings as the original CUDA pair. The original
pair remains the staged MPI path. The custom prebuilt LAMMPS integration below
does not enable Kokkos; use the standard source build for this new pair.

For full molecular atom storage (`atom_style full`) enable `PKG_MOLECULE=ON`.
The CSVR/DCD density protocol additionally requires `PKG_EXTRA-FIX=ON` and
`PKG_EXTRA-DUMP=ON`. These packages are enabled in the Rusty development source
build. The LSC Kokkos pair synchronizes the charge DualView before device writes,
so host exchange, sorting and restart readback can update its storage safely.
Charges remain predictions of the exported LSC model; this synchronization does
not make the previous LAMMPS charges an input to that model.

`test_lammps_lsc_kokkos_charges.py` covers forced host rebuilds for full/charge
atom styles, full-atom NPT/restart and the atomic style without charge storage,
in FP32 and mixed precision. Its optional original-density checks also require
`LSC_FULL_ATOM_DENSITY_CASE`, an isolated directory with the original input,
model/export, frozen native FP64 outputs and `validate_density.py`. The tracked
basic checks use the ordinary frozen stage2 fixture.

The device input API consumes owned node types, owned sender/receiver indices
and physical receiver-minus-sender image vectors. Callers finish input writes
before `prepare_graph_device`; the evaluator copies those arrays on-device and
retains the existing geometry validation. Its device force and charge pointers
are borrowed outputs valid until the next evaluation. No host `LSCGraph` or
host feature-exchange scans are needed on this route. Optional per-atom energy
partitioning is also performed on the GPU.

The helpers use the existing Rusty environment paths. `build_lammps.sh` integrates
a prebuilt native library, which makes iterative LSC work much quicker than
rebuilding all LAMMPS/Kokkos sources. Rebuild the MPI diagnostic driver after
changing the pair class layout or signatures.

## Standard source installation

The regular installer includes both LSC and Polar styles:

```bash
bash pair_symmetrix/install.sh /path/to/fresh/lammps
cmake -S /path/to/fresh/lammps/cmake -B /path/to/build \
  -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=ON -D BUILD_OMP=ON \
  -D PKG_KSPACE=ON -D FFT=FFTW3 -D FFT_SINGLE=OFF \
  -D SYMMETRIX_KOKKOS=OFF -D SYMMETRIX_LSC_MESH=ON
cmake --build /path/to/build -j 4
```

FFTW3 development headers/library are required for the optional CPU mesh backend.
For CUDA, use the Kokkos/CUDA build recipe in `pair_symmetrix/README.md`, enable
KSPACE and `SYMMETRIX_LSC_MESH`, and select the appropriate GPU architecture.
The public Symmetrix target propagates C++20, and the installer explicitly links
cudart, cuBLAS and cuFFT. Do not run the standard installer over the custom
prebuilt integration: use a fresh scratch LAMMPS tree.

## Correctness and timing

Run GPU tests through the allocated job, for example:

```bash
srun --jobid=JOB_ID --overlap --chdir=/tmp -N1 -n1 -c4 --exact bash -l RUN_SCRIPT
```

`RUN_SCRIPT` should activate the environment, export the node-local caches above,
and set `LSC_LAMMPS_BINARY`, `LSC_LAMMPS_LAUNCHER` (matching `mpirun -np 1
--bind-to none`), `LSC_FORWARD_MPI_DRIVER`, `LSC_REQUIRE_CUDA=1` and
`LSC_REQUIRE_LAMMPS=1`. Then run:

```bash
python -m pytest -q -s symmetrix/test/lsc symmetrix/test/test_cubic_spline.py
```

Strict old-CUDA preservation checks also accept `LSC_STAGED_CUDA_BASELINE`,
`LSC_CUDA_DIRECT_BASELINE`, and `LSC_CUDA_DIRECT_REPEATABILITY`. Use independent
old-code controls recorded on the same GPU family; do not replace the frozen
native ASE references or widen tolerances to absorb a regression. Summation order
in FP32 reverse atomics can vary, while forward values have stricter gates.

`benchmark_h100.py` compares full ASE and prepared-topology numerical evaluation
with native MACE-SCF. `benchmark_h100_lammps.py` measures actual LAMMPS loops,
separately recording neighbor reuse and forced rebuilds. Model loading, run-0
validation and warm-up are excluded from loop throughput. A calculator benchmark
and a LAMMPS benchmark are distinct workflows.

`validate_water_md.py --output /path/to/results` runs the 1,536-atom water box at
0.5 fs: 1 ps NVT at 300 K, 5 ps NVE, then 5 ps NPT at 300 K and 1 bar, in both
neural precisions. It checks restart continuation and native FP64 snapshot E/F/
stress/charge/dipole errors. Potential-only pressure is used for comparison with
ASE potential stress; thermostat/barostat pressure still includes kinetic terms.
Short trajectories do not establish converged thermodynamic averages.

The retained single-rank optimization skips H1/H2 MPI transport only when all
periodic ghost features have already been published locally. Multi-rank
communication is unchanged. Candidate gradient gathers were accuracy-tested but
rejected after timing. The retained CUDA pair uses a Polar-style receiver-batched
second-layer reverse kernel: it caches message adjoints per receiver and stages
edge geometry in shared memory, while preserving the exported CG coefficients
and the four-warp edge reduction. Sender-feature derivatives still use atomics.
The native Python class defaults to the legacy path for comparison; set
`model.use_batched_reverse = True` to select the same kernel as the CUDA pair.
`batched_reverse_calls` confirms that this path was exercised. Forward energy,
charges and dipoles are identical; force accumulation may differ by FP32
roundoff and is checked at the existing native FP64 tolerances.

## Clean source installation

The standard `pair_symmetrix/install.sh /path/to/lammps` links the pair sources
and adds libsymmetrix to LAMMPS's CMake build. Enable `PKG_KSPACE=ON`,
`SYMMETRIX_LSC_MESH=ON`, `FFT=FFTW3`, `FFT_SINGLE=OFF`, `BUILD_MPI=ON` and
`BUILD_OMP=ON`. The CPU-only build may use `SYMMETRIX_KOKKOS=OFF`. C++20 now
propagates through the library target.

For a CUDA/Kokkos build, additionally set `PKG_KOKKOS=ON`,
`Kokkos_ENABLE_CUDA=ON`, `Kokkos_ARCH_AMPERE80=ON`, and
`CMAKE_CUDA_ARCHITECTURES=80`, and use Kokkos's `nvcc_wrapper` as the C++ compiler.
The isolated clean-build test disables unrelated default KokkosKernels explicit
instantiations with `KokkosKernels_ADD_DEFAULT_ETI=OFF`; all tested LSC and Polar
CUDA kernels are unchanged. The exact configure/build launchers and logs are in
`.lsc-reference-logs/production-20261118/source-install/`.

`compare_polar_lammps.py` compares both actual pair styles on identical 192- and
1,536-atom starting boxes in one binary. It validates each against its own native
FP64 model and records loading/warm-up exclusions and model/precision settings.
`benchmark_optimization.py` compares the archived original LSC library and the
retained optimization using the same equilibrated 300 K restart and 0.5 fs step.

With `atom_style charge`, the CUDA pair updates `q` through the existing
physical-only charge readback. It downloads local energy/density-gradient data
when `pe/atom` is requested; obtaining the density gradient currently retains
the full force-diagnostic readback on that path. Neural feature diagnostics remain
available through the native API for debugging and tests. Ordinary MD with global
energy and virial output avoids those dumps.
