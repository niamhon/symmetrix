# Clean Rusty runtime

The clean LSC release is named `lsc-clean-20261124` in
`https://github.com/niamhon/symmetrix`, branch `symmetrix/lsc-clean`. Its physical
runtime is:

`/mnt/home/noneill/ceph/symmetrix-lsc-releases/releases/lsc-clean-20261124`

After validation, the convenience entry point is:

```bash
source /mnt/home/noneill/Programs/symmetrix-lsc/activate.sh
```

For each new project, resolve
`/mnt/home/noneill/Programs/symmetrix-lsc/current` once, check that its manifest
says `validated-clean-release`, and record the physical release path in scripts
and provenance. Do not redirect an existing project's pinned path automatically.

The activation selects GCC 13.3.0/CUDA 12.8.0, clears development Python paths,
and defines `LSC_RELEASE_ROOT`, `LSC_LAMMPS_BINARY`, `LSC_LAMMPS_LAUNCHER` and
`python_bin`. Python is the release's `venv/bin/python`; LAMMPS is its `bin/lmp`.
This is a noneditable package and a frozen standalone LAMMPS executable, using
Rusty's immutable Nix/system MPI/CUDA/compiler libraries. No development build
directory or development shared library is required at runtime.

The release combines the clean LSC + ASE source with the upstream Polar GPU ASE
interface. Its LAMMPS executable reuses the validated compact-plus-charge-sync
implementation. Source commits, per-artifact build revisions, wheel and binary
hashes, validation results and dependency origins are in `manifest.json`.
Package version `0.3.0` alone is not an implementation identifier.

## LSC ASE API

```python
from symmetrix.experimental import MACELSC

atoms.info["external_field"] = [0.0, 0.0, 0.0]
fast = MACELSC(
    str(export_path), device="cuda", dtype="float32",
    electrostatics="mesh", mesh_spacing=0.7, mesh_order=8,
)
atoms.calc = fast
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
charges = atoms.get_charges()
dipole = atoms.get_dipole_moment()
```

This ASE path uses native CUDA, not a LAMMPS Kokkos pair. It selects the optimized
LSC path; do not add legacy diagnostic flags. CPU uses `device="cpu"` and
`dtype="float64"`. Mixed GPU mode is `dtype="float32-mixed"`, only when
explicitly requested. Fully open clusters use `electrostatics="direct"` and
omit mesh options. Partial periodicity is unsupported. Field/charge/dipole and
periodic-image conventions must match the native reference.

## LSC LAMMPS API

```lammps
units metal
atom_style charge
atom_modify map array
boundary p p p
read_data initial.data
pair_style symmetrix/mace/lsc/cuda/kk mesh 0.7 order 8
pair_coeff * * model.symmetrix.json O H
```

`O H` means atom type 1 is O and type 2 is H; read the actual input's mapping.
GPU launch options are:

```bash
MPI=/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun
"$MPI" --bind-to none -np 1 "$LSC_LAMMPS_BINARY" \
  -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps </dev/null
```

Use `newton pair on`, atom IDs and `units metal`. CPU uses
`symmetrix/mace/lsc` without the Kokkos/GPU launch options and is FP64. The
ordinary CUDA pair is retained, but the optimized LAMMPS route is `/cuda/kk`.
FP32 is default; add `float32-mixed` to the pair settings only on explicit
request. Direct electrostatics requires one process; mesh requires full 3D
periodicity. Multi-rank infrastructure exists; production multi-GPU scaling
needs its own validation.

Electrostatics is included; do not double-count it with a Coulomb pair or
`kspace_style`. No D3/D4 correction is supplied by this clean branch. Preserve
the user's actual complete-potential specification. Predicted charges are model
outputs. Global stress/per-atom energies are supported, per-atom virials are
not. Use `compute mu all pair symmetrix/mace/lsc/cuda/kk` for the total dipole.
The optional `field Ex Ey Ez` uses eV/(e Angstrom), with energy convention
`+ E dot mu`. Reissue pair settings and coefficients after `read_restart`.

For NPT, retain kinetic pressure in the barostat. Potential-only validation
uses `compute vir all pressure NULL virial`. Ordinary `fix npt` is distinct from
NPH plus CSVR; do not combine two temperature controllers unintentionally.

## Launching a project

Use `ssh -x rusty` / `scp -o ForwardX11=no`. Small exports and analysis can use
login CPUs; GPU checks require an allocated GPU. Inspect current allocations
and avoid unrelated jobs. One H100 is the measured reference hardware; SM80/PTX
compatibility alone is not proof of A100/H200 validation. Check the release's
recorded CPU-host tests before assuming CPU ISA portability.

Use the project's requested temperature, pressure, timestep and resources. A
typical one-GPU Slurm script should source the resolved release `activate.sh`,
set its CPU thread counts, use per-job caches under `/tmp`, print actual GPU
identity, and invoke the pinned binary with the options above. Keep simulation
files and large artifacts on Ceph. Python MPI subprocesses need `stdin=DEVNULL`.

## Separate code and installations

- Native reference/training remains `Programs/mace-scf-dev-niamh` at `4d1fe8a`
  with `Programs/venvs/mace-scf`; the original PyTorch calculator is unchanged.
- Development remains `Programs/symmetrix-dev` and its existing CPU/CUDA
  environments. Its current checkout contains dispersion work and is not the
  default for new scientific projects.
- Legacy stable `Programs/symmetrix-lsc-stable/releases/2762b9f` is preserved;
  projects already pinned to it keep that version. Its old wrapper does not
  identify the clean release.
- i-PI/batching and FourierD3 are separate application/feature installations.
  Preserve their jobs and manifests. Future LSD dispersion work starts from the
  clean release tag, not from an unrecorded dirty source tree.

For another host, use the chosen tag's main README, `tools/lsc/CLEAN_RELEASE.md`
and LSC pair documentation, initializing pinned submodules and building native
artifacts for that host. Rusty development helper scripts contain Rusty paths.
Never rebuild this frozen release in place or assume a Rusty wheel is portable
to a different Python/CUDA/MPI stack.
