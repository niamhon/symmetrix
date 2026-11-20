# LSC tester checkpoint: lsc-validated-20261120

This named checkpoint contains the tested compact-first-layer LSC evaluator,
its Kokkos charge-synchronization fix, and the snapshot/NVE validation files.
The evaluator is unchanged from `b5238fee78b01cedddc7bd151dec1750bebe5624`.
It is a tester checkpoint of the fork, not a promotion of the separate Rusty
stable installation. Later development does not move this tag.

## Get the source

```bash
git clone --recursive --branch lsc-validated-20261120 \
  https://github.com/niamhon/symmetrix.git symmetrix-lsc
cd symmetrix-lsc
```

Build the Python package and LAMMPS against this same checkout. See the
[Python instructions](../../symmetrix/README.md),
[LAMMPS instructions](../../pair_symmetrix/README.md) and the
[LSC source-build notes](README.md#standard-source-installation).
The `tools/lsc` build/runner helpers use the existing Rusty setup; adapt their
paths before using another host. No compiled binaries, environments, large
models, JSON exports or trajectories are included in Git.

Loading an LSC Torch checkpoint for conversion requires compatible MACE-SCF
and graph-longrange dependencies in addition to the usual MACE dependencies.
The original native reference was MACE-SCF commit
`4d1fe8a40481be0c73507f0761f5eaf2b1d32468`, with mace-scf 0.4.4,
mace-torch 0.3.14, graph-longrange 0.4.3 and e3nn 0.4.4.
The Rusty checks used GCC 13.3, CUDA 12.8 and an H100 PCIe.
These are recorded reference versions; other versions, GPUs and compilers need
validation. Runtime LAMMPS uses the native evaluator.

## Convert a model

The same command as Polar dispatches automatically to the LSC exporter:

```bash
symmetrix_extract_mace --model model.model --chemical-symbols H O \
  --num-spline-points 4096 --r-min 0.5 --output model.symmetrix.json
```

Replace the elements with those needed for the model. The exporter checks the
supported single-head, two-layer, 128-channel scalar/vector LSC architecture;
it does not support every architecture that MACE-SCF can train. Unsupported
architectures fail explicitly. Do not use `--compress-basis` for LSC: compact
GPU evaluation is selected internally. Export and run with the same Symmetrix
checkout, and retain model/export hashes.

## Run one GPU

A minimal fully periodic input, with type 1 H and type 2 O, is:

```lammps
units metal
atom_style charge
boundary p p p
atom_modify map yes
newton on
read_data system.data
pair_style symmetrix/mace/lsc/cuda mesh 0.7 order 8
pair_coeff * * model.symmetrix.json H O
```

Use the CUDA/Kokkos build with:

```bash
mpirun -np 1 lmp -k on g 1 -sf kk \
  -pk kokkos newton on neigh half -in in.lammps
```

Append `float32-mixed` to `pair_style` for mixed neural arithmetic. The GPU
default is FP32; electrostatics and physical reductions use FP64 in both modes.
Use `symmetrix/mace/lsc` for CPU FP64. Nonperiodic clusters use `direct` and
nonperiodic boundaries. Direct is the current LSC default; specify `mesh`
explicitly for periodic MD. Mesh spacing 0.7 Angstrom/order 8 is the tested
setting, not an accuracy guarantee for every new model.

The fastest `/cuda/kk` path is currently one MPI process. Multi-rank mesh MD
uses the ordinary `/cuda` pair without `-sf kk`; independent multi-GPU scaling
has not been established. This checkpoint has no Polar-style `grid`, `charge`
or `spin` override. Charge follows the exported per-species formal charges and
conserved learned transfers. Optional `field Ex Ey Ez` retains the native
`+ field dot dipole` convention.

Electrostatics is included: do not add another Coulomb pair or `kspace_style`.
This pair adds no separate D3/D4 term; establish the intended complete potential
from the checkpoint's training provenance. Global stress and per-atom energies
are supported; per-atom stress and partially periodic slabs are not. Reissue
`pair_style` and `pair_coeff` after `read_restart`. Full details are in the
[LSC pair reference](../../pair_symmetrix/pair_symmetrix_mace_lsc.rst).

## Validation evidence and limits

The unchanged native `MACELocalSplitCharges` PyTorch/ASE calculator, FP64 and
uncompiled, supplies the reference. All Symmetrix comparisons below use actual
LAMMPS. Frozen predictions were saved before comparison.

- Snapshot pilot: 45 benzene/EMC configurations (clusters, isolated atoms and
  periodic cells), each in FP32 and mixed, giving 90 LAMMPS comparisons. The
  final 186 distinct passing checks include water regressions, coordinate and
  species conventions, conservation, CPU FP64 spot checks and native finite
  differences. The first run passed 182; four cluster-rotation assertions were
  corrected after independently diagnosing the native finite Cartesian dipole
  stencil, then those four passed. No parity gate or production kernel changed.
- NVE screen: 36 trajectories, three liquids/backends, timesteps 0.125, 0.25,
  0.5 and 1 fs, 1 ps each. All 49 saved-output checks passed. Maximum matched
  native/Symmetrix RMS-energy difference was 0.1373%; fluctuations scaled
  approximately with timestep squared.
- Water extension: three 10 ps trajectories at 0.5 fs. Energy RMS values were
  0.0158782, 0.0154968 and 0.0159690 meV/atom for native, FP32 and mixed.
  The FP32 fitted energy slope was larger than native over this window; one
  independently evolving trajectory per backend does not establish a persistent
  bias or a general absence of secular drift. Benzene/EMC retain 1 ps windows.

The publication recheck passed all 139 selected saved-result tests: the 90
organic snapshot comparisons and 49 NVE checks. This recheck ran on the login
CPU and did not launch new GPU trajectories. Syntax and frozen-fixture hashes
were also checked.

These check implementation parity and finite-window conservation, not DFT
accuracy, universal architecture support or converged thermodynamics.
[ORGANIC_VALIDATION.md](ORGANIC_VALIDATION.md) and
[NVE_VALIDATION.md](NVE_VALIDATION.md) give the methods and unchanged gates.
The four small organic fixture/reference files and their manifest are versioned
under `symmetrix/test/lsc/data/organic_stage3/`. Model files and full generated
results remain external. The opt-in historical runners retain explicit Rusty
paths and exact binary hashes; a collaborator building on another host should
validate their own checkpoint against the native calculator.

Original artifacts on Rusty/Ceph:

- `/mnt/home/noneill/ceph/symmetrix-lsc-validation/organic-stage3`
- `/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120`

The original LAMMPS executable SHA-256 is
`8b0fbe2e4224e2d3e910fbf2aa95c6b7b50eede38096d847ac8a872c3726197a`.
The JSON provenance manifest records model, export, data and source hashes.
Validation-only descendant commits are allowed by the runners only while the
implementation/build files remain identical to `b5238fe`; binary checks remain
exact. The publication commit does not rebuild or modify the stable release.
