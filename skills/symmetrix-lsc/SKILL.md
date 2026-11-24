---
name: symmetrix-lsc
description: Export LocalSplitCharges MACE-SCF models for the native Symmetrix LSC backend, validate predictions against ASE, and prepare or run LAMMPS simulations using the validated clean Rusty installation. Use for this LSC implementation, rather than MACE-POLAR, foundation MACE, or model training.
---

# Symmetrix LSC

Use the validated clean Rusty release for new scientific projects, tag
`lsc-clean-20261124` on branch `symmetrix/lsc-clean` of
`https://github.com/niamhon/symmetrix`. Read its manifest before use; the physical
release is `/mnt/home/noneill/ceph/symmetrix-lsc-releases/releases/lsc-clean-20261124`.
The new-project wrapper is `/mnt/home/noneill/Programs/symmetrix-lsc/activate.sh`.
Legacy stable `2762b9f` remains separate for projects already pinned to it.

The exporter, native CPU/CUDA evaluator, ASE interface and LAMMPS Kokkos pair
belong to Symmetrix. Pinned original MACE-SCF supplies the independent PyTorch
reference. Development and dispersion experiments remain separate. Resolve the
release once per project and record its physical path. Preserve released
artifacts; do not rebuild a frozen release in place.

Read [the project workflow](references/project-workflow.md) for a new model or
simulation and [the clean runtime reference](references/clean-runtime.md) for
Rusty paths, the ASE API and LAMMPS launch settings.

## Essential decisions

- Establish the requested checkpoint, structure, type mapping, PBC, observables
  and simulation settings from the project. The previous water example's 330 K
  and 1 ns are not defaults for other projects.
- Determine whether added dispersion is needed from the user's instruction or
  model/training provenance. A `nod4` filename alone is insufficient. This LSC
  pair does not add D4. If the intended complete potential is unresolved,
  continue export/component validation while obtaining that specification
  before launching physical production MD. Existing explicit provenance or a
  user-provided complete model is sufficient; do not request repeated approval.
- Validate the actual new checkpoint. Frozen stage-2 water fixtures are
  development regressions, not predictions for every future model.
- Boundaries supported by the pair are finite clusters and fully periodic 3D
  cells. Mesh requires full 3D periodicity. Slabs, compensating-jellium variants
  and per-atom virials are unsupported. Preserve architecture validation;
  report unsupported models instead of silently changing them.
- Use FP32 (`float32`) neural evaluation for new GPU runs unless the user
  explicitly specifies another mode. Enable `float32-mixed` only on explicit
  request. CPU evaluation is fixed at FP64. GPU electrostatics and reductions
  use FP64 in both neural modes.
- Electrostatics is already included. Do not double-count it using a separate
  Coulomb pair or `kspace_style`. Required added dispersion is a distinct
  interaction needing a validated setup.
- Preserve prior runs and frozen references. Isolate validation output and
  start a replacement model from the intended configuration, not an unrelated
  model's partial checkpoints.

Complete execution already authorized by the user. Invoking this skill alone
does not authorize job submission, cancellation of other jobs, publication or
PRs. Diagnose a failure as model/export/binary/launcher/input related before
changing physics or tolerances; stop unchanged retries and report the blocker.

Record model/export hashes, code revisions, loaded modules/binaries, precision,
mesh settings and numerical checks. Distinguish submission/startup from a
completed trajectory or converged thermodynamic average.
