---
name: symmetrix-lsc
description: Export LocalSplitCharges MACE-SCF models for the native Symmetrix LSC backend, validate predictions against ASE, and prepare or run LAMMPS simulations for new projects. Use for this LSC implementation and the existing Rusty environments, rather than MACE-POLAR, foundation MACE, or model training.
---

# Symmetrix LSC

Use `https://github.com/niamhon/symmetrix`, branch `symmetrix/lsc-mace`.
The exporter, native CPU/CUDA evaluator and LAMMPS pair belong to Symmetrix;
original MACE-SCF supplies the ASE reference. Inspect the actual branch, working
tree and loaded module: a directory named `symmetrix-dev` does not establish its
version.

Read [the project workflow](references/project-workflow.md) for a new model or
simulation. Read [the runtime reference](references/rusty-runtime.md) for Rusty
paths, environments, launchers and installation on another host. Reuse an
existing suitable setup; installing on a new host is a separate task.

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
- CPU uses FP64. GPU neural modes are FP32 and `float32-mixed`, with FP64
  electrostatics/reductions; GPU FP64 neural evaluation is not implemented.
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
