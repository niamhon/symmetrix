# Matched NVE timestep screen

`nve_timestep.py` evaluates the existing stage-3 water, benzene and EMC models
without changing the evaluator or a build. It uses the pinned compact-plus-charge-
synchronization development binary, not the stable release. All generated models,
states, trajectories and reports live on Ceph under
`/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120`.

## Protocol

* Original `water_init.pdb`, `benzene_init.extxyz` and `emc_init.extxyz` come from
  the corresponding model projects' `simulations/nvt/1` folders. Coordinates,
  cells and masses are retained. The immutable manifest records paths and hashes.
* Water has 384 atoms and starts at 330 K; benzene has 996 atoms and EMC 990 atoms,
  both starting at 300 K. For the organics, use the saved momenta. For water,
  generate Maxwell momenta with seed 20261120. Remove center-of-mass momentum and
  rescale once to `(3N-3) kB T / 2`, then freeze the state. These are initial states,
  not an additional equilibration or a claim about an equilibrium temperature.
* Restart the identical state for each backend and timestep. Use 1 ps per run,
  timesteps 0.125, 0.25, 0.5 and 1.0 fs, fixed fully periodic cells, no constraints,
  thermostat or barostat. This is 36 independent trajectories, not one trajectory
  that changes timestep halfway through.
* Reference: unchanged `MACELocalSplitCharges`, native PyTorch CUDA FP64,
  `use_compile=False`, `pbc_handling="mixed_periodic"`, ASE `VelocityVerlet`.
* Symmetrix: actual LAMMPS `fix nve`, `symmetrix/mace/lsc/cuda/kk`, one MPI rank
  and one H100, FP32 or `float32-mixed`, mesh spacing 0.7 Å and order 8.
  These precision names describe the neural kernels; electrostatics and physical
  reductions retain FP64 in both modes.
* Use the complete supplied models without an additional D4 term, matching their
  original native simulation scripts. This test measures the existing potential;
  it does not compare a model with a different dispersion correction.

LAMMPS `metal` and ASE use slightly different physical conversion constants.
Match kinetic energies and Verlet updates using the actual LAMMPS `mvv2e`:
`v_metal = v_ASE / sqrt(mvv2e)` and
`dt_metal = dt_fs * ase.units.fs * sqrt(mvv2e)`. Explicitly assign the same atomic
masses. Report time in the requested fs convention. Initial kinetic energies must
agree within `1e-10 eV/atom`. Initial potential/force predictions retain the
existing parity gates (`1e-6 eV/atom`; force `atol=8e-5, rtol=1e-4`).

## Execution and recovery

Activate the existing development CUDA environment and set `LSC_NVE_ROOT`,
`LSC_LAMMPS_BINARY` and `LSC_LAMMPS_LAUNCHER` as in the project run record.
`prepare` freezes inputs once; `run` executes trajectories sequentially on one
allocated GPU; `analyse` regenerates tables; `audit` verifies preserved artifacts.
`plot_nve_timestep.py` creates scientific PNG/PDF figures from completed results.
The expensive NVE run is never launched implicitly by pytest.

Each trajectory writes its energy CSV while running. Complete trajectories have
a `summary.json`, compressed series and final state/dump. A restart reuses
completed trajectories and writes a new attempt for incomplete/failed runs;
failures and their logs remain available. A nonfinite state or a total-energy
departure exceeding 1 eV/atom is a runaway guard, not an accuracy threshold.

## Interpretation

Log full-step potential, kinetic and total energy at a common 1 fs cadence.
Report signed OLS total-energy slope in meV/atom/ps, centered RMS, peak-to-peak
span, maximum departure from the initial value and final departure. Also report
a fit to ten 0.1 ps block means with a nominal Student-t interval. Ten blocks may
remain correlated; that interval is indicative, not a reliable independent-
sample uncertainty estimate or proof of zero secular drift.

Velocity Verlet can conserve an approximate shadow Hamiltonian while the actual
total energy oscillates. On a 1 ps window, these bounded oscillations can produce
nonzero fitted slopes. Compare energy traces and timestep dependence alongside
slopes. The strongest implementation check is agreement between native and both
Symmetrix modes at matched timesteps, followed by smaller fluctuations as the
timestep decreases. A short screen cannot establish long-time stability,
transport properties, or a converged density. Do not loosen tolerances or alter
kernels in response to a failed run; preserve and diagnose the discrepancy.

## Saved-output checks and longer water run

After completing the sweep, recheck the saved results without starting MD:

```bash
LSC_RUN_NVE=1 python -m pytest symmetrix/test/lsc/test_nve_timestep.py -q
```

`nve_long_water.py` explicitly runs a separate 10 ps water extension at 0.5 fs
for all three backends. It is not invoked by pytest. Its historical results
remain under `long-water-10ps/`; benzene and EMC retain 1 ps windows only.

Implementation identity is checked against `b5238fe` even when the checkout
contains later validation/documentation commits. The frozen binary, models,
exports, states, native source and stable installation keep their independent
hash checks. Existing archived manifests and audit results are not rewritten
when publishing the named checkpoint. These scripts retain Rusty input paths
and require the corresponding external artifacts.
