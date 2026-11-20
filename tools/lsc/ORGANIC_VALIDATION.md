# Frozen benzene/EMC snapshot pilot

The pilot compares the unchanged native PyTorch/ASE `MACELocalSplitCharges`
calculator (FP64, uncompiled) with the development LAMMPS Kokkos pair in FP32
and mixed precision. It is model-component parity, not a comparison with DFT
labels. No dispersion correction, kernel change, retraining or new build is
introduced. The binary is pinned to the selected compact-plus-charge-sync
baseline `b5238fe` and checked by SHA-256.

`tools/lsc/organic_pilot.py` has separate prepare, reference, compare and report
stages. Never generate expected predictions inside pytest. All model/export
files and run artifacts live under `LSC_ORGANIC_DATA` (default:
`/mnt/home/noneill/ceph/symmetrix-lsc-validation/organic-stage3`). Small frozen
geometries, predictions and their manifest are also copied into
`symmetrix/test/lsc/data/organic_stage3` for version control. Existing references
are never overwritten. Use a new data root for a different model selection.

Models are seed-1 stage-3 checkpoints from the user's benzene and EMC gen2
e100_force100/55a experiments. For each, select ten molecular clusters at
atom-count quantiles and ten periodic structures at volume quantiles, retaining
the original source index as a tie-breaker. Add the two/three isolated-atom
entries, giving 22 benzene and 23 EMC cases. Positions and cells round-trip
float64 exactly; the normal extxyz writer's eight-decimal formatting is not
used. Checkpoints, datasets, exports, fixtures, native predictions, the original
MACE-SCF commit, versions, baseline executable and implementation sources are
hashed in the manifest.

In the existing development CUDA environment, after setting the usual module
and temporary-cache paths:

```bash
python -u tools/lsc/organic_pilot.py prepare   # CPU; run once
python -u tools/lsc/organic_pilot.py reference # allocated GPU; run once
python -u tools/lsc/organic_pilot.py compare   # allocated GPU
python -u tools/lsc/organic_pilot.py report
export LSC_RUN_ORGANIC=1
export LSC_ORGANIC_REUSE_RESULTS=1
python -m pytest symmetrix/test/lsc/test_organic_pilot.py -q \
    --basetemp="$LSC_ORGANIC_DATA/checks/pytest" \
    --junitxml="$LSC_ORGANIC_DATA/results/pytest-organic.xml"
```

Set `LSC_LAMMPS_BINARY` to the pinned executable and `LSC_LAMMPS_LAUNCHER` to
the matching Open MPI command, `mpirun --bind-to none -np 1`. Keep subprocess
stdin disconnected. Use one MPI rank and GPU. Reference generation and pytest
must be inside the allocation; prepare may run on the login CPU. Kernels and
the stable installation are unchanged. This is not a timing benchmark.

The compare stage runs 90 real LAMMPS evaluations and saves all raw outputs
before producing the error table. `LSC_ORGANIC_REUSE_RESULTS=1` lets the snapshot
pytest cases independently compare these already-saved raw arrays to frozen
native predictions without launching the same 90 evaluations again. Without
that switch, snapshot tests run LAMMPS afresh. Explicit opt-in with missing
inputs fails rather than skipping; without opt-in the module is skipped.

Energy error is absolute eV/atom. Force/stress/charge/dipole summaries include
both maximum component error and component RMS error. Forces/stress/charges use
`atol + rtol * abs(reference)` componentwise. The existing gates are retained:
energy 1e-6 eV/atom; forces 8e-5/1e-4; stress 5e-7/1e-4; charges 2e-6/1e-5;
total dipole absolute 3e-5 eA. Stress is potential-only, converted from LAMMPS
bar and mapped to ASE sign/Voigt conventions. There is no meaningful cluster
stress comparison.

LAMMPS restricted-triclinic rotations are applied to vectors and stress tensors
on output. Dipole errors use the same polarization branch as native ASE: record
the raw output and subtract only the exactly known formal-charge image shift
`sum(q_formal * (x_lammps - x_reference))`. Do not substitute predicted partial
charges in this correction. Cluster coordinates are not translated.

The remaining tests exercise translation, proper rotation (including the
periodic cell), atom permutation, reversed species numbering, wrapping, charge
conservation, CPU FP64 export spot checks, and independent native energy finite
differences of cluster forces and all six periodic stress components at two
step sizes. The native transformed predictions are saved before evaluating
LAMMPS; their own symmetry relations are checked independently. Physical
errors are never hidden by widening gates. Failure inputs and logs are retained.

The native real-space backend represents dipoles with charges displaced by
0.02 A along fixed Cartesian axes. Exact rotational invariance of the total
cluster energy/forces is therefore not a valid assertion for these references.
Cluster rotation tests retain all original LAMMPS/native gates, check scalar
charges and vector dipoles, reconstruct the native fixed-density electrostatic
energy independently in FP64, and require the remaining local-energy change to
vanish. Native total-energy and force anisotropy are recorded as diagnostics,
not concealed or repaired. Periodic native rotation invariance is still tested.
The first attempt's four false total-cluster-energy assertions are preserved in
run1, with the independent diagnosis in results/rotation-diagnostic.json.

No MD, new allocation, OMol/SPICE download, stable promotion, commit or PR is
part of this pilot.

## Versioned checkpoint and archived evidence

The original validation was run at implementation commit `b5238fe`. The named
`lsc-validated-20261120` checkpoint adds only validation tools, frozen fixtures
and documentation. Preparation accepts descendants only after checking that
the implementation and build configuration still match that commit; it records
both the implementation pin and actual checkout. The original frozen manifests
and predictions remain unchanged. Binary SHA-256 checks are still exact.

These runners reproduce a specific Rusty experiment. `LSC_ORGANIC_DATA` selects
its external artifact root; original model checkpoints, exports and raw outputs
are not shipped in Git. The versioned `data/organic_stage3/manifest.json` records
their hashes, source indices and the correspondence to the four small fixture/
reference files. See [CHECKPOINT.md](CHECKPOINT.md) for the tester workflow.
