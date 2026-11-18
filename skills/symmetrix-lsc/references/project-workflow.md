# New-project workflow

## Inspect and export

Read project instructions and inputs. Establish the model's training/dispersion
provenance, species, coordinates, formal-charge/field requirements, PBC and atom
type numbering. Retain an existing input's requested temperature, pressure,
timestep, duration and output choices unless explaining a necessary correction.

Use an environment containing this fork and reference MACE-SCF. Export to a new
project file, preserving existing exports:

```bash
python -m symmetrix.cli.extract_mace \
  --model /absolute/project/model.model --chemical-symbols H O \
  --num-spline-points 4096 --r-min 0.5 \
  --output /absolute/project/model.symmetrix.json
```

Replace the symbols with the model species. The example uses validated water
spline settings; ensure the lower bound covers relevant new-project distances.
Retain architecture validation. Confirm `model_type=LocalSplitCharges` and
`export_status=parameters_and_splines`; record cutoff, species, architecture,
versions and hashes. LSC basis compression is not part of this supported path.

## Validate before production

Use representative configurations with identical positions, cells and identities
on both paths, including the actual bulk case for NPT. Save native predictions
before comparing. Do not select a checkpoint by the first `*.model` glob match.

Set PyTorch's default dtype to FP64 and construct the reference:

```python
from mace_scf.calculators.localsources import MACELocalSplitCharges

native = MACELocalSplitCharges(
    model_path=str(model_path), device="cpu", default_dtype="float64",
    pbc_handling="mixed_periodic", use_compile=False,
)
native.model.eval()
for parameter in native.model.parameters():
    parameter.requires_grad_(False)
native.calculate(atoms, properties=["energy", "forces", "stress"])
```

Use `realspace` for a nonperiodic cluster and omit stress when not meaningful.
Native CUDA FP64 is also usable in a suitable allocation. Keep stress enabled
for NPT; a compiled calculator lacking stress cannot validate that workflow.

The native Symmetrix ASE wrapper is:

```python
from symmetrix.lsc_calculator import LSCCalculator

fast = LSCCalculator(
    str(export_path), device="cuda", precision="float32",
    pbc_handling="mixed_periodic", external_field=(0.0, 0.0, 0.0),
)
if fast.device == "cuda":
    fast.model.use_batched_reverse = True
fast.model.set_mesh_electrostatics(0.7, 8)
fast.calculate(atoms, properties=["energy", "forces", "stress"])
```

Adapt the field to the requested physics. CPU uses `device="cpu"`, FP64; GPU
mixed uses `precision="float32-mixed"`. Omit the mesh setter for direct
evaluation. Spacing 0.7 Å/order 8 is a tested starting point, not a universal
mesh-accuracy guarantee. Cluster calculations must use direct mode.

The CUDA LAMMPS pair already selects receiver-batched reverse evaluation. The
Python class retains a legacy reverse default for diagnostics; enable the flag
above to select the same optimized path when using its ASE wrapper.

Existing native-reference gates are energy error `<1e-6 eV/atom`, forces
`allclose(atol=8e-5 eV/Å, rtol=1e-4)` and periodic stress
`allclose(atol=5e-7 eV/Å³, rtol=1e-4)`. Preserve these unless the project calls
for stricter accuracy. Compare charges/dipoles when needed: native keys are
`partial_charges`/`dipole`, wrapper keys `charges`/`dipole`. Use consistent
coordinates and polarization branches. Do not widen tolerances to pass failure.

Also validate the actual LAMMPS pair. Compare potential-only pressure
(`compute vir all pressure NULL virial`) against ASE potential stress. LAMMPS
bar components `[xx,yy,zz,xy,xz,yz]` map to ASE by:

```python
ase_stress = -pressure[[0, 1, 2, 5, 4, 3]] / 1.6021765e6
```

The MD barostat must still include kinetic pressure. Run a short isolated case
with the intended integrator and outputs; check finite observables and
trajectory/restart creation. A 1000-step startup check is not equilibration.

## LAMMPS input

Example for fully periodic GPU evaluation:

```lammps
units metal
atom_style charge
boundary p p p
atom_modify map yes
newton on
read_data initial.data
pair_style symmetrix/mace/lsc/cuda mesh 0.7 order 8
pair_coeff * * model.symmetrix.json O H
```

`O H` means type 1 is O and type 2 is H; derive this from the actual data, not
JSON species order. Existing `atom_style full` inputs require MOLECULE support.

- CPU pair is `symmetrix/mace/lsc`; CUDA is `symmetrix/mace/lsc/cuda`.
- Append `float32-mixed` for mixed neural arithmetic.
- Direct mode requires one MPI process. Mesh supports normal multi-rank MD.
- CUDA invokes its own native evaluator: no `/kk` suffix or required `-sf kk`.
- Use `units metal`, atom IDs and `newton pair on`. Global stress and per-atom
  energies are supported; per-atom virials are not.
- With a charge-capable atom style, `q` is a predicted model output, not an
  independent charge-equilibration variable.
- `compute mu all pair symmetrix/mace/lsc/cuda` returns total dipole components.
  The optional `field Ex Ey Ez` is in eV/(e Å), energy convention `+ E·mu`.
- Reissue `pair_style` and `pair_coeff` after `read_restart`: model parameters
  are not stored in binary restart files.
- CSVR NPT uses an NPH barostat plus CSVR temperature control. Avoid applying
  Nose–Hoover temperature control simultaneously. A single `fix npt` is the
  ordinary Nose–Hoover alternative.
- Global E/virial is the efficient MD path. Per-atom energy and neural diagnostic
  readbacks cost additional time; preserve them when the project needs them.

## Results

Record job ID, actual GPU/rank assignment and startup checks after submission.
For averages, verify time units and discard interval and report the available
window. Average instantaneous mass/volume for density; account for correlations
in error estimates. Disclose model, box-size and thermostat/barostat differences
in comparisons. Distinguish actual LAMMPS throughput from ASE/prepared-graph
timing; exclude loading/warm-up and synchronize GPU timing harnesses.
