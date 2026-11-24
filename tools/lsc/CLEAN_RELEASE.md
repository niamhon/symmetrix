# Clean LSC Symmetrix release

This branch starts at the clean LSC + ASE checkpoint `9d9265b` and integrates
the public Polar ASE GPU interface from upstream commits `eccc39c` and `6df3a70`.
Their original authorship is retained. No subsequent dispersion integration is
included, and no LSC numerical kernel is changed by the Polar integration.

The LSC training and independent reference implementation remains
`ACEsuit/mace-scf`, pinned on Rusty at `4d1fe8a`.

## Installed runtime

The versioned Rusty release is prepared at:

`/mnt/home/noneill/ceph/symmetrix-lsc-releases/releases/lsc-clean-20261124`

Use its `activate.sh` after its manifest reports validation success. This selects
the noneditable Python package and frozen LAMMPS executable, clears development
Python paths, and exposes `LSC_RELEASE_ROOT` and `LSC_LAMMPS_BINARY`. Existing
projects should record the resolved versioned path rather than follow a mutable
`current` link. Do not rebuild a validated release in place.

The release reuses the validated standalone LAMMPS compact-plus-charge-sync
binary; the Python package is rebuilt to add Polar's GPU binding. Its manifest
records both source/binary revisions. System MPI/CUDA/compiler libraries are
provided by Rusty's immutable Nix paths. GPU compilation targets SM80 plus PTX;
actual validation hardware is recorded separately. CPU FP64 support and CPU ISA
portability must be assessed from the release validation, not inferred from GPU
compatibility.

## Export and use LSC

```bash
python -m symmetrix.cli.extract_mace \
  --model /absolute/model.model --chemical-symbols H O \
  --num-spline-points 4096 --r-min 0.5 \
  --output /absolute/model.symmetrix.json
```

Use the model's actual species. Architecture validation remains enabled. Verify
the spline lower bound against relevant distances. No dispersion correction is
added by this branch; choose a complete model or supply its intended correction
separately in a dedicated project.

```python
from symmetrix.experimental import MACELSC
atoms.calc = MACELSC("model.symmetrix.json", device="cuda", dtype="float32",
                     electrostatics="mesh", mesh_spacing=0.7, mesh_order=8)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

LSC fully open clusters use direct electrostatics; fully periodic 3D cells may
use direct or mesh. CPU evaluation is FP64. Mixed GPU precision is opt-in.
See `ASE_CALCULATOR.md` for charge, dipole, field and cache conventions.

```lammps
units metal
atom_style charge
atom_modify map array
boundary p p p
read_data water.data
pair_style symmetrix/mace/lsc/cuda/kk mesh 0.7 order 8
pair_coeff * * model.symmetrix.json H O
```

The example means atom type 1 is H and type 2 is O; match the actual data file.
Launch with `-k on g 1 -sf kk -pk kokkos newton on neigh half`. Electrostatics is
already included; do not add a second Coulomb solver. Mesh parameters above are
validated examples, not universal accuracy settings.

## Other models through ASE

```python
from symmetrix.experimental import MACEPolar, MACEFoundation
polar = MACEPolar("polar.json", device="cuda", dtype="float32")
mp0 = MACEFoundation("mp0-H-O.json", device="cuda", dtype="float32")
```

Polar's integrated public ASE interface requires full 3D periodicity and uses
direct reciprocal-space electrostatics, rebuilding neighbors each call. It does
not yet expose LAMMPS's mesh path. Its size scaling and memory limit therefore
differ from Polar's mesh LAMMPS route. Charges/spin can come from `atoms.info`.
The native CUDA ASE route is distinct from the LAMMPS Kokkos pair route.

## Tests, versions and future work

The opt-in `symmetrix/test/test_release_interfaces.py` reads a frozen fixture
manifest through `SYMMETRIX_RELEASE_FIXTURES`. It checks Polar/foundation GPU
energy, forces, stress and per-atom sums against original FP64 native predictions,
plus Polar invalid-input and charge/spin cache behavior. It never regenerates
references. Existing LSC tests retain their native-reference tolerances.

The release manifest, validation XML/JSON, input/model/export hashes and measured
MD timings are authoritative. Package version `0.3.0` alone does not identify
this fork's implementation. Dispersion development belongs on a later separate
branch created from the tested clean release. Existing LSD/FourierD3 work is
preserved; integrating it is outside this release.
