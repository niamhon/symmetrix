"""One-GPU LSC device graph: physical outputs and moving/reordered atoms."""
import os
import re
import shlex
import subprocess

import numpy as np
import pytest
from symmetrix.lsc_calculator import LSCCalculator

from .test_lammps_lsc import setup
from .test_cuda_geometry import export_file

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda, pytest.mark.lsc_lammps]
STYLE = "symmetrix/mace/lsc/cuda/kk"


def launch(binary, commands, work):
    path = work / "in.kokkos"
    path.write_text("\n".join(commands) + "\n")
    launcher = shlex.split(os.environ.get("LSC_LAMMPS_LAUNCHER", ""))
    proc = subprocess.run(
        launcher + [str(binary), "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                    "newton", "on", "neigh", "half", "-log", "none", "-in", str(path)],
        cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    (work / "lammps.stdout").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout[-14000:] + proc.stderr[-3000:]
    return proc.stdout


def commands_for(tmp_path, case, precision, mesh=False):
    binary, atoms, commands = setup(tmp_path, case, precision)
    commands.insert(commands.index(next(c for c in commands if c.startswith("read_data "))),
                    "atom_modify map array")
    suffix = " float32-mixed" if precision == "float32-mixed" else ""
    setting = "mesh .7 order 8" if mesh else "direct"
    commands = [(f"pair_style {STYLE} {setting}{suffix}" if c.startswith("pair_style ") else c)
                for c in commands]
    return binary, atoms, commands


@pytest.mark.parametrize("precision", ["float32", "float32-mixed"])
@pytest.mark.parametrize("case,mesh", [("dimer", False), ("bulk", False), ("bulk", True)])
@pytest.mark.parametrize("field", [(0., 0., 0.), (.01, -.02, .03)])
def test_device_pair_outputs(export_file, tmp_path, precision, case, mesh, field, record_property):
    binary, atoms, commands = commands_for(tmp_path, case, precision, mesh)
    commands = [c + " field " + " ".join(str(x) for x in field) if c.startswith("pair_style ") else c
                for c in commands]
    dump = tmp_path / "forces.dump"
    commands += ["compute eat all pe/atom", "compute esum all reduce sum c_eat",
                 f"compute mu all pair {STYLE}",
                 "thermo_style custom step pe c_esum pxx pyy pzz pxy pxz pyz c_mu[1] c_mu[2] c_mu[3]",
                 "thermo_modify norm no format float %.17g", "run 0",
                 'print "DEVICE_RESULT $(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g) $(c_esum:%.17g) $(c_mu[1]:%.17g) $(c_mu[2]:%.17g) $(c_mu[3]:%.17g)"',
                 f"write_dump all custom {dump} id type fx fy fz q c_eat modify sort id format float %.17g"]
    output = launch(binary, commands, tmp_path)
    match = re.search(r"^DEVICE_RESULT (.*)$", output, re.M)
    assert match, output
    values = np.fromstring(match.group(1), sep=" ")
    rows = np.loadtxt(dump.read_text().splitlines()[9:])
    reference = LSCCalculator(export_file, device="cpu", external_field=field)
    if mesh:
        reference.model.set_mesh_electrostatics(.7, 8)
    reference.calculate(atoms, properties=["energy", "forces", "stress"])
    expected = reference.results
    assert abs(values[0] - expected["energy"]) < 5e-5
    np.testing.assert_allclose(rows[:, 2:5], expected["forces"], atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(rows[:, 5], expected["charges"], atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(values[8:11], expected["dipole"], atol=3e-5, rtol=0)
    if case == "bulk":
        stress = -values[1:7][[0, 1, 2, 5, 4, 3]] / 1.6021765e6
        np.testing.assert_allclose(stress, expected["stress"], atol=5e-7, rtol=1e-4)
    assert abs(values[0] - values[7]) < 2e-8
    assert abs(rows[:, 6].sum() - values[0]) < 2e-8
    assert abs(rows[:, 5].sum()) < 2e-8
    errors = dict(energy_eV=abs(values[0] - expected["energy"]),
                  force_eV_A=float(np.max(abs(rows[:, 2:5] - expected["forces"]))))
    for name, value in errors.items():
        record_property(name, value)
    print("LSC_KOKKOS_POINT", precision, case, mesh, field, errors)


@pytest.mark.parametrize("precision", ["float32", "float32-mixed"])
@pytest.mark.parametrize("case,mesh", [("dimer", False), ("bulk", True)])
def test_device_pair_short_md(export_file, tmp_path, precision, case, mesh, record_property):
    binary, atoms, commands = commands_for(tmp_path, case, precision, mesh)
    dump = tmp_path / "trajectory.dump"
    commands += ["atom_modify sort 1 0.0", "velocity all create 10 19371 mom yes rot yes loop geom",
                 "fix integrate all nve", "timestep .0001", "thermo 1",
                 "thermo_style custom step pe ke etotal temp", "thermo_modify norm no format float %.17g",
                 "run 5", "run 5",
                 f"write_dump all custom {dump} id type x y z fx fy fz q modify sort id format float %.17g"]
    output = launch(binary, commands, tmp_path)
    thermo = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0].isdigit():
            thermo.append([float(x) for x in fields])
    values = np.asarray(thermo)
    assert len(values) == 12 and np.isfinite(values).all(), output
    drift = float(np.max(abs(values[:, 3] - values[0, 3])) / len(atoms))
    assert drift < 1e-4
    rows = np.loadtxt(dump.read_text().splitlines()[9:])
    atoms.positions = rows[:, 2:5]
    reference = LSCCalculator(export_file, device="cpu")
    if mesh:
        reference.model.set_mesh_electrostatics(.7, 8)
    reference.calculate(atoms, properties=["energy", "forces"])
    assert abs(values[-1, 1] - reference.results["energy"]) < 5e-5
    np.testing.assert_allclose(rows[:, 5:8], reference.results["forces"], atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(rows[:, 8], reference.results["charges"], atol=2e-6, rtol=1e-5)
    record_property("drift_eV_atom", drift)
    print("LSC_KOKKOS_NVE", precision, case, "steps=10", drift)
