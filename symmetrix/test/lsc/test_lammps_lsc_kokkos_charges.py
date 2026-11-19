"""Charge DualView ownership across host rebuilds and full-atom MD/restarts."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from ase.io import read, write
from symmetrix.lsc_calculator import LSCCalculator

from .test_cuda_geometry import export_file
from .test_lammps_lsc_kokkos import STYLE, commands_for, launch

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda, pytest.mark.lsc_lammps]


def atom_commands(tmp_path, precision, atom_style):
    binary, atoms, commands = commands_for(tmp_path, 'bulk', precision, mesh=True)
    atoms.set_initial_charges(np.zeros(len(atoms)))
    if atom_style == 'full':
        atoms.new_array('mol-id', np.arange(len(atoms), dtype=int) // 3 + 1)
    write(tmp_path / 'atoms.data', atoms, format='lammps-data', atom_style=atom_style,
          specorder=['H', 'O'], masses=True)
    commands = ['atom_style ' + atom_style if c.startswith('atom_style ') else c
                for c in commands]
    return binary, atoms, commands


def output_commands(dump, label):
    return ['compute vir all pressure NULL virial',
            f'compute mu all pair {STYLE}',
            'thermo_style custom step pe temp press vol density',
            'thermo_modify norm no format float %.17g',
            f'print "{label} $(pe:%.17g) $(c_vir[1]:%.17g) $(c_vir[2]:%.17g) $(c_vir[3]:%.17g) $(c_vir[4]:%.17g) $(c_vir[5]:%.17g) $(c_vir[6]:%.17g) $(c_mu[1]:%.17g) $(c_mu[2]:%.17g) $(c_mu[3]:%.17g)"',
            f'write_dump all custom {dump} id type x y z fx fy fz q modify sort id format float %.17g']


def check_prediction(output, dump, label, export_file, record_property):
    match = re.search(r'^' + label + r' (.*)$', output, re.M)
    assert match, output[-5000:]
    values = np.fromstring(match.group(1), sep=' ')
    rows = np.loadtxt(dump.read_text().splitlines()[9:])
    atoms = read(dump, format='lammps-dump-text', specorder=['H', 'O'])
    reference = LSCCalculator(export_file, device='cpu')
    reference.model.set_mesh_electrostatics(.7, 8)
    reference.calculate(atoms, properties=['energy', 'forces', 'stress'])
    expected = reference.results
    assert abs(values[0] - expected['energy']) < 5e-5
    np.testing.assert_allclose(rows[:, 5:8], expected['forces'], atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(rows[:, 8], expected['charges'], atol=2e-6, rtol=1e-5)
    stress = -values[1:7][[0, 1, 2, 5, 4, 3]] / 1.6021765e6
    np.testing.assert_allclose(stress, expected['stress'], atol=5e-7, rtol=1e-4)
    np.testing.assert_allclose(values[7:10], expected['dipole'], atol=3e-5, rtol=0)
    assert abs(rows[:, 8].sum()) < 2e-8
    errors = dict(energy_error_eV=float(abs(values[0] - expected['energy'])),
                  force_error_eV_A=float(np.max(abs(rows[:, 5:8] - expected['forces']))),
                  charge_error_e=float(np.max(abs(rows[:, 8] - expected['charges']))))
    for name, value in errors.items():
        record_property(label + '_' + name, value)
    return errors


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
@pytest.mark.parametrize('atom_style', ['full', 'charge'])
def test_charge_sync_after_host_rebuild(export_file, tmp_path, precision, atom_style, record_property):
    binary, atoms, commands = atom_commands(tmp_path, precision, atom_style)
    # Exchange and sorting can make the entire host charge view current even
    # though LSC overwrites only owned charges in its device output kernel.
    commands.insert(0, 'package kokkos newton on neigh half sort host comm/exchange no')
    commands = ['neigh_modify every 1 delay 0 check no' if c.startswith('neigh_modify ') else c
                for c in commands]
    dump = tmp_path / 'final.dump'
    commands += ['atom_modify sort 1 0.0', 'velocity all create 10 19371 mom yes rot yes loop geom',
                 'fix integrate all nve', 'timestep .0001', 'thermo 1',
                 'dump qtrace all custom 1 charges.dump id type q', 'dump_modify qtrace sort id',
                 'run 10']
    # Define global observables before their final force evaluation.
    outputs = output_commands(dump, 'CHARGE_SYNC')
    commands += outputs[:4] + ['run 0'] + outputs[4:]
    output = launch(binary, commands, tmp_path)
    errors = check_prediction(output, dump, 'CHARGE_SYNC', export_file, record_property)
    assert 'Dangerous builds' in output
    print('LSC_KOKKOS_HOST_REBUILD', atom_style, precision, errors)


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
def test_full_atom_npt_restart(export_file, tmp_path, precision, record_property):
    binary, atoms, commands = atom_commands(tmp_path, precision, 'full')
    commands = ['neigh_modify every 5 delay 0 check no' if c.startswith('neigh_modify ') else c
                for c in commands]
    commands += ['change_box all triclinic', 'atom_modify sort 5 0.0',
                 'velocity all create 330 745 mom yes dist gaussian',
                 'fix md all nph iso 1 1 1', 'fix thermostat all temp/csvr 330 330 .1 1726',
                 'timestep .0005', 'thermo 10',
                 'thermo_style custom step pe temp press vol density',
                 'dump qtrace all custom 10 trajectory.dump id type x y z q',
                 'dump_modify qtrace sort id', 'restart 50 checkpoint.1 checkpoint.2', 'run 100',
                 'write_restart saved.restart']
    # Restarted charges are restored on the host; the same device pair must
    # safely recompute them while keeping molecular storage and output usable.
    restart = ['clear', 'units metal', 'atom_modify map array', 'read_restart saved.restart']
    restart += [c for c in commands if c.startswith(('pair_style ', 'pair_coeff ', 'neighbor ', 'neigh_modify '))]
    restart += ['fix md all nph iso 1 1 1', 'fix thermostat all temp/csvr 330 330 .1 1726',
                'timestep .0005', 'thermo 10']
    dump = tmp_path / 'restarted.dump'
    outputs = output_commands(dump, 'FULL_RESTART')
    restart += outputs[:4] + ['run 100'] + outputs[4:]
    output = launch(binary, commands + restart, tmp_path)
    errors = check_prediction(output, dump, 'FULL_RESTART', export_file, record_property)
    assert (tmp_path / 'saved.restart').stat().st_size > 0
    assert any((tmp_path / name).is_file() for name in ['checkpoint.1', 'checkpoint.2'])
    assert (tmp_path / 'trajectory.dump').stat().st_size > 0
    thermo = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 6 and fields[0].isdigit():
            thermo.append([float(value) for value in fields])
    assert len(thermo) == 22 and np.isfinite(thermo).all()
    assert int(thermo[-1][0]) == 200
    print('LSC_KOKKOS_FULL_NPT_RESTART', precision, 'steps=200', errors)


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
def test_original_density_input(tmp_path, precision, record_property):
    case = os.environ.get('LSC_FULL_ATOM_DENSITY_CASE')
    if not case:
        pytest.skip('set LSC_FULL_ATOM_DENSITY_CASE to the isolated stage3 density fixture')
    case = Path(case)
    for name in ['coord.lmp', 'forcefield.lmp']:
        shutil.copy2(case / name, tmp_path / name)
    model = next(case.glob('*.json'))
    (tmp_path / model.name).symlink_to(model)
    binary = Path(os.environ['LSC_LAMMPS_BINARY'])
    script = case / 'validate_density.py'
    proc = subprocess.run([sys.executable, str(script), '--work', str(tmp_path), '--binary', str(binary),
                           '--precision', precision], capture_output=True, text=True, timeout=600)
    (tmp_path / 'validation.stdout').write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, proc.stdout[-8000:] + proc.stderr[-4000:]
    result = json.loads((tmp_path / 'results.json').read_text())
    assert result['passed'] and result['steps'] == 1000
    for name, value in result['accuracy'].items():
        if name != 'passed':
            record_property(name, value)
    print('LSC_KOKKOS_ORIGINAL_DENSITY', precision, result)


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
def test_atomic_without_charge_storage(export_file, tmp_path, precision, record_property):
    binary, atoms, commands = atom_commands(tmp_path, precision, 'atomic')
    dump = tmp_path / 'atomic.dump'
    commands += ['run 0', 'print "ATOMIC_ENERGY $(pe:%.17g)"',
                 f'write_dump all custom {dump} id fx fy fz modify sort id format float %.17g']
    output = launch(binary, commands, tmp_path)
    energy = float(re.search(r'^ATOMIC_ENERGY (.*)$', output, re.M).group(1))
    rows = np.loadtxt(dump.read_text().splitlines()[9:])
    reference = LSCCalculator(export_file, device='cpu')
    reference.model.set_mesh_electrostatics(.7, 8)
    reference.calculate(atoms, properties=['energy', 'forces'])
    assert abs(energy - reference.results['energy']) < 5e-5
    np.testing.assert_allclose(rows[:, 1:4], reference.results['forces'], atol=8e-5, rtol=1e-4)
    record_property('force_error_eV_A', float(np.max(abs(rows[:, 1:4] - reference.results['forces']))))
    print('LSC_KOKKOS_ATOMIC_NO_Q', precision)
