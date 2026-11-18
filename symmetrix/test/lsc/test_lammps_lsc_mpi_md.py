"""Normal LAMMPS run 0/Verlet tests, without the staged diagnostic driver."""
import os
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
import symmetrix
from ase.io import write

from .reference import REPO, load_atoms
from .test_forward_mpi import launch
from .test_staged_api import backend

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_lammps]


def evaluate(directory, case, backend, ranks, field=False, steps=0, empty=False,
             resize=False, initial_split=None, rebalance=False, per_atom_energy=True):
    directory.mkdir(parents=True, exist_ok=True)
    atoms = load_atoms(case).copy()
    atoms.set_pbc(True)
    if case == 'dimer':
        atoms.positions += np.array([7.5 if empty else 15., 15., 15.])-atoms.positions.mean(axis=0)
    else:
        atoms.wrap()
    if steps:
        # Place an atom just left of the x-domain boundary. A common velocity
        # moves it across during the ten-step trajectory, in both decompositions.
        middle = atoms.cell[0, 0]/2
        chosen = np.argmin(abs(atoms.positions[:, 0]-middle))
        atoms.positions[:, 0] += middle-.03-atoms.positions[chosen, 0]
        atoms.wrap()
    data = directory/'atoms.data'
    write(data, atoms, format='lammps-data', atom_style='charge', masses=True,
          specorder=['H', 'O'])
    build = 'cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    binary = Path(os.environ.get('LSC_LAMMPS_BINARY',
                  str(REPO.parent/f'lammps-lsc-dev/build-{build}/lmp')))
    assert binary.is_file(), f'Missing development executable: {binary}'
    style = 'symmetrix/mace/lsc'+('' if backend == 'cpu' else '/cuda')
    option = ' float32-mixed' if backend == 'float32-mixed' else ''
    if field:
        option += ' field .01 -.02 .03'
    path = directory/'in.md'
    commands = [
        'units metal', 'atom_style charge', 'boundary p p p',
        f'processors {ranks} 1 1', 'atom_modify sort 1 0.0', f'read_data {data}',
        'newton on', f'pair_style {style} mesh .7 order 8{option}',
        f'pair_coeff * * {REPO}/.lsc-reference-logs/water-stage2-energy.json H O',
        'neighbor .4 bin', 'neigh_modify every 1 delay 0 check no',
        'compute eat all pe/atom', 'compute esum all reduce sum c_eat',
        f'compute mu all pair {style}',
        # The second consumer must not sum already-reduced dipoles again.
        'fix musample all ave/time 1 1 1 c_mu[1] c_mu[2] c_mu[3] file '+str(directory/'mu.dat'),
        'thermo 1',
        'thermo_style custom step pe ke etotal c_esum pxx pyy pzz pxy pxz pyz c_mu[1] c_mu[2] c_mu[3]',
        'thermo_modify norm no format float %.17g',
        f'dump states all custom 1 {directory}/states.dump id proc xu yu zu vx vy vz fx fy fz q c_eat',
        'dump_modify states sort id format float %.17g',
    ]
    if not per_atom_energy:
        # Keep the reader's column layout while requesting no per-atom PE.
        commands = [line.replace('c_esum', 'v_esum').replace('c_eat', 'v_eat')
                    for line in commands if not line.startswith(('compute eat ', 'compute esum '))]
        index = commands.index('thermo 1')
        commands[index:index] = ['variable esum equal 0.0', 'variable eat atom 0.0']
    if initial_split is not None:
        assert ranks == 3
        commands.insert(commands.index('newton on'),
                        f'balance 0.0 x {initial_split} 0.6666666666666666')
    if resize:
        commands.insert(commands.index('newton on'), 'change_box all triclinic')
    if steps:
        commands += ['velocity all create 10 19371 mom yes rot yes loop geom',
                     'velocity all set 100 NULL NULL sum yes', 'fix integrate all nve',
                     'timestep .0001', 'run 5', 'run 5']
    else:
        commands += ['run 0']
    if resize or rebalance:
        change = ('balance 0.0 x .32 0.6666666666666666' if rebalance else
                  'change_box all x scale 1.001 xy delta .01 remap')
        commands += ['unfix musample', 'undump states', 'reset_timestep 1',
                     change,
                     f'dump states all custom 1 {directory}/states.dump id proc xu yu zu vx vy vz fx fy fz q c_eat',
                     'dump_modify states append yes sort id format float %.17g', 'run 0']
    path.write_text('\n'.join(commands)+'\n')
    result = launch(ranks, binary, ['-in', path, '-log', 'none'])
    (directory/'lammps.stdout').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout[-16000:]+result.stderr[-4000:]
    samples = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 14 and re.fullmatch(r'\d+', fields[0]):
            samples.append([float(x) for x in fields])
    thermo = np.asarray(samples)
    assert len(thermo) == (12 if steps else 2 if resize or rebalance else 1), result.stdout
    # Consecutive runs repeat step 5. Keep the final evaluation of each step.
    thermo = np.asarray(list({row[0]: row for row in thermo}.values()))
    lines = (directory/'states.dump').read_text().splitlines()
    frames = {}
    for start, line in enumerate(lines):
        if line == 'ITEM: TIMESTEP':
            step = int(lines[start+1]); count = int(lines[start+3])
            values = np.asarray([[float(x) for x in row.split()]
                                 for row in lines[start+9:start+9+count]])
            frames[step] = values[np.argsort(values[:, 0])]
    assert frames and len(frames) == (11 if steps else 2 if resize or rebalance else 1)
    assert all(np.isfinite(frame).all() for frame in frames.values())
    if per_atom_energy:
        np.testing.assert_allclose(thermo[:, 1], thermo[:, 4], atol=2e-8, rtol=0)
    for step, frame in frames.items():
        row = thermo[thermo[:, 0] == step][-1]
        if per_atom_energy:
            assert abs(frame[:, 12].sum()-row[1]) < 2e-8
        assert abs(frame[:, 11].sum()) < 2e-8
    mu = np.loadtxt(directory/'mu.dat', comments='#', ndmin=2)
    for row in mu:
        expected = thermo[thermo[:, 0] == row[0]][-1, 11:14]
        np.testing.assert_allclose(row[1:], expected, atol=5e-5, rtol=2e-6)
    return thermo, frames


def compare(one, two, backend, record_property):
    a, af = one; b, bf = two
    cpu = backend == 'cpu'
    energy_error = float(np.max(abs(a[:, 1:5]-b[:, 1:5])))
    assert energy_error < (2e-8 if cpu else 5e-5)
    stress_error = float(np.max(abs(a[:, 5:11]-b[:, 5:11]))/1.6021765e6)
    assert stress_error < (2e-10 if cpu else 5e-7)
    np.testing.assert_allclose(a[:, 11:14], b[:, 11:14], atol=2e-8 if cpu else 3e-5, rtol=0)
    force_error = 0.
    for step in af:
        x, y = af[step], bf[step]
        np.testing.assert_array_equal(x[:, 0], y[:, 0])
        np.testing.assert_allclose(x[:, 2:8], y[:, 2:8], atol=2e-8 if cpu else 2e-5, rtol=0)
        np.testing.assert_allclose(x[:, 8:11], y[:, 8:11], atol=2e-8 if cpu else 8e-5, rtol=0)
        np.testing.assert_allclose(x[:, 11], y[:, 11], atol=2e-10 if cpu else 2e-6, rtol=0)
        np.testing.assert_allclose(x[:, 12], y[:, 12], atol=2e-8 if cpu else 5e-5, rtol=0)
        force_error = max(force_error, float(abs(x[:, 8:11]-y[:, 8:11]).max()))
    for key, value in dict(backend=backend, energy_abs_eV=energy_error,
                           force_max_abs_eV_A=force_error, stress_max_abs_eV_A3=stress_error).items():
        record_property(key, value)
    print('LAMMPS_MPI_MD', backend, 'energy', energy_error, 'forces', force_error,
          'stress', stress_error)


@pytest.mark.parametrize('case', ['dimer', 'bulk'])
@pytest.mark.parametrize('field', [False, True], ids=['zero', 'field'])
def test_normal_mpi_point_prediction(backend, case, field, tmp_path, record_property):
    one = evaluate(tmp_path/'one', case, backend, 1, field=field)
    two = evaluate(tmp_path/'two', case, backend, 2, field=field)
    assert set(two[1][0][:, 1]) == {0, 1}
    compare(one, two, backend, record_property)


@pytest.mark.parametrize('case', ['dimer', 'bulk'])
def test_normal_mpi_nve_and_migration(backend, case, tmp_path, record_property):
    one = evaluate(tmp_path/'one', case, backend, 1, steps=10)
    two = evaluate(tmp_path/'two', case, backend, 2, steps=10)
    compare(one, two, backend, record_property)
    assert np.any(two[1][0][:, 1] != two[1][10][:, 1]), 'No atom changed MPI owner'
    drift = float(np.max(abs(two[0][:, 3]-two[0][0, 3]))/len(two[1][0]))
    assert drift < 1e-4
    record_property('energy_drift_eV_per_atom', drift)
    print('LAMMPS_MPI_NVE', backend, case, 'steps=10 migrations=',
          np.count_nonzero(two[1][0][:, 1] != two[1][10][:, 1]), 'drift=', drift)


def test_normal_mpi_empty_rank(backend, tmp_path, record_property):
    one = evaluate(tmp_path/'one', 'dimer', backend, 1, empty=True, field=True)
    two = evaluate(tmp_path/'two', 'dimer', backend, 2, empty=True, field=True)
    assert len(set(two[1][0][:, 1])) == 1
    compare(one, two, backend, record_property)


def test_normal_mpi_box_change(backend, tmp_path, record_property):
    one = evaluate(tmp_path/'one', 'bulk', backend, 1, resize=True)
    two = evaluate(tmp_path/'two', 'bulk', backend, 2, resize=True)
    compare(one, two, backend, record_property)


def test_normal_mpi_rebalance(backend, tmp_path, record_property):
    # Both runs end with the same nonuniform decomposition. Only ranks 0/1
    # change bounds during rebalancing; rank 2 must still recreate its plans.
    fixed = evaluate(tmp_path/'fixed', 'bulk', backend, 3, initial_split=.32)
    changed = evaluate(tmp_path/'changed', 'bulk', backend, 3,
                       initial_split=.40, rebalance=True)
    before, after = changed[1][0], changed[1][1]
    assert np.any(before[:, 1] != after[:, 1])
    np.testing.assert_array_equal(before[before[:, 1] == 2, 0],
                                  after[after[:, 1] == 2, 0])
    compare(fixed, (changed[0][-1:], {0: after}), backend, record_property)


@pytest.mark.parametrize('case', ['dimer', 'bulk'])
def test_normal_one_rank_preserved(backend, case, tmp_path, record_property):
    """Baseline captured by the unchanged pair on the same H100 before rebuilding."""
    key = backend+'/'+case
    build = 'cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    baseline = Path(os.environ.get('LSC_MPI_MD_BASELINE',
                                  str(REPO/f'.lsc-reference-logs/mpi-md-before-{build}.npz')))
    metadata = json.loads(baseline.with_suffix(".json").read_text())
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(baseline, allow_pickle=False) as data:
        expected = (data[key+'/thermo'], {int(step): data[key+'/frame'+str(int(step))]
                    for step in data[key+'/steps']})
    actual = evaluate(tmp_path/'one', case, backend, 1, steps=10, field=True)
    if backend == 'cpu':
        np.testing.assert_array_equal(actual[0], expected[0])
        for step in actual[1]:
            np.testing.assert_array_equal(actual[1][step], expected[1][step])
    else:
        # The forward path is deterministic; CUDA atomic force reductions have
        # existing roundoff. Use the pre-existing one-rank GPU test tolerances.
        compare(expected, actual, backend, record_property)


@pytest.mark.parametrize('case', ['dimer', 'bulk'])
def test_q_only_matches_per_atom_energy(backend, case, tmp_path):
    """Writing predicted q must agree with the path also requesting pe/atom."""
    with_energy = evaluate(tmp_path/'energy', case, backend, 1, field=True)
    charges_only = evaluate(tmp_path/'charges', case, backend, 1, field=True,
                            per_atom_energy=False)
    a, af = with_energy; b, bf = charges_only
    # Forward energy, dipole and charges are independent of requested readbacks.
    np.testing.assert_array_equal(a[:, [0, 1, 2, 3, 11, 12, 13]],
                                  b[:, [0, 1, 2, 3, 11, 12, 13]])
    limit = 2e-10 if backend == 'cpu' else 5e-7
    np.testing.assert_allclose(a[:, 5:11]/1.6021765e6, b[:, 5:11]/1.6021765e6,
                               atol=limit, rtol=0)
    for step in af:
        x, y = af[step], bf[step]
        np.testing.assert_array_equal(x[:, :8], y[:, :8])
        np.testing.assert_array_equal(x[:, 11], y[:, 11])
        assert np.max(abs(y[:, 11])) > 1e-12
        np.testing.assert_allclose(x[:, 8:11], y[:, 8:11],
                                   atol=2e-8 if backend == 'cpu' else 8e-5, rtol=0)
