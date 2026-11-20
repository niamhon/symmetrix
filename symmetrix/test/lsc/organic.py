"""Frozen organic-model validation. Reference generation is explicitly separate.

Only this test helper adapts coordinate frames and polarization branches; it
does not change either evaluator or any accuracy threshold.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path('/mnt/home/noneill/ceph/symmetrix-lsc-validation/organic-stage3')
PIN = 'b5238fee78b01cedddc7bd151dec1750bebe5624'
BINARY_SHA = '8b0fbe2e4224e2d3e910fbf2aa95c6b7b50eede38096d847ac8a872c3726197a'
STYLE = 'symmetrix/mace/lsc/cuda/kk'
TOLERANCES = dict(energy=1e-6, forces=(8e-5, 1e-4), stress=(5e-7, 1e-4),
                  charges=(2e-6, 1e-5), dipole=(3e-5, 0.0))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def implementation_checkpoint():
    """Accept test/documentation descendants only when the implementation is unchanged."""
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    ancestor = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', PIN, head], cwd=REPO,
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if ancestor.returncode:
        raise RuntimeError(f'Checkout {head} does not descend from validated LSC {PIN}')
    paths = ['libsymmetrix', 'pair_symmetrix', 'symmetrix/source',
             'symmetrix/CMakeLists.txt', 'symmetrix/pyproject.toml', '.gitmodules', 'VERSION']
    unchanged = subprocess.run(
        ['git', 'diff', '--quiet', PIN, '--', *paths], cwd=REPO,
        stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if unchanged.returncode:
        raise RuntimeError('Implementation differs from the validated LSC checkpoint')
    return head


def root():
    return Path(os.environ.get('LSC_ORGANIC_DATA', DEFAULT_ROOT))


def manifest():
    return json.loads((root() / 'manifest.json').read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_case(name, case_id):
    from ase.io import read
    return read(root() / 'fixtures' / f'{name}.extxyz', index=case_id)


def reference(name, case_id):
    with np.load(root() / 'references' / f'{name}.npz', allow_pickle=False) as data:
        prefix = f'{case_id:03d}__'
        return {k[len(prefix):]: data[k].copy() for k in data if k.startswith(prefix)}


def native_calculator(name, mode, device='cuda'):
    import torch
    from mace_scf.calculators.localsources import MACELocalSplitCharges
    torch.set_default_dtype(torch.float64)
    calc = MACELocalSplitCharges(
        model_path=str(root() / 'models' / f'{name}.model'), device=device,
        default_dtype='float64', pbc_handling=mode, use_compile=False)
    calc.model.eval()
    for parameter in calc.model.parameters():
        parameter.requires_grad_(False)
    return calc


def evaluate_native(atoms, calc):
    a = atoms.copy()
    props = ['energy', 'forces'] + (['stress'] if all(a.pbc) else [])
    calc.calculate(a, properties=props)
    result = {key: np.asarray(calc.results[key], dtype=np.float64).copy()
              for key in ['energy', 'forces', 'dipole', 'partial_charges',
                          'partial_dipoles', 'density_coefficients'] +
                         (['stress'] if all(a.pbc) else [])}
    result['charges'] = result.pop('partial_charges')
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite native prediction')
    return result


def compare(actual, expected, atoms):
    e = abs(float(actual['energy']) - float(expected['energy'])) / len(atoms)
    metrics = dict(energy_eV_atom=e)
    passed = dict(energy=e < TOLERANCES['energy'])
    units = dict(forces='eV_A', stress='eV_A3', charges='e', dipole='eA')
    for key in ['forces', 'charges', 'dipole'] + (['stress'] if all(atoms.pbc) else []):
        delta = np.asarray(actual[key]) - expected[key]
        metrics[f'{key}_max_{units[key]}'] = float(np.max(abs(delta)))
        metrics[f'{key}_rms_{units[key]}'] = float(np.sqrt(np.mean(delta**2)))
        atol, rtol = TOLERANCES[key]
        passed[key] = bool(np.all(abs(delta) <= atol + rtol * abs(expected[key])))
    metrics['gates'] = passed
    metrics['passed'] = all(passed.values())
    return metrics


def _data(atoms, species, path):
    """Write full-precision coordinates without translating or wrapping atoms.

    Rotated periodic cells are mapped to restricted triclinic form by a proper
    orthogonal rotation. Returned vectors/tensors are rotated back afterwards.
    """
    if all(atoms.pbc):
        a, b, _ = atoms.cell.array
        ex = a / np.linalg.norm(a)
        ey = b - np.dot(b, ex) * ex
        ey /= np.linalg.norm(ey)
        rotation = np.column_stack([ex, ey, np.cross(ex, ey)])
        cell = atoms.cell.array @ rotation
        if np.linalg.det(cell) <= 0:
            raise ValueError('Cell must have positive volume')
        lo, hi = np.zeros(3), np.diag(cell)
    else:
        rotation = np.eye(3)
        lo = atoms.positions.min(axis=0) - 8.0
        hi = atoms.positions.max(axis=0) + 8.0
        cell = np.diag(hi - lo)
    xyz = atoms.positions @ rotation
    lines = ['Organic LSC frozen snapshot', '', f'{len(atoms)} atoms',
             f'{len(species)} atom types', '']
    for d, label in enumerate(['x', 'y', 'z']):
        lines.append(f'{lo[d]:.17g} {hi[d]:.17g} {label}lo {label}hi')
    if all(atoms.pbc):
        lines.append(f'{cell[1,0]:.17g} {cell[2,0]:.17g} {cell[2,1]:.17g} xy xz yz')
    lines += ['', 'Masses', '']
    for i, z in enumerate(species, 1):
        lines.append(f'{i} {atomic_masses[z]:.17g}')
    lines += ['', 'Atoms # charge', '']
    for i, (z, x) in enumerate(zip(atoms.numbers, xyz), 1):
        lines.append(f'{i} {species.index(int(z))+1} 0 ' + ' '.join(f'{v:.17g}' for v in x))
    path.write_text('\n'.join(lines) + '\n')
    return rotation


def lammps(name, atoms, precision, work, reverse_species=False):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    info = manifest()['models'][name]
    species = list(info['atomic_numbers'])
    if reverse_species:
        species.reverse()
    rotation = _data(atoms, species, work / 'atoms.data')
    periodic = all(atoms.pbc)
    setting = 'mesh 0.7 order 8' if periodic else 'direct'
    suffix = ' float32-mixed' if precision == 'float32-mixed' else ''
    commands = [
        'units metal', 'atom_style charge', 'atom_modify map array sort 0 0.0',
        'boundary ' + ('p p p' if periodic else 'f f f'), 'box tilt large',
        f'read_data {work / "atoms.data"}',
        f'pair_style {STYLE} {setting}{suffix}',
        f'pair_coeff * * {root() / "models" / (name + ".json")} ' +
        ' '.join(chemical_symbols[z] for z in species),
        'neighbor 0.4 bin', 'neigh_modify every 1 delay 0 check yes',
        'compute vir all pressure NULL virial', f'compute mu all pair {STYLE}',
        'thermo_style custom step pe c_vir[1] c_vir[2] c_vir[3] c_vir[4] c_vir[5] c_vir[6] c_mu[1] c_mu[2] c_mu[3]',
        'thermo_modify norm no format float %.17g', 'run 0',
        'print "ORGANIC_RESULT $(pe:%.17g) $(c_vir[1]:%.17g) $(c_vir[2]:%.17g) $(c_vir[3]:%.17g) $(c_vir[4]:%.17g) $(c_vir[5]:%.17g) $(c_vir[6]:%.17g) $(c_mu[1]:%.17g) $(c_mu[2]:%.17g) $(c_mu[3]:%.17g)"',
        f'write_dump all custom {work / "outputs.dump"} id type x y z fx fy fz q modify sort id format float %.17g']
    inp = work / 'in.lammps'
    inp.write_text('\n'.join(commands) + '\n')
    binary = Path(os.environ.get('LSC_LAMMPS_BINARY',
        REPO / '.lsc-reference-logs/kokkos-charge-sync-20261119/bin/lmp-fixed'))
    launcher = shlex.split(os.environ.get('LSC_LAMMPS_LAUNCHER', ''))
    args = launcher + [str(binary), '-k', 'on', 'g', '1', '-sf', 'kk', '-pk',
                       'kokkos', 'newton', 'on', 'neigh', 'half', '-log', 'none', '-in', str(inp)]
    t = time.monotonic()
    proc = subprocess.run(args, cwd=work, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=300)
    (work / 'lammps.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stdout[-7000:] + proc.stderr[-2000:])
    match = re.search(r'^ORGANIC_RESULT (.*)$', proc.stdout, re.M)
    if not match:
        raise RuntimeError('Missing LAMMPS result: ' + proc.stdout[-5000:])
    values = np.fromstring(match.group(1), sep=' ')
    data = np.atleast_2d(np.loadtxt((work / 'outputs.dump').read_text().splitlines()[9:]))
    data = data[np.argsort(data[:, 0])]
    np.testing.assert_array_equal(data[:, 0], np.arange(1, len(atoms)+1))
    observed_species = np.array(species)[data[:, 1].astype(int)-1]
    np.testing.assert_array_equal(observed_species, atoms.numbers)
    positions = data[:, 2:5] @ rotation.T
    formal = {z: q for z, q in zip(info['atomic_numbers'], info['formal_charges'])}
    branch_shift = np.sum(np.array([formal[int(z)] for z in atoms.numbers])[:, None] *
                          (positions - atoms.positions), axis=0)
    result = dict(energy=values[0], forces=data[:, 5:8] @ rotation.T,
                  charges=data[:, 8], dipole_raw=values[7:10] @ rotation.T,
                  positions=positions, dipole_branch_shift=branch_shift,
                  wall_seconds=time.monotonic()-t)
    result['dipole'] = result['dipole_raw'] - branch_shift
    if periodic:
        xx, yy, zz, xy, xz, yz = -values[1:7] / 1.6021766208e6
        stress = rotation @ np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]) @ rotation.T
        result['stress'] = stress[[0,1,2,1,0,0],[0,1,2,2,2,1]]
    else:
        np.testing.assert_allclose(positions, atoms.positions, atol=1e-10, rtol=0)
    for k, v in result.items():
        if not np.isfinite(v).all():
            raise ValueError(f'Nonfinite LAMMPS output {k}')
    np.savez_compressed(work / 'prediction.npz', **result)
    return result


def transformed_case(atoms, kind):
    a = atoms.copy()
    rng = np.random.default_rng(20261120)
    rotation = np.eye(3)
    permutation = np.arange(len(a))
    if kind == 'translation':
        a.positions += [1.234, -2.345, 0.789]
    elif kind == 'rotation':
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation[:, -1] *= np.linalg.det(rotation)
        a.positions = a.positions @ rotation
        a.cell = a.cell.array @ rotation
    elif kind == 'permutation':
        permutation = rng.permutation(len(a))
        a = a[permutation]
    elif kind == 'wrapping':
        a.wrap()
    elif kind != 'species_order':
        raise ValueError(kind)
    return a, rotation, permutation


def cpu_prediction(name, atoms, mesh=False):
    from symmetrix.lsc_calculator import LSCCalculator
    calc = LSCCalculator(root() / 'models' / f'{name}.json', device='cpu',
                         pbc_handling='mixed_periodic' if all(atoms.pbc) else 'realspace')
    if mesh:
        calc.model.set_mesh_electrostatics(0.7, 8)
    a = atoms.copy()
    calc.calculate(a, properties=['energy', 'forces'] + (['stress'] if all(a.pbc) else []))
    return {k: np.asarray(v).copy() for k, v in calc.results.items()}


def fixed_density_cluster_energy(name, atoms, density):
    """Independent FP64 real-space formula, using frozen native densities.

    The reference uses a finite Cartesian displaced-charge stencil for dipoles;
    its total cluster energy is therefore not exactly rotation invariant.
    Reconstruct that contribution so tests can check local-energy invariance
    without altering the original model or suppressing the observed anisotropy.
    """
    from scipy.special import erf
    data=json.loads((root()/'models'/f'{name}.json').read_text())
    p=data['electrostatics']['energy_parameters']
    real=p['realspace'];offset=real['offset']
    offsets=np.vstack([np.zeros(3),real['offset_vectors']])
    rho=np.asarray(density,dtype=np.float64)
    q=np.column_stack([rho[:,0]-rho[:,1:].sum(axis=1)/offset,
                       rho[:,3]/offset,rho[:,1]/offset,rho[:,2]/offset])
    energy=0.
    for i in range(len(atoms)):
        for j in range(i+1,len(atoms)):
            delta=atoms.positions[i]+offsets[:,None,:]-atoms.positions[j]-offsets[None,:,:]
            distance=np.linalg.norm(delta,axis=2)
            kernel=erf(distance/(2*real['sigma']))/(distance+real['distance_epsilon'])
            energy+=np.sum(q[i,:,None]*q[j,None,:]*kernel)
    energy*=p['field_constant']/(4*np.pi)
    if data['electrostatics']['include_self_interaction']:
        energy+=0.5*np.sum(rho**2*np.asarray(real['self_overlap']))
    return float(energy)
