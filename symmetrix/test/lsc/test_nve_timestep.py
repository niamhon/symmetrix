"""Opt-in checks of saved NVE runs; never launch MD or replace a reference."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get('LSC_NVE_ROOT', '/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120'))
MODELS = ['water', 'benzene', 'emc']
BACKENDS = ['native', 'float32', 'float32-mixed']
DTS = [.125, .25, .5, 1.]
pytestmark = pytest.mark.skipif(os.environ.get('LSC_RUN_NVE') != '1',
                                reason='Opt in to saved NVE checks with LSC_RUN_NVE=1')


def records():
    return json.loads((ROOT/'results/runs.json').read_text())


def record(model, backend, dt):
    matches = [r for r in records() if (r['model'], r['backend'], r['timestep_fs']) == (model, backend, dt)]
    assert len(matches) == 1
    assert matches[0]['status'] == 'complete', matches[0]
    return matches[0]


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('dt', DTS)
def test_completed_physical_trajectory(model, backend, dt):
    r = record(model, backend, dt)
    with np.load(Path(r['work'])/'series.npz', allow_pickle=False) as data:
        s = data['series']
    assert s.shape == (1001, 5)
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s[:, 0], np.arange(1001), atol=1e-12, rtol=0)
    np.testing.assert_allclose(s[:, 1]+s[:, 2], s[:, 3], atol=3e-9, rtol=0)
    assert (s[:, 2] > 0).all() and (s[:, 4] > 0).all()
    assert np.max(abs(s[:, 3]-s[0, 3]))/r['atoms'] < 1.0  # Catastrophic-runaway guard only.
    with (Path(r['work'])/'energy.csv').open() as stream:
        reader = csv.reader(stream)
        assert next(reader) == ['time_fs', 'potential_eV', 'kinetic_eV', 'total_eV', 'temperature_K']
        saved = np.array([[float(x) for x in row] for row in reader])
    np.testing.assert_array_equal(saved, s)
    assert r['steps'] == round(r['duration_fs']/dt)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('dt', DTS)
def test_matched_initial_energy_and_forces(model, dt):
    native = record(model, 'native', dt)
    with np.load(Path(native['work'])/'series.npz', allow_pickle=False) as d:
        first = d['series'][0].copy()
    for backend in BACKENDS[1:]:
        r = record(model, backend, dt)
        assert r['initial_energy_error_eV_atom'] < 1e-6
        assert r['initial_kinetic_error_eV_atom'] < 1e-10
        with np.load(Path(r['work'])/'series.npz', allow_pickle=False) as d:
            initial = d['series'][0]
        assert abs(initial[1]-first[1])/r['atoms'] < 1e-6
        np.testing.assert_allclose(initial[2], first[2], atol=1e-8, rtol=0)
        with np.load(ROOT/'states'/f'{model}.native-initial.npz', allow_pickle=False) as ref:
            dump = np.loadtxt((Path(r['work'])/'initial.dump').read_text().splitlines()[9:])
            np.testing.assert_array_equal(dump[:, 0], np.arange(1, r['atoms']+1))
            np.testing.assert_allclose(dump[:, 5:8], ref['forces'], atol=8e-5, rtol=1e-4)


def test_frozen_inputs_and_preserved_installations():
    m = json.loads((ROOT/'manifest.json').read_text())
    assert len(records()) == 36
    assert m['duration_fs'] == 1000
    for name, info in m['models'].items():
        for p, key in [(ROOT/'states'/f'{name}.npz', 'state_sha256'),
                       (ROOT/'models'/f'{name}.model', 'model_sha256'),
                       (ROOT/'models'/f'{name}.json', 'export_sha256')]:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == info[key]
        with np.load(ROOT/'states'/f'{name}.npz', allow_pickle=False) as state:
            assert state['pbc'].all()
            np.testing.assert_allclose(state['momenta'].sum(axis=0), 0, atol=1e-10, rtol=0)
    audit = json.loads((ROOT/'results/audit.json').read_text())
    assert all(audit.values()), audit
