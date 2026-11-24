"""Opt-in clean-release checks against frozen original FP64 native predictions.

Set SYMMETRIX_RELEASE_FIXTURES to a manifest containing data/export/reference
paths. No native reference is generated or modified by these tests.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase.io import read


@pytest.fixture(scope='module')
def fixtures():
    path=os.environ.get('SYMMETRIX_RELEASE_FIXTURES')
    if not path:pytest.skip('opt in with SYMMETRIX_RELEASE_FIXTURES')
    return json.loads(Path(path).read_text())


def atoms(fixtures):
    return read(fixtures['data'],format='lammps-data',atom_style='charge',Z_of_type={1:1,2:8},units='metal')


def calculator(kind,precision,fixtures):
    from symmetrix.experimental import MACEFoundation,MACEPolar
    cls=MACEPolar if kind=='polar' else MACEFoundation
    return cls(fixtures[kind]['export'],device='cuda',dtype=precision)


@pytest.mark.parametrize('kind',['polar','mp'])
@pytest.mark.parametrize('precision',['float32','float32-mixed'])
def test_saved_native_efs(kind,precision,fixtures):
    a=atoms(fixtures);a.calc=calculator(kind,precision,fixtures)
    with np.load(fixtures[kind]['reference']) as ref:
        assert abs(a.get_potential_energy()-float(ref['energy']))/len(a)<1e-6
        np.testing.assert_allclose(a.get_forces(),ref['forces'],atol=8e-5,rtol=1e-4)
        np.testing.assert_allclose(a.get_stress(),ref['stress'],atol=5e-7,rtol=1e-4)
    assert abs(a.get_potential_energies().sum()-a.get_potential_energy())/len(a)<1e-8


@pytest.mark.parametrize('device,dtype,match',[('gpu',None,'device'),('cpu','float32','float64'),('cuda','float64','float32')])
def test_polar_rejects_invalid_backend(device,dtype,match):
    from symmetrix.experimental import MACEPolar
    with pytest.raises(ValueError,match=match):MACEPolar('missing.json',device=device,dtype=dtype)


def test_polar_rejects_invalid_atoms(fixtures):
    calc=calculator('polar','float32',fixtures)
    a=atoms(fixtures);a.pbc=[True,True,False];a.calc=calc
    with pytest.raises(ValueError,match='periodic'):a.get_potential_energy()
    b=atoms(fixtures);b.numbers[0]=54;b.calc=calc
    with pytest.raises(ValueError,match='atomic numbers'):b.get_potential_energy()
    c=atoms(fixtures);c.cell[2]=c.cell[0];c.calc=calc
    with pytest.raises(ValueError,match='volume'):c.get_potential_energy()


def test_polar_charge_spin_cache(fixtures):
    a=atoms(fixtures);a.calc=calculator('polar','float32',fixtures)
    e=a.get_potential_energy();a.info.update(charge=1,spin=2)
    assert 'info' in a.calc.check_state(a)
    charged=a.get_potential_energy()
    assert np.isfinite(charged) and charged!=e
    b=atoms(fixtures);b.info.update(charge=1,spin=2);b.calc=calculator('polar','float32',fixtures)
    assert abs(b.get_potential_energy()-charged)<1e-8
