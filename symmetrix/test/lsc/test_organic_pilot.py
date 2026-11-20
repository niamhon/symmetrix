"""Opt-in snapshot parity for frozen benzene/EMC stage-3 models.

Run tools/lsc/organic_pilot.py prepare/reference/compare explicitly first.
References are never generated or replaced by pytest.
"""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import organic as o

pytestmark = [pytest.mark.lsc_organic, pytest.mark.lsc_cuda, pytest.mark.lsc_lammps,
              pytest.mark.skipif(os.environ.get('LSC_RUN_ORGANIC')!='1',
                                 reason='opt in with LSC_RUN_ORGANIC=1')]
MODELS = ['benzene', 'emc']
PRECISIONS = ['float32', 'float32-mixed']
CASES = [(name,i) for name,n in [('benzene',22),('emc',23)] for i in range(n)]


@pytest.fixture(scope='module', autouse=True)
def integrity():
    m = o.manifest()
    assert m['provenance']['commit'] == o.PIN
    binary = os.environ.get('LSC_LAMMPS_BINARY',m['provenance']['binary'])
    assert o.sha256(binary) == o.BINARY_SHA
    assert m['reference_dtype']=='float64' and not m['reference_use_compile']
    for name,info in m['models'].items():
        for directory,suffix,key in [('models','model','model_sha256'),
                                     ('models','json','export_sha256'),
                                     ('fixtures','extxyz','fixtures_sha256'),
                                     ('references','npz','reference_sha256')]:
            assert o.sha256(o.root()/directory/f'{name}.{suffix}') == info[key]


def assert_prediction(actual,expected,atoms):
    assert abs(float(actual['energy'])-float(expected['energy']))/len(atoms)<1e-6
    for key,atol,rtol in [('forces',8e-5,1e-4),('charges',2e-6,1e-5),('dipole',3e-5,0)]:
        np.testing.assert_allclose(actual[key],expected[key],atol=atol,rtol=rtol,err_msg=key)
    if all(atoms.pbc):
        np.testing.assert_allclose(actual['stress'],expected['stress'],atol=5e-7,rtol=1e-4)


@pytest.mark.parametrize('name,case_id',CASES)
@pytest.mark.parametrize('precision',PRECISIONS)
def test_snapshot(name,case_id,precision,tmp_path,record_property):
    atoms = o.load_case(name,case_id)
    if os.environ.get('LSC_ORGANIC_REUSE_RESULTS') == '1':
        # Only reuse the pinned binary's freshly recorded comparison-stage data.
        work=o.root()/'results'/name/precision/f'{case_id:03d}'
        with np.load(work/'prediction.npz',allow_pickle=False) as p:
            actual={k:p[k].copy() for k in p}
    else:
        actual=o.lammps(name,atoms,precision,tmp_path)
    expected=o.reference(name,case_id)
    metrics=o.compare(actual,expected,atoms)
    for k,v in metrics.items():
        if isinstance(v,(int,float)):record_property(k,v)
    assert_prediction(actual,expected,atoms)
    info=o.manifest()['models'][name]
    formal=np.array(info['formal_charges'])
    target=sum(formal[info['atomic_numbers'].index(int(z))] for z in atoms.numbers)
    assert abs(actual['charges'].sum()-target)<1e-8


@lru_cache(None)
def torch_calculator(name,periodic):
    return o.native_calculator(name,'mixed_periodic' if periodic else 'realspace')


def representative(name,periodic):
    info=o.manifest()['models'][name]
    group='periodic' if periodic else 'cluster'
    return next(c['id'] for c in info['cases'] if c['group']==group)


TRANSFORMS=[(p,k) for p in [False,True]
            for k in ['translation','rotation','permutation','species_order']+(['wrapping'] if p else [])]


@pytest.mark.parametrize('name',MODELS)
@pytest.mark.parametrize('precision',PRECISIONS)
@pytest.mark.parametrize('periodic,kind',TRANSFORMS)
def test_coordinate_and_species_conventions(name,precision,periodic,kind,tmp_path,record_property):
    case=representative(name,periodic)
    original=o.load_case(name,case)
    atoms,Q,permutation=o.transformed_case(original,kind)
    expected=o.evaluate_native(atoms,torch_calculator(name,periodic))
    np.savez_compressed(tmp_path/'native.npz',**expected)
    actual=o.lammps(name,atoms,precision,tmp_path,reverse_species=kind=='species_order')
    metrics=o.compare(actual,expected,atoms)
    o.atomic_json(tmp_path/'accuracy.json',metrics)
    record_property('force_error',metrics['forces_max_eV_A'])
    assert_prediction(actual,expected,atoms)
    # Independent native invariants catch mistakes in the transform/reference plumbing.
    base=o.reference(name,case)
    energy_change=float(expected['energy'])-float(base['energy'])
    force_change=float(np.max(abs(expected['forces']-base['forces'][permutation]@Q)))
    if kind=='rotation' and not periodic:
        # Native real-space dipoles use a finite Cartesian charge stencil.
        # Account for its orientation dependence; do not loosen parity gates.
        electrostatic_change=(o.fixed_density_cluster_energy(name,atoms,expected['density_coefficients'])-
                              o.fixed_density_cluster_energy(name,original,base['density_coefficients']))
        diagnostic=dict(native_energy_change_eV=energy_change,
                        native_electrostatic_change_eV=electrostatic_change,
                        remaining_local_change_eV=energy_change-electrostatic_change,
                        native_force_covariance_error_eV_A=force_change)
        o.atomic_json(tmp_path/'native-rotation-diagnostic.json',diagnostic)
        record_property('native_cluster_rotation_energy_eV',energy_change)
        record_property('native_cluster_rotation_force_eV_A',force_change)
        assert abs(energy_change-electrostatic_change)<1e-8
    else:
        assert abs(energy_change)/len(atoms)<1e-9
        np.testing.assert_allclose(expected['forces'],base['forces'][permutation]@Q,atol=1e-7,rtol=1e-8)
    np.testing.assert_allclose(expected['charges'],base['charges'][permutation],atol=1e-9,rtol=1e-8)
    info=o.manifest()['models'][name]
    formal=np.array([info['formal_charges'][info['atomic_numbers'].index(int(z))] for z in atoms.numbers])
    shift=np.sum(formal[:,None]*(atoms.positions-original.positions[permutation]@Q),axis=0)
    np.testing.assert_allclose(expected['dipole'],base['dipole']@Q+shift,atol=1e-7,rtol=1e-8)


@pytest.mark.parametrize('name',MODELS)
@pytest.mark.parametrize('periodic',[False,True])
def test_cpu_export_spot_check(name,periodic,tmp_path,record_property):
    case=representative(name,periodic);atoms=o.load_case(name,case)
    expected=o.reference(name,case)
    for mesh in ([False,True] if periodic else [False]):
        actual=o.cpu_prediction(name,atoms,mesh=mesh)
        np.savez_compressed(tmp_path/f'cpu-mesh-{mesh}.npz',**actual)
        metrics=o.compare(actual,expected,atoms)
        o.atomic_json(tmp_path/f'cpu-mesh-{mesh}.json',metrics)
        record_property(f'force_error_mesh_{mesh}',metrics['forces_max_eV_A'])
        assert_prediction(actual,expected,atoms)


@pytest.mark.parametrize('name',MODELS)
def test_native_force_finite_difference(name,tmp_path,record_property):
    case=representative(name,False);a=o.load_case(name,case);calc=torch_calculator(name,False)
    direction=np.random.default_rng(20261120).normal(size=a.positions.shape)
    direction/=np.linalg.norm(direction)
    expected=o.reference(name,case)
    target=-float(np.sum(expected['forces']*direction))
    estimates=[]
    for step in [1e-4,5e-5]:
        energy=[]
        for sign in [-1,1]:
            b=a.copy();b.positions+=sign*step*direction
            energy.append(float(o.evaluate_native(b,calc)['energy']))
        estimates.append((energy[1]-energy[0])/(2*step))
    o.atomic_json(tmp_path/'finite-difference.json',dict(analytic=target,numerical=estimates))
    error=abs(estimates[-1]-target);record_property('directional_force_error_eV_A',error)
    assert error<5e-5
    assert abs(estimates[1]-estimates[0])<5e-5


@pytest.mark.parametrize('name',MODELS)
def test_native_stress_finite_difference(name,tmp_path,record_property):
    case=representative(name,True);a=o.load_case(name,case);calc=torch_calculator(name,True)
    target=o.reference(name,case)['stress'];estimates=[]
    for step in [1e-5,5e-6]:
        values=[]
        for i,j in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
            strain=np.zeros((3,3));strain[i,j]=step if i==j else step/2
            if i!=j:strain[j,i]=step/2
            energy=[]
            for sign in [-1,1]:
                b=a.copy();deformation=np.eye(3)+sign*strain
                b.positions=a.positions@deformation;b.cell=a.cell.array@deformation
                energy.append(float(o.evaluate_native(b,calc)['energy']))
            values.append((energy[1]-energy[0])/(2*step*a.get_volume()))
        estimates.append(values)
    o.atomic_json(tmp_path/'finite-difference.json',dict(analytic=target.tolist(),numerical=estimates))
    error=float(np.max(abs(np.asarray(estimates[-1])-target)))
    record_property('stress_fd_error_eV_A3',error)
    assert error<2e-8
    assert np.max(abs(np.asarray(estimates[1])-estimates[0]))<2e-8
