"""One-process fixed-density mesh energy versus the native direct operator."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import symmetrix
from .reference import DATA,REPO,load_atoms

pytestmark=pytest.mark.lsc_reference

@pytest.fixture(scope='module')
def model():
    if not symmetrix.lsc_cpu_mesh_compiled():
        if os.environ.get('LSC_REQUIRE_CPU_MESH')=='1':pytest.fail('CPU mesh/FFTW support is required')
        pytest.skip('requires SYMMETRIX_LSC_MESH=ON and FFTW3')
    return symmetrix.MACELSC(str(REPO/'.lsc-reference-logs/water-stage2-energy.json'))

@pytest.fixture(scope='module')
def bulk():
    atoms=load_atoms('bulk')
    with np.load(DATA/'bulk.reference.npz',allow_pickle=False) as saved:rho=saved['density_coefficients'].reshape(-1,4).copy()
    return [atoms.positions.copy(),rho,atoms.cell.array.copy(),atoms.pbc.copy()]

@pytest.fixture(scope='module')
def no_self_model(model,tmp_path_factory):
    data=json.loads((REPO/'.lsc-reference-logs/water-stage2-energy.json').read_text())
    data['electrostatics']['include_self_interaction']=False
    path=tmp_path_factory.mktemp('lsc-mesh-no-self')/'model.json';path.write_text(json.dumps(data))
    return symmetrix.MACELSC(str(path))


def test_bulk_energy_convergence(model,bulk,record_property):
    direct=model.evaluate_direct_electrostatics(*bulk,'mixed_periodic')
    errors=[];gradient_errors=[]
    for spacing in [.7,.5,.35]:
        mesh=symmetrix.LSCCPUMesh(spacing,8);actual=mesh.evaluate_energy(model,*bulk)
        error=abs(actual['energy']-direct['energy']);errors.append(error)
        gradient_errors.append(float(np.max(abs(actual['density_gradient']-direct['density_gradient']))))
        assert mesh.can_evaluate_energy and mesh.can_evaluate_forces and mesh.can_evaluate_stress
        assert not actual['can_evaluate_forces'] and not actual['can_evaluate_stress']
        assert abs(actual['energy']-actual['reciprocal_energy']-actual['self_correction'])<1e-14
        record_property(f'energy_error_spacing_{spacing}',error)
        record_property(f'density_gradient_error_spacing_{spacing}',gradient_errors[-1])
        print(f'CPU_MESH spacing={spacing} grid={actual["grid_shape"]} energy_error_eV={error:.12e} density_gradient_error={gradient_errors[-1]:.12e}')
    assert errors[0]<5e-7 and errors[2]<2e-9
    assert errors[2]<errors[1]<errors[0]
    assert gradient_errors[2]<gradient_errors[0]/20

@pytest.mark.parametrize('component',['charge','dipole','mixed'])
def test_density_components(model,bulk,component):
    args=[x.copy() for x in bulk]
    if component=='charge':args[1][:,1:]=0
    if component=='dipole':args[1][:,0]=0
    mesh=symmetrix.LSCCPUMesh(.35,8)
    actual=mesh.evaluate_energy(model,*args);direct=model.evaluate_direct_electrostatics(*args,'mixed_periodic')
    assert abs(actual['energy']-direct['energy'])<2e-9


def test_density_gradient_and_energy_identity(model,bulk):
    mesh=symmetrix.LSCCPUMesh(.5,8);actual=mesh.evaluate_energy(model,*bulk)
    assert abs(actual['energy']-.5*np.sum(bulk[1]*actual['density_gradient']))<1e-11
    direction=np.random.default_rng(93).normal(size=bulk[1].shape);direction/=np.linalg.norm(direction)
    energies=[];step=1e-5
    for sign in [-1,1]:
        args=list(bulk);args[1]=bulk[1]+sign*step*direction
        energies.append(mesh.evaluate_energy(model,*args)['energy'])
    difference=(energies[1]-energies[0])/(2*step)
    assert abs(difference-np.sum(actual['density_gradient']*direction))<2e-8


def test_self_correction(model,no_self_model,bulk):
    mesh=symmetrix.LSCCPUMesh(.35,8)
    with_self=mesh.evaluate_energy(model,*bulk);without=mesh.evaluate_energy(no_self_model,*bulk)
    direct_with=model.evaluate_direct_electrostatics(*bulk,'mixed_periodic');direct_without=no_self_model.evaluate_direct_electrostatics(*bulk,'mixed_periodic')
    assert abs(without['energy']-direct_without['energy'])<2e-9
    assert abs(without['self_correction']-(direct_without['energy']-direct_with['energy']))<1e-10
    np.testing.assert_allclose(with_self['density_gradient']-without['density_gradient'],direct_with['density_gradient']-direct_without['density_gradient'],atol=1e-12,rtol=0)


def test_periodic_wrapping_and_reuse(model,bulk):
    mesh=symmetrix.LSCCPUMesh(.35,8);original=mesh.evaluate_energy(model,*bulk)
    shifted=list(bulk);shifted[0]=bulk[0]+np.array([2,-1,3])@bulk[2]
    translated=mesh.evaluate_energy(model,*shifted)
    assert abs(original['energy']-translated['energy'])<1e-12
    np.testing.assert_allclose(original['density_gradient'],translated['density_gradient'],atol=2e-12,rtol=0)
    shear=np.array([[1.05,.12,0],[0,1.,.08],[0,0,.98]])
    tilted=list(bulk);tilted[0]=bulk[0]@shear;tilted[2]=bulk[2]@shear
    actual=mesh.evaluate_energy(model,*tilted);direct=model.evaluate_direct_electrostatics(*tilted,'mixed_periodic')
    assert abs(actual['energy']-direct['energy'])<5e-9
    again=mesh.evaluate_energy(model,*bulk)
    assert abs(again['energy']-original['energy'])<1e-13


def test_zero_density_and_empty_graph(model,bulk):
    mesh=symmetrix.LSCCPUMesh(.7,8)
    args=list(bulk);args[1]=np.zeros_like(bulk[1]);actual=mesh.evaluate_energy(model,*args)
    assert actual['energy']==0 and not np.any(actual['density_gradient'])
    args[0]=np.empty((0,3));args[1]=np.empty((0,4));actual=mesh.evaluate_energy(model,*args)
    assert actual['energy']==0 and actual['density_gradient'].shape==(0,4)

@pytest.mark.parametrize('spacing,order',[(0.,8),(np.nan,8),(.7,3),(.7,11)])
def test_invalid_mesh_options(model,spacing,order):
    with pytest.raises(ValueError,match='spacing|order'):symmetrix.LSCCPUMesh(spacing,order)


def test_invalid_periodicity_and_inputs(model,bulk):
    mesh=symmetrix.LSCCPUMesh(.7,8)
    dimer=load_atoms('dimer');rho=np.zeros((len(dimer),4))
    with pytest.raises(ValueError,match='periodic'):mesh.evaluate_energy(model,dimer.positions,rho,dimer.cell.array,dimer.pbc)
    singular=list(bulk);singular[2]=np.zeros((3,3))
    with pytest.raises(ValueError,match='nonsingular'):mesh.evaluate_energy(model,*singular)
    bad=list(bulk);bad[1]=bulk[1].copy();bad[1][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):mesh.evaluate_energy(model,*bad)
    huge=symmetrix.LSCCPUMesh(1e-8,8)
    with pytest.raises(ValueError,match='too large'):huge.evaluate_energy(model,*bulk)
    # Failure recovery uses the original valid grid and density again.
    assert np.isfinite(mesh.evaluate_energy(model,*bulk)['energy'])


def test_optional_stub_builds_without_fftw(tmp_path):
    source=tmp_path/'stub.cpp';source.write_text('#include "mace_lsc_mesh.hpp"\n#include <stdexcept>\nint main(){if(lsc_cpu_mesh_compiled())return 1;try{LSCCPUMesh mesh;}catch(const std::runtime_error&){return 0;}return 2;}\n')
    binary=tmp_path/'stub'
    result=subprocess.run(['g++','-std=c++20','-I'+str(REPO/'libsymmetrix/source'),str(source),str(REPO/'libsymmetrix/source/mace_lsc_mesh_stub.cpp'),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    subprocess.run([str(binary)],check=True)
