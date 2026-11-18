"""Basic GPU mesh energy/potentials versus FFTW and direct with identical rho/parameters."""
import numpy as np
import pytest
import symmetrix
from .test_cpu_mesh_energy import model,bulk,no_self_model
from .test_cuda_loader import require_cuda

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_cuda]

@pytest.fixture(scope='module')
def gpu_mesh():
    require_cuda()
    assert symmetrix.lsc_cuda_mesh_compiled()
    return symmetrix.LSCCudaMesh(.7,8)

@pytest.mark.parametrize('tilted',[False,True])
def test_mesh_energy_potential_convergence(model,bulk,gpu_mesh,tilted,record_property):
    args=[x.copy() for x in bulk]
    if tilted:
        transform=np.array([[1.05,.12,0],[0,1.,.08],[0,0,.98]])
        args[0]=args[0]@transform;args[2]=args[2]@transform
    direct=model.evaluate_direct_electrostatics(*args,'mixed_periodic');errors=[]
    for spacing in [.7,.35]:
        gpu=symmetrix.LSCCudaMesh(spacing,8);cpu=symmetrix.LSCCPUMesh(spacing,8)
        actual=gpu.evaluate_energy(model,*args);expected=cpu.evaluate_energy(model,*args)
        assert abs(actual['energy']-expected['energy'])<2e-12
        np.testing.assert_allclose(actual['density_gradient'],expected['density_gradient'],atol=3e-12,rtol=0)
        assert actual['grid_shape']==expected['grid_shape']
        assert not actual['can_evaluate_forces'] and not actual['can_evaluate_stress']
        errors.append(abs(actual['energy']-direct['energy']))
        assert errors[-1]<(5e-7 if spacing==.7 else 6e-9)
        record_property(f'energy_error_{spacing}',errors[-1]);record_property('device',gpu.device_name)
        print(f'CUDA_MESH_ENERGY tilted={tilted} spacing={spacing} device={gpu.device_name} error={errors[-1]:.12e}')
    assert errors[1]<errors[0]/20

@pytest.mark.parametrize('component',['charge','dipole','mixed'])
def test_density_components_and_self(model,no_self_model,bulk,gpu_mesh,component):
    args=[x.copy() for x in bulk]
    if component=='charge':args[1][:,1:]=0
    if component=='dipole':args[1][:,0]=0
    cpu=symmetrix.LSCCPUMesh(.7,8)
    for selected in [model,no_self_model]:
        actual=gpu_mesh.evaluate_energy(selected,*args);expected=cpu.evaluate_energy(selected,*args)
        assert abs(actual['energy']-expected['energy'])<2e-12
        np.testing.assert_allclose(actual['density_gradient'],expected['density_gradient'],atol=3e-12,rtol=0)
        assert abs(actual['energy']-.5*np.sum(args[1]*actual['density_gradient']))<2e-12


def test_potential_finite_difference(model,bulk,gpu_mesh):
    actual=gpu_mesh.evaluate_energy(model,*bulk)
    direction=np.random.default_rng(193).normal(size=bulk[1].shape);direction/=np.linalg.norm(direction)
    step=1e-5;values=[]
    for sign in [-1,1]:
        args=list(bulk);args[1]=bulk[1]+sign*step*direction;values.append(gpu_mesh.evaluate_energy(model,*args)['energy'])
    assert abs((values[1]-values[0])/(2*step)-np.sum(actual['density_gradient']*direction))<2e-8


def test_reuse_wrapping_and_empty(model,bulk,gpu_mesh):
    original=gpu_mesh.evaluate_energy(model,*bulk);generation=gpu_mesh.plan_generation
    shifted=list(bulk);shifted[0]=bulk[0]+np.array([2,-1,3])@bulk[2]
    wrapped=gpu_mesh.evaluate_energy(model,*shifted)
    assert gpu_mesh.plan_generation==generation
    np.testing.assert_allclose(wrapped['density_gradient'],original['density_gradient'],atol=3e-12,rtol=0)
    scaled=list(bulk);scaled[0]=bulk[0]*1.2;scaled[2]=bulk[2]*1.2
    gpu_mesh.evaluate_energy(model,*scaled);assert gpu_mesh.plan_generation==generation+1
    args=list(bulk);args[1]=np.zeros_like(args[1]);zero=gpu_mesh.evaluate_energy(model,*args)
    assert zero['energy']==0 and not np.any(zero['density_gradient'])
    args[0]=np.empty((0,3));args[1]=np.empty((0,4));empty=gpu_mesh.evaluate_energy(model,*args)
    assert empty['energy']==0 and empty['density_gradient'].shape==(0,4)


def test_invalid_inputs(model,bulk,gpu_mesh):
    for spacing,order in [(0.,8),(np.nan,8),(.7,3),(.7,11)]:
        with pytest.raises(ValueError,match='spacing|order'):symmetrix.LSCCudaMesh(spacing,order)
    for index,value,match in [(3,np.zeros(3,dtype=bool),'periodic'),(2,np.zeros((3,3)),'nonsingular'),(0,np.full_like(bulk[0],np.nan),'finite')]:
        args=list(bulk);args[index]=value
        with pytest.raises(ValueError,match=match):gpu_mesh.evaluate_energy(model,*args)
    with pytest.raises(ValueError,match='too large'):symmetrix.LSCCudaMesh(1e-8,8).evaluate_energy(model,*bulk)
    assert np.isfinite(gpu_mesh.evaluate_energy(model,*bulk)['energy'])
