"""Small direct-regression and independent callback-contract checks."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import symmetrix
from .reference import REPO
from .test_cuda_loader import require_cuda

pytestmark=pytest.mark.lsc_reference
BACKENDS=['cpu',pytest.param('float32',marks=pytest.mark.lsc_cuda),pytest.param('float32-mixed',marks=pytest.mark.lsc_cuda)]

@pytest.fixture(scope='module')
def saved():
    path=REPO/'.lsc-reference-logs/electrostatics-interface-baseline-cpu.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['result_sha256']
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}

@pytest.fixture(scope='module',params=BACKENDS)
def backend(request):return request.param

@pytest.fixture
def model(backend):
    path=REPO/'.lsc-reference-logs/water-stage2-energy.json'
    if backend=='cpu':return symmetrix.MACELSC(str(path))
    require_cuda();return symmetrix.MACELSCCuda(str(path),backend)

def inputs(saved,case):
    return [saved[f'{case}/{name}'].copy() for name in ['types','edges','vectors','positions','cell','pbc']]

def physical(model,backend):
    if backend=='cpu':
        out={name:np.asarray(getattr(model,name)) for name in ['forces','position_forces','edge_forces','density_energy_gradient','partial_charges','partial_dipoles','density_coefficients']}
        for name,width in [('forces',3),('position_forces',3),('edge_forces',3),('density_energy_gradient',4)]:out[name]=out[name].reshape(-1,width)
    else:
        out=model.download_forces();out.update(model.download_charges(False))
    for name in ['energy','local_energy','electrostatic_energy','field_energy','dipole','stress','virial','strain_derivative']:out[name]=np.asarray(getattr(model,name))
    return out

@pytest.mark.parametrize('case',['dimer','bulk'])
def test_direct_unchanged(model,backend,saved,case,record_property):
    source='cpu' if backend=='cpu' else 'cuda';precision='float64' if backend=='cpu' else backend
    path=REPO/f'.lsc-reference-logs/electrostatics-interface-baseline-{source}.npz'
    if source=='cuda':path=Path(os.environ.get('LSC_CUDA_DIRECT_BASELINE',str(path)))
    meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['result_sha256']
    if source=='cuda' and 'device' in meta:assert meta['device']==model.device_name,'direct baseline must match the GPU hardware'
    limits = {} if backend == "cpu" else json.loads(Path(os.environ.get("LSC_CUDA_DIRECT_REPEATABILITY",str(REPO/".lsc-reference-logs/electrostatics-interface-cuda-repeatability.json"))).read_text())["bounds"]
    maximum=0.
    assert not model.has_electrostatics_hook
    with np.load(path,allow_pickle=False) as old:
        for label,field in [('zero',(0,0,0)),('field',(.01,-.02,.03))]:
            model.compute_energy_forces(*inputs(saved,case),'auto',field,compute_stress=True)
            for name,actual in physical(model,backend).items():
                values=np.array([old[f'{precision}/{case}/{label}/{i}/{name}'] for i in range(3)])
                gap=np.maximum(values.min(axis=0)-actual,actual-values.max(axis=0))
                maximum=max(maximum,float(np.max(gap,initial=0)))
                # GPU bounds come from an independent run of the unchanged old
                # native library, not from differences produced by this patch.
                np.testing.assert_array_less(gap,1e-10 if backend=='cpu' else limits[name])
    record_property('backend',backend);record_property('case',case);record_property('max_change_outside_before_repeat_range',maximum)
    print(f'INTERFACE_DIRECT {backend} {case} max_change={maximum:.12e}')

@pytest.mark.parametrize('case',['dimer','bulk'])
def test_cpu_direct_passthrough(saved,case):
    model=symmetrix.MACELSC(str(REPO/'.lsc-reference-logs/water-stage2-energy.json'))
    args=inputs(saved,case);field=(.01,-.02,.03)
    model.compute_energy_forces(*args,'auto',field,compute_stress=True)
    expected=physical(model,'cpu');seen=[]
    def hook(request):
        seen.append(request)
        return model.evaluate_direct_electrostatics(request['positions'],request['density_coefficients'],request['cell'],request['pbc'],request['pbc_handling'])
    model.set_electrostatics_hook(hook)
    assert model.has_electrostatics_hook and not model.has_energy_results and not model.has_force_results
    model.compute_energy_forces(*args,'auto',field,compute_stress=True)
    assert len(seen)==1
    for key,value in physical(model,'cpu').items():np.testing.assert_allclose(value,expected[key],atol=1e-10,rtol=0)
    assert seen[0]['pbc_handling']==('realspace' if case=='dimer' else 'mixed_periodic')
    assert seen[0]['parameters']['include_self'] is True
    assert seen[0]['parameters']['kspace_cutoff']==model.kspace_cutoff
    model.set_electrostatics_hook(None)
    assert not model.has_electrostatics_hook and not model.has_energy_results
    model.compute_energy_forces(*args,'auto',field,compute_stress=True)
    for key,value in physical(model,'cpu').items():np.testing.assert_allclose(value,expected[key],atol=1e-10,rtol=0)


def quadratic(request):
    rho=request['density_coefficients'];xyz=request['positions'];volume=abs(np.linalg.det(request['cell']))
    a,b,c=.05,1e-5,.001
    return dict(energy=.5*a*np.sum(rho*rho)+.5*b*np.sum(xyz*xyz)+c*np.log(volume),
                density_gradient=a*rho,position_gradient=b*xyz,
                strain_derivative=b*xyz.T@xyz+c*np.eye(3))

@pytest.mark.parametrize('case',['dimer','bulk'])
def test_hook_energy_and_analytic_reverse(model,backend,saved,case,record_property):
    args=inputs(saved,case);seen=[]
    def hook(request):seen.append(request);return quadratic(request)
    model.set_electrostatics_hook(hook)
    model.compute_energy_forces(*args,'auto',compute_stress=True)
    assert len(seen)==1
    assert abs(model.electrostatic_energy-quadratic(seen[0])['energy'])<1e-12
    assert abs(model.energy-model.local_energy-model.electrostatic_energy)<3e-11
    result=physical(model,backend)
    # Fixed-density position forces and density seeds are checked independently.
    np.testing.assert_allclose(result['position_forces'],-1e-5*args[3],atol=1e-12,rtol=0)
    np.testing.assert_allclose(result['density_energy_gradient'],.05*seen[0]['density_coefficients'],atol=1e-12,rtol=0)
    if backend!='cpu':
        model.evaluate_forces(True);assert len(seen)==1
    if case=='dimer':
        direction=np.random.default_rng(71).normal(size=args[3].shape);direction/=np.linalg.norm(direction)
        step=1e-5 if backend=='cpu' else 1e-3
        energies=[]
        for sign in [-1,1]:
            shifted=list(args);shifted[3]=args[3]+sign*step*direction
            shifted[2]=args[2]+sign*step*(direction[args[1][1]]-direction[args[1][0]])
            energies.append(model.compute_energy(*shifted,'auto'))
        error=abs((energies[1]-energies[0])/(2*step)+np.sum(result['forces']*direction))
        assert error<(3e-6 if backend=='cpu' else 3e-3)
    else:
        from ase import Atoms
        from ase.neighborlist import neighbor_list
        step=1e-5 if backend=='cpu' else 1e-3;energies=[]
        for sign in [-1,1]:
            transform=np.diag([1+sign*step,1,1]);shifted=list(args)
            for index in [3,4]:shifted[index]=args[index]@transform
            atoms=Atoms(numbers=np.asarray(model.atomic_numbers)[args[0]],positions=shifted[3],cell=shifted[4],pbc=args[5])
            sender,receiver,vectors=neighbor_list('ijD',atoms,model.r_cut,self_interaction=False)
            shifted[1]=np.array([sender,receiver]);shifted[2]=vectors
            energies.append(model.compute_energy(*shifted,'auto'))
        error=abs((energies[1]-energies[0])/(2*step)/abs(np.linalg.det(args[4]))-result['stress'][0])
        assert error<(2e-8 if backend=='cpu' else 3e-6)
    record_property('derivative_fd_error',error)
    print(f'INTERFACE_HOOK {backend} {case} fd_error={error:.12e}')
    model.set_electrostatics_hook(None)
    assert not model.has_energy_results and not model.has_force_results and not model.has_stress_results
    model.compute_energy_forces(*args,'auto',compute_stress=True)
    assert not model.has_electrostatics_hook

@pytest.mark.parametrize('failure',['shape','nonfinite'])
def test_invalid_hook_result(saved,failure):
    model=symmetrix.MACELSC(str(REPO/'.lsc-reference-logs/water-stage2-energy.json'))
    def hook(request):
        result=quadratic(request)
        if failure=='shape':result['density_gradient']=result['density_gradient'][:,:3]
        else:result['energy']=np.nan
        return result
    model.set_electrostatics_hook(hook)
    with pytest.raises(ValueError,match='shape|nonfinite'):model.compute_energy_forces(*inputs(saved,'dimer'),'auto')
    assert not model.has_energy_results and not model.has_force_results
    model.set_electrostatics_hook(None)
    model.compute_energy_forces(*inputs(saved,'dimer'),'auto')
    assert model.has_force_results


def test_empty_hook_and_reset(model,backend):
    calls=[]
    def hook(request):
        calls.append(request)
        return dict(energy=2.,density_gradient=np.empty((0,4)),position_gradient=np.empty((0,3)),strain_derivative=np.eye(3)*.1)
    args=[np.empty(0,dtype=np.int64),np.empty((2,0),dtype=np.int64),np.empty((0,3)),np.empty((0,3)),np.eye(3)*30,np.zeros(3,dtype=bool)]
    model.set_electrostatics_hook(hook)
    assert model.compute_energy_forces(*args,'auto',compute_stress=True)==2.
    assert len(calls)==1
    np.testing.assert_allclose(model.stress,np.array([.1,.1,.1,0,0,0])/27000,atol=1e-20,rtol=1e-14)
    model.set_electrostatics_hook(None)
    assert model.compute_energy_forces(*args,'auto',compute_stress=True)==0.
