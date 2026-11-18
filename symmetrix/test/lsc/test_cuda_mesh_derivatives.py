"""Analytic one-process mesh derivatives, direct convergence and finite differences."""
import numpy as np
import pytest
import symmetrix
from .test_cpu_mesh_energy import model,bulk,no_self_model
from .test_cuda_loader import require_cuda

@pytest.fixture(scope="module",autouse=True)
def cuda_required():require_cuda()

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_cuda]

@pytest.mark.parametrize('tilted',[False,True])
def test_fixed_density_direct_convergence(model,bulk,tilted,record_property):
    args=[x.copy() for x in bulk]
    if tilted:
        shear=np.array([[1.05,.12,0],[0,1.,.08],[0,0,.98]])
        args[0]=args[0]@shear;args[2]=args[2]@shear
    direct=model.evaluate_direct_electrostatics(*args,'mixed_periodic')
    errors=[]
    for spacing in [.7,.35]:
        mesh=symmetrix.LSCCudaMesh(spacing,8);actual=mesh.evaluate(model,*args)
        old=mesh.evaluate_energy(model,*args)
        assert actual['can_evaluate_forces'] and actual['can_evaluate_stress']
        assert abs(actual['energy']-old['energy'])<2e-15
        np.testing.assert_allclose(actual['density_gradient'],old['density_gradient'],atol=2e-15,rtol=0)
        row={key:float(np.max(abs(actual[key]-direct[key]))) for key in ['energy','position_gradient','density_gradient','strain_derivative']}
        errors.append(row)
        for key,value in row.items():record_property(f'{key}_error_spacing_{spacing}',value)
        print(f'CUDA_MESH_DERIV tilted={tilted} spacing={spacing} errors={row}')
        bounds=[5e-7,2e-6,2e-6,2e-5] if spacing==.7 else [6e-9,3e-8,3e-8,3e-7]
        for key,bound in zip(row,bounds):assert row[key]<bound,(key,row)
    assert errors[1]['position_gradient']<errors[0]['position_gradient']/20
    assert errors[1]['strain_derivative']<errors[0]['strain_derivative']/20

@pytest.mark.parametrize('tilted',[False,True])
def test_fixed_density_finite_differences(model,bulk,tilted):
    args=[x.copy() for x in bulk]
    if tilted:
        transform=np.array([[1.,.1,0],[0,1.,.08],[0,0,1.]])
        args[0]=args[0]@transform;args[2]=args[2]@transform
    mesh=symmetrix.LSCCudaMesh(.5,8);actual=mesh.evaluate(model,*args)
    step=1e-5;direction=np.random.default_rng(169).normal(size=args[0].shape);direction/=np.linalg.norm(direction)
    energies=[]
    for sign in [-1,1]:
        changed=list(args);changed[0]=args[0]+sign*step*direction
        energies.append(mesh.evaluate_energy(model,*changed)['energy'])
    assert abs((energies[1]-energies[0])/(2*step)-np.sum(actual['position_gradient']*direction))<2e-8
    for a,b in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
        direction=np.zeros((3,3));direction[a,b]=direction[b,a]=1 if a==b else .5
        energies=[]
        for sign in [-1,1]:
            transform=np.eye(3)+sign*step*direction;changed=list(args)
            changed[0]=args[0]@transform;changed[2]=args[2]@transform
            energies.append(mesh.evaluate_energy(model,*changed)['energy'])
        numerical=(energies[1]-energies[0])/(2*step)
        assert abs(numerical-np.sum(actual['strain_derivative']*direction))<3e-8,(a,b,numerical,actual['strain_derivative'])


def test_self_correction_has_no_explicit_geometry_derivative(model,no_self_model,bulk):
    mesh=symmetrix.LSCCudaMesh(.5,8)
    actual=mesh.evaluate(model,*bulk);without=mesh.evaluate(no_self_model,*bulk)
    for key in ['position_gradient','strain_derivative']:np.testing.assert_allclose(actual[key],without[key],atol=2e-14,rtol=0)
    direct=no_self_model.evaluate_direct_electrostatics(*bulk,'mixed_periodic')
    np.testing.assert_allclose(without['position_gradient'],direct['position_gradient'],atol=3e-7,rtol=0)


def test_empty_and_zero_density(model,bulk):
    mesh=symmetrix.LSCCudaMesh(.7,8);args=list(bulk);args[1]=np.zeros_like(args[1])
    for empty in [False,True]:
        if empty:args[0]=np.empty((0,3));args[1]=np.empty((0,4))
        actual=mesh.evaluate(model,*args)
        assert actual['energy']==0
        for key in ['density_gradient','position_gradient','strain_derivative']:assert not np.any(actual[key])
        assert actual['position_gradient'].shape==(0 if empty else len(bulk[0]),3)



@pytest.mark.parametrize('precision',['float32','float32-mixed'])
@pytest.mark.parametrize('field',[(0,0,0),(.01,-.02,.03)])
def test_full_gpu_mesh_against_direct(model,precision,field,record_property):
    from .reference import load_atoms,REPO
    from symmetrix.lsc_calculator import LSCCalculator
    atoms=load_atoms('bulk');calc=LSCCalculator(REPO/'.lsc-reference-logs/water-stage2-energy.json',device='cuda',precision=precision,external_field=field)
    calc.calculate(atoms);direct={k:np.asarray(v).copy() for k,v in calc.results.items()}
    for spacing in [.7,.35]:
        calc.model.set_mesh_electrostatics(spacing,8)
        assert not calc.model.has_energy_results and not calc.model.has_force_results
        calc.calculate(atoms)
        errors={key:float(np.max(abs(np.asarray(calc.results[key])-direct[key]))) for key in ['energy','forces','stress']}
        for key,value in errors.items():record_property(f'{key}_error_{spacing}',value)
        print(f'CUDA_MESH_MODEL precision={precision} field={field} spacing={spacing} errors={errors}')
        assert errors['energy']<5e-7
        assert errors['forces']<3e-6
        assert errors['stress']<5e-9
        np.testing.assert_allclose(calc.results['charges'],direct['charges'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(calc.results['dipole'],direct['dipole'],atol=2e-7,rtol=0)
    calc.model.set_electrostatics_hook(None);assert not calc.model.has_energy_results
    calc.calculate(atoms)
    for key,limit in [('energy',1e-8),('forces',3e-6),('stress',5e-9)]:np.testing.assert_allclose(calc.results[key],direct[key],atol=limit,rtol=0)


@pytest.mark.parametrize('precision',['float32','float32-mixed'])
def test_full_gpu_finite_difference_and_nonperiodic_rejection(model,precision,record_property):
    from .reference import load_atoms,REPO
    from symmetrix.lsc_calculator import LSCCalculator
    atoms=load_atoms('bulk');calc=LSCCalculator(REPO/'.lsc-reference-logs/water-stage2-energy.json',device='cuda',precision=precision,external_field=(.01,-.02,.03))
    calc.model.set_mesh_electrostatics(.5,8);calc.calculate(atoms)
    forces=calc.results['forces'].copy();stress=calc.results['stress'].copy();step=1e-3
    direction=np.random.default_rng(194).normal(size=forces.shape);direction/=np.linalg.norm(direction)
    energies=[]
    for sign in [-1,1]:
        changed=atoms.copy();changed.positions+=sign*step*direction;calc.calculate(changed);energies.append(calc.results['energy'])
    force_error=abs((energies[1]-energies[0])/(2*step)+np.sum(forces*direction));assert force_error<3e-3
    errors=[]
    for component,(a,b) in [(0,(0,0)),(3,(1,2))]:
        direction=np.zeros((3,3));direction[a,b]=direction[b,a]=1 if a==b else .5;energies=[]
        for sign in [-1,1]:
            changed=atoms.copy();changed.set_cell(atoms.cell.array@(np.eye(3)+sign*step*direction),scale_atoms=True)
            calc.calculate(changed);energies.append(calc.results['energy'])
        error=abs((energies[1]-energies[0])/(2*step)/atoms.get_volume()-stress[component]);errors.append(error);assert error<3e-6
    record_property('force_direction_fd_error',force_error);record_property('stress_fd_error',max(errors))
    print(f'CUDA_MESH_FULL_FD precision={precision} force_error={force_error:.12e} stress_error={max(errors):.12e}')
    with pytest.raises(ValueError,match='periodic'):calc.calculate(load_atoms('dimer'))
    calc.model.set_electrostatics_hook(None);calc.calculate(load_atoms('dimer'));assert calc.model.has_force_results
