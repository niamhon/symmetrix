"""Analytic one-process mesh derivatives, direct convergence and finite differences."""
import numpy as np
import pytest
import symmetrix
from .test_cpu_mesh_energy import model,bulk,no_self_model

pytestmark=pytest.mark.lsc_reference

@pytest.mark.parametrize('tilted',[False,True])
def test_fixed_density_direct_convergence(model,bulk,tilted,record_property):
    args=[x.copy() for x in bulk]
    if tilted:
        shear=np.array([[1.05,.12,0],[0,1.,.08],[0,0,.98]])
        args[0]=args[0]@shear;args[2]=args[2]@shear
    direct=model.evaluate_direct_electrostatics(*args,'mixed_periodic')
    errors=[]
    for spacing in [.7,.35]:
        mesh=symmetrix.LSCCPUMesh(spacing,8);actual=mesh.evaluate(model,*args)
        old=mesh.evaluate_energy(model,*args)
        assert actual['can_evaluate_forces'] and actual['can_evaluate_stress']
        assert abs(actual['energy']-old['energy'])<2e-15
        np.testing.assert_allclose(actual['density_gradient'],old['density_gradient'],atol=2e-15,rtol=0)
        row={key:float(np.max(abs(actual[key]-direct[key]))) for key in ['energy','position_gradient','density_gradient','strain_derivative']}
        errors.append(row)
        for key,value in row.items():record_property(f'{key}_error_spacing_{spacing}',value)
        print(f'CPU_MESH_DERIV tilted={tilted} spacing={spacing} errors={row}')
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
    mesh=symmetrix.LSCCPUMesh(.5,8);actual=mesh.evaluate(model,*args)
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
    mesh=symmetrix.LSCCPUMesh(.5,8)
    actual=mesh.evaluate(model,*bulk);without=mesh.evaluate(no_self_model,*bulk)
    for key in ['position_gradient','strain_derivative']:np.testing.assert_array_equal(actual[key],without[key])
    direct=no_self_model.evaluate_direct_electrostatics(*bulk,'mixed_periodic')
    np.testing.assert_allclose(without['position_gradient'],direct['position_gradient'],atol=3e-7,rtol=0)


def test_empty_and_zero_density(model,bulk):
    mesh=symmetrix.LSCCPUMesh(.7,8);args=list(bulk);args[1]=np.zeros_like(args[1])
    for empty in [False,True]:
        if empty:args[0]=np.empty((0,3));args[1]=np.empty((0,4))
        actual=mesh.evaluate(model,*args)
        assert actual['energy']==0
        for key in ['density_gradient','position_gradient','strain_derivative']:assert not np.any(actual[key])
        assert actual['position_gradient'].shape==(0 if empty else len(bulk[0]),3)


def calculator(spacing=None,field=None):
    from symmetrix.lsc_calculator import LSCCalculator
    from .reference import REPO
    calc=LSCCalculator(REPO/'.lsc-reference-logs/water-stage2-energy.json',external_field=field)
    if spacing is not None:calc.model.set_mesh_electrostatics(spacing,8)
    return calc


@pytest.mark.parametrize('spacing',[.7,.35])
@pytest.mark.parametrize('field',[(0,0,0),(.01,-.02,.03)])
def test_full_model_against_direct(model,spacing,field,record_property):
    from .reference import load_atoms
    atoms=load_atoms('bulk');direct=calculator(field=field);mesh=calculator(spacing,field)
    direct.calculate(atoms);mesh.calculate(atoms)
    errors={key:float(np.max(abs(np.asarray(mesh.results[key])-direct.results[key]))) for key in ['energy','forces','stress']}
    for key,value in errors.items():record_property(key+'_error',value)
    print(f'CPU_MESH_MODEL spacing={spacing} field={field} errors={errors}')
    limits=[5e-7,3e-6,2e-8] if spacing==.7 else [2e-9,4e-8,3e-10]
    for key,bound in zip(errors,limits):assert errors[key]<bound,(key,errors)
    for key in ['charges','dipole']:np.testing.assert_array_equal(mesh.results[key],direct.results[key])
    assert mesh.model.has_electrostatics_hook
    mesh.model.set_electrostatics_hook(None)
    assert not mesh.model.has_energy_results and not mesh.model.has_force_results and not mesh.model.has_stress_results
    mesh.calculate(atoms)
    for key in ['energy','forces','stress','charges','dipole']:
        np.testing.assert_array_equal(mesh.results[key],direct.results[key])


def test_full_model_finite_differences(model,record_property):
    from .reference import load_atoms
    atoms=load_atoms('bulk');calc=calculator(.5,(.01,-.02,.03));calc.calculate(atoms)
    # Total energy is about -133214 eV: a 1e-5 A direction step loses
    # precision even with the unchanged direct solver (see FD step control).
    forces=calc.results['forces'].copy();stress=calc.results['stress'].copy();step=1e-4
    # Recalculate rebuilds image neighbours after every position/cell change.
    rng=np.random.default_rng(170);direction=rng.normal(size=forces.shape);direction/=np.linalg.norm(direction)
    energies=[]
    for sign in [-1,1]:
        changed=atoms.copy();changed.positions+=sign*step*direction;calc.calculate(changed)
        energies.append(calc.results['energy'])
    force_error=abs((energies[1]-energies[0])/(2*step)+np.sum(forces*direction))
    assert force_error<3e-6
    stress_errors=[];step=1e-5
    for component,(a,b) in enumerate([(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]):
        direction=np.zeros((3,3));direction[a,b]=direction[b,a]=1 if a==b else .5
        energies=[]
        for sign in [-1,1]:
            changed=atoms.copy();changed.set_cell(atoms.cell.array@(np.eye(3)+sign*step*direction),scale_atoms=True)
            calc.calculate(changed);energies.append(calc.results['energy'])
        error=abs((energies[1]-energies[0])/(2*step)/atoms.get_volume()-stress[component]);stress_errors.append(error)
        assert error<2e-8,(component,error)
    record_property('force_direction_fd_error',force_error);record_property('stress_max_fd_error',max(stress_errors))
    print(f'CPU_MESH_FULL_FD force_error={force_error:.12e} stress_error={max(stress_errors):.12e}')


def test_full_mesh_rejects_nonperiodic_and_restores_direct(model):
    from .reference import load_atoms
    calc=calculator(.7)
    with pytest.raises(ValueError,match='periodic'):calc.calculate(load_atoms('dimer'))
    assert not calc.model.has_energy_results and not calc.model.has_force_results
    calc.model.set_electrostatics_hook(None);calc.calculate(load_atoms('dimer'))
    direct=calculator();direct.calculate(load_atoms('dimer'))
    for key in ['energy','forces','stress','charges','dipole']:
        np.testing.assert_array_equal(calc.results[key],direct.results[key])
