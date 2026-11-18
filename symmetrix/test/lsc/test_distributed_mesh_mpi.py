"""Distributed LSC grid halos/FFT, with the serial mesh as an independent oracle."""
import json,os
from pathlib import Path
import numpy as np
import pytest
import symmetrix
from ase.io import write
from .reference import REPO,load_atoms
from .test_staged_api import backend
from .test_forward_mpi import launch

pytestmark=pytest.mark.lsc_reference
MODEL=REPO/'.lsc-reference-logs/water-stage2-energy.json'

def setup(directory,case,backend,ranks,field,empty=False,tilted=False,no_self=False,spacing=.7):
    directory.mkdir(parents=True,exist_ok=True);atoms=load_atoms(case).copy();atoms.pbc=True
    if case=='dimer':atoms.positions+=np.array([7.5 if empty else 15.,15.,15.])-atoms.positions.mean(axis=0)
    else:atoms.wrap()
    if tilted:
        cell=atoms.cell.array.copy();cell[1,0]+=.09*cell[1,1];cell[2,0]+=.03*cell[2,2];cell[2,1]+=.06*cell[2,2];atoms.set_cell(cell,scale_atoms=True)
    data=directory/'atoms.data';write(data,atoms,format='lammps-data',atom_style='atomic',masses=True,specorder=['H','O'])
    model=MODEL
    if no_self:
        payload=json.loads(MODEL.read_text());payload['electrostatics']['include_self_interaction']=False
        model=directory/'model.json';model.write_text(json.dumps(payload))
    style='symmetrix/mace/lsc'+('' if backend=='cpu' else '/cuda');precision=' float32-mixed' if backend=='float32-mixed' else ''
    path=directory/'setup.in';path.write_text(f'''units metal
atom_style atomic
boundary p p p
processors {ranks} 1 1
atom_modify sort 0 0.0
read_data {data}
newton on
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
pair_style {style}{precision} mesh {spacing} order 8 field {' '.join(map(str,field))}
pair_coeff * * {model} H O
''')
    return path,atoms,model

def evaluate(directory,case,backend,ranks,field,stage="mesh",**options):
    setup_file,atoms,model=setup(directory,case,backend,ranks,field,**options)
    build='cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    driver=Path(os.environ.get('LSC_FORWARD_MPI_DRIVER',str(REPO/f'.lsc-reference-logs/lsc-forward-mpi-{build}')));assert driver.is_file()
    prefix=directory/'result';out=launch(ranks,driver,[setup_file,prefix,backend,stage,model]);assert out.returncode==0,out.stdout+out.stderr
    data=[json.loads(Path(f'{prefix}.rank{r}.json').read_text()) for r in range(ranks)]
    return data,atoms,model

def arrays(data,atoms):
    n=len(atoms);rho=np.empty((n,4));phi=np.empty((n,4));pg=np.empty((n,3));forces=np.empty((n,3));ids=[]
    for item in data:
        tags=np.array(item['atom_ids'][:item['num_owned']],dtype=int)-1;ids.extend(tags)
        rho[tags]=np.array(item['density_coefficients']).reshape(-1,4)
        phi[tags]=np.array(item['potentials']).reshape(-1,4)[:len(tags)]
        pg[tags]=np.array(item['mesh_position_gradient']).reshape(-1,3);forces[tags]=np.array(item['forces']).reshape(-1,3)
        assert item['unused_bounds_poisoned']
        assert item['mesh_plan_generation']==item['mesh_reused_plan_generation']==1
        assert abs(item['energy']-item['mesh_repeat_energy'])<1e-9
    assert sorted(ids)==list(range(n))
    for item in data:
        for row,tag in enumerate(item['atom_ids'][item['num_owned']:],item['num_owned']):np.testing.assert_array_equal(np.array(item['potentials']).reshape(-1,4)[row],phi[tag-1])
    return rho,phi,pg,forces

def check_oracle(data,atoms,model,backend,directory,spacing=.7):
    rho,phi,pg,force=arrays(data,atoms)
    if backend!='cpu':
        # Match the existing CUDA export's FP32 constants; mesh arithmetic stays double.
        payload=json.loads(Path(model).read_text())
        def rounded(value):
            if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
            if isinstance(value,list):return [rounded(v) for v in value]
            return float(np.float32(value)) if isinstance(value,float) else value
        payload['electrostatics']=rounded(payload['electrostatics'])
        model=directory/'oracle-packed.json';model.write_text(json.dumps(payload))
    control=symmetrix.MACELSC(str(model));mesh=symmetrix.LSCCPUMesh(spacing,8)
    ref=mesh.evaluate(control,atoms.positions,rho,atoms.cell.array,atoms.pbc,'auto')
    energy=sum(item['mesh_energy_rank'] for item in data);strain=np.sum([item['mesh_strain_rank'] for item in data],axis=0).reshape(3,3)
    # CUDA's packed physical constants retain the established FP32 parameter scale.
    np.testing.assert_allclose(phi,ref['density_gradient'],atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(pg,ref['position_gradient'],atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(strain,ref['strain_derivative'],atol=2e-9,rtol=2e-10)
    assert abs(energy-ref['energy'])<2e-9
    assert abs(energy-.5*np.sum(rho*phi))<2e-9
    total=int(np.prod(data[0]['mesh_shape']));assert sum(item['mesh_local_modes'] for item in data)==total
    if len(data)>1:assert all(0<item['mesh_local_modes']<total for item in data)
    for item in data:
        if item['mesh_device_fft_calls']:assert item['mesh_device_fft_calls']==6
    return dict(energy_error=abs(energy-ref['energy']),potential_error=float(abs(phi-ref['density_gradient']).max()),position_gradient_error=float(abs(pg-ref['position_gradient']).max()),strain_error=float(abs(strain-ref['strain_derivative']).max())),force

@pytest.mark.parametrize('kind',['empty','tilted','no-self'])
def test_mesh_special_cases(backend,kind,tmp_path,record_property):
    case='dimer' if kind=='empty' else 'bulk';options={'empty':kind=='empty','tilted':kind=='tilted','no_self':kind=='no-self'}
    data,atoms,model=evaluate(tmp_path/kind,case,backend,2,(.01,-.02,.03),**options)
    errors,forces=check_oracle(data,atoms,model,backend,tmp_path)
    if kind=='empty':assert any(x['num_owned']==0 and not x['forces'] for x in data)
    for k,v in dict(backend=backend,case=kind,**errors).items():record_property(k,v)

@pytest.mark.parametrize('case',['dimer','bulk'])
@pytest.mark.parametrize('field',[(0.,0.,0.),(.01,-.02,.03)],ids=['zero','field'])
def test_distributed_mesh_one_two_ranks(backend,case,field,tmp_path,record_property):
    one,atoms,model=evaluate(tmp_path/'one',case,backend,1,field)
    two,_,_=evaluate(tmp_path/'two',case,backend,2,field)
    errors,force=check_oracle(two,atoms,model,backend,tmp_path)
    native,_,_=evaluate(tmp_path/'native',case,backend,1,field,stage='serial')
    native_force={tag:force for row in native for tag,force in zip(row['atom_ids'][:row['num_owned']],np.array(row['forces']).reshape(-1,3))}
    expected=np.array([native_force[tag] for tag in range(1,len(atoms)+1)])
    np.testing.assert_allclose(arrays(two,atoms)[3],expected,atol=2e-8 if backend=='cpu' else 8e-5,rtol=0 if backend=='cpu' else 1e-4)
    np.testing.assert_allclose(two[0]['stress'],native[0]['stress'],atol=2e-10 if backend=='cpu' else 5e-7,rtol=0 if backend=='cpu' else 1e-4)
    assert abs(two[0]['energy']-native[0]['energy'])<(2e-8 if backend=='cpu' else 5e-5)
    a=arrays(one,atoms);b=arrays(two,atoms)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-9 if backend=='cpu' else 8e-5,rtol=0 if backend=='cpu' else 1e-4)
    np.testing.assert_allclose(two[0]['stress'],one[0]['stress'],atol=2e-10 if backend=='cpu' else 5e-7,rtol=0 if backend=='cpu' else 1e-4)
    assert abs(two[0]['energy']-one[0]['energy'])<(2e-8 if backend=='cpu' else 5e-5)
    for x in two:
        np.testing.assert_array_equal(x['stress'],two[0]['stress']);assert x['energy']==two[0]['energy']
        assert bool(x['mesh_device_fft_calls'])==(backend!='cpu')
    dipole1=np.sum([x['dipole'] for x in one],axis=0);dipole2=np.sum([x['dipole'] for x in two],axis=0)
    np.testing.assert_allclose(dipole1,dipole2,atol=2e-9 if backend=='cpu' else 3e-5,rtol=0)
    for k,v in dict(backend=backend,case=case,field=field,force_rank_error=float(abs(a[3]-b[3]).max()),energy_rank_error=abs(two[0]['energy']-one[0]['energy']),**errors).items():record_property(k,v)
    print('DISTRIBUTED_MESH',backend,case,field,errors,'F_ranks',float(abs(a[3]-b[3]).max()))
