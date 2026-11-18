"""Real LAMMPS reverse charge sums and forward full Coulomb potentials."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import symmetrix
from .reference import REPO
from .test_staged_api import backend
from .test_forward_mpi import setup_input,launch

pytestmark=pytest.mark.lsc_reference

def evaluate(directory,case,backend,ranks,empty=False):
    build='cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    driver=Path(os.environ.get('LSC_FORWARD_MPI_DRIVER',str(REPO/f'.lsc-reference-logs/lsc-forward-mpi-{build}')))
    assert driver.is_file(),'Build tools/lsc/build_forward_mpi_driver.sh first'
    setup,n=setup_input(directory,case,backend,ranks,empty)
    prefix=directory/'result'
    result=launch(ranks,driver,[setup,prefix,backend,'charges',REPO/'.lsc-reference-logs/water-stage2-energy.json'])
    assert result.returncode==0,result.stdout+result.stderr
    outputs=[json.loads(Path(str(prefix)+f'.rank{rank}.json').read_text()) for rank in range(ranks)]
    return outputs,n

def rows(outputs):
    owned={}
    for item in outputs:
        n=item['num_owned'];m=len(item['atom_ids'])
        item['rho']=np.asarray(item['density_coefficients']).reshape(n,4)
        item['phi']=np.asarray(item['potentials']).reshape(m,4)
        expected=np.asarray(item['reference_owned_potentials']).reshape(n,4)
        # Full positive dE/drho, not a half-potential or just scalar voltages.
        np.testing.assert_array_equal(item['phi'][:n],expected)
        for i,tag in enumerate(item['atom_ids'][:n]):
            assert tag not in owned;owned[tag]=(item['rho'][i],item['phi'][i])
    return owned

@pytest.mark.parametrize('case',['dimer','bulk'])
def test_charge_and_potential_one_two_ranks(backend,case,tmp_path,record_property):
    one,n=evaluate(tmp_path/'one',case,backend,1)
    two,_=evaluate(tmp_path/'two',case,backend,2)
    serial=rows(one);parallel=rows(two);assert len(serial)==len(parallel)==n
    density_error=potential_error=0.;remote=0
    for tag in serial:
        np.testing.assert_allclose(parallel[tag][0],serial[tag][0],atol=2e-9 if backend=='cpu' else 2e-6,rtol=0 if backend=='cpu' else 1e-5)
        np.testing.assert_allclose(parallel[tag][1],serial[tag][1],atol=2e-9 if backend=='cpu' else 3e-5,rtol=0 if backend=='cpu' else 1e-4)
        density_error=max(density_error,float(abs(parallel[tag][0]-serial[tag][0]).max()))
        potential_error=max(potential_error,float(abs(parallel[tag][1]-serial[tag][1]).max()))
    for item in two:
        local=set(item['atom_ids'][:item['num_owned']])
        for i,tag in enumerate(item['atom_ids'][item['num_owned']:],item['num_owned']):
            np.testing.assert_array_equal(item['phi'][i],parallel[tag][1]);remote+=tag not in local
    assert remote>0
    net=sum(item['rho'][:,0].sum() for item in two)
    assert abs(net)<1e-11
    dipole1=np.sum([item['dipole'] for item in one],axis=0);dipole2=np.sum([item['dipole'] for item in two],axis=0)
    np.testing.assert_allclose(dipole2,dipole1,atol=2e-9 if backend=='cpu' else 3e-5,rtol=0)
    energy=two[0]['reference_electrostatic_energy']
    contraction=.5*sum(np.sum(item['rho']*item['phi'][:item['num_owned']]) for item in two)
    assert abs(contraction-energy)<1e-10
    for key,value in dict(backend=backend,case=case,remote_rows=remote,density_max_abs_error=density_error,potential_max_abs_error=potential_error,net_charge=net,dipole_max_abs_error=float(abs(dipole2-dipole1).max()),energy_identity_error=abs(contraction-energy)).items():record_property(key,value)
    print('CHARGE_MPI',backend,case,'remote',remote,'rho',density_error,'phi',potential_error,'net_charge',net)

def test_charge_and_potential_empty_rank(backend,tmp_path):
    output,n=evaluate(tmp_path/'empty','dimer',backend,2,empty=True)
    owned=rows(output);assert len(owned)==n and any(item['num_owned']==0 for item in output)
    for item in output:
        for i,tag in enumerate(item['atom_ids'][item['num_owned']:],item['num_owned']):np.testing.assert_array_equal(item['phi'][i],owned[tag][1])
    assert abs(sum(item['rho'][:,0].sum() for item in output))<1e-11
