"""Real LAMMPS callbacks: return H2/H1 derivatives and ghost forces."""
import json,os
from pathlib import Path
import numpy as np
import pytest
import symmetrix
from .reference import REPO
from .test_staged_api import backend
from .test_forward_mpi import setup_input,launch

pytestmark=pytest.mark.lsc_reference

def evaluate(directory,case,backend,ranks,field,empty=False):
    build='cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    driver=Path(os.environ.get('LSC_FORWARD_MPI_DRIVER',str(REPO/f'.lsc-reference-logs/lsc-forward-mpi-{build}')))
    assert driver.is_file()
    setup,n=setup_input(directory,case,backend,ranks,empty)
    text=setup.read_text();text=text.replace('\npair_coeff',' field '+' '.join(map(str,field))+'\npair_coeff');setup.write_text(text)
    prefix=directory/'result';out=launch(ranks,driver,[setup,prefix,backend,'reverse',REPO/'.lsc-reference-logs/water-stage2-energy.json'])
    assert out.returncode==0,out.stdout+out.stderr
    return [json.loads(Path(f'{prefix}.rank{r}.json').read_text()) for r in range(ranks)],n

def rows(outputs):
    return {tag:force for out in outputs for tag,force in zip(out['atom_ids'][:out['num_owned']],np.asarray(out['forces']).reshape(-1,3))}

@pytest.mark.parametrize('case',['dimer','bulk'])
@pytest.mark.parametrize('field',[(0.,0.,0.),(.01,-.02,.03)],ids=['zero','field'])
def test_reverse_one_two_ranks(backend,case,field,tmp_path,record_property):
    one,n=evaluate(tmp_path/'one',case,backend,1,field);two,_=evaluate(tmp_path/'two',case,backend,2,field)
    serial=rows(one);parallel=rows(two);assert len(serial)==len(parallel)==n
    a=np.array([serial[k] for k in sorted(serial)]);b=np.array([parallel[k] for k in sorted(serial)])
    atol=2e-8 if backend=='cpu' else 8e-5;rtol=0 if backend=='cpu' else 1e-4
    np.testing.assert_allclose(b,a,atol=atol,rtol=rtol)
    np.testing.assert_allclose(b.sum(axis=0),0,atol=2e-8 if backend=='cpu' else 3e-5,rtol=0)
    for out in two:
        assert out['energy']==two[0]['energy']
        np.testing.assert_array_equal(out['stress'],two[0]['stress'])
        np.testing.assert_allclose(out['stress'],one[0]['stress'],atol=2e-10 if backend=='cpu' else 5e-7,rtol=rtol)
        np.testing.assert_allclose(out['virial'],one[0]['virial'],atol=2e-8 if backend=='cpu' else 1e-3,rtol=rtol)
        assert abs(out['energy']-one[0]['energy'])<(2e-8 if backend=='cpu' else 5e-5)
    for k,v in dict(backend=backend,case=case,field=field,force_max_abs_error=float(abs(b-a).max()),stress_max_abs_error=float(abs(np.array(two[0]['stress'])-one[0]['stress']).max()),energy_abs_error=abs(two[0]['energy']-one[0]['energy'])).items():record_property(k,v)
    print('REVERSE_MPI',backend,case,field,'F',float(abs(b-a).max()))

def test_reverse_empty_rank(backend,tmp_path):
    one,n=evaluate(tmp_path/'one','dimer',backend,1,(.01,-.02,.03),empty=True)
    two,_=evaluate(tmp_path/'two','dimer',backend,2,(.01,-.02,.03),empty=True)
    assert any(x['num_owned']==0 and not x['forces'] for x in two)
    a=rows(one);b=rows(two);assert len(a)==len(b)==n
    for tag in a:np.testing.assert_allclose(b[tag],a[tag],atol=2e-8 if backend=='cpu' else 8e-5,rtol=0 if backend=='cpu' else 1e-4)
    np.testing.assert_allclose(two[0]['stress'],one[0]['stress'],atol=2e-10 if backend=='cpu' else 5e-7,rtol=0)
