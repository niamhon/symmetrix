"""Exercise the real LAMMPS forward callbacks with 1/2 MPI ranks on one node."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import pytest
import symmetrix
from ase.io import write

from .reference import REPO, load_atoms
from .test_staged_api import backend, saved

pytestmark = pytest.mark.lsc_reference
MPI = '/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun'


def setup_input(directory, case, backend, ranks, empty=False, run=False):
    directory.mkdir(parents=True, exist_ok=True)
    atoms = load_atoms(case).copy()
    if case == 'dimer':
        atoms.positions += np.array([7.5 if empty else 15., 15., 15.])-atoms.positions.mean(axis=0)
    else:
        atoms.wrap()
    data = directory/'atoms.data'
    write(data, atoms, format='lammps-data', atom_style='atomic', masses=True, specorder=['H', 'O'])
    style = 'symmetrix/mace/lsc' + ('' if backend=='cpu' else '/cuda')
    option = ' float32-mixed' if backend=='float32-mixed' else ''
    periodic = 'p p p' if case=='bulk' else 'f f f'
    path = directory/'setup.in'
    path.write_text(f'''units metal
atom_style atomic
boundary {periodic}
processors {ranks} 1 1
atom_modify sort 0 0.0
read_data {data}
newton on
neighbor 0.4 bin
neigh_modify every 1 delay 0 check yes
pair_style {style}{option}
pair_coeff * * {REPO}/.lsc-reference-logs/water-stage2-energy.json H O
''' + ('run 0\n' if run else ''))
    return path, len(atoms)


def launch(ranks, binary, args):
    launcher = shlex.split(os.environ.get('LSC_FORWARD_MPI_LAUNCHER', MPI+' --bind-to none --oversubscribe'))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([*launcher, '-np', str(ranks), str(binary), *map(str,args)],
                            capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=120)
    return result


def evaluate(directory, case, backend, ranks, empty=False):
    build = 'cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    driver = Path(os.environ.get('LSC_FORWARD_MPI_DRIVER', str(REPO/f'.lsc-reference-logs/lsc-forward-mpi-{build}')))
    assert driver.is_file(), 'Build tools/lsc/build_forward_mpi_driver.sh in the matching environment'
    setup, n = setup_input(directory, case, backend, ranks, empty)
    prefix = directory/'result'
    result = launch(ranks, driver, [setup, prefix, backend])
    assert result.returncode == 0, result.stdout+result.stderr
    outputs = [json.loads(Path(str(prefix)+f'.rank{i}.json').read_text()) for i in range(ranks)]
    return outputs, n


def rows(outputs):
    owned = {}
    for item in outputs:
        ids = item['atom_ids']; n = item['num_owned']
        width = 512
        for layer in [1,2]:item[f'H{layer}']=np.asarray(item[f'H{layer}']).reshape(len(ids),width)
        for slot, tag in enumerate(ids[:n]):
            assert tag not in owned
            owned[tag] = (item['H1'][slot], item['H2'][slot], item['local_node_energies'][slot])
    return owned


def check_transport(outputs, owned):
    ghost_count=0;remote_count=0
    for item in outputs:
        ids=item['atom_ids'];n=item['num_owned'];local=set(ids[:n])
        for slot,tag in enumerate(ids[n:],n):
            ghost_count += 1;remote_count += tag not in local
            for layer in [1,2]:
                # Transport is lossless, even when the forward MPI route relays
                # periodic ghosts across more than one LAMMPS swap.
                np.testing.assert_array_equal(item[f'H{layer}'][slot], owned[tag][layer-1])
    return ghost_count,remote_count


@pytest.mark.parametrize('case',['dimer','bulk'])
def test_lammps_forward_one_and_two_ranks(backend,saved,case,tmp_path,record_property):
    one,n=evaluate(tmp_path/'one',case,backend,1)
    two,_=evaluate(tmp_path/'two',case,backend,2)
    serial=rows(one);parallel=rows(two)
    assert len(serial)==len(parallel)==n
    ghosts,remote=check_transport(two,parallel)
    assert remote>0 and all(item['num_owned']>0 for item in two)
    check_transport(one,serial)
    atol,rtol=(2e-8,2e-7) if backend=='cpu' else (3e-5,3e-4)
    max_error=0.
    precision='float64' if backend=='cpu' else backend
    for layer in [1,2]:
        frozen=saved[f'{precision}/{case}/direct/zero/0/H{layer}'].reshape(n,-1)
        for tag in serial:
            np.testing.assert_allclose(serial[tag][layer-1],frozen[tag-1],atol=atol,rtol=rtol)
            np.testing.assert_allclose(parallel[tag][layer-1],serial[tag][layer-1],atol=atol,rtol=rtol)
            max_error=max(max_error,float(abs(parallel[tag][layer-1]-serial[tag][layer-1]).max()))
    energy_error=abs(sum(item['local_energy'] for item in two)-sum(item['local_energy'] for item in one))
    assert energy_error<(2e-9 if backend=='cpu' else 5e-5)
    for key,value in dict(backend=backend,case=case,ghost_rows=ghosts,remote_rows=remote,feature_max_abs_error=max_error,local_energy_abs_error=energy_error).items():record_property(key,value)
    print('FORWARD_MPI',backend,case,'ghosts',ghosts,'remote',remote,'features',max_error,'local_energy',energy_error)


def test_lammps_forward_empty_rank(backend,tmp_path,record_property):
    output,n=evaluate(tmp_path/'empty','dimer',backend,2,empty=True)
    owned=rows(output);assert len(owned)==n
    assert any(item['num_owned']==0 and item['local_energy']==0 for item in output)
    ghosts,remote=check_transport(output,owned)
    record_property('ghost_rows',ghosts);record_property('remote_rows',remote)


def test_multi_rank_direct_requires_mesh(tmp_path):
    setup,_=setup_input(tmp_path/'guard','bulk','cpu',2,run=True)
    build='cuda' if symmetrix.lsc_cuda_compiled() else 'cpu'
    binary=Path(os.environ.get('LSC_LAMMPS_BINARY',f'/mnt/home/noneill/Programs/lammps-lsc-dev/build-{build}/lmp'))
    result=launch(2,binary,['-in',setup,'-log','none'])
    assert result.returncode!=0
    assert 'Multi-rank LSC MD requires mesh electrostatics' in result.stdout+result.stderr
