"""One-process CUDA mesh LAMMPS prediction and a basic ten-step MD smoke test."""
import re
import numpy as np
import pytest
from .test_lammps_lsc import setup,run
from symmetrix.lsc_calculator import LSCCalculator
from .reference import REPO
from .test_cuda_loader import require_cuda
from .test_cpu_mesh_energy import model

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_lammps,pytest.mark.lsc_cuda]

@pytest.fixture(params=["float32","float32-mixed"])
def precision(request):
    require_cuda();return request.param

def calculator(precision,spacing=None):
    calc=LSCCalculator(REPO/".lsc-reference-logs/water-stage2-energy.json",device="cuda",precision=precision)
    if spacing is not None:calc.model.set_mesh_electrostatics(spacing,8)
    return calc


def mesh_setup(tmp_path,precision,md=False):
    binary,atoms,commands=setup(tmp_path,'bulk',precision,md=md)
    suffix=' float32-mixed' if precision=='float32-mixed' else ''
    commands=[('pair_style symmetrix/mace/lsc/cuda mesh 0.7 order 8'+suffix if c.startswith('pair_style ') else c) for c in commands]
    return binary,atoms,commands


def test_mesh_pair_point_prediction(model,tmp_path,precision,record_property):
    binary,atoms,commands=mesh_setup(tmp_path,precision);dump=tmp_path/'forces.dump'
    commands += ['compute eat all pe/atom','compute esum all reduce sum c_eat',
                 'compute mu all pair symmetrix/mace/lsc/cuda',
                 'thermo_style custom step pe c_esum pxx pyy pzz pxy pxz pyz c_mu[1] c_mu[2] c_mu[3]',
                 'thermo_modify format float %.17g','run 0',
                 'print "LSC_MESH_RESULT $(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g) $(c_esum:%.17g) $(c_mu[1]:%.17g) $(c_mu[2]:%.17g) $(c_mu[3]:%.17g)"',
                 f'write_dump all custom {dump} id type fx fy fz q c_eat modify sort id format float %.17g']
    output=run(binary,commands,tmp_path);match=re.search(r'^LSC_MESH_RESULT (.*)$',output,re.MULTILINE)
    assert match,output
    values=np.fromstring(match.group(1),sep=' ');rows=np.loadtxt(dump.read_text().splitlines()[9:]);rows=rows[np.argsort(rows[:,0])]
    mesh=calculator(precision,.7);mesh.calculate(atoms);expected=mesh.results
    direct=calculator(precision);direct.calculate(atoms)
    # LAMMPS metal units use this historical nktv2p constant (src/update.cpp),
    # slightly different from ASE's conversion. Undo the actual LAMMPS factor.
    stress=-values[1:7][[0,1,2,5,4,3]]/1.6021765e6
    assert abs(values[0]-expected['energy'])<5e-5
    np.testing.assert_allclose(rows[:,2:5],expected['forces'],atol=8e-5,rtol=0)
    np.testing.assert_allclose(stress,expected['stress'],atol=5e-7,rtol=0)
    np.testing.assert_allclose(rows[:,5],expected['charges'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(values[8:11],expected['dipole'],atol=3e-5,rtol=0)
    assert abs(values[0]-values[7])<2e-8
    assert abs(rows[:,6].sum()-values[0])<2e-8
    errors=dict(energy=abs(values[0]-direct.results['energy']),forces=float(np.max(abs(rows[:,2:5]-direct.results['forces']))),stress=float(np.max(abs(stress-direct.results['stress']))))
    assert errors['energy']<5e-5 and errors['forces']<8e-5 and errors['stress']<5e-7
    for key,value in errors.items():record_property(key+'_direct_error',value)
    record_property('precision',precision)
    print(f'LAMMPS_CUDA_MESH precision={precision} errors_vs_direct={errors}')


def test_mesh_short_nve(model,tmp_path,precision,record_property):
    binary,atoms,commands=mesh_setup(tmp_path,precision,md=True)
    commands += ['velocity all create 10 19371 mom yes rot yes','fix integrate all nve','timestep 0.0001',
                 'thermo 1','thermo_style custom step pe ke etotal temp','thermo_modify format float %.17g','run 10']
    output=run(binary,commands,tmp_path);samples=[]
    for line in output.splitlines():
        fields=line.split()
        if len(fields)==5 and fields[0].isdigit():
            try:samples.append([float(v) for v in fields])
            except ValueError:pass
    values=np.array(samples);assert len(values)==11,output
    drift=float(np.max(abs(values[:,3]-values[0,3]))/len(atoms));assert drift<1e-4
    record_property('energy_drift_eV_per_atom',drift);record_property('precision',precision)
    print(f'LAMMPS_CUDA_MESH_NVE precision={precision} steps=10 timestep_fs=0.1 drift_eV_per_atom={drift:.12e}')
