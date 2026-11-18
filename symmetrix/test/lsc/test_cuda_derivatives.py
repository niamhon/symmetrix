"""Basic full-model GPU force/stress parity and a directional force check."""
import numpy as np
import pytest
import symmetrix
from .reference import DATA,load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import require_cuda
from .test_native_local import graph_for_case

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_cuda]

@pytest.fixture(scope="module",params=["float32","float32-mixed"])
def gpu(request,export_file):
    require_cuda();return symmetrix.MACELSCCuda(str(export_file),request.param)

@pytest.fixture(scope="module")
def native(export_file):return symmetrix.MACELSC(str(export_file))

def inputs(model,case):
    numbers,edges,vectors=graph_for_case(model,case);atoms=load_atoms(case)
    types=np.array([model.atomic_numbers.tolist().index(int(z)) for z in numbers])
    return [types,edges,vectors,atoms.positions.copy(),atoms.cell.array.copy(),atoms.pbc.copy(),"auto"]

@pytest.mark.parametrize("case",["dimer","bulk"])
def test_full_forces_and_stress(gpu,native,lsc_model,case,record_property):
    args=inputs(lsc_model,case)
    expected_energy=native.compute_energy_forces(*args,compute_stress=True)
    actual=gpu.compute_energy_forces(*args,compute_stress=True)
    results=gpu.download_forces()
    reference=np.asarray(native.forces).reshape(-1,3)
    np.testing.assert_allclose(results["forces"],reference,atol=8e-5,rtol=1e-4)
    np.testing.assert_allclose(gpu.stress,native.stress,atol=5e-7,rtol=1e-4)
    np.testing.assert_allclose(gpu.virial,native.virial,atol=1e-3,rtol=1e-4)
    assert abs(actual-expected_energy)<5e-5
    with np.load(DATA/f"{case}.reference.npz",allow_pickle=False) as frozen:
        np.testing.assert_allclose(results["forces"],frozen["forces"],atol=8e-5,rtol=1e-4)
        if case=="bulk":np.testing.assert_allclose(gpu.stress,frozen["stress"],atol=5e-7,rtol=1e-4)
    assembled=results["position_forces"].copy()
    np.add.at(assembled,args[1][1],results["edge_forces"])
    np.add.at(assembled,args[1][0],-results["edge_forces"])
    np.testing.assert_allclose(results["forces"],assembled,atol=2e-12,rtol=0)
    assert gpu.can_evaluate and gpu.can_evaluate_forces and gpu.has_force_results and gpu.has_stress_results
    stats={"force_max_abs_eV_A":float(abs(results["forces"]-reference).max()),"stress_max_abs_eV_A3":float(abs(np.array(gpu.stress)-native.stress).max()),"energy_abs_eV":abs(actual-expected_energy)}
    for key,value in dict(device=gpu.device_name,precision=gpu.precision,case=case,**stats).items():record_property(key,value)
    print(f"CUDA_DERIVATIVES {gpu.precision} {case} "+" ".join(f"{k}={v:.6e}" for k,v in stats.items()))

@pytest.mark.parametrize("case",["dimer","bulk"])
def test_field_derivatives_and_state(gpu,native,lsc_model,case):
    args=inputs(lsc_model,case);field=np.array([0.01,-0.02,0.03])
    native.compute_energy_forces(*args,field,compute_stress=True)
    gpu.compute_energy_forces(*args,field,compute_stress=True)
    np.testing.assert_allclose(gpu.download_forces(False)["forces"],np.asarray(native.forces).reshape(-1,3),atol=8e-5,rtol=1e-4)
    np.testing.assert_allclose(gpu.stress,native.stress,atol=5e-7,rtol=1e-4)
    gpu.compute_local(*args[:3])
    assert not gpu.has_force_results and not gpu.has_energy_results
    with pytest.raises(RuntimeError,match="evaluate forces first"):gpu.download_forces()


def test_dimer_force_direction(gpu,lsc_model):
    args=inputs(lsc_model,"dimer")
    gpu.compute_energy_forces(*args)
    forces=gpu.download_forces(False)["forces"]
    direction=np.random.default_rng(19).normal(size=forces.shape);direction/=np.linalg.norm(direction)
    energies=[];step=1e-3
    for sign in [-1,1]:
        shifted=list(args)
        shifted[3]=args[3]+sign*step*direction
        shifted[2]=args[2]+sign*step*(direction[args[1][1]]-direction[args[1][0]])
        energies.append(gpu.compute_energy(*shifted))
    derivative=(energies[1]-energies[0])/(2*step)
    assert abs(derivative+np.sum(forces*direction))<3e-3
