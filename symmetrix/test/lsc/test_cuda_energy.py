"""Basic finite-cluster/bulk GPU electrostatic and total-energy comparisons."""
import json
import numpy as np
import pytest
import symmetrix

from .reference import DATA, load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import require_cuda
from .test_native_local import graph_for_case

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_cuda]


@pytest.fixture(scope="module",params=["float32","float32-mixed"])
def gpu(request,export_file):
    require_cuda()
    return symmetrix.MACELSCCuda(str(export_file),request.param)


@pytest.fixture(scope="module")
def native(export_file):
    return symmetrix.MACELSC(str(export_file))


@pytest.mark.parametrize("case",["dimer","bulk"])
def test_total_electrostatic_and_field_energy(gpu,native,lsc_model,case,record_property):
    numbers,edges,vectors=graph_for_case(lsc_model,case)
    atoms=load_atoms(case)
    types=np.array([gpu.atomic_numbers.index(int(z)) for z in numbers])
    mode="realspace" if case=="dimer" else "mixed_periodic"
    args=(types,edges,vectors,atoms.positions.copy(),atoms.cell.array.copy(),atoms.pbc.copy(),mode)
    expected=native.compute_energy(*args)
    actual=gpu.compute_energy(*args)
    assert gpu.can_evaluate_energy and gpu.has_energy_results and gpu.can_evaluate
    assert abs(actual-expected)<5e-5
    assert abs(gpu.electrostatic_energy-native.electrostatic_energy)<3e-5
    with np.load(DATA/f"{case}.reference.npz",allow_pickle=False) as frozen:
        assert abs(actual-float(frozen["energy"]))<5e-5
    electrostatic=gpu.electrostatic_energy
    assert abs(actual-(gpu.local_energy+electrostatic+gpu.field_energy))<1e-10
    field=np.array([0.01,-0.02,0.03])
    changed=gpu.evaluate_energy(atoms.cell.array,atoms.pbc,"auto",field)
    assert abs(changed-actual-np.dot(field,gpu.dipole))<1e-9
    assert abs(gpu.field_energy-np.dot(field,gpu.dipole))<1e-12
    stats={"total_energy_abs_eV":abs(actual-expected),"electrostatic_energy_abs_eV":abs(electrostatic-native.electrostatic_energy)}
    for key,value in dict(device=gpu.device_name,precision=gpu.precision,case=case,**stats).items():record_property(key,value)
    print(f"CUDA_ENERGY {gpu.precision} {case} "+" ".join(f"{k}={v:.6e}" for k,v in stats.items()))


@pytest.fixture(scope="module")
def no_self(export_file,tmp_path_factory):
    document=json.loads(export_file.read_text())
    document["electrostatics"]["include_self_interaction"]=False
    path=tmp_path_factory.mktemp("cuda-energy-no-self")/"model.json"
    path.write_text(json.dumps(document,allow_nan=False))
    return path


@pytest.mark.parametrize("precision",["float32","float32-mixed"])
def test_without_self_interaction(no_self,lsc_model,precision):
    require_cuda()
    gpu=symmetrix.MACELSCCuda(str(no_self),precision)
    native=symmetrix.MACELSC(str(no_self))
    for case in ["dimer","bulk"]:
        numbers,edges,vectors=graph_for_case(lsc_model,case);atoms=load_atoms(case)
        types=np.array([gpu.atomic_numbers.index(int(z)) for z in numbers])
        args=(types,edges,vectors,atoms.positions.copy(),atoms.cell.array.copy(),atoms.pbc.copy(),"auto")
        assert abs(gpu.compute_energy(*args)-native.compute_energy(*args))<5e-5
