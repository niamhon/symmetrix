"""Basic ASE integration against the frozen dimer and bulk predictions."""
import numpy as np
import pytest
from symmetrix.lsc_calculator import LSCCalculator
from .reference import DATA, REPO, load_atoms
from .test_cuda_loader import require_cuda

pytestmark=pytest.mark.lsc_reference

@pytest.mark.parametrize("backend",["cpu",pytest.param("float32",marks=pytest.mark.lsc_cuda),pytest.param("float32-mixed",marks=pytest.mark.lsc_cuda)])
@pytest.mark.parametrize("case",["dimer","bulk"])
def test_ase_calculator(backend,case):
    if backend!="cpu":require_cuda()
    model=REPO/".lsc-reference-logs/water-stage2-energy.json"
    atoms=load_atoms(case)
    atoms.calc=LSCCalculator(model,device="cpu" if backend=="cpu" else "cuda",precision="float64" if backend=="cpu" else backend)
    forces=atoms.get_forces()
    with np.load(DATA/f"{case}.reference.npz",allow_pickle=False) as expected:
        assert abs(atoms.get_potential_energy()-float(expected["energy"]))<5e-5
        np.testing.assert_allclose(forces,expected["forces"],atol=8e-5,rtol=1e-4)
        np.testing.assert_allclose(atoms.get_charges(),expected["partial_charges"],atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(atoms.get_dipole_moment(),expected["dipole"],atol=3e-5,rtol=0)
        if case=="bulk":np.testing.assert_allclose(atoms.get_stress(),expected["stress"],atol=5e-7,rtol=1e-4)
