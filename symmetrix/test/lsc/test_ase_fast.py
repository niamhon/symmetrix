"""Polar-style ASE API, checked against independent frozen native predictions."""
import sys
import subprocess

import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from ase.md.verlet import VelocityVerlet
from ase import Atoms, units
from ase.neighborlist import neighbor_list

from symmetrix.experimental import MACELSC
from symmetrix.lsc_calculator import LSCCalculator
from .reference import DATA, REPO, load_atoms
from .test_cuda_loader import require_cuda

pytestmark = pytest.mark.lsc_reference
MODEL = REPO / ".lsc-reference-logs/water-stage2-energy.json"


@pytest.fixture(scope="module", params=["cpu",
    pytest.param("float32", marks=pytest.mark.lsc_cuda),
    pytest.param("float32-mixed", marks=pytest.mark.lsc_cuda)])
def backend(request):
    if request.param != "cpu":
        require_cuda()
    return request.param


def options(backend):
    return dict(device="cpu" if backend == "cpu" else "cuda",
                dtype="float64" if backend == "cpu" else backend)


@pytest.fixture(scope="module")
def calc(backend):
    return MACELSC(MODEL, **options(backend))


def assert_prediction(atoms, calc, expected):
    atoms.calc = calc
    assert abs(atoms.get_potential_energy() - float(expected["energy"])) / len(atoms) < 1e-6
    np.testing.assert_allclose(atoms.get_forces(), expected["forces"], atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(atoms.get_charges(), expected["partial_charges"], atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(atoms.get_dipole_moment(), expected["dipole"], atol=3e-5, rtol=0)
    if atoms.pbc.all():
        np.testing.assert_allclose(atoms.get_stress(), expected["stress"], atol=5e-7, rtol=1e-4)
    assert abs(atoms.get_potential_energies().sum() - atoms.get_potential_energy()) / len(atoms) < 1e-8


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_frozen_snapshots(calc, case):
    with np.load(DATA / f"{case}.reference.npz", allow_pickle=False) as expected:
        assert_prediction(load_atoms(case), calc, expected)


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_compiled_neighbors(calc, case):
    a = load_atoms(case)
    i, j, S = calc._neighbor_list(a.positions, a.cell.array, a.pbc, calc.evaluator.r_cut)
    ai, aj, aS = neighbor_list("ijS", a, calc.evaluator.r_cut, self_interaction=False)
    actual = sorted(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))
    expected = sorted(zip(ai.tolist(), aj.tolist(), map(tuple, aS.tolist())))
    assert actual == expected


def test_periodic_self_images(calc):
    a = Atoms("OH2", positions=[[1, 1, 1], [1.96, 1, 1], [.76, 1.93, 1]],
              cell=[4, 4, 4], pbc=True)
    i, j, S = calc._neighbor_list(a.positions, a.cell.array, a.pbc, calc.evaluator.r_cut)
    ai, aj, aS = neighbor_list("ijS", a, calc.evaluator.r_cut, self_interaction=False)
    assert sorted(zip(i.tolist(), j.tolist(), map(tuple, S.tolist()))) == sorted(
        zip(ai.tolist(), aj.tolist(), map(tuple, aS.tolist())))
    assert np.any((i == j) & np.any(S != 0, axis=1))


def test_cellless_cluster_and_permutation(calc):
    a = load_atoms("dimer"); a.set_cell(np.zeros((3, 3))); a.calc = calc
    force = a.get_forces(); energy = a.get_potential_energy()
    b = a[::-1]; b.calc = calc
    np.testing.assert_allclose(b.get_forces(), force[::-1], atol=8e-5, rtol=1e-4)
    assert abs(b.get_potential_energy() - energy) / len(a) < 1e-6


def test_cache_and_lazy_results(calc):
    a = load_atoms("bulk"); a.calc = calc; calc.reset()
    calls = []
    builder = calc._neighbor_list
    def count(*args):
        calls.append(1)
        return builder(*args)
    calc._neighbor_list = count
    try:
        e = a.get_potential_energy()
        assert set(calc.results) == {"energy", "free_energy", "forces", "stress"}
        a.get_forces(); a.get_stress(); a.get_charges(); a.get_dipole_moment()
        node = a.get_potential_energies()
        assert len(calls) == 1
        assert abs(node.sum() - e) / len(a) < 1e-8
        a.positions[0, 0] += 1e-3
        a.get_forces()
        assert len(calls) == 2
        a.set_cell(a.cell.array * 1.001, scale_atoms=True)
        a.get_stress()
        assert len(calls) == 3
    finally:
        calc._neighbor_list = builder


def test_external_field_invalidation(calc):
    a = load_atoms("dimer"); a.calc = calc; calc.reset()
    e0 = a.get_potential_energy(); mu = a.get_dipole_moment()
    field = np.array([.01, -.02, .015])
    a.info["external_field"] = field
    e1 = a.get_potential_energy()
    assert e1 - e0 == pytest.approx(np.dot(field, mu), abs=2e-8)
    field[0] += .01
    e2 = a.get_potential_energy()
    assert e2 - e1 == pytest.approx(.01 * mu[0], abs=2e-8)
    assert abs(a.get_potential_energies().sum() - e2) / len(a) < 1e-8


def test_solver_and_pbc_switching(calc):
    a = load_atoms("bulk"); a.calc = calc; calc.reset()
    e0 = a.get_potential_energy()
    assert calc._solver == "mesh"
    a.pbc = False
    assert np.isfinite(a.get_potential_energy())
    assert calc._solver == "direct" and not calc.evaluator.has_electrostatics_hook
    a.pbc = True
    assert a.get_potential_energy() == pytest.approx(e0, abs=1e-8)
    assert calc._solver == "mesh"


def test_direct_compatibility(backend):
    a = load_atoms("bulk")
    modern = MACELSC(MODEL, electrostatics="direct", **options(backend))
    old = LSCCalculator(MODEL, device=modern.device, precision=modern.dtype)
    props = ["energy", "forces", "stress", "charges", "dipole"]
    modern.calculate(a, props); old.calculate(a, props)
    assert abs(modern.results["energy"] - old.results["energy"]) / len(a) < 1e-6
    for name, atol, rtol in [("forces", 8e-5, 1e-4), ("stress", 5e-7, 1e-4),
                              ("charges", 2e-6, 1e-5), ("dipole", 3e-5, 0)]:
        np.testing.assert_allclose(modern.results[name], old.results[name], atol=atol, rtol=rtol)


def test_kernel_selection(calc, backend):
    if backend == "cpu":
        pytest.skip("CUDA dispatch counters apply only to the GPU")
    calc.calculate(load_atoms("bulk"), ["forces"])
    e = calc.evaluator
    assert e.batched_reverse_calls > 0
    assert e.batched_source_forward_calls > 0 and e.batched_source_reverse_calls > 0
    if e.can_compact_first_layer:
        assert e.compact_first_forward_calls > 0 and e.compact_first_reverse_calls > 0


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_short_ase_md(calc, case):
    a = load_atoms(case); a.calc = calc
    a.set_momenta(np.zeros((len(a), 3)))
    e0 = a.get_total_energy()
    VelocityVerlet(a, timestep=.1 * units.fs, logfile=None).run(10)
    assert np.isfinite(a.get_forces()).all() and np.isfinite(a.get_total_energy())
    assert abs(a.get_total_energy() - e0) / len(a) < 1e-3


def test_input_failures(calc):
    a = load_atoms("bulk"); a.pbc = [True, True, False]
    with pytest.raises(ValueError, match="all three"):
        calc.calculate(a)
    a.pbc = True; a.cell[2] = a.cell[1]
    with pytest.raises(ValueError, match="nonsingular"):
        calc.calculate(a)
    a = load_atoms("dimer")
    with pytest.raises(PropertyNotImplementedError, match="fully periodic"):
        calc.calculate(a, ["stress"])
    a.numbers[0] = 2
    with pytest.raises(ValueError, match="not in the model"):
        calc.calculate(a)


def test_explicit_mesh_and_field_override(backend):
    c = MACELSC(MODEL, electrostatics="mesh", external_field=(0, 0, 0), **options(backend))
    a = load_atoms("dimer")
    with pytest.raises(ValueError, match="Mesh electrostatics"):
        c.calculate(a)
    a = load_atoms("bulk"); a.calc = c
    e = a.get_potential_energy()
    a.info["external_field"] = [.1, .2, .3]
    assert a.get_potential_energy() == e
    with pytest.raises(AttributeError):
        c.mesh_spacing = .5


def test_constructor_failures():
    for kwargs in [dict(device="other"), dict(device="cpu", dtype="float32"),
                   dict(dtype="float64"), dict(mesh_spacing=0), dict(mesh_order=3),
                   dict(external_field=[0, 1]), dict(electrostatics="other")]:
        with pytest.raises(ValueError):
            MACELSC(MODEL, **kwargs)


def test_runtime_does_not_require_torch():
    code = ("import sys; sys.modules['torch'] = None; "
            "from symmetrix.experimental import MACELSC; "
            f"c=MACELSC({str(MODEL)!r},device='cpu'); "
            "assert sys.modules['torch'] is None")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
