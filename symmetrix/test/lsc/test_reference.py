"""Test the original ASE calculator against frozen predictions, not an LSC port."""

import importlib.metadata

import numpy as np
import pytest

from .reference import (
    DATA,
    MODEL_SHA256,
    REFERENCE_VERSIONS,
    evaluate,
    load_atoms,
    load_manifest,
    make_calculator,
    model_path,
    sha256,
)

pytestmark = pytest.mark.lsc_reference

# These tolerances describe CPU float64 reference reproducibility. Future
# spline/CUDA/mixed-precision implementations need their own accuracy contract.
TOLERANCES = {
    "energy": (1e-8, 0.0),  # Absolute total-energy tolerance; do not scale by E0.
    "forces": (2e-8, 1e-8),
    "dipole": (2e-9, 1e-9),
    "partial_charges": (1e-10, 1e-9),
    "partial_dipoles": (1e-10, 1e-9),
    "density_coefficients": (1e-10, 1e-9),
    "stress": (1e-10, 1e-8),
}


@pytest.fixture(scope="module")
def manifest():
    data = load_manifest()
    assert data["schema_version"] == 1
    assert data["model"]["sha256"] == MODEL_SHA256
    return data


@pytest.fixture(scope="module")
def checkpoint():
    path = model_path()
    if not path.is_file():
        pytest.fail(
            f"Missing reference checkpoint: {path}. Set LSC_REFERENCE_MODEL to a "
            "copy of the pinned stage-2 model; see the LSC test README."
        )
    assert sha256(path) == MODEL_SHA256, "The reference checkpoint has changed"
    return path


@pytest.fixture(scope="module", params=["dimer", "bulk"])
def case(request, manifest):
    name = request.param
    return name, manifest["cases"][name]


@pytest.fixture(scope="module")
def prediction(case, checkpoint):
    name, spec = case
    calculator = make_calculator(checkpoint, spec["pbc_handling"])
    return evaluate(load_atoms(name), calculator, spec["include_stress"])


def test_reference_environment_matches_snapshot(manifest):
    actual = {
        name: importlib.metadata.version(name) for name in REFERENCE_VERSIONS
    }
    assert actual == manifest["reference"]["packages"] == REFERENCE_VERSIONS


def test_saved_fixture_integrity(case):
    name, spec = case
    assert sha256(DATA / f"{name}.extxyz") == spec["geometry_sha256"]
    assert sha256(DATA / f"{name}.reference.npz") == spec["outputs_sha256"]
    atoms = load_atoms(name)
    assert len(atoms) == spec["num_atoms"]
    np.testing.assert_array_equal(atoms.pbc, spec["pbc"])
    np.testing.assert_array_equal(atoms.cell.array, spec["cell_A"])
    assert not any(key.startswith("REF_") for key in atoms.info)
    assert not any(key.startswith("REF_") for key in atoms.arrays)


def test_reference_outputs_match_saved_predictions(case, prediction):
    name, spec = case
    with np.load(DATA / f"{name}.reference.npz", allow_pickle=False) as expected:
        assert set(prediction) == set(expected.files)
        assert ("stress" in prediction) == spec["include_stress"]
        for field, actual in prediction.items():
            assert actual.shape == expected[field].shape, field
            assert np.isfinite(actual).all(), field
            atol, rtol = TOLERANCES[field]
            np.testing.assert_allclose(
                actual,
                expected[field],
                atol=atol,
                rtol=rtol,
                err_msg=f"{name}: {field} changed from the frozen ASE baseline",
            )


def test_formal_charge_is_conserved(case, prediction):
    _, spec = case
    np.testing.assert_allclose(
        prediction["partial_charges"].sum(),
        spec["total_formal_charge"],
        atol=1e-10,
        rtol=0,
    )


def test_dimer_forces_are_negative_energy_derivative(checkpoint):
    atoms = load_atoms("dimer")
    calculator = make_calculator(checkpoint, "realspace")
    direction = np.random.default_rng(18423).normal(size=atoms.positions.shape)
    direction -= direction.mean(axis=0)  # Exercise deformation, not translation.
    direction /= np.linalg.norm(direction)
    step = 1e-4  # Angstrom
    energies = []
    for sign in [1, -1]:
        displaced = atoms.copy()
        displaced.positions += sign * step * direction
        displaced.calc = calculator
        energies.append(displaced.get_potential_energy())
    finite_difference = (energies[0] - energies[1]) / (2 * step)
    with np.load(DATA / "dimer.reference.npz", allow_pickle=False) as expected:
        force_derivative = -np.sum(expected["forces"] * direction)
    np.testing.assert_allclose(
        finite_difference,
        force_derivative,
        atol=5e-5,
        rtol=1e-5,
        err_msg="Dimer forces disagree with an independent directional energy derivative",
    )
