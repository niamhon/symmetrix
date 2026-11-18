"""Native total energy against frozen ASE and independent GTO energy routines."""

import copy
import json

import numpy as np
import pytest
import symmetrix
import torch
from symmetrix.extract_mace_lsc import extract_mace_data_lsc

from .reference import DATA, load_atoms
from .test_native_local import graph_for_case

pytestmark = pytest.mark.lsc_reference


@pytest.fixture(scope="module")
def energy_document(lsc_model):
    return extract_mace_data_lsc(lsc_model, [1, 8])


def load_native(document, path):
    path.write_text(json.dumps(document, allow_nan=False))
    return symmetrix.MACELSC(str(path))


@pytest.fixture(scope="module")
def energy_native(energy_document, tmp_path_factory):
    return load_native(
        energy_document, tmp_path_factory.mktemp("native-lsc-energy") / "energy.json"
    )


def run_energy(native, model, case, *, mode=None, field=(0, 0, 0), shift=None):
    numbers, edges, vectors = graph_for_case(model, case)
    atoms = load_atoms(case)
    positions = atoms.positions.copy()
    if shift is not None:
        positions += shift
    types = np.asarray(
        [native.atomic_numbers.index(int(z)) for z in numbers], dtype=np.int64
    )
    mode = mode or ("realspace" if case == "dimer" else "mixed_periodic")
    returned = native.compute_energy(
        types,
        edges,
        vectors,
        positions,
        atoms.cell.array.copy(),
        atoms.pbc.copy(),
        mode,
        field,
    )
    assert returned == native.energy
    assert native.has_energy_results and native.can_evaluate_energy
    assert (
        native.can_evaluate is True
    )  # Native E/F/stress are supported for this export.
    assert (
        abs(
            returned
            - (native.local_energy + native.electrostatic_energy + native.field_energy)
        )
        < 1e-12
    )
    return returned, positions, atoms.cell.array.copy(), atoms.pbc.copy()


def original_coulomb(native, positions, cell, pbc, mode, include_self=True):
    """Use original library code, with native density, to isolate the energy kernel."""
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.kspace import compute_k_vectors_flat

    coulomb = GTOElectrostaticEnergy(
        1,
        native.density_smearing_width,
        native.kspace_cutoff,
        include_self_interaction=include_self,
        pbc_handling=mode,
    ).double()
    # Realspace ignores the cell; use a finite fallback only to construct unused k data.
    box = np.asarray(cell)
    if abs(np.linalg.det(box)) == 0:
        box = np.eye(3) * 30
    lattice = torch.tensor(box)[None]
    reciprocal = 2 * torch.pi * torch.linalg.inv(lattice.transpose(1, 2))
    with torch.no_grad():
        k, norm2, kbatch, k0 = compute_k_vectors_flat(
            native.kspace_cutoff, lattice, reciprocal
        )
        result = coulomb(
            k_vectors=k,
            k_norm2=norm2,
            k_vector_batch=kbatch,
            k0_mask=k0,
            source_feats=torch.tensor(native.density_coefficients).reshape(-1, 4),
            node_positions=torch.tensor(positions),
            batch=torch.zeros(len(positions), dtype=torch.int64),
            volume=torch.linalg.det(lattice).abs(),
            pbc=torch.tensor(pbc)[None],
        )
    return float(result.item())


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_total_and_coulomb_match_reference(
    energy_native, lsc_model, case, record_property
):
    total, xyz, cell, pbc = run_energy(energy_native, lsc_model, case)
    mode = "realspace" if case == "dimer" else "mixed_periodic"
    expected_coulomb = original_coulomb(energy_native, xyz, cell, pbc, mode)
    kernel_error = abs(energy_native.electrostatic_energy - expected_coulomb)
    assert kernel_error < 2e-9
    with np.load(DATA / f"{case}.reference.npz", allow_pickle=False) as reference:
        total_error = abs(total - float(reference["energy"]))
        assert total_error < 2e-6
    record_property("atoms", len(xyz))
    for name, value in {
        "total_energy_eV": total,
        "electrostatic_energy_eV": energy_native.electrostatic_energy,
        "total_energy_abs_eV": total_error,
        "same_density_coulomb_abs_eV": kernel_error,
    }.items():
        record_property(name, value)
    print(
        f"ENERGY_PARITY case={case} atoms={len(xyz)} total_abs_eV={total_error:.12e} same_density_coulomb_abs_eV={kernel_error:.12e}"
    )


def test_export_energy_constants(energy_document, lsc_model):
    from graph_longrange.utils import FIELD_CONSTANT

    p = energy_document["electrostatics"]["energy_parameters"]
    assert p["format_version"] == 1
    assert p["field_constant"] == FIELD_CONSTANT
    assert p["realspace"]["offset"] == lsc_model.coulomb_energy.realspace_energy.offset
    np.testing.assert_array_equal(
        p["realspace"]["offset_vectors"],
        torch.stack(
            [
                lsc_model.coulomb_energy.realspace_energy.x,
                lsc_model.coulomb_energy.realspace_energy.y,
                lsc_model.coulomb_energy.realspace_energy.z,
            ]
        ).numpy(),
    )
    np.testing.assert_array_equal(
        p["reciprocal"]["self_overlap"],
        lsc_model.coulomb_energy.self_interaction_terms.overlap_constants.numpy(),
    )


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_no_self_interaction_matches_original(
    energy_document, lsc_model, tmp_path, case
):
    document = copy.deepcopy(energy_document)
    document["electrostatics"]["include_self_interaction"] = False
    native = load_native(document, tmp_path / "no-self.json")
    _, xyz, cell, pbc = run_energy(native, lsc_model, case)
    expected = original_coulomb(
        native,
        xyz,
        cell,
        pbc,
        "realspace" if case == "dimer" else "mixed_periodic",
        include_self=False,
    )
    assert abs(native.electrostatic_energy - expected) < 2e-9


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_modes_field_and_translation(energy_native, lsc_model, case):
    baseline, *_ = run_energy(energy_native, lsc_model, case)
    for mode in ["auto", ""] if case == "dimer" else ["pbc", "auto"]:
        # Empty string means the checkpoint's realspace default, tested explicitly below.
        if mode == "":
            numbers, edges, vectors = graph_for_case(lsc_model, case)
            atoms = load_atoms(case)
            types = np.asarray(
                [energy_native.atomic_numbers.index(int(z)) for z in numbers]
            )
            actual = energy_native.compute_energy(
                types, edges, vectors, atoms.positions, atoms.cell.array, atoms.pbc
            )
        else:
            actual, *_ = run_energy(energy_native, lsc_model, case, mode=mode)
        assert abs(actual - baseline) < 1e-10
    shifted, *_ = run_energy(energy_native, lsc_model, case, shift=[0.3, -0.7, 0.2])
    assert abs(shifted - baseline) < 2e-9
    field = np.asarray([0.03, -0.02, 0.01])
    changed, *_ = run_energy(energy_native, lsc_model, case, field=field)
    expected = np.dot(energy_native.dipole, field)
    assert abs(energy_native.field_energy - expected) < 1e-14
    assert abs((changed - baseline) - expected) < 1e-10


def test_triclinic_reciprocal_with_dipoles(energy_native, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    types = np.asarray([energy_native.atomic_numbers.index(int(z)) for z in numbers])
    xyz = load_atoms("dimer").positions.copy()
    cell = np.asarray([[12.0, 1.5, -0.3], [0.2, 13.0, 1.2], [-0.4, 0.6, 14.0]])
    pbc = np.ones(3, dtype=bool)
    energy_native.compute_energy(types, edges, vectors, xyz, cell, pbc, "pbc")
    expected = original_coulomb(energy_native, xyz, cell, pbc, "pbc")
    assert abs(energy_native.electrostatic_energy - expected) < 2e-10
    # Translating one atom by a lattice vector leaves the periodic density energy unchanged.
    previous = energy_native.electrostatic_energy
    xyz[0] += cell[1]
    energy_native.compute_energy(types, edges, vectors, xyz, cell, pbc, "pbc")
    assert abs(energy_native.electrostatic_energy - previous) < 2e-10


def test_isolated_charge_pair_and_empty_state(energy_native, energy_document):
    types = np.asarray([0, 1], dtype=np.int64)
    edges, vectors = np.empty((2, 0), dtype=np.int64), np.empty((0, 3))
    xyz = np.asarray([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
    cell, pbc = np.zeros((3, 3)), np.zeros(3, dtype=bool)
    energy_native.compute_energy(types, edges, vectors, xyz, cell, pbc, "realspace")
    from math import erf, pi

    p = energy_document["electrostatics"]["energy_parameters"]
    expected = (
        p["field_constant"]
        / (4 * pi)
        * (-2)
        * erf(7 / (2 * p["realspace"]["sigma"]))
        / (7 + p["realspace"]["distance_epsilon"])
    )
    expected += 0.5 * (1 + 4) * p["realspace"]["self_overlap"][0]
    assert abs(energy_native.electrostatic_energy - expected) < 1e-12
    energy_native.compute_charges(types, edges, vectors, xyz)
    assert not energy_native.has_energy_results
    assert (
        energy_native.energy
        == energy_native.electrostatic_energy
        == energy_native.field_energy
        == 0
    )
    assert (
        energy_native.compute_energy(
            np.empty(0, dtype=np.int64),
            edges,
            vectors,
            np.empty((0, 3)),
            cell,
            pbc,
            "realspace",
        )
        == 0
    )
    assert energy_native.has_energy_results
    energy_native.compute_local(types, edges, vectors)
    assert not energy_native.has_energy_results


def test_legacy_export_requires_reexport(energy_document, tmp_path):
    document = copy.deepcopy(energy_document)
    del document["electrostatics"]["energy_parameters"]
    native = load_native(document, tmp_path / "legacy.json")
    assert native.can_evaluate_local and native.can_evaluate_sources
    assert not native.can_evaluate_energy
    with pytest.raises(ValueError, match="re-export"):
        native.compute_energy(
            np.empty(0, dtype=np.int64),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.zeros((3, 3)),
            np.zeros(3, dtype=bool),
        )


@pytest.mark.parametrize(
    "variant",
    [
        "cell_shape",
        "cell_nan",
        "singular",
        "pbc_shape",
        "pbc_dtype",
        "partial_pbc",
        "mixed_nonperiodic",
        "slab",
        "unknown_mode",
        "field_nan",
        "position_nan",
    ],
)
def test_invalid_energy_inputs(energy_native, variant):
    types, edges, vectors = (
        np.asarray([0], dtype=np.int64),
        np.empty((2, 0), dtype=np.int64),
        np.empty((0, 3)),
    )
    xyz, cell, pbc, mode, field = (
        np.zeros((1, 3)),
        np.eye(3) * 10,
        np.ones(3, dtype=bool),
        "pbc",
        [0.0, 0.0, 0.0],
    )
    if variant == "cell_shape":
        cell = cell.ravel()
    elif variant == "cell_nan":
        cell[0, 0] = np.nan
    elif variant == "singular":
        cell[0] = 0
    elif variant == "pbc_shape":
        pbc = pbc[None]
    elif variant == "pbc_dtype":
        pbc = pbc.astype(int)
    elif variant == "partial_pbc":
        pbc[0] = False
    elif variant == "mixed_nonperiodic":
        pbc[:] = False
        mode = "mixed_periodic"
    elif variant == "slab":
        mode = "slab"
    elif variant == "unknown_mode":
        mode = "wrong"
    elif variant == "field_nan":
        field[1] = np.nan
    elif variant == "position_nan":
        xyz[0, 0] = np.nan
    with pytest.raises(ValueError, match="MACELSC:"):
        energy_native.compute_energy(types, edges, vectors, xyz, cell, pbc, mode, field)


@pytest.mark.parametrize(
    ("part", "key", "value"),
    [
        (None, "format_version", 2),
        (None, "field_constant", -1),
        ("reciprocal", "sigma2", 0),
        ("reciprocal", "a", [1]),
        ("realspace", "offset_vectors", [[0, 0, 0]]),
        ("realspace", "self_overlap", [1, 1, -1, 1]),
    ],
)
def test_invalid_energy_parameters(energy_document, tmp_path, part, key, value):
    document = copy.deepcopy(energy_document)
    target = document["electrostatics"]["energy_parameters"]
    if part is not None:
        target = target[part]
    target[key] = value
    with pytest.raises(ValueError, match="electrostatics.energy_parameters"):
        load_native(document, tmp_path / "invalid.json")
