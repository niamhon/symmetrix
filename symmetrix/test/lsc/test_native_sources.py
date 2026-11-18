"""Native LSC transfers and dipoles versus original source blocks and ASE snapshots."""

import json

import numpy as np
import pytest
import symmetrix
import torch
from symmetrix.extract_mace_lsc import extract_mace_data_lsc

from .reference import DATA, load_atoms
from .test_native_local import graph_for_case, hidden_layout, torch_local

pytestmark = pytest.mark.lsc_reference
C = 128


@pytest.fixture(scope="module")
def source_native(lsc_model, tmp_path_factory):
    document = extract_mace_data_lsc(lsc_model, [1, 8])
    path = tmp_path_factory.mktemp("native-lsc-sources") / "sources.json"
    path.write_text(json.dumps(document, allow_nan=False))
    return symmetrix.MACELSC(str(path))


def original_hidden(values):
    return torch.tensor(
        np.concatenate(
            [
                values[:, :C],
                values[:, C:]
                .reshape(len(values), 3, C)
                .transpose(0, 2, 1)
                .reshape(len(values), 3 * C),
            ],
            axis=1,
        )
    )


def torch_sources(model, numbers, edges, vectors, positions):
    from mace_scf.electrostatics.utils import compute_polarization

    local = torch_local(model, numbers, edges, vectors)
    with torch.no_grad():
        n = len(numbers)
        types = torch.tensor(
            [model.atomic_numbers.tolist().index(int(z)) for z in numbers]
        )
        attrs = torch.nn.functional.one_hot(types, len(model.atomic_numbers)).to(
            torch.float64
        )
        formal = model.formal_charges(attrs, torch.zeros(n))
        edge_index = torch.tensor(edges, dtype=torch.int64)
        xyz = torch.tensor(vectors, dtype=torch.float64)
        coords = torch.tensor(positions, dtype=torch.float64)
        batch = torch.zeros(n, dtype=torch.int64)
        harmonics = model.spherical_harmonics(xyz[:, [1, 2, 0]])
        radial, cutoff = model.radial_embedding(
            torch.linalg.vector_norm(xyz, dim=1, keepdim=True),
            attrs,
            edge_index,
            model.atomic_numbers,
        )
        assert cutoff is None
        density = torch.zeros(n, 4)
        density[:, 0] = formal
        flux = torch.zeros(len(vectors))
        features, contributions, transfers = [], [], []
        for t, source in enumerate(model.lr_source_maps):
            h = original_hidden(local[f"H{t + 1}"])
            ox = source.oxidation_state_linear(
                source.oxidation_basis(
                    (formal[:, None] - source.oxidation_state_range[0]) * source.factor
                    - 1
                )
            )
            features.append(hidden_layout(source.oxidation_state_tp(h, ox)))
            contribution, transfer = source(
                node_attrs=attrs,
                node_formal_charges=formal,
                node_feats=h,
                edge_attrs=harmonics,
                edge_feats=radial,
                edge_index=edge_index,
                edge_vectors=xyz,
                batch=batch,
                num_graphs=1,
            )
            contributions.append(contribution.numpy().copy())
            transfers.append(transfer[:, 0].numpy().copy())
            density += contribution
            flux += transfer[:, 0]
        dipole = compute_polarization(density, flux, xyz, edge_index, batch, 1)[0]
        dipole += (formal[:, None] * coords).sum(0)
        return {
            "source_features": np.asarray(features),
            "layer_density_coefficients": np.asarray(contributions),
            "layer_edge_fluxes": np.asarray(transfers),
            "density_coefficients": density.numpy().copy(),
            "partial_charges": density[:, 0].numpy().copy(),
            "partial_dipoles": density[:, [3, 1, 2]].numpy().copy(),
            "edge_fluxes": flux.numpy().copy(),
            "dipole": dipole.numpy().copy(),
        }


def run_sources(native, numbers, edges, vectors, positions):
    types = np.asarray(
        [native.atomic_numbers.index(int(z)) for z in numbers], dtype=np.int64
    )
    returned = native.compute_charges(types, edges, vectors, positions)
    n, e = len(numbers), edges.shape[1]
    result = {
        "source_features": np.asarray(native.source_features).reshape(2, n, 4 * C),
        "layer_density_coefficients": np.asarray(
            native.layer_density_coefficients
        ).reshape(2, n, 4),
        "layer_edge_fluxes": np.asarray(native.layer_edge_fluxes).reshape(2, e),
        "density_coefficients": np.asarray(native.density_coefficients).reshape(n, 4),
        "partial_charges": np.asarray(native.partial_charges),
        "partial_dipoles": np.asarray(native.partial_dipoles).reshape(n, 3),
        "edge_fluxes": np.asarray(native.edge_fluxes),
        "dipole": np.asarray(native.dipole),
    }
    np.testing.assert_array_equal(returned, result["dipole"])
    assert native.has_charge_results is True
    assert native.can_evaluate_sources is True
    assert native.can_evaluate is True
    return result


def test_source_blocks_at_spline_knots(source_native, lsc_model):
    numbers = np.asarray([1, 1, 8, 1, 1, 8])
    edges = np.asarray(
        [[0, 1, 2, 3, 4, 5, 0, 5], [2, 2, 3, 5, 5, 0, 5, 2]], dtype=np.int64
    )
    rng = np.random.default_rng(316)
    vectors = rng.normal(size=(8, 3))
    radii = np.linspace(0.5, 6.0, 1024)[[71, 120, 243, 330, 580, 680, 734, 900]]
    vectors *= radii[:, None] / np.linalg.norm(vectors, axis=1)[:, None]
    positions = rng.normal(size=(6, 3))
    expected = torch_sources(lsc_model, numbers, edges, vectors, positions)
    actual = run_sources(source_native, numbers, edges, vectors, positions)
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], atol=2e-10, rtol=2e-11, err_msg=name
        )


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_sources_match_frozen_ase(source_native, lsc_model, case, record_property):
    numbers, edges, vectors = graph_for_case(lsc_model, case)
    positions = load_atoms(case).positions.copy()
    actual = run_sources(source_native, numbers, edges, vectors, positions)
    with np.load(DATA / f"{case}.reference.npz", allow_pickle=False) as expected:
        for name in ["density_coefficients", "partial_charges", "partial_dipoles"]:
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                atol=2e-8,
                rtol=1e-7,
                err_msg=f"{case}: {name}",
            )
        np.testing.assert_allclose(
            actual["dipole"], expected["dipole"], atol=2e-6, rtol=0
        )
        errors = {
            "charge_max_abs_e": float(
                np.max(abs(actual["partial_charges"] - expected["partial_charges"]))
            ),
            "atomic_dipole_max_abs_eA": float(
                np.max(abs(actual["partial_dipoles"] - expected["partial_dipoles"]))
            ),
            "dipole_max_abs_eA": float(
                np.max(abs(actual["dipole"] - expected["dipole"]))
            ),
        }
    formal = np.asarray(source_native.formal_charges)[
        [source_native.atomic_numbers.index(int(z)) for z in numbers]
    ]
    errors["charge_conservation_abs_e"] = abs(
        float(actual["partial_charges"].sum() - formal.sum())
    )
    assert errors["charge_conservation_abs_e"] < 1e-10
    np.testing.assert_allclose(
        actual["layer_density_coefficients"][:, :, 0].sum(1), 0, atol=2e-12, rtol=0
    )
    np.testing.assert_allclose(
        actual["density_coefficients"][:, [3, 1, 2]],
        actual["partial_dipoles"],
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual["dipole"],
        np.asarray(source_native.formal_charge_dipole)
        + source_native.transfer_dipole
        + source_native.atomic_dipole_sum,
        atol=2e-13,
        rtol=0,
    )
    record_property("atoms", len(numbers))
    record_property("edges", edges.shape[1])
    for name, value in errors.items():
        record_property(name, value)
    print(
        f"SOURCE_PARITY case={case} atoms={len(numbers)} edges={edges.shape[1]} "
        + " ".join(f"{key}={value:.6e}" for key, value in errors.items())
    )


def test_source_rotation_translation_and_permutations(source_native, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    positions = load_atoms("dimer").positions.copy()
    original = run_sources(source_native, numbers, edges, vectors, positions)
    rng = np.random.default_rng(742)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = run_sources(
        source_native, numbers, edges, vectors @ rotation, positions @ rotation
    )
    np.testing.assert_allclose(
        rotated["partial_charges"], original["partial_charges"], atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        rotated["edge_fluxes"], original["edge_fluxes"], atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        rotated["partial_dipoles"],
        original["partial_dipoles"] @ rotation,
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        rotated["dipole"], original["dipole"] @ rotation, atol=2e-11, rtol=2e-11
    )
    translated = run_sources(
        source_native, numbers, edges, vectors, positions + [1.2, -2.1, 0.7]
    )
    np.testing.assert_allclose(
        translated["dipole"], original["dipole"], atol=2e-11, rtol=2e-11
    )
    permutation = rng.permutation(len(numbers))
    inverse = np.argsort(permutation)
    order = rng.permutation(edges.shape[1])
    permuted = run_sources(
        source_native,
        numbers[permutation],
        inverse[edges[:, order]],
        vectors[order],
        positions[permutation],
    )
    np.testing.assert_allclose(
        permuted["partial_charges"],
        original["partial_charges"][permutation],
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        permuted["partial_dipoles"],
        original["partial_dipoles"][permutation],
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        permuted["layer_edge_fluxes"],
        original["layer_edge_fluxes"][:, order],
        atol=2e-11,
        rtol=2e-11,
    )
    np.testing.assert_allclose(
        permuted["dipole"], original["dipole"], atol=2e-11, rtol=2e-11
    )


def test_periodic_flux_polarization_branch(source_native, lsc_model):
    numbers = np.asarray([8, 1, 1])
    positions = np.asarray([[0.2, 0.0, 0.0], [9.2, 0.0, 0.0], [1.2, 1.0, 0.0]])
    edges = np.asarray([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=np.int64)
    vectors = positions[edges[1]] - positions[edges[0]]
    vectors -= 10 * np.rint(vectors / 10)
    original = run_sources(source_native, numbers, edges, vectors, positions)
    expected = torch_sources(lsc_model, numbers, edges, vectors, positions)
    np.testing.assert_allclose(
        original["dipole"], expected["dipole"], atol=2e-8, rtol=1e-8
    )
    naive = (original["partial_charges"][:, None] * positions).sum(0) + original[
        "partial_dipoles"
    ].sum(0)
    assert np.max(abs(naive - original["dipole"])) > 1e-4
    wrapped = positions.copy()
    wrapped[1, 0] -= 10
    changed = run_sources(source_native, numbers, edges, vectors, wrapped)
    np.testing.assert_allclose(
        changed["partial_charges"], original["partial_charges"], atol=0, rtol=0
    )
    # Moving an H formal charge of +1 across the box changes this branch by -10 eA.
    np.testing.assert_allclose(
        changed["dipole"] - original["dipole"], [-10, 0, 0], atol=2e-12, rtol=0
    )


def test_source_species_subset(lsc_model, tmp_path):
    document = extract_mace_data_lsc(lsc_model, [8])
    path = tmp_path / "oxygen-sources.json"
    path.write_text(json.dumps(document, allow_nan=False))
    native = symmetrix.MACELSC(str(path))
    numbers = np.asarray([8, 8, 8])
    edges = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    radii = np.linspace(0.5, 6.0, 1024)[[120, 120, 250, 250]]
    vectors = radii[:, None] * np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    actual = run_sources(native, numbers, edges, vectors, positions)
    expected = torch_sources(lsc_model, numbers, edges, vectors, positions)
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], atol=2e-10, rtol=2e-11, err_msg=name
        )


def test_source_empty_cutoff_and_state_reset(source_native, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    positions = load_atoms("dimer").positions.copy()
    first = run_sources(source_native, numbers, edges, vectors, positions)
    assert source_native.compute_charges(
        np.empty(0, dtype=np.int64),
        np.empty((2, 0), dtype=np.int64),
        np.empty((0, 3)),
        np.empty((0, 3)),
    ) == [0, 0, 0]
    assert source_native.has_charge_results is True
    assert (
        source_native.partial_charges
        == source_native.partial_dipoles
        == source_native.edge_fluxes
        == []
    )
    types = np.asarray([0, 1], dtype=np.int64)
    no_edges = np.empty((2, 0), dtype=np.int64)
    no_vectors = np.empty((0, 3))
    coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0]])
    source_native.compute_charges(types, no_edges, no_vectors, coords)
    np.testing.assert_array_equal(source_native.partial_charges, [1, -2])
    np.testing.assert_array_equal(source_native.partial_dipoles, np.zeros(6))
    np.testing.assert_allclose(source_native.dipole, [-2, -4, 2], atol=1e-13, rtol=0)
    source_native.compute_charges(
        types, no_edges, no_vectors, coords + [0.3, 0.2, -0.1]
    )
    np.testing.assert_allclose(
        source_native.dipole, [-2.3, -4.2, 2.1], atol=1e-13, rtol=0
    )
    source_native.compute_charges(
        types,
        np.asarray([[0], [1]], dtype=np.int64),
        np.asarray([[6.0, 0.0, 0.0]]),
        coords,
    )
    np.testing.assert_allclose(source_native.edge_fluxes, 0, atol=1e-13, rtol=0)
    source_native.compute_local(types, no_edges, no_vectors)
    assert source_native.has_charge_results is False
    assert (
        source_native.partial_charges
        == source_native.partial_dipoles
        == source_native.edge_fluxes
        == []
    )
    repeated = run_sources(source_native, numbers, edges, vectors, positions)
    for name in first:
        np.testing.assert_array_equal(repeated[name], first[name])


def test_more_knots_improve_source_accuracy(source_native, lsc_model, tmp_path):
    document = extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=64)
    path = tmp_path / "coarse-sources.json"
    path.write_text(json.dumps(document, allow_nan=False))
    coarse = symmetrix.MACELSC(str(path))
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    positions = load_atoms("dimer").positions.copy()
    with np.load(DATA / "dimer.reference.npz", allow_pickle=False) as expected:
        coarse_error = np.max(
            abs(
                run_sources(coarse, numbers, edges, vectors, positions)[
                    "partial_charges"
                ]
                - expected["partial_charges"]
            )
        )
        fine_error = np.max(
            abs(
                run_sources(source_native, numbers, edges, vectors, positions)[
                    "partial_charges"
                ]
                - expected["partial_charges"]
            )
        )
    assert coarse_error > 1e-10
    assert fine_error < 0.1 * coarse_error


@pytest.mark.parametrize(
    "variant",
    [
        "position_shape",
        "position_count",
        "position_complex",
        "nan_position",
        "inf_position",
        "invalid_node",
        "invalid_edge",
        "nan_vector",
    ],
)
def test_sources_reject_invalid_inputs(source_native, variant):
    types = np.asarray([0, 1], dtype=np.int64)
    edges = np.asarray([[0], [1]], dtype=np.int64)
    vectors = np.asarray([[1.0, 0.0, 0.0]])
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    if variant == "position_shape":
        positions = positions.ravel()
    elif variant == "position_count":
        positions = positions[:1]
    elif variant == "position_complex":
        positions = positions.astype(complex)
    elif variant == "nan_position":
        positions[0, 0] = float("nan")
    elif variant == "inf_position":
        positions[0, 0] = float("inf")
    elif variant == "invalid_node":
        types[0] = 2
    elif variant == "invalid_edge":
        edges[0, 0] = 2
    elif variant == "nan_vector":
        vectors[0, 0] = float("nan")
    with pytest.raises(ValueError, match="MACELSC:"):
        source_native.compute_charges(types, edges, vectors, positions)
