"""C++ local LSC layers/readouts against the original CPU float64 PyTorch blocks."""

import json

import numpy as np
import pytest
import symmetrix
import torch
from symmetrix.extract_mace_lsc import extract_mace_data_lsc

from .reference import load_atoms

pytestmark = pytest.mark.lsc_reference
C = 128


@pytest.fixture(scope="module")
def local_native(lsc_model, tmp_path_factory):
    document = extract_mace_data_lsc(lsc_model, [1, 8])
    path = tmp_path_factory.mktemp("native-lsc-local") / "water-local.json"
    path.write_text(json.dumps(document, allow_nan=False))
    return symmetrix.MACELSC(str(path))


def graph_for_case(model, case):
    from mace import data
    from mace.modules.utils import get_edge_vectors_and_lengths
    from mace.tools import AtomicNumberTable

    atoms = load_atoms(case)
    config = data.config_from_atoms(atoms, head_name=model.heads[0])
    graph = data.AtomicData.from_config(
        config,
        z_table=AtomicNumberTable(model.atomic_numbers.tolist()),
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    vectors, _ = get_edge_vectors_and_lengths(
        graph.positions, graph.edge_index, graph.shifts
    )
    return atoms.numbers.copy(), graph.edge_index.numpy(), vectors.numpy()


def hidden_layout(x):
    values = x.detach().numpy()
    return np.concatenate(
        [
            values[:, :C],
            values[:, C:]
            .reshape(len(values), C, 3)
            .transpose(0, 2, 1)
            .reshape(len(values), 3 * C),
        ],
        axis=1,
    ).copy()


def torch_local(model, numbers, edge_index, vectors):
    """Run original blocks directly, omitting the separate charge/electrostatic path."""
    with torch.no_grad():
        types = torch.tensor(
            [model.atomic_numbers.tolist().index(int(z)) for z in numbers]
        )
        attrs = torch.nn.functional.one_hot(types, len(model.atomic_numbers)).to(
            torch.float64
        )
        edges = torch.tensor(edge_index, dtype=torch.int64)
        xyz = torch.tensor(vectors, dtype=torch.float64)
        lengths = torch.linalg.vector_norm(xyz, dim=1, keepdim=True)
        harmonics = model.spherical_harmonics(xyz[:, [1, 2, 0]])
        radial, cutoff = model.radial_embedding(
            lengths, attrs, edges, model.atomic_numbers
        )
        assert cutoff is None
        formal = model.formal_charges(attrs, torch.zeros(len(numbers)))
        features = model.oxidation_state_mixer(
            attrs, model.node_embedding(attrs), formal
        )
        result = {"initial_features": features.numpy().copy()}
        energies = []
        heads = torch.zeros(len(numbers), dtype=torch.int64)
        for t, (interaction, product, readout) in enumerate(
            zip(model.interactions, model.products, model.readouts)
        ):
            a, skip = interaction(
                node_attrs=attrs,
                node_feats=features,
                edge_attrs=harmonics,
                edge_feats=radial,
                edge_index=edges,
            )
            result[f"A{t}"] = (
                a.numpy().transpose(0, 2, 1).reshape(len(numbers), 16 * C).copy()
            )
            result[f"M{t}"] = hidden_layout(product.symmetric_contractions(a, attrs))
            features = product(node_feats=a, sc=skip, node_attrs=attrs)
            result[f"H{t + 1}"] = hidden_layout(features)
            energies.append(readout(features, heads).reshape(-1).numpy().copy())
        result["layer_node_energies"] = np.asarray(energies)
        e0 = model.atomic_energies_fn(attrs).reshape(len(numbers), -1)[:, 0].numpy()
        result["local_node_energies"] = e0 + energies[0] + energies[1]
        result["local_energy"] = float(result["local_node_energies"].sum())
    return result


def run_native(native, numbers, edges, vectors):
    types = np.asarray(
        [native.atomic_numbers.index(int(z)) for z in numbers], dtype=np.int64
    )
    returned = native.compute_local(types, edges, vectors)
    n = len(numbers)
    result = {
        name: np.asarray(getattr(native, name)).reshape(n, width)
        for name, width in [
            ("initial_features", C),
            ("A0", 16 * C),
            ("A1", 16 * C),
            ("M0", 4 * C),
            ("M1", 4 * C),
            ("H1", 4 * C),
            ("H2", 4 * C),
        ]
    }
    result["layer_node_energies"] = np.asarray(native.layer_node_energies)
    result["local_node_energies"] = np.asarray(native.local_node_energies)
    result["local_energy"] = returned
    assert returned == native.local_energy
    return result


def test_local_layers_at_spline_knots(local_native, lsc_model):
    # Independent edge vectors deliberately land on knots: this isolates C++
    # embedding/layout/convolution/product arithmetic from interpolation error.
    numbers = np.asarray([1, 1, 8, 1, 1, 8])
    edges = np.asarray(
        [[0, 1, 2, 3, 4, 5, 0, 5], [2, 2, 3, 5, 5, 0, 5, 2]], dtype=np.int64
    )
    lengths = np.linspace(0.5, 6.0, 1024)[[71, 120, 243, 330, 580, 680, 734, 900]]
    vectors = np.random.default_rng(116).normal(size=(8, 3))
    vectors *= lengths[:, None] / np.linalg.norm(vectors, axis=1)[:, None]
    expected = torch_local(lsc_model, numbers, edges, vectors)
    actual = run_native(local_native, numbers, edges, vectors)
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], atol=2e-10, rtol=2e-11, err_msg=name
        )
    assert local_native.can_evaluate_local is True
    assert local_native.can_evaluate is True


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_real_geometry_local_parity(local_native, lsc_model, case, record_property):
    numbers, edges, vectors = graph_for_case(lsc_model, case)
    expected = torch_local(lsc_model, numbers, edges, vectors)
    actual = run_native(local_native, numbers, edges, vectors)
    for name in ["initial_features", "A0", "A1", "M0", "M1", "H1", "H2"]:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            atol=2e-8,
            rtol=2e-7,
            err_msg=f"{case}: {name}",
        )
    np.testing.assert_allclose(
        actual["layer_node_energies"],
        expected["layer_node_energies"],
        atol=1e-8,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        actual["local_node_energies"],
        expected["local_node_energies"],
        atol=1e-8,
        rtol=0,
    )
    assert abs(actual["local_energy"] - expected["local_energy"]) < 2e-6
    errors = {
        "H1_max_abs": float(np.max(abs(actual["H1"] - expected["H1"]))),
        "H2_max_abs": float(np.max(abs(actual["H2"] - expected["H2"]))),
        "readout_max_abs_eV": float(
            np.max(abs(actual["layer_node_energies"] - expected["layer_node_energies"]))
        ),
        "local_energy_abs_eV": abs(actual["local_energy"] - expected["local_energy"]),
    }
    record_property("atoms", len(numbers))
    record_property("edges", edges.shape[1])
    for name, value in errors.items():
        record_property(name, value)
    print(
        f"LOCAL_PARITY case={case} atoms={len(numbers)} edges={edges.shape[1]} "
        + " ".join(f"{key}={value:.6e}" for key, value in errors.items())
    )


def test_local_energy_rotation_and_ordering(local_native, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    original = run_native(local_native, numbers, edges, vectors)
    rng = np.random.default_rng(383)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = run_native(local_native, numbers, edges, vectors @ rotation)
    np.testing.assert_allclose(
        rotated["layer_node_energies"],
        original["layer_node_energies"],
        atol=2e-11,
        rtol=2e-11,
    )
    assert abs(rotated["local_energy"] - original["local_energy"]) < 2e-9
    permutation = rng.permutation(len(numbers))
    inverse = np.argsort(permutation)
    order = rng.permutation(edges.shape[1])
    permuted = run_native(
        local_native, numbers[permutation], inverse[edges[:, order]], vectors[order]
    )
    np.testing.assert_allclose(
        permuted["H2"], original["H2"][permutation], atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        permuted["local_node_energies"],
        original["local_node_energies"][permutation],
        atol=2e-9,
        rtol=0,
    )


def test_local_species_subset(lsc_model, tmp_path):
    data_out = extract_mace_data_lsc(lsc_model, [8])
    path = tmp_path / "oxygen-local.json"
    path.write_text(json.dumps(data_out, allow_nan=False))
    native = symmetrix.MACELSC(str(path))
    numbers = np.asarray([8, 8, 8])
    edges = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    radii = np.linspace(0.5, 6.0, 1024)[[120, 120, 250, 250]]
    vectors = radii[:, None] * np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    actual = run_native(native, numbers, edges, vectors)
    expected = torch_local(lsc_model, numbers, edges, vectors)
    np.testing.assert_allclose(actual["H2"], expected["H2"], atol=2e-10, rtol=2e-11)
    np.testing.assert_allclose(
        actual["layer_node_energies"],
        expected["layer_node_energies"],
        atol=2e-10,
        rtol=2e-11,
    )


def test_empty_graph_cutoff_and_repeated_calls(local_native, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    first = run_native(local_native, numbers, edges, vectors)
    assert (
        local_native.compute_local(
            np.empty(0, dtype=np.int64),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 3)),
        )
        == 0
    )
    assert local_native.H1 == local_native.H2 == local_native.local_node_energies == []
    no_edges = np.empty((2, 0), dtype=np.int64)
    no_vectors = np.empty((0, 3))
    isolated = run_native(local_native, np.asarray([1, 8]), no_edges, no_vectors)
    expected = torch_local(lsc_model, np.asarray([1, 8]), no_edges, no_vectors)
    np.testing.assert_allclose(
        isolated["layer_node_energies"],
        expected["layer_node_energies"],
        atol=2e-11,
        rtol=2e-11,
    )
    at_cutoff = run_native(
        local_native,
        np.asarray([1, 8]),
        np.asarray([[0], [1]], dtype=np.int64),
        np.asarray([[6.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(at_cutoff["H2"], isolated["H2"], atol=2e-11, rtol=2e-11)
    repeated = run_native(local_native, numbers, edges, vectors)
    np.testing.assert_array_equal(repeated["H2"], first["H2"])
    assert repeated["local_energy"] == first["local_energy"]


def test_more_knots_improve_local_readout_accuracy(local_native, lsc_model, tmp_path):
    document = extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=64)
    path = tmp_path / "coarse-local.json"
    path.write_text(json.dumps(document, allow_nan=False))
    coarse = symmetrix.MACELSC(str(path))
    numbers, edges, vectors = graph_for_case(lsc_model, "dimer")
    expected = torch_local(lsc_model, numbers, edges, vectors)["layer_node_energies"]
    coarse_error = np.max(
        abs(
            run_native(coarse, numbers, edges, vectors)["layer_node_energies"]
            - expected
        )
    )
    fine_error = np.max(
        abs(
            run_native(local_native, numbers, edges, vectors)["layer_node_energies"]
            - expected
        )
    )
    assert coarse_error > 1e-10
    assert fine_error < coarse_error * 0.1


@pytest.mark.parametrize(
    "variant",
    [
        "node_shape",
        "edge_shape",
        "vector_shape",
        "edge_count",
        "node_float",
        "edge_float",
        "node_type",
        "edge_negative",
        "edge_atom",
        "edge_overflow",
        "nan",
        "inf",
        "too_close",
        "too_far",
        "complex",
    ],
)
def test_local_rejects_invalid_graph(local_native, variant):
    types = np.asarray([0, 1], dtype=np.int64)
    edges = np.asarray([[0], [1]], dtype=np.int64)
    vectors = np.asarray([[1.0, 0.0, 0.0]])
    if variant == "node_shape":
        types = types[:, None]
    elif variant == "edge_shape":
        edges = edges.T
    elif variant == "vector_shape":
        vectors = vectors.ravel()
    elif variant == "edge_count":
        vectors = np.zeros((2, 3))
    elif variant == "node_float":
        types = types.astype(float)
    elif variant == "edge_float":
        edges = edges.astype(float)
    elif variant == "node_type":
        types[0] = 2
    elif variant == "edge_negative":
        edges[0, 0] = -1
    elif variant == "edge_atom":
        edges[0, 0] = 2
    elif variant == "edge_overflow":
        edges[0, 0] = 2**32
    elif variant == "nan":
        vectors[0, 0] = float("nan")
    elif variant == "inf":
        vectors[0, 0] = float("inf")
    elif variant == "too_close":
        vectors[0, 0] = 0.49
    elif variant == "too_far":
        vectors[0, 0] = 6.01
    elif variant == "complex":
        vectors = vectors.astype(complex)
    with pytest.raises(ValueError, match="MACELSC:"):
        local_native.compute_local(types, edges, vectors)


@pytest.mark.parametrize(
    ("name", "value"), [("normalization", "integral"), ("normalize", False)]
)
def test_export_requires_matching_harmonics(lsc_model, monkeypatch, name, value):
    monkeypatch.setattr(lsc_model.spherical_harmonics, name, value)
    with pytest.raises(
        NotImplementedError, match="component-normalized unit-vector harmonics"
    ):
        extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=8)
