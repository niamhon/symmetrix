"""Serial staging and periodic ghost bookkeeping, before adding MPI transport."""
import os
import hashlib
import json

import numpy as np
import pytest
import symmetrix

from pathlib import Path
from .reference import REPO
from .test_cuda_loader import require_cuda

pytestmark = pytest.mark.lsc_reference
BACKENDS = ["cpu", pytest.param("float32", marks=pytest.mark.lsc_cuda),
            pytest.param("float32-mixed", marks=pytest.mark.lsc_cuda)]
CASES = [("dimer", "direct"), ("bulk", "direct"), ("bulk", "mesh")]
FIELDS = [("zero", (0., 0., 0.)), ("field", (.01, -.02, .03))]
CPU_ARRAYS = ["initial_features", "A0", "A1", "M0", "M1", "H1", "H2",
              "local_node_energies", "layer_node_energies", "source_features",
              "layer_density_coefficients", "layer_edge_fluxes", "partial_charges",
              "partial_dipoles", "density_coefficients", "edge_fluxes", "forces",
              "edge_forces", "position_forces", "density_energy_gradient"]
SCALARS = ["energy", "local_energy", "electrostatic_energy", "field_energy",
           "dipole", "stress", "virial", "strain_derivative"]


@pytest.fixture(scope="module", params=BACKENDS)
def backend(request):
    if request.param != "cpu":
        require_cuda()
    return request.param


@pytest.fixture(scope="module")
def saved(backend):
    source = "cpu" if backend == "cpu" else "cuda"
    path = REPO / f".lsc-reference-logs/staged-api-before-{source}.npz"
    if source == "cuda":
        path = Path(os.environ.get("LSC_STAGED_CUDA_BASELINE", str(path)))
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["result_sha256"]
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture
def model(backend):
    path = str(REPO / ".lsc-reference-logs/water-stage2-energy.json")
    return (symmetrix.MACELSC(path) if backend == "cpu"
            else symmetrix.MACELSCCuda(path, backend))


def inputs(saved, case):
    return [saved[f"{case}/{name}"].copy()
            for name in ["types", "edges", "vectors", "positions", "cell", "pbc"]]


def graph(args, images=False):
    types, edges, vectors = args[:3]
    count = len(types)
    ids = np.arange(count, dtype=np.int64) + 2**40
    if images:
        types = np.tile(types, 2)
        ids = np.tile(ids, 2)
        edges = edges.copy()
        edges[0] += count
    return symmetrix.LSCGraph(types, edges, vectors, num_owned=count, atom_ids=ids)


def results(model, backend):
    if backend == "cpu":
        out = {key: np.asarray(getattr(model, key)) for key in CPU_ARRAYS}
    else:
        out = model.download_local()
        out.update(model.download_charges())
        out.update(model.download_forces())
    out.update({key: np.asarray(getattr(model, key)) for key in SCALARS})
    return out


def unchanged(actual, saved, backend, case, solver, label, record_property, route):
    precision = "float64" if backend == "cpu" else backend
    # These GPU derivative bounds were measured with an independent, unchanged
    # old binary on H100 (32 repeats), before this refactor. No ASE tolerance is
    # relaxed, and deterministic forward outputs retain the 1e-9 floor.
    bounds_path = Path(os.environ.get("LSC_CUDA_DIRECT_REPEATABILITY", str(REPO /
        ".lsc-reference-logs/gpu-mesh-old-h100-repeatability.json")))
    bounds = {} if backend == "cpu" else json.loads(bounds_path.read_text())["bounds"]
    maximum = 0.
    for name, value in actual.items():
        prefix = f"{precision}/{case}/{solver}/{label}"
        old = np.array([saved[f"{prefix}/{rep}/{name}"].reshape(-1) for rep in range(3)])
        value = np.asarray(value).reshape(-1)
        assert value.shape == old.shape[1:]
        gap = np.maximum(old.min(axis=0)-value, value-old.max(axis=0))
        field_gap = float(np.max(gap, initial=0))
        maximum = max(maximum, field_gap)
        record_property(f"{route}_{name}_max_gap", field_gap)
        np.testing.assert_array_less(gap, 1e-10 if backend == "cpu" else bounds.get(name, 1e-9),
                                     err_msg=f"{route}: {prefix}/{name}")
    record_property(f"{route}_max_gap_from_before", maximum)
    print(f"STAGED_UNCHANGED {backend} {case} {solver} {label} {route} max_gap={maximum:.12e}")


@pytest.mark.parametrize("case,solver", CASES)
@pytest.mark.parametrize("label,field", FIELDS)
@pytest.mark.parametrize("images", [False, True], ids=["owned", "periodic-images"])
def test_stages_preserve_before_outputs(model, backend, saved, case, solver, label, field, images, record_property):
    args = inputs(saved, case)
    if solver == "mesh":
        model.set_mesh_electrostatics(.7, 8)
    model.compute_energy_forces(*args, "auto", field, compute_stress=True)
    unchanged(results(model, backend), saved, backend, case, solver, label, record_property, "one_call")
    layout = graph(args, images)
    model.prepare_graph(layout)
    assert model.evaluation_stage == "graph"
    assert model.graph_layout.num_owned == len(args[0])
    np.testing.assert_array_equal(model.graph_layout.edge_index, layout.edge_index)
    model.compute_first_layer()
    assert model.evaluation_stage == "first_layer" and not model.has_local_results
    first = model.download_layer_features(1)
    np.testing.assert_allclose(first.reshape(-1), saved[f'{"float64" if backend == "cpu" else backend}/{case}/{solver}/{label}/0/H1'].reshape(-1), atol=1e-10 if backend == "cpu" else 1e-9, rtol=0)
    model.compute_second_layer()
    assert model.evaluation_stage == "second_layer"
    second = model.download_layer_features(2)
    model.compute_readouts()
    assert model.evaluation_stage == "readouts" and model.has_local_results
    model.evaluate_charges(args[3])
    assert model.evaluation_stage == "sources"
    model.evaluate_energy(args[4], args[5], "auto", field)
    assert model.evaluation_stage == "energy"
    model.reverse_charges()
    assert model.evaluation_stage == "reverse_sources" and not model.has_force_results
    for layer in [1, 2]:
        gradient = model.download_layer_features(layer, gradients=True)
        assert gradient.shape == first.shape and np.isfinite(gradient).all()
        assert np.any(gradient != 0)
    model.reverse_second_layer()
    assert model.evaluation_stage == "reverse_second_layer"
    model.reverse_first_layer()
    assert model.evaluation_stage == "reverse_first_layer"
    model.finalize_forces(compute_stress=True)
    assert model.evaluation_stage == "forces" and model.has_stress_results
    np.testing.assert_array_equal(model.download_layer_features(1), first)
    np.testing.assert_array_equal(model.download_layer_features(2), second)
    unchanged(results(model, backend), saved, backend, case, solver, label, record_property, "images" if images else "staged")


def test_stage_order_restart_and_empty(model, backend, saved):
    args = inputs(saved, "dimer")
    errors = (ValueError, RuntimeError)
    assert model.evaluation_stage == "empty"
    for method in [model.compute_first_layer, model.compute_second_layer, model.compute_readouts,
                   model.reverse_charges, model.reverse_second_layer, model.reverse_first_layer, model.finalize_forces]:
        with pytest.raises(errors):
            method()
    model.prepare_graph(graph(args))
    with pytest.raises(errors):
        model.download_layer_features(1)
    model.compute_first_layer()
    with pytest.raises(errors):
        model.download_layer_features(2)
    with pytest.raises(errors):
        model.download_layer_features(1, gradients=True)
    with pytest.raises(errors):
        model.evaluate_charges(args[3])
    model.compute_second_layer(); model.compute_readouts()
    with pytest.raises(errors):
        model.reverse_charges()
    model.evaluate_charges(args[3]); model.evaluate_energy(args[4], args[5], "auto")
    model.evaluate_forces(True)
    with pytest.raises(errors):
        model.finalize_forces(True)  # No double accumulation of strain.
    model.compute_first_layer()
    assert not model.has_local_results and not model.has_charge_results and not model.has_energy_results and not model.has_force_results
    with pytest.raises(errors):
        model.download_layer_features(1, gradients=True)
    # Prepare from the evaluator's own layout, then replace with an empty rank.
    model.prepare_graph(model.graph_layout)
    empty = symmetrix.LSCGraph(np.empty(0, dtype=int), np.empty((2, 0), dtype=int), np.empty((0, 3)), num_owned=0)
    model.prepare_graph(empty)
    model.compute_first_layer(); model.compute_second_layer(); model.compute_readouts()
    model.evaluate_charges(np.empty((0, 3))); model.evaluate_energy(np.eye(3)*30, np.zeros(3, dtype=bool), "auto")
    model.evaluate_forces(True)
    assert model.energy == 0 and np.array(model.stress).sum() == 0
    assert model.download_layer_features(1, gradients=True).shape == (0, 4*model.num_channels)
    assert results(model, backend)["forces"].size == 0


def test_remote_ghost_and_bad_geometry_rejected(model, backend, saved):
    args = inputs(saved, "dimer")
    model.compute_energy_forces(*args, "auto")
    remote = symmetrix.LSCGraph(np.array([0, 0]), np.array([[1], [0]]), np.array([[1., 0., 0.]]), 1, np.array([7, 99]))
    assert remote.has_remote_ghosts
    model.prepare_graph(remote)
    assert model.evaluation_stage == "graph" and not model.has_force_results
    model.compute_first_layer()
    with pytest.raises(ValueError, match="forward H1 exchange is incomplete"):
        model.compute_second_layer()
    bad = args[:3]
    bad[2] = args[2].copy(); bad[2][0] = 0
    with pytest.raises(ValueError, match="distance"):
        model.prepare_graph(symmetrix.LSCGraph(*bad))
    assert model.evaluation_stage == "empty"
    with pytest.raises((ValueError, RuntimeError)):
        model.compute_first_layer()
    model.prepare_graph(graph(args))
    model.compute_first_layer()
    assert model.evaluation_stage == "first_layer"


def test_graph_preserves_slots_ids_and_images():
    types = np.array([0, 1, 1, 0, 1])
    ids = np.array([2**40, 2**40+1, 2**40+1, 2**40, 2**40+9], dtype=np.int64)
    edges = np.array([[2, 3], [0, 1]])
    vectors = np.array([[1., 0., 0.], [-9., 0., 0.]])
    layout = symmetrix.LSCGraph(types, edges, vectors, 2, ids)
    types[:] = 9; ids[:] = 9; edges[:] = 0; vectors[:] = 0
    assert (layout.num_owned, layout.num_nodes, layout.num_ghosts, layout.num_edges) == (2, 5, 3, 2)
    np.testing.assert_array_equal(layout.local_owners, [0, 1, 1, 0, -1])
    np.testing.assert_array_equal(layout.edge_index, [[2, 3], [0, 1]])
    np.testing.assert_array_equal(layout.edge_vectors[:, 0], [1, -9])
    assert layout.atom_ids[0] == 2**40 and layout.has_remote_ghosts
    with pytest.raises(ValueError):
        layout.edge_vectors[0, 0] = 123


@pytest.mark.parametrize("bad", ["receiver-ghost", "sender-range", "duplicate-owner", "species-copy", "id-shape", "id-float", "id-overflow", "owned-range", "nan", "inf", "vectors-shape"])
def test_invalid_graph_layout(bad):
    types = np.array([0, 1, 1]); edges = np.array([[2], [0]])
    vectors = np.array([[1., 0., 0.]]); ids = np.array([7, 8, 8]); owned = 2
    if bad == "receiver-ghost": edges[1, 0] = 2
    if bad == "sender-range": edges[0, 0] = 3
    if bad == "duplicate-owner": ids = np.array([7, 7, 8]); types[:] = 0
    if bad == "species-copy": types[2] = 0
    if bad == "id-shape": ids = np.array([7, 8])
    if bad == "id-float": ids = ids.astype(float)
    if bad == "id-overflow": ids = np.array([7, 8, 2**64-1], dtype=np.uint64)
    if bad == "owned-range": owned = 4
    if bad in ["nan", "inf"]: vectors[0, 0] = float(bad)
    if bad == "vectors-shape": vectors = np.zeros((1, 2))
    with pytest.raises((ValueError, OverflowError)):
        symmetrix.LSCGraph(types, edges, vectors, owned, ids)
