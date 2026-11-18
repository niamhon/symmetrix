"""Basic GPU charge-source parity, conservation and periodic polarization checks."""
import numpy as np
import pytest
import symmetrix

from .reference import DATA, load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import require_cuda
from .test_native_local import graph_for_case
from .test_native_sources import run_sources, torch_sources

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda]


@pytest.fixture(scope="module", params=["float32", "float32-mixed"])
def gpu(request, export_file):
    require_cuda()
    return symmetrix.MACELSCCuda(str(export_file), request.param)


@pytest.fixture(scope="module")
def native(export_file):
    return symmetrix.MACELSC(str(export_file))


@pytest.fixture(scope="module", params=["dimer", "bulk"])
def case(request, native, lsc_model):
    name = request.param
    numbers, edges, vectors = graph_for_case(lsc_model, name)
    positions = load_atoms(name).positions.copy()
    cpu = run_sources(native, numbers, edges, vectors, positions)
    original = torch_sources(lsc_model, numbers, edges, vectors, positions)
    types = np.array([native.atomic_numbers.index(int(z)) for z in numbers])
    return name, types, numbers, edges, vectors, positions, cpu, original


def compare(actual, expected):
    for name in ["source_features", "layer_edge_fluxes", "edge_fluxes"]:
        np.testing.assert_allclose(actual[name], expected[name], atol=3e-6, rtol=3e-5, err_msg=name)
    for name in ["layer_density_coefficients", "density_coefficients", "partial_charges", "partial_dipoles"]:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-6, rtol=1e-5, err_msg=name)
    np.testing.assert_allclose(actual["dipole"], expected["dipole"], atol=3e-5, rtol=0)


def test_dimer_bulk_sources(gpu, case, record_property):
    name, types, _, edges, vectors, positions, cpu, original = case
    returned = gpu.compute_charges(types, edges, vectors, positions)
    actual = gpu.download_charges()
    compare(actual, cpu)
    compare(actual, original)
    with np.load(DATA / f"{name}.reference.npz", allow_pickle=False) as frozen:
        for key in ["density_coefficients", "partial_charges", "partial_dipoles"]:
            np.testing.assert_allclose(actual[key], frozen[key], atol=2e-6, rtol=1e-5, err_msg=key)
        np.testing.assert_allclose(actual["dipole"], frozen["dipole"], atol=3e-5, rtol=0)
    np.testing.assert_array_equal(returned, actual["dipole"])
    formal = gpu.download_parameter("formal_charges")[types].astype(float)
    assert abs(actual["partial_charges"].sum() - formal.sum()) < 1e-11
    np.testing.assert_allclose(actual["layer_density_coefficients"][:, :, 0].sum(1), 0, atol=1e-11, rtol=0)
    # Reconstruct the conservative divergence without using the GPU CSR.
    for t in range(2):
        divergence = np.zeros(len(types))
        np.add.at(divergence, edges[1], actual["layer_edge_fluxes"][t].astype(float))
        np.add.at(divergence, edges[0], -actual["layer_edge_fluxes"][t].astype(float))
        np.testing.assert_allclose(actual["layer_density_coefficients"][t, :, 0], divergence, atol=2e-13, rtol=0)
    np.testing.assert_array_equal(actual["partial_dipoles"], actual["density_coefficients"][:, [3, 1, 2]])
    np.testing.assert_allclose(actual["dipole"], actual["formal_charge_dipole"] + actual["transfer_dipole"] + actual["atomic_dipole_sum"], atol=2e-13, rtol=0)
    assert gpu.can_evaluate_sources and gpu.has_charge_results and gpu.has_local_results
    assert gpu.can_evaluate
    stats = {"charge_max_abs_e": float(abs(actual["partial_charges"] - cpu["partial_charges"]).max()),
             "atomic_dipole_max_abs_eA": float(abs(actual["partial_dipoles"] - cpu["partial_dipoles"]).max()),
             "dipole_max_abs_eA": float(abs(actual["dipole"] - cpu["dipole"]).max()),
             "charge_conservation_abs_e": abs(float(actual["partial_charges"].sum() - formal.sum()))}
    for key, value in dict(device=gpu.device_name, precision=gpu.precision, case=name,
                           source_sgemm_calls=gpu.source_sgemm_calls,
                           source_tensor_core_gemm_calls=gpu.source_tensor_core_gemm_calls, **stats).items():
        record_property(key, value)
    print(f"CUDA_SOURCES {gpu.precision} {name} " + " ".join(f"{k}={v:.6e}" for k, v in stats.items()))


def test_transformations_and_reuse(gpu, case):
    _, types, _, edges, vectors, positions, _, _ = case
    gpu.compute_charges(types, edges, vectors, positions)
    original = gpu.download_charges()
    high_water = gpu.workspace_bytes
    gpu.evaluate_charges(positions)
    assert gpu.workspace_bytes == high_water
    for key, value in gpu.download_charges().items():
        np.testing.assert_array_equal(value, original[key])
    rotation, _ = np.linalg.qr(np.random.default_rng(74).normal(size=(3, 3)))
    gpu.compute_charges(types, edges, vectors @ rotation, positions @ rotation)
    changed = gpu.download_charges()
    np.testing.assert_allclose(changed["partial_charges"], original["partial_charges"], atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(changed["partial_dipoles"], original["partial_dipoles"] @ rotation, atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(changed["dipole"], original["dipole"] @ rotation, atol=3e-5, rtol=0)


def test_periodic_image_polarization(gpu, native):
    numbers, types = np.array([8, 1, 1]), np.array([1, 0, 0])
    positions = np.array([[0.2, 0., 0.], [9.2, 0., 0.], [1.2, 1., 0.]])
    edges = np.array([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    vectors = positions[edges[1]] - positions[edges[0]]
    vectors -= 10*np.rint(vectors/10)
    cpu = run_sources(native, numbers, edges, vectors, positions)
    gpu.compute_charges(types, edges, vectors, positions)
    original = gpu.download_charges()
    compare(original, cpu)
    naive = (original["partial_charges"][:, None]*positions).sum(0) + original["partial_dipoles"].sum(0)
    assert np.max(abs(naive - original["dipole"])) > 1e-4
    wrapped = positions.copy()
    wrapped[1, 0] -= 10
    gpu.evaluate_charges(wrapped)
    np.testing.assert_allclose(np.array(gpu.dipole) - original["dipole"], [-10, 0, 0], atol=1e-12, rtol=0)


def test_empty_isolated_invalid_and_reset(gpu):
    edges, vectors = np.empty((2, 0), dtype=int), np.empty((0, 3))
    types, positions = np.array([0, 1]), np.array([[0., 0., 0.], [1., 2., -1.]])
    gpu.compute_charges(types, edges, vectors, positions)
    actual = gpu.download_charges()
    np.testing.assert_array_equal(actual["partial_charges"], [1, -2])
    np.testing.assert_array_equal(actual["partial_dipoles"], np.zeros((2, 3)))
    np.testing.assert_allclose(gpu.dipole, [-2, -4, 2], atol=1e-12, rtol=0)
    gpu.evaluate_charges(positions + [0.3, 0.2, -0.1])
    np.testing.assert_allclose(gpu.dipole, [-2.3, -4.2, 2.1], atol=1e-12, rtol=0)
    for value in [np.nan, np.inf]:
        bad = positions.copy(); bad[0, 0] = value
        with pytest.raises(ValueError, match="positions must be finite"):
            gpu.evaluate_charges(bad)
        assert not gpu.has_charge_results
        with pytest.raises(RuntimeError, match="evaluate charge sources first"):
            gpu.download_charges()
    assert gpu.compute_charges(np.array([], dtype=int), edges, vectors, np.empty((0, 3))) == [0, 0, 0]
    assert gpu.download_charges()["source_features"].shape == (2, 0, 512)
    gpu.compute_local(types, edges, vectors)
    assert not gpu.has_charge_results
    with pytest.raises(RuntimeError, match="evaluate charge sources first"):
        gpu.download_charges()
