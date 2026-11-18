"""GPU local interactions/readouts versus native FP64 and original MACE-SCF."""

import copy
import json

import numpy as np
import pytest
import symmetrix

from .test_cuda_geometry import export_file
from .test_cuda_loader import cuda_document, require_cuda
from .test_native_local import graph_for_case, run_native, torch_local

pytestmark = pytest.mark.lsc_reference
FEATURES = ["initial_features", "A0", "M0", "H1", "A1", "M1", "H2"]


@pytest.mark.parametrize("layer", [0, 1])
def test_cuda_rejects_unsupported_convolution_order(cuda_document, layer, tmp_path):
    document = copy.deepcopy(cuda_document)
    block = document[f"conv_tp_{layer}"]
    # Reordering complete terms is valid for the generic CPU evaluator, but
    # must not silently reach a differently indexed specialized GPU kernel.
    for key in ["spec", "coefficients"]:
        block[key][0], block[key][1] = block[key][1], block[key][0]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(document, allow_nan=False))
    assert symmetrix.MACELSC(str(path)).can_evaluate_local
    with pytest.raises(ValueError, match=f"unsupported GPU convolution instructions in conv_tp_{layer}"):
        symmetrix.prepare_lsc_cuda_parameters(str(path), "float32")


@pytest.fixture(scope="module", params=["float32", "float32-mixed"])
def local_gpu(request, export_file):
    require_cuda()
    return symmetrix.MACELSCCuda(str(export_file), request.param)


@pytest.fixture(scope="module")
def native(export_file):
    return symmetrix.MACELSC(str(export_file))


@pytest.fixture(scope="module", params=["dimer", "bulk"])
def local_case(request, native, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, request.param)
    cpu = run_native(native, numbers, edges, vectors)
    original = torch_local(lsc_model, numbers, edges, vectors)
    for name in FEATURES + ["layer_node_energies", "local_node_energies"]:
        np.testing.assert_allclose(cpu[name], original[name], atol=2e-8, rtol=2e-7, err_msg=name)
    assert abs(cpu["local_energy"] - original["local_energy"]) < 2e-6
    types = np.array([native.atomic_numbers.index(int(z)) for z in numbers])
    return request.param, types, edges, vectors, cpu, original


def max_abs(actual, expected):
    return float(np.max(np.abs(actual - expected), initial=0))


def assert_local(actual, expected, energy):
    for name in FEATURES:
        assert actual[name].dtype == np.float32
        np.testing.assert_allclose(actual[name], expected[name], atol=3e-5, rtol=3e-4, err_msg=name)
    assert actual["layer_node_energies"].dtype == np.float32
    assert actual["local_node_energies"].dtype == np.float64
    np.testing.assert_allclose(actual["layer_node_energies"], expected["layer_node_energies"], atol=1e-5, rtol=1e-4)
    # Absolute errors on the small learned readouts as well as total energies
    # prevent the large atomic reference energy from hiding a broken layer.
    np.testing.assert_allclose(actual["local_node_energies"], expected["local_node_energies"], atol=2e-5, rtol=0)
    nodes = len(actual["local_node_energies"])
    assert abs(energy - expected["local_energy"]) < max(5e-5, nodes * 2e-6)


@pytest.mark.lsc_cuda
class TestLocalGpu:
    def test_dimer_bulk_all_layers(self, local_gpu, local_case, record_property):
        case, types, edges, vectors, cpu, original = local_case
        gpu = local_gpu
        energy = gpu.compute_local(types, edges, vectors)
        actual = gpu.download_local()
        assert_local(actual, cpu, energy)
        assert_local(actual, original, energy)
        assert gpu.can_evaluate_local and gpu.has_local_results
        assert gpu.has_geometry and gpu.can_evaluate
        assert gpu.can_evaluate_energy and hasattr(gpu, "compute_energy_forces")
        assert gpu.sgemm_calls > 0
        if gpu.precision == "float32":
            assert gpu.tensor_core_gemm_calls == 0
        atomic = gpu.download_parameter("atomic_energies")[types]
        assembled = atomic + actual["layer_node_energies"][0].astype(float) + actual["layer_node_energies"][1].astype(float)
        np.testing.assert_allclose(actual["local_node_energies"], assembled, atol=2e-12, rtol=0)
        assert abs(energy - actual["local_node_energies"].sum()) < 1e-8
        stats = {name + "_max_abs": max_abs(actual[name], cpu[name]) for name in FEATURES}
        stats.update(readout_max_abs_eV=max_abs(actual["layer_node_energies"], cpu["layer_node_energies"]),
                     node_energy_max_abs_eV=max_abs(actual["local_node_energies"], cpu["local_node_energies"]),
                     local_energy_abs_eV=abs(energy - cpu["local_energy"]),
                     local_energy_vs_original_abs_eV=abs(energy - original["local_energy"]))
        for key, value in dict(device=gpu.device_name, precision=gpu.precision, case=case,
                               nodes=len(types), edges=len(vectors), sgemm_calls=gpu.sgemm_calls,
                               tensor_core_gemm_calls=gpu.tensor_core_gemm_calls, **stats).items():
            record_property(key, value)
        print(f"CUDA_LOCAL {gpu.precision} {case} N={len(types)} E={len(vectors)} "
              + " ".join(f"{key}={value:.6e}" for key, value in stats.items()))

    def test_reuse_resident_geometry(self, local_gpu, local_case):
        _, types, edges, vectors, cpu, _ = local_case
        local_gpu.compute_geometry(types, edges, vectors)
        assert not local_gpu.has_local_results
        energy = local_gpu.evaluate_local()
        first = local_gpu.download_local()
        high_water = local_gpu.workspace_bytes
        assert_local(first, cpu, energy)
        assert local_gpu.evaluate_local() == energy
        assert local_gpu.workspace_bytes == high_water
        for key, values in local_gpu.download_local().items():
            np.testing.assert_array_equal(values, first[key])
        # The independent spline diagnostic must not overwrite local features.
        local_gpu.evaluate_splines("radial_0")
        assert local_gpu.has_local_results and local_gpu.has_spline_results
        for key, values in local_gpu.download_local().items():
            np.testing.assert_array_equal(values, first[key])

    def test_rotation_and_atom_edge_order(self, local_gpu, local_case):
        _, types, edges, vectors, _, _ = local_case
        gpu = local_gpu
        energy = gpu.compute_local(types, edges, vectors)
        original = gpu.download_local()
        rng = np.random.default_rng(471)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        changed_energy = gpu.compute_local(types, edges, vectors @ rotation)
        rotated = gpu.download_local()
        np.testing.assert_allclose(rotated["layer_node_energies"], original["layer_node_energies"], atol=1e-5, rtol=1e-4)
        assert abs(changed_energy - energy) < max(5e-5, len(types)*2e-6)
        order = rng.permutation(len(vectors))
        permutation = rng.permutation(len(types))
        inverse = np.argsort(permutation)
        permuted_energy = gpu.compute_local(types[permutation], inverse[edges[:, order]], vectors[order])
        permuted = gpu.download_local()
        for name in FEATURES:
            np.testing.assert_allclose(permuted[name], original[name][permutation], atol=3e-5, rtol=3e-4, err_msg=name)
        np.testing.assert_allclose(permuted["layer_node_energies"], original["layer_node_energies"][:, permutation], atol=1e-5, rtol=1e-4)
        assert abs(permuted_energy-energy) < max(5e-5, len(types)*2e-6)

    def test_empty_isolated_and_invalid_state(self, local_gpu, native):
        gpu = local_gpu
        empty_edges, empty_vectors = np.empty((2, 0), dtype=int), np.empty((0, 3))
        types = np.array([0, 1, 0])
        expected = run_native(native, np.array([1, 8, 1]), empty_edges, empty_vectors)
        assert_local(gpu_download_after_compute(gpu, types, empty_edges, empty_vectors), expected, gpu.local_energy)
        assert gpu.compute_local(np.array([], dtype=int), empty_edges, empty_vectors) == 0
        zero = gpu.download_local()
        assert zero["A0"].shape == (0, 2048) and zero["H2"].shape == (0, 512)
        assert zero["layer_node_energies"].shape == (2, 0)
        assert gpu.has_local_results and gpu.sgemm_calls == gpu.tensor_core_gemm_calls == 0
        with pytest.raises(ValueError, match="node index"):
            gpu.compute_local(types, np.array([[0], [3]]), np.array([[1., 0, 0]]))
        assert not gpu.has_geometry and not gpu.has_local_results and gpu.local_energy == 0
        with pytest.raises(RuntimeError, match="evaluate local interactions first"):
            gpu.download_local()
        with pytest.raises(RuntimeError, match="compute geometry first"):
            gpu.evaluate_local()


def gpu_download_after_compute(gpu, types, edges, vectors):
    gpu.compute_local(types, edges, vectors)
    return gpu.download_local()


@pytest.fixture(scope="module")
def modified_file(export_file, tmp_path_factory):
    document = json.loads(export_file.read_text())
    for t in range(2):
        document[f"conv_tp_{t}"]["coefficients"][0] *= 1.125
        weights = np.asarray(document[f"M{t}_weights"])
        weights[0] *= 0.9
        weights[1] *= 1.1
        document[f"M{t}_weights"] = weights.tolist()
        for e, skip in enumerate(document[f"skip_{t}"]):
            skip["weights"][0] = (np.asarray(skip["weights"][0]) * (1.1 + 0.1*e)).tolist()
            skip["bias"] = np.full(128, (e+1)*0.002).tolist()
        document[f"linear_{t}"]["bias"] = np.full(128, 0.003).tolist()
        document[f"product_linear_{t}"]["bias"] = np.full(128, -0.002).tolist()
        document[f"readout_{t}"]["bias"] = [0.013*(t+1)]
    path = tmp_path_factory.mktemp("cuda-local-modified") / "water.json"
    path.write_text(json.dumps(document, allow_nan=False))
    return path


@pytest.mark.lsc_cuda
@pytest.mark.parametrize("precision", ["float32", "float32-mixed"])
def test_exported_species_weights_biases_and_cg_are_used(modified_file, precision):
    require_cuda()
    native = symmetrix.MACELSC(str(modified_file))
    gpu = symmetrix.MACELSCCuda(str(modified_file), precision)
    numbers = np.array([1, 8, 1])
    types = np.array([0, 1, 0])
    edges = np.array([[0, 1, 2, 0], [1, 2, 0, 2]])
    vectors = np.array([[1.2, 0.3, -0.4], [-0.7, 1.5, 0.2], [0.2, -0.5, 1.8], [0.9, 1.2, 0.8]])
    expected = run_native(native, numbers, edges, vectors)
    energy = gpu.compute_local(types, edges, vectors)
    assert_local(gpu.download_local(), expected, energy)


@pytest.mark.lsc_cuda
def test_mixed_tensor_core_path_on_large_isolated_batch(export_file, native, record_property):
    require_cuda()
    gpu = symmetrix.MACELSCCuda(str(export_file), "float32-mixed")
    # Small dimer/bulk GEMMs fall below Polar's tensor-core dispatch threshold.
    # Repeated isolated H/O atoms exercise that branch without a huge edge list.
    count = 32768
    types = np.arange(count) % 2
    edges, vectors = np.empty((2, 0), dtype=int), np.empty((0, 3))
    two = run_native(native, np.array([1, 8]), edges, vectors)
    energy = gpu.compute_local(types, edges, vectors)
    assert gpu.tensor_cores and gpu.tensor_core_gemm_calls > 0
    assert gpu.sgemm_calls > 0 # scalar readouts use FP32 GEMM
    actual = gpu.download_local()
    expected = {name: two[name][types] for name in FEATURES}
    expected["layer_node_energies"] = two["layer_node_energies"][:, types]
    expected["local_node_energies"] = two["local_node_energies"][types]
    expected["local_energy"] = float(expected["local_node_energies"].sum())
    assert_local(actual, expected, energy)
    record_property("device", gpu.device_name)
    record_property("nodes", count)
    record_property("tensor_core_gemm_calls", gpu.tensor_core_gemm_calls)
    record_property("sgemm_calls", gpu.sgemm_calls)
    record_property("node_energy_max_abs_eV", max_abs(actual["local_node_energies"], expected["local_node_energies"]))
    print(f"CUDA_LOCAL_TENSOR_CORE device={gpu.device_name} N={count} "
          f"tensor_core_gemm_calls={gpu.tensor_core_gemm_calls} sgemm_calls={gpu.sgemm_calls}")
