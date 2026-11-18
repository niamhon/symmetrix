"""LSC CUDA parameter packing and actual device upload; no inference kernels yet."""

import copy
import json
import os

import numpy as np
import pytest
import symmetrix
from symmetrix.extract_mace_lsc import extract_mace_data_lsc

from .reference import REPO

pytestmark = pytest.mark.lsc_reference


@pytest.fixture(scope="module")
def cuda_document(lsc_model):
    # Layout tests need the real learned parameters, not 1,024 interpolation knots.
    return extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=16)


@pytest.fixture(scope="module")
def cuda_file(cuda_document, tmp_path_factory):
    path = tmp_path_factory.mktemp("lsc-cuda-loader") / "water.json"
    path.write_text(json.dumps(cuda_document, allow_nan=False))
    return path


@pytest.fixture(scope="module", params=["float32", "float32-mixed"])
def packed(request, cuda_file):
    return symmetrix.prepare_lsc_cuda_parameters(str(cuda_file), request.param)


def write_document(document, tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(document, allow_nan=False))
    return str(path)


def require_cuda():
    if not symmetrix.lsc_cuda_available():
        if os.environ.get("LSC_REQUIRE_CUDA") == "1":
            pytest.fail(
                "GPU test run requires a CUDA build and an allocated CUDA device"
            )
        pytest.skip("requires the LSC CUDA build and a GPU allocation")


def test_packed_metadata_and_dtypes(packed, cuda_document):
    assert packed.num_channels == 128 and packed.num_interactions == 2
    assert packed.num_elements == 2 and packed.atomic_numbers == [1, 8]
    assert packed.num_knots == 16 and packed.r_cut == 6
    assert packed.r_min == 0.5
    assert packed.parameter_bytes == sum(
        item["bytes"] for item in packed.manifest.values()
    )
    for name, info in packed.manifest.items():
        values = packed.parameter(name)
        assert list(values.shape) == info["shape"]
        assert values.dtype == np.dtype(info["dtype"])
        assert values.nbytes == info["bytes"]
        assert np.isfinite(values).all(), name
    np.testing.assert_array_equal(
        packed.parameter("atomic_energies"), cuda_document["atomic_energies"]
    )
    assert packed.parameter("atomic_energies").dtype == np.float64
    assert packed.parameter("linear_up_0.weights.0").dtype == np.float32
    assert not hasattr(packed, "compute_energy")


def test_learned_weights_and_species_mapping(packed, cuda_document):
    for layer in range(2):
        suffix = f"_{layer}"
        for name in ["linear_up", "linear", "product_linear", "readout"]:
            block = cuda_document[name + suffix]
            for l, weights in enumerate(block["weights"]):
                expected = np.asarray(weights, dtype=np.float32).reshape(
                    block["in"][l], block["out"][l]
                )
                np.testing.assert_array_equal(
                    packed.parameter(f"{name}{suffix}.weights.{l}"), expected
                )
        np.testing.assert_array_equal(
            packed.parameter(f"M{layer}_weights"),
            np.asarray(cuda_document[f"M{layer}_weights"], dtype=np.float32),
        )
        for species, block in enumerate(cuda_document["skip" + suffix]):
            for l, weights in enumerate(block["weights"]):
                np.testing.assert_array_equal(
                    packed.parameter(f"skip{suffix}.{species}.weights.{l}"),
                    np.asarray(weights, dtype=np.float32).reshape(
                        block["in"][l], block["out"][l]
                    ),
                )
        source = cuda_document["sources"][layer]
        np.testing.assert_array_equal(
            packed.parameter(f"sources.{layer}.transfer_scale"),
            np.asarray([source["transfer_scale"]], dtype=np.float32),
        )
        for name in ["linear1", "linear2", "multipoles", "transfer_readout"]:
            block = source[name]
            for l, weights in enumerate(block["weights"]):
                np.testing.assert_array_equal(
                    packed.parameter(f"sources.{layer}.{name}.weights.{l}"),
                    np.asarray(weights, dtype=np.float32).reshape(
                        block["in"][l], block["out"][l]
                    ),
                )
    energy = cuda_document["electrostatics"]["energy_parameters"]
    np.testing.assert_array_equal(
        packed.parameter("electrostatics.reciprocal.self_overlap"),
        np.asarray(energy["reciprocal"]["self_overlap"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        packed.parameter("electrostatics.realspace.offset_vectors"),
        np.asarray(energy["realspace"]["offset_vectors"], dtype=np.float32),
    )


def test_initial_embedding_matches_native_cpu(packed, cuda_file):
    cpu = symmetrix.MACELSC(str(cuda_file))
    cpu.compute_local(
        np.asarray([0, 1], dtype=np.int64),
        np.empty((2, 0), dtype=np.int64),
        np.empty((0, 3)),
    )
    expected = np.asarray(cpu.initial_features).reshape(2, 128).astype(np.float32)
    np.testing.assert_allclose(
        packed.parameter("initial_embedding"), expected, atol=2e-7, rtol=2e-7
    )


def expected_cubic(document, layer):
    values = np.asarray(document[f"radial_spline_values_{layer}"][0]).T
    derivs = np.asarray(document[f"radial_spline_derivs_{layer}"][0]).T
    weights = np.asarray(document[f"radial_weights_{layer}"])
    v, d = values @ weights, derivs @ weights
    h = (document["r_cut"] - document["radial_spline_r_min"]) / (len(v) - 1)
    a0, a1 = v[:-1], h * d[:-1]
    a2 = 3 * np.diff(v, axis=0) - h * (2 * d[:-1] + d[1:])
    a3 = -2 * np.diff(v, axis=0) + h * (d[:-1] + d[1:])
    return np.stack(
        [a0 + a1 / 2 + a2 / 4 + a3 / 8, a1 + a2 + 0.75 * a3, a2 + 1.5 * a3, a3], axis=1
    ).astype(np.float32)


def test_radial_coefficients_follow_polar_layout(packed, cuda_document):
    for layer in range(2):
        expected = expected_cubic(cuda_document, layer)
        if layer == 1 and packed.precision == "float32-mixed":
            np.testing.assert_allclose(
                packed.parameter("radial_1.c01"), expected[:, :2], atol=2e-6, rtol=2e-6
            )
            np.testing.assert_array_equal(
                packed.parameter("radial_1.c23"),
                expected[:, 2:].transpose(0, 2, 1).astype(np.float16),
            )
        else:
            np.testing.assert_allclose(
                packed.parameter(f"radial_{layer}.coefficients"),
                expected,
                atol=2e-6,
                rtol=2e-6,
            )


def test_mixed_weight_high_low_and_transpose(packed):
    name = "linear_up_1.weights.0"
    if packed.precision == "float32":
        assert not any(".mixed." in key for key in packed.manifest)
        return
    weights = packed.parameter(name)
    shift = int(packed.parameter(name + ".mixed.shift")[0])
    scaled = np.ldexp(weights, shift)
    hi = scaled.astype(np.float16)
    lo = (scaled - hi.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(packed.parameter(name + ".mixed.forward_hi"), hi.T)
    np.testing.assert_array_equal(packed.parameter(name + ".mixed.forward_lo"), lo.T)
    np.testing.assert_array_equal(packed.parameter(name + ".mixed.reverse_hi"), hi)
    np.testing.assert_array_equal(packed.parameter(name + ".mixed.reverse_lo"), lo)
    recovered = np.ldexp(hi.astype(np.float32) + lo.astype(np.float32), -shift)
    np.testing.assert_allclose(recovered, weights, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("precision", ["float64", "float16", "fp32"])
def test_rejects_unsupported_precision(cuda_file, precision):
    with pytest.raises(ValueError, match="float32 or float32-mixed"):
        symmetrix.prepare_lsc_cuda_parameters(str(cuda_file), precision)
    with pytest.raises(ValueError, match="float32 or float32-mixed"):
        symmetrix.MACELSCCuda(str(cuda_file), precision)


@pytest.mark.parametrize(
    "variant", ["architecture", "missing_energy", "product_table", "fp32_overflow"]
)
def test_validation_and_precision_guards(cuda_document, tmp_path, variant):
    document = copy.deepcopy(cuda_document)
    if variant == "architecture":
        document["num_channels"] = 64
    elif variant == "missing_energy":
        del document["electrostatics"]["energy_parameters"]
    elif variant == "product_table":
        document["B0_coefficients"][0] += 0.1
    elif variant == "fp32_overflow":
        document["node_embedding"][0][0] = 1e100
    with pytest.raises(ValueError, match="MACELSC"):
        symmetrix.prepare_lsc_cuda_parameters(
            write_document(document, tmp_path), "float32"
        )


def test_mixed_spline_range_guard(cuda_document, tmp_path):
    document = copy.deepcopy(cuda_document)
    document["radial_weights_1"] = (
        np.asarray(document["radial_weights_1"]) * 1e12
    ).tolist()
    path = write_document(document, tmp_path)
    symmetrix.prepare_lsc_cuda_parameters(path, "float32")
    with pytest.raises(ValueError, match="FP16 range"):
        symmetrix.prepare_lsc_cuda_parameters(path, "float32-mixed")


def test_mixed_rounding_edges(cuda_document, tmp_path):
    document = copy.deepcopy(cuda_document)
    weights = np.zeros((128, 128), dtype=np.float32)
    weights.flat[:8] = [
        4096,
        0,
        -0.0,
        2.0**-24,
        2.0**-25,
        3 * 2.0**-25,
        1 + 2.0**-11,
        1 + 3 * 2.0**-11,
    ]
    document["linear_up_1"]["weights"][0] = weights.ravel().tolist()
    packed = symmetrix.prepare_lsc_cuda_parameters(
        write_document(document, tmp_path), "float32-mixed"
    )
    name = "linear_up_1.weights.0"
    assert packed.parameter(name + ".mixed.shift")[0] == 0
    np.testing.assert_array_equal(
        packed.parameter(name + ".mixed.reverse_hi"), weights.astype(np.float16)
    )
    expected_lo = (weights - weights.astype(np.float16).astype(np.float32)).astype(
        np.float16
    )
    np.testing.assert_array_equal(
        packed.parameter(name + ".mixed.reverse_lo"), expected_lo
    )


def test_species_subset_packing(lsc_model, cuda_document, tmp_path):
    document = extract_mace_data_lsc(lsc_model, [8], num_spline_points=16)
    packed = symmetrix.prepare_lsc_cuda_parameters(
        write_document(document, tmp_path), "float32"
    )
    assert packed.atomic_numbers == [8] and packed.num_elements == 1
    np.testing.assert_array_equal(packed.parameter("source_element_indices"), [1])
    np.testing.assert_array_equal(
        packed.parameter("M0_weights"),
        np.asarray(cuda_document["M0_weights"], dtype=np.float32)[1:2],
    )


def test_build_capability_is_explicit(cuda_file):
    assert isinstance(symmetrix.lsc_cuda_compiled(), bool)
    assert isinstance(symmetrix.lsc_cuda_available(), bool)
    if not symmetrix.lsc_cuda_compiled():
        assert not symmetrix.lsc_cuda_available()
        with pytest.raises(RuntimeError, match="CUDA support is not built"):
            symmetrix.MACELSCCuda(str(cuda_file))


@pytest.mark.lsc_cuda
@pytest.mark.parametrize("precision", ["float32", "float32-mixed"])
def test_actual_device_parameter_round_trip(cuda_file, precision, record_property):
    require_cuda()
    host = symmetrix.prepare_lsc_cuda_parameters(str(cuda_file), precision)
    gpu = symmetrix.MACELSCCuda(str(cuda_file), precision)
    assert gpu.parameters_on_device and gpu.can_evaluate
    assert gpu.device_parameter_bytes == host.parameter_bytes
    assert gpu.manifest == host.manifest
    assert gpu.float32_mixed is (precision == "float32-mixed")
    for name in host.manifest:
        expected, actual = host.parameter(name), gpu.download_parameter(name)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), name
    assert gpu.can_evaluate_energy and hasattr(
        gpu, "compute_energy_forces"
    )
    for key, value in {
        "device": gpu.device_name,
        "precision": precision,
        "device_parameter_bytes": gpu.device_parameter_bytes,
        "parameter_arrays": len(gpu.manifest),
    }.items():
        record_property(key, value)
    print(
        f"CUDA_UPLOAD precision={precision} device={gpu.device_name} arrays={len(gpu.manifest)} bytes={gpu.device_parameter_bytes}"
    )


@pytest.mark.lsc_cuda
@pytest.mark.parametrize("precision", ["float32", "float32-mixed"])
def test_full_stage2_export_loads_on_gpu(precision, record_property):
    require_cuda()
    path = REPO / ".lsc-reference-logs/water-stage2-energy.json"
    assert path.is_file()
    gpu = symmetrix.MACELSCCuda(str(path), precision)
    assert gpu.num_knots == 1024 and gpu.atomic_numbers == [1, 8]
    assert gpu.num_channels == 128 and gpu.num_interactions == 2
    assert gpu.parameters_on_device and gpu.device_parameter_bytes > 1_000_000
    record_property("device", gpu.device_name)
    record_property("precision", precision)
    record_property("stage2_device_parameter_bytes", gpu.device_parameter_bytes)
    print(
        f"STAGE2_UPLOAD precision={precision} device={gpu.device_name} bytes={gpu.device_parameter_bytes}"
    )


@pytest.mark.lsc_cuda
def test_unknown_parameter_and_repeated_loading(cuda_file):
    require_cuda()
    for precision in ["float32", "float32-mixed", "float32"]:
        gpu = symmetrix.MACELSCCuda(str(cuda_file), precision)
        with pytest.raises(IndexError, match="unknown parameter"):
            gpu.download_parameter("does.not.exist")
        assert gpu.parameters_on_device
