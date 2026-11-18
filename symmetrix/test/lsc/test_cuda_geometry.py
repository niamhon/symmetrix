"""Actual CUDA geometry/radial primitives, independent of future LSC interactions."""

import hashlib
import json

import numpy as np
import pytest
import symmetrix
import torch
from scipy.interpolate import CubicHermiteSpline

from .reference import REPO
from .test_cuda_loader import require_cuda
from .test_native_local import graph_for_case

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda]
RADIALS = ["radial_0", "radial_1", "sources.0.radial", "sources.1.radial"]
EXPORT_SHA = "7bbbe952d1e5b819a2925047943be217181116de6fb5f357e58ab28527b46f65"


@pytest.fixture(scope="module")
def export_file():
    require_cuda()
    path = REPO / ".lsc-reference-logs/water-stage2-energy.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPORT_SHA
    return path


@pytest.fixture(scope="module")
def document(export_file):
    return json.loads(export_file.read_text())


@pytest.fixture(scope="module", params=["float32", "float32-mixed"])
def gpu(request, export_file):
    require_cuda()
    result = symmetrix.MACELSCCuda(str(export_file), request.param)
    assert result.parameters_on_device and result.can_evaluate
    return result


@pytest.fixture(scope="module", params=["dimer", "bulk"])
def case_graph(request, lsc_model):
    numbers, edges, vectors = graph_for_case(lsc_model, request.param)
    species = lsc_model.atomic_numbers.tolist()
    types = np.asarray([species.index(int(z)) for z in numbers], dtype=np.int64)
    return request.param, types, edges, vectors


@pytest.fixture(scope="module")
def cpu_splines(document):
    grid = np.linspace(document["radial_spline_r_min"], document["r_cut"], 1024)
    out = {}
    for name in RADIALS:
        if name.startswith("radial_"):
            i = name[-1]
            values = document[f"radial_spline_values_{i}"][0]
            derivatives = document[f"radial_spline_derivs_{i}"][0]
            weights = document[f"radial_weights_{i}"]
        else:
            source = document["sources"][int(name.split(".")[1])]
            values, derivatives, weights = (
                source["radial_spline_values"], source["radial_spline_derivs"],
                source["radial_weights"],
            )
        v = np.asarray(values).T @ np.asarray(weights)
        d = np.asarray(derivatives).T @ np.asarray(weights)
        spline = CubicHermiteSpline(grid, v, d)
        h = (grid[-1] - grid[0]) / (len(grid) - 1)
        a1 = h * d[:-1]
        a2 = 3 * np.diff(v, axis=0) - h * (2 * d[:-1] + d[1:])
        a3 = -2 * np.diff(v, axis=0) + h * (d[:-1] + d[1:])
        spline.centered_coefficients = np.stack([
            v[:-1] + a1 / 2 + a2 / 4 + a3 / 8,
            a1 + a2 + 0.75 * a3, a2 + 1.5 * a3, a3,
        ], axis=1)
        out[name] = spline
    return out


def assert_csr(result, edges, nodes):
    for label, endpoints in [("incoming", edges[1]), ("outgoing", edges[0])]:
        order = np.argsort(endpoints, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(endpoints, minlength=nodes))]
        np.testing.assert_array_equal(result[label + "_starts"], starts)
        np.testing.assert_array_equal(result[label + "_edges"], order)
        assert result[label + "_edges"].dtype == np.int32


def errors(actual, expected):
    return float(np.max(np.abs(actual - expected), initial=0))


def test_geometry_and_angular_derivatives(gpu, case_graph, lsc_model, record_property):
    case, types, edges, vectors = case_graph
    gpu.compute_geometry(types, edges, vectors)
    actual = gpu.download_geometry()
    radii = np.linalg.norm(vectors, axis=1)
    expected_geometry = np.column_stack((vectors / radii[:, None], radii))
    np.testing.assert_allclose(actual["geometry"], expected_geometry, atol=3e-7, rtol=2e-7)
    xyz = torch.tensor(vectors, requires_grad=True)
    y = lsc_model.spherical_harmonics(xyz[:, [1, 2, 0]]) / (2 * np.sqrt(np.pi))
    grad = torch.stack([
        torch.autograd.grad(y[:, m].sum(), xyz, retain_graph=True)[0]
        for m in range(16)
    ], dim=2).detach().numpy()
    expected_y = y.detach().numpy()
    np.testing.assert_allclose(actual["spherical_harmonics"], expected_y, atol=5e-7, rtol=2e-6)
    np.testing.assert_allclose(actual["spherical_harmonics_gradient"], grad, atol=2e-6, rtol=4e-6)
    assert_csr(actual, edges, len(types))
    assert gpu.has_geometry and not gpu.has_spline_results and gpu.can_evaluate
    assert gpu.num_nodes == len(types) and gpu.num_edges == len(vectors)
    assert actual["geometry"].dtype == np.float32
    for key, value in {
        "device": gpu.device_name, "precision": gpu.precision, "case": case,
        "nodes": len(types), "edges": len(vectors),
        "geometry_max_abs": errors(actual["geometry"], expected_geometry),
        "harmonics_max_abs": errors(actual["spherical_harmonics"], expected_y),
        "angular_derivative_max_abs": errors(actual["spherical_harmonics_gradient"], grad),
    }.items():
        record_property(key, value)
    print(f"CUDA_GEOMETRY {gpu.precision} {case} N={len(types)} E={len(vectors)} "
          f"Y_error={errors(actual['spherical_harmonics'], expected_y):.3e} "
          f"dY_error={errors(actual['spherical_harmonics_gradient'], grad):.3e}")


@pytest.fixture(scope="module")
def gpu_tables(gpu, cpu_splines):
    tables = {}
    for name in RADIALS:
        reference = cpu_splines[name].centered_coefficients
        if gpu.float32_mixed and name == "radial_1":
            c01 = gpu.download_parameter(name + ".c01").astype(np.float64)
            c23 = gpu.download_parameter(name + ".c23").astype(np.float64).transpose(0, 2, 1)
            stored = np.concatenate([c01, c23], axis=1)
        else:
            stored = gpu.download_parameter(name + ".coefficients").astype(np.float64)
        # Check storage precision independently of the evaluation kernel.
        bound = 2 * np.finfo(np.float32).eps * np.abs(reference) + 1e-10
        if gpu.float32_mixed and name == "radial_1":
            bound[:, 2:] += (np.finfo(np.float16).eps / 2 * np.abs(reference[:, 2:])
                            + 2.0**-25)
        assert np.all(np.abs(stored - reference) <= bound)
        tables[name] = stored, gpu.download_parameter(name + ".grid")
    return tables


def check_splines(gpu, name, radii, spline, gpu_tables):
    """Separate CUDA arithmetic from unavoidable coordinate/storage rounding.

    For p(u)=sum(c_j*u**j), bound coefficient errors term by term and
    coordinate errors with max|p'| (or max|p''| for radial derivatives).
    This compares against the independent FP64 Hermite spline, without
    choosing a larger global tolerance from the observed test errors.
    """
    result = gpu.download_splines()
    geometry = gpu.download_geometry()
    stored, grid = gpu_tables[name]
    h = (spline.x[-1] - spline.x[0]) / (len(spline.x) - 1)
    inv_h, exact_inv_h = float(grid[2]), 1 / h
    maxima = dict(value_max_abs=0., radial_derivative_max_abs=0.,
                  value_precision_bound=0., radial_derivative_precision_bound=0.,
                  arithmetic_value_max_abs=0., arithmetic_derivative_max_abs=0.)
    # Bound peak host memory for the bulk graph's 22 million radial outputs.
    for first in range(0, len(radii), 512):
        sl = slice(first, first + 512)
        k = geometry["intervals"][sl]
        u = geometry["spline_coordinate"][sl, None].astype(np.float64)
        exact_u = ((radii[sl] - spline.x[0]) / h - k - 0.5)[:, None]
        du, umax = np.abs(u - exact_u), np.maximum(np.abs(u), np.abs(exact_u))
        c, original = stored[k], spline.centered_coefficients[k]
        delta = np.abs(c - original)
        polynomial_value = sum(c[:, j] * u**j for j in range(4))
        derivative = inv_h * sum(j * c[:, j] * u**(j-1) for j in range(1, 4))
        # Conservative bound for FP32 Horner arithmetic, including FMA choices.
        eps = np.finfo(np.float32).eps
        round_v = 8 * eps * sum(np.abs(c[:, j] * u**j) for j in range(4)) + 1e-10
        round_d = 8 * eps * inv_h * sum(np.abs(j*c[:, j]*u**(j-1)) for j in range(1, 4)) + 1e-8
        actual_v, actual_d = result["values"][sl], result["radial_derivatives"][sl]
        assert np.all(np.abs(actual_v - polynomial_value) <= round_v)
        assert np.all(np.abs(actual_d - derivative) <= round_d)
        coefficient_v = sum(delta[:, j] * np.abs(u)**j for j in range(4))
        coefficient_d = inv_h * sum(j*delta[:, j]*np.abs(u)**(j-1) for j in range(1, 4))
        coordinate_v = du * sum(j*np.abs(original[:, j])*umax**(j-1) for j in range(1, 4))
        coordinate_d = du * inv_h * sum(j*(j-1)*np.abs(original[:, j])*umax**(j-2) for j in range(2, 4))
        spacing_d = abs(inv_h - exact_inv_h) * sum(
            j*np.abs(original[:, j])*np.abs(exact_u)**(j-1) for j in range(1, 4))
        bound_v = coefficient_v + coordinate_v + round_v + 5e-10
        bound_d = coefficient_d + coordinate_d + spacing_d + round_d + 2e-7
        expected_v, expected_d = spline(radii[sl]), spline(radii[sl], 1)
        np.testing.assert_allclose(actual_v, expected_v, atol=1e-5, rtol=2e-5)
        assert np.all(np.abs(actual_v - expected_v) <= bound_v)
        assert np.all(np.abs(actual_d - expected_d) <= bound_d)
        for key, value in {
            "value_max_abs": errors(actual_v, expected_v),
            "radial_derivative_max_abs": errors(actual_d, expected_d),
            "value_precision_bound": float(bound_v.max()),
            "radial_derivative_precision_bound": float(bound_d.max()),
            "arithmetic_value_max_abs": errors(actual_v, polynomial_value),
            "arithmetic_derivative_max_abs": errors(actual_d, derivative),
        }.items():
            maxima[key] = max(maxima[key], value)
    return maxima


@pytest.mark.parametrize("name", RADIALS)
def test_radial_values_and_derivatives(gpu, case_graph, cpu_splines, gpu_tables, name, record_property):
    case, types, edges, vectors = case_graph
    gpu.compute_geometry(types, edges, vectors)
    gpu.evaluate_splines(name)
    stats = check_splines(gpu, name, np.linalg.norm(vectors, axis=1), cpu_splines[name], gpu_tables)
    assert gpu.has_spline_results and gpu.spline_name == name
    for key, value in dict(precision=gpu.precision, case=case, spline=name, **stats).items():
        record_property(key, value)
    print(f"CUDA_SPLINE {gpu.precision} {case} {name} "
          f"value_error={stats['value_max_abs']:.3e} dR_error={stats['radial_derivative_max_abs']:.3e} "
          f"GPU_dR_arithmetic_error={stats['arithmetic_derivative_max_abs']:.3e}")


@pytest.mark.parametrize("name", RADIALS)
def test_knots_endpoints_and_last_interval(gpu, cpu_splines, gpu_tables, name):
    grid = cpu_splines[name].x
    radii = np.r_[grid[[0, 1, 127, 511, 1022, 1023]],
                  (grid[:-1:127] + grid[1::127]) / 2]
    vectors = np.column_stack((radii, np.zeros((len(radii), 2))))
    edges = np.tile([[0], [1]], (1, len(radii)))
    gpu.compute_geometry(np.array([0, 1]), edges, vectors)
    geo = gpu.download_geometry()
    assert geo["intervals"][0] == 0 and geo["intervals"][5] == 1022
    np.testing.assert_allclose(geo["spline_coordinate"][[0, 5]], [-0.5, 0.5], atol=1e-4)
    assert (geo["intervals"] >= 0).all() and (geo["intervals"] <= 1022).all()
    gpu.evaluate_splines(name)
    check_splines(gpu, name, radii, cpu_splines[name], gpu_tables)


def test_edge_atom_permutation_and_direction(gpu, case_graph):
    _, types, edges, vectors = case_graph
    gpu.compute_geometry(types, edges, vectors)
    original = gpu.download_geometry()
    rng = np.random.default_rng(183)
    order = rng.permutation(len(vectors))
    atoms = rng.permutation(len(types))
    inverse = np.argsort(atoms)
    changed_edges = inverse[edges[:, order]]
    gpu.compute_geometry(types[atoms], changed_edges, vectors[order])
    changed = gpu.download_geometry()
    for key in ["geometry", "spherical_harmonics", "spherical_harmonics_gradient", "intervals", "spline_coordinate"]:
        np.testing.assert_array_equal(changed[key], original[key][order])
    assert_csr(changed, changed_edges, len(types))
    gpu.compute_geometry(types, edges[::-1], -vectors)
    reverse = gpu.download_geometry()
    parity = np.r_[1, np.full(3, -1), np.ones(5), np.full(7, -1)]
    np.testing.assert_allclose(reverse["spherical_harmonics"], original["spherical_harmonics"] * parity, atol=2e-7, rtol=0)
    np.testing.assert_allclose(reverse["spherical_harmonics_gradient"], -original["spherical_harmonics_gradient"] * parity, atol=2e-7, rtol=0)
    assert_csr(reverse, edges[::-1], len(types))


def test_empty_graph_state_and_workspace_reuse(gpu, lsc_model):
    for case in ["bulk", "dimer"]:
        numbers, edges, vectors = graph_for_case(lsc_model, case)
        types = np.asarray([gpu.atomic_numbers.index(int(z)) for z in numbers])
        gpu.compute_geometry(types, edges, vectors)
        gpu.evaluate_splines("radial_1")
        if case == "bulk":
            high_water = gpu.workspace_bytes
        else:
            assert gpu.workspace_bytes == high_water
    for types in [np.array([], dtype=int), np.array([0, 1])]:
        gpu.compute_geometry(types, np.empty((2, 0), dtype=int), np.empty((0, 3)))
        assert gpu.has_geometry and not gpu.has_spline_results and gpu.spline_name == ""
        with pytest.raises(RuntimeError, match="evaluate splines first"):
            gpu.download_splines()
        geo = gpu.download_geometry()
        assert geo["geometry"].shape == (0, 4)
        assert geo["spherical_harmonics_gradient"].shape == (0, 3, 16)
        np.testing.assert_array_equal(geo["incoming_starts"], np.zeros(len(types) + 1))
        for name in RADIALS:
            gpu.evaluate_splines(name)
            out = gpu.download_splines()
            assert out["values"].shape == (0, 1280 if name == "radial_1" else 512 if name == "radial_0" else 256)
        assert gpu.workspace_bytes == high_water


@pytest.mark.parametrize("bad,match", [
    ("species", "species index"), ("node", "node index"),
    ("nan", "finite"), ("inf", "finite"), ("minus_inf", "finite"),
    ("zero", "edge distance"), ("short", "edge distance"),
    ("long", "edge distance"), ("overflow", "edge distance"),
])
def test_device_validation_invalidates_old_results(gpu, bad, match):
    types = np.array([0, 1])
    edges = np.array([[0], [1]])
    vectors = np.array([[1.0, 0, 0]])
    gpu.compute_geometry(types, edges, vectors)
    gpu.evaluate_splines("radial_0")
    if bad == "species":
        types[1] = 2
    elif bad == "node":
        edges[1, 0] = 2
    else:
        vectors[0, 0] = {"nan": np.nan, "inf": np.inf, "minus_inf": -np.inf,
                         "zero": 0, "short": np.nextafter(gpu.r_min, 0),
                         "long": np.nextafter(gpu.r_cut, np.inf), "overflow": 1e308}[bad]
    with pytest.raises(ValueError, match=match):
        gpu.compute_geometry(types, edges, vectors)
    assert not gpu.has_geometry and not gpu.has_spline_results
    with pytest.raises(RuntimeError, match="compute geometry first"):
        gpu.download_geometry()
    with pytest.raises(RuntimeError, match="compute geometry first"):
        gpu.evaluate_splines("radial_0")


@pytest.mark.parametrize("bad,match", [
    ("shape", "expected node_types"), ("float_indices", "integer dtype"),
    ("complex_vectors", "real numeric dtype"), ("negative_index", "index is out of range"),
])
def test_python_input_checks(gpu, bad, match):
    types, edges, vectors = np.array([0, 1]), np.array([[0], [1]]), np.array([[1., 0, 0]])
    if bad == "shape":
        vectors = vectors.T
    elif bad == "float_indices":
        edges = edges.astype(float)
    elif bad == "complex_vectors":
        vectors = vectors.astype(complex)
    else:
        edges[0, 0] = -1
    with pytest.raises(ValueError, match=match):
        gpu.compute_geometry(types, edges, vectors)


def test_unknown_spline_and_no_implicit_energy(gpu):
    gpu.compute_geometry(np.array([0, 1]), np.array([[0], [1]]), np.array([[1., 0, 0]]))
    gpu.evaluate_splines("radial_0")
    with pytest.raises(ValueError, match="unknown radial spline"):
        gpu.evaluate_splines("not.a.spline")
    assert gpu.has_geometry and not gpu.has_spline_results
    with pytest.raises(RuntimeError, match="evaluate splines first"):
        gpu.download_splines()
    assert gpu.can_evaluate
    assert gpu.can_evaluate_energy and hasattr(gpu, "compute_energy_forces")
