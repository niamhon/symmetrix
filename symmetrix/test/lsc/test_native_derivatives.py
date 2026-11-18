"""Analytic native LSC derivatives versus frozen ASE, autograd and energy differences."""

import copy

import numpy as np
import pytest
import torch
from symmetrix.extract_mace_lsc import extract_mace_data_lsc

from .reference import DATA, load_atoms
from .test_native_energy import load_native
from .test_native_local import graph_for_case

pytestmark = pytest.mark.lsc_reference


@pytest.fixture(scope="module")
def derivative_document(lsc_model):
    return extract_mace_data_lsc(lsc_model, [1, 8])


@pytest.fixture(scope="module")
def derivative_native(derivative_document, tmp_path_factory):
    return load_native(
        derivative_document,
        tmp_path_factory.mktemp("native-derivatives") / "model.json",
    )


@pytest.fixture(scope="module")
def no_self_native(derivative_document, tmp_path_factory):
    document = copy.deepcopy(derivative_document)
    document["electrostatics"]["include_self_interaction"] = False
    return load_native(
        document, tmp_path_factory.mktemp("native-derivatives-no-self") / "model.json"
    )


def graph_inputs(native, model, case):
    numbers, edges, vectors = graph_for_case(model, case)
    atoms = load_atoms(case)
    types = np.asarray(
        [native.atomic_numbers.index(int(z)) for z in numbers], dtype=np.int64
    )
    return [
        types,
        edges,
        vectors,
        atoms.positions.copy(),
        atoms.cell.array.copy(),
        atoms.pbc.copy(),
    ]


def run_derivatives(native, inputs, mode, *, field=(0, 0, 0), stress=False):
    energy = native.compute_energy_forces(*inputs, mode, field, compute_stress=stress)
    assert native.has_energy_results and native.has_force_results
    assert native.has_stress_results is stress
    assert (
        native.can_evaluate
        and native.can_evaluate_forces
        and native.can_evaluate_stress
    )
    n, e = len(inputs[0]), inputs[1].shape[1]
    result = {
        "energy": energy,
        "forces": np.asarray(native.forces).reshape(n, 3),
        "stress": np.asarray(native.stress),
        "virial": np.asarray(native.virial),
    }
    assembled = np.asarray(native.position_forces).reshape(n, 3).copy()
    edge = np.asarray(native.edge_forces).reshape(e, 3)
    np.add.at(assembled, inputs[1][1], edge)
    np.add.at(assembled, inputs[1][0], -edge)
    np.testing.assert_allclose(result["forces"], assembled, atol=2e-13, rtol=0)
    if stress:
        volume = abs(np.linalg.det(inputs[4]))
        np.testing.assert_allclose(
            result["stress"],
            -result["virial"][[0, 1, 2, 5, 4, 3]] / volume,
            atol=2e-15,
            rtol=1e-13,
        )
    return result


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_native_forces_stress_match_frozen_ase(
    derivative_native, lsc_model, case, record_property
):
    inputs = graph_inputs(derivative_native, lsc_model, case)
    result = run_derivatives(
        derivative_native,
        inputs,
        "realspace" if case == "dimer" else "mixed_periodic",
        stress=case == "bulk",
    )
    with np.load(DATA / f"{case}.reference.npz", allow_pickle=False) as expected:
        np.testing.assert_allclose(
            result["forces"], expected["forces"], atol=1e-5, rtol=1e-7
        )
        assert abs(result["energy"] - float(expected["energy"])) < 2e-6
        errors = {
            "energy_abs_eV": abs(result["energy"] - float(expected["energy"])),
            "force_max_abs_eV_A": float(
                np.max(abs(result["forces"] - expected["forces"]))
            ),
            "force_rms_eV_A": float(
                np.sqrt(np.mean((result["forces"] - expected["forces"]) ** 2))
            ),
        }
        if case == "bulk":
            np.testing.assert_allclose(
                result["stress"], expected["stress"], atol=5e-8, rtol=0
            )
            errors["stress_max_abs_eV_A3"] = float(
                np.max(abs(result["stress"] - expected["stress"]))
            )
    errors["net_force_max_abs_eV_A"] = float(np.max(abs(result["forces"].sum(0))))
    assert errors["net_force_max_abs_eV_A"] < 2e-10
    record_property("atoms", len(inputs[0]))
    record_property("edges", inputs[1].shape[1])
    for name, value in errors.items():
        record_property(name, value)
    print(
        f"DERIVATIVE_PARITY case={case} "
        + " ".join(f"{name}={value:.12e}" for name, value in errors.items())
    )


def torch_coulomb_derivatives(native, inputs, mode):
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.kspace import compute_k_vectors_flat

    density = torch.tensor(native.density_coefficients, requires_grad=True).reshape(
        -1, 4
    )
    positions = torch.tensor(inputs[3], requires_grad=True)
    displacement = torch.zeros(3, 3, requires_grad=True)
    deformation = torch.eye(3) + 0.5 * (displacement + displacement.T)
    cell = (torch.tensor(inputs[4]) @ deformation)[None]
    xyz = positions @ deformation
    reciprocal = 2 * torch.pi * torch.linalg.inv(cell.transpose(1, 2))
    k, norm2, kbatch, k0 = compute_k_vectors_flat(
        native.kspace_cutoff, cell, reciprocal
    )
    coulomb = GTOElectrostaticEnergy(
        1,
        native.density_smearing_width,
        native.kspace_cutoff,
        include_self_interaction=native.include_self_interaction,
        pbc_handling=mode,
    ).double()
    energy = coulomb(
        k_vectors=k,
        k_norm2=norm2,
        k_vector_batch=kbatch,
        k0_mask=k0,
        source_feats=density,
        node_positions=xyz,
        batch=torch.zeros(len(positions), dtype=torch.int64),
        volume=torch.linalg.det(cell).abs(),
        pbc=torch.tensor(inputs[5])[None],
    ).sum()
    gradients = torch.autograd.grad(energy, (density, positions, displacement))
    return [gradient.detach().numpy() for gradient in gradients]


@pytest.mark.parametrize("case", ["dimer", "bulk"])
@pytest.mark.parametrize("self_interaction", [True, False])
def test_electrostatic_adjoints_match_autograd(
    derivative_native,
    no_self_native,
    lsc_model,
    case,
    self_interaction,
    record_property,
):
    native = derivative_native if self_interaction else no_self_native
    inputs = graph_inputs(native, lsc_model, case)
    mode = "realspace" if case == "dimer" else "mixed_periodic"
    run_derivatives(native, inputs, mode, stress=case == "bulk")
    density, positions, strain = torch_coulomb_derivatives(native, inputs, mode)
    actual_density = np.asarray(native.density_energy_gradient).reshape(-1, 4)
    actual_position = -np.asarray(native.position_forces).reshape(-1, 3)
    actual_strain = np.asarray(native.electrostatic_strain_derivative).reshape(3, 3)
    for actual, expected in [
        (actual_density, density),
        (actual_position, positions),
        (actual_strain, strain),
    ]:
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)
    record_property(
        "density_adjoint_max_abs", float(np.max(abs(actual_density - density)))
    )
    record_property(
        "position_adjoint_max_abs_eV_A", float(np.max(abs(actual_position - positions)))
    )
    record_property(
        "coulomb_strain_max_abs_eV", float(np.max(abs(actual_strain - strain)))
    )


def variable_energy(native):
    # Atomic reference energies are constant. Omit them for numerically stable
    # differences; they contribute exactly zero to forces and stress.
    return float(
        np.asarray(native.layer_node_energies).sum()
        + native.electrostatic_energy
        + native.field_energy
    )


def safe_step(native, inputs, edge_direction, maximum=1e-5):
    radii = np.linalg.norm(inputs[2], axis=1)
    speed = np.max(np.linalg.norm(edge_direction, axis=1))
    margin = min(
        float(np.min(radii - native.radial_spline_r_min)),
        float(np.min(native.r_cut - radii)),
    )
    step = min(maximum, 0.2 * margin / max(speed, 1e-20))
    assert step > 1e-8, "Finite difference would cross the supported radial domain"
    return step


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_forces_are_native_energy_derivatives(
    derivative_native, lsc_model, case, record_property
):
    native = derivative_native
    inputs = graph_inputs(native, lsc_model, case)
    mode = "realspace" if case == "dimer" else "mixed_periodic"
    original = run_derivatives(native, inputs, mode)
    direction = np.random.default_rng(920).normal(size=inputs[3].shape)
    direction /= np.linalg.norm(direction)
    edge_direction = direction[inputs[1][1]] - direction[inputs[1][0]]
    step = safe_step(native, inputs, edge_direction)
    energies = []
    for sign in [-1, 1]:
        displaced = inputs.copy()
        displaced[2] = inputs[2] + sign * step * edge_direction
        displaced[3] = inputs[3] + sign * step * direction
        native.compute_energy(*displaced, mode)
        energies.append(variable_energy(native))
    finite_difference = (energies[1] - energies[0]) / (2 * step)
    analytic = -float(np.sum(original["forces"] * direction))
    error = abs(finite_difference - analytic)
    assert error < 2e-6
    record_property("force_directional_fd_abs_eV_A", error)
    print(f"FORCE_FD case={case} error_eV_A={error:.12e} step_A={step:.6e}")


@pytest.mark.parametrize(
    ("component", "a", "b"),
    [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 1, 2), (4, 0, 2), (5, 0, 1)],
)
def test_bulk_stress_is_native_strain_derivative(
    derivative_native, lsc_model, component, a, b, record_property
):
    native = derivative_native
    inputs = graph_inputs(native, lsc_model, "bulk")
    original = run_derivatives(native, inputs, "mixed_periodic", stress=True)
    direction = np.zeros((3, 3))
    direction[a, b] = direction[b, a] = 1 if a == b else 0.5
    step = safe_step(native, inputs, inputs[2] @ direction)
    energies = []
    for sign in [-1, 1]:
        transform = np.eye(3) + sign * step * direction
        displaced = inputs.copy()
        displaced[2], displaced[3], displaced[4] = [
            inputs[k] @ transform for k in [2, 3, 4]
        ]
        native.compute_energy(*displaced, "mixed_periodic")
        energies.append(variable_energy(native))
    finite_difference = (energies[1] - energies[0]) / (
        2 * step * abs(np.linalg.det(inputs[4]))
    )
    error = abs(finite_difference - original["stress"][component])
    assert error < 2e-8
    record_property("stress_fd_abs_eV_A3", error)
    print(f"STRESS_FD component={component} error_eV_A3={error:.12e} step={step:.6e}")


def original_field_derivatives(model, atoms, field, mode):
    from graph_longrange.energy import GTOElectrostaticEnergy
    from mace import data
    from mace.tools import AtomicNumberTable

    graph = data.AtomicData.from_config(
        data.config_from_atoms(atoms, head_name=model.heads[0]),
        z_table=AtomicNumberTable(model.atomic_numbers.tolist()),
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    batch = graph.to_dict()
    batch.update(
        batch=torch.zeros(len(atoms), dtype=torch.int64),
        ptr=torch.tensor([0, len(atoms)]),
        head=batch["head"].reshape(1),
        pbc=torch.tensor(atoms.pbc).reshape(1, 3),
        external_field=torch.tensor(field).reshape(1, 3),
        charges=torch.zeros(len(atoms)),
    )
    original = model.coulomb_energy
    model.coulomb_energy = GTOElectrostaticEnergy(
        1,
        original.density_smearing_width,
        original.kspace_cutoff,
        include_self_interaction=original.include_self_interaction,
        pbc_handling=mode,
    ).double()
    try:
        return {
            name: value.detach().numpy().copy()
            for name, value in model(
                batch, compute_force=True, compute_stress=bool(atoms.pbc.all())
            ).items()
            if name in {"energy", "forces", "stress"} and value is not None
        }
    finally:
        model.coulomb_energy = original


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_external_field_derivatives_match_original(derivative_native, lsc_model, case):
    inputs = graph_inputs(derivative_native, lsc_model, case)
    mode = "realspace" if case == "dimer" else "mixed_periodic"
    field = [0.03, -0.02, 0.01]
    actual = run_derivatives(
        derivative_native, inputs, mode, field=field, stress=case == "bulk"
    )
    expected = original_field_derivatives(lsc_model, load_atoms(case), field, mode)
    assert abs(actual["energy"] - float(expected["energy"].item())) < 2e-6
    np.testing.assert_allclose(
        actual["forces"], expected["forces"], atol=1e-5, rtol=1e-7
    )
    if case == "bulk":
        matrix = expected["stress"].reshape(3, 3)
        np.testing.assert_allclose(
            actual["stress"],
            matrix[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]],
            atol=5e-8,
            rtol=0,
        )


def test_translation_atom_and_edge_permutation(derivative_native, lsc_model):
    native = derivative_native
    inputs = graph_inputs(native, lsc_model, "dimer")
    expected = run_derivatives(native, inputs, "realspace")
    shifted = inputs.copy()
    shifted[3] = inputs[3] + [0.3, -0.2, 0.7]
    actual = run_derivatives(native, shifted, "realspace")
    np.testing.assert_allclose(actual["forces"], expected["forces"], atol=2e-10, rtol=0)
    rng = np.random.default_rng(1320)
    permutation = rng.permutation(len(inputs[0]))
    inverse = np.argsort(permutation)
    order = rng.permutation(inputs[1].shape[1])
    reordered = [
        inputs[0][permutation],
        inverse[inputs[1][:, order]],
        inputs[2][order],
        inputs[3][permutation],
        inputs[4],
        inputs[5],
    ]
    actual = run_derivatives(native, reordered, "realspace")
    np.testing.assert_allclose(
        actual["forces"], expected["forces"][permutation], atol=2e-10, rtol=0
    )


def test_bulk_force_and_stress_rotation(derivative_native, lsc_model):
    from ase.stress import voigt_6_to_full_3x3_stress

    inputs = graph_inputs(derivative_native, lsc_model, "bulk")
    expected = run_derivatives(derivative_native, inputs, "mixed_periodic", stress=True)
    rotation, _ = np.linalg.qr(np.random.default_rng(270).normal(size=(3, 3)))
    rotated = inputs.copy()
    for k in [2, 3, 4]:
        rotated[k] = inputs[k] @ rotation
    actual = run_derivatives(derivative_native, rotated, "mixed_periodic", stress=True)
    np.testing.assert_allclose(
        actual["forces"], expected["forces"] @ rotation, atol=2e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        voigt_6_to_full_3x3_stress(actual["stress"]),
        rotation.T @ voigt_6_to_full_3x3_stress(expected["stress"]) @ rotation,
        atol=2e-12,
        rtol=1e-10,
    )


def test_isolated_pair_empty_graph_and_result_clearing(
    derivative_native, derivative_document
):
    from math import erf, exp, pi, sqrt

    native = derivative_native
    inputs = [
        np.asarray([0, 1]),
        np.empty((2, 0), dtype=np.int64),
        np.empty((0, 3)),
        np.asarray([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        np.eye(3) * 30,
        np.zeros(3, dtype=bool),
    ]
    actual = run_derivatives(native, inputs, "realspace", stress=True)
    p = derivative_document["electrostatics"]["energy_parameters"]
    sigma, epsilon = p["realspace"]["sigma"], p["realspace"]["distance_epsilon"]
    r = 7.0
    denominator = r + epsilon
    radial = (
        p["field_constant"]
        / (4 * pi)
        * (-2)
        * (
            exp(-r * r / (4 * sigma * sigma)) / (sigma * sqrt(pi) * denominator)
            - erf(r / (2 * sigma)) / (denominator * denominator)
        )
    )
    np.testing.assert_allclose(
        actual["forces"], [[radial, 0, 0], [-radial, 0, 0]], atol=2e-12, rtol=0
    )
    field = np.asarray([0.02, -0.01, 0.03])
    changed = run_derivatives(native, inputs, "realspace", field=field)
    np.testing.assert_allclose(
        changed["forces"] - actual["forces"],
        -np.asarray([[1], [-2]]) * field,
        atol=2e-12,
        rtol=0,
    )
    assert not native.has_stress_results and native.stress == [0] * 6
    native.compute_energy(*inputs, "realspace")
    assert not native.has_force_results and not native.has_stress_results
    assert native.forces == native.edge_forces == native.position_forces == []
    assert native.virial == native.stress == [0] * 6
    empty = [
        np.empty(0, dtype=np.int64),
        inputs[1],
        inputs[2],
        np.empty((0, 3)),
        inputs[4],
        inputs[5],
    ]
    result = run_derivatives(native, empty, "realspace", stress=True)
    assert result["energy"] == 0 and result["forces"].shape == (0, 3)
    native.compute_charges(*inputs[:4])
    assert not native.has_force_results and not native.has_stress_results


def test_legacy_export_cannot_evaluate_derivatives(derivative_document, tmp_path):
    document = copy.deepcopy(derivative_document)
    del document["electrostatics"]["energy_parameters"]
    native = load_native(document, tmp_path / "legacy.json")
    assert (
        not native.can_evaluate_forces
        and not native.can_evaluate_stress
        and not native.can_evaluate
    )
    with pytest.raises(ValueError, match="re-export"):
        native.compute_energy_forces(
            np.empty(0, dtype=np.int64),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.eye(3),
            np.zeros(3, dtype=bool),
        )


@pytest.mark.parametrize(
    "variant",
    ["zero_stress_cell", "nan_cell", "nan_position", "partial_pbc", "nan_field"],
)
def test_derivative_input_validation(derivative_native, variant):
    inputs = [
        np.asarray([0]),
        np.empty((2, 0), dtype=np.int64),
        np.empty((0, 3)),
        np.zeros((1, 3)),
        np.eye(3) * 10,
        np.zeros(3, dtype=bool),
    ]
    field, mode = [0.0, 0.0, 0.0], "realspace"
    if variant == "zero_stress_cell":
        inputs[4][:] = 0
    elif variant == "nan_cell":
        inputs[4][0, 0] = np.nan
    elif variant == "nan_position":
        inputs[3][0, 0] = np.nan
    elif variant == "partial_pbc":
        inputs[5][0] = True
        mode = "auto"
    elif variant == "nan_field":
        field[0] = np.nan
    with pytest.raises(ValueError, match="MACELSC:"):
        derivative_native.compute_energy_forces(
            *inputs, mode, field, compute_stress=True
        )


def test_more_knots_improve_force_accuracy(derivative_native, lsc_model, tmp_path):
    document = extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=64)
    coarse = load_native(document, tmp_path / "coarse.json")
    inputs = graph_inputs(derivative_native, lsc_model, "dimer")
    fine_result = run_derivatives(derivative_native, inputs, "realspace")
    coarse_result = run_derivatives(coarse, inputs, "realspace")
    with np.load(DATA / "dimer.reference.npz", allow_pickle=False) as expected:
        fine_error = np.max(abs(fine_result["forces"] - expected["forces"]))
        coarse_error = np.max(abs(coarse_result["forces"] - expected["forces"]))
    assert coarse_error > 1e-10
    assert fine_error < 0.1 * coarse_error
