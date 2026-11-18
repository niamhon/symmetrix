"""Numerical contracts for the exported local LSC blocks, before native LSC exists."""

import json

import numpy as np
import pytest
import symmetrix
import torch
from symmetrix.extract_mace_lsc import _radial_spline, extract_mace_data_lsc
from symmetrix.extract_mace_polar import _layout

pytestmark = pytest.mark.lsc_reference


@pytest.fixture(scope="module")
def parameters(lsc_model):
    # Exercise only serialized data in the reconstruction helpers below.
    return json.loads(
        json.dumps(extract_mace_data_lsc(lsc_model, [1, 8]), allow_nan=False)
    )


def to_layout(x, irreps):
    return x.detach().numpy()[:, _layout(irreps)[0]]


def apply_linear(data, x):
    parts, offset = [], 0
    for l, (n_in, n_out, weights) in enumerate(
        zip(data["in"], data["out"], data["weights"])
    ):
        block = x[:, offset : offset + (2 * l + 1) * n_in].reshape(
            len(x), 2 * l + 1, n_in
        )
        y = block @ np.asarray(weights).reshape(n_in, n_out)
        if l == 0:
            y += np.asarray(data["bias"])
        parts.append(y.reshape(len(x), -1))
        offset += (2 * l + 1) * n_in
    return np.concatenate(parts, axis=1)


def apply_modulation(weights, x, scalar):
    parts, offset = [], 0
    for l, flat in enumerate(weights):
        channels = len(scalar[0])
        block = x[:, offset : offset + (2 * l + 1) * channels].reshape(
            len(x), -1, channels
        )
        modulation = scalar @ np.asarray(flat).reshape(channels, channels)
        parts.append((block * modulation[:, None, :]).reshape(len(x), -1))
        offset += (2 * l + 1) * channels
    return np.concatenate(parts, axis=1)


def apply_tp(data, x, harmonics, radial):
    out = np.zeros((len(x), sum((2 * l + 1) * n for l, n in enumerate(data["out"]))))
    channels = data["num_channels"]
    for (q, a, b, w), coefficient in zip(data["spec"], data["coefficients"]):
        out[:, q : q + channels] += (
            coefficient
            * x[:, a : a + channels]
            * harmonics[:, b, None]
            * radial[:, w : w + channels]
        )
    return out


def apply_contraction(data, layer, a, species):
    # a is [atom][lm][channel]. The existing Symmetrix recursive product
    # format first constructs B polynomials, then applies species weights M.
    spec = data[f"B{layer}_spec"]
    size = max([a.shape[1]] + [q + 1 for q, _, _ in spec])
    b = np.zeros((len(a), size, a.shape[-1]))
    b[:, : a.shape[1]] = a
    for (q, left, right), coefficient in zip(spec, data[f"B{layer}_coefficients"]):
        b[:, q] += coefficient * b[:, left] * b[:, right]
    out = np.zeros((len(a), 4, a.shape[-1]))
    weights = np.asarray(data[f"M{layer}_weights"])[species]
    for term, (q, source, count) in enumerate(data[f"M{layer}_spec"]):
        out[:, q : q + count] += (
            b[:, source : source + count] * weights[:, term, None, :]
        )
    return out.reshape(len(a), -1)


def oxidation_embedding(data, charges):
    normalized = (charges - data["range"][0]) * data["factor"] - 1
    # MACE's basis is T1..T8, rather than T0..T7.
    basis = np.polynomial.chebyshev.chebvander(normalized, data["num_basis"])[:, 1:]
    return apply_linear(data["linear"], basis)


def radial_values(data, r_min, r_cut, scale, points):
    # Read the exact values/slopes through the existing native spline class.
    splines = [
        symmetrix.CubicSpline(r_min, r_cut, values, derivatives)
        for values, derivatives in zip(
            data["radial_spline_values"], data["radial_spline_derivs"]
        )
    ]
    evaluated = np.asarray(
        [[s.evaluate_deriv(float(r)) for s in splines] for r in points]
    )
    weights = np.asarray(data["radial_weights"])
    return evaluated[:, :, 0] @ weights / scale, evaluated[:, :, 1] @ weights / scale


def interaction_radial(data, layer):
    return {
        "radial_spline_values": data[f"radial_spline_values_{layer}"][0],
        "radial_spline_derivs": data[f"radial_spline_derivs_{layer}"][0],
        "radial_weights": data[f"radial_weights_{layer}"],
    }


def test_species_and_oxidation_embeddings(lsc_model, parameters):
    model, data = lsc_model, parameters
    attrs = torch.eye(2, dtype=torch.float64)
    with torch.no_grad():
        expected = model.node_embedding(attrs).numpy()
    np.testing.assert_allclose(data["node_embedding"], expected, atol=2e-14, rtol=2e-14)
    charges = np.asarray([-6, -2, 0, 1, 6], dtype=float)
    with torch.no_grad():
        expected = model.oxidation_state_mixer.ox_embedding(
            torch.tensor(charges)
        ).numpy()
    np.testing.assert_allclose(
        oxidation_embedding(data["oxidation_state_mixer"]["embedding"], charges),
        expected,
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_interaction_linear_maps(lsc_model, parameters, layer):
    interaction = lsc_model.interactions[layer]
    rng = np.random.default_rng(114 + layer)
    x = torch.tensor(rng.normal(size=(3, interaction.linear_up.irreps_in.dim)))
    z = to_layout(x, interaction.linear_up.irreps_in)
    with torch.no_grad():
        expected = to_layout(interaction.linear_up(x), interaction.linear_up.irreps_out)
    np.testing.assert_allclose(
        apply_linear(parameters[f"linear_up_{layer}"], z),
        expected,
        atol=2e-12,
        rtol=2e-12,
    )
    for species in [0, 1]:
        attrs = torch.eye(2, dtype=torch.float64)[[species] * len(x)]
        with torch.no_grad():
            expected = to_layout(
                interaction.skip_tp(x, attrs), interaction.skip_tp.irreps_out
            )
        np.testing.assert_allclose(
            apply_linear(parameters[f"skip_{layer}"][species], z),
            expected,
            atol=2e-12,
            rtol=2e-12,
        )
    x = torch.tensor(rng.normal(size=(3, interaction.linear.irreps_in.dim)))
    with torch.no_grad():
        expected = (
            to_layout(interaction.linear(x), interaction.linear.irreps_out)
            / interaction.avg_num_neighbors
        )
    np.testing.assert_allclose(
        apply_linear(
            parameters[f"linear_{layer}"], to_layout(x, interaction.linear.irreps_in)
        ),
        expected,
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.parametrize(
    "block", ["interaction0", "interaction1", "source0", "source1"]
)
def test_convolution_couplings(lsc_model, parameters, block):
    layer = int(block[-1])
    tp = (
        lsc_model.interactions
        if block.startswith("interaction")
        else lsc_model.lr_source_maps
    )[layer].conv_tp
    data = (
        parameters[f"conv_tp_{layer}"]
        if block.startswith("interaction")
        else parameters["sources"][layer]["conv_tp"]
    )
    rng = np.random.default_rng(421 + layer)
    x = torch.tensor(rng.normal(size=(5, tp.irreps_in1.dim)))
    harmonics = torch.tensor(rng.normal(size=(5, tp.irreps_in2.dim)))
    weights = torch.tensor(rng.normal(size=(5, tp.weight_numel)))
    with torch.no_grad():
        expected = to_layout(tp(x, harmonics, weights), tp.irreps_out)
    actual = apply_tp(
        data, to_layout(x, tp.irreps_in1), harmonics.numpy(), weights.numpy()
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("layer", [0, 1])
def test_products_and_energy_readouts(lsc_model, parameters, layer):
    product = lsc_model.products[layer]
    species = np.asarray([0, 1, 1, 0])
    attrs = torch.eye(2, dtype=torch.float64)[species]
    a = np.random.default_rng(882 + layer).normal(size=(4, 16, 128))
    actual = apply_contraction(parameters, layer, a, species)
    with torch.no_grad():
        expected = to_layout(
            product.symmetric_contractions(torch.tensor(a.transpose(0, 2, 1)), attrs),
            lsc_model.hidden_irreps,
        )
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    actual = apply_linear(parameters[f"product_linear_{layer}"], actual)
    with torch.no_grad():
        original = product(torch.tensor(a.transpose(0, 2, 1)), None, attrs)
        expected = to_layout(original, lsc_model.hidden_irreps)
        energy = lsc_model.readouts[layer].linear(original).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        apply_linear(parameters[f"readout_{layer}"], actual),
        energy,
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_charge_source_from_json(lsc_model, parameters, layer):
    source, data = lsc_model.lr_source_maps[layer], parameters["sources"][layer]
    rng = np.random.default_rng(7341 + layer)
    species = np.asarray([0, 0, 1, 0, 0, 1])
    attrs = torch.eye(2, dtype=torch.float64)[species]
    charges = np.asarray(parameters["formal_charges"])[species]
    x = torch.tensor(rng.normal(size=(6, 512)))
    edge_index = np.asarray([[0, 1, 2, 3, 4, 5, 0, 5], [2, 2, 3, 5, 5, 0, 5, 2]])
    sender, receiver = edge_index
    harmonics = torch.tensor(rng.normal(size=(8, 16)))
    grid = np.linspace(parameters["radial_spline_r_min"], parameters["r_cut"], 1024)
    radii = grid[[71, 120, 243, 330, 580, 680, 734, 900]]
    radial, _ = radial_values(
        data,
        parameters["radial_spline_r_min"],
        parameters["r_cut"],
        parameters["radial_spline_scale"],
        radii,
    )
    ox = oxidation_embedding(data["oxidation_embedding"], charges)
    features = apply_modulation(
        data["oxidation"], to_layout(x, lsc_model.hidden_irreps), ox
    )
    bond = (
        apply_linear(data["linear1"], features)[sender]
        + apply_linear(data["linear2"], features)[receiver]
    )
    message = apply_tp(data["conv_tp"], bond, harmonics.numpy(), radial)
    transfer = (
        apply_linear(data["transfer_readout"], message)[:, 0] * data["transfer_scale"]
    )
    density = apply_linear(data["multipoles"], features)
    assert data["zero_charges"] is True
    density[:, 0] = np.bincount(receiver, weights=transfer, minlength=6) - np.bincount(
        sender, weights=transfer, minlength=6
    )
    with torch.no_grad():
        edge_features, _ = lsc_model.radial_embedding(
            torch.tensor(radii)[:, None],
            attrs,
            torch.tensor(edge_index),
            lsc_model.atomic_numbers,
        )
        expected, expected_transfer = source(
            node_attrs=attrs,
            node_formal_charges=torch.tensor(charges),
            node_feats=x,
            edge_attrs=harmonics,
            edge_feats=edge_features,
            edge_index=torch.tensor(edge_index),
            edge_vectors=torch.zeros(8, 3),
            batch=torch.zeros(6, dtype=torch.int64),
            num_graphs=1,
        )
    np.testing.assert_allclose(
        transfer, expected_transfer.numpy()[:, 0], atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(density, expected.numpy(), atol=2e-11, rtol=2e-11)
    assert abs(density[:, 0].sum()) < 2e-13


@pytest.mark.parametrize(
    "block", ["interaction0", "interaction1", "source0", "source1"]
)
def test_native_spline_values_and_radial_derivatives(
    lsc_model, parameters, block, record_property
):
    layer = int(block[-1])
    interaction = block.startswith("interaction")
    net = (lsc_model.interactions if interaction else lsc_model.lr_source_maps)[
        layer
    ].conv_tp_weights
    data = (
        interaction_radial(parameters, layer)
        if interaction
        else parameters["sources"][layer]
    )
    # Midpoints include the first and final intervals, rather than checking
    # knots alone (where interpolation values agree by construction).
    knots = np.linspace(0.5, 6.0, 1024)
    intervals = np.unique(np.r_[0, 1, np.linspace(2, 1021, 61).astype(int), 1022])
    points = np.r_[0.5, (knots[intervals] + knots[intervals + 1]) / 2, 6.0]
    x = torch.tensor(points)

    def original(r):
        features, _ = lsc_model.radial_embedding(
            r[:, None], None, None, lsc_model.atomic_numbers
        )
        return net(features)

    values, derivatives = torch.autograd.functional.jvp(original, x, torch.ones_like(x))
    actual, actual_derivative = radial_values(
        data, 0.5, 6.0, parameters["radial_spline_scale"], points
    )
    value_error = np.max(np.abs(actual - values.detach().numpy()))
    derivative_error = np.max(np.abs(actual_derivative - derivatives.detach().numpy()))
    record_property("maximum_value_error", float(value_error))
    record_property("maximum_derivative_error", float(derivative_error))
    assert value_error < 1e-7 * max(1.0, float(values.abs().max()))
    assert derivative_error < 1e-4 * max(1.0, float(derivatives.abs().max()))
    np.testing.assert_allclose(actual[-1], 0, atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual_derivative[-1], 0, atol=1e-13, rtol=0)


def test_more_knots_improve_radial_accuracy(lsc_model):
    net = lsc_model.lr_source_maps[0].conv_tp_weights
    points = np.linspace(0.513, 5.93, 71)
    x = torch.tensor(points)

    def original(r):
        return net(
            lsc_model.radial_embedding(
                r[:, None], None, None, lsc_model.atomic_numbers
            )[0]
        )

    expected, expected_derivative = torch.autograd.functional.jvp(
        original, x, torch.ones_like(x)
    )
    errors = []
    for count in [256, 512]:
        data = _radial_spline(lsc_model, net, np.linspace(0.5, 6.0, count))
        values, derivatives = radial_values(data, 0.5, 6.0, 2 * np.sqrt(np.pi), points)
        errors.append(
            (
                np.max(abs(values - expected.detach().numpy())),
                np.max(abs(derivatives - expected_derivative.detach().numpy())),
            )
        )
    assert errors[1][0] < 0.2 * errors[0][0]
    assert errors[1][1] < 0.3 * errors[0][1]


def test_export_preserves_model_and_random_state(lsc_model):
    before = {key: tensor.clone() for key, tensor in lsc_model.state_dict().items()}
    random_state = torch.get_rng_state()
    data = extract_mace_data_lsc(lsc_model, [8], num_spline_points=32)
    assert torch.equal(random_state, torch.get_rng_state())
    assert all(
        torch.equal(before[key], tensor)
        for key, tensor in lsc_model.state_dict().items()
    )
    np.testing.assert_allclose(
        data["node_embedding"][0],
        lsc_model.node_embedding(torch.eye(2, dtype=torch.float64)[[1]])
        .detach()
        .numpy()[0],
    )
    assert len(data["skip_0"]) == len(data["M0_weights"]) == 1


def test_nonfinite_weights_are_rejected(lsc_model, monkeypatch):
    linear = lsc_model.node_embedding.linear
    weight = linear.weight.detach().clone()
    weight[0] = float("nan")
    monkeypatch.setitem(linear._parameters, "weight", torch.nn.Parameter(weight))
    with pytest.raises(ValueError, match="nonfinite learned parameters"):
        extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=32)
