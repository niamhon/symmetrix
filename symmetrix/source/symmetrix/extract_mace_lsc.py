"""LSC-MACE parameters and splines, following the PolarMACE JSON conventions.

This stage exports the local neural model. A native LSC evaluator and its
electrostatics implementation are still required before the file is runnable.
"""

import math
from numbers import Integral

import numpy as np
import torch
from scipy.interpolate import CubicSpline

with torch.serialization.safe_globals([slice]):
    from e3nn.o3 import Irreps


def _require(condition, message):
    if not condition:
        raise NotImplementedError(f"Unsupported LocalSplitCharges export: {message}")


def _require_type(module, expected, label):
    actual = type(module).__name__
    _require(actual == expected, f"{label} must be {expected}; found {actual}")


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid LSC {label}: expected a positive finite value")
    return value


def _oxidation_embedding(block):
    bounds = block.oxidation_state_range.detach().cpu()
    if (
        bounds.shape != (2,)
        or not torch.isfinite(bounds).all()
        or bounds[0] >= bounds[1]
    ):
        raise ValueError(
            "Invalid LSC oxidation_state_range: expected two increasing finite values"
        )
    _require_type(block.oxidation_basis, "ChebychevBasis", "oxidation basis")
    linear = block.oxidation_state_linear
    _require(
        Irreps(linear.irreps_in) == Irreps("8x0e"),
        "eight oxidation basis functions are required",
    )
    _require(
        Irreps(linear.irreps_out) == Irreps("128x0e"),
        "oxidation embeddings must have 128 scalar channels",
    )
    # Read each block's buffers. This checkpoint's initial and transfer
    # embeddings have different ranges; global training defaults are unsafe.
    return {
        "basis": "ChebychevBasis",
        "num_basis": 8,
        "range": bounds.tolist(),
        "factor": _positive(block.factor, "oxidation embedding factor"),
    }


def validate_lsc_architecture(model):
    """Validate the first supported LSC family and return descriptive sections.

    Validation uses explicit exceptions, so Python's ``-O`` cannot disable it.
    This is intentionally narrower than all architectures MACE-SCF can train.
    """
    _require_type(model, "LocalSplitCharges", "model")
    _require(len(model.heads) == 1, "only single-head models are supported")
    _require(
        all(
            len(items) == 2
            for items in (
                model.interactions,
                model.products,
                model.readouts,
                model.lr_source_maps,
            )
        ),
        "exactly two interactions, products, energy readouts, and charge-source maps are required",
    )
    _require(int(model.num_interactions) == 2, "num_interactions must be two")
    hidden = Irreps("128x0e+128x1o")
    angular = Irreps.spherical_harmonics(3)
    target = Irreps("128x0e+128x1o+128x2e+128x3o")
    charges = Irreps.spherical_harmonics(1)
    _require(
        model.spherical_harmonics.normalization == "component"
        and model.spherical_harmonics.normalize,
        "component-normalized unit-vector harmonics are required",
    )
    _require(
        Irreps(model.hidden_irreps) == hidden, "hidden irreps must be 128x0e+128x1o"
    )
    _require(
        model.spherical_harmonics._lmax == 3,
        "edge spherical harmonics must have l_max=3",
    )
    _require(
        Irreps(model.sh_irreps) == angular,
        "natural-parity spherical harmonics through l=3 are required",
    )
    _require(
        Irreps(model.charges_irreps) == charges,
        "density irreps must contain charge and dipole components only",
    )
    _require(
        not hasattr(model, "polarizability_readouts"),
        "polarizability readouts are not supported yet",
    )
    _require(
        not hasattr(model, "scale_shift"),
        "a scale_shift block is not supported by this LSC family",
    )

    numbers = model.atomic_numbers.detach().cpu()
    if numbers.ndim != 1 or not numbers.numel():
        raise ValueError("Invalid LSC atomic_numbers: expected a nonempty vector")
    zs = numbers.tolist()
    if any(not isinstance(z, int) or z < 1 or z > 118 for z in zs) or len(
        set(zs)
    ) != len(zs):
        raise ValueError(
            "Invalid LSC atomic_numbers: elements must be unique integers from 1 to 118"
        )
    _require(
        Irreps(model.node_embedding.linear.irreps_in) == Irreps(f"{len(zs)}x0e"),
        "node embedding must match the species count",
    )
    _require(
        Irreps(model.node_embedding.linear.irreps_out) == Irreps("128x0e"),
        "initial node embedding must have 128 scalar channels",
    )

    radial = model.radial_embedding
    _require_type(radial, "RadialEmbeddingBlock", "radial embedding")
    _require_type(radial.bessel_fn, "BesselBasis", "radial basis")
    _require_type(radial.cutoff_fn, "PolynomialCutoff", "radial cutoff")
    _require(radial.out_dim == 8, "eight radial basis functions are required")
    _require(int(radial.cutoff_fn.p) == 5, "the polynomial cutoff must have p=5")
    _require(
        getattr(radial, "apply_cutoff", True),
        "the radial cutoff must be applied before the radial networks",
    )
    _require(
        not hasattr(radial, "distance_transform"),
        "distance-transformed radial embeddings are not supported",
    )

    _require_type(
        model.formal_charges, "PerSpeciesFormalChargesBlock", "formal charges"
    )
    formal = model.formal_charges.formal_charges.detach().cpu()
    if formal.shape != numbers.shape or not torch.isfinite(formal).all():
        raise ValueError(
            "Invalid LSC formal_charges: expected one finite value per species"
        )
    energies = model.atomic_energies_fn.atomic_energies.detach().cpu()
    if energies.numel() != len(zs) or not torch.isfinite(energies).all():
        raise ValueError(
            "Invalid LSC atomic energies: expected one finite energy per species"
        )
    _positive(model.r_max, "cutoff")

    _require_type(
        model.oxidation_state_mixer, "SumOxidationStateMixer", "oxidation-state mixer"
    )
    initial_oxidation = _oxidation_embedding(model.oxidation_state_mixer.ox_embedding)
    layers, sources = [], []
    for index, (interaction, product, readout, source) in enumerate(
        zip(model.interactions, model.products, model.readouts, model.lr_source_maps)
    ):
        label = f"layer {index}"
        _require_type(
            interaction, "RealAgnosticResidualInteractionBlock", f"{label} interaction"
        )
        _require(
            Irreps(interaction.target_irreps) == target,
            f"{label} interaction target irreps are unsupported",
        )
        input_irreps = Irreps("128x0e") if index == 0 else hidden
        _require(
            Irreps(interaction.linear_up.irreps_in) == input_irreps,
            f"{label} input irreps are unsupported",
        )
        _require_type(product, "EquivariantProductBasisBlock", f"{label} product")
        _require(
            Irreps(product.linear.irreps_out) == hidden,
            f"{label} product must preserve scalar and vector features",
        )
        _require(
            product.use_sc and not getattr(product, "use_agnostic_product", False),
            f"{label} requires residual, element-dependent products",
        )
        contractions = product.symmetric_contractions.contractions
        _require(
            len(contractions) == 2 and all(c.correlation == 3 for c in contractions),
            f"{label} products must have correlation order 3",
        )
        _require(
            all(c.weights_max.shape[0] == len(zs) for c in contractions),
            f"{label} product weights must match the species count",
        )
        _require_type(readout, "LinearReadoutBlock", f"{label} energy readout")
        _require(
            Irreps(readout.linear.irreps_in) == hidden
            and Irreps(readout.linear.irreps_out) == Irreps("0e"),
            f"{label} energy readout must map the hidden features to one scalar",
        )
        _require_type(
            source,
            "OxidationDependentSymmetricPredictionSourceBlock",
            f"{label} charge source",
        )
        _require(
            all(
                Irreps(linear.irreps_in) == hidden
                and Irreps(linear.irreps_out) == hidden
                for linear in (source.linear1, source.linear2)
            ),
            f"{label} sender and receiver maps must preserve the hidden irreps",
        )
        _require(
            Irreps(source.oxidation_state_tp.irreps_in1) == hidden
            and Irreps(source.oxidation_state_tp.irreps_in2) == Irreps("128x0e")
            and Irreps(source.oxidation_state_tp.irreps_out) == hidden,
            f"{label} oxidation tensor-product irreps are unsupported",
        )
        _require(
            Irreps(source.conv_tp.irreps_in1) == hidden
            and Irreps(source.conv_tp.irreps_in2) == angular
            and Irreps(source.conv_tp.irreps_out) == Irreps("128x0e"),
            f"{label} transfer tensor-product irreps are unsupported",
        )
        _require(
            source.conv_tp.weight_numel == 256,
            f"{label} transfer requires 256 radial weights",
        )
        _require(
            list(source.conv_tp_weights.hs) == [8, 32, 32, 32, 256],
            f"{label} transfer radial network must be 8/32/32/32/256",
        )
        _require(
            Irreps(source.linear.irreps_in) == Irreps("128x0e")
            and Irreps(source.linear.irreps_out) == Irreps("0e"),
            f"{label} transfer readout must map the scalar channels to one scalar",
        )
        _require(
            Irreps(source.multipole_block.linear.irreps_in) == hidden
            and Irreps(source.multipole_block.linear.irreps_out) == charges
            and source.multipole_block.zero_charges,
            f"{label} requires a charge-free local dipole readout",
        )
        layers.append(
            {
                "interaction": type(interaction).__name__,
                "input_irreps": str(input_irreps),
                "target_irreps": str(target),
                "output_irreps": str(hidden),
                "correlation": 3,
                "avg_num_neighbors": _positive(
                    interaction.avg_num_neighbors, "average neighbor count"
                ),
                "energy_readout": type(readout).__name__,
            }
        )
        sources.append(
            {
                "type": type(source).__name__,
                "oxidation_embedding": _oxidation_embedding(source),
                "radial_mlp": list(source.conv_tp_weights.hs),
                "num_radial_weights": 256,
                "transfer_scale": 1.0 / 40.0,
                "multipole_irreps": str(charges),
            }
        )

    energy = model.coulomb_energy
    _require_type(energy, "GTOElectrostaticEnergy", "electrostatics")
    _require(energy.density_max_l == 1, "electrostatic density_max_l must be one")
    _positive(model.kspace_cutoff, "reciprocal-space cutoff")
    return {
        "architecture": {
            "num_interactions": 2,
            "hidden_irreps": str(hidden),
            "layers": layers,
            "radial_embedding": {
                "basis": "BesselBasis",
                "num_basis": 8,
                "cutoff": "PolynomialCutoff",
                "p": 5,
            },
        },
        "oxidation_state_mixer": {
            "type": "SumOxidationStateMixer",
            "embedding": initial_oxidation,
        },
        "sources": sources,
        "electrostatics": {
            "density_max_l": 1,
            "density_smearing_width": _positive(
                energy.density_smearing_width, "density smearing width"
            ),
            "include_self_interaction": bool(energy.include_self_interaction),
            "default_pbc_handling": energy.pbc_handling,
            "density_coefficient_order": ["q", "dipole_y", "dipole_z", "dipole_x"],
        },
    }


def _uvu_spec(tp):
    """Channelwise TP couplings in Polar's [l][m][channel] feature layout.

    Each term contributes, for every channel u:
      out[q+u] += coefficient * x[a+u] * Y[b] * radial[w+u].
    Probing the original operation includes its e3nn path normalization. Unlike
    phi1_spec, this also handles several paths accumulating into one output,
    as required by the LSC scalar transfer tensor product.
    """
    from .extract_mace_polar import _layout

    p, mul_in, _ = _layout(tp.irreps_in1)
    q, mul_out, _ = _layout(tp.irreps_out)
    in_layout, out_layout = np.argsort(p), np.argsort(q)
    input_offsets = np.cumsum([0] + [n * ir.dim for n, ir in tp.irreps_in1])
    edge_offsets = np.cumsum([0] + [n * ir.dim for n, ir in tp.irreps_in2])
    output_offsets = np.cumsum([0] + [n * ir.dim for n, ir in tp.irreps_out])
    channels = tp.irreps_in1[0].mul
    spec, coefficients, weight_offset = [], [], 0
    for instruction in tp.instructions:
        i, j, k = instruction.i_in1, instruction.i_in2, instruction.i_out
        _require(
            instruction.connection_mode == "uvu"
            and instruction.has_weight
            and tp.irreps_in1[i].mul == tp.irreps_out[k].mul == channels
            and tp.irreps_in2[j].mul == 1
            and tuple(instruction.path_shape) == (channels, 1),
            "convolutions require channelwise uvu paths and one edge harmonic per l",
        )
        d1, d2, d3 = (
            tp.irreps_in1[i].ir.dim,
            tp.irreps_in2[j].ir.dim,
            tp.irreps_out[k].ir.dim,
        )
        m1, m2 = np.meshgrid(np.arange(d1), np.arange(d2), indexing="ij")
        m1, m2 = m1.ravel(), m2.ravel()
        x = torch.zeros(len(m1), len(p), dtype=torch.float64)
        y = torch.zeros(len(m1), tp.irreps_in2.dim, dtype=torch.float64)
        w = torch.zeros(len(m1), tp.weight_numel, dtype=torch.float64)
        x[np.arange(len(m1)), input_offsets[i] + m1] = 1.0
        y[np.arange(len(m1)), edge_offsets[j] + m2] = 1.0
        w[:, weight_offset] = 1.0
        with torch.no_grad():
            values = tp(x, y, w)[:, output_offsets[k] : output_offsets[k] + d3].numpy()
        for sample, m3 in zip(*np.nonzero(np.abs(values) > 1e-12)):
            spec.append(
                [
                    int(out_layout[output_offsets[k] + m3]),
                    int(in_layout[input_offsets[i] + m1[sample]]),
                    int(edge_offsets[j] + m2[sample]),
                    weight_offset,
                ]
            )
            coefficients.append(float(values[sample, m3]))
        weight_offset += channels
    _require(weight_offset == tp.weight_numel, "unsupported convolution weight layout")
    return {
        "num_channels": channels,
        "in": mul_in,
        "out": mul_out,
        "num_weights": tp.weight_numel,
        "spec": spec,
        "coefficients": coefficients,
    }


def _radial_spline(model, net, radii):
    """Spline the hidden radial functions; retain the last effective projection.

    As in Polar, values and derivatives are [function][knot], scaled by
    2*sqrt(pi). The radial MLP's normalized activations are evaluated directly;
    its last linear map is probed to include its normalization as well.
    """
    layers = list(net.children())
    _require_type(net, "FullyConnectedNet", "radial network")
    _require(
        bool(layers) and layers[-1].act is None,
        "radial networks need a linear final layer",
    )
    with torch.no_grad():
        hidden, cutoff = model.radial_embedding(
            torch.tensor(radii, dtype=torch.float64)[:, None],
            None,
            None,
            model.atomic_numbers,
        )
        _require(cutoff is None, "the radial embedding must include its cutoff")
        for layer in layers[:-1]:
            hidden = layer(hidden)
        weights = layers[-1](torch.eye(hidden.shape[1], dtype=torch.float64)).numpy()
        values = 2 * np.sqrt(np.pi) * hidden.numpy()
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("Invalid LSC radial network: nonfinite exported values")
    spline = CubicSpline(radii, values, axis=0, bc_type=("not-a-knot", "clamped"))
    return {
        "radial_spline_values": spline(radii).T.tolist(),
        "radial_spline_derivs": spline(radii, 1).T.tolist(),
        "radial_weights": weights.tolist(),
    }


def _electrostatic_energy_parameters(energy):
    """Save checkpoint constants, including the reference real-space FD kernel."""
    from graph_longrange.utils import FIELD_CONSTANT

    basis = energy.density_basis
    _require_type(basis.radial_spline, "RadialIntegralDirect", "Fourier radial basis")
    _require(
        basis.max_l == 1 and basis.num_sigma == 1 and basis.normalize == "multipoles",
        "one multipole-normalized charge/dipole Gaussian is required",
    )
    _require(
        basis.real_phase_factors.tolist() == [1, 0, 0, 0]
        and basis.imag_phase_factors.tolist() == [0, -1, -1, -1]
        and basis.permute_indices.tolist() == [1, 2, 0],
        "unsupported Fourier phase or component order",
    )
    real = energy.realspace_energy
    _require_type(real, "RealSpaceFiniteDiffereneEnergy", "real-space energy")
    _require(
        real.density_max_l == 1
        and real.include_self_interaction == energy.include_self_interaction,
        "real-space density/self-interaction settings must match",
    )
    for block in (energy.self_interaction_terms, real.self_interaction):
        _require(
            block.select_indices.tolist() == [0, 1, 2, 3],
            "unsupported self-interaction component order",
        )
    radial = basis.radial_spline
    amplitudes = (
        (torch.stack([radial.pref0, radial.pref1], dim=-1) * basis.cl_scale)
        .detach()
        .cpu()
    )
    result = {
        "format_version": 1,
        "field_constant": float(FIELD_CONSTANT),
        "reciprocal": {
            "sigma2": float(radial.sigma2[0]),
            "a": amplitudes[0].tolist(),
            "self_overlap": energy.self_interaction_terms.overlap_constants.detach()
            .cpu()
            .tolist(),
        },
        "realspace": {
            "sigma": float(real.density_smearing_width),
            "offset": float(real.offset),
            "offset_vectors": torch.stack([real.x, real.y, real.z])
            .detach()
            .cpu()
            .tolist(),
            # charges_energy_from_graph uses this denominator regularization.
            "distance_epsilon": 1e-6,
            "self_overlap": real.self_interaction.overlap_constants.detach()
            .cpu()
            .tolist(),
        },
    }
    return result


def _learned_parameters_and_splines(model, output, num_spline_points, r_min):
    from .extract_mace_data import symmetric_contraction
    from .extract_mace_polar import _linear, _modulation

    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Invalid LSC model: nonfinite learned parameters")
    indices = output["source_element_indices"]
    attrs = torch.eye(len(model.atomic_numbers), dtype=torch.float64)[indices]
    hidden = model.hidden_irreps
    radii = np.linspace(r_min, output["r_cut"], num_spline_points)
    with torch.no_grad():
        output["node_embedding"] = model.node_embedding(attrs).tolist()
    output["feature_layout"] = "[l][m][channel]"
    output["radial_spline_r_min"] = float(radii[0])
    output["radial_spline_scale"] = float(2 * np.sqrt(np.pi))
    # These LSC radial functions are species-independent. Keep Polar's outer
    # table dimension, storing one shared table rather than duplicating pairs.
    output["radial_pair_dependent"] = False
    embedding = model.oxidation_state_mixer.ox_embedding
    output["oxidation_state_mixer"]["embedding"]["linear"] = _linear(
        embedding.oxidation_state_linear, "8x0e", "128x0e"
    )
    for t, (interaction, product, readout, source) in enumerate(
        zip(model.interactions, model.products, model.readouts, model.lr_source_maps)
    ):
        radial = _radial_spline(model, interaction.conv_tp_weights, radii)
        output[f"radial_spline_values_{t}"] = [radial["radial_spline_values"]]
        output[f"radial_spline_derivs_{t}"] = [radial["radial_spline_derivs"]]
        output[f"radial_weights_{t}"] = radial["radial_weights"]
        output[f"linear_up_{t}"] = _linear(
            interaction.linear_up,
            interaction.linear_up.irreps_in,
            interaction.linear_up.irreps_out,
        )
        output[f"conv_tp_{t}"] = _uvu_spec(interaction.conv_tp)
        output[f"linear_{t}"] = _linear(
            interaction.linear,
            interaction.linear.irreps_in,
            interaction.linear.irreps_out,
            1 / interaction.avg_num_neighbors,
        )
        output[f"skip_{t}"] = [
            _linear(
                lambda x, a=a, skip=interaction.skip_tp: skip(x, attrs[[a] * len(x)]),
                interaction.skip_tp.irreps_in1,
                hidden,
            )
            for a in range(len(indices))
        ]
        contraction = symmetric_contraction(t, product.symmetric_contractions, indices)
        # The shared helper returns tuple indices; keep the LSC dictionary
        # identical to its JSON round-trip without changing the Polar helper.
        contraction[f"B{t}_spec"] = [list(term) for term in contraction[f"B{t}_spec"]]
        output.update(contraction)
        output[f"product_linear_{t}"] = _linear(product.linear, hidden, hidden)
        output[f"readout_{t}"] = _linear(readout.linear, hidden, "0e")
        exported_source = output["sources"][t]
        exported_source["oxidation_embedding"]["linear"] = _linear(
            source.oxidation_state_linear, "8x0e", "128x0e"
        )
        # Polar's probing helper uses random test inputs. Preserve caller RNG.
        with torch.random.fork_rng(devices=[]):
            exported_source["oxidation"] = _modulation(
                source.oxidation_state_tp, hidden
            )
        for name in ("linear1", "linear2"):
            exported_source[name] = _linear(getattr(source, name), hidden, hidden)
        exported_source["conv_tp"] = _uvu_spec(source.conv_tp)
        exported_source.update(_radial_spline(model, source.conv_tp_weights, radii))
        exported_source["transfer_readout"] = _linear(source.linear, "128x0e", "0e")
        exported_source["multipoles"] = _linear(
            source.multipole_block.linear, hidden, model.charges_irreps
        )
        exported_source["zero_charges"] = bool(source.multipole_block.zero_charges)
    output["electrostatics"]["energy_parameters"] = _electrostatic_energy_parameters(
        model.coulomb_energy
    )
    output["export_status"] = "parameters_and_splines"


def extract_mace_data_lsc(
    model,
    atomic_numbers,
    num_spline_points=1024,
    r_min=0.5,
    head=None,
    compression_margin=None,
):
    """Export the supported LSC local neural model using Polar's conventions."""
    sections = validate_lsc_architecture(model)
    if head is not None and head != model.heads[0]:
        raise ValueError(
            f"Unknown LSC head {head!r}; available head: {model.heads[0]!r}"
        )
    if compression_margin is not None:
        raise ValueError("LSC architecture export does not support --compress-basis")
    if (
        isinstance(num_spline_points, bool)
        or not isinstance(num_spline_points, Integral)
        or num_spline_points < 2
    ):
        raise ValueError("LSC num_spline_points must be an integer of at least two")
    r_cut = float(model.r_max)
    if not math.isfinite(float(r_min)) or not 0 < r_min < r_cut:
        raise ValueError("LSC r_min must be positive and smaller than the model cutoff")
    zs = model.atomic_numbers.tolist()
    selected = list(
        zs if atomic_numbers is None or len(atomic_numbers) == 0 else atomic_numbers
    )
    if any(isinstance(z, bool) or not isinstance(z, Integral) for z in selected):
        raise ValueError("LSC atomic_numbers must contain integers")
    if len(set(selected)) != len(selected):
        raise ValueError("LSC atomic_numbers must not contain duplicate species")
    unknown = sorted(set(selected) - set(zs))
    if unknown:
        raise ValueError(f"LSC species are absent from the checkpoint: {unknown}")
    selected = sorted(int(z) for z in selected)
    indices = [zs.index(z) for z in selected]
    energies = model.atomic_energies_fn.atomic_energies.detach().cpu().reshape(-1)
    formal = model.formal_charges.formal_charges.detach().cpu()
    output = {
        # Match extract_mace_data_polar's common model header and conventions.
        "format_version": 3,
        "model_type": "LocalSplitCharges",
        "num_channels": 128,
        "r_cut": r_cut,
        "l_max": 3,
        "L_max": 1,
        "atomic_numbers": selected,
        "num_elements": len(selected),
        "atomic_energies": energies[indices].tolist(),
        # Updated only after all learned parameters and splines are exported.
        "export_status": "architecture_only",
        "head": model.heads[0],
        "source_num_elements": len(zs),
        "source_element_indices": indices,
        "formal_charges": formal[indices].tolist(),
        "kspace_cutoff": float(model.kspace_cutoff),
        "export_options": {
            "num_spline_points": int(num_spline_points),
            "r_min": float(r_min),
        },
        **sections,
    }
    _learned_parameters_and_splines(model, output, num_spline_points, r_min)
    return output
