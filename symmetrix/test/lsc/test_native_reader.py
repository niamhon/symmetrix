"""Load the real LSC export through C++; evaluation is a later stage."""

import copy
import json
import re

import numpy as np
import pytest
import symmetrix
from symmetrix.extract_mace_lsc import extract_mace_data_lsc

pytestmark = pytest.mark.lsc_reference


def write_document(path, data):
    path.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
    return str(path)


@pytest.fixture(scope="module")
def native_document(lsc_model, tmp_path_factory):
    # The learned weights are real. Fewer knots keep reader tests small; the
    # existing numerical suite tests spline accuracy separately at 1,024 knots.
    data = extract_mace_data_lsc(lsc_model, [1, 8], num_spline_points=8)
    path = tmp_path_factory.mktemp("native-lsc-reader") / "water-lsc.json"
    write_document(path, data)
    return data, path


def test_native_reader_loads_the_real_export(native_document):
    data, path = native_document
    model = symmetrix.MACELSC(filename=str(path))
    assert type(model).__module__ == "symmetrix.symmetrix"
    for name in (
        "format_version",
        "model_type",
        "export_status",
        "head",
        "num_channels",
        "num_elements",
        "source_num_elements",
        "source_element_indices",
        "atomic_numbers",
        "atomic_energies",
        "formal_charges",
        "feature_layout",
        "r_cut",
        "radial_spline_r_min",
        "radial_spline_scale",
        "kspace_cutoff",
    ):
        assert getattr(model, name) == data[name], name
    assert (model.num_interactions, model.l_max, model.L_max) == (2, 3, 1)
    assert model.can_evaluate is True
    assert not hasattr(model, "compute_node_energies_forces")
    np.testing.assert_array_equal(model.node_embedding, data["node_embedding"])
    np.testing.assert_array_equal(
        model.energy_readout_weights,
        [data[f"readout_{i}"]["weights"][0] for i in range(2)],
    )
    assert model.interaction_radial_shapes == [[64, 8, 512], [64, 8, 1280]]
    assert model.source_radial_shapes == [[32, 8, 256], [32, 8, 256]]
    assert model.oxidation_state_ranges == [[-6, 6], [-4, 4], [-4, 4]]
    assert model.oxidation_state_factors == pytest.approx([1 / 6, 0.25, 0.25])
    assert (
        model.density_smearing_width == data["electrostatics"]["density_smearing_width"]
    )
    assert model.include_self_interaction is True
    assert model.default_pbc_handling == "realspace"


def test_native_reader_loads_a_species_subset(lsc_model, tmp_path):
    data = extract_mace_data_lsc(lsc_model, [8], num_spline_points=8)
    model = symmetrix.MACELSC(write_document(tmp_path / "oxygen.json", data))
    assert model.atomic_numbers == [8]
    assert model.num_elements == 1
    assert model.source_num_elements == 2
    assert model.source_element_indices == [1]
    assert model.formal_charges == [-2]
    np.testing.assert_array_equal(model.node_embedding, data["node_embedding"])
    assert model.atomic_energies == data["atomic_energies"]


def test_native_reader_owns_loaded_arrays(native_document, tmp_path):
    data = copy.deepcopy(native_document[0])
    data["node_embedding"][0][0] = 0.125
    data["readout_1"]["weights"][0][0] = -0.375
    path = tmp_path / "owned.json"
    model = symmetrix.MACELSC(write_document(path, data))
    path.unlink()
    assert model.node_embedding[0][0] == 0.125
    assert model.energy_readout_weights[1][0] == -0.375
    with pytest.raises(AttributeError):
        model.num_channels = 64
    numbers = model.atomic_numbers
    numbers[0] = 6
    assert model.atomic_numbers == [1, 8]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_type",), "PolarMACE", "model_type"),
        (("format_version",), 2, "format_version"),
        (("format_version",), True, "expected an integer"),
        (("format_version",), 2**64 - 1, "integer is out of range"),
        (("export_status",), "architecture_only", "parameters_and_splines"),
        (("export_status",), "complete", "parameters_and_splines"),
        (("num_channels",), 64, "architecture"),
        (("feature_layout",), "e3nn", "feature_layout"),
        (("r_cut",), 0, "r_cut"),
        (("radial_spline_r_min",), 7, "radial_spline_r_min"),
        (("radial_spline_scale",), 1, "radial_spline_scale"),
        (("radial_pair_dependent",), True, "radial_pair_dependent"),
        (
            ("export_options", "num_spline_points"),
            1,
            "export_options.num_spline_points",
        ),
        (("export_options", "r_min"), 0.6, "export_options.r_min"),
        (("source_element_indices",), [0, 0], "source_element_indices"),
        (("source_element_indices",), [0, 2], "source_element_indices"),
        (("atomic_numbers",), [1, 1], "atomic_numbers"),
        (("formal_charges",), [1], "formal_charges"),
        (("node_embedding", 0), [0], "node_embedding"),
        (("architecture", "num_interactions"), 3, "architecture.num_interactions"),
        (
            ("architecture", "layers", 0, "interaction"),
            "Identity",
            "architecture.layers[0].interaction",
        ),
        (
            ("oxidation_state_mixer", "embedding", "range"),
            [6, -6],
            "oxidation_state_mixer.embedding.range",
        ),
        (("linear_up_0", "weights", 0), [0], "linear_up_0.weights"),
        (("skip_1",), [], "skip_1"),
        (("conv_tp_1", "spec", 0), [2_000_000_000, 0, 0, 0], "convolution indices"),
        (("B0_spec", 0), [16, 16, 0], "product dependency indices"),
        (("M0_spec", 0), [0, 2_000_000_000, 1], "undefined product basis value"),
        (("M1_weights", 0, 0), [0], "M1_weights"),
        (("radial_spline_values_0", 0, 0), [0], "radial_spline_0.values"),
        (("radial_spline_derivs_1", 0, 0), [0], "radial_spline_1.derivs"),
        (("radial_weights_1", 0), [0], "radial_spline_1.weights"),
        (("sources", 0, "oxidation"), [], "sources[0].oxidation"),
        (("sources", 0, "zero_charges"), False, "sources[0].zero_charges"),
        (("sources", 1, "transfer_scale"), 0.1, "sources[1].transfer_scale"),
        (
            ("sources", 1, "radial_spline_derivs", 0),
            [0],
            "sources[1].radial_spline.derivs",
        ),
        (
            ("electrostatics", "density_smearing_width"),
            0,
            "electrostatics.density_smearing_width",
        ),
        (("electrostatics", "include_self_interaction"), "true", "expected a boolean"),
        (
            ("electrostatics", "default_pbc_handling"),
            "unknown",
            "unknown boundary mode",
        ),
        (
            ("electrostatics", "density_coefficient_order"),
            ["q", "x", "y", "z"],
            "density_coefficient_order",
        ),
    ],
)
def test_native_reader_rejects_incompatible_data(
    native_document, tmp_path, path, value, message
):
    data = copy.deepcopy(native_document[0])
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    filename = write_document(tmp_path / "invalid.json", data)
    with pytest.raises(ValueError, match=re.escape(message)):
        symmetrix.MACELSC(filename)


@pytest.mark.parametrize(
    "path",
    [
        ("node_embedding",),
        ("readout_1", "bias"),
        ("sources", 0, "conv_tp"),
        ("electrostatics", "include_self_interaction"),
    ],
)
def test_native_reader_reports_missing_fields(native_document, tmp_path, path):
    data = copy.deepcopy(native_document[0])
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(ValueError, match="missing required field"):
        symmetrix.MACELSC(write_document(tmp_path / "missing-field.json", data))


@pytest.mark.parametrize("content", ['{"format_version":', '{"format_version": NaN}'])
def test_native_reader_reports_invalid_json(tmp_path, content):
    path = tmp_path / "invalid-json.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="MACELSC: JSON: cannot parse"):
        symmetrix.MACELSC(str(path))


def test_native_reader_reports_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="MACELSC: cannot open"):
        symmetrix.MACELSC(str(tmp_path / "absent.json"))


@pytest.mark.parametrize("path", [("r_cut",), ("linear_up_0", "weights", 0, 0)])
def test_native_reader_rejects_nonfinite_values(native_document, tmp_path, path):
    data = copy.deepcopy(native_document[0])
    parent = data
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = "__lsc_overflow_test__"
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"))
    assert payload.count('"__lsc_overflow_test__"') == 1
    destination = tmp_path / "overflow.json"
    destination.write_text(payload.replace('"__lsc_overflow_test__"', "1e400"))
    with pytest.raises(ValueError) as error:
        symmetrix.MACELSC(str(destination))
    # JSON libraries may reject overflow while parsing. With fast-math builds
    # it can reach the reader, which must reject it explicitly in either field.
    message = str(error.value)
    assert "MACELSC:" in message
    assert "expected a finite number" in message or "JSON: cannot parse" in message
    if "expected a finite number" in message:
        assert path[0] in message
