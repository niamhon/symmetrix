"""Contracts for LSC dispatch, architecture checks, and the partial JSON format."""

import importlib
import json
import sys

import numpy as np
import pytest
import torch
from symmetrix.extract_mace_lsc import (
    extract_mace_data_lsc,
    validate_lsc_architecture,
)

from .reference import load_manifest

# The production LSC module above initializes legacy e3nn constants inside its
# scoped safe_globals context before this lookup. Test that import path too.
Irreps = importlib.import_module("e3nn.o3").Irreps

pytestmark = pytest.mark.lsc_reference
exporter = importlib.import_module("symmetrix.extract_mace_data")


def test_dispatch_selects_lsc(lsc_checkpoint):
    data = exporter.extract_mace_data(lsc_checkpoint, ["O", "H"])
    assert data["model_type"] == "LocalSplitCharges"
    assert data["atomic_numbers"] == [1, 8]
    assert data["export_status"] == "parameters_and_splines"


def test_dispatch_precedes_generic_head_conversion(lsc_model, monkeypatch):
    monkeypatch.setattr(lsc_model, "heads", ["Default", "other"])
    monkeypatch.setattr(exporter.torch, "load", lambda *args, **kwargs: lsc_model)

    def forbidden_conversion(*args, **kwargs):
        pytest.fail("LSC was passed through generic MACE head conversion")

    monkeypatch.setattr(exporter, "remove_pt_head", forbidden_conversion)
    with pytest.raises(NotImplementedError, match="single-head"):
        exporter.extract_mace_data("unused.model", [1, 8])


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("one_layer", "exactly two"),
        ("hidden_channels", "hidden irreps"),
        ("interaction", "interaction must be"),
        ("scalar_final_features", "preserve scalar and vector"),
        ("energy_readout", "energy readout must be"),
        ("charge_source", "charge source must be"),
        ("transfer_readout", "transfer readout must map"),
        ("formal_charges", "PerSpeciesFormalChargesBlock"),
        ("oxidation_mixer", "SumOxidationStateMixer"),
        ("higher_density", "charge and dipole"),
        ("correlation", "correlation order 3"),
        ("polarizability", "polarizability"),
        ("cutoff_placement", "before the radial networks"),
        ("distance_transform", "distance-transformed"),
    ],
)
def test_architecture_rejects_unsupported_variants(
    lsc_model, monkeypatch, variant, message
):
    model = lsc_model
    if variant == "one_layer":
        monkeypatch.setitem(
            model._modules, "interactions", torch.nn.ModuleList([model.interactions[0]])
        )
    elif variant == "hidden_channels":
        monkeypatch.setattr(model, "hidden_irreps", Irreps("64x0e+64x1o"))
    elif variant == "interaction":
        monkeypatch.setitem(
            model._modules,
            "interactions",
            torch.nn.ModuleList([torch.nn.Identity(), model.interactions[1]]),
        )
    elif variant == "scalar_final_features":
        monkeypatch.setattr(model.products[1].linear, "irreps_out", Irreps("128x0e"))
    elif variant == "energy_readout":
        monkeypatch.setitem(
            model._modules,
            "readouts",
            torch.nn.ModuleList([model.readouts[0], torch.nn.Identity()]),
        )
    elif variant == "charge_source":
        monkeypatch.setitem(
            model._modules,
            "lr_source_maps",
            torch.nn.ModuleList([torch.nn.Identity(), model.lr_source_maps[1]]),
        )
    elif variant == "transfer_readout":
        monkeypatch.setattr(model.lr_source_maps[0].linear, "irreps_out", Irreps("1o"))
    elif variant == "formal_charges":
        from mace_scf.electrostatics.bonded_blocks import PerAtomFormalChargesBlock

        monkeypatch.setitem(
            model._modules, "formal_charges", PerAtomFormalChargesBlock()
        )
    elif variant == "oxidation_mixer":
        monkeypatch.setitem(
            model._modules, "oxidation_state_mixer", torch.nn.Identity()
        )
    elif variant == "higher_density":
        monkeypatch.setattr(model, "charges_irreps", Irreps.spherical_harmonics(2))
    elif variant == "correlation":
        monkeypatch.setattr(
            model.products[0].symmetric_contractions.contractions[0], "correlation", 2
        )
    elif variant == "cutoff_placement":
        monkeypatch.setattr(model.radial_embedding, "apply_cutoff", False)
    elif variant == "distance_transform":
        monkeypatch.setitem(
            model.radial_embedding._modules, "distance_transform", torch.nn.Identity()
        )
    elif variant == "polarizability":
        monkeypatch.setitem(
            model._modules,
            "polarizability_readouts",
            torch.nn.ModuleList([torch.nn.Identity()]),
        )
    with pytest.raises(NotImplementedError, match=message):
        validate_lsc_architecture(model)


def test_architecture_records_the_frozen_model(lsc_model):
    data = validate_lsc_architecture(lsc_model)
    expected = load_manifest()["model"]["architecture"]
    assert data["architecture"]["hidden_irreps"] == expected["hidden_irreps"]
    assert [
        layer["interaction"] for layer in data["architecture"]["layers"]
    ] == expected["interactions"]
    assert [
        layer["energy_readout"] for layer in data["architecture"]["layers"]
    ] == expected["readouts"]
    assert [source["type"] for source in data["sources"]] == expected["source_maps"]
    assert (
        data["electrostatics"]["density_smearing_width"]
        == expected["density_smearing_width_A"]
    )
    assert data["oxidation_state_mixer"]["embedding"]["range"] == [-6.0, 6.0]
    assert data["oxidation_state_mixer"]["embedding"]["factor"] == pytest.approx(1 / 6)
    for source in data["sources"]:
        assert source["oxidation_embedding"]["range"] == [-4.0, 4.0]
        assert source["oxidation_embedding"]["factor"] == 0.25
        assert source["transfer_scale"] == 1 / 40


def test_architecture_rejects_invalid_saved_constants(lsc_model, monkeypatch):
    monkeypatch.setitem(lsc_model._buffers, "r_max", torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="cutoff"):
        validate_lsc_architecture(lsc_model)


def test_export_has_polars_common_header(lsc_model):
    data = extract_mace_data_lsc(lsc_model, [8, 1])
    assert {
        "format_version",
        "model_type",
        "num_channels",
        "r_cut",
        "l_max",
        "L_max",
        "atomic_numbers",
        "num_elements",
        "atomic_energies",
    } <= data.keys()
    assert data["format_version"] == 3
    assert data["num_channels"] == 128
    assert (data["r_cut"], data["l_max"], data["L_max"]) == (6.0, 3, 1)
    assert data["formal_charges"] == [1.0, -2.0]
    assert data["head"] == "Default"
    assert data["export_status"] == "parameters_and_splines"
    assert "radial_spline_values_0" in data
    assert "product_linear_0" in data
    np.testing.assert_array_equal(
        data["atomic_energies"], lsc_model.atomic_energies_fn.atomic_energies.numpy()
    )
    assert json.loads(json.dumps(data, allow_nan=False)) == data


def test_species_selection_keeps_the_source_index(lsc_model):
    data = extract_mace_data_lsc(lsc_model, [8])
    assert data["atomic_numbers"] == [8]
    assert data["num_elements"] == 1
    assert data["source_num_elements"] == 2
    assert data["source_element_indices"] == [1]
    assert data["formal_charges"] == [-2.0]
    assert data["atomic_energies"] == [
        float(lsc_model.atomic_energies_fn.atomic_energies[1])
    ]


@pytest.mark.parametrize(
    ("species", "message"),
    [([1, 1], "duplicate"), ([6], "absent"), ([1.5], "integers")],
)
def test_species_selection_rejects_invalid_values(lsc_model, species, message):
    with pytest.raises(ValueError, match=message):
        extract_mace_data_lsc(lsc_model, species)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"head": "mbpol"}, "Unknown LSC head"),
        ({"compression_margin": 0.5}, "compress-basis"),
        ({"num_spline_points": 1}, "at least two"),
        ({"r_min": 0}, "r_min"),
        ({"r_min": 6.0}, "r_min"),
    ],
)
def test_export_rejects_unsupported_options(lsc_model, options, message):
    with pytest.raises(ValueError, match=message):
        extract_mace_data_lsc(lsc_model, [1, 8], **options)


def test_existing_polar_dispatch_is_unchanged(monkeypatch):
    polar_module = importlib.import_module("symmetrix.extract_mace_polar")
    model = type(
        "PolarMACE",
        (),
        {
            "heads": ["Default"],
            "to": lambda self, **kwargs: self,
            "eval": lambda self: self,
        },
    )()
    monkeypatch.setattr(exporter.torch, "load", lambda *args, **kwargs: model)
    calls = []
    expected = {"model_type": "PolarMACE"}

    def extract_polar(actual, species, points, r_min):
        calls.append((actual, species, points, r_min))
        return expected

    monkeypatch.setattr(polar_module, "extract_mace_data_polar", extract_polar)
    assert (
        exporter.extract_mace_data(
            "unused.model", [1, 8], num_spline_points=256, r_min=0.6
        )
        is expected
    )
    assert calls == [(model, [1, 8], 256, 0.6)]


def test_standard_mace_still_uses_its_own_checks(monkeypatch):
    model = type(
        "ScaleShiftMACE",
        (),
        {
            "heads": ["Default"],
            "interactions": [object()],
            "to": lambda self, **kwargs: self,
            "eval": lambda self: self,
        },
    )()
    monkeypatch.setattr(exporter.torch, "load", lambda *args, **kwargs: model)
    with pytest.raises(RuntimeError, match="two-layer MACE"):
        exporter.extract_mace_data("unused.model", [1, 8])


def test_cli_writes_parameters_and_splines(
    lsc_checkpoint, tmp_path, monkeypatch, capsys
):
    cli = importlib.import_module("symmetrix.cli.extract_mace")
    destination = tmp_path / "water-lsc-architecture.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetrix_extract_mace",
            "--model",
            str(lsc_checkpoint),
            "--chemical-symbols",
            "O",
            "H",
            "--head",
            "Default",
            "--num-spline-points",
            "256",
            "--r-min",
            "0.6",
            "--output",
            str(destination),
        ],
    )
    cli.main()
    data = json.loads(destination.read_text())
    assert data["model_type"] == "LocalSplitCharges"
    assert data["format_version"] == 3
    assert data["export_status"] == "parameters_and_splines"
    assert data["export_options"] == {"num_spline_points": 256, "r_min": 0.6}
    assert data["atomic_numbers"] == [1, 8]
    assert set(data["extracted_with"]) == {
        "symmetrix",
        "mace-torch",
        "torch",
        "mace-scf",
        "graph-longrange",
        "e3nn",
    }
    assert "LSC PARAMETERS AND SPLINES" in capsys.readouterr().out


def test_cli_keeps_polar_version_metadata_unchanged(tmp_path, monkeypatch):
    cli = importlib.import_module("symmetrix.cli.extract_mace")
    monkeypatch.setattr(
        cli,
        "extract_mace_data",
        lambda *args, **kwargs: {"model_type": "PolarMACE", "format_version": 3},
    )
    destination = tmp_path / "polar.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "symmetrix_extract_mace",
            "--model",
            "unused.model",
            "--output",
            str(destination),
        ],
    )
    cli.main()
    data = json.loads(destination.read_text())
    assert set(data["extracted_with"]) == {"symmetrix", "mace-torch", "torch"}
    assert "export_status" not in data


def test_existing_mace_reader_rejects_the_lsc_type(lsc_model, tmp_path):
    import symmetrix

    destination = tmp_path / "lsc-architecture.json"
    destination.write_text(json.dumps(extract_mace_data_lsc(lsc_model, [1, 8])))
    with pytest.raises(RuntimeError, match="not a MACE model"):
        symmetrix.MACE(str(destination))
