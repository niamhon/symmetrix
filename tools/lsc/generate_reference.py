#!/usr/bin/env python3
"""Freeze two explicitly selected cases with the original MACE-SCF calculator.

Run this using the reference environment. Existing baselines are never replaced.
"""

import argparse
import datetime
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from ase import Atoms
from ase.io import read

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "symmetrix" / "test"))
from lsc.reference import (  # noqa: E402
    DATA,
    DEFAULT_MODEL,
    MODEL_SHA256,
    REFERENCE_VERSIONS,
    evaluate,
    make_calculator,
    sha256,
)

RUN = Path(
    "/mnt/home/noneill/ceph/reactive_water-1/dft2dft/fresh_baselines/"
    "direct-cluster-to-bulk/inter-intra/pbed3/dipole_test/03_train/"
    "energy100_force100/seed1"
)
MODEL_NAME = "direct_c2b_pbed3_energy100_force100_seed1_stage2.model"


def git_head(path):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def write_geometry(path, source):
    """Write round-trip float64 positions, with no DFT labels/calculator attached."""
    atoms = Atoms(
        numbers=source.numbers.copy(),
        positions=source.positions.copy(),
        cell=source.cell.array.copy(),
        pbc=source.pbc.copy(),
    )
    # The reference data loader supplies this same 30 A cell for nonperiodic
    # inputs without a cell. Make that effective input explicit and reproducible.
    if not atoms.pbc.any() and not atoms.cell.array.any():
        atoms.set_cell(np.eye(3) * 30.0)
    molids = source.arrays.get("molID")
    properties = "species:S:1:pos:R:3" + (":molID:I:1" if molids is not None else "")
    lattice = " ".join(format(float(x), ".17g") for x in atoms.cell.array.flat)
    pbc = " ".join("T" if x else "F" for x in atoms.pbc)
    lines = [
        str(len(atoms)),
        f'Lattice="{lattice}" Properties={properties} pbc="{pbc}" '
        'total_charge=0.0 external_field="0 0 0"',
    ]
    for index, (symbol, position) in enumerate(zip(atoms.symbols, atoms.positions)):
        line = symbol + " " + " ".join(format(float(x), ".17g") for x in position)
        if molids is not None:
            line += f" {int(molids[index])}"
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    saved = read(path)
    np.testing.assert_array_equal(saved.positions, source.positions)
    np.testing.assert_array_equal(saved.numbers, source.numbers)
    return saved


def architecture(model):
    contractions = model.products[0].symmetric_contractions.contractions
    return {
        "class": type(model).__name__,
        "heads": list(model.heads),
        "atomic_numbers": model.atomic_numbers.tolist(),
        "num_interactions": len(model.interactions),
        "hidden_irreps": str(model.hidden_irreps),
        "max_ell": model.spherical_harmonics._lmax,
        "correlation": [int(c.correlation) for c in contractions],
        "r_max_A": float(model.r_max),
        "interactions": [type(x).__name__ for x in model.interactions],
        "readouts": [type(x).__name__ for x in model.readouts],
        "source_maps": [type(x).__name__ for x in model.lr_source_maps],
        "oxidation_state_mixer": type(model.oxidation_state_mixer).__name__,
        "formal_charge_block": type(model.formal_charges).__name__,
        "formal_charges": model.formal_charges.formal_charges.tolist(),
        "density_max_l": int(model.coulomb_energy.density_max_l),
        "density_smearing_width_A": float(model.coulomb_energy.density_smearing_width),
        "kspace_cutoff_inverse_A": float(model.kspace_cutoff),
        "include_self_interaction": bool(model.coulomb_energy.include_self_interaction),
        "parameter_dtype": str(next(model.parameters()).dtype),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    parser.add_argument("--model-copy", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite saved references: {args.output_dir}")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    versions = {name: importlib.metadata.version(name) for name in REFERENCE_VERSIONS}
    if versions != REFERENCE_VERSIONS:
        raise RuntimeError(f"Reference dependency versions changed: {versions}")
    model_source = args.run_dir / MODEL_NAME
    if sha256(model_source) != MODEL_SHA256:
        raise RuntimeError("The selected stage-2 checkpoint has changed")
    args.model_copy.parent.mkdir(parents=True, exist_ok=True)
    if args.model_copy.exists():
        if sha256(args.model_copy) != MODEL_SHA256:
            raise RuntimeError("Existing frozen model has a different checksum")
    else:
        shutil.copyfile(model_source, args.model_copy)
        args.model_copy.chmod(0o444)
    assert sha256(args.model_copy) == MODEL_SHA256
    cases = {
        "dimer": {
            "source": args.run_dir / "test.xyz",
            "frame_index": 154,
            "atoms": 6,
            "pbc_handling": "realspace",
            "include_stress": False,
        },
        "bulk": {
            "source": args.run_dir.parent / "periodic_gen2_pbed3_filtered.extxyz",
            "frame_index": 0,
            "atoms": 192,
            "pbc_handling": "mixed_periodic",
            "include_stress": True,
        },
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=args.output_dir.parent) as tmp:
        out = Path(tmp) / "water_stage2"
        out.mkdir()
        manifest = {
            "schema_version": 1,
            "purpose": "Frozen original MACE-SCF predictions; no Symmetrix LSC evaluator exists yet",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": {"source": str(model_source), "sha256": MODEL_SHA256,
                      "default_copy_relative_to_repo": str(DEFAULT_MODEL.relative_to(REPO))},
            "reference": {
                "python": sys.version,
                "python_executable": sys.executable,
                "packages": versions,
                "device": "cpu",
                "dtype": "float64",
                "compile": False,
                "threads": args.threads,
                "symmetrix_base_commit": git_head(REPO),
            },
            "units": {"energy": "eV", "forces": "eV/Angstrom", "dipole": "e*Angstrom",
                      "partial_charges": "e", "partial_dipoles": "e*Angstrom",
                      "stress": "eV/Angstrom^3"},
            "density_coefficient_order": ["q", "dipole_y", "dipole_z", "dipole_x"],
            "stress_voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
            "cases": {},
        }
        for name, spec in cases.items():
            source_atoms = read(spec["source"], index=spec["frame_index"])
            if len(source_atoms) != spec["atoms"]:
                raise RuntimeError(f"Unexpected source frame: {name}")
            geometry = out / f"{name}.extxyz"
            atoms = write_geometry(geometry, source_atoms)
            calculator = make_calculator(args.model_copy, spec["pbc_handling"])
            if name == "dimer":
                manifest["model"]["architecture"] = architecture(calculator.model)
                import mace_scf
                import graph_longrange
                source_root = Path(inspect.getfile(mace_scf)).resolve().parents[1]
                manifest["reference"].update(
                    mace_scf_source=str(source_root),
                    mace_scf_commit=git_head(source_root),
                    graph_longrange_source=str(Path(inspect.getfile(graph_longrange)).parent),
                )
            print(f"Evaluating {name}: {len(atoms)} atoms, {spec['pbc_handling']}", flush=True)
            values = evaluate(atoms, calculator, spec["include_stress"])
            if abs(values["partial_charges"].sum()) > 1e-10:
                raise RuntimeError(f"Charge conservation failed for {name}")
            archive = out / f"{name}.reference.npz"
            np.savez_compressed(archive, **values)
            manifest["cases"][name] = {
                "source": str(spec["source"]),
                "source_file_sha256": sha256(spec["source"]),
                "outputs_sha256": sha256(archive),
                "frame_index": spec["frame_index"],
                "source_label": source_atoms.info.get("label"),
                "num_atoms": len(atoms),
                "pbc": atoms.pbc.tolist(),
                "cell_A": atoms.cell.array.tolist(),
                "cell_note": "Explicit 30 A nonperiodic cell matches reference loader fallback" if name == "dimer" else "Original cell",
                "pbc_handling": spec["pbc_handling"],
                "include_stress": spec["include_stress"],
                "total_formal_charge": 0.0,
                "external_field": [0.0, 0.0, 0.0],
                "geometry_file": geometry.name,
                "geometry_sha256": sha256(geometry),
                "reference_file": archive.name,
                "reference_sha256": sha256(archive),
                "outputs": {key: {"shape": list(value.shape)} for key, value in values.items()},
                "energy_eV": float(values["energy"]),
            }
            print(f"Saved {name}: energy={float(values['energy']):.12f} eV", flush=True)
            del calculator
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(out, args.output_dir)
    print(f"Reference outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
