"""Access the frozen MACE-SCF baseline without requiring an LSC implementation."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).parent / "data" / "water_stage2"
DEFAULT_MODEL = REPO / ".lsc-reference-data" / "water_stage2.model"
MODEL_SHA256 = "47a453e928adaf66efe67f5b4e9436322cbdc77ca211aa610dbd55a650634a79"
REFERENCE_VERSIONS = {
    "mace-torch": "0.3.14",
    "mace-scf": "0.4.4",
    "graph-longrange": "0.4.3",
    "torch": "2.14.0",
    "e3nn": "0.4.4",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "ase": "3.29.0",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path():
    return Path(os.environ.get("LSC_REFERENCE_MODEL", DEFAULT_MODEL)).expanduser()


def load_manifest():
    return json.loads((DATA / "manifest.json").read_text())


def load_atoms(case, data_dir=DATA):
    from ase.io import read

    return read(Path(data_dir) / f"{case}.extxyz", index=0)


def make_calculator(model, pbc_handling):
    from mace_scf.calculators.localsources import MACELocalSplitCharges

    calculator = MACELocalSplitCharges(
        model_path=str(model),
        device="cpu",
        default_dtype="float64",
        pbc_handling=pbc_handling,
        use_compile=False,
    )
    calculator.model.eval()
    return calculator


def evaluate(atoms, calculator, include_stress):
    """Copy ASE outputs; never use dataset labels as reference predictions."""
    atoms = atoms.copy()
    atoms.calc = calculator
    atoms.get_potential_energy()
    names = [
        "energy",
        "forces",
        "dipole",
        "partial_charges",
        "partial_dipoles",
        "density_coefficients",
    ]
    if include_stress:
        names.append("stress")
    result = {
        name: np.array(calculator.results[name], dtype=np.float64, copy=True)
        for name in names
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite reference {name}")
    return result
