"""Opt-in ASE checks on the frozen benzene/EMC pilot; never regenerate references."""
import json
import os

import numpy as np
import pytest

from symmetrix.experimental import MACELSC
from . import organic as o
from .test_cuda_loader import require_cuda

pytestmark = [pytest.mark.lsc_organic,
              pytest.mark.skipif(os.environ.get("LSC_RUN_ORGANIC") != "1",
                                 reason="opt in with LSC_RUN_ORGANIC=1")]


@pytest.fixture(scope="module", autouse=True)
def integrity():
    for name, info in o.manifest()["models"].items():
        for folder, ext, key in [("models", "model", "model_sha256"),
                                 ("models", "json", "export_sha256"),
                                 ("fixtures", "extxyz", "fixtures_sha256"),
                                 ("references", "npz", "reference_sha256")]:
            assert o.sha256(o.root() / folder / f"{name}.{ext}") == info[key]


@pytest.fixture(scope="module", params=[
    pytest.param((name, backend), marks=() if backend == "cpu" else pytest.mark.lsc_cuda,
                 id=f"{name}-{backend}")
    for name in ["benzene", "emc"] for backend in ["cpu", "float32", "float32-mixed"]])
def model_backend(request):
    name, backend = request.param
    if backend != "cpu":
        require_cuda()
    calc = MACELSC(o.root() / "models" / f"{name}.json",
                   device="cpu" if backend == "cpu" else "cuda",
                   dtype="float64" if backend == "cpu" else backend)
    return name, backend, calc


def test_frozen_organic_snapshots(model_backend, tmp_path, record_property):
    name, backend, calc = model_backend
    cases = o.manifest()["models"][name]["cases"]
    if backend == "cpu":
        cases = ([next(c for c in cases if c["group"] == g) for g in ["cluster", "periodic"]]
                 + [c for c in cases if c["group"] == "isolated"])
    rows = []
    for case in cases:
        a = o.load_case(name, case["id"]); a.calc = calc
        props = ["energy", "forces", "charges", "dipole", "energies"]
        if a.pbc.all():
            props.append("stress")
        calc.calculate(a, props)
        actual = {k: np.asarray(v).copy() for k, v in calc.results.items()}
        metrics = o.compare(actual, o.reference(name, case["id"]), a)
        metrics.update(case_id=case["id"], group=case["group"], atoms=len(a))
        metrics["partition_error_eV_atom"] = abs(actual["energies"].sum() - actual["energy"]) / len(a)
        metrics["passed"] &= bool(metrics["partition_error_eV_atom"] < 1e-8)
        rows.append(metrics)
        np.savez_compressed(tmp_path / f"{case['id']:03d}.npz", **actual)
        (tmp_path / "errors.json").write_text(json.dumps(rows, indent=2) + "\n")
    record_property("configurations", len(rows))
    record_property("max_force_error", max(r["forces_max_eV_A"] for r in rows))
    assert all(r["passed"] for r in rows), [r for r in rows if not r["passed"]]
