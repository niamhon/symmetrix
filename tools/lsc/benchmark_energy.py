#!/usr/bin/env python3
"""Bounded CPU float64 energy-only timing; excludes loading and neighbor building."""

import json
import os
import platform
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mace import data
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import AtomicNumberTable

import symmetrix


def median_time(function, repeats=5):
    function()  # warm up each implementation separately; first-call setup excluded
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def pytorch_energy(model, batch):
    with torch.no_grad():
        return float(
            model(batch, compute_force=False, compute_stress=False)["energy"].item()
        )


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "symmetrix/test"))
    from lsc.reference import DATA, DEFAULT_MODEL, load_atoms, make_calculator

    native = symmetrix.MACELSC(
        str(repo / ".lsc-reference-logs/water-stage2-energy.json")
    )
    report = {
        "host": platform.node(),
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "dtype": "float64",
        "scope": "warm energy-only inference; prebuilt graph; no force/stress, model load or neighbor construction",
        "torch_compile": False,
        "cases": {},
    }
    for case, mode in [("dimer", "realspace"), ("bulk", "mixed_periodic")]:
        model = make_calculator(DEFAULT_MODEL, mode).model
        atoms = load_atoms(case)
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
            external_field=torch.zeros(1, 3),
            charges=torch.zeros(len(atoms)),
        )
        batch["pbc"] = torch.tensor(atoms.pbc).reshape(1, 3)
        batch["head"] = batch["head"].reshape(1)
        vectors, _ = get_edge_vectors_and_lengths(
            graph.positions, graph.edge_index, graph.shifts
        )
        types = np.asarray(
            [native.atomic_numbers.index(int(z)) for z in atoms.numbers], dtype=np.int64
        )
        edges, xyz, positions = (
            graph.edge_index.numpy(),
            vectors.detach().numpy(),
            atoms.positions.copy(),
        )
        cell, pbc = atoms.cell.array.copy(), atoms.pbc.copy()

        run_native = partial(
            native.compute_energy, types, edges, xyz, positions, cell, pbc, mode
        )
        run_torch = partial(pytorch_energy, model, batch)

        native_e, torch_e = run_native(), run_torch()
        with np.load(DATA / f"{case}.reference.npz", allow_pickle=False) as reference:
            assert abs(torch_e - float(reference["energy"])) < 1e-9
        assert abs(native_e - torch_e) < 2e-6
        native_times, torch_times = median_time(run_native), median_time(run_torch)
        ratio = torch_times["median_seconds"] / native_times["median_seconds"]
        report["cases"][case] = {
            "atoms": len(atoms),
            "edges": edges.shape[1],
            "native": native_times,
            "pytorch": torch_times,
            "pytorch_over_native": ratio,
            "energy_abs_eV": abs(native_e - torch_e),
        }
        print(
            f"ENERGY_TIMING {case} native_ms={native_times['median_seconds'] * 1000:.3f} pytorch_ms={torch_times['median_seconds'] * 1000:.3f} ratio={ratio:.3f}",
            flush=True,
        )
    (repo / ".lsc-reference-logs/native-energy-benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
