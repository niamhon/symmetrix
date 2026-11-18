#!/usr/bin/env python3
"""Same-node ASE energy/force/stress timings and repeated-bulk validation."""
import argparse
import gc
import hashlib
import json
import os
import socket
import time
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from ase.calculators.calculator import all_changes
from symmetrix.lsc_calculator import LSCCalculator
from mace_scf.calculators.localsources import MACELocalSplitCharges

p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--repeats",type=int,default=5);args=p.parse_args()
repo=Path(__file__).resolve().parents[2]
data=repo/"symmetrix/test/lsc/data/water_stage2"
checkpoint=repo/".lsc-reference-data/water_stage2.model"
export=repo/".lsc-reference-logs/water-stage2-energy.json"
assert torch.cuda.is_available(),"Run this benchmark in a GPU allocation"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
torch.set_default_dtype(torch.float64)
record={"host":socket.gethostname(),"job_id":os.environ.get("SLURM_JOB_ID"),"device":torch.cuda.get_device_name(),
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"export_sha256":hashlib.sha256(export.read_bytes()).hexdigest(),
        "scope":"Full ASE calculate calls including neighbor construction, energy, forces, stress and result transfer; no torch.compile or cuEquivariance.",
        "rows":[]}

def save():Path(args.output).write_text(json.dumps(record,indent=2)+"\n")

def run(calc,atoms,cuda):
    if cuda:torch.cuda.synchronize()
    start=time.perf_counter()
    calc.calculate(atoms,properties=["energy","forces","stress"],system_changes=all_changes)
    if cuda:torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    result={k:np.array(calc.results[k],copy=True) for k in ["energy","forces","stress"]}
    return elapsed,result

for case in ["dimer","bulk","bulk_2x2x2"]:
    atoms=read(data/("bulk.extxyz" if case.startswith("bulk") else "dimer.extxyz"))
    if case=="bulk_2x2x2":atoms=atoms.repeat((2,2,2))
    mode="realspace" if case=="dimer" else "mixed_periodic"
    original=None
    for name,device,precision in [("pytorch-cpu-float64","cpu","float64"),("pytorch-cuda-float64","cuda","float64"),
                                   ("native-cpu-float64","cpu","float64"),("native-cuda-float32","cuda","float32"),
                                   ("native-cuda-mixed","cuda","float32-mixed")]:
        calc=None
        row={"case":case,"atoms":len(atoms),"backend":name,"device":device,"neural_precision":precision}
        try:
            if name.startswith("pytorch"):
                calc=MACELocalSplitCharges(str(checkpoint),device=device,default_dtype="float64",pbc_handling=mode,use_compile=False)
            else:calc=LSCCalculator(export,device=device,precision=precision,pbc_handling=mode)
            for _ in range(2):_,result=run(calc,atoms,device=="cuda")
            count=min(args.repeats,3) if case=="bulk_2x2x2" else args.repeats
            times=[]
            for _ in range(count):
                elapsed,result=run(calc,atoms,device=="cuda");times.append(elapsed)
            if original is None:original=result
            force_error=float(np.max(abs(result["forces"]-original["forces"])))
            energy_per_atom=abs(float(result["energy"]-original["energy"]))/len(atoms)
            stress_error=float(np.max(abs(result["stress"]-original["stress"])))
            np.testing.assert_allclose(result["forces"],original["forces"],atol=8e-5,rtol=1e-4)
            assert energy_per_atom<1e-6,(name,case,energy_per_atom)
            np.testing.assert_allclose(result["stress"],original["stress"],atol=5e-7,rtol=1e-4)
            if case=="bulk_2x2x2":
                with np.load(data/"bulk.reference.npz",allow_pickle=False) as frozen:
                    np.testing.assert_allclose(result["forces"],np.tile(frozen["forces"],(8,1)),atol=8e-5,rtol=1e-4)
                    assert abs(float(result["energy"])-8*float(frozen["energy"]))/len(atoms)<1e-6
                    np.testing.assert_allclose(result["stress"],frozen["stress"],atol=5e-7,rtol=1e-4)
                if name=="native-cuda-mixed":
                    assert calc.model.tensor_core_gemm_calls>0,"Repeated bulk must exercise mixed GEMMs"
            row.update(status="passed",median_seconds=float(np.median(times)),minimum_seconds=float(min(times)),repetitions=count,
                       force_max_abs_eV_A=force_error,energy_abs_eV_per_atom=energy_per_atom,stress_max_abs_eV_A3=stress_error)
            if name.startswith("native-cuda"):
                row.update(sgemm_calls=calc.model.sgemm_calls,tensor_core_gemm_calls=calc.model.tensor_core_gemm_calls,
                           native_owned_device_bytes=calc.model.device_parameter_bytes+calc.model.workspace_bytes)
            print("BENCHMARK",json.dumps(row),flush=True)
        except torch.OutOfMemoryError as error:
            row.update(status="out_of_memory",error=str(error));print("BENCHMARK_OOM",name,case,flush=True)
        finally:
            record["rows"].append(row);save()
            if calc is not None:del calc
            gc.collect();torch.cuda.empty_cache()
assert all(r.get("status")=="passed" for r in record["rows"] if r["backend"].startswith("native"))
