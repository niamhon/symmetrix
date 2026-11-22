#!/usr/bin/env python3
"""Synchronized complete ASE calls; native FP64 predictions are saved first."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ase.calculators.calculator import all_changes
from ase.io import read


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prediction(calc, atoms, native=False):
    props = ["energy", "forces", "stress"]
    calc.calculate(atoms, properties=props if native else props + ["charges", "dipole"],
                   system_changes=all_changes)
    keys = dict(energy="energy", forces="forces", stress="stress", dipole="dipole",
                charges="partial_charges" if native else "charges")
    return {k: np.asarray(calc.results[v]).copy() for k, v in keys.items()}


def errors(actual, expected, n):
    out = dict(energy_eV_atom=abs(float(actual["energy"])-float(expected["energy"]))/n)
    passed = out["energy_eV_atom"] < 1e-6
    for k, atol, rtol in [("forces",8e-5,1e-4),("stress",5e-7,1e-4),
                          ("charges",2e-6,1e-5),("dipole",3e-5,0)]:
        out[k+"_max"] = float(np.max(abs(actual[k]-expected[k])))
        passed &= bool(np.allclose(actual[k],expected[k],atol=atol,rtol=rtol))
    out["passed"] = bool(passed)
    return out


def measure(calc, frames, torch, warm, blocks, calls):
    props = ["energy", "forces", "stress"]
    for i in range(warm):
        calc.calculate(frames[i % len(frames)], properties=props, system_changes=all_changes)
    torch.cuda.synchronize()
    raw = []
    for block in range(blocks):
        sample = []
        for i in range(calls):
            a = frames[(i+block) % len(frames)]
            torch.cuda.synchronize(); start = time.perf_counter()
            calc.calculate(a, properties=props, system_changes=all_changes)
            torch.cuda.synchronize()
            sample.append((time.perf_counter()-start)*1000)
        raw.append(sample)
    return dict(ms_per_call=float(np.median(raw)), block_medians_ms=np.median(raw,axis=1).tolist(),
                p10_p90_ms=np.percentile(raw,[10,90]).tolist(), samples_ms=raw)


def neighbors(calc, a, blocks, calls):
    from ase.neighborlist import neighbor_list
    routines = {
        "ase": lambda: neighbor_list("ijD", a, calc.evaluator.r_cut, self_interaction=False),
        "compiled": lambda: calc._neighbor_list(a.positions,a.cell.array,a.pbc,calc.evaluator.r_cut)}
    result = {}
    for name, fn in routines.items():
        fn(); values=[]
        for _ in range(blocks*calls):
            t=time.perf_counter(); fn(); values.append((time.perf_counter()-t)*1000)
        result[name] = dict(ms_per_call=float(np.median(values)),samples_ms=values)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export",type=Path,required=True)
    parser.add_argument("--native-model",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sizes",type=int,nargs="+",default=[192,1536,10368])
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--blocks",type=int,default=3)
    parser.add_argument("--calls",type=int,default=20)
    parser.add_argument("--resume",action="store_true",help="Preserve and reuse completed timing rows")
    parser.add_argument("--tiled-reference",action="store_true",
                        help="After a recorded full-native OOM, derive the large reference from native 192-atom repeats")
    parser.add_argument("--native-oom-log",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    if args.tiled_reference:
        assert args.resume and args.native_oom_log is not None
        assert 'out of memory' in args.native_oom_log.read_text().lower()
    import torch
    import mace_scf
    import symmetrix
    from mace_scf.calculators.localsources import MACELocalSplitCharges
    from symmetrix.experimental import MACELSC
    from symmetrix.lsc_calculator import LSCCalculator
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    assert "H100" in torch.cuda.get_device_name()
    root=dict(model_sha256=sha(args.native_model),export_sha256=sha(args.export),
              native_source=mace_scf.__file__,symmetrix_source=symmetrix.__file__,
              gpu=torch.cuda.get_device_name(),torch=torch.__version__,
              warmup=args.warmup,blocks=args.blocks,calls_per_block=args.calls,
              requested_properties=["energy","forces","stress"],
              scope="Complete ASE calculate, including neighbor construction and transfers; "
                    "forced cache misses; initialization/warmup excluded; device synchronized per call.",
              native_dtype="float64",native_compile=False,mesh_spacing=.7,mesh_order=8,
              rows=[],neighbors=[],completed=False)
    if args.resume:
        previous=json.loads((args.output/"results.json").read_text())
        assert previous['model_sha256']==root['model_sha256'] and previous['export_sha256']==root['export_sha256']
        assert not previous['completed']
        if not (args.output/'results-before-resume.json').exists():
            save(args.output/'results-before-resume.json',previous)
        root=previous
    save(args.output/"results.json",root)
    for n in args.sizes:
        if len([r for r in root['rows'] if r['atoms']==n and 'ms_per_call' in r])==9:
            print('PRESERVED_COMPLETED_SIZE',n,flush=True)
            continue
        data=args.data_dir/f"water-{n}.data"
        a=read(data,format="lammps-data",atom_style="charge",Z_of_type={1:1,2:8},units="metal")
        assert len(a)==n
        frames=[a.copy()]
        rng=np.random.default_rng(20261121+n)
        for _ in range(4):
            b=a.copy();b.positions+=rng.normal(0,.001,b.positions.shape);frames.append(b)
        tiled=args.tiled_reference and n>1536
        if tiled:
            base=read(args.data_dir/'water-192.data',format='lammps-data',atom_style='charge',
                      Z_of_type={1:1,2:8},units='metal')
            transform=a.cell.array @ np.linalg.inv(base.cell.array)
            repeats=np.rint(np.diag(transform)).astype(int)
            np.testing.assert_allclose(transform,np.diag(repeats),atol=1e-12,rtol=0)
            repeated=base.repeat(repeats)
            np.testing.assert_allclose(repeated.positions,a.positions,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(repeated.numbers,a.numbers)
            baseframes=[base.copy()]
            for _ in range(4):
                b=base.copy();b.positions+=rng.normal(0,.001,b.positions.shape);baseframes.append(b)
            frames=[b.repeat(repeats) for b in baseframes]
            save(args.output/f'native-{n}-oom.json',dict(atoms=n,status='oom',
                 log=str(args.native_oom_log),log_sha256=sha(args.native_oom_log),
                 reference='Fresh native FP64 192-atom frames, periodically repeated',repeats=repeats.tolist()))
        suffix='-tiled' if tiled else ''
        framefile=args.output/f"frames-{n}{suffix}.npz"
        assert not framefile.exists()
        np.savez_compressed(framefile,positions=np.array([b.positions for b in frames]),
                            numbers=a.numbers,cell=a.cell.array,pbc=a.pbc)
        native=MACELocalSplitCharges(str(args.native_model),device="cuda",default_dtype="float64",
                                    pbc_handling="mixed_periodic",use_compile=False)
        native.model.eval()
        for parameter in native.model.parameters():
            assert parameter.dtype==torch.float64
            parameter.requires_grad_(False)
        expected=[prediction(native,b,True) for b in (baseframes if tiled else frames)]
        if tiled:
            np.savez_compressed(args.output/f'native-base192-for-{n}.npz',
                **{f'{i}__{k}':v for i,p in enumerate(expected) for k,v in p.items()})
            multiple=n//192
            assert all(abs(p['charges'].sum())<1e-8 for p in expected)
            expected=[dict(energy=p['energy']*multiple,forces=np.tile(p['forces'],(multiple,1)),
                           stress=p['stress'].copy(),charges=np.tile(p['charges'],multiple),
                           dipole=p['dipole']*multiple) for p in expected]
        np.savez_compressed(args.output/f"native-{n}{suffix}.npz",
            **{f"{i}__{k}":v for i,p in enumerate(expected) for k,v in p.items()})
        row=dict(backend="native",precision="float64",solver="direct",atoms=n,data_sha256=sha(data),
                 status='oom' if tiled else 'passed',reference_method='periodic tiling of native 192' if tiled else 'full native')
        if not tiled:
            row.update(measure(native,frames,torch,args.warmup,args.blocks,args.calls))
        root["rows"].append(row);save(args.output/"results.json",root)
        print("ASE_BENCH",n,"native",row.get('ms_per_call','OOM; reference tiled'),flush=True)
        del native;gc.collect();torch.cuda.empty_cache()
        jobs=[(kind,precision,solver) for solver in ["direct","mesh"]
              for precision in ["float32","float32-mixed"] for kind in ["legacy","fast"]]
        rng.shuffle(jobs)
        for kind,precision,solver in jobs:
            if kind=="legacy":
                calc=LSCCalculator(args.export,device="cuda",precision=precision)
                if solver=="mesh":calc.model.set_mesh_electrostatics(.7,8)
            else:
                calc=MACELSC(args.export,device="cuda",dtype=precision,electrostatics=solver)
            checks=[errors(prediction(calc,b),ref,n) for b,ref in zip(frames,expected)]
            row=dict(backend=kind,precision=precision,solver=solver,atoms=n,accuracy=checks,
                     status='passed' if all(c['passed'] for c in checks) else 'accuracy_failed',
                     reference_method='periodic tiling of native 192' if tiled else 'full native')
            root["rows"].append(row);save(args.output/"results.json",root)
            assert all(c["passed"] for c in checks),row
            row.update(measure(calc,frames,torch,args.warmup,args.blocks,args.calls))
            if kind=="fast":
                row["dispatch"]={key:getattr(calc.evaluator,key) for key in
                    ["batched_reverse_calls","batched_source_forward_calls","batched_source_reverse_calls",
                     "compact_first_forward_calls","compact_first_reverse_calls","tensor_core_gemm_calls"]}
                if solver=="mesh" and precision=="float32":
                    root["neighbors"].append(dict(atoms=n,**neighbors(calc,a,3,5)))
            save(args.output/"results.json",root)
            print("ASE_BENCH",n,kind,precision,solver,row["ms_per_call"],flush=True)
            del calc;gc.collect();torch.cuda.empty_cache()
    root["completed"]=True;save(args.output/"results.json",root)


if __name__=="__main__":
    main()
