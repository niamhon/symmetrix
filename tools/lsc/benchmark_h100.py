#!/usr/bin/env python3
"""Synchronized H100 E/F/stress benchmark: full ASE and prepared host graphs."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time

import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.io import read
from mace_scf.calculators.localsources import MACELocalSplitCharges
from symmetrix.lsc_calculator import LSCCalculator

REPO = Path('/mnt/home/noneill/Programs/symmetrix-dev')
DATA = REPO/'symmetrix/test/lsc/data/water_stage2'
CHECKPOINT = REPO/'.lsc-reference-data/water_stage2.model'
EXPORT = REPO/'.lsc-reference-logs/water-stage2-energy.json'
EXPECTED_MODEL = '47a453e928adaf66efe67f5b4e9436322cbdc77ca211aa610dbd55a650634a79'
EXPECTED_EXPORT = '7bbbe952d1e5b819a2925047943be217181116de6fb5f357e58ab28527b46f65'
KINDS = ['mace-fp64', 'mace-fp32', 'sym-fp32-direct', 'sym-mixed-direct',
         'sym-fp32-mesh', 'sym-mixed-mesh']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_status():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version,utilization.gpu,clocks.sm,power.draw',
                                    '--format=csv,noheader'], text=True).strip()


def copied(result):
    keys = ['energy', 'forces', 'stress', 'dipole']
    out = {k: np.array(result[k], dtype=np.float64, copy=True) for k in keys}
    out['charges'] = np.array(result['charges'] if 'charges' in result else result['partial_charges'], dtype=np.float64, copy=True)
    return out


class Evaluator:
    def __init__(self, kind):
        self.kind = kind
        self.native = kind.startswith('mace-')
        self.dtype = torch.float64 if kind == 'mace-fp64' else torch.float32
        self.configure()
        if self.native:
            self.calc = MACELocalSplitCharges(str(CHECKPOINT), device='cuda',
                default_dtype='float64' if self.dtype == torch.float64 else 'float32',
                pbc_handling='mixed_periodic', use_compile=False)
            self.calc.model.to(dtype=self.dtype).eval()
            for parameter in self.calc.model.parameters():
                parameter.requires_grad_(False)
            actual = {str(p.dtype) for p in self.calc.model.parameters() if p.is_floating_point()}
            assert actual == {str(self.dtype)}, actual
            self.metadata = dict(parameter_dtypes=sorted(actual), frozen_weights=True,
                                 compile=False, cueq_conversion=False)
        else:
            precision = 'float32-mixed' if 'mixed' in kind else 'float32'
            self.calc = LSCCalculator(EXPORT, device='cuda', precision=precision,
                                      pbc_handling='mixed_periodic', external_field=(0., 0., 0.))
            if kind.endswith('mesh'):
                self.calc.model.set_mesh_electrostatics(.7, 8)
            self.metadata = dict(neural_precision=precision, physical_precision='float64',
                                 electrostatics='mesh .7 order 8' if kind.endswith('mesh') else 'direct reciprocal')
        self.host = []

    def configure(self):
        torch.set_default_dtype(self.dtype)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def prepare(self, frames, graph64):
        if self.native:
            self.configure()
            self.host = [next(iter(self.calc._build_data_loader(a))).to_dict() for a in frames]
            assert all(not value.is_cuda for data in self.host for value in data.values() if torch.is_tensor(value))
        else:
            self.host = graph64

    def call(self, frame, scope, frames):
        atoms = frames[frame]
        if scope == 'ase':
            self.calc.calculate(atoms, properties=['energy', 'forces', 'stress'], system_changes=all_changes)
            return self.calc.results
        if self.native:
            # Fresh CUDA tensors each call; the master host graph stays on CPU.
            data = {k: value.to('cuda') if torch.is_tensor(value) else value
                    for k, value in self.host[frame].items()}
            output = self.calc.model(data, compute_force=True, compute_stress=True)
            return self.calc._build_results(atoms, output)
        values = self.host[frame]
        # The topology is prepared, but update geometry inside the timer just
        # as the native model does when it constructs vectors from positions.
        vectors = values[3][values[1][1]]-values[3][values[1][0]]+values[6]
        args = [values[0], values[1], vectors, *values[3:6]]
        self.calc.model.compute_energy_forces(*args, 'mixed_periodic', (0., 0., 0.), compute_stress=True)
        model = self.calc.model
        return dict(energy=model.energy, forces=model.download_forces(False)['forces'],
                    stress=np.asarray(model.stress), dipole=np.asarray(model.dipole),
                    charges=model.download_charges(False)['partial_charges'])


def call_timed(evaluator, frame, scope, frames):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = evaluator.call(frame, scope, frames)
    torch.cuda.synchronize()
    return (time.perf_counter_ns()-start)*1e-9, result


def accuracy(actual, expected, atoms, kind):
    n = len(atoms)
    energy = abs(float(actual['energy'])-float(expected['energy']))
    # FP32 total energies carry a large atomic-energy offset. Declare its
    # representational allowance before timing; retain original force/stress gates.
    ulp = float(abs(np.spacing(np.float32(expected['energy'])))) if kind == 'mace-fp32' else 0.
    energy_limit = n*1e-6 + 4*ulp
    result = dict(energy_abs_eV=energy, energy_abs_eV_per_atom=energy/n,
                  energy_limit_eV=energy_limit, fp32_total_energy_ulp_eV=ulp,
                  force_max_abs_eV_A=float(np.max(abs(actual['forces']-expected['forces']))),
                  stress_max_abs_eV_A3=float(np.max(abs(actual['stress']-expected['stress']))),
                  dipole_max_abs_eA=float(np.max(abs(actual['dipole']-expected['dipole']))),
                  charge_max_abs_e=float(np.max(abs(actual['charges']-expected['charges']))),
                  net_charge_e=float(np.sum(actual['charges'])))
    assert all(np.isfinite(v).all() for v in actual.values()), result
    result['passed'] = bool(energy < energy_limit and
        np.allclose(actual['forces'], expected['forces'], atol=8e-5, rtol=1e-4) and
        np.allclose(actual['stress'], expected['stress'], atol=5e-7, rtol=1e-4))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--calls', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--scopes', nargs='+', choices=['ase','prepared'], default=['ase','prepared'])
    args = parser.parse_args()
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    assert sha(CHECKPOINT) == EXPECTED_MODEL and sha(EXPORT) == EXPECTED_EXPORT
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.set_num_interop_threads(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID'),
        hardware=gpu_status(), python=os.sys.executable,
        versions={p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'ase', 'mace-scf', 'mace-torch']},
        checkpoint_sha256=sha(CHECKPOINT), export_sha256=sha(EXPORT), bulk_sha256=sha(DATA/'bulk.extxyz'),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        caches={k:os.environ.get(k) for k in ['CUDA_CACHE_PATH','PYTORCH_KERNEL_CACHE_PATH','XDG_CACHE_HOME','TMPDIR']},
        source_heads={name:subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
                      for name,path in [('mace_scf',str(REPO.parent/'mace-scf-dev-niamh')),('symmetrix',str(REPO))]},
        settings=vars(args), observables=['energy','forces','stress','dipole','charges'],
        scope_notes=dict(ase='Full calculate(all_changes), neighbor graph construction, input transfer, E/F/stress and host outputs.',
                         prepared='Prebuilt CPU topology/DataLoader graphs; per-call edge-vector update, fresh input transfer, evaluator and host outputs. Native builds vectors inside its forward; Symmetrix refreshes vectors in timed NumPy code. Uses the same native FP64 neighbor topology for Symmetrix.',
                         precision='Native FP32 explicitly casts in-memory weights/buffers. Symmetrix FP32/mixed retain FP64 physical operations. Mixed uses high/low FP16 tensor-core GEMMs, not blanket FP16 or TF32.',
                         compilation='No torch.compile: current native compiled calculator rejects stress. This calculator has no implemented cuEq conversion switch.'),
        accuracy_gates=dict(energy='N*1e-6 eV; native FP32 adds 4 ULPs of its total-energy representation',
                            forces='atol 8e-5 eV/A, rtol 1e-4', stress='atol 5e-7 eV/A^3, rtol 1e-4',
                            dipole_charges='Reported separately, not used to widen force/stress gates'),
        cases=[], rows=[])
    save = lambda: output.write_text(json.dumps(record, indent=2)+'\n')
    save()
    rng = np.random.default_rng(20261118)
    for repeat in [1, 2]:
        base = read(DATA/'bulk.extxyz')
        base.wrap()
        if repeat > 1:
            base = base.repeat((repeat, repeat, repeat))
        base.info['external_field'] = np.zeros(3)
        frames = [base.copy()]
        for i in range(1, args.frames):
            atoms = base.copy()
            atoms.positions += rng.normal(0., .002, size=atoms.positions.shape)
            atoms.wrap()
            frames.append(atoms)
        case = f'bulk_{len(base)}'
        frame_file = output.parent/(case+'-frames.npz')
        if not frame_file.exists():
            np.savez(frame_file, positions=np.asarray([a.positions for a in frames]), cell=base.cell.array,
                     numbers=base.numbers, pbc=base.pbc)
        else:
            with np.load(frame_file) as saved:
                np.testing.assert_array_equal(saved['positions'], [a.positions for a in frames])
        case_record = dict(case=case, atoms=len(base), frame_sha256=sha(frame_file),
                           perturbation_sigma_A=.002, shape=list(base.cell.lengths()), accuracy=[])
        record['cases'].append(case_record)
        evaluators = {kind: Evaluator(kind) for kind in KINDS}
        reference = evaluators['mace-fp64']
        reference.configure()
        cpu_data = [next(iter(reference.calc._build_data_loader(atoms))).to_dict() for atoms in frames]
        graph64 = []
        for atoms, data in zip(frames, cpu_data):
            types = data['node_attrs'].argmax(dim=1).numpy().astype(np.int64)
            edges = data['edge_index'].numpy().astype(np.int64)
            positions = data['positions'].numpy().astype(np.float64)
            vectors = positions[edges[1]]-positions[edges[0]]+data['shifts'].numpy()
            graph64.append([types, edges, vectors, atoms.positions.copy(), atoms.cell.array.copy(), atoms.pbc.copy(), data['shifts'].numpy().copy()])
        for evaluator in evaluators.values():
            evaluator.prepare(frames, graph64)
        expected = []
        reference.configure()
        for i in range(len(frames)):
            _, result = call_timed(reference, i, 'ase', frames)
            expected.append(copied(result))
        with np.load(DATA/'bulk.reference.npz', allow_pickle=False) as frozen:
            mult = repeat**3
            np.testing.assert_allclose(expected[0]['forces'], np.tile(frozen['forces'], (mult, 1)), atol=8e-5, rtol=1e-4)
            assert abs(float(expected[0]['energy'])-mult*float(frozen['energy']))/len(base) < 1e-6
            np.testing.assert_allclose(expected[0]['stress'], frozen['stress'], atol=5e-7, rtol=1e-4)
        np.savez(output.parent/(case+'-reference.npz'), **{f'{i}/{k}':v for i,row in enumerate(expected) for k,v in row.items()})
        for scope in args.scopes:
            rows = {}
            for kind, evaluator in evaluators.items():
                evaluator.configure()
                validations=[]
                for i in range(len(frames)):
                    _, result = call_timed(evaluator, i, scope, frames)
                    validations.append(accuracy(copied(result), expected[i], frames[i], kind))
                valid = all(v['passed'] for v in validations)
                row = dict(case=case, atoms=len(base), backend=kind, scope=scope,
                           metadata=evaluator.metadata, accuracy=validations, accuracy_passed=valid,
                           output_dtypes={k:str(np.asarray(v).dtype) for k,v in result.items() if k in ['energy','forces','stress','dipole']},
                           status='validated', blocks=[])
                record['rows'].append(row)
                rows[kind] = row
                print('ACCURACY', case, scope, kind, valid, json.dumps(validations[0]), flush=True)
                for i in range(args.warmup if not args.preflight else 1):
                    call_timed(evaluator, i % len(frames), scope, frames)
            save()
            if args.preflight:
                continue
            for block in range(args.blocks):
                order = list(KINDS)
                random.Random(1729+block).shuffle(order)
                for kind in order:
                    row, evaluator = rows[kind], evaluators[kind]
                    evaluator.configure()
                    call_timed(evaluator, block % len(frames), scope, frames)
                    times=[]
                    for i in range(args.calls):
                        elapsed,_ = call_timed(evaluator, i % len(frames), scope, frames)
                        times.append(elapsed)
                    entry=dict(block=block, seconds=times, median_ms=float(np.median(times)*1000), gpu=gpu_status())
                    row['blocks'].append(entry)
                    all_times=np.asarray([value for b in row['blocks'] for value in b['seconds']])
                    row.update(status='passed', median_ms=float(np.median(all_times)*1000),
                               p10_ms=float(np.percentile(all_times,10)*1000),p90_ms=float(np.percentile(all_times,90)*1000),
                               block_median_min_ms=min(b['median_ms'] for b in row['blocks']),
                               block_median_max_ms=max(b['median_ms'] for b in row['blocks']))
                    if not evaluator.native:
                        row['tensor_core_gemm_calls']=evaluator.calc.model.tensor_core_gemm_calls
                        row['sgemm_calls']=evaluator.calc.model.sgemm_calls
                    print('TIMING',case,scope,kind,block,row['median_ms'],flush=True)
                    save()
        # Model lifetime and allocator cleanup are outside timed sections.
        del reference, evaluators
        gc.collect()
        torch.cuda.empty_cache()
    record['finished_hardware']=gpu_status()
    record['all_accuracy_passed']=all(r['accuracy_passed'] for r in record['rows'])
    record['symmetrix_accuracy_passed']=all(r['accuracy_passed'] for r in record['rows'] if r['backend'].startswith('sym-'))
    save()
    assert record['symmetrix_accuracy_passed'], 'A Symmetrix backend failed its unchanged accuracy gates.'


if __name__ == '__main__':
    main()
