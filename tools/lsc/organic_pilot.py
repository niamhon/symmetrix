#!/usr/bin/env python3
"""Prepare/freeze organic references, then run an explicit snapshot pilot."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'symmetrix/test/lsc'))
import organic as o
import numpy as np
from ase.io import iread, read, write

BASE = Path('/mnt/home/noneill/ceph/reactive_water-1/dft2dft/fresh_baselines/direct-cluster-to-bulk/inter-intra/organic-liquids')
SOURCES = {
    'benzene': (BASE / 'benzene/gen2/models/e100_force100/55a', 'periodic_reference.extxyz'),
    'emc': (BASE / 'emc/gen2/models/e100_force100/55a', 'gen1gen2_emc_periodic.extxyz')}


def provenance():
    def git(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    native = Path('/mnt/home/noneill/Programs/mace-scf-dev-niamh')
    checkout_commit = o.implementation_checkpoint()
    assert git(native) == '4d1fe8a40481be0c73507f0761f5eaf2b1d32468'
    binary = Path(os.environ.get('LSC_LAMMPS_BINARY', REPO / '.lsc-reference-logs/kokkos-charge-sync-20261119/bin/lmp-fixed'))
    assert o.sha256(binary) == o.BINARY_SHA
    stable = Path('/mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f')
    files = list((REPO / 'libsymmetrix/source').glob('mace_lsc*'))
    files += list((REPO / 'pair_symmetrix').glob('pair_symmetrix_mace_lsc*'))
    return dict(commit=o.PIN, checkout_commit=checkout_commit, native_commit=git(native), binary=str(binary), binary_sha256=o.sha256(binary),
                unchanged_sources={str(p.relative_to(REPO)): o.sha256(p) for p in files if p.is_file()},
                stable_binary_sha256=o.sha256(stable / 'bin/lmp'),
                versions={k: importlib.metadata.version(k) for k in ['torch','mace-scf','mace-torch','graph-longrange','ase','numpy','e3nn']})


def _formal(data):
    # Schema location is checked explicitly, never infer charges from elements.
    for k in ['lsc_architecture', 'architecture', 'lsc']:
        if k in data and isinstance(data[k], dict) and 'formal_charges' in data[k]:
            return data[k]['formal_charges']
    if 'formal_charges' in data:
        return data['formal_charges']
    raise ValueError('Export has no explicit formal-charge metadata')


def prepare():
    import torch
    from ase.neighborlist import neighbor_list
    from symmetrix.extract_mace_data import extract_mace_data
    torch.set_default_dtype(torch.float64)
    out = o.root()
    if (out / 'manifest.json').exists():
        raise FileExistsError('Preserving prepared fixtures: manifest already exists')
    for d in ['models', 'fixtures', 'references', 'results', 'checks']:
        (out / d).mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    purpose='Native PyTorch/ASE versus LAMMPS Symmetrix component parity',
                    tolerances=o.TOLERANCES, provenance=provenance(), models={})
    for name, (source, periodic) in SOURCES.items():
        checkpoint = source / 'seed1' / f'direct_c2b_{name}_pbed3_gen2_e100_force100_seed1_stage3.model'
        model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        numbers = [int(z) for z in model.atomic_numbers]
        cutoff = float(model.r_max)
        copied = out / 'models' / f'{name}.model'
        if copied.exists():
            assert o.sha256(copied) == o.sha256(checkpoint)
        else:
            shutil.copy2(checkpoint, copied)
        print('EXPORT_START', name, flush=True)
        export = extract_mace_data(str(copied), numbers, num_spline_points=4096, r_min=0.5)
        formal = _formal(export)
        path = out / 'models' / f'{name}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(export, allow_nan=False))
        info = dict(source_model=str(checkpoint), model_sha256=o.sha256(copied),
                    export_sha256=o.sha256(path), atomic_numbers=numbers,
                    formal_charges=formal, cutoff_A=cutoff, spline_points=4096, r_min_A=0.5,
                    cases=[], sources={})
        selected = []
        for group, source_file in [('cluster',source / 'seed1/test.xyz'), ('periodic',source / periodic)]:
            frames = list(iread(source_file, index=':'))
            molecular = [i for i,a in enumerate(frames) if len(a)>1]
            ordered = sorted(molecular, key=lambda i:(len(frames[i]) if group=='cluster' else frames[i].get_volume(),i))
            indices = [ordered[i] for i in np.rint(np.linspace(0,len(ordered)-1,10)).astype(int)]
            if group == 'cluster':
                indices += [i for i,a in enumerate(frames) if len(a)==1]
            info['sources'][group] = dict(path=str(source_file), sha256=o.sha256(source_file), frames=len(frames))
            for index in indices:
                a = frames[index].copy()
                a.calc = None
                assert set(a.numbers).issubset(numbers)
                assert a.pbc.all() if group=='periodic' else not a.pbc.any()
                distances = neighbor_list('d',a,0.5,self_interaction=False)
                if len(distances):
                    raise ValueError(f'{name} {group} frame {index} below spline r_min: {distances.min()}')
                # Preserve positions, cell and source metadata, excluding label arrays.
                metadata = dict(a.info)
                for k in list(a.arrays):
                    if k not in ['numbers','positions','molID','mol_id']:
                        del a.arrays[k]
                a.info = metadata
                case = dict(id=len(selected), group=group if len(a)>1 else 'isolated',
                            source_index=index, atoms=len(a), formula=a.get_chemical_formula(),
                            pbc=a.pbc.tolist(), cell=a.cell.array.tolist())
                selected.append(a); info['cases'].append(case)
        fixture = out / 'fixtures' / f'{name}.extxyz'
        # ASE's standard extxyz writer uses eight decimal places for positions;
        # these frozen tests require exact float64 coordinate round trips.
        from ase.io.extxyz import key_val_dict_to_str
        with fixture.open('w') as stream:
            for a in selected:
                extra=[k for k in a.arrays if k not in ['numbers','positions']]
                properties='species:S:1:pos:R:3'
                for k in extra:
                    if a.arrays[k].ndim != 1:
                        raise ValueError(f'Unexpected fixture array dimensions: {k}')
                    properties+=f':{k}:I:1'
                header={**a.info,'Lattice':a.cell.array.T,'Properties':properties,'pbc':a.pbc}
                stream.write(str(len(a))+'\n'+key_val_dict_to_str(header)+'\n')
                for i,(symbol,x) in enumerate(zip(a.get_chemical_symbols(),a.positions)):
                    stream.write(symbol+' '+' '.join(f'{v:.17g}' for v in x)+
                                 ''.join(' '+str(int(a.arrays[k][i])) for k in extra)+'\n')
        reread = list(iread(fixture,index=':'))
        for a,b in zip(selected,reread):
            np.testing.assert_array_equal(a.numbers,b.numbers)
            np.testing.assert_array_equal(a.positions,b.positions)
            np.testing.assert_array_equal(a.cell.array,b.cell.array)
        info['fixtures_sha256'] = o.sha256(fixture)
        manifest['models'][name] = info
        print('PREPARED',name,len(selected),'cases',flush=True)
    o.atomic_json(out / 'manifest.json', manifest)


def freeze():
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'Reference stage requires allocated GPU'
    m = o.manifest()
    for name, info in m['models'].items():
        destination = o.root() / 'references' / f'{name}.npz'
        if destination.exists():
            raise FileExistsError('Never regenerate existing frozen predictions: '+str(destination))
        arrays = {}
        calculators = {}
        for case in info['cases']:
            a = o.load_case(name,case['id'])
            mode = 'mixed_periodic' if all(a.pbc) else 'realspace'
            if mode not in calculators:
                calculators[mode] = o.native_calculator(name,mode)
            values = o.evaluate_native(a,calculators[mode])
            for key,value in values.items():
                arrays[f'{case["id"]:03d}__{key}'] = value
            print('FROZEN',name,case['id'],case['group'],len(a),flush=True)
        np.savez_compressed(destination,**arrays)
        info['reference_sha256'] = o.sha256(destination)
        del calculators
        torch.cuda.empty_cache()
    m['reference_device'] = torch.cuda.get_device_name()
    m['reference_dtype'] = 'float64'
    m['reference_use_compile'] = False
    o.atomic_json(o.root()/'manifest.json',m)
    # Small frozen files are version-controlled; large models and logs stay on Ceph.
    tracked = REPO/'symmetrix/test/lsc/data/organic_stage3'
    tracked.mkdir(parents=True,exist_ok=True)
    for name in m['models']:
        for kind,suffix in [('fixtures','extxyz'),('references','npz')]:
            dest=tracked/f'{name}.{suffix}'
            if dest.exists():raise FileExistsError(dest)
            shutil.copy2(o.root()/kind/f'{name}.{suffix}',dest)
    shutil.copy2(o.root()/'manifest.json',tracked/'manifest.json')


def comparisons():
    m=o.manifest(); rows=[]
    for name,info in m['models'].items():
        for precision in ['float32','float32-mixed']:
            for case in info['cases']:
                a=o.load_case(name,case['id']);work=o.root()/'results'/name/precision/f'{case["id"]:03d}'
                execution_error=None
                try:
                    actual=o.lammps(name,a,precision,work)
                    errors=o.compare(actual,o.reference(name,case['id']),a)
                    expected_q=sum(info['formal_charges'][info['atomic_numbers'].index(int(z))] for z in a.numbers)
                    errors['charge_conservation_error_e']=abs(float(actual['charges'].sum())-expected_q)
                    errors['passed'] &= errors['charge_conservation_error_e']<1e-8
                except Exception as error:
                    errors=dict(passed=False,error=repr(error))
                    execution_error=error
                row=dict(model=name,precision=precision,case=case['id'],group=case['group'],atoms=len(a),**errors)
                o.atomic_json(work/'accuracy.json',row);rows.append(row)
                o.atomic_json(o.root()/'results/comparisons.json',rows)
                print('COMPARED',name,precision,case['id'],json.dumps(errors),flush=True)
                if execution_error is not None:
                    raise RuntimeError('Stopped after first evaluator/launch failure; preserve logs and diagnose') from execution_error
    report()


def report():
    rows=json.loads((o.root()/'results/comparisons.json').read_text())
    summaries=[]
    for name in SOURCES:
        for precision in ['float32','float32-mixed']:
            for group in ['all','cluster','periodic','isolated']:
                subset=[r for r in rows if r['model']==name and r['precision']==precision
                        and (group=='all' or r['group']==group)]
                if not subset:continue
                valid=[r for r in subset if 'energy_eV_atom' in r]
                energies=[r['energy_eV_atom'] for r in valid]
                s=dict(model=name,precision=precision,group=group,cases=len(subset),
                       failures=sum(not r['passed'] for r in subset))
                if energies:
                    s['energy_eV_atom']={k:float(v) for k,v in dict(
                        mean=np.mean(energies),rms=np.sqrt(np.mean(np.square(energies))),
                        median=np.median(energies),p95=np.percentile(energies,95),max=max(energies)).items()}
                for key in ['forces','stress','charges','dipole']:
                    differences=[]
                    for r in valid:
                        with np.load(o.root()/'results'/name/precision/f'{r["case"]:03d}'/'prediction.npz') as data:
                            if key in data:
                                differences.append((data[key]-o.reference(name,r['case'])[key]).ravel())
                    if differences:
                        delta=np.concatenate(differences)
                        s[key]=dict(max=float(np.max(abs(delta))),rms=float(np.sqrt(np.mean(delta**2))),
                                    components=int(delta.size))
                summaries.append(s)
    o.atomic_json(o.root()/'results/summary.json',summaries)
    import csv
    with (o.root()/'results/case-errors.csv').open('w') as stream:
        keys=['model','precision','case','group','atoms','energy_eV_atom','forces_max_eV_A',
              'forces_rms_eV_A','stress_max_eV_A3','stress_rms_eV_A3','charges_max_e','dipole_max_eA','passed']
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    lines=['# Organic snapshot parity', '', '| Model | Precision | Cases | E max (eV/atom) | F max (eV/A) | F RMS max | S max (eV/A3) | S RMS max | Failed |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in SOURCES:
        for precision in ['float32','float32-mixed']:
            subset=[r for r in rows if r['model']==name and r['precision']==precision]
            metrics=[]
            for key in ['energy_eV_atom','forces_max_eV_A','forces_rms_eV_A','stress_max_eV_A3','stress_rms_eV_A3']:
                values=[r[key] for r in subset if key in r]
                metrics.append(f'{max(values):.6e}' if values else 'unavailable')
            lines.append('| '+ ' | '.join([name,precision,str(len(subset)),*metrics,str(sum(not r['passed'] for r in subset))])+' |')
    lines+=['','The summary reports maxima across the selected cases, including maxima of per-case RMS errors.',
            'Each native reference is the unchanged PyTorch/ASE FP64 calculator. Dataset DFT labels are not used.',
            'LAMMPS dipoles are compared on the native coordinate branch using recorded formal-charge image shifts.']
    (o.root()/'results/ERRORS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def audit():
    m=o.manifest();p=m['provenance']
    changed=[f for f,h in p['unchanged_sources'].items() if o.sha256(REPO/f)!=h]
    binary=o.sha256(p['binary'])
    stable=Path('/mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f/bin/lmp')
    stable_sha=o.sha256(stable)
    stable_library_sha=o.sha256(stable.parent.parent/'lib/liblammps.so.0')
    source_changed=[]
    for name,info in m['models'].items():
        if o.sha256(info['source_model'])!=info['model_sha256']:source_changed.append(name)
        for directory,suffix,key in [('models','model','model_sha256'),('models','json','export_sha256'),
                                     ('fixtures','extxyz','fixtures_sha256'),('references','npz','reference_sha256')]:
            assert o.sha256(o.root()/directory/f'{name}.{suffix}')==info[key]
    result=dict(implementation_sources_unchanged=not changed,changed_implementation_sources=changed,
                binary_unchanged=binary==p['binary_sha256']==o.BINARY_SHA,
                stable_binary_unchanged=stable_sha==p['stable_binary_sha256'],
                stable_library_unchanged=stable_library_sha=='587e91763c8daead33e225ae6b6de89d2eb9d8e0d0cccbedb7f6809595e85656',
                source_models_unchanged=not source_changed,
                reference_hashes_verified=True,baseline_commit=o.PIN)
    result['native_git_status']=subprocess.check_output(
        ['git','-C','/mnt/home/noneill/Programs/mace-scf-dev-niamh','status','--porcelain','--untracked-files=no'],text=True)
    result['job_id']=os.environ.get('SLURM_JOB_ID')
    result['host']=subprocess.check_output(['hostname'],text=True).strip()
    o.atomic_json(o.root()/'results/audit.json',result)
    assert all(result[k] for k in ['implementation_sources_unchanged','binary_unchanged',
                                  'stable_binary_unchanged','stable_library_unchanged','source_models_unchanged','reference_hashes_verified'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','reference','compare','report','audit'])
    p.add_argument('--root',default=str(o.DEFAULT_ROOT))
    a=p.parse_args();os.environ['LSC_ORGANIC_DATA']=a.root
    dict(prepare=prepare,reference=freeze,compare=comparisons,report=report,audit=audit)[a.stage]()
