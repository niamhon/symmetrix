#!/usr/bin/env python3
"""Matched native ASE / actual LAMMPS NVE timestep conservation screen."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.io import read

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'symmetrix/test/lsc'))
import organic as shared

ROOT=Path(os.environ.get('LSC_NVE_ROOT','/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120'))
BASE=Path('/mnt/home/noneill/ceph/reactive_water-1/dft2dft/fresh_baselines/direct-cluster-to-bulk/inter-intra')
SNAPSHOTS=Path('/mnt/home/noneill/ceph/symmetrix-lsc-validation/organic-stage3')
MVV2E=1.0364269e-4  # Verified LAMMPS metal units, src/update.cpp.
TIME_PS_PER_FS=units.fs*np.sqrt(MVV2E)
VELOCITY_TO_METAL=1/np.sqrt(MVV2E)
TIMESTEPS=[0.5,1.0,0.25,0.125]  # Complete the production-relevant points first.
BACKENDS=['native','float32','float32-mixed']
SPECS={
 'water':dict(folder=BASE/'pbed3/energy100_force100/adam/65a_clusters/simulations/nvt/1',
              initial='water_init.pdb',model='direct_c2b_pbed3_c65_r55_energy100_force100_adam_seed1_stage3.model',temperature=330),
 'benzene':dict(folder=BASE/'organic-liquids/benzene/gen2/models/e100_force100/55a/simulations/nvt/1',
                initial='benzene_init.extxyz',model='direct_c2b_benzene_pbed3_gen2_e100_force100_seed1_stage3.model',temperature=300),
 'emc':dict(folder=BASE/'organic-liquids/emc/gen2/models/e100_force100/55a/simulations/nvt/1',
            initial='emc_init.extxyz',model='direct_c2b_emc_pbed3_gen2_e100_force100_seed1_stage3.model',temperature=300)}


def manifest():return json.loads((ROOT/'manifest.json').read_text())
def atomic(path,value):shared.atomic_json(path,value)
def sha(path):return shared.sha256(path)
def binary():return Path(os.environ.get('LSC_LAMMPS_BINARY',REPO/'.lsc-reference-logs/kokkos-charge-sync-20261119/bin/lmp-fixed'))


def state(name):
    with np.load(ROOT/'states'/f'{name}.npz',allow_pickle=False) as data:
        a=Atoms(numbers=data['numbers'],positions=data['positions'],cell=data['cell'],
                pbc=data['pbc'],masses=data['masses'],momenta=data['momenta'])
    return a


def prepare(duration):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
    assert not (ROOT/'manifest.json').exists(),'Preserving existing NVE manifest'
    for d in ['models','states','runs','results','plots']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    checkout_commit=shared.implementation_checkpoint()
    assert sha(binary())==shared.BINARY_SHA
    stable=Path('/mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f')
    sources=list((REPO/'libsymmetrix/source').glob('mace_lsc*'))+list((REPO/'pair_symmetrix').glob('pair_symmetrix_mace_lsc*'))
    m=dict(schema_version=1,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           duration_fs=duration,timesteps_fs=TIMESTEPS,backends=BACKENDS,log_interval_fs=1.,
           integrator='Velocity Verlet; no thermostat/barostat/constraints',
           electrostatics='native mixed_periodic; LAMMPS mesh 0.7 A/order 8',
           baseline_commit=shared.PIN,checkout_commit=checkout_commit,binary=str(binary()),binary_sha256=sha(binary()),
           stable_binary_sha256=sha(stable/'bin/lmp'),stable_library_sha256=sha(stable/'lib/liblammps.so.0'),
           source_sha256={str(p.relative_to(REPO)):sha(p) for p in sources if p.is_file()},
           unit_conversion=dict(lammps_mvv2e=MVV2E,ase_fs=units.fs,time_ps_per_requested_fs=float(TIME_PS_PER_FS),
                                velocity_metal_per_ase=float(VELOCITY_TO_METAL),
                                explanation='Match kinetic energy and the Verlet update using each engine unit constants.'),models={})
    organic=json.loads((SNAPSHOTS/'manifest.json').read_text())
    for name,spec in SPECS.items():
        folder=spec['folder'];initial=folder/spec['initial'];source_model=folder/spec['model']
        a=read(initial,index=0);a.calc=None
        assert a.cell.rank==3 and not a.constraints
        a.pbc=True  # Matches the original simulation scripts.
        assert set(a.numbers)<=({1,8} if name=='water' else {1,6} if name=='benzene' else {1,6,8})
        assert np.allclose(a.cell.array,np.diag(np.diag(a.cell.array)),atol=1e-12)
        saved=a.has('momenta');original_temperature=float(a.get_temperature()) if saved else None
        if not saved:MaxwellBoltzmannDistribution(a,temperature_K=spec['temperature'],rng=np.random.default_rng(20261120))
        Stationary(a,preserve_temperature=False)
        target=(3*len(a)-3)*units.kB*spec['temperature']/2
        a.set_momenta(a.get_momenta()*np.sqrt(target/a.get_kinetic_energy()))
        source_export=(REPO/'.lsc-reference-logs/kokkos-charge-sync-20261119/speed/models/lsc-stage3.json'
                       if name=='water' else SNAPSHOTS/'models'/f'{name}.json')
        expected=('5e775df4dda17c438cfd1f2894ab2ccd97b20cf07a29dfc89b2400794f2fe9d4'
                  if name=='water' else organic['models'][name]['model_sha256'])
        assert sha(source_model)==expected
        for src,dest in [(source_model,ROOT/'models'/f'{name}.model'),(source_export,ROOT/'models'/f'{name}.json')]:
            if dest.exists():assert sha(src)==sha(dest)
            else:shutil.copy2(src,dest)
        export=json.loads((ROOT/'models'/f'{name}.json').read_text())
        np.savez_compressed(ROOT/'states'/f'{name}.npz',numbers=a.numbers,positions=a.positions,
                            cell=a.cell.array,pbc=a.pbc,masses=a.get_masses(),momenta=a.get_momenta())
        m['models'][name]=dict(initial_source=str(initial),initial_source_sha256=sha(initial),source_model=str(source_model),
            model_sha256=expected,export_sha256=sha(ROOT/'models'/f'{name}.json'),
            state_sha256=sha(ROOT/'states'/f'{name}.npz'),atoms=len(a),formula=a.get_chemical_formula(),
            cell_A=a.cell.array.tolist(),target_temperature_K=spec['temperature'],saved_velocities=saved,
            original_saved_temperature_K=original_temperature,initial_kinetic_eV=float(a.get_kinetic_energy()),
            momentum_conditioning='Remove COM once, rescale once to (3N-3)kBT/2; no constraints during NVE',
            atomic_numbers=export['atomic_numbers'])
        print('PREPARED',name,len(a),'K0',a.get_kinetic_energy(),flush=True)
    atomic(ROOT/'manifest.json',m)


def calc(name):
    import torch
    from mace_scf.calculators.localsources import MACELocalSplitCharges
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    c=MACELocalSplitCharges(str(ROOT/'models'/f'{name}.model'),device='cuda',default_dtype='float64',
                            pbc_handling='mixed_periodic',use_compile=False)
    c.model.eval()
    for p in c.model.parameters():p.requires_grad_(False)
    return c


def drift_metrics(series,atoms):
    from scipy.stats import t as student_t
    values=np.asarray(series,dtype=float)
    time_ps=values[:,0]/1000
    energy=(values[:,3]-values[0,3])/atoms  # Full-step total energy, eV/atom.
    slope,intercept=np.polyfit(time_ps,energy,1)
    blocks=[]
    for block in range(int(np.floor(time_ps[-1]/0.1+1e-9))):
        mask=(time_ps>=block*.1)&(time_ps<(block+1)*.1)
        if np.sum(mask)>=80:blocks.append([float(np.mean(time_ps[mask])),float(np.mean(energy[mask]))])
    out=dict(slope_meV_atom_ps=float(slope*1000),energy_span_meV_atom=float(np.ptp(energy)*1000),
             energy_rms_meV_atom=float(np.std(energy)*1000),max_abs_energy_change_meV_atom=float(np.max(abs(energy))*1000),
             final_energy_change_meV_atom=float(energy[-1]*1000),
             initial_temperature_K=float(values[0,4]),final_temperature_K=float(values[-1,4]))
    if len(blocks)>=4:
        b=np.asarray(blocks);coef=np.polyfit(b[:,0],b[:,1],1)
        residual=b[:,1]-np.polyval(coef,b[:,0]);se=np.sqrt(np.sum(residual**2)/(len(b)-2)/np.sum((b[:,0]-b[:,0].mean())**2))
        out.update(block_slope_meV_atom_ps=float(coef[0]*1000),
                   block_slope_95_halfwidth=float(student_t.ppf(.975,len(b)-2)*se*1000),blocks=len(b))
    return out


def native_run(name,dt,work,calculator,duration):
    from ase.md.verlet import VelocityVerlet
    a=state(name);a.calc=calculator
    e0=float(a.get_potential_energy());f0=a.get_forces().copy()
    initial=ROOT/'states'/f'{name}.native-initial.npz'
    if not initial.exists():np.savez_compressed(initial,energy=e0,forces=f0,stress=a.get_stress())
    steps=int(round(duration/dt));stride=int(round(1/dt));rows=[]
    dyn=VelocityVerlet(a,dt*units.fs)
    last=[time.monotonic()]
    with (work/'energy.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['time_fs','potential_eV','kinetic_eV','total_eV','temperature_K'])
        def observe():
            pe=float(a.get_potential_energy());ke=float(a.get_kinetic_energy())
            row=[dyn.nsteps*dt,pe,ke,pe+ke,2*ke/((3*len(a)-3)*units.kB)]
            if not np.isfinite(row).all():raise FloatingPointError('Nonfinite native NVE state')
            if abs(row[3]-(rows[0][3] if rows else row[3]))/len(a)>1.0:raise FloatingPointError('NVE energy runaway > 1 eV/atom')
            rows.append(row);writer.writerow([f'{v:.17g}' for v in row]);stream.flush()
            if time.monotonic()-last[0]>30:
                print('PROGRESS',name,'native',dt,dyn.nsteps,'/',steps,flush=True);last[0]=time.monotonic()
        dyn.attach(observe,interval=stride)
        started=time.monotonic();dyn.run(steps);elapsed=time.monotonic()-started
    np.savez_compressed(work/'series.npz',series=np.asarray(rows))
    np.savez_compressed(work/'final-state.npz',positions=a.positions,momenta=a.get_momenta(),cell=a.cell.array)
    return dict(wall_seconds=elapsed,steps=steps,initial_kinetic_eV=rows[0][2],**drift_metrics(rows,len(a)))


def lammps_run(name,backend,dt,work,duration):
    import re,shlex
    a=state(name);m=manifest();species=m['models'][name]['atomic_numbers']
    rotation=shared._data(a,species,work/'atoms.data')
    velocity=(a.get_velocities()*VELOCITY_TO_METAL)@rotation
    with (work/'atoms.data').open('a') as stream:
        stream.write('\nVelocities\n\n')
        for i,v in enumerate(velocity,1):stream.write(str(i)+' '+' '.join(f'{x:.17g}' for x in v)+'\n')
    steps=int(round(duration/dt));stride=int(round(1/dt));suffix=' float32-mixed' if backend=='float32-mixed' else ''
    commands=['units metal','atom_style charge','atom_modify map array sort 0 0.0','boundary p p p',
              f'read_data {work/"atoms.data"}']
    for i,z in enumerate(species,1):
        mass=a.get_masses()[a.numbers==z];assert np.ptp(mass)<1e-14
        commands.append(f'mass {i} {mass[0]:.17g}')
    commands += [f'pair_style {shared.STYLE} mesh 0.7 order 8{suffix}',
                 f'pair_coeff * * {ROOT/"models"/(name+".json")} '+' '.join(__import__('ase.data',fromlist=['chemical_symbols']).chemical_symbols[z] for z in species),
                 'neighbor 0.4 bin','neigh_modify every 1 delay 0 check yes',
                 f'timestep {dt*TIME_PS_PER_FS:.17g}','fix integrate all nve',f'thermo {stride}',
                 'thermo_style custom step pe ke etotal temp','thermo_modify norm no format float %.17g flush yes',
                 'run 0',f'write_dump all custom {work/"initial.dump"} id type x y z fx fy fz modify sort id format float %.17g',
                 f'run {steps}',f'write_dump all custom {work/"final.dump"} id type xu yu zu vx vy vz modify sort id format float %.17g']
    (work/'in.lammps').write_text('\n'.join(commands)+'\n')
    launcher=shlex.split(os.environ.get('LSC_LAMMPS_LAUNCHER',''))
    args=launcher+[str(binary()),'-k','on','g','1','-sf','kk','-pk','kokkos','newton','on','neigh','half','-log','none','-in',str(work/'in.lammps')]
    started=time.monotonic()
    with (work/'lammps.log').open('w') as stream:
        p=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                         cwd=work,timeout=3600)
    if p.returncode:raise RuntimeError((work/'lammps.log').read_text()[-7000:])
    samples={}
    for line in (work/'lammps.log').read_text().splitlines():
        fields=line.split()
        if len(fields)==5 and fields[0].isdigit():
            try:values=[float(x) for x in fields]
            except ValueError:continue
            if not np.isfinite(values).all():raise FloatingPointError('Nonfinite LAMMPS NVE state')
            step=int(values[0]);pe,ke,total=values[1:4]
            samples[step]=[step*dt,pe,ke,total,2*ke/((3*len(a)-3)*units.kB)]
    assert max(samples)==steps and len(samples)==int(duration)+1
    rows=np.array([samples[s] for s in sorted(samples)])
    np.savetxt(work/'energy.csv',rows,delimiter=',',header='time_fs,potential_eV,kinetic_eV,total_eV,temperature_K',comments='',fmt='%.17g')
    np.savez_compressed(work/'series.npz',series=rows)
    with np.load(ROOT/'states'/f'{name}.native-initial.npz') as ref:
        dump=np.atleast_2d(np.loadtxt((work/'initial.dump').read_text().splitlines()[9:]));dump=dump[np.argsort(dump[:,0])]
        initial_error=abs(rows[0,1]-float(ref['energy']))/len(a)
        force_error=float(abs(dump[:,5:8]@rotation.T-ref['forces']).max())
        assert initial_error<1e-6
        np.testing.assert_allclose(dump[:,5:8]@rotation.T,ref['forces'],atol=8e-5,rtol=1e-4)
    k0=float(a.get_kinetic_energy());assert abs(rows[0,2]-k0)/len(a)<1e-10
    if np.max(abs(rows[:,3]-rows[0,3]))/len(a)>1.0:raise FloatingPointError('NVE energy runaway > 1 eV/atom')
    return dict(wall_seconds=time.monotonic()-started,steps=steps,initial_energy_error_eV_atom=initial_error,
                initial_force_error_eV_A=force_error,initial_kinetic_error_eV_atom=abs(rows[0,2]-k0)/len(a),
                initial_kinetic_eV=rows[0,2],**drift_metrics(rows,len(a)))


def run():
    import torch
    m=manifest();assert sha(binary())==m['binary_sha256']==shared.BINARY_SHA
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    calculators={};records=[]
    for dt in m['timesteps_fs']:
        for name in SPECS:
            for backend in BACKENDS:
                folder=ROOT/'runs'/name/backend/f'dt-{dt:g}'
                done=[]
                for p in sorted(folder.glob('attempt*/summary.json')):
                    d=json.loads(p.read_text())
                    if d['status']=='complete':done.append(d)
                if done:records.append(done[-1]);continue
                folder.mkdir(parents=True,exist_ok=True)
                work=folder/f'attempt{len(list(folder.glob("attempt*")))+1}';work.mkdir()
                row=dict(model=name,backend=backend,timestep_fs=dt,duration_fs=m['duration_fs'],
                         atoms=m['models'][name]['atoms'],work=str(work),status='running')
                atomic(ROOT/'progress.json',row);print('START',json.dumps(row),flush=True)
                try:
                    if backend=='native':
                        if name not in calculators:calculators[name]=calc(name)
                        data=native_run(name,dt,work,calculators[name],m['duration_fs'])
                    else:
                        torch.cuda.empty_cache()
                        data=lammps_run(name,backend,dt,work,m['duration_fs'])
                    row.update(status='complete',**data)
                except Exception as error:
                    row.update(status='failed',error=repr(error))
                    atomic(work/'summary.json',row);records.append(row)
                    atomic(ROOT/'results/runs.json',records);atomic(ROOT/'progress.json',row)
                    raise
                atomic(work/'summary.json',row);records.append(row)
                atomic(ROOT/'results/runs.json',records);atomic(ROOT/'progress.json',row)
                print('FINISHED',json.dumps(row),flush=True)
    analyse()


def analyse():
    rows=json.loads((ROOT/'results/runs.json').read_text())
    keys=['model','backend','timestep_fs','duration_fs','status','slope_meV_atom_ps',
          'block_slope_meV_atom_ps','block_slope_95_halfwidth','energy_span_meV_atom','energy_rms_meV_atom',
          'max_abs_energy_change_meV_atom','final_energy_change_meV_atom','wall_seconds']
    with (ROOT/'results/summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# NVE timestep screen','',
           '| Liquid | Backend | dt (fs) | Slope (meV/atom/ps) | Block slope ±95% | Span (meV/atom) | RMS (meV/atom) |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(rows,key=lambda r:(list(SPECS).index(r['model']),r['timestep_fs'],BACKENDS.index(r['backend']))):
        if r['status']=='complete':
            values=[r['model'],r['backend'],f'{r["timestep_fs"]:g}',f'{r["slope_meV_atom_ps"]:.4g}',
                    f'{r.get("block_slope_meV_atom_ps",0):.4g} ± {r.get("block_slope_95_halfwidth",0):.3g}',
                    f'{r["energy_span_meV_atom"]:.4g}',f'{r["energy_rms_meV_atom"]:.4g}']
            lines.append('| '+' | '.join(values)+' |')
    lines+=['','Slopes over this short window are apparent trends. Bounded Verlet oscillations can dominate; '
            'use excursion/RMS versus timestep alongside slopes. Block intervals do not establish long-time secular drift.']
    (ROOT/'results/TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def audit():
    m=manifest();stable=Path('/mnt/home/noneill/Programs/symmetrix-lsc-stable/releases/2762b9f')
    native=Path('/mnt/home/noneill/Programs/mace-scf-dev-niamh')
    native_pin='4d1fe8a40481be0c73507f0761f5eaf2b1d32468'
    result=dict(binary_unchanged=sha(binary())==m['binary_sha256'],
                development_source_unchanged=bool(shared.implementation_checkpoint()) and m['baseline_commit']==shared.PIN,
                native_reference_unchanged=subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD'],text=True).strip()==native_pin and not subprocess.check_output(['git','-C',str(native),'status','--short'],text=True).strip(),
                implementation_diff_clean=subprocess.run(['git','-C',str(REPO),'diff','--quiet','HEAD','--','libsymmetrix/source','pair_symmetrix']).returncode==0,
                stable_binary_unchanged=sha(stable/'bin/lmp')==m['stable_binary_sha256'],
                stable_library_unchanged=sha(stable/'lib/liblammps.so.0')==m['stable_library_sha256'],
                implementation_unchanged=all(sha(REPO/p)==h for p,h in m['source_sha256'].items()),
                models_and_states_unchanged=all(sha(ROOT/'models'/f'{name}.model')==v['model_sha256'] and
                    sha(ROOT/'models'/f'{name}.json')==v['export_sha256'] and
                    sha(ROOT/'states'/f'{name}.npz')==v['state_sha256'] for name,v in m['models'].items()))
    atomic(ROOT/'results/audit.json',result);assert all(result.values());print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','analyse','audit'])
    parser.add_argument('--duration-fs',type=int,default=1000)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.duration_fs)
    else:dict(run=run,analyse=analyse,audit=audit)[args.stage]()
