#!/usr/bin/env python3
"""Time the actual LSC and Polar LAMMPS pair styles on identical input boxes."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import numpy as np
from ase import Atoms
from ase.io import write

REPO=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--launcher',default='mpirun --bind-to none -np 1')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--blocks',type=int,default=3)
    args=p.parse_args();root=args.root.resolve();work=root/'polar-lammps';work.mkdir(exist_ok=True)
    result=dict(job=os.environ.get('SLURM_JOB_ID'),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip(),
        binary=str(args.binary),ranks=1,threads=os.environ.get('OMP_NUM_THREADS'),dt_fs=.1,temperature_K=10,
        warmup_steps=20,measured_steps=args.steps,blocks=args.blocks,rows=[],
        scope='Actual Verlet loop, global energy and pressure each step, normal neighbor reuse; distinct models, identical starting water geometry. Loading and warm-up excluded. No D4 correction added.')
    save=lambda:(root/'polar-lammps-results.json').write_text(json.dumps(result,indent=2)+'\n')
    for n in [192,1536]:
        with np.load(root.parent/f'h100-speed-20261118/bulk_{n}-frames.npz') as data:
            box=Atoms(numbers=data['numbers'],positions=data['positions'][0],cell=data['cell'],pbc=data['pbc'])
        datafile=work/f'water-{n}.data';write(datafile,box,format='lammps-data',masses=True,specorder=['H','O'])
        for kind in ['lsc','polar']:
            file=REPO/'.lsc-reference-logs/water-stage2-energy.json' if kind=='lsc' else root/'polar-example/MACE-POLAR-1-M-C-Cl-F-H-N-O.json'
            refpath=root.parent/f'h100-speed-20261118/bulk_{n}-reference.npz' if kind=='lsc' else root/f'polar-reference-{n}.npz'
            with np.load(refpath) as refs:expected={key:refs[('0/' if kind=='lsc' else '')+key] for key in ['energy','forces','stress']}
            for precision in ['float32','float32-mixed']:
                for method in ['direct','mesh']:
                    name=f'{n}-{kind}-{precision}-{method}';path=work/(name+'.in');dump=work/(name+'.dump')
                    style='symmetrix/mace/lsc/cuda' if kind=='lsc' else 'symmetrix/mace/polar/cuda'
                    settings=('direct' if method=='direct' else 'mesh .7 order 8')+(' float32-mixed' if precision.endswith('mixed') else '')
                    commands=['units metal','atom_style atomic','atom_modify map array','boundary p p p','processors 1 1 1',f'read_data {datafile}',
                        f'pair_style {style} {settings}',f'pair_coeff * * {file} H O','neighbor .4 bin','neigh_modify every 1 delay 0 check yes','thermo 1',
                        'thermo_style custom step pe ke etotal pxx pyy pzz pxy pxz pyz','thermo_modify norm no format float %.17g','run 0',
                        'print "POLAR_POINT $(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g)"',
                        f'write_dump all custom {dump} id fx fy fz modify sort id format float %.17g',
                        'velocity all create 10 19371 mom yes rot yes loop geom','fix integrate all nve','timestep .0001','run 20',*[f'run {args.steps}']*args.blocks]
                    path.write_text('\n'.join(commands)+'\n')
                    kokkos=['-k','on','g','1','-sf','kk','-pk','kokkos','newton','on','neigh','half'] if kind=='polar' else []
                    run=subprocess.run([*shlex.split(args.launcher),str(args.binary),*kokkos,'-log','none','-in',str(path)],stdin=subprocess.DEVNULL,text=True,capture_output=True,timeout=600)
                    path.with_suffix('.log').write_text(run.stdout+run.stderr)
                    assert run.returncode==0,run.stdout[-5000:]+run.stderr[-2000:]
                    values=np.fromstring(re.search(r'^POLAR_POINT (.*)$',run.stdout,re.MULTILINE).group(1),sep=' ')
                    forces=np.loadtxt(dump.read_text().splitlines()[9:])[:,1:4]
                    stress=-values[1:7][[0,1,2,5,4,3]]/1.6021765e6
                    errors=dict(energy_eV_atom=abs(values[0]-float(expected['energy']))/n,forces_eV_A=float(np.max(abs(forces-expected['forces']))),stress_eV_A3=float(np.max(abs(stress-expected['stress']))))
                    accuracy=bool(errors['energy_eV_atom']<1e-6 and np.allclose(forces,expected['forces'],atol=8e-5,rtol=1e-4) and np.allclose(stress,expected['stress'],atol=5e-7,rtol=1e-4))
                    loops=re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps',run.stdout)
                    assert len(loops)==args.blocks+2,loops
                    times=[float(t)*1000/int(steps) for t,steps in loops[-args.blocks:]]
                    row=dict(n=n,model=kind,precision=precision,electrostatics=method,block_ms_step=times,median_ms_step=float(np.median(times)),accuracy_passed=accuracy,errors=errors)
                    result['rows'].append(row);save();print('PAIR_COMPARISON',json.dumps(row),flush=True)
    save()

if __name__=='__main__':main()
