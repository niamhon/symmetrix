#!/usr/bin/env python3
"""One-rank Symmetrix LAMMPS point validation and warmed NVE loop throughput."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from ase import Atoms
from ase.io import write

REPO=Path('/mnt/home/noneill/Programs/symmetrix-dev')
BINARY=REPO.parent/'lammps-lsc-dev/build-cuda/lmp'
MPI='/mnt/sw/nix/store/pa59vldasm7gxpr7dkijhk09q5qq63q1-openmpi-4.1.8/bin/mpirun'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--warmup',type=int,default=20);p.add_argument('--repeats',type=int,default=3);p.add_argument('--sizes',type=int,nargs='+',default=[192,1536]);args=p.parse_args()
 output=Path(args.output);root=output.parent;work=root/'lammps';work.mkdir(exist_ok=True)
 record=dict(job_id=os.environ.get('SLURM_JOB_ID'),binary=str(BINARY),ranks=1,threads=int(os.environ['OMP_NUM_THREADS']),
  settings=vars(args),scope='Actual LAMMPS Verlet loop, energy and global pressure every step; excludes loading, run-0 validation and warmup. Separate workflow from ASE benchmark.',rows=[])
 save=lambda:output.write_text(json.dumps(record,indent=2)+'\n')
 for n in args.sizes:
  with np.load(root/f'bulk_{n}-frames.npz') as data:
   atoms=Atoms(numbers=data['numbers'],positions=data['positions'][0],cell=data['cell'],pbc=data['pbc'])
  with np.load(root/f'bulk_{n}-reference.npz') as data:expected={k:data['0/'+k].copy() for k in ['energy','forces','stress','dipole']}
  datafile=work/f'bulk_{n}.data';write(datafile,atoms,format='lammps-data',atom_style='atomic',specorder=['H','O'],masses=True)
  for precision in ['float32','float32-mixed']:
   for solver in ['direct','mesh']:
    for neighbors in ['reuse','rebuild']:
     option=' float32-mixed' if precision=='float32-mixed' else ''
     electro='direct' if solver=='direct' else 'mesh .7 order 8'
     row=dict(atoms=n,precision=precision,solver=solver,neighbors=neighbors,trials=[])
     for repeat in range(args.repeats):
      name=f'{n}-{precision}-{solver}-{neighbors}-{repeat}';path=work/(name+'.in');dump=work/(name+'.dump')
      commands=['units metal','atom_style atomic','boundary p p p','processors 1 1 1',f'read_data {datafile}',
       f'pair_style symmetrix/mace/lsc/cuda {electro}{option}',f'pair_coeff * * {REPO}/.lsc-reference-logs/water-stage2-energy.json H O',
       'neighbor .4 bin','neigh_modify every 1 delay 0 check '+('yes' if neighbors=='reuse' else 'no'),
       'compute mu all pair symmetrix/mace/lsc/cuda','thermo 1',
       'thermo_style custom step pe ke etotal pxx pyy pzz pxy pxz pyz','thermo_modify norm no format float %.17g','run 0',
       'print "SPEED_POINT $(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g) $(c_mu[1]:%.17g) $(c_mu[2]:%.17g) $(c_mu[3]:%.17g)"',
       f'write_dump all custom {dump} id fx fy fz modify sort id format float %.17g',
       'velocity all create 10 19371 mom yes rot yes loop geom','fix integrate all nve','timestep .0001',
       f'run {args.warmup}',f'run {args.steps}']
      path.write_text('\n'.join(commands)+'\n')
      start=time.perf_counter();result=subprocess.run([MPI,'--bind-to','none','-np','1',str(BINARY),'-log','none','-in',str(path)],text=True,capture_output=True,stdin=subprocess.DEVNULL,timeout=300)
      (work/(name+'.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stdout[-6000:]+result.stderr[-2000:]
      match=re.search(r'^SPEED_POINT (.*)$',result.stdout,re.MULTILINE);assert match
      values=np.fromstring(match.group(1),sep=' ');forces=np.loadtxt(dump.read_text().splitlines()[9:])[:,1:4]
      stress=-values[1:7][[0,1,2,5,4,3]]/1.6021765e6
      assert abs(values[0]-float(expected['energy']))/n<1e-6
      np.testing.assert_allclose(forces,expected['forces'],atol=8e-5,rtol=1e-4)
      np.testing.assert_allclose(stress,expected['stress'],atol=5e-7,rtol=1e-4)
      np.testing.assert_allclose(values[7:10],expected['dipole'],atol=3e-5*n/192,rtol=0)
      loops=re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps',result.stdout)
      assert len(loops)==3 and int(loops[-1][1])==args.steps,loops
      builds=[int(value) for value in re.findall(r'Neighbor list builds = (\d+)',result.stdout)]
      samples=[]
      for line in result.stdout.splitlines():
       fields=line.split()
       if len(fields)==10 and fields[0].isdigit():
        try:samples.append([float(v) for v in fields])
        except ValueError:pass
      samples=np.asarray(samples[-args.steps-1:]);assert len(samples)==args.steps+1
      drift=float(np.max(abs(samples[:,3]-samples[0,3]))/n)
      trial=dict(ms_per_step=float(loops[-1][0])*1000/args.steps,neighbor_builds=builds[-1],energy_drift_eV_per_atom=drift,
       point_energy_error_eV=abs(values[0]-float(expected['energy'])),point_force_error_eV_A=float(np.max(abs(forces-expected['forces']))),
       point_stress_error_eV_A3=float(np.max(abs(stress-expected['stress']))),wall_seconds_with_startup=time.perf_counter()-start)
      row['trials'].append(trial);print('LAMMPS',n,precision,solver,neighbors,repeat,trial,flush=True)
     times=[v['ms_per_step'] for v in row['trials']]
     row.update(median_ms_per_step=float(np.median(times)),minimum_ms_per_step=min(times),maximum_ms_per_step=max(times),steps=args.steps,warmup_steps=args.warmup)
     record['rows'].append(row);save()
 save()

if __name__=='__main__':main()
