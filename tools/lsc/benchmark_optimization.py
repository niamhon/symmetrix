#!/usr/bin/env python3
"""Before/after actual LAMMPS timing from one equilibrated 300 K restart."""
import argparse,json,os,re,shlex,subprocess
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
 p.add_argument('--launcher',default='mpirun --bind-to none -np 1');p.add_argument('--steps',type=int,default=500)
 p.add_argument('--ensemble',choices=['nve','npt'],default='nve');args=p.parse_args();root=args.root.resolve();work=root/('production-timing-'+args.ensemble);work.mkdir(exist_ok=True)
 repo=Path(__file__).resolve().parents[2]
 rows=[]
 initial_points={}
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()
 for precision in ['float32','float32-mixed']:
  for version in ['before','after']:
   name=precision+'-'+version;path=work/(name+'.in')
   restart=root/'long-md'/precision/'equilibrated.restart'
   commands=['units metal','atom_style charge',f'read_restart {restart}',
    'pair_style symmetrix/mace/lsc/cuda mesh .7 order 8'+(' float32-mixed' if precision.endswith('mixed') else ''),
    f'pair_coeff * * {repo}/.lsc-reference-logs/water-stage2-energy.json H O','neighbor .4 bin','neigh_modify every 1 delay 0 check yes',
    'thermo 1','thermo_style custom step temp pe ke etotal press','thermo_modify norm no format float %.17g',
    'timestep .0005','fix integrate all '+('nve' if args.ensemble=='nve' else 'npt temp 300 300 .1 iso 1 1 1'),'run 20',*[f'run {args.steps}']*3]
   path.write_text('\n'.join(commands)+'\n')
   # Preload the archived original DSO into the LAMMPS worker only, never mpirun/Python.
   override=['env',f'LD_PRELOAD={root}/before/liblammps-cuda.so.0'] if version=='before' else []
   result=subprocess.run([*shlex.split(args.launcher),*override,str(args.binary),'-log','none','-in',str(path)],
    stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=600)
   path.with_suffix('.log').write_text(result.stdout+result.stderr)
   assert result.returncode==0,result.stdout[-5000:]+result.stderr[-1500:]
   loops=re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps',result.stdout);assert len(loops)==4,loops
   times=[float(t)*1000/int(n) for t,n in loops[-3:]]
   point=next(np.fromstring(line,sep=' ') for line in result.stdout.splitlines() if len(line.split())==6 and line.split()[0].isdigit())
   if version=='before':initial_points[precision]=point
   else:
    np.testing.assert_allclose(point[2:5],initial_points[precision][2:5],atol=1536e-6,rtol=0)
    assert abs(point[5]-initial_points[precision][5])<.8
   row=dict(precision=precision,version=version,ms_step=times,median_ms_step=float(np.median(times)),initial_thermo=point.tolist(),neighbor_builds=[int(x) for x in re.findall(r'Neighbor list builds = (\d+)',result.stdout)])
   rows.append(row);print('PRODUCTION_TIMING',row,flush=True)
   (root/('production-timing-'+args.ensemble+'.json')).write_text(json.dumps(dict(job=os.environ.get('SLURM_JOB_ID'),gpu=gpu,ensemble=args.ensemble,pressure_target_bar=1. if args.ensemble=='npt' else None,timestep_fs=.5,initial_temperature_K=300.,atoms=1536,thermo_every=1,rows=rows),indent=2)+'\n')

if __name__=='__main__':main()
