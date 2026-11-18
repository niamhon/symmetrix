#!/usr/bin/env python3
"""Single-GPU water MD, restart and native MACE-SCF trajectory checks.

The MD and snapshot checks are sequential so they do not contend for the GPU.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import numpy as np
from ase.io import read, write

REPO = Path(__file__).resolve().parents[2]

def launch(binary, launcher, commands, path):
    path.write_text('\n'.join(commands)+'\n')
    start = time.perf_counter()
    result = subprocess.run([*shlex.split(launcher), str(binary), '-log', 'none', '-in', str(path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3600)
    path.with_name(path.name+'.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout[-5000:]+result.stderr[-1000:]
    return result.stdout, time.perf_counter()-start


def thermo(text):
    rows=[]
    for line in text.splitlines():
        values=line.split()
        if len(values)==17 and values[0].isdigit():
            try: rows.append([float(v) for v in values])
            except ValueError: pass
    result=np.asarray(rows)
    assert len(result)>0 and np.isfinite(result).all()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=REPO.parent/'lammps-lsc-dev/build-cuda/lmp')
    p.add_argument('--launcher',default=os.environ.get('LSC_LAMMPS_LAUNCHER','mpirun --bind-to none -np 1'))
    p.add_argument('--equil-steps',type=int,default=2000)
    p.add_argument('--nve-steps',type=int,default=10000)
    p.add_argument('--npt-steps',type=int,default=10000)
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--dump-every',type=int,default=1000)
    args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    from benchmark_h100 import Evaluator, copied
    box=read(REPO/'symmetrix/test/lsc/data/water_stage2/bulk.extxyz').repeat((args.repeat,)*3)
    box.wrap()
    data=root/'water.data';write(data,box,format='lammps-data',atom_style='charge',masses=True,specorder=['H','O'])
    record=dict(job_id=os.environ.get('SLURM_JOB_ID'),atoms=len(box),temperature_K=300.,pressure_bar=1.,timestep_fs=.5,
        thermostat_damping_ps=.1,barostat_damping_ps=1.,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},rows=[])
    save=lambda:(root/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    native=Evaluator('mace-fp64')
    for precision in ['float32','float32-mixed']:
        work=root/precision;work.mkdir(exist_ok=True)
        option=' float32-mixed' if precision.endswith('mixed') else ''
        pair=['pair_style symmetrix/mace/lsc/cuda mesh .7 order 8'+option,
            f'pair_coeff * * {REPO}/.lsc-reference-logs/water-stage2-energy.json H O']
        common=['neighbor .4 bin','neigh_modify every 1 delay 0 check yes',
            'compute mu all pair symmetrix/mace/lsc/cuda', 'compute vir all pressure NULL virial', 'thermo 100',
            'thermo_style custom step temp press pe ke etotal vol density c_vir[1] c_vir[2] c_vir[3] c_vir[4] c_vir[5] c_vir[6] c_mu[1] c_mu[2] c_mu[3]',
            'thermo_modify norm no format float %.17g','timestep .0005']
        commands=['units metal','atom_style charge','boundary p p p',f'read_data {data}',*pair,*common,
            f'dump states all custom {args.dump_every} {work}/trajectory.dump id type xu yu zu vx vy vz fx fy fz q',
            'dump_modify states sort id format float %.17g',
            'velocity all create 300 19371 mom yes rot yes loop geom',
            'fix integrate all nvt temp 300 300 .1',f'run {args.equil_steps}',f'write_restart {work}/equilibrated.restart',
            'unfix integrate','fix integrate all nve',f'run {args.nve_steps}',f'write_restart {work}/nve.restart',
            'unfix integrate','fix integrate all npt temp 300 300 .1 iso 1 1 1',f'run {args.npt_steps}',f'write_restart {work}/npt.restart']
        text,wall=launch(args.binary,args.launcher,commands,work/'in.md')
        samples=thermo(text)
        nve=samples[(samples[:,0]>=args.equil_steps)&(samples[:,0]<=args.equil_steps+args.nve_steps)]
        # Remove duplicate boundary rows; absolute energy includes a large atomic offset.
        nve=np.asarray(list({r[0]:r for r in nve}.values()))
        nve_max=float(np.max(abs(nve[:,5]-nve[0,5]))/len(box))
        slope=float(np.polyfit((nve[:,0]-nve[0,0])*.0005,nve[:,5]-nve[0,5],1)[0]/len(box))
        np.savetxt(work/'thermo.txt',samples)
        frames=read(work/'trajectory.dump',index=':',format='lammps-dump-text',specorder=['H','O'])
        # Check beginning, both sides of the NVE/NPT boundary and final NPT cell.
        targets=[0,args.equil_steps,args.equil_steps+args.nve_steps,args.equil_steps+args.nve_steps+args.npt_steps//2,args.equil_steps+args.nve_steps+args.npt_steps]
        indices=sorted(set(min(range(len(frames)),key=lambda i:abs(int(frames[i].info['timestep'])-target)) for target in targets))
        checks=[]
        for index in indices:
            frame=frames[index];step=int(frame.info['timestep'])
            numerical=frame.calc.results['forces'].copy()
            charges=frame.get_initial_charges().copy()
            native.configure();native.calc.calculate(frame,properties=['energy','forces','stress'])
            expected=copied(native.calc.results)
            row=samples[samples[:,0]==step][-1]
            stress=-row[8:14][[0,1,2,5,4,3]]/1.6021765e6
            eerr=abs(row[3]-float(expected['energy']))/len(box)
            ferr=float(np.max(abs(numerical-expected['forces'])))
            serr=float(np.max(abs(stress-expected['stress'])))
            passed=bool(eerr<1e-6 and np.allclose(numerical,expected['forces'],atol=8e-5,rtol=1e-4) and np.allclose(stress,expected['stress'],atol=5e-7,rtol=1e-4))
            muerr=float(np.max(abs(row[14:17]-expected['dipole'])))
            qerr=float(np.max(abs(charges-expected['charges'])))
            passed=passed and muerr<3e-5*len(box)/192 and qerr<2e-6
            np.testing.assert_allclose(row[14:17],expected['dipole'],atol=3e-5*len(box)/192,rtol=0)
            np.testing.assert_allclose(charges,expected['charges'],atol=3e-5,rtol=1e-4)
            check=dict(step=step,dipole_error_eA=muerr,charge_error_e=qerr,energy_error_eV_atom=eerr,force_error_eV_A=ferr,stress_error_eV_A3=serr,passed=passed)
            checks.append(check);print('SNAPSHOT',precision,check,flush=True)
        # Restore the NVT endpoint, reissue pair settings, and continue 100 NVE steps.
        restart=['units metal','atom_style charge',f'read_restart {work}/equilibrated.restart',*pair,*common,'fix integrate all nve','thermo 100','run 100']
        rt,_=launch(args.binary,args.launcher,restart,work/'in.restart')
        restored=thermo(rt)
        original=samples[samples[:,0]==args.equil_steps+100][0]
        delta=np.max(abs(restored[-1,3:6]-original[3:6]))
        # Reordering at restart can change FP32 sums; use the established energy scale.
        assert delta/len(box)<1e-6,delta
        npt=np.asarray(list({r[0]:r for r in samples[samples[:,0]>=args.equil_steps+args.nve_steps]}.values()))
        loops=re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps',text)
        result=dict(precision=precision,wall_seconds=wall,nve_max_excursion_eV_atom=nve_max,nve_linear_drift_eV_atom_ps=slope,
            final_density_g_cm3=float(samples[-1,7]),npt_mean_temperature_K=float(np.mean(npt[:,1])),npt_mean_pressure_bar=float(np.mean(npt[:,2])),
            npt_volume_min_max_A3=[float(np.min(npt[:,6])),float(np.max(npt[:,6]))],restart_energy_error_eV_atom=float(delta/len(box)),
            loops=[dict(seconds=float(t),steps=int(n),ms_step=1000*float(t)/int(n)) for t,n in loops],snapshots=checks,
            passed=bool(all(c['passed'] for c in checks)))
        record['rows'].append(result);save();print('MD_RESULT',json.dumps(result),flush=True)
        assert result['passed'],result
    save()

if __name__=='__main__':main()
