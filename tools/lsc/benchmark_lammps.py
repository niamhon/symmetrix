#!/usr/bin/env python3
"""LAMMPS NVE loop timings; loading and warm-up are excluded."""
import argparse
import json
import os
import re
import shlex
import socket
import subprocess
from pathlib import Path
import numpy as np
import torch
from ase.io import read, write

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
binary = repo.parent / 'lammps-lsc-dev/build-cuda/lmp'
work = repo / '.lsc-reference-logs' / f'lammps-timing-{os.environ.get("SLURM_JOB_ID", "manual")}'
work.mkdir(exist_ok=True)
record = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
              scope='LAMMPS NVE loop only; one MPI process, four CPU threads, 0.1 fs timestep; excludes model loading and 10-step warm-up.', rows=[])
for case in ['dimer', 'bulk', 'bulk_2x2x2']:
    atoms = read(repo / 'symmetrix/test/lsc/data/water_stage2' / ('dimer.extxyz' if case == 'dimer' else 'bulk.extxyz'))
    if case == 'dimer': atoms.center()
    else: atoms.wrap()
    if case == 'bulk_2x2x2': atoms = atoms.repeat((2, 2, 2))
    data = work / f'{case}.data'
    write(data, atoms, format='lammps-data', atom_style='atomic', specorder=['H', 'O'], masses=True)
    steps = 50 if case == 'bulk_2x2x2' else 100
    for backend in ['cpu', 'float32', 'float32-mixed']:
        style = 'symmetrix/mace/lsc' + ('' if backend == 'cpu' else '/cuda')
        option = ' float32-mixed' if backend == 'float32-mixed' else ''
        commands = ['units metal', 'atom_style atomic', 'boundary ' + ('f f f' if case == 'dimer' else 'p p p'),
                    f'read_data {data}', f'pair_style {style} direct{option}',
                    f'pair_coeff * * {repo}/.lsc-reference-logs/water-stage2-energy.json H O',
                    'neighbor 0.2 bin', 'neigh_modify every 1 delay 0 check yes',
                    'velocity all create 10 19371 mom yes rot yes', 'fix integrate all nve', 'timestep 0.0001',
                    'thermo 100', 'thermo_style custom step pe ke etotal temp', 'run 10', f'run {steps}']
        path = work / f'{case}-{backend}.in'
        path.write_text('\n'.join(commands) + '\n')
        times = []
        for repeat in range(2):
            launcher = shlex.split(os.environ.get("LSC_LAMMPS_LAUNCHER", ""))
            result = subprocess.run(launcher + [str(binary), '-log', 'none', '-in', str(path)], text=True, capture_output=True, timeout=600)
            (work / f'{case}-{backend}-{repeat}.log').write_text(result.stdout + result.stderr)
            assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
            loops = re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps', result.stdout)
            assert len(loops) == 2 and int(loops[-1][1]) == steps, result.stdout[-4000:]
            times.append(float(loops[-1][0]) / steps)
        row = dict(case=case, atoms=len(atoms), backend=backend, steps=steps, repeats=2,
                   median_ms_per_step=float(np.median(times)) * 1000, seconds_per_step=times)
        record['rows'].append(row)
        Path(args.output).write_text(json.dumps(record, indent=2) + '\n')
        print('LAMMPS_TIMING', json.dumps(row), flush=True)
