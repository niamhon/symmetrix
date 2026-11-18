#!/usr/bin/env python3
"""Save pre-interface native results without changing the frozen ASE references."""
import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from symmetrix import MACELSC, MACELSCCuda

parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','cuda'],required=True);args=parser.parse_args()
repo=Path(__file__).resolve().parents[2];export=repo/'.lsc-reference-logs/water-stage2-energy.json'
path=repo/f'.lsc-reference-logs/electrostatics-interface-baseline-{args.backend}.npz'
if path.exists():raise SystemExit(f'Refusing to overwrite {path}')
arrays={};modes=['float64'] if args.backend=='cpu' else ['float32','float32-mixed']
for precision in modes:
    model=MACELSC(str(export)) if args.backend=='cpu' else MACELSCCuda(str(export),precision)
    for case in ['dimer','bulk']:
        atoms=read(repo/f'symmetrix/test/lsc/data/water_stage2/{case}.extxyz')
        sender,receiver,vectors=neighbor_list('ijD',atoms,model.r_cut,self_interaction=False)
        types=np.array([model.atomic_numbers.index(int(z)) for z in atoms.numbers],dtype=np.int64)
        values=[types,np.array([sender,receiver]),vectors,atoms.positions.copy(),atoms.cell.array.copy(),atoms.pbc.copy()]
        for key,value in zip(['types','edges','vectors','positions','cell','pbc'],values):arrays[f'{case}/{key}']=value
        for label,field in [('zero',(0,0,0)),('field',(.01,-.02,.03))]:
            for repeat in range(3):
                model.compute_energy_forces(*values,'auto',field,compute_stress=True)
                if args.backend=='cuda':
                    result=model.download_forces();result.update(model.download_charges(False))
                else:
                    result={k:np.asarray(getattr(model,k)) for k in ['forces','position_forces','edge_forces','density_energy_gradient','partial_charges','partial_dipoles','density_coefficients']}
                    result['forces']=result['forces'].reshape(-1,3);result['position_forces']=result['position_forces'].reshape(-1,3)
                    result['edge_forces']=result['edge_forces'].reshape(-1,3);result['density_energy_gradient']=result['density_energy_gradient'].reshape(-1,4)
                for key in ['energy','local_energy','electrostatic_energy','field_energy','dipole','stress','virial','strain_derivative']:result[key]=np.asarray(getattr(model,key))
                for key,value in result.items():arrays[f'{precision}/{case}/{label}/{repeat}/{key}']=np.asarray(value)
                print('BASELINE',precision,case,label,repeat,model.energy,flush=True)
np.savez_compressed(path,**arrays)
meta=dict(host=socket.gethostname(),python=sys.executable,job_id=os.getenv('SLURM_JOB_ID'),backend=args.backend,
          export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
          description='Native pre-interface results, three repeats, dimer/bulk, zero/nonzero field; original ASE references untouched.')
path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
print(path,meta,flush=True)
