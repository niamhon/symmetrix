#!/usr/bin/env python3
"""Supplement the matched timestep screen with 10 ps water at 0.5 fs."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import nve_timestep as n

ROOT = n.ROOT/'long-water-10ps'


def main():
    ROOT.mkdir(exist_ok=True)
    # Wait without creating a CUDA context or competing with the primary sweep.
    while not (n.ROOT/'results/summary.csv').exists():
        p = json.loads((n.ROOT/'progress.json').read_text())
        if p['status'] == 'failed': raise RuntimeError('Primary sweep failed: '+json.dumps(p))
        time.sleep(30)
    primary = json.loads((n.ROOT/'results/runs.json').read_text())
    assert len(primary) == 36 and all(r['status'] == 'complete' for r in primary)
    import torch
    assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name()
    m = n.manifest()
    assert n.sha(n.binary()) == m['binary_sha256'] == n.shared.BINARY_SHA
    protocol = dict(model='water',atoms=m['models']['water']['atoms'],duration_fs=10000,
                    timestep_fs=.5,backends=n.BACKENDS,initial_state=m['models']['water'],
                    baseline_commit=m['baseline_commit'],binary_sha256=m['binary_sha256'],
                    reference='Unchanged MACELocalSplitCharges CUDA FP64 / ASE VelocityVerlet',
                    symmetrix='Actual LAMMPS fix nve, mesh 0.7 A/order 8, FP32 or mixed',
                    purpose='Longer-window check at the operating timestep; no additional equilibration')
    n.atomic(ROOT/'protocol.json',protocol)
    records=[]; calculator=None
    for backend in n.BACKENDS:
        folder=ROOT/backend;folder.mkdir(exist_ok=True)
        complete=[]
        for p in sorted(folder.glob('attempt*/summary.json')):
            r=json.loads(p.read_text())
            if r['status']=='complete':complete.append(r)
        if complete:records.append(complete[-1]);continue
        work=folder/f'attempt{len(list(folder.glob("attempt*")))+1}';work.mkdir()
        r=dict(model='water',backend=backend,timestep_fs=.5,duration_fs=10000,
               atoms=protocol['atoms'],work=str(work),status='running')
        n.atomic(ROOT/'progress.json',r);print('START_LONG',json.dumps(r),flush=True)
        try:
            if backend=='native':
                calculator=n.calc('water')
                data=n.native_run('water',.5,work,calculator,10000)
            else:
                torch.cuda.empty_cache()
                data=n.lammps_run('water',backend,.5,work,10000)
            r.update(status='complete',**data)
        except Exception as error:
            r.update(status='failed',error=repr(error));n.atomic(work/'summary.json',r)
            n.atomic(ROOT/'progress.json',r);raise
        n.atomic(work/'summary.json',r);records.append(r)
        n.atomic(ROOT/'records.json',records);n.atomic(ROOT/'progress.json',r)
        print('FINISHED_LONG',json.dumps(r),flush=True)
    n.audit()
    shutil.copy2(n.ROOT/'results/audit.json',ROOT/'audit.json')
    report(records)


def report(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    names={'native':'Native PyTorch/ASE FP64','float32':'Symmetrix FP32','float32-mixed':'Symmetrix mixed'}
    colors={'native':'#262626','float32':'#2372b7','float32-mixed':'#df7b17'}
    styles={'native':'-','float32':'--','float32-mixed':':'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(10.5,6.5),sharex=True,layout='constrained')
    checks=[]
    for r in records:
        b=r['backend'];work=Path(r['work'])
        with np.load(work/'series.npz',allow_pickle=False) as d:s=d['series']
        assert s.shape==(10001,5) and np.isfinite(s).all()
        np.testing.assert_array_equal(s[:,0],np.arange(10001))
        np.testing.assert_allclose(s[:,1]+s[:,2],s[:,3],atol=3e-9,rtol=0)
        with (work/'energy.csv').open() as stream:
            reader=csv.reader(stream);next(reader);saved=np.array([[float(x) for x in row] for row in reader])
        np.testing.assert_array_equal(saved,s)
        e=(s[:,3]-s[0,3])/r['atoms']*1000;t=s[:,0]/1000
        assert np.max(abs(e))<1000  # The same 1 eV/atom runaway guard.
        axes[0].plot(t,e,color=colors[b],ls=styles[b],lw=.65,alpha=.7,label=names[b])
        blocks=np.array([[np.mean(t[j:j+100]),np.mean(e[j:j+100])] for j in range(0,10000,100)])
        axes[1].plot(blocks[:,0],blocks[:,1],color=colors[b],ls=styles[b],lw=1.3,label=names[b])
        checks.append(dict(backend=b,samples=len(s),bookkeeping_passed=True,csv_matches_npz=True,
                           first_ps_metrics=n.drift_metrics(s[:1001],r['atoms'])))
    axes[0].set_ylabel('Δ total energy\n(meV/atom)');axes[0].legend(fontsize=8,frameon=False,ncol=3)
    axes[1].set_ylabel('0.1 ps block mean\n(meV/atom)');axes[1].set_xlabel('Time (ps)')
    fig.suptitle('Water · 384 atoms · 0.5 fs · one H100\n10 ps per backend, identical frozen starting state')
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'water-10ps.{suffix}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    n.atomic(ROOT/'saved-output-checks.json',checks)
    lines=['# Water: longer NVE check at 0.5 fs','','All three runs cover 10 ps, restarting the same '
        '330 K, 384-atom frozen state used in the primary timestep screen. The native calculator, '
        'model, Symmetrix binary, mesh and unit conversion are identical. No thermostat, barostat '
        'or constraints are used.','','| Backend | Slope (meV/atom/ps) | RMS (meV/atom) | Span (meV/atom) | Block slope ± nominal 95% |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append(f'| {names[r["backend"]]} | {r["slope_meV_atom_ps"]:.6g} | {r["energy_rms_meV_atom"]:.6g} | '
            f'{r["energy_span_meV_atom"]:.6g} | {r["block_slope_meV_atom_ps"]:.6g} ± {r["block_slope_95_halfwidth"]:.3g} |')
    lines+=['','![Water energy conservation over 10 ps](water-10ps.png)','','Initial energy/force and '
        'kinetic-energy checks use the unchanged primary gates. All three saved 10,001-sample '
        'series passed time-window, finite-value, energy-bookkeeping and CSV/NPZ identity checks. '
        'The source/binary/stable/frozen-input audit passed again after these trajectories.',
        '', 'Independent chaotic trajectories may lose phase agreement while conserving energy '
        'equally well. Block means and fitted slopes are finite-window diagnostics; correlated '
        'blocks make the nominal intervals indicative. This extension strengthens the water '
        'check; benzene and EMC remain the 1 ps timestep pilot. Longer independently equilibrated '
        'replicates would be needed for robust general secular-drift estimates.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(Path(__file__),ROOT/'nve_long_water.py')
    print('LONG_WATER_COMPLETE',flush=True)


if __name__=='__main__':main()
