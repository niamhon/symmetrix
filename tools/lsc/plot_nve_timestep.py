#!/usr/bin/env python3
"""Render conservation measurements; this script never runs a calculator."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get('LSC_NVE_ROOT', '/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120'))
MODELS = ['water', 'benzene', 'emc']
BACKENDS = ['native', 'float32', 'float32-mixed']
LABELS = {'native': 'Native PyTorch/ASE FP64', 'float32': 'Symmetrix FP32',
          'float32-mixed': 'Symmetrix mixed'}
COLORS = {'native': '#262626', 'float32': '#2372b7', 'float32-mixed': '#df7b17'}
MARKERS = {'native': 'o', 'float32': 's', 'float32-mixed': '^'}
STYLES = {'native': '-', 'float32': '--', 'float32-mixed': ':'}


def save(fig, name):
    for suffix in ['png', 'pdf']:
        fig.savefig(ROOT/'plots'/f'{name}.{suffix}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main(traces_only=False):
    manifest = json.loads((ROOT/'manifest.json').read_text())
    rows = json.loads((ROOT/'results/runs.json').read_text())
    complete = [r for r in rows if r['status'] == 'complete']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'pdf.fonttype': 42})
    if traces_only:
        assert len([r for r in complete if r['timestep_fs'] == .5]) == 9
        plot_traces(complete)
        return
    assert len(complete) == 36, 'Plot the complete matched sweep, not an incomplete selection'
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10), sharex='col', layout='constrained')
    for i, name in enumerate(MODELS):
        title = f'{name.upper() if name == "emc" else name.title()} · {manifest["models"][name]["atoms"]:,} atoms'
        for backend in BACKENDS:
            group = sorted([r for r in complete if r['model'] == name and r['backend'] == backend],
                           key=lambda r: r['timestep_fs'])
            dt = [r['timestep_fs'] for r in group]
            kw = dict(color=COLORS[backend], marker=MARKERS[backend], label=LABELS[backend],
                      ls=STYLES[backend], lw=1.4, ms=6, fillstyle='none', alpha=.9)
            axes[i, 0].loglog(dt, [r['energy_rms_meV_atom'] for r in group], **kw)
            axes[i, 1].plot(dt, [r['slope_meV_atom_ps'] for r in group], **kw)
        anchor = next(r['energy_rms_meV_atom'] for r in complete
                      if r['model'] == name and r['backend'] == 'native' and r['timestep_fs'] == .5)
        guide_dt = np.array([.125, .25, .5, 1.])
        axes[i, 0].plot(guide_dt, anchor*(guide_dt/.5)**2, color='.65', ls='-.',
                        lw=1, zorder=0, label='dt² guide (anchored at native 0.5 fs)')
        axes[i, 0].set_ylabel('RMS energy fluctuation\n(meV/atom)')
        axes[i, 1].set_ylabel('Fitted slope\n(meV/atom/ps)')
        axes[i, 0].set_title(title, loc='left', fontweight='bold')
        axes[i, 1].axhline(0, color='.5', lw=.7, zorder=0)
        for ax in axes[i]:
            ax.set_xscale('log', base=2)
            ax.set_xticks([.125, .25, .5, 1.], labels=['0.125', '0.25', '0.5', '1.0'])
    axes[0, 0].legend(fontsize=8, frameon=False, loc='upper left')
    for ax in axes[-1]: ax.set_xlabel('Timestep (fs)')
    fig.suptitle('LSC NVE timestep screen · one H100 · 1 ps per trajectory\n'
                 'Same frozen starting state per liquid; fixed cell; no thermostat', fontsize=13)
    fig.supxlabel('RMS is centered about each trajectory mean. Short-window slopes do not establish long-time drift.', fontsize=9)
    save(fig, 'nve-timestep')
    plot_traces(complete)


def plot_traces(complete):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True, layout='constrained')
    for i, name in enumerate(MODELS):
        for backend in BACKENDS:
            r = next(r for r in complete if r['model'] == name and r['backend'] == backend and r['timestep_fs'] == .5)
            with np.load(Path(r['work'])/'series.npz', allow_pickle=False) as data:
                series = data['series']
            delta = (series[:, 3]-series[0, 3])/r['atoms']*1000
            axes[i].plot(series[:, 0]/1000, delta, color=COLORS[backend],
                         lw=1.7 if backend == 'native' else 1.1, ls=STYLES[backend],
                         label=LABELS[backend], alpha=.85)
        axes[i].set_title(name.upper() if name == 'emc' else name.title(), loc='left', fontweight='bold')
        axes[i].set_ylabel('Δ total energy\n(meV/atom)')
        axes[i].axhline(0, color='.5', lw=.6)
    axes[0].legend(fontsize=9, frameon=False)
    axes[-1].set_xlabel('Time (ps)')
    fig.suptitle('Total-energy conservation at 0.5 fs\nEach curve subtracts its own initial total energy', fontsize=13)
    save(fig, 'nve-energy-0.5fs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces-only', action='store_true', help='Plot the nine completed 0.5 fs runs while the sweep continues')
    main(parser.parse_args().traces_only)
