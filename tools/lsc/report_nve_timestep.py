#!/usr/bin/env python3
"""Generate the NVE report from complete saved trajectories and their audits."""
import json
import os
import re
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get('LSC_NVE_ROOT', '/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120'))
MODELS = ['water', 'benzene', 'emc']
BACKENDS = ['native', 'float32', 'float32-mixed']
LABELS = {'native': 'Native ASE FP64', 'float32': 'Symmetrix FP32', 'float32-mixed': 'Symmetrix mixed'}


def main():
    m = json.loads((ROOT/'manifest.json').read_text())
    runtime = json.loads((ROOT/'results/runtime.json').read_text())
    rows = json.loads((ROOT/'results/runs.json').read_text())
    assert len(rows) == 36 and all(r['status'] == 'complete' for r in rows)
    audit = json.loads((ROOT/'results/audit.json').read_text())
    assert all(audit.values()), audit
    pytest_log = (ROOT/'results/pytest.log').read_text()
    passed = re.search(r'(\d+) passed', pytest_log)
    assert passed and int(passed.group(1)) == 49, pytest_log
    lookup = {(r['model'], r['backend'], r['timestep_fs']): r for r in rows}
    paired = []
    scaling = {}
    for name in MODELS:
        for backend in BACKENDS:
            group = sorted([r for r in rows if r['model'] == name and r['backend'] == backend], key=lambda r: r['timestep_fs'])
            scaling[f'{name}/{backend}'] = float(np.polyfit(np.log([r['timestep_fs'] for r in group]),
                                      np.log([r['energy_rms_meV_atom'] for r in group]), 1)[0])
            if backend == 'native': continue
            for r in group:
                native = lookup[name, 'native', r['timestep_fs']]
                with np.load(Path(native['work'])/'series.npz', allow_pickle=False) as d: ref = d['series'].copy()
                with np.load(Path(r['work'])/'series.npz', allow_pickle=False) as d: actual = d['series'].copy()
                ref_delta = (ref[:, 3]-ref[0, 3])/r['atoms']*1000
                actual_delta = (actual[:, 3]-actual[0, 3])/r['atoms']*1000
                paired.append(dict(model=name, backend=backend, timestep_fs=r['timestep_fs'],
                    slope_difference_meV_atom_ps=r['slope_meV_atom_ps']-native['slope_meV_atom_ps'],
                    rms_ratio=r['energy_rms_meV_atom']/native['energy_rms_meV_atom'],
                    max_relative_energy_trace_difference_meV_atom=float(np.max(abs(actual_delta-ref_delta)))))
    comparison = dict(paired=paired, fitted_RMS_timestep_exponents=scaling,
                      maximum_absolute_slope_difference_meV_atom_ps=max(abs(p['slope_difference_meV_atom_ps']) for p in paired),
                      maximum_absolute_RMS_relative_difference=max(abs(p['rms_ratio']-1) for p in paired))
    (ROOT/'results/backend-comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    lines = ['# LSC NVE energy conservation versus timestep', '',
        'Completed on one H100 PCIe, allocation 7506125 / workergpu151. All 36 independent trajectories '
        'cover 1 ps, starting from the same frozen state for each liquid. The development implementation '
        'and stable installation remained unchanged.', '',
        '**This is a short conservation screen.** Fitted 1 ps slopes are apparent trends; '
        'bounded Verlet oscillations and the initial vibrational phase can contribute. '
        'These results do not establish long-time secular drift or an equilibrated liquid property.', '',
        '## Starting states and physics', '',
        '| Liquid | Atoms | Initial temperature (K) | Initial configuration |',
        '| --- | ---: | ---: | --- |']
    for name, v in m['models'].items():
        lines.append(f'| {name} | {v["atoms"]} | {v["target_temperature_K"]} | `{v["initial_source"]}` |')
    lines += ['', 'The original coordinates, cells and masses were preserved. Saved organic momenta were '
        'used; water momenta were generated reproducibly. Center-of-mass momentum was removed and '
        'the temperature set once before freezing each starting state. No thermostat, barostat or '
        'constraints were applied during NVE. No extra D4 term was added, matching the original '
        'full-model native simulation scripts.', '',
        'Native reference: unchanged MACELocalSplitCharges, CUDA FP64, uncompiled, ASE VelocityVerlet. '
        'Symmetrix: actual LAMMPS fix nve with the pinned Kokkos LSC pair, one rank, mesh 0.7 Å/order 8, '
        'FP32 or mixed neural precision, with FP64 electrostatics/reductions in both modes. '
        'Explicit masses and the recorded metal/ASE unit conversion match '
        'the kinetic energy and Verlet update. Energies are sampled at full steps every 1 fs.', '',
        '## Complete measurements', '',
        'Energy RMS is the centered standard deviation of total energy per atom. Span is maximum '
        'minus minimum. Slope is an ordinary least-squares fit to total energy versus time. '
        'All energy quantities remove the constant initial total-energy offset.', '',
        '| Liquid | dt (fs) | Backend | Slope (meV/atom/ps) | RMS (meV/atom) | Span (meV/atom) |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for name in MODELS:
        for dt in sorted(m['timesteps_fs']):
            for backend in BACKENDS:
                r = lookup[name, backend, dt]
                lines.append(f'| {name} | {dt:g} | {LABELS[backend]} | {r["slope_meV_atom_ps"]:.6g} | '
                             f'{r["energy_rms_meV_atom"]:.6g} | {r["energy_span_meV_atom"]:.6g} |')
    lines += ['', 'See `results/summary.csv` for maximum/final energy departures, block-fit slopes, '
        'nominal block-fit intervals and wall times. Ten 0.1 ps blocks can remain correlated; the '
        'intervals are indicative and should not be treated as independent-sample confidence limits.', '',
        '![Conservation versus timestep](plots/nve-timestep.png)', '',
        '![Energy traces at 0.5 fs](plots/nve-energy-0.5fs.png)', '',
        '## Backend comparison and timestep scaling', '',
        f'The largest absolute matched-backend slope difference over the sweep is '
        f'{comparison["maximum_absolute_slope_difference_meV_atom_ps"]:.6g} meV/atom/ps. '
        f'The largest relative difference in energy RMS is '
        f'{comparison["maximum_absolute_RMS_relative_difference"]*100:.4g}%. '
        'These compare conservation statistics at the same timestep, not a force prediction on '
        'the same later snapshot: each trajectory evolves independently.', '',
        '| Liquid | Native RMS exponent | FP32 RMS exponent | Mixed RMS exponent |',
        '| --- | ---: | ---: | ---: |']
    for name in MODELS:
        lines.append('| '+name+' | '+' | '.join(f'{scaling[f"{name}/{b}"]:.4f}' for b in BACKENDS)+' |')
    lines += ['', 'The exponent fits RMS ∝ timestep^p over all four tested timesteps. '
        'A value near two is consistent with the leading velocity-Verlet integration error. '
        'It is a description of this finite window, not a guarantee at other temperatures or timesteps.', '',
        '## Verification and provenance', '',
        'All initial LAMMPS energies and force components passed the unchanged snapshot gates: '
        'energy < 1e-6 eV/atom; forces atol 8e-5 eV/Å and rtol 1e-4. Initial kinetic energies '
        'agreed within 1e-10 eV/atom. The separate pytest log records the saved-trajectory checks. '
        'No numerical drift acceptance gate was invented for this pilot.', '',
        '**49 saved-trajectory and preservation checks passed.** These verify complete time windows, '
        'energy bookkeeping, frozen inputs, initial-state parity and unchanged installations; '
        'they do not impose a new arbitrary threshold on the measured drift.', '',
        f'* Development commit: `{m["baseline_commit"]}`',
        f'* LAMMPS binary SHA256: `{m["binary_sha256"]}`',
        f'* Native source commit: `{runtime["sources"]["native_mace_scf"]["commit"]}`',
        '* Python/packages: '+runtime['python']+'; '+', '.join(f'{k} {v}' for k,v in runtime['packages'].items()),
        '* Model/export/state hashes, source paths, masses, cell and unit conversion: `manifest.json`.',
        '* Stable binary/library, implementation and frozen-input audit: `results/audit.json`.',
        '* Raw energy CSV/NPZ, initial/final dumps or states, exact LAMMPS inputs and logs: `runs/`.',
        '* Pairwise statistics: `results/backend-comparison.json`.', '',
        'A longer NVE run with several independently equilibrated starting states would be the next '
        'step for a robust secular-drift estimate. This screen specifically tests the requested '
        'timestep dependence and implementation agreement.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__': main()
