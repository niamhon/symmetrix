"""Short real NVT/NVE/NPT and restart smoke test; long runs use the same driver."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from .reference import REPO
from .test_cuda_loader import require_cuda

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda, pytest.mark.lsc_lammps]

def test_water_npt_and_restart(tmp_path):
    require_cuda()
    binary=Path(os.environ.get('LSC_LAMMPS_BINARY',str(REPO.parent/'lammps-lsc-dev/build-cuda/lmp')))
    if not binary.is_file():
        pytest.fail(f'Missing LAMMPS executable: {binary}')
    result=subprocess.run([sys.executable,str(REPO/'tools/lsc/validate_water_md.py'),
        '--output',str(tmp_path),'--binary',str(binary),'--equil-steps','100',
        '--nve-steps','100','--npt-steps','100','--dump-every','100','--repeat','1'],
        text=True,capture_output=True,stdin=subprocess.DEVNULL,timeout=300)
    assert result.returncode==0,result.stdout[-5000:]+result.stderr[-3000:]
    report=json.loads((tmp_path/'results.json').read_text())
    assert report['atoms']==192 and report['timestep_fs']==.5
    assert len(report['rows'])==2
    for row in report['rows']:
        assert row['passed']
        work=tmp_path/row['precision']
        assert (work/'in.md.log').is_file() and (work/'in.restart.log').is_file()
        assert row['restart_energy_error_eV_atom']<1e-6
        assert row['npt_volume_min_max_A3'][0]<row['npt_volume_min_max_A3'][1]
        assert {v['step'] for v in row['snapshots']} >= {0,100,200,300}
