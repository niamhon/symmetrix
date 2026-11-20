#!/usr/bin/env bash
# Complete analysis of the existing sweep; never launch MD or rebuild software.
set -e
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LSC_NVE_ROOT=/mnt/home/noneill/ceph/symmetrix-lsc-validation/nve-timestep-20261120
export LSC_LAMMPS_BINARY=/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/kokkos-charge-sync-20261119/bin/lmp-fixed
export XDG_CACHE_HOME=/tmp/noneill-nve-report/cache MPLCONFIGDIR=/tmp/noneill-nve-report/matplotlib
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd /mnt/home/noneill/Programs/symmetrix-dev
python -u - <<'PY'
import json, os, time
from pathlib import Path
r=Path(os.environ['LSC_NVE_ROOT'])
while True:
    progress=json.loads((r/'progress.json').read_text())
    if progress['status']=='failed':
        raise RuntimeError('Preserved failed attempt: '+json.dumps(progress))
    if (r/'results/summary.csv').exists():
        rows=json.loads((r/'results/runs.json').read_text())
        if len(rows)==36 and all(x['status']=='complete' for x in rows):break
    time.sleep(30)
print('All 36 trajectories complete; starting audit and report', flush=True)
PY
python tools/lsc/nve_timestep.py audit
export LSC_RUN_NVE=1
pytest -q symmetrix/test/lsc/test_nve_timestep.py --junitxml="$LSC_NVE_ROOT/results/pytest.xml" > "$LSC_NVE_ROOT/results/pytest.log" 2>&1
cat "$LSC_NVE_ROOT/results/pytest.log"
python tools/lsc/plot_nve_timestep.py
python tools/lsc/report_nve_timestep.py
mkdir -p "$LSC_NVE_ROOT/code"
cp tools/lsc/nve_timestep.py tools/lsc/plot_nve_timestep.py tools/lsc/report_nve_timestep.py tools/lsc/NVE_VALIDATION.md tools/lsc/finish_nve_timestep.sh symmetrix/test/lsc/test_nve_timestep.py "$LSC_NVE_ROOT/code/"
python - <<'PY'
import hashlib,json,os,tarfile
from pathlib import Path
r=Path(os.environ['LSC_NVE_ROOT']);dest=r.parent/'nve-timestep-20261120-evidence.tar.gz'
with tarfile.open(dest,'w:gz') as out:
    for name in ['manifest.json','REPORT.md','states','runs','results','plots','code']:
        out.add(r/name,arcname=name)
info={'path':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
(r/'results/archive.json').write_text(json.dumps(info,indent=2)+'\n')
print('EVIDENCE',json.dumps(info))
PY
