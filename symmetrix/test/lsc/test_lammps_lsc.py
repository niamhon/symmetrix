"""Basic direct LSC pair-style point predictions and short NVE checks."""
import os
import re
import shlex
import subprocess
from pathlib import Path
import numpy as np
import pytest
from ase.io import write
from .reference import DATA,REPO,load_atoms
from .test_cuda_loader import require_cuda

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_lammps]
BACKENDS=["cpu",pytest.param("float32",marks=pytest.mark.lsc_cuda),pytest.param("float32-mixed",marks=pytest.mark.lsc_cuda)]


def setup(tmp_path,case,backend,md=False):
    if backend!="cpu":require_cuda()
    root=REPO.parent/"lammps-lsc-dev"
    binary=Path(os.environ.get("LSC_LAMMPS_BINARY",str(root/("build-cpu" if backend=="cpu" else "build-cuda")/"lmp")))
    if not binary.is_file():
        if os.environ.get("LSC_REQUIRE_LAMMPS")=="1":pytest.fail(f"Missing LSC LAMMPS executable: {binary}")
        pytest.skip("requires the LSC LAMMPS development build")
    atoms=load_atoms(case)
    if case=="dimer":atoms.center()
    else:atoms.wrap()
    data=tmp_path/"atoms.data"
    write(data,atoms,format="lammps-data",atom_style="charge" if not md else "atomic",specorder=["H","O"],masses=True)
    style="symmetrix/mace/lsc" if backend=="cpu" else "symmetrix/mace/lsc/cuda"
    precision=" float32-mixed" if backend=="float32-mixed" else ""
    commands=["units metal","atom_style "+("atomic" if md else "charge"),"boundary "+("p p p" if case=="bulk" else "f f f"),
              f"read_data {data}",f"pair_style {style} direct{precision}",
              f"pair_coeff * * {REPO/'.lsc-reference-logs/water-stage2-energy.json'} H O",
              "neighbor 0.2 bin","neigh_modify every 1 delay 0 check yes"]
    return binary,atoms,commands


def run(binary,commands,tmp_path):
    path=tmp_path/"in.lsc";path.write_text("\n".join(commands)+"\n")
    launcher=shlex.split(os.environ.get("LSC_LAMMPS_LAUNCHER", ""))
    result=subprocess.run(launcher+[str(binary),"-log","none","-in",str(path)],cwd=tmp_path,text=True,capture_output=True,timeout=300)
    (tmp_path/"lammps.stdout").write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout[-12000:]+result.stderr[-4000:]
    return result.stdout


@pytest.mark.parametrize("backend",BACKENDS)
@pytest.mark.parametrize("case",["dimer","bulk"])
def test_pair_point_prediction(backend,case,tmp_path,record_property):
    binary,atoms,commands=setup(tmp_path,case,backend)
    dump=tmp_path/"forces.dump"
    commands += ["compute eat all pe/atom","compute esum all reduce sum c_eat",
                 "compute mu all pair "+("symmetrix/mace/lsc" if backend=="cpu" else "symmetrix/mace/lsc/cuda"),
                 "thermo_style custom step pe c_esum pxx pyy pzz pxy pxz pyz c_mu[1] c_mu[2] c_mu[3]","thermo_modify format float %.17g",
                 "run 0",'print "LSC_RESULT $(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pxy:%.17g) $(pxz:%.17g) $(pyz:%.17g) $(c_esum:%.17g) $(c_mu[1]:%.17g) $(c_mu[2]:%.17g) $(c_mu[3]:%.17g)"',
                 f"write_dump all custom {dump} id type fx fy fz q c_eat modify sort id format float %.17g"]
    output=run(binary,commands,tmp_path)
    match=re.search(r"^LSC_RESULT (.*)$",output,re.MULTILINE);assert match,output
    values=np.fromstring(match.group(1),sep=" ")
    rows=np.loadtxt(dump.read_text().splitlines()[9:]);rows=rows[np.argsort(rows[:,0])]
    with np.load(DATA/f"{case}.reference.npz",allow_pickle=False) as expected:
        assert abs(values[0]-float(expected["energy"]))<5e-5
        np.testing.assert_allclose(rows[:,2:5],expected["forces"],atol=8e-5,rtol=1e-4)
        np.testing.assert_allclose(rows[:,5],expected["partial_charges"],atol=2e-6,rtol=1e-5)
        if case=="bulk":
            stress=-values[1:7][[0,1,2,5,4,3]]/1.6021766208e6
            np.testing.assert_allclose(stress,expected["stress"],atol=5e-7,rtol=1e-4)
        original=load_atoms(case)
        formal=np.where(atoms.numbers==1,1.,-2.)
        expected_dipole=expected["dipole"]+np.sum(formal[:,None]*(atoms.positions-original.positions),axis=0)
        np.testing.assert_allclose(values[8:11],expected_dipole,atol=3e-5,rtol=0)
        force_error=float(abs(rows[:,2:5]-expected["forces"]).max())
        energy_error=abs(values[0]-float(expected["energy"]))
    assert abs(values[0]-values[7])<2e-8
    assert abs(rows[:,6].sum()-values[0])<2e-8
    for key,value in dict(backend=backend,case=case,binary=str(binary),energy_abs_eV=energy_error,force_max_abs_eV_A=force_error).items():record_property(key,value)
    print(f"LAMMPS_LSC {backend} {case} energy_error={energy_error:.6e} force_error={force_error:.6e}")


@pytest.mark.parametrize("backend",BACKENDS)
@pytest.mark.parametrize("case",["dimer","bulk"])
def test_short_nve(backend,case,tmp_path,record_property):
    binary,atoms,commands=setup(tmp_path,case,backend,md=True)
    commands += ["velocity all create 10 19371 mom yes rot yes","fix integrate all nve","timestep 0.0001",
                 "thermo 1","thermo_style custom step pe ke etotal temp","thermo_modify format float %.17g","run 10"]
    output=run(binary,commands,tmp_path)
    samples=[]
    for line in output.splitlines():
        fields=line.split()
        if len(fields)==5 and fields[0].isdigit():
            try:samples.append([float(v) for v in fields])
            except ValueError:pass
    values=np.array(samples);assert len(values)==11,output
    drift=float(np.max(abs(values[:,3]-values[0,3]))/len(atoms))
    assert drift<1e-4
    record_property("backend",backend);record_property("case",case);record_property("energy_drift_eV_per_atom",drift)
    print(f"LAMMPS_NVE {backend} {case} steps=10 timestep_fs=0.1 drift_eV_per_atom={drift:.6e}")
