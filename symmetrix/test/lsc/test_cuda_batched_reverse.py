import numpy as np
import pytest
import symmetrix
from .reference import load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import require_cuda
from .test_cuda_derivatives import inputs
pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_cuda]
@pytest.mark.parametrize('precision',['float32','float32-mixed'])
@pytest.mark.parametrize('case,mesh',[('dimer',False),('bulk',False),('bulk',True)])
@pytest.mark.parametrize('field',[(0.,0.,0.),(.01,-.02,.03)])
def test_batched_reverse(export_file,lsc_model,precision,case,mesh,field,record_property):
    require_cuda();args=inputs(lsc_model,case)
    native=symmetrix.MACELSC(str(export_file));legacy=symmetrix.MACELSCCuda(str(export_file),precision);batch=symmetrix.MACELSCCuda(str(export_file),precision)
    assert not batch.use_batched_reverse;batch.use_batched_reverse=True
    for model in [native,legacy,batch]:
        if mesh:model.set_mesh_electrostatics(.5,8)
    energies=[model.compute_energy_forces(*args,external_field=field,compute_stress=True) for model in [native,legacy,batch]]
    assert batch.batched_reverse_calls==1 and legacy.batched_reverse_calls==0
    assert energies[1]==energies[2]
    np.testing.assert_array_equal(legacy.dipole,batch.dipole)
    np.testing.assert_array_equal(legacy.download_charges()['partial_charges'],batch.download_charges()['partial_charges'])
    forces=batch.download_forces()['forces'];expected=np.asarray(native.forces).reshape(-1,3)
    np.testing.assert_allclose(forces,expected,atol=8e-5,rtol=1e-4)
    np.testing.assert_allclose(batch.stress,native.stress,atol=5e-7,rtol=1e-4)
    assert abs(energies[2]-energies[0])/len(load_atoms(case))<1e-6
    error=float(np.max(abs(forces-expected)));record_property('force_error_eV_A',error)
    print('BATCH_REVERSE',precision,case,mesh,field,error)
