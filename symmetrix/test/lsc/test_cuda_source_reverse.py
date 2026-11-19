"""Receiver-batched LSC transfer reverse versus the original path and ASE."""
import numpy as np
import pytest
import symmetrix
import json

from .reference import DATA, load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import require_cuda
from .test_cuda_derivatives import inputs

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda]


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
@pytest.mark.parametrize('case,mesh', [('dimer', False), ('bulk', False), ('bulk', True)])
@pytest.mark.parametrize('field', [(0., 0., 0.), (.01, -.02, .03)])
def test_source_reverse(export_file, lsc_model, precision, case, mesh, field, record_property):
    require_cuda()
    args = inputs(lsc_model, case)
    native = symmetrix.MACELSC(str(export_file))
    legacy = symmetrix.MACELSCCuda(str(export_file), precision)
    batch = symmetrix.MACELSCCuda(str(export_file), precision)
    assert not batch.use_batched_source_reverse
    batch.use_batched_source_reverse = True
    # Hold the existing local-convolution optimization fixed on both paths.
    legacy.use_batched_reverse = batch.use_batched_reverse = True
    for model in [native, legacy, batch]:
        if mesh:
            model.set_mesh_electrostatics(.5, 8)
    energies = [model.compute_energy_forces(*args, external_field=field, compute_stress=True)
                for model in [native, legacy, batch]]
    assert batch.batched_source_reverse_calls == 2
    assert legacy.batched_source_reverse_calls == 0
    assert energies[1] == energies[2]
    np.testing.assert_array_equal(legacy.dipole, batch.dipole)
    np.testing.assert_array_equal(legacy.download_charges()['partial_charges'],
                                  batch.download_charges()['partial_charges'])
    forces = batch.download_forces()['forces']
    expected = np.asarray(native.forces).reshape(-1, 3)
    for reference in [expected, legacy.download_forces()['forces']]:
        np.testing.assert_allclose(forces, reference, atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(batch.stress, native.stress, atol=5e-7, rtol=1e-4)
    assert abs(energies[2] - energies[0]) / len(load_atoms(case)) < 1e-6
    if not any(field):
        with np.load(DATA / f'{case}.reference.npz', allow_pickle=False) as frozen:
            np.testing.assert_allclose(forces, frozen['forces'], atol=8e-5, rtol=1e-4)
            if case == 'bulk':
                np.testing.assert_allclose(batch.stress, frozen['stress'], atol=5e-7, rtol=1e-4)
    error = float(np.max(abs(forces - expected)))
    record_property('force_error_eV_A', error)
    print('SOURCE_REVERSE', precision, case, mesh, field, error)
    batch.compute_energy_forces(*args, external_field=field, compute_stress=True)
    assert batch.batched_source_reverse_calls == 2  # Reset per evaluation.


@pytest.fixture(scope='module')
def reordered_sources(export_file, tmp_path_factory):
    document = json.loads(export_file.read_text())
    for source in document['sources']:
        for key in ['spec', 'coefficients']:
            source['conv_tp'][key].reverse()
    path = tmp_path_factory.mktemp('reordered-source') / 'model.json'
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
def test_source_reverse_fallback(reordered_sources, lsc_model, precision):
    # Reordering terms preserves the potential but does not match the specialized
    # instruction ordering. The original generic reverse must remain available.
    require_cuda()
    args = inputs(lsc_model, 'bulk')
    legacy = symmetrix.MACELSCCuda(str(reordered_sources), precision)
    requested = symmetrix.MACELSCCuda(str(reordered_sources), precision)
    requested.use_batched_source_reverse = True
    for model in [legacy, requested]:
        model.set_mesh_electrostatics(.5, 8)
        model.compute_energy_forces(*args, external_field=(.01, -.02, .03), compute_stress=True)
    assert requested.batched_source_reverse_calls == 0
    assert requested.energy == legacy.energy
    np.testing.assert_allclose(requested.download_forces()['forces'],
                               legacy.download_forces()['forces'], atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(requested.stress, legacy.stress, atol=5e-7, rtol=1e-4)
