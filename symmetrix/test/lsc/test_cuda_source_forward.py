"""Scalar-source forward batching versus generic CUDA, native FP64 and saved ASE."""
import numpy as np
import pytest
import symmetrix

from .reference import DATA, load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import require_cuda
from .test_cuda_derivatives import inputs
from .test_cuda_source_reverse import reordered_sources

pytestmark = [pytest.mark.lsc_reference, pytest.mark.lsc_cuda]


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
@pytest.mark.parametrize('case,mesh', [('dimer', False), ('bulk', False), ('bulk', True)])
@pytest.mark.parametrize('field', [(0., 0., 0.), (.01, -.02, .03)])
def test_source_forward(export_file, lsc_model, precision, case, mesh, field, record_property):
    require_cuda()
    args = inputs(lsc_model, case)
    native = symmetrix.MACELSC(str(export_file))
    legacy = symmetrix.MACELSCCuda(str(export_file), precision)
    batch = symmetrix.MACELSCCuda(str(export_file), precision)
    assert not batch.use_batched_source_forward
    batch.use_batched_source_forward = True
    for model in [legacy, batch]:
        model.use_batched_reverse = model.use_batched_source_reverse = True
    for model in [native, legacy, batch]:
        if mesh:
            model.set_mesh_electrostatics(.5, 8)
    energies = [model.compute_energy_forces(*args, external_field=field, compute_stress=True)
                for model in [native, legacy, batch]]
    assert batch.batched_source_forward_calls == 2
    assert legacy.batched_source_forward_calls == 0
    assert energies[1] == energies[2]
    old, actual = legacy.download_charges(), batch.download_charges()
    for key in old:
        np.testing.assert_array_equal(actual[key], old[key], err_msg=key)
    forces = batch.download_forces()['forces']
    expected = np.asarray(native.forces).reshape(-1, 3)
    np.testing.assert_allclose(forces, expected, atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(batch.stress, native.stress, atol=5e-7, rtol=1e-4)
    assert abs(energies[2] - energies[0]) / len(load_atoms(case)) < 1e-6
    with np.load(DATA / f'{case}.reference.npz', allow_pickle=False) as frozen:
        for key in ['density_coefficients', 'partial_charges', 'partial_dipoles']:
            np.testing.assert_allclose(actual[key], frozen[key], atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(actual['dipole'], frozen['dipole'], atol=3e-5, rtol=0)
        if not any(field):
            np.testing.assert_allclose(forces, frozen['forces'], atol=8e-5, rtol=1e-4)
            if case == 'bulk':
                np.testing.assert_allclose(batch.stress, frozen['stress'], atol=5e-7, rtol=1e-4)
    formal = batch.download_parameter('formal_charges')[args[0]].astype(float)
    assert abs(actual['partial_charges'].sum() - formal.sum()) < 1e-11
    np.testing.assert_allclose(actual['layer_density_coefficients'][:, :, 0].sum(1), 0, atol=1e-11, rtol=0)
    # Reuse existing local features; the launch count belongs to this evaluation.
    batch.evaluate_charges(args[3])
    assert batch.batched_source_forward_calls == 2
    np.testing.assert_array_equal(batch.download_charges()['layer_edge_fluxes'], actual['layer_edge_fluxes'])
    error = float(np.max(abs(forces - expected)))
    record_property('force_error_eV_A', error)
    print('SOURCE_FORWARD', precision, case, mesh, field, error)


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
def test_source_forward_fallback(reordered_sources, lsc_model, precision):
    require_cuda()
    args = inputs(lsc_model, 'bulk')
    legacy = symmetrix.MACELSCCuda(str(reordered_sources), precision)
    requested = symmetrix.MACELSCCuda(str(reordered_sources), precision)
    requested.use_batched_source_forward = True
    for model in [legacy, requested]:
        model.set_mesh_electrostatics(.5, 8)
        model.compute_energy_forces(*args, external_field=(.01, -.02, .03), compute_stress=True)
    assert requested.batched_source_forward_calls == 0
    assert requested.energy == legacy.energy
    for key, value in legacy.download_charges().items():
        np.testing.assert_array_equal(requested.download_charges()[key], value)
    np.testing.assert_allclose(requested.download_forces()['forces'],
                               legacy.download_forces()['forces'], atol=8e-5, rtol=1e-4)
    np.testing.assert_allclose(requested.stress, legacy.stress, atol=5e-7, rtol=1e-4)


@pytest.mark.parametrize('precision', ['float32', 'float32-mixed'])
def test_source_forward_empty_and_isolated(export_file, precision):
    require_cuda()
    model = symmetrix.MACELSCCuda(str(export_file), precision)
    model.use_batched_source_forward = True
    edges, vectors = np.empty((2, 0), dtype=int), np.empty((0, 3))
    model.compute_charges(np.array([0, 1]), edges, vectors, np.array([[0., 0., 0.], [1., 2., -1.]]))
    assert model.batched_source_forward_calls == 0
    np.testing.assert_array_equal(model.download_charges()['partial_charges'], [1, -2])
    np.testing.assert_array_equal(model.download_charges()['partial_dipoles'], np.zeros((2, 3)))
    model.compute_charges(np.empty(0, dtype=int), edges, vectors, np.empty((0, 3)))
    assert model.batched_source_forward_calls == 0
    assert model.download_charges()['layer_edge_fluxes'].shape == (2, 0)
    np.testing.assert_array_equal(model.dipole, [0, 0, 0])
