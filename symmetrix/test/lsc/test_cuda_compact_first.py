"""Exact compact first-layer packing, full-model derivatives and halo parity."""
import json
import numpy as np
import pytest
import symmetrix
from scipy.interpolate import CubicHermiteSpline

from .reference import DATA, load_atoms
from .test_cuda_geometry import export_file
from .test_cuda_loader import cuda_document, cuda_file, require_cuda
from .test_cuda_derivatives import inputs
from .test_forward_halo import partition, exchange

pytestmark=[pytest.mark.lsc_reference,pytest.mark.lsc_cuda]

def configure(path,precision,compact):
    model=symmetrix.MACELSCCuda(str(path),precision)
    model.use_batched_reverse=model.use_batched_source_reverse=model.use_batched_source_forward=True
    model.use_compact_first_layer=compact
    return model

@pytest.mark.parametrize('precision',['float32','float32-mixed'])
def test_compact_parameter_maps(cuda_file,cuda_document,precision):
    packed=symmetrix.prepare_lsc_cuda_parameters(str(cuda_file),precision)
    native=symmetrix.MACELSC(str(cuda_file))
    native.compute_local(np.array([0,1]),np.empty((2,0),int),np.empty((0,3)))
    initial=np.asarray(native.initial_features).reshape(2,128)
    up=initial@np.asarray(cuda_document['linear_up_0']['weights'][0]).reshape(128,128)+cuda_document['linear_up_0']['bias']
    U=np.asarray(cuda_document['radial_weights_0'])
    assert U.shape==(64,512)
    for l in range(4):
        W=np.asarray(cuda_document['linear_0']['weights'][l]).reshape(128,128)
        expected=np.concatenate([(U[:,l*128:(l+1)*128]*up[b])@W for b in range(2)])
        np.testing.assert_allclose(packed.parameter(f'compact_first.weights.{l}'),expected,atol=2e-7,rtol=2e-6)
    table=packed.parameter('compact_first.basis')
    assert table.shape==(15,4,64) and table.dtype==np.float32
    x=np.linspace(.5,6.,16);h=x[1]-x[0]
    spline=CubicHermiteSpline(x,np.asarray(cuda_document['radial_spline_values_0'][0]).T,
                             np.asarray(cuda_document['radial_spline_derivs_0'][0]).T)
    for k in [0,7,14]:
        t=.173;coeff=table[k].astype(float)
        value=coeff[0]+t*(coeff[1]+t*(coeff[2]+t*coeff[3]))
        derivative=(coeff[1]+t*(2*coeff[2]+3*t*coeff[3]))/h
        np.testing.assert_allclose(value,spline(x[k]+h*(t+.5)),atol=2e-7,rtol=2e-6)
        np.testing.assert_allclose(derivative,spline(x[k]+h*(t+.5),1),atol=2e-7,rtol=2e-6)

@pytest.mark.parametrize('precision',['float32','float32-mixed'])
@pytest.mark.parametrize('case,mesh',[('dimer',False),('bulk',False),('bulk',True)])
@pytest.mark.parametrize('field',[(0.,0.,0.),(.01,-.02,.03)])
def test_compact_full_model(export_file,lsc_model,precision,case,mesh,field,record_property):
    require_cuda();args=inputs(lsc_model,case)
    native=symmetrix.MACELSC(str(export_file))
    legacy=configure(export_file,precision,False);compact=configure(export_file,precision,True)
    assert compact.can_compact_first_layer
    for model in [native,legacy,compact]:
        if mesh:model.set_mesh_electrostatics(.5,8)
    energies=[m.compute_energy_forces(*args,external_field=field,compute_stress=True) for m in [native,legacy,compact]]
    assert compact.compact_first_forward_calls==compact.compact_first_reverse_calls==1
    assert legacy.compact_first_forward_calls==legacy.compact_first_reverse_calls==0
    for e in energies[1:]:assert abs(e-energies[0])/len(args[0])<1e-6
    actual=compact.download_forces()['forces'];expected=np.asarray(native.forces).reshape(-1,3)
    for reference in [expected,legacy.download_forces()['forces']]:
        np.testing.assert_allclose(actual,reference,atol=8e-5,rtol=1e-4)
    for model in [native,legacy]:
        np.testing.assert_allclose(compact.stress,model.stress,atol=5e-7,rtol=1e-4)
        np.testing.assert_allclose(compact.virial,model.virial,atol=1e-3,rtol=1e-4)
    old=legacy.download_local();features=compact.download_local()
    for name in ['A0','M0','H1','A1','M1','H2']:
        np.testing.assert_allclose(features[name],old[name],atol=3e-5,rtol=3e-4,err_msg=name)
    charges=compact.download_charges()
    with np.load(DATA/f'{case}.reference.npz',allow_pickle=False) as frozen:
        for key in ['density_coefficients','partial_charges','partial_dipoles']:
            np.testing.assert_allclose(charges[key],frozen[key],atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(charges['dipole'],frozen['dipole'],atol=3e-5,rtol=0)
        if not any(field):
            np.testing.assert_allclose(actual,frozen['forces'],atol=8e-5,rtol=1e-4)
            if case=='bulk':np.testing.assert_allclose(compact.stress,frozen['stress'],atol=5e-7,rtol=1e-4)
    formal=compact.download_parameter('formal_charges')[args[0]].astype(float)
    assert abs(charges['partial_charges'].sum()-formal.sum())<1e-11
    np.testing.assert_allclose(charges['layer_density_coefficients'][:,:,0].sum(1),0,atol=1e-11,rtol=0)
    record_property('force_error_eV_A',float(abs(actual-expected).max()))
    record_property('A0_legacy_error',float(abs(features['A0']-old['A0']).max()))
    print('COMPACT_FULL_MODEL',precision,case,mesh,field,float(abs(actual-expected).max()))

@pytest.fixture(scope='module')
def wider_basis(export_file,tmp_path_factory):
    # Padding a zero basis function preserves the potential but exercises the
    # generic fallback for a width outside this 64-function specialization.
    document=json.loads(export_file.read_text())
    for key in ['radial_spline_values_0','radial_spline_derivs_0']:
        document[key][0].append([0.]*len(document[key][0][0]))
    document['radial_weights_0'].append([0.]*512)
    path=tmp_path_factory.mktemp('compact-fallback')/'model.json';path.write_text(json.dumps(document))
    return path

@pytest.mark.parametrize('precision',['float32','float32-mixed'])
def test_compact_fallback(wider_basis,lsc_model,precision):
    require_cuda();args=inputs(lsc_model,'dimer')
    a=configure(wider_basis,precision,False);b=configure(wider_basis,precision,True)
    assert not b.can_compact_first_layer
    for model in [a,b]:model.compute_energy_forces(*args,compute_stress=True)
    assert b.compact_first_forward_calls==b.compact_first_reverse_calls==0
    assert a.energy==b.energy
    np.testing.assert_allclose(a.download_forces()['forces'],b.download_forces()['forces'],atol=8e-5,rtol=1e-4)

@pytest.mark.parametrize('precision',['float32','float32-mixed'])
def test_compact_empty_and_state(export_file,lsc_model,precision):
    require_cuda();model=configure(export_file,precision,True);native=symmetrix.MACELSC(str(export_file))
    edges=np.empty((2,0),int);vectors=np.empty((0,3));cell=np.eye(3)*20;pbc=np.zeros(3,bool)
    for types,positions in [(np.array([0,1]),np.array([[0.,0.,0.],[5.,6.,7.]])),(np.empty(0,int),np.empty((0,3)))]:
        args=[types,edges,vectors,positions,cell,pbc,'auto']
        for m in [native,model]:m.compute_energy_forces(*args,compute_stress=True)
        assert model.compact_first_forward_calls==int(len(types)>0) and model.compact_first_reverse_calls==0
        np.testing.assert_allclose(model.download_forces()['forces'],np.asarray(native.forces).reshape(-1,3),atol=8e-5,rtol=1e-4)
    args=inputs(lsc_model,'dimer');model.compute_energy(*args)
    model.use_compact_first_layer=False;model.evaluate_forces(compute_stress=True)
    assert model.compact_first_reverse_calls==1  # Reverse uses the cached forward route.
    model.compute_energy_forces(*args,compute_stress=True)
    assert model.compact_first_forward_calls==model.compact_first_reverse_calls==0

@pytest.mark.parametrize('precision',['float32','float32-mixed'])
def test_compact_directional_force(export_file,lsc_model,precision):
    require_cuda();model=configure(export_file,precision,True);args=inputs(lsc_model,'dimer')
    model.compute_energy_forces(*args)
    force=model.download_forces()['forces']
    direction=np.random.default_rng(193).normal(size=force.shape);direction/=np.linalg.norm(direction)
    values=[];step=1e-3
    for sign in [-1,1]:
        shifted=list(args);shifted[3]=args[3]+sign*step*direction
        shifted[2]=args[2]+sign*step*(direction[args[1][1]]-direction[args[1][0]])
        values.append(model.compute_energy(*shifted))
    assert abs((values[1]-values[0])/(2*step)+np.sum(force*direction))<3e-3

@pytest.mark.parametrize('precision',['float32','float32-mixed'])
@pytest.mark.parametrize('case',['dimer','bulk'])
def test_compact_owned_ghost_features(export_file,lsc_model,precision,case):
    require_cuda();args=inputs(lsc_model,case)
    serial=configure(export_file,precision,False);serial.compute_local(*args[:3])
    expected=serial.download_local();owners=np.arange(len(args[0]))%2
    layouts=[partition(args,owners,rank) for rank in range(2)]
    models=[configure(export_file,precision,True) for _ in layouts]
    for model,(graph,own,order) in zip(models,layouts):model.prepare_graph(graph);model.compute_first_layer()
    for layer in [1,2]:
        if layer==2:
            for model in models:model.compute_second_layer()
        exchange(models,layouts,owners,layer)
        for model,(graph,own,order) in zip(models,layouts):
            assert model.compact_first_forward_calls==1
            np.testing.assert_allclose(model.download_layer_features(layer),expected[f'H{layer}'][own],atol=2e-5,rtol=1e-4)
            np.testing.assert_allclose(model.download_forward_features(layer),expected[f'H{layer}'][order],atol=2e-5,rtol=1e-4)
    assert abs(sum(m.compute_readouts() for m in models)-serial.local_energy)<5e-5
