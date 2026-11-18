"""Analytic reverse across two partitions, including actual ghost force sums."""
import numpy as np
import pytest
import symmetrix
from .reference import REPO
from .test_staged_api import backend,saved,inputs
from .test_forward_halo import partition,exchange
from .test_charge_exchange import create,reverse,source_result,exchange_potentials

pytestmark=pytest.mark.lsc_reference

def setup(args,backend,owners):
    layouts=[partition(args,owners,r) for r in range(2)];models=[create(backend) for _ in layouts]
    for m,(g,own,order) in zip(models,layouts):m.prepare_graph(g);m.compute_first_layer()
    exchange(models,layouts,owners,1)
    for m in models:m.compute_second_layer()
    exchange(models,layouts,owners,2)
    for m,(_,own,_) in zip(models,layouts):m.compute_readouts();m.compute_partial_charges(args[3][own])
    reverse(models,layouts,owners)
    rho=np.empty((len(owners),4))
    for m,(_,own,_) in zip(models,layouts):rho[own]=source_result(m,backend)['density_coefficients']
    direct=symmetrix.MACELSC(str(REPO/'.lsc-reference-logs/water-stage2-energy.json')).evaluate_direct_electrostatics(args[3],rho,args[4],args[5],'auto')
    exchange_potentials(models,layouts,owners,direct['density_gradient'])
    return models,layouts,rho,direct

def return_rows(models,layouts,owners,layer=None):
    packets=[m.pack_reverse_forces(len(own),len(order)-len(own)) if layer is None else m.pack_reverse_gradients(layer,len(own),len(order)-len(own)) for m,(_,own,order) in zip(models,layouts)]
    for (_,own,order),packet in zip(layouts,packets):
        for rank in range(2):
            select=np.array([int(owners[tag])==rank for tag in order[len(own):]],dtype=bool)
            tags=order[len(own):][select];slots=np.array([np.flatnonzero(layouts[rank][1]==tag)[0] for tag in tags],dtype=int)
            if layer is None:models[rank].unpack_reverse_forces(slots,packet[select])
            else:models[rank].unpack_reverse_gradients(layer,slots,packet[select])
    for m in models:
        if layer is None:m.complete_force_exchange()
        else:m.complete_reverse_gradient_exchange(layer)

def force_result(m,backend):
    n=m.graph_layout.num_owned
    return np.asarray(m.forces).reshape(n,3) if backend=='cpu' else m.download_forces(False)['forces']

@pytest.mark.parametrize('case',['dimer','bulk'])
@pytest.mark.parametrize('field',[(0.,0.,0.),(.01,-.02,.03)],ids=['zero','field'])
def test_reverse_two_partitions(backend,saved,case,field,record_property):
    args=inputs(saved,case);n=len(args[0]);owners=np.arange(n)%2
    models,layouts,rho,direct=setup(args,backend,owners)
    for m,(_,own,_) in zip(models,layouts):
        m.begin_distributed_reverse(direct['position_gradient'][own],np.asarray(direct['strain_derivative']).reshape(-1)/2,args[4].reshape(-1),field,float(.5*np.sum(rho[own]*direct['density_gradient'][own])))
        with pytest.raises((ValueError,RuntimeError),match='phase'):m.reverse_distributed_second_layer()
        with pytest.raises(ValueError,match='incomplete'):m.complete_reverse_gradient_exchange(2)
    return_rows(models,layouts,owners,2)
    for m in models:m.reverse_distributed_second_layer()
    return_rows(models,layouts,owners,1)
    for m in models:m.reverse_distributed_first_layer();m.finalize_distributed_forces(True);assert not m.has_force_results
    return_rows(models,layouts,owners)
    forces=np.empty((n,3))
    for m,(_,own,_) in zip(models,layouts):forces[own]=force_result(m,backend);assert m.distributed_reverse_phase==7 and m.has_force_results
    ref=create(backend);ref.compute_energy_forces(*args[:4],args[4],args[5],'auto',field,True)
    expected=force_result(ref,backend);energy=sum(m.energy for m in models);stress=sum(np.asarray(m.stress) for m in models)
    atol=2e-9 if backend=='cpu' else 8e-5;rtol=0 if backend=='cpu' else 1e-4
    np.testing.assert_allclose(forces,expected,atol=atol,rtol=rtol)
    np.testing.assert_allclose(stress,ref.stress,atol=2e-11 if backend=='cpu' else 5e-7,rtol=rtol)
    assert abs(energy-ref.energy)<(2e-9 if backend=='cpu' else 5e-5)
    np.testing.assert_allclose(forces.sum(axis=0),0,atol=2e-9 if backend=='cpu' else 3e-5,rtol=0)
    for key,value in dict(backend=backend,case=case,field=field,force_max_abs_error=float(abs(forces-expected).max()),stress_max_abs_error=float(abs(stress-ref.stress).max()),energy_abs_error=abs(energy-ref.energy)).items():record_property(key,value)
    print('REVERSE_NATIVE',backend,case,field,'F',float(abs(forces-expected).max()),'stress',float(abs(stress-ref.stress).max()))

def test_reverse_packets_and_reset(backend,saved):
    args=inputs(saved,'dimer');owners=np.arange(len(args[0]))%2;models,layouts,rho,direct=setup(args,backend,owners);m=models[0];g,own,order=layouts[0]
    with pytest.raises(ValueError):m.pack_reverse_gradients(2,len(own),1)
    m.begin_distributed_reverse(direct['position_gradient'][own],np.asarray(direct['strain_derivative']).reshape(-1)/2,args[4].reshape(-1),(0.,0.,0.),0.)
    with pytest.raises(ValueError):m.unpack_reverse_gradients(2,np.array([0]),np.full((1,512),np.nan))
    with pytest.raises(ValueError):m.pack_reverse_gradients(2,0,1)
    packet=m.pack_reverse_gradients(2,len(own),len(order)-len(own));assert np.isfinite(packet).all()
    with pytest.raises(ValueError,match='already returned'):m.pack_reverse_gradients(2,len(own),1)
    with pytest.raises(ValueError,match='after'):m.unpack_reverse_gradients(2,np.array([len(own)]),np.ones((1,512)))
    with pytest.raises((ValueError,RuntimeError),match='distributed reverse'):m.reverse_charges()
    m.set_owned_potentials(direct['density_gradient'][own]);assert m.distributed_reverse_phase==0 and not m.has_force_results and not m.has_energy_results
    with pytest.raises(ValueError):m.begin_distributed_reverse(direct['position_gradient'][own],np.zeros(9),args[4].reshape(-1),(0.,0.,0.),0.)
    m.unpack_potentials(len(own),np.zeros((len(order)-len(own),4)));m.complete_potential_exchange()
    m.begin_distributed_reverse(direct['position_gradient'][own],np.zeros(9),args[4].reshape(-1),(0.,0.,0.),0.)
    m.set_electrostatics_hook(None);assert m.distributed_reverse_phase==0 and not m.has_energy_results
    m.prepare_graph(g);assert m.distributed_reverse_phase==0
    # An unused remote slot on a zero-owned rank must remain safe through reverse.
    empty=symmetrix.LSCGraph(np.array([0]),np.empty((2,0),int),np.empty((0,3)),0,np.array([99]))
    m.prepare_graph(empty);m.compute_first_layer();m.compute_second_layer();m.compute_readouts()
    m.compute_partial_charges(np.empty((0,3)));m.complete_charge_exchange()
    m.set_owned_potentials(np.empty((0,4)));m.complete_potential_exchange()
    m.begin_distributed_reverse(np.empty((0,3)),np.zeros(9),np.eye(3).reshape(-1),(0.,0.,0.),0.)
    np.testing.assert_array_equal(m.pack_reverse_gradients(2,0,1),0)
    m.complete_reverse_gradient_exchange(2);m.reverse_distributed_second_layer()
    np.testing.assert_array_equal(m.pack_reverse_gradients(1,0,1),0)
    m.complete_reverse_gradient_exchange(1);m.reverse_distributed_first_layer();m.finalize_distributed_forces(True)
    np.testing.assert_array_equal(m.pack_reverse_forces(0,1),0);m.complete_force_exchange()
    assert force_result(m,backend).shape==(0,3) and m.has_force_results
