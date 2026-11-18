"""Partial-source conservation, additive returns and full density potentials."""
import numpy as np
import pytest
import symmetrix
from .reference import REPO
from .test_staged_api import backend, saved, model, inputs
from .test_forward_halo import partition, exchange

pytestmark=pytest.mark.lsc_reference

def create(backend):
    path=str(REPO/'.lsc-reference-logs/water-stage2-energy.json')
    return symmetrix.MACELSC(path) if backend=='cpu' else symmetrix.MACELSCCuda(path,backend)

def source_result(model,backend):
    if backend=='cpu':
        n=model.graph_layout.num_owned
        return dict(density_coefficients=np.asarray(model.density_coefficients).reshape(n,4),
                    layer_density_coefficients=np.asarray(model.layer_density_coefficients).reshape(2,n,4),
                    partial_charges=np.asarray(model.partial_charges),partial_dipoles=np.asarray(model.partial_dipoles).reshape(n,3))
    return model.download_charges()

def reverse(models,layouts,owners):
    # Return every image slot once. Repeated destination indices must add.
    packets=[m.pack_reverse_charges(len(own),len(order)-len(own)) for m,(_,own,order) in zip(models,layouts)]
    for (graph,own,order),packet in zip(layouts,packets):
        for rank in range(len(models)):
            select=np.array([int(owners[tag])==rank for tag in order[len(own):]])
            tags=order[len(own):][select]
            slots=np.array([np.flatnonzero(layouts[rank][1]==tag)[0] for tag in tags],dtype=int)
            models[rank].unpack_reverse_charges(slots,packet[select])
    for m in models:m.complete_charge_exchange()

def exchange_potentials(models,layouts,owners,phi):
    for m,(_,own,order) in zip(models,layouts):m.set_owned_potentials(phi[own])
    for target,(_,own,order) in enumerate(layouts):
        rows=[]
        for tag in order[len(own):]:
            rank=int(owners[tag]);slot=np.flatnonzero(layouts[rank][1]==tag)
            rows.append(models[rank].pack_potentials(slot)[0])
        models[target].unpack_potentials(len(own),np.asarray(rows).reshape(-1,4))
        models[target].complete_potential_exchange()

@pytest.mark.parametrize('case',['dimer','bulk'])
def test_two_partitions_charge_and_potential(backend,saved,case,record_property):
    args=inputs(saved,case);count=len(args[0]);owners=np.arange(count)%2
    layouts=[partition(args,owners,r) for r in range(2)];models=[create(backend) for _ in layouts]
    for m,(g,own,order) in zip(models,layouts):m.prepare_graph(g);m.compute_first_layer()
    exchange(models,layouts,owners,1)
    for m in models:m.compute_second_layer()
    exchange(models,layouts,owners,2)
    for m,(_,own,order) in zip(models,layouts):m.compute_readouts();m.compute_partial_charges(args[3][own])
    partial_sum=sum(m.download_partial_charges().sum(axis=0) for m in models)
    np.testing.assert_allclose(partial_sum,0,atol=1e-11,rtol=0)
    reverse(models,layouts,owners)
    rho=np.zeros((count,4));layer=np.zeros((2,count,4));dipole=np.zeros(3)
    for m,(_,own,order) in zip(models,layouts):
        result=source_result(m,backend);rho[own]=result['density_coefficients'];layer[:,own]=result['layer_density_coefficients'];dipole+=m.dipole
        assert m.has_charge_results and m.charge_exchange_stage=='assembled_charges' and not m.has_energy_results
        with pytest.raises((ValueError,RuntimeError),match='distributed electrostatic energy'):m.evaluate_energy(args[4],args[5],'auto')
    precision='float64' if backend=='cpu' else backend;prefix=f'{precision}/{case}/direct/zero/0'
    expected=saved[f'{prefix}/density_coefficients'].reshape(count,4)
    np.testing.assert_allclose(rho,expected,atol=2e-10 if backend=='cpu' else 2e-6,rtol=0 if backend=='cpu' else 1e-5)
    np.testing.assert_allclose(layer[:,:,0].sum(axis=1),0,atol=1e-11,rtol=0)
    np.testing.assert_allclose(rho[:,0].sum(),0,atol=1e-11,rtol=0)
    np.testing.assert_allclose(dipole,saved[f'{prefix}/dipole'],atol=2e-9 if backend=='cpu' else 3e-5,rtol=0)
    control=symmetrix.MACELSC(str(REPO/'.lsc-reference-logs/water-stage2-energy.json'))
    direct=control.evaluate_direct_electrostatics(args[3],rho,args[4],args[5],'auto')
    phi=direct['density_gradient']
    assert abs(.5*np.sum(rho*phi)-direct['energy'])<1e-10
    exchange_potentials(models,layouts,owners,phi)
    for m,(_,own,order) in zip(models,layouts):
        assert m.potential_exchange_complete() and m.charge_exchange_stage=='potentials_complete'
        np.testing.assert_array_equal(m.download_electrostatic_potentials(),phi[order])
    record_property('backend',backend);record_property('case',case)
    record_property('density_max_abs_error',float(abs(rho-expected).max()))
    record_property('dipole_max_abs_error',float(abs(dipole-saved[f'{prefix}/dipole']).max()))
    print('CHARGE_EXCHANGE',backend,case,'rho_error',float(abs(rho-expected).max()),'net_charge',rho[:,0].sum())

def test_charge_packet_and_potential_state(model,backend,saved):
    args=inputs(saved,'dimer');owners=np.arange(len(args[0]))%2
    graph,own,order=partition(args,owners,0);precision='float64' if backend=='cpu' else backend
    with pytest.raises((ValueError,RuntimeError)):model.compute_partial_charges(args[3])
    model.prepare_graph(graph);model.compute_first_layer()
    h1=saved[f'{precision}/dimer/direct/zero/0/H1'].reshape(len(args[0]),-1)
    model.unpack_forward_features(1,len(own),h1[order[len(own):]]);model.compute_second_layer()
    h2=saved[f'{precision}/dimer/direct/zero/0/H2'].reshape(len(args[0]),-1)
    model.unpack_forward_features(2,len(own),h2[order[len(own):]]);model.compute_readouts()
    model.compute_partial_charges(args[3][own]);assert not model.has_charge_results
    with pytest.raises(ValueError,match='incomplete'):model.complete_charge_exchange()
    with pytest.raises(ValueError):model.set_owned_potentials(np.zeros((len(own),4)))
    with pytest.raises(ValueError):model.pack_reverse_charges(0,1)
    with pytest.raises(ValueError,match='finite'):model.unpack_reverse_charges(np.array([0]),np.full((1,2),np.nan))
    before=model.download_partial_charges()[0].copy()
    model.unpack_reverse_charges(np.array([0,0]),np.array([[.1,.2],[.3,.4]]))
    np.testing.assert_allclose(model.download_partial_charges()[0],before+[.4,.6],atol=1e-12,rtol=0)
    model.pack_reverse_charges(len(own),len(order)-len(own))
    with pytest.raises(ValueError,match='already returned'):model.pack_reverse_charges(len(own),1)
    with pytest.raises(ValueError,match='after'):model.unpack_reverse_charges(np.array([len(own)]),np.ones((1,2)))
    model.complete_charge_exchange()
    with pytest.raises(ValueError):model.complete_charge_exchange()
    with pytest.raises(ValueError,match='finite'):model.set_owned_potentials(np.full((len(own),4),np.inf))
    values=np.arange(len(own)*4,dtype=float).reshape(-1,4)
    model.set_owned_potentials(values)
    with pytest.raises(ValueError,match='incomplete'):model.complete_potential_exchange()
    with pytest.raises(ValueError):model.unpack_potentials(0,np.zeros((1,4)))
    with pytest.raises(ValueError,match='finite'):model.unpack_potentials(len(own),np.full((1,4),np.nan))
    model.unpack_potentials(len(own),np.ones((len(order)-len(own),4)))
    model.complete_potential_exchange();assert model.potential_exchange_complete()
    model.compute_first_layer();assert model.charge_exchange_stage=='empty' and not model.potential_exchange_complete()
    with pytest.raises(ValueError):model.download_electrostatic_potentials()
    empty=symmetrix.LSCGraph(np.array([0]),np.empty((2,0),int),np.empty((0,3)),0,np.array([99]))
    model.prepare_graph(empty);model.compute_first_layer();model.compute_second_layer();model.compute_readouts()
    model.compute_partial_charges(np.empty((0,3)));model.complete_charge_exchange()
    model.set_owned_potentials(np.empty((0,4)));model.complete_potential_exchange()
    assert model.potential_exchange_complete() and source_result(model,backend)['density_coefficients'].shape==(0,4)
