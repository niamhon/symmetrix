"""Small native halo tests; real MPI uses the LAMMPS callbacks separately."""
import numpy as np
import pytest
import symmetrix

from .test_staged_api import backend, saved, model, inputs

pytestmark = pytest.mark.lsc_reference


def partition(args, owners, rank):
    types, edges, vectors = args[:3]
    own = np.flatnonzero(owners == rank)
    ghost = np.flatnonzero(owners != rank)
    order = np.r_[own, ghost]
    # Extra image slots test both local copies and repeated remote copies.
    if len(own): order = np.r_[order, own[0]]
    if len(ghost): order = np.r_[order, ghost[0]]
    slot = np.empty(len(types), dtype=int)
    slot[np.r_[own, ghost]] = np.arange(len(types))
    keep = owners[edges[1]] == rank
    local_edges = slot[edges[:, keep]]
    if len(ghost): local_edges[0, edges[0, keep] == ghost[0]] = len(order)-1
    graph = symmetrix.LSCGraph(types[order], local_edges, vectors[keep], len(own),
                              np.asarray(order, dtype=np.int64)+2**40)
    return graph, own, order


def exchange(models, layouts, ownership, layer):
    for target, (graph, own, order) in enumerate(layouts):
        rows = []
        for atom in order[len(own):]:
            owner = int(ownership[atom])
            source_owned = layouts[owner][1]
            source_slot = np.flatnonzero(source_owned == atom)
            rows.append(models[owner].pack_forward_features(layer, source_slot)[0])
        values = np.asarray(rows).reshape(-1, 4*models[target].num_channels)
        models[target].unpack_forward_features(layer, len(own), values)
        models[target].complete_forward_exchange(layer)


@pytest.mark.parametrize("case", ["dimer", "bulk"])
def test_two_partitions_match_serial(saved, backend, case, record_property):
    args = inputs(saved, case)
    ownership = np.arange(len(args[0])) % 2
    layouts = [partition(args, ownership, rank) for rank in range(2)]
    from .reference import REPO
    path = str(REPO / '.lsc-reference-logs/water-stage2-energy.json')
    models = [(symmetrix.MACELSC(path) if backend == 'cpu' else symmetrix.MACELSCCuda(path, backend)) for _ in layouts]
    precision = 'float64' if backend == 'cpu' else backend
    prefix = f'{precision}/{case}/direct/zero/0'
    errors = []
    for mdl, (graph, own, order) in zip(models, layouts):
        mdl.prepare_graph(graph); mdl.compute_first_layer()
        assert not mdl.forward_exchange_complete(1)
        with pytest.raises(ValueError, match='H1 exchange is incomplete'): mdl.compute_second_layer()
    for layer in [1, 2]:
        if layer == 2:
            for mdl in models: mdl.compute_second_layer()
        exchange(models, layouts, ownership, layer)
        expected = saved[f'{prefix}/H{layer}'].reshape(len(args[0]), -1)
        for mdl, (graph, own, order) in zip(models, layouts):
            halo = mdl.download_forward_features(layer)
            assert halo.shape == (len(order), 4*mdl.num_channels)
            actual = mdl.download_layer_features(layer)
            limit = 2e-10 if backend == 'cpu' else 2e-5
            np.testing.assert_allclose(actual, expected[own], atol=limit, rtol=0 if backend == 'cpu' else 1e-4)
            np.testing.assert_allclose(halo, expected[order], atol=limit, rtol=0 if backend == 'cpu' else 1e-4)
            errors.append(float(abs(actual-expected[own]).max(initial=0)))
            # A received ghost can be forwarded again (LAMMPS multi-hop swaps).
            if len(order)>len(own):
                np.testing.assert_array_equal(mdl.pack_forward_features(layer, np.array([len(own)]))[0], halo[len(own)])
    total = 0.
    for mdl, (graph, own, order) in zip(models, layouts):
        total += mdl.compute_readouts()
        with pytest.raises((ValueError, RuntimeError), match='remote charge accumulation'):
            mdl.evaluate_charges(args[3][own])
    energy_error = abs(total-float(saved[f'{prefix}/local_energy']))
    assert energy_error < (2e-10 if backend=='cpu' else 5e-5)
    record_property('feature_max_abs_error', max(errors)); record_property('local_energy_abs_error', energy_error)
    print('FORWARD_HALO', backend, case, 'feature_error', max(errors), 'local_energy_error', energy_error)


def test_receive_validation_and_restart(model, saved, backend):
    args = inputs(saved, 'dimer'); ownership = np.arange(len(args[0])) % 2
    graph, own, order = partition(args, ownership, 0)
    model.prepare_graph(graph); model.compute_first_layer()
    width = 4*model.num_channels
    remote_slot = next(i for i in range(len(own),len(order)) if graph.local_owners[i] < 0)
    with pytest.raises(ValueError, match='unreceived ghost'): model.pack_forward_features(1,np.array([remote_slot]))
    with pytest.raises(ValueError, match='fit ghost rows'): model.unpack_forward_features(1,0,np.zeros((1,width)))
    for value in [np.nan,np.inf]:
        rows=np.zeros((1,width));rows[0,0]=value
        with pytest.raises(ValueError,match='finite'):model.unpack_forward_features(1,remote_slot,rows)
    if backend != 'cpu':
        with pytest.raises(ValueError,match='representable'):model.unpack_forward_features(1,remote_slot,np.full((1,width),1e100))
    with pytest.raises(ValueError,match='H1 exchange is incomplete'):model.complete_forward_exchange(1)
    model.unpack_forward_features(1,len(own),np.zeros((len(order)-len(own),width)))
    model.complete_forward_exchange(1);model.compute_second_layer()
    with pytest.raises(ValueError,match='H2 exchange is incomplete'):model.compute_readouts()
    with pytest.raises(ValueError,match='advancing'):model.unpack_forward_features(1,remote_slot,np.zeros((1,width)))
    model.compute_first_layer()
    assert not model.forward_exchange_complete(1)
    with pytest.raises(ValueError,match='unreceived ghost'):model.pack_forward_features(1,np.array([remote_slot]))
    # Empty owned rank with unused remote slots is valid and returns zero energy.
    empty=symmetrix.LSCGraph(np.array([0]),np.empty((2,0),int),np.empty((0,3)),0,np.array([99]))
    model.prepare_graph(empty);model.compute_first_layer();model.compute_second_layer()
    assert model.compute_readouts()==0
    assert model.download_layer_features(2).shape==(0,width)
    assert model.download_forward_features(2).shape==(1,width)

    bad=symmetrix.LSCGraph(np.array([model.num_elements]),np.empty((2,0),int),np.empty((0,3)))
    with pytest.raises(ValueError,match='species'):model.prepare_graph(bad)
    assert not model.forward_exchange_complete(1) and not model.forward_exchange_complete(2)
    with pytest.raises((ValueError,RuntimeError)):model.download_forward_features(2)
