#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "mace_lsc_graph.hpp"
#include "mace_lsc_graph_input.hpp"

namespace lsc_binding {
template<class T> py::array_t<T> graph_array(const std::vector<T>& values)
{
    py::array_t<T> result(values.size());
    std::copy(values.begin(),values.end(),result.mutable_data());
    result.attr("setflags")(false);
    return result;
}
inline void bind_graph(py::module_& m)
{
    py::class_<LSCGraph>(m,"LSCGraph","Owned atom slots first, followed by ghost slots; physical image vectors stay on edges.")
        .def(py::init([](const py::array& types,const py::array& edges,const py::array& vectors,
                        py::object owned,py::object atom_ids){
            auto input=graph_input(types,edges,vectors);
            std::vector<std::int64_t> ids;
            if(!atom_ids.is_none()){
                auto array=py::array::ensure(atom_ids);
                if(!array || array.ndim()!=1 || array.size()!=types.size() ||
                    (array.dtype().kind()!='i' && array.dtype().kind()!='u'))
                    throw py::value_error("LSCGraph: expected atom_ids[N] with integer dtype");
                // uint64 values above INT64_MAX wrap negative and are rejected
                // by LSCGraph; tags are never narrowed to 32-bit edge indices.
                auto values=py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>::ensure(array);
                ids.assign(values.data(),values.data()+values.size());
            }
            return LSCGraph(input.types,input.senders(),input.receivers(),
                {input.vectors.data(),size_t(input.vectors.size())},owned.is_none()?-1:owned.cast<int>(),ids);
        }),py::arg("node_types"),py::arg("edge_index"),py::arg("edge_vectors"),
           py::arg("num_owned")=py::none(),py::arg("atom_ids")=py::none())
        .def_property_readonly("num_owned",&LSCGraph::num_owned)
        .def_property_readonly("num_nodes",&LSCGraph::num_nodes)
        .def_property_readonly("num_ghosts",&LSCGraph::num_ghosts)
        .def_property_readonly("num_edges",&LSCGraph::num_edges)
        .def_property_readonly("node_types",[](const LSCGraph& g){return graph_array(g.types());})
        .def_property_readonly("atom_ids",[](const LSCGraph& g){return graph_array(g.atom_ids());})
        .def_property_readonly("local_owners",[](const LSCGraph& g){return graph_array(g.local_owners());})
        .def_property_readonly("has_remote_ghosts",&LSCGraph::has_remote_ghosts)
        .def_property_readonly("edge_index",[](const LSCGraph& g){
            py::array_t<int> a({2,g.num_edges()});
            std::copy(g.senders().begin(),g.senders().end(),a.mutable_data());
            std::copy(g.receivers().begin(),g.receivers().end(),a.mutable_data()+g.num_edges());
            a.attr("setflags")(false);return a;
        })
        .def_property_readonly("edge_vectors",[](const LSCGraph& g){
            py::array_t<double> a({g.num_edges(),3});
            std::copy(g.edge_vectors().begin(),g.edge_vectors().end(),a.mutable_data());
            a.attr("setflags")(false);return a;
        });
}

template<class Model> void bind_stages(py::class_<Model>& cls)
{
    cls.def("begin_distributed_reverse",[](Model& model,const py::array& pg,const std::array<double,9>& strain,
            const std::array<double,9>& cell,const std::array<double,3>& field,double energy){
            if(pg.ndim()!=2 || pg.shape(1)!=3)throw py::value_error("expected owned position gradient[N_owned,3]");
            auto input=real_array(pg,"position gradient");py::gil_scoped_release release;
            model.begin_distributed_reverse({input.data(),size_t(input.size())},strain,cell,field,energy);
        },py::arg("position_gradient"),py::arg("strain_derivative"),py::arg("cell"),py::arg("field"),py::arg("electrostatic_energy"))
       .def_property_readonly("distributed_reverse_phase",&Model::distributed_reverse_phase)
       .def("pack_reverse_gradients",[](Model& model,int layer,int first,int count){
            if(count<0)throw py::value_error("negative reverse count");py::array_t<double> out({count,4*model.num_channels});
            {py::gil_scoped_release release;model.pack_reverse_gradients(layer,first,count,{out.mutable_data(),size_t(out.size())});}return out;
        })
       .def("unpack_reverse_gradients",[](Model& model,int layer,const py::array& slots,const py::array& values){
            if(slots.ndim()!=1 || values.ndim()!=2 || values.shape(0)!=slots.size() || values.shape(1)!=4*model.num_channels)throw py::value_error("expected slots[n], gradients[n,4*C]");
            auto ids=indices(slots,"gradient slots");auto v=real_array(values,"gradients");py::gil_scoped_release release;model.unpack_reverse_gradients(layer,ids,{v.data(),size_t(v.size())});
        })
       .def("complete_reverse_gradient_exchange",&Model::complete_reverse_gradient_exchange,py::call_guard<py::gil_scoped_release>())
       .def("reverse_distributed_second_layer",&Model::reverse_distributed_second_layer,py::call_guard<py::gil_scoped_release>())
       .def("reverse_distributed_first_layer",&Model::reverse_distributed_first_layer,py::call_guard<py::gil_scoped_release>())
       .def("finalize_distributed_forces",&Model::finalize_distributed_forces,py::arg("compute_stress")=false,py::call_guard<py::gil_scoped_release>())
       .def("pack_reverse_forces",[](Model& model,int first,int count){
            if(count<0)throw py::value_error("negative reverse count");py::array_t<double> out({count,3});
            {py::gil_scoped_release release;model.pack_reverse_forces(first,count,{out.mutable_data(),size_t(out.size())});}return out;
        })
       .def("unpack_reverse_forces",[](Model& model,const py::array& slots,const py::array& values){
            if(slots.ndim()!=1 || values.ndim()!=2 || values.shape(0)!=slots.size() || values.shape(1)!=3)throw py::value_error("expected slots[n], forces[n,3]");
            auto ids=indices(slots,"force slots");auto v=real_array(values,"forces");py::gil_scoped_release release;model.unpack_reverse_forces(ids,{v.data(),size_t(v.size())});
        })
       .def("complete_force_exchange",&Model::complete_force_exchange,py::call_guard<py::gil_scoped_release>());
    cls.def_property_readonly("charge_exchange_stage",&Model::charge_exchange_stage)
       .def("compute_partial_charges",[](Model& model,const py::array& positions){
            if(positions.ndim()!=2 || positions.shape(1)!=3)throw py::value_error("expected owned positions[N_owned,3]");
            auto input=real_array(positions,"positions");const auto count=size_t(input.size());auto data=input.data();
            py::gil_scoped_release release;model.compute_partial_charges({data,count});
        },py::arg("positions"))
       .def("pack_reverse_charges",[](Model& model,int first,int count){
            if(count<0)throw py::value_error("expected nonnegative charge row count");
            py::array_t<double> out({count,2});auto data=out.mutable_data();const auto size=size_t(out.size());
            {py::gil_scoped_release release;model.pack_reverse_charges(first,count,{data,size});}return out;
        },py::arg("first"),py::arg("count"))
       .def("unpack_reverse_charges",[](Model& model,const py::array& slots,const py::array& values){
            if(slots.ndim()!=1 || values.ndim()!=2 || values.shape(0)!=slots.size() || values.shape(1)!=2)
                throw py::value_error("expected receive slots[n] and layer charges[n,2]");
            auto indices=lsc_binding::indices(slots,"charge slots");auto input=real_array(values,"charges");
            const auto count=size_t(input.size());auto data=input.data();py::gil_scoped_release release;
            model.unpack_reverse_charges(indices,{data,count});
        },py::arg("slots"),py::arg("values"))
       .def("complete_charge_exchange",&Model::complete_charge_exchange,py::call_guard<py::gil_scoped_release>())
       .def("set_owned_potentials",[](Model& model,const py::array& values){
            if(values.ndim()!=2 || values.shape(1)!=4)throw py::value_error("expected owned potentials[N_owned,4]");
            auto input=real_array(values,"potentials");const auto count=size_t(input.size());auto data=input.data();
            py::gil_scoped_release release;model.set_owned_potentials({data,count});
        },py::arg("values"))
       .def("pack_potentials",[](const Model& model,const py::array& slots){
            if(slots.ndim()!=1)throw py::value_error("expected send slots[n]");auto indices=lsc_binding::indices(slots,"potential slots");
            py::array_t<double> out({py::ssize_t(indices.size()),py::ssize_t(4)});auto data=out.mutable_data();const auto count=size_t(out.size());
            {py::gil_scoped_release release;model.pack_potentials(indices,{data,count});}return out;
        },py::arg("slots"))
       .def("unpack_potentials",[](Model& model,int first,const py::array& values){
            if(values.ndim()!=2 || values.shape(1)!=4)throw py::value_error("expected received potentials[n,4]");
            auto input=real_array(values,"potentials");const auto count=size_t(input.size());auto data=input.data();
            py::gil_scoped_release release;model.unpack_potentials(first,{data,count});
        },py::arg("first"),py::arg("values"))
       .def("complete_potential_exchange",&Model::complete_potential_exchange,py::call_guard<py::gil_scoped_release>())
       .def("potential_exchange_complete",&Model::potential_exchange_complete);
    cls.def("pack_forward_features",[](const Model& model,int layer,const py::array& slots){
            if(slots.ndim()!=1)throw py::value_error("expected send slots[n]");
            auto indices=lsc_binding::indices(slots,"feature slots");
            py::array_t<double> out({py::ssize_t(indices.size()),py::ssize_t(4*model.num_channels)});
            auto data=out.mutable_data();const auto count=size_t(out.size());
            {py::gil_scoped_release release;model.pack_forward_features(layer,indices,{data,count});}
            return out;
        },py::arg("layer"),py::arg("slots"))
       .def("unpack_forward_features",[](Model& model,int layer,int first,const py::array& values){
            if(values.ndim()!=2 || values.shape(1)!=4*model.num_channels)throw py::value_error("expected feature rows[n,4*C]");
            auto input=lsc_binding::real_array(values,"features");
            py::gil_scoped_release release;model.unpack_forward_features(layer,first,{input.data(),size_t(input.size())});
        },py::arg("layer"),py::arg("first"),py::arg("values"))
       .def("complete_forward_exchange",&Model::complete_forward_exchange,py::arg("layer"),py::call_guard<py::gil_scoped_release>())
       .def("forward_exchange_complete",&Model::forward_exchange_complete,py::arg("layer"))
       .def("prepare_graph",&Model::prepare_graph,py::arg("graph"),py::call_guard<py::gil_scoped_release>())
       .def_property_readonly("graph_layout",[](const Model& model){return LSCGraph(model.graph_layout());})
       .def_property_readonly("evaluation_stage",&Model::evaluation_stage)
       .def("compute_first_layer",&Model::compute_first_layer,py::call_guard<py::gil_scoped_release>())
       .def("compute_second_layer",&Model::compute_second_layer,py::call_guard<py::gil_scoped_release>())
       .def("compute_readouts",&Model::compute_readouts,py::call_guard<py::gil_scoped_release>())
       .def("reverse_charges",&Model::reverse_charges,py::call_guard<py::gil_scoped_release>())
       .def("reverse_second_layer",&Model::reverse_second_layer,py::call_guard<py::gil_scoped_release>())
       .def("reverse_first_layer",&Model::reverse_first_layer,py::call_guard<py::gil_scoped_release>())
       .def("finalize_forces",&Model::finalize_forces,py::arg("compute_stress")=false,
            py::call_guard<py::gil_scoped_release>());
}
} // namespace lsc_binding
