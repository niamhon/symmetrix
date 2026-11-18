#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "mace_lsc_cuda.hpp"
#include "mace_lsc_electrostatics_bindings.hpp"
#include "mace_lsc_graph_input.hpp"
#include "mace_lsc_stages_bindings.hpp"

namespace py=pybind11;
namespace {
py::array numpy(const LSCCudaArray& array)
{
    std::vector<py::ssize_t> shape(array.shape.begin(),array.shape.end());
    py::array out(py::dtype(array.dtype),shape);
    if(array.bytes()>0)std::memcpy(out.mutable_data(),array.data(),array.bytes());
    return out;
}
py::dict arrays(const std::map<std::string,LSCCudaArray>& input)
{
    py::dict out;for(const auto& [name,array]:input)out[py::str(name)]=numpy(array);return out;
}
py::dict manifest(const std::map<std::string,LSCCudaArray>& arrays)
{
    py::dict out;
    for(const auto& [name,array]:arrays){
        py::dict info;info["dtype"]=array.dtype;info["shape"]=array.shape;info["bytes"]=array.bytes();out[py::str(name)]=info;
    }
    return out;
}
struct BoxInput {std::array<double,9> cell{};std::array<bool,3> pbc{};};
BoxInput box_input(const py::array& cell,const py::array& pbc)
{
    if(cell.ndim()!=2 or cell.shape(0)!=3 or cell.shape(1)!=3)throw std::invalid_argument("MACELSCCuda: expected cell[3,3]");
    if(pbc.ndim()!=1 or pbc.size()!=3 or pbc.dtype().kind()!='b')throw std::invalid_argument("MACELSCCuda: expected pbc[3] with boolean dtype");
    auto lattice=lsc_binding::real_array(cell,"cell");
    auto periodic=py::array_t<bool,py::array::c_style|py::array::forcecast>::ensure(pbc);
    BoxInput result;std::copy_n(lattice.data(),9,result.cell.begin());std::copy_n(periodic.data(),3,result.pbc.begin());return result;
}
} // namespace

void bind_mace_lsc_cuda(py::module_& m)
{
    m.def("lsc_cuda_compiled",&lsc_cuda_compiled);
    m.def("lsc_cuda_available",&lsc_cuda_available);
    py::class_<LSCCudaParameters>(m,"LSCCudaParameters","Validated host packing for the LSC CUDA kernels.")
        .def_readonly("precision",&LSCCudaParameters::precision)
        .def_readonly("num_elements",&LSCCudaParameters::num_elements)
        .def_readonly("num_channels",&LSCCudaParameters::num_channels)
        .def_readonly("num_interactions",&LSCCudaParameters::num_interactions)
        .def_readonly("num_knots",&LSCCudaParameters::num_knots)
        .def_readonly("r_cut",&LSCCudaParameters::r_cut)
        .def_readonly("r_min",&LSCCudaParameters::r_min)
        .def_readonly("atomic_numbers",&LSCCudaParameters::atomic_numbers)
        .def_property_readonly("parameter_bytes",&LSCCudaParameters::bytes)
        .def_property_readonly("manifest",[](const LSCCudaParameters& p){return manifest(p.arrays);})
        .def("parameter",[](const LSCCudaParameters& p,const std::string& name){return numpy(p.arrays.at(name));},py::arg("name"));
    m.def("prepare_lsc_cuda_parameters",[](const std::string& filename,const std::string& precision){
        py::gil_scoped_release release;return prepare_lsc_cuda_parameters(filename,precision);
    },py::arg("filename"),py::arg("precision")="float32");
    auto cls=py::class_<MACELSCCuda>(m,"MACELSCCuda","Native LSC CUDA energy, charge transfers, dipoles, analytic forces and stress.");
    lsc_binding::bind_stages(cls);
    cls.def("download_partial_charges",[](const MACELSCCuda& self){LSCCudaArray out;
        {py::gil_scoped_release release;out=self.download_partial_charges();}return numpy(out);})
       .def("download_electrostatic_potentials",[](const MACELSCCuda& self){LSCCudaArray out;
        {py::gil_scoped_release release;out=self.download_electrostatic_potentials();}return numpy(out);});
    cls.def("download_forward_features",[](const MACELSCCuda& self,int layer){
        LSCCudaArray out;{py::gil_scoped_release release;out=self.download_forward_features(layer);}return numpy(out);
    },py::arg("layer"));
    cls.def("download_layer_features",[](MACELSCCuda& self,int layer,bool gradients){
        LSCCudaArray out;{py::gil_scoped_release release;out=self.download_layer_features(layer,gradients);}
        return numpy(out);
    },py::arg("layer"),py::arg("gradients")=false);
    cls.def(py::init([](const std::string& filename,const std::string& precision){
            py::gil_scoped_release release;return std::make_unique<MACELSCCuda>(filename,precision);
        }),py::arg("filename"),py::arg("precision")="float32")
        .def_readwrite("use_batched_reverse",&MACELSCCuda::use_batched_reverse)
        .def_readonly("batched_reverse_calls",&MACELSCCuda::batched_reverse_calls)
        .def_readonly("precision",&MACELSCCuda::precision)
        .def_readonly("float32_mixed",&MACELSCCuda::float32_mixed)
        .def_readonly("device_name",&MACELSCCuda::device_name)
        .def_readonly("device_id",&MACELSCCuda::device_id)
        .def_readonly("compute_capability_major",&MACELSCCuda::compute_capability_major)
        .def_readonly("compute_capability_minor",&MACELSCCuda::compute_capability_minor)
        .def_readonly("tensor_cores",&MACELSCCuda::tensor_cores)
        .def_readonly("num_elements",&MACELSCCuda::num_elements)
        .def_readonly("num_channels",&MACELSCCuda::num_channels)
        .def_readonly("num_interactions",&MACELSCCuda::num_interactions)
        .def_readonly("num_knots",&MACELSCCuda::num_knots)
        .def_readonly("r_cut",&MACELSCCuda::r_cut)
        .def_readonly("r_min",&MACELSCCuda::r_min)
        .def_readonly("atomic_numbers",&MACELSCCuda::atomic_numbers)
        .def_readonly("parameters_on_device",&MACELSCCuda::parameters_on_device)
        .def_readonly("device_parameter_bytes",&MACELSCCuda::device_parameter_bytes)
        .def_property_readonly("can_evaluate_forces",&MACELSCCuda::can_evaluate_forces)
        .def_property_readonly("can_evaluate_stress",&MACELSCCuda::can_evaluate_stress)
        .def_readonly("has_force_results",&MACELSCCuda::has_force_results)
        .def_readonly("has_stress_results",&MACELSCCuda::has_stress_results)
        .def_readonly("stress",&MACELSCCuda::stress)
        .def_readonly("virial",&MACELSCCuda::virial)
        .def_readonly("strain_derivative",&MACELSCCuda::strain_derivative)
        .def("compute_energy_forces",[](MACELSCCuda& self,const py::array& types,const py::array& edges,const py::array& vectors,
             const py::array& positions,const py::array& cell,const py::array& pbc,const std::string& mode,const std::array<double,3>& field,bool stress){
            auto graph=lsc_binding::graph_input(types,edges,vectors);const auto box=box_input(cell,pbc);
            if(positions.ndim()!=2 or positions.shape(0)!=types.size() or positions.shape(1)!=3)throw std::invalid_argument("MACELSCCuda: expected positions[N,3]");
            auto xyz=lsc_binding::real_array(positions,"positions");py::gil_scoped_release release;
            return self.compute_energy_forces(graph.types,graph.senders(),graph.receivers(),{graph.vectors.data(),size_t(graph.vectors.size())},
                {xyz.data(),size_t(xyz.size())},box.cell,box.pbc,mode,field,stress);
        },py::arg("node_types"),py::arg("edge_index"),py::arg("edge_vectors"),py::arg("positions"),py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",py::arg("external_field")=std::array<double,3>{},py::arg("compute_stress")=false)
        .def("evaluate_forces",&MACELSCCuda::evaluate_forces,py::arg("compute_stress")=false,py::call_guard<py::gil_scoped_release>())
        .def("download_forces",[](const MACELSCCuda& self,bool diagnostics){
            std::map<std::string,LSCCudaArray> out;{py::gil_scoped_release release;out=self.download_forces(diagnostics);}return arrays(out);
        },py::arg("diagnostics")=true)
        .def_property_readonly("can_evaluate_energy",&MACELSCCuda::can_evaluate_energy)
        .def_readonly("has_energy_results",&MACELSCCuda::has_energy_results)
        .def_property_readonly("has_electrostatics_hook",&MACELSCCuda::has_electrostatics_hook)
        .def("set_mesh_electrostatics",&MACELSCCuda::set_mesh_electrostatics,py::arg("spacing")=.7,py::arg("order")=8,
             "Use the native one-process GPU mesh and analytic derivatives; requires full periodicity. None hook restores direct.")
        .def("set_electrostatics_hook",[](MACELSCCuda& self,const py::object& callback){
            self.set_electrostatics_host_hook(lsc_binding::electrostatics_python_hook(callback));
        },py::arg("callback"),
        "Python host electrostatics callback for checking custom solvers; copies inputs/results. Native C++ device hooks keep buffers on the GPU. None restores direct.")
        .def_readonly("energy",&MACELSCCuda::energy)
        .def_readonly("electrostatic_energy",&MACELSCCuda::electrostatic_energy)
        .def_readonly("field_energy",&MACELSCCuda::field_energy)
        .def("compute_energy",[](MACELSCCuda& self,const py::array& types,const py::array& edges,const py::array& vectors,
             const py::array& positions,const py::array& cell,const py::array& pbc,const std::string& mode,const std::array<double,3>& field){
            auto graph=lsc_binding::graph_input(types,edges,vectors);const auto box=box_input(cell,pbc);
            if(positions.ndim()!=2 or positions.shape(0)!=types.size() or positions.shape(1)!=3)throw std::invalid_argument("MACELSCCuda: expected positions[N,3]");
            auto xyz=lsc_binding::real_array(positions,"positions");py::gil_scoped_release release;
            return self.compute_energy(graph.types,graph.senders(),graph.receivers(),{graph.vectors.data(),size_t(graph.vectors.size())},
                {xyz.data(),size_t(xyz.size())},box.cell,box.pbc,mode,field);
        },py::arg("node_types"),py::arg("edge_index"),py::arg("edge_vectors"),py::arg("positions"),py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",py::arg("external_field")=std::array<double,3>{})
        .def("evaluate_energy",[](MACELSCCuda& self,const py::array& cell,const py::array& pbc,const std::string& mode,const std::array<double,3>& field){
            const auto box=box_input(cell,pbc);py::gil_scoped_release release;return self.evaluate_energy(box.cell,box.pbc,mode,field);
        },py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",py::arg("external_field")=std::array<double,3>{})
        .def_property_readonly("can_evaluate_sources",&MACELSCCuda::can_evaluate_sources)
        .def_readonly("has_charge_results",&MACELSCCuda::has_charge_results)
        .def_readonly("dipole",&MACELSCCuda::dipole)
        .def_readonly("formal_charge_dipole",&MACELSCCuda::formal_charge_dipole)
        .def_readonly("transfer_dipole",&MACELSCCuda::transfer_dipole)
        .def_readonly("atomic_dipole_sum",&MACELSCCuda::atomic_dipole_sum)
        .def_readonly("source_sgemm_calls",&MACELSCCuda::source_sgemm_calls)
        .def_readonly("source_tensor_core_gemm_calls",&MACELSCCuda::source_tensor_core_gemm_calls)
        .def("compute_charges",[](MACELSCCuda& self,const py::array& types,const py::array& edges,
                                  const py::array& vectors,const py::array& positions){
            auto graph=lsc_binding::graph_input(types,edges,vectors);
            if(positions.ndim()!=2 or positions.shape(0)!=types.size() or positions.shape(1)!=3)
                throw std::invalid_argument("MACELSCCuda: expected positions[N,3]");
            auto xyz=lsc_binding::real_array(positions,"positions");
            py::gil_scoped_release release;
            return self.compute_charges(graph.types,graph.senders(),graph.receivers(),
                {graph.vectors.data(),size_t(graph.vectors.size())},{xyz.data(),size_t(xyz.size())});
        },py::arg("node_types"),py::arg("edge_index"),py::arg("edge_vectors"),py::arg("positions"))
        .def("evaluate_charges",[](MACELSCCuda& self,const py::array& positions){
            if(positions.ndim()!=2 or positions.shape(0)!=self.num_nodes or positions.shape(1)!=3)
                throw std::invalid_argument("MACELSCCuda: expected positions[N,3]");
            auto xyz=lsc_binding::real_array(positions,"positions");
            py::gil_scoped_release release;return self.evaluate_charges({xyz.data(),size_t(xyz.size())});
        },py::arg("positions"))
        .def("download_charges",[](const MACELSCCuda& self,bool diagnostics){
            std::map<std::string,LSCCudaArray> out;{py::gil_scoped_release release;out=self.download_charges(diagnostics);}return arrays(out);
        },py::arg("diagnostics")=true)
        .def_property_readonly("can_evaluate_local",&MACELSCCuda::can_evaluate_local)
        .def_readonly("has_local_results",&MACELSCCuda::has_local_results)
        .def_readonly("local_energy",&MACELSCCuda::local_energy)
        .def_property_readonly("sgemm_calls",&MACELSCCuda::sgemm_calls)
        .def_property_readonly("tensor_core_gemm_calls",&MACELSCCuda::tensor_core_gemm_calls)
        .def("compute_local",[](MACELSCCuda& self,const py::array& types,const py::array& edges,const py::array& vectors){
            auto graph=lsc_binding::graph_input(types,edges,vectors);
            py::gil_scoped_release release;
            return self.compute_local(graph.types,graph.senders(),graph.receivers(),
                                      {graph.vectors.data(),size_t(graph.vectors.size())});
        },py::arg("node_types"),py::arg("edge_index"),py::arg("edge_vectors"))
        .def("evaluate_local",&MACELSCCuda::evaluate_local,py::call_guard<py::gil_scoped_release>())
        .def("download_local",[](const MACELSCCuda& self,bool diagnostics){
            std::map<std::string,LSCCudaArray> out;{py::gil_scoped_release release;out=self.download_local(diagnostics);}return arrays(out);
        },py::arg("diagnostics")=true)
        .def_readonly("num_nodes",&MACELSCCuda::num_nodes)
        .def_readonly("num_edges",&MACELSCCuda::num_edges)
        .def_readonly("has_geometry",&MACELSCCuda::has_geometry)
        .def_readonly("has_spline_results",&MACELSCCuda::has_spline_results)
        .def_readonly("spline_name",&MACELSCCuda::spline_name)
        .def_property_readonly("workspace_bytes",&MACELSCCuda::workspace_bytes)
        .def("compute_geometry",[](MACELSCCuda& self,const py::array& types,const py::array& edges,const py::array& vectors){
            auto graph=lsc_binding::graph_input(types,edges,vectors);
            py::gil_scoped_release release;
            self.compute_geometry(graph.types,graph.senders(),graph.receivers(),
                                  {graph.vectors.data(),size_t(graph.vectors.size())});
        },py::arg("node_types"),py::arg("edge_index"),py::arg("edge_vectors"))
        .def("evaluate_splines",&MACELSCCuda::evaluate_splines,py::arg("name"),py::call_guard<py::gil_scoped_release>())
        .def("download_geometry",[](const MACELSCCuda& self){
            std::map<std::string,LSCCudaArray> out;{py::gil_scoped_release release;out=self.download_geometry();}return arrays(out);
        })
        .def("download_splines",[](const MACELSCCuda& self){
            std::map<std::string,LSCCudaArray> out;{py::gil_scoped_release release;out=self.download_splines();}return arrays(out);
        })
        .def_property_readonly("can_evaluate",&MACELSCCuda::can_evaluate)
        .def_property_readonly("manifest",[](const MACELSCCuda& p){return manifest(p.parameter_metadata());})
        .def("download_parameter",[](const MACELSCCuda& p,const std::string& name){
            LSCCudaArray out;{py::gil_scoped_release release;out=p.download_parameter(name);}return numpy(out);
        },py::arg("name"));
}
