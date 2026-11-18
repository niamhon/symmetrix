#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mace_lsc.hpp"
#include "mace_lsc_mesh.hpp"
#include "mace_lsc_cuda_mesh.hpp"
#include "mace_lsc_electrostatics_bindings.hpp"

namespace py=pybind11;
namespace {
struct MeshInput {
    py::array_t<double,py::array::c_style|py::array::forcecast> xyz,rho,lattice;
    std::array<double,9> box;
    std::array<bool,3> flags;
    MeshInput(py::array positions,py::array density,py::array cell,py::array pbc) {
        if(positions.ndim()!=2 || positions.shape(1)!=3 || density.ndim()!=2 ||
           density.shape(0)!=positions.shape(0) || density.shape(1)!=4)
            throw py::value_error("expected positions[N,3] and density[N,4]");
        if(cell.ndim()!=2 || cell.shape(0)!=3 || cell.shape(1)!=3 ||
           pbc.ndim()!=1 || pbc.size()!=3 || pbc.dtype().kind()!='b')
            throw py::value_error("expected cell[3,3] and boolean pbc[3]");
        xyz=lsc_binding::real_array(positions,"positions");rho=lsc_binding::real_array(density,"density");
        lattice=lsc_binding::real_array(cell,"cell");
        const auto periodic=py::array_t<bool,py::array::c_style|py::array::forcecast>::ensure(pbc);
        std::copy_n(lattice.data(),9,box.begin());std::copy_n(periodic.data(),3,flags.begin());
    }
};
} // namespace

void bind_mace_lsc_mesh(py::module_& m)
{
    m.def("lsc_cpu_mesh_compiled",&lsc_cpu_mesh_compiled);
    py::class_<LSCCPUMesh>(m,"LSCCPUMesh",
        "One-process CPU Gaussian mesh with analytic position, density and cell derivatives; no distributed MPI.")
        .def(py::init<double,int>(),py::arg("spacing")=.7,py::arg("order")=8)
        .def_readonly("spacing",&LSCCPUMesh::spacing)
        .def_readonly("order",&LSCCPUMesh::order)
        .def_property_readonly("can_evaluate_energy",[](const LSCCPUMesh&){return true;})
        .def_property_readonly("can_evaluate_forces",[](const LSCCPUMesh&){return true;})
        .def_property_readonly("can_evaluate_stress",[](const LSCCPUMesh&){return true;})
        .def("evaluate",[](LSCCPUMesh& self,const MACELSC& model,
             py::array positions,py::array density,py::array cell,py::array pbc,const std::string& mode){
            const MeshInput input(positions,density,cell,pbc);
            const auto& xyz=input.xyz;const auto& rho=input.rho;
            const auto& box=input.box;const auto& flags=input.flags;
            LSCElectrostaticsResult result;
            {py::gil_scoped_release release;
             result=model.evaluate_mesh_electrostatics_full(self,{xyz.data(),size_t(xyz.size())},
                    {rho.data(),size_t(rho.size())},box,flags,mode);}
            auto out=lsc_binding::electrostatics_result(result,size_t(positions.shape(0)));
            out["can_evaluate_forces"]=true;out["can_evaluate_stress"]=true;
            return out;
        },py::arg("model"),py::arg("positions"),py::arg("density_coefficients"),py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",
        "Evaluate energy and full dE/drho, dE/dr and dE/dstrain at fixed density. No model state changes.")
        .def("evaluate_energy",[](LSCCPUMesh& self,const MACELSC& model,
             py::array positions,py::array density,py::array cell,py::array pbc,const std::string& mode){
            const MeshInput input(positions,density,cell,pbc);
            const auto& xyz=input.xyz;const auto& rho=input.rho;
            const auto& box=input.box;const auto& flags=input.flags;
            LSCMeshEnergyResult result;
            {py::gil_scoped_release release;
             result=model.evaluate_mesh_electrostatics(self,{xyz.data(),size_t(xyz.size())},
                    {rho.data(),size_t(rho.size())},box,flags,mode);}
            py::dict out;
            out["energy"]=result.energy;out["reciprocal_energy"]=result.reciprocal_energy;
            out["self_correction"]=result.self_correction;out["grid_shape"]=result.grid_shape;
            out["density_gradient"]=lsc_binding::electrostatics_array(result.density_gradient,{positions.shape(0),4});
            out["can_evaluate_forces"]=false;out["can_evaluate_stress"]=false;
            return out;
        },py::arg("model"),py::arg("positions"),py::arg("density_coefficients"),py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",
        "Evaluate periodic Gaussian charge/dipole mesh energy and full dE/drho. Reuses FFT plans; does not change model state or install a force callback.");
    m.def("lsc_cuda_mesh_compiled",&lsc_cuda_mesh_compiled);
    py::class_<LSCCudaMesh>(m,"LSCCudaMesh",
        "One-process cuFFT mesh. Fixed-density diagnostic calls explicitly copy host inputs/results; native device calls retain GPU buffers.")
        .def(py::init<double,int>(),py::arg("spacing")=.7,py::arg("order")=8)
        .def_readonly("spacing",&LSCCudaMesh::spacing).def_readonly("order",&LSCCudaMesh::order)
        .def_readonly("device_id",&LSCCudaMesh::device_id).def_readonly("device_name",&LSCCudaMesh::device_name)
        .def_readonly("grid_shape",&LSCCudaMesh::grid_shape).def_readonly("plan_generation",&LSCCudaMesh::plan_generation)
        .def_property_readonly("can_evaluate_energy",[](const LSCCudaMesh&){return true;})
        .def_property_readonly("can_evaluate_forces",[](const LSCCudaMesh&){return true;})
        .def_property_readonly("can_evaluate_stress",[](const LSCCudaMesh&){return true;})
        .def("evaluate",[](LSCCudaMesh& self,const MACELSC& model,
             py::array positions,py::array density,py::array cell,py::array pbc,const std::string& mode){
            const MeshInput input(positions,density,cell,pbc);LSCElectrostaticsResult result;
            {py::gil_scoped_release release;
             result=self.evaluate(model,{input.xyz.data(),size_t(input.xyz.size())},
                 {input.rho.data(),size_t(input.rho.size())},input.box,input.flags,mode);}
            auto out=lsc_binding::electrostatics_result(result,size_t(positions.shape(0)));
            out["can_evaluate_forces"]=true;out["can_evaluate_stress"]=true;return out;
        },py::arg("model"),py::arg("positions"),py::arg("density_coefficients"),py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",
        "Explicit fixed-density host diagnostic for the GPU mesh energy and full analytic gradients.")
        .def("evaluate_energy",[](LSCCudaMesh& self,const MACELSC& model,
             py::array positions,py::array density,py::array cell,py::array pbc,const std::string& mode){
            const MeshInput input(positions,density,cell,pbc);LSCMeshEnergyResult result;
            {py::gil_scoped_release release;
             result=self.evaluate_energy(model,{input.xyz.data(),size_t(input.xyz.size())},
                 {input.rho.data(),size_t(input.rho.size())},input.box,input.flags,mode);}
            py::dict out;out["energy"]=result.energy;out["reciprocal_energy"]=result.reciprocal_energy;
            out["self_correction"]=result.self_correction;out["grid_shape"]=result.grid_shape;
            out["density_gradient"]=lsc_binding::electrostatics_array(result.density_gradient,{positions.shape(0),4});
            out["can_evaluate_forces"]=false;out["can_evaluate_stress"]=false;return out;
        },py::arg("model"),py::arg("positions"),py::arg("density_coefficients"),py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",
        "Fixed-density GPU energy/potential diagnostic with the CPU model's exact parameters and explicit host copies.");

}
