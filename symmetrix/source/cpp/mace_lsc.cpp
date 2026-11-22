#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "mace_lsc.hpp"
#include "mace_lsc_electrostatics_bindings.hpp"
#include "mace_lsc_graph_input.hpp"
#include "mace_lsc_stages_bindings.hpp"

namespace py = pybind11;

namespace {
using namespace lsc_binding;

struct EnergyInput {
    GraphInput graph;
    RealArray positions;
    std::array<double, 9> cell;
    std::array<bool, 3> pbc;
};

EnergyInput energy_input(const py::array& node_types, const py::array& edge_index,
                        const py::array& edge_vectors, const py::array& positions,
                        const py::array& cell, const py::array& pbc)
{
    auto graph = graph_input(node_types, edge_index, edge_vectors);
    if (positions.ndim() != 2 or positions.shape(0) != node_types.size() or positions.shape(1) != 3)
        throw std::invalid_argument("MACELSC: expected positions[N,3]");
    if (cell.ndim() != 2 or cell.shape(0) != 3 or cell.shape(1) != 3)
        throw std::invalid_argument("MACELSC: expected cell[3,3]");
    if (pbc.ndim() != 1 or pbc.size() != 3 or pbc.dtype().kind() != 'b')
        throw std::invalid_argument("MACELSC: expected pbc[3] with boolean dtype");
    auto xyz = real_array(positions, "positions");
    const auto lattice = real_array(cell, "cell");
    const auto periodic = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(pbc);
    EnergyInput input{std::move(graph), std::move(xyz), {}, {}};
    std::copy_n(lattice.data(), 9, input.cell.begin());
    std::copy_n(periodic.data(), 3, input.pbc.begin());
    return input;
}

double evaluate(MACELSC& self, const EnergyInput& input, const std::string& mode,
                const std::array<double, 3>& field, bool derivatives, bool stress)
{
    const auto& graph = input.graph;
    const std::span<const double> vectors(graph.vectors.data(), graph.vectors.size());
    const std::span<const double> positions(input.positions.data(), input.positions.size());
    py::gil_scoped_release release;
    if (derivatives)
        self.compute_energy_forces(graph.types, graph.senders(), graph.receivers(), vectors,
                                   positions, input.cell, input.pbc, mode, field, stress);
    else
        self.compute_energy(graph.types, graph.senders(), graph.receivers(), vectors,
                            positions, input.cell, input.pbc, mode, field);
    return self.energy;
}
} // namespace

void bind_mace_lsc(py::module_& m)
{
    lsc_binding::bind_graph(m);
    auto cls=py::class_<MACELSC>(m, "MACELSC",
                       "Native LSC energy, forces, stress, charge transfers and dipoles on a supplied graph.");
    lsc_binding::bind_stages(cls);
    cls.def("download_partial_charges",[](const MACELSC& self){auto values=self.download_partial_charges();
        py::array_t<double> out({self.graph_layout().num_nodes(),2});std::copy(values.begin(),values.end(),out.mutable_data());return out;})
       .def("download_electrostatic_potentials",[](const MACELSC& self){auto values=self.electrostatic_potentials();
        py::array_t<double> out({self.graph_layout().num_nodes(),4});std::copy(values.begin(),values.end(),out.mutable_data());return out;});
    cls.def("download_forward_features",[](const MACELSC& self,int layer){
            auto values=self.forward_features(layer);
            py::array_t<double> out({py::ssize_t(self.graph_layout().num_nodes()),py::ssize_t(4*self.num_channels)});
            std::copy(values.begin(),values.end(),out.mutable_data());return out;
        },py::arg("layer"));
    cls.def_readonly("has_local_results",&MACELSC::has_local_results)
       .def("download_layer_features",[](MACELSC& self,int layer,bool gradients){
            const auto values=gradients?self.layer_feature_gradients(layer):self.layer_features(layer);
            py::array_t<double> out({py::ssize_t(self.graph_layout().num_owned()),py::ssize_t(4*self.num_channels)});
            std::copy(values.begin(),values.end(),out.mutable_data());return out;
        },py::arg("layer"),py::arg("gradients")=false)
       .def("evaluate_charges",[](MACELSC& self,const py::array& positions){
            if(positions.ndim()!=2 || positions.shape(1)!=3)throw py::value_error("expected positions[N_owned,3]");
            auto xyz=real_array(positions,"positions");py::gil_scoped_release release;
            return self.evaluate_charges({xyz.data(),size_t(xyz.size())});
        },py::arg("positions"))
       .def("evaluate_energy",[](MACELSC& self,const py::array& cell,const py::array& pbc,
                                const std::string& mode,const std::array<double,3>& field){
            if(cell.ndim()!=2 || cell.shape(0)!=3 || cell.shape(1)!=3 ||
                pbc.ndim()!=1 || pbc.size()!=3 || pbc.dtype().kind()!='b')
                throw py::value_error("expected cell[3,3] and boolean pbc[3]");
            auto lattice=real_array(cell,"cell");
            auto periodic=py::array_t<bool,py::array::c_style|py::array::forcecast>::ensure(pbc);
            std::array<double,9> box;std::array<bool,3> flags;
            std::copy_n(lattice.data(),9,box.begin());std::copy_n(periodic.data(),3,flags.begin());
            py::gil_scoped_release release;return self.evaluate_energy(box,flags,mode,field);
        },py::arg("cell"),py::arg("pbc"),py::arg("pbc_handling")="auto",py::arg("external_field")=std::array<double,3>{})
       .def("evaluate_forces",&MACELSC::evaluate_forces,py::arg("compute_stress")=false,
            py::call_guard<py::gil_scoped_release>());
    cls.def("download_node_energies",[](const MACELSC& self){
        std::vector<double> values;
        {py::gil_scoped_release release;values=self.node_energies();}
        py::array_t<double> out(values.size());
        std::copy(values.begin(),values.end(),out.mutable_data());return out;
    },"Read the total-energy partition, including electrostatic and field energy.");
    cls.def(py::init<std::string>(), py::arg("filename"))
        .def_readonly("format_version", &MACELSC::format_version)
        .def_readonly("model_type", &MACELSC::model_type)
        .def_readonly("export_status", &MACELSC::export_status)
        .def_readonly("head", &MACELSC::head)
        .def_readonly("num_elements", &MACELSC::num_elements)
        .def_readonly("source_num_elements", &MACELSC::source_num_elements)
        .def_readonly("source_element_indices", &MACELSC::source_element_indices)
        .def_readonly("num_channels", &MACELSC::num_channels)
        .def_readonly("num_interactions", &MACELSC::num_interactions)
        .def_readonly("l_max", &MACELSC::l_max)
        .def_readonly("L_max", &MACELSC::L_max)
        .def_readonly("feature_layout", &MACELSC::feature_layout)
        .def_readonly("atomic_numbers", &MACELSC::atomic_numbers)
        .def_readonly("atomic_energies", &MACELSC::atomic_energies)
        .def_readonly("formal_charges", &MACELSC::formal_charges)
        .def_readonly("node_embedding", &MACELSC::node_embedding)
        .def_readonly("r_cut", &MACELSC::r_cut)
        .def_readonly("radial_spline_r_min", &MACELSC::radial_spline_r_min)
        .def_readonly("radial_spline_scale", &MACELSC::radial_spline_scale)
        .def_readonly("interaction_radial_shapes", &MACELSC::interaction_radial_shapes)
        .def_readonly("source_radial_shapes", &MACELSC::source_radial_shapes)
        .def_readonly("oxidation_state_ranges", &MACELSC::oxidation_state_ranges)
        .def_readonly("oxidation_state_factors", &MACELSC::oxidation_state_factors)
        .def_readonly("kspace_cutoff", &MACELSC::kspace_cutoff)
        .def_readonly("density_smearing_width", &MACELSC::density_smearing_width)
        .def_readonly("include_self_interaction", &MACELSC::include_self_interaction)
        .def_readonly("default_pbc_handling", &MACELSC::default_pbc_handling)
        .def_property_readonly("energy_readout_weights", &MACELSC::energy_readout_weights)
        .def_property_readonly("can_evaluate_local", &MACELSC::can_evaluate_local)
        .def_property_readonly("can_evaluate_sources", &MACELSC::can_evaluate_sources)
        .def_property_readonly("can_evaluate_energy", &MACELSC::can_evaluate_energy)
        .def_property_readonly("can_evaluate_forces", &MACELSC::can_evaluate_forces)
        .def_property_readonly("can_evaluate_stress", &MACELSC::can_evaluate_stress)
        .def_readonly("forces", &MACELSC::forces)
        .def_readonly("edge_forces", &MACELSC::edge_forces)
        .def_readonly("position_forces", &MACELSC::position_forces)
        .def_readonly("density_energy_gradient", &MACELSC::density_energy_gradient)
        .def_readonly("strain_derivative", &MACELSC::strain_derivative)
        .def_readonly("electrostatic_strain_derivative", &MACELSC::electrostatic_strain_derivative)
        .def_readonly("virial", &MACELSC::virial)
        .def_readonly("stress", &MACELSC::stress)
        .def_readonly("has_force_results", &MACELSC::has_force_results)
        .def_readonly("has_stress_results", &MACELSC::has_stress_results)
        .def_property_readonly("has_electrostatics_hook", &MACELSC::has_electrostatics_hook)
        .def("set_mesh_electrostatics", &MACELSC::set_mesh_electrostatics,
             py::arg("spacing")=.7,py::arg("order")=8,
             "Use native CPU mesh energy and analytic derivatives in the LSC reverse pass. Requires full periodicity; None hook restores direct.")
        .def("set_electrostatics_hook", [](MACELSC& self, const py::object& callback) {
            self.set_electrostatics_hook(electrostatics_python_hook(callback));
        }, py::arg("callback"),
        "Replace electrostatics with callback(request)->result; None restores direct. Return full gradients, excluding fields/local energy.")
        .def("evaluate_direct_electrostatics", [](const MACELSC& self, py::array positions,
             py::array density, py::array cell, py::array pbc, const std::string& mode) {
            if (positions.ndim() != 2 || positions.shape(1) != 3 || density.ndim() != 2 ||
                density.shape(0) != positions.shape(0) || density.shape(1) != 4)
                throw py::value_error("expected positions[N,3] and density[N,4]");
            if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3 ||
                pbc.ndim() != 1 || pbc.size() != 3 || pbc.dtype().kind() != 'b')
                throw py::value_error("expected cell[3,3] and boolean pbc[3]");
            const auto xyz = real_array(positions, "positions"), rho = real_array(density, "density");
            const auto lattice = real_array(cell, "cell");
            const auto periodic = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(pbc);
            std::array<double,9> box; std::array<bool,3> flags;
            std::copy_n(lattice.data(), 9, box.begin()); std::copy_n(periodic.data(), 3, flags.begin());
            LSCElectrostaticsResult result;
            { py::gil_scoped_release release;
              result = self.evaluate_direct_electrostatics({xyz.data(),size_t(xyz.size())},
                  {rho.data(),size_t(rho.size())},box,flags,mode); }
            return electrostatics_result(result, size_t(positions.shape(0)));
        }, py::arg("positions"), py::arg("density_coefficients"), py::arg("cell"), py::arg("pbc"), py::arg("pbc_handling") = "",
        "Evaluate direct electrostatic energy/gradients at fixed density without changing model state or invoking a hook.")
        .def_readonly("energy", &MACELSC::energy)
        .def_readonly("electrostatic_energy", &MACELSC::electrostatic_energy)
        .def_readonly("field_energy", &MACELSC::field_energy)
        .def_readonly("has_energy_results", &MACELSC::has_energy_results)
        .def_readonly("initial_features", &MACELSC::initial_features)
        .def_readonly("A0", &MACELSC::A0)
        .def_readonly("A1", &MACELSC::A1)
        .def_readonly("M0", &MACELSC::M0)
        .def_readonly("M1", &MACELSC::M1)
        .def_readonly("H1", &MACELSC::H1)
        .def_readonly("H2", &MACELSC::H2)
        .def_readonly("local_node_energies", &MACELSC::local_node_energies)
        .def_readonly("layer_node_energies", &MACELSC::layer_node_energies)
        .def_readonly("local_energy", &MACELSC::local_energy)
        .def_readonly("source_features", &MACELSC::source_features)
        .def_readonly("layer_density_coefficients", &MACELSC::layer_density_coefficients)
        .def_readonly("layer_edge_fluxes", &MACELSC::layer_edge_fluxes)
        .def_readonly("density_coefficients", &MACELSC::density_coefficients)
        .def_readonly("partial_charges", &MACELSC::partial_charges)
        .def_readonly("partial_dipoles", &MACELSC::partial_dipoles)
        .def_readonly("edge_fluxes", &MACELSC::edge_fluxes)
        .def_readonly("dipole", &MACELSC::dipole)
        .def_readonly("formal_charge_dipole", &MACELSC::formal_charge_dipole)
        .def_readonly("transfer_dipole", &MACELSC::transfer_dipole)
        .def_readonly("atomic_dipole_sum", &MACELSC::atomic_dipole_sum)
        .def_readonly("has_charge_results", &MACELSC::has_charge_results)
        .def("compute_local", [](MACELSC& self, py::array node_types,
                                  py::array edge_index, py::array edge_vectors) {
            const auto graph = graph_input(node_types, edge_index, edge_vectors);
            double energy;
            {
                py::gil_scoped_release release;
                self.compute_local(graph.types, graph.senders(), graph.receivers(),
                                   std::span<const double>(graph.vectors.data(), graph.vectors.size()));
                energy = self.local_energy;
            }
            return energy;
        }, py::arg("node_types"), py::arg("edge_index"), py::arg("edge_vectors"),
        "Compute atomic reference plus local readout energies on a supplied directed graph; no electrostatics or forces.")
        .def("compute_charges", [](MACELSC& self, py::array node_types, py::array edge_index,
                                    py::array edge_vectors, py::array positions) {
            const auto graph = graph_input(node_types, edge_index, edge_vectors);
            if (positions.ndim() != 2 or positions.shape(0) != node_types.size() or positions.shape(1) != 3)
                throw std::invalid_argument("MACELSC: expected positions[N,3]");
            const auto xyz = real_array(positions, "positions");
            std::array<double, 3> result;
            {
                py::gil_scoped_release release;
                self.compute_charges(graph.types, graph.senders(), graph.receivers(),
                                     std::span<const double>(graph.vectors.data(), graph.vectors.size()),
                                     std::span<const double>(xyz.data(), xyz.size()));
                result = self.dipole;
            }
            return result;
        }, py::arg("node_types"), py::arg("edge_index"), py::arg("edge_vectors"), py::arg("positions"),
        "Compute local layers, conservative charge transfers and dipoles; return total dipole in e*Angstrom. No energy calculation or forces in this method.")
        .def("compute_energy", [](MACELSC& self, py::array node_types, py::array edge_index,
                                   py::array edge_vectors, py::array positions, py::array cell,
                                   py::array pbc, const std::string& pbc_handling,
                                   const std::array<double, 3>& external_field) {
            return evaluate(self, energy_input(node_types, edge_index, edge_vectors, positions, cell, pbc),
                            pbc_handling, external_field, false, false);
        }, py::arg("node_types"), py::arg("edge_index"), py::arg("edge_vectors"),
        py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::arg("pbc_handling") = "",
        py::arg("external_field") = std::array<double, 3>{0, 0, 0},
        "Compute total energy in eV on a supplied graph, clearing derivative results.")
        .def("compute_energy_forces", [](MACELSC& self, py::array node_types, py::array edge_index,
                                          py::array edge_vectors, py::array positions, py::array cell,
                                          py::array pbc, const std::string& pbc_handling,
                                          const std::array<double, 3>& external_field, bool compute_stress) {
            return evaluate(self, energy_input(node_types, edge_index, edge_vectors, positions, cell, pbc),
                            pbc_handling, external_field, true, compute_stress);
        }, py::arg("node_types"), py::arg("edge_index"), py::arg("edge_vectors"),
        py::arg("positions"), py::arg("cell"), py::arg("pbc"), py::arg("pbc_handling") = "",
        py::arg("external_field") = std::array<double, 3>{0, 0, 0}, py::arg("compute_stress") = false,
        "Compute native energy and analytic forces; optionally ASE-order stress for a nonsingular cell. Return energy (eV); flat forces are eV/Angstrom. Supports finite realspace and fully periodic pbc/mixed_periodic/auto.")
        .def_property_readonly("can_evaluate", &MACELSC::can_evaluate);
}
