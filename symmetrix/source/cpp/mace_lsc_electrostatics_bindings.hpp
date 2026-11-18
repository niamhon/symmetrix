#pragma once
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mace_lsc_electrostatics.hpp"
#include "mace_lsc_graph_input.hpp"

namespace lsc_binding {
namespace py = pybind11;
inline py::array_t<double> electrostatics_array(std::span<const double> values,
                                               std::vector<py::ssize_t> shape)
{
    py::array_t<double> result(shape);
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}
inline py::dict electrostatics_parameters(const LSCElectrostaticsParameters& p)
{
    py::dict result;
    result["field_constant"] = p.field_constant; result["kspace_cutoff"] = p.kspace_cutoff;
    result["sigma2"] = p.sigma2; result["a"] = p.a;
    result["reciprocal_self"] = p.reciprocal_self; result["realspace_self"] = p.realspace_self;
    result["sigma"] = p.sigma; result["offset"] = p.offset; result["epsilon"] = p.epsilon;
    result["displacements"] = p.displacements; result["include_self"] = p.include_self;
    return result;
}
inline py::dict electrostatics_result(const LSCElectrostaticsResult& result, size_t nodes)
{
    py::dict out;
    out["energy"] = result.energy;
    out["density_gradient"] = electrostatics_array(result.density_gradient, {py::ssize_t(nodes), 4});
    out["position_gradient"] = electrostatics_array(result.position_gradient, {py::ssize_t(nodes), 3});
    out["strain_derivative"] = electrostatics_array(result.strain_derivative, {3, 3});
    return out;
}
inline LSCElectrostaticsHook electrostatics_python_hook(const py::object& callback)
{
    if (callback.is_none()) return {};
    if (!PyCallable_Check(callback.ptr())) throw py::type_error("electrostatics hook must be callable or None");
    return [function = py::reinterpret_borrow<py::function>(callback)](const LSCElectrostaticsInput& input) {
        py::gil_scoped_acquire acquire;
        const auto nodes = py::ssize_t(input.positions.size()/3);
        // Copies deliberately prevent Python from retaining borrowed native memory.
        py::dict request;
        request["positions"] = electrostatics_array(input.positions, {nodes, 3});
        request["density_coefficients"] = electrostatics_array(input.density, {nodes, 4});
        request["cell"] = electrostatics_array(input.context.cell, {3, 3});
        py::array_t<bool> periodic(3);
        std::copy(input.context.pbc.begin(), input.context.pbc.end(), periodic.mutable_data());
        request["pbc"] = periodic; request["pbc_handling"] = input.context.mode;
        request["parameters"] = electrostatics_parameters(input.context.parameters);
        const auto values = function(request).cast<py::dict>();
        LSCElectrostaticsResult result;
        result.energy = values["energy"].cast<double>();
        auto read = [&](const char* name, py::ssize_t rows, py::ssize_t columns) {
            const auto value = values[name].cast<py::array>();
            if (value.ndim() != 2 || value.shape(0) != rows || value.shape(1) != columns)
                throw py::value_error(std::string("LSC electrostatics hook: invalid shape for ") + name);
            return real_array(value, name);
        };
        const auto density = read("density_gradient", nodes, 4);
        const auto positions = read("position_gradient", nodes, 3);
        const auto strain = read("strain_derivative", 3, 3);
        result.density_gradient.assign(density.data(), density.data()+density.size());
        result.position_gradient.assign(positions.data(), positions.data()+positions.size());
        std::copy_n(strain.data(), 9, result.strain_derivative.begin());
        validate_lsc_electrostatics_result(result, size_t(nodes));
        return result;
    };
}
} // namespace lsc_binding
