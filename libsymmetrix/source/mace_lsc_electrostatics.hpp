#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// The values used by this backend, including CUDA's existing FP32 parameter
// packing. No model/export format changes are needed to select another solver.
struct LSCElectrostaticsParameters {
    double field_constant = 0, kspace_cutoff = 0, sigma2 = 0;
    std::array<double, 2> a{};
    std::array<double, 4> reciprocal_self{}, realspace_self{};
    double sigma = 0, offset = 0, epsilon = 0;
    std::array<double, 9> displacements{};
    bool include_self = false;
};

struct LSCElectrostaticsContext {
    std::array<double, 9> cell{}; // Row lattice vectors, Angstrom.
    std::array<bool, 3> pbc{};
    std::string mode; // Resolved realspace, pbc or mixed_periodic.
    LSCElectrostaticsParameters parameters;
};

// Borrowed host buffers, valid only for the duration of the callback.
struct LSCElectrostaticsInput {
    std::span<const double> positions; // [N,3], Angstrom.
    std::span<const double> density;   // [N,4]: q, dipole_y, dipole_z, dipole_x.
    LSCElectrostaticsContext context;
};

// Complete electrostatic contribution only: exclude local energy and fields.
// Gradients are positive dE/dx, not forces or Polar's half-potential psi.
struct LSCElectrostaticsResult {
    double energy = 0;                       // eV.
    std::vector<double> density_gradient;     // [N,4], dE/drho.
    std::vector<double> position_gradient;    // [N,3], dE/dr at fixed rho/cell.
    // Full dE/dstrain at fixed Cartesian density components, with positions
    // and cell deformed together. Includes position terms; do not add r*g again.
    std::array<double, 9> strain_derivative{};
};
using LSCElectrostaticsHook = std::function<LSCElectrostaticsResult(const LSCElectrostaticsInput&)>;

inline bool lsc_electrostatics_finite(double value)
{
    return (std::bit_cast<std::uint64_t>(value) & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
inline void validate_lsc_electrostatics_result(const LSCElectrostaticsResult& result, size_t nodes)
{
    if (result.density_gradient.size() != nodes * 4 || result.position_gradient.size() != nodes * 3)
        throw std::invalid_argument("LSC electrostatics hook: expected density_gradient[N,4] and position_gradient[N,3]");
    if (!lsc_electrostatics_finite(result.energy))
        throw std::invalid_argument("LSC electrostatics hook: nonfinite energy");
    for (const auto* values : {&result.density_gradient, &result.position_gradient})
        for (double value : *values)
            if (!lsc_electrostatics_finite(value)) throw std::invalid_argument("LSC electrostatics hook: nonfinite gradient");
    for (double value : result.strain_derivative)
        if (!lsc_electrostatics_finite(value)) throw std::invalid_argument("LSC electrostatics hook: nonfinite strain derivative");
}

// Forward declaration keeps the interface available to CPU builds without CUDA
// headers. This is the same pointer type as cudaStream_t in CUDA translation units.
struct CUstream_st;
struct LSCCudaElectrostaticsInput {
    size_t num_nodes = 0;
    const double* positions = nullptr; // Borrowed device [N,3].
    const double* density = nullptr;   // Borrowed device [N,4].
    LSCElectrostaticsContext context;   // Host metadata.
    CUstream_st* stream = nullptr;     // Launch work on this stream.
};
struct LSCCudaElectrostaticsOutput {
    // Zero-initialized borrowed device buffers. Fill on input.stream; no pointer
    // may be retained after the callback. Values have the host result semantics.
    double* energy = nullptr;
    double* density_gradient = nullptr;
    double* position_gradient = nullptr;
    double* strain_derivative = nullptr;
};
using LSCCudaElectrostaticsHook = std::function<void(const LSCCudaElectrostaticsInput&, const LSCCudaElectrostaticsOutput&)>;
