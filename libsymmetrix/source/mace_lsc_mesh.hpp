#pragma once

#include <array>
#include <memory>
#include <vector>
#include "mace_lsc_electrostatics.hpp"

// Energy-only diagnostic result. evaluate() supplies the complete derivative contract.
struct LSCMeshEnergyResult {
    double energy = 0;
    double reciprocal_energy = 0;
    double self_correction = 0;
    std::vector<double> density_gradient; // Full dE/drho, [q,dy,dz,dx].
    std::array<int, 3> grid_shape{};       // [nx,ny,nz], x fastest in storage.
};

bool lsc_cpu_mesh_compiled();

// Whole periodic grid on one process. Same assignment/deconvolution and
// Gaussian kernel and analytic derivatives as Polar; no distributed MPI.
class LSCCPUMesh {
public:
    explicit LSCCPUMesh(double spacing = 0.7, int order = 8);
    ~LSCCPUMesh();
    LSCCPUMesh(const LSCCPUMesh&) = delete;
    LSCCPUMesh& operator=(const LSCCPUMesh&) = delete;
    const double spacing;
    const int order;
    LSCMeshEnergyResult evaluate_energy(const LSCElectrostaticsInput& input);
    LSCElectrostaticsResult evaluate(const LSCElectrostaticsInput& input);
private:
    LSCMeshEnergyResult evaluate_impl(const LSCElectrostaticsInput& input,
        std::vector<double>* position_gradient, std::array<double,9>* strain_derivative);
    struct Workspace;
    std::unique_ptr<Workspace> workspace;
};
