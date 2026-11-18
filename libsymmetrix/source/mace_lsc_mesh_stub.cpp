#include "mace_lsc_mesh.hpp"
#include "mace_lsc.hpp"
#include <stdexcept>
struct LSCCPUMesh::Workspace {};
bool lsc_cpu_mesh_compiled() { return false; }
LSCCPUMesh::LSCCPUMesh(double mesh_spacing,int spline_order) : spacing(mesh_spacing),order(spline_order)
{
    throw std::runtime_error("LSCCPUMesh: CPU mesh is not built; enable SYMMETRIX_LSC_MESH with FFTW3");
}
LSCCPUMesh::~LSCCPUMesh()=default;
LSCMeshEnergyResult LSCCPUMesh::evaluate_energy(const LSCElectrostaticsInput&)
{
    throw std::runtime_error("LSCCPUMesh: CPU mesh is not built");
}
LSCMeshEnergyResult MACELSC::evaluate_mesh_electrostatics(LSCCPUMesh&,
    std::span<const double>,std::span<const double>,const std::array<double,9>&,
    const std::array<bool,3>&,const std::string&) const
{
    throw std::runtime_error("LSCCPUMesh: CPU mesh is not built");
}

LSCElectrostaticsResult LSCCPUMesh::evaluate(const LSCElectrostaticsInput&)
{
    throw std::runtime_error("LSCCPUMesh: CPU mesh is not built");
}
LSCElectrostaticsResult MACELSC::evaluate_mesh_electrostatics_full(LSCCPUMesh&,
    std::span<const double>,std::span<const double>,const std::array<double,9>&,
    const std::array<bool,3>&,const std::string&) const
{
    throw std::runtime_error("LSCCPUMesh: CPU mesh is not built");
}

void MACELSC::set_mesh_electrostatics(double,int)
{
    throw std::runtime_error("LSCCPUMesh: CPU mesh is not built; enable SYMMETRIX_LSC_MESH with FFTW3");
}
