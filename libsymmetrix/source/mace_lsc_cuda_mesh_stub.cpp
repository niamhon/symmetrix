#include "mace_lsc_cuda_mesh.hpp"
#include "mace_lsc_cuda.hpp"
#include <stdexcept>
struct LSCCudaMesh::Workspace {};
bool lsc_cuda_mesh_compiled(){return false;}
LSCCudaMesh::LSCCudaMesh(double spacing,int order):spacing(spacing),order(order){throw std::runtime_error("LSCCudaMesh: CUDA mesh is not built");}
LSCCudaMesh::~LSCCudaMesh()=default;
void LSCCudaMesh::evaluate_energy(const LSCCudaElectrostaticsInput&,double*,double*){throw std::runtime_error("LSCCudaMesh: CUDA mesh is not built");}
LSCMeshEnergyResult LSCCudaMesh::evaluate_energy(const MACELSC&,std::span<const double>,std::span<const double>,
    const std::array<double,9>&,const std::array<bool,3>&,const std::string&){throw std::runtime_error("LSCCudaMesh: CUDA mesh is not built");}

void LSCCudaMesh::evaluate(const LSCCudaElectrostaticsInput&,const LSCCudaElectrostaticsOutput&){throw std::runtime_error("LSCCudaMesh: CUDA mesh is not built");}
LSCElectrostaticsResult LSCCudaMesh::evaluate(const MACELSC&,std::span<const double>,std::span<const double>,
    const std::array<double,9>&,const std::array<bool,3>&,const std::string&){throw std::runtime_error("LSCCudaMesh: CUDA mesh is not built");}
void MACELSCCuda::set_mesh_electrostatics(double,int){throw std::runtime_error("LSCCudaMesh: CUDA mesh is not built");}
