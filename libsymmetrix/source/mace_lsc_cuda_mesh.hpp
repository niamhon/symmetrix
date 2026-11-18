#pragma once
#include "mace_lsc_mesh.hpp"
#include <span>
#include <string>
class MACELSC;
bool lsc_cuda_mesh_compiled();

// One complete periodic grid on one GPU/process. Neural precision is independent
// of these double-precision mesh operations, as in Polar's local cuFFT solver.
class LSCCudaMesh {
public:
    explicit LSCCudaMesh(double spacing=.7,int order=8);
    ~LSCCudaMesh();
    LSCCudaMesh(const LSCCudaMesh&)=delete;
    LSCCudaMesh& operator=(const LSCCudaMesh&)=delete;
    const double spacing;
    const int order;
    int device_id=-1;
    std::string device_name;
    std::array<int,3> grid_shape{};
    size_t plan_generation=0;
    // Energy-only device entry point; neither host input copies nor readbacks.
    void evaluate_energy(const LSCCudaElectrostaticsInput&,double* energy,double* density_gradient);
    // Full positive gradients; output buffers are filled on input.stream.
    void evaluate(const LSCCudaElectrostaticsInput&,const LSCCudaElectrostaticsOutput&);
    LSCElectrostaticsResult evaluate(const MACELSC& model,std::span<const double> positions,
        std::span<const double> density,const std::array<double,9>& cell,
        const std::array<bool,3>& pbc,const std::string& mode="auto");
    // Explicit diagnostic bridge; the CPU model supplies the reference parameters.
    LSCMeshEnergyResult evaluate_energy(const MACELSC& model,std::span<const double> positions,
        std::span<const double> density,const std::array<double,9>& cell,
        const std::array<bool,3>& pbc,const std::string& mode="auto");
private:
    struct Workspace;
    std::unique_ptr<Workspace> workspace;
    void evaluate_impl(const LSCCudaElectrostaticsInput&,double* energy,double* density_gradient,
        double* position_gradient=nullptr,double* strain_derivative=nullptr);
};
