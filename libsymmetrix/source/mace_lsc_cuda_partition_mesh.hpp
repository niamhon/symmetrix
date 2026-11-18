#pragma once
#include "mace_lsc_partition_mesh.hpp"
#include <memory>
// Device mesh arithmetic/local cuFFT with explicit host-staged MPI transport.
class LSCCudaMeshPartition {
public:
    LSCCudaMeshPartition();~LSCCudaMeshPartition();
    std::vector<double> spread(lsc_partition::Grid,std::span<const double> positions,std::span<const double> density);
    lsc_partition::KernelResult convolve(lsc_partition::Grid,std::span<const double> values);
    LSCElectrostaticsResult gather(lsc_partition::Grid,std::span<const double> potential);
    void fft_local(std::span<double> values,int length,int total,int direction);
    size_t fft_calls=0;
private:
    struct Workspace;std::unique_ptr<Workspace> workspace;
};
