#pragma once
#include "mace_lsc_cuda_local.cuh"
namespace lsc_cuda {
// Owned calculations retain their existing layout. This separate halo includes
// every LAMMPS slot, with zero/unready rows for ghosts owned by other ranks.
__global__ void publish_forward_halo(int owned,int all,int C,const int* owners,
                                     const float* features,float* halo)
{
    const size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(k>=size_t(all)*4*C)return;
    const int node=k/(4*C),lm=(k%(4*C))/C,u=k%C,owner=owners[node];
    halo[local_row(lm,node,all)*C+u]=owner>=0?features[local_row(lm,owner,owned)*C+u]:0.f;
}
__global__ void pack_forward_halo(int count,int all,int C,const int* slots,
                                  const float* halo,double* values)
{
    const size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(k>=size_t(count)*4*C)return;
    const int row=k/(4*C),lm=(k%(4*C))/C,u=k%C;
    values[k]=halo[local_row(lm,slots[row],all)*C+u];
}
__global__ void unpack_forward_halo(int count,int first,int all,int C,
                                    const double* values,float* halo)
{
    const size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(k>=size_t(count)*4*C)return;
    const int node=first+k/(4*C),lm=(k%(4*C))/C,u=k%C;
    halo[local_row(lm,node,all)*C+u]=float(values[k]);
}
} // namespace lsc_cuda
