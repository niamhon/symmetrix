#pragma once
#include "mace_lsc_cuda_local.cuh"

namespace lsc_cuda {
// Polar's modulation operation: separate scalar/vector channel weights,
// with the same vector weight applied to all three magnetic components.
__global__ void source_modulate(int N,const float* hidden,const float* weights,float* features)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    constexpr int C=local_channels;
    if(i>=size_t(N)*C)return;
    const int node=i/C,u=i%C;
    for(int lm=0;lm<4;++lm){
        const size_t row=local_row(lm,node,N)*C+u;
        features[row]=hidden[row]*weights[(size_t(lm>0)*N+node)*C+u];
    }
}
__device__ __forceinline__ float source_warp_sum(float value)
{
    for(int offset=16;offset>0;offset>>=1)value+=__shfl_xor_sync(~0u,value,offset);
    return value;
}
// One block per directed edge, one lane per channel. Fuse the source UVU
// tensor product and scalar readout, avoiding an E x 128 message allocation.
// The validated scalar-output specification and its actual coefficients are
// consumed directly, including valid reorderings of the source instructions.
__global__ void source_flux(int N,const int* senders,const int* receivers,
    const float* Y,const int* intervals,const float* coordinate,
    const float* radial,const int* spec,const float* coefficients,int terms,
    const float* sender_features,const float* receiver_features,
    const float* readout,const float* bias,const float* scale,float* flux,int* error)
{
    constexpr int C=local_channels,F=2*C;
    __shared__ float warp_sums[C/32];
    const int edge=blockIdx.x,u=threadIdx.x;
    const int sender=senders[edge],receiver=receivers[edge],knot=intervals[edge];
    const float x=coordinate[edge];float R[2],derivative;
    for(int p=0;p<2;++p)cubic(radial+size_t(knot)*4*F,F,p*C+u,x,R[p],derivative);
    float message=0;
    for(int t=0;t<terms;++t){
        const int lm=spec[4*t+1]/C,angular=spec[4*t+2],path=spec[4*t+3]/C;
        const float features=sender_features[local_row(lm,sender,N)*C+u]
                            +receiver_features[local_row(lm,receiver,N)*C+u];
        message+=(coefficients[t]*Y[size_t(edge)*16+angular])*features*R[path];
    }
    const float sum=source_warp_sum(message*readout[u]);
    if((u&31)==0)warp_sums[u/32]=sum;
    __syncthreads();
    if(u==0){
        float value=0;for(int w=0;w<C/32;++w)value+=warp_sums[w];
        value=(value+bias[0])*scale[0];flux[edge]=value;
        if((__float_as_uint(value)&0x7f800000u)==0x7f800000u)atomicOr(error,2);
    }
}

__device__ __forceinline__ bool source_finite(double value)
{
    return (__double_as_longlong(value)&0x7ff0000000000000LL)!=0x7ff0000000000000LL;
}
__global__ void validate_source_positions(size_t count,const double* positions,int* error)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count and not source_finite(positions[i]))atomicOr(error,1);
}
// FP32 neural readouts are accumulated in FP64. Each directed transfer appears
// exactly once with each sign; no global charge-normalization correction.
__global__ void source_density(int N,const int* incoming_starts,const int* incoming_edges,
    const int* outgoing_starts,const int* outgoing_edges,const float* flux,
    const float* multipoles,double* density,int* error)
{
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=N)return;
    double incoming=0,outgoing=0;
    for(int j=incoming_starts[node];j<incoming_starts[node+1];++j)incoming+=double(flux[incoming_edges[j]]);
    for(int j=outgoing_starts[node];j<outgoing_starts[node+1];++j)outgoing+=double(flux[outgoing_edges[j]]);
    density[size_t(node)*4]=incoming-outgoing;
    // Multipole projection uses Polar's l-major device layout. Its scalar
    // output is unused by LSC; replace it with the conservative divergence.
    for(int lm=1;lm<4;++lm){
        const double value=multipoles[size_t(N)+size_t(node)*3+lm-1];
        density[size_t(node)*4+lm]=value;
        if(not source_finite(value))atomicOr(error,2);
    }
}
__global__ void source_nodes(int N,const int* types,const float* formal,const double* positions,
    const double* layers,double* density,double* charges,double* dipoles,
    size_t stride,double* terms,int* error)
{
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=N)return;
    double rho[4];
    for(int lm=0;lm<4;++lm){
        const size_t i=size_t(node)*4+lm;
        rho[lm]=(lm==0?double(formal[types[node]]):0.)+layers[i]+layers[size_t(N)*4+i];
        density[i]=rho[lm];if(not source_finite(rho[lm]))atomicOr(error,2);
    }
    charges[node]=rho[0];
    constexpr int xyz[3]={3,1,2};
    for(int d=0;d<3;++d){
        const double dipole=rho[xyz[d]];dipoles[size_t(node)*3+d]=dipole;
        terms[size_t(d)*stride+node]=double(formal[types[node]])*positions[size_t(node)*3+d];
        terms[size_t(d+3)*stride+node]=dipole;
    }
}
__global__ void source_edges(int E,const float* layers,const double* vectors,
    double* flux,size_t stride,double* terms)
{
    const int edge=blockIdx.x*blockDim.x+threadIdx.x;
    if(edge>=E)return;
    const double value=double(layers[edge])+double(layers[size_t(E)+edge]);flux[edge]=value;
    // All directed edges contribute; there is no extra factor of one half.
    for(int d=0;d<3;++d)terms[size_t(d+6)*stride+edge]=value*vectors[size_t(edge)*3+d];
}
__global__ void source_dipole(const double* formal,const double* atomic,const double* transfer,
                             double* total,int* error)
{
    const int d=threadIdx.x;
    if(d<3){
        const double value=formal[d]+transfer[d]+atomic[d];total[d]=value;
        if(not source_finite(value))atomicOr(error,2);
    }
}
} // namespace lsc_cuda
