#pragma once
#include "mace_lsc_cuda_sources.cuh"
namespace lsc_cuda {
__global__ void seed_partial_layers(int owned,int all,const double* layers,double* partial){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(owned)*8)return;
    int t=k/(size_t(owned)*4);size_t j=k%(size_t(owned)*4);partial[size_t(t)*all*4+j]=layers[k];
}
__global__ void extract_partial_layers(int owned,int all,const double* partial,double* layers){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(owned)*8)return;
    int t=k/(size_t(owned)*4);size_t j=k%(size_t(owned)*4);layers[k]=partial[size_t(t)*all*4+j];
}
__global__ void extract_source_features(int owned,int all,int C,const float* halo,float* features){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(owned)*8*C)return;
    int t=k/(size_t(owned)*4*C);size_t j=k%(size_t(owned)*4*C);
    int node=j/(4*C),lm=(j%(4*C))/C,u=j%C;
    features[size_t(t)*owned*4*C+local_row(lm,node,owned)*C+u]=halo[size_t(t)*all*4*C+local_row(lm,node,all)*C+u];
}
__global__ void pack_partial_charges(int first,int count,int all,double* rho,double* values,bool consume){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(count)*2)return;
    int slot=first+k/2,t=k%2;size_t j=(size_t(t)*all+slot)*4;values[k]=rho[j];if(consume)rho[j]=0;
}
__global__ void add_partial_charges(int count,int all,const int* slots,const double* values,double* rho,int* error){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(count)*2)return;
    size_t j=(size_t(k%2)*all+slots[k/2])*4;double old=atomicAdd(rho+j,values[k]);
    if(!source_finite(old+values[k]))atomicOr(error,2);
}
__global__ void publish_potentials(int owned,int all,const int* owners,const double* input,double* halo){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(all)*4)return;
    int owner=owners[k/4];halo[k]=owner>=0?input[size_t(owner)*4+k%4]:0.;
}
__global__ void pack_density_potentials(int count,const int* slots,const double* phi,double* values){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k<size_t(count)*4)values[k]=phi[size_t(slots[k/4])*4+k%4];
}
} // namespace lsc_cuda
