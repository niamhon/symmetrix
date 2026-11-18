#pragma once
#include "mace_lsc_cuda_reverse.cuh"
namespace lsc_cuda {
__global__ void reverse_feature_prefix(int rows,int from,int to,int C,const float* input,float* output,bool add){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(rows)*4*C)return;
    int n=k/(4*C),lm=(k%(4*C))/C,u=k%C;float x=input[local_row(lm,n,from)*C+u];auto dst=output+local_row(lm,n,to)*C+u;if(add)*dst+=x;else *dst=x;
}
__global__ void reverse_feature_packet(int count,int first,int all,int C,float* x,double* packet){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(count)*4*C)return;
    int n=first+k/(4*C),lm=(k%(4*C))/C,u=k%C;auto ptr=x+local_row(lm,n,all)*C+u;packet[k]=*ptr;*ptr=0;
}
__global__ void reverse_feature_add(int count,int all,int C,const int* slots,const double* packet,float* x){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=size_t(count)*4*C)return;
    int n=slots[k/(4*C)],lm=(k%(4*C))/C,u=k%C;atomicAdd(x+local_row(lm,n,all)*C+u,float(packet[k]));
}
__global__ void reverse_density_seed(int owned,int all,const double* phi,double* adj){
    size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k<size_t(all)*4)adj[k]=(k/4<owned || k%4==0)?phi[k]:0.;
}
__global__ void reverse_force_slots(int owned,int all,const int* types,const float* formal,const double* positions,
    const int* is,const int* ie,const int* os,const int* oe,const double* pg,const double* eg,
    double fx,double fy,double fz,size_t stride,double* terms,double* forces,double* pf,int* error){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=all)return;double field[3]={fx,fy,fz};
    for(int d=0;d<3;++d){double f=i<owned?-pg[size_t(i)*3+d]:0.;if(i<owned)pf[size_t(i)*3+d]=f;
        for(int j=is[i];j<is[i+1];++j)f-=eg[size_t(ie[j])*3+d];for(int j=os[i];j<os[i+1];++j)f+=eg[size_t(oe[j])*3+d];
        forces[size_t(i)*3+d]=f;if(!source_finite(f))atomicOr(error,1);
        if(i<owned)for(int a=0;a<3;++a)terms[size_t(3*a+d)*stride+i]=positions[size_t(i)*3+a]*double(formal[types[i]])*field[d];}
}
__global__ void reverse_force_packet(int count,int first,double* x,double* packet){size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k<size_t(count)*3){packet[k]=x[size_t(first)*3+k];x[size_t(first)*3+k]=0;}}
__global__ void reverse_force_add(int count,const int* slots,const double* packet,double* x){size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(k<size_t(count)*3)atomicAdd(x+size_t(slots[k/3])*3+k%3,packet[k]);}
} // namespace lsc_cuda
