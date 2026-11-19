#pragma once
#include "mace_lsc_cuda_sources.cuh"

namespace lsc_cuda {
constexpr int compact_basis=64;

// Foundation Phi0 pattern. One receiver/species block accumulates 64 radial
// functions, then Q maps the species/basis density to the original A0 layout.
__global__ void __launch_bounds__(compact_basis,8) compact_first_density(
    int N,int T,const int* starts,const int* order,const int* senders,const int* types,
    const float* Y,const int* intervals,const float* coordinate,const float* basis,
    const float* cg,float* density)
{
    constexpr int B=compact_basis,Batch=32;
    struct Edge{float y[16],t;int k,b;};
    __shared__ Edge staged[Batch];
    const int n=blockIdx.x,b=blockIdx.y,c=threadIdx.x,F=T*B;
    float acc[16]={};
    for(int first=starts[n];first<starts[n+1];first+=Batch){
        const int count=min(Batch,starts[n+1]-first);
        __syncthreads();
        if(c<count){
            const int e=order[first+c];auto& q=staged[c];
            q.k=intervals[e];q.t=coordinate[e];q.b=types[senders[e]];
            for(int lm=0;lm<16;++lm)q.y[lm]=Y[size_t(e)*16+lm];
        }
        __syncthreads();
        for(int j=0;j<count;++j){
            const auto& q=staged[j];if(q.b!=b)continue;
            const float* p=basis+size_t(q.k)*4*B;
            const float value=p[c]+q.t*(p[B+c]+q.t*(p[2*B+c]+q.t*p[3*B+c]));
            #pragma unroll
            for(int lm=0;lm<16;++lm)acc[lm]+=value*(cg[lm]*q.y[lm]);
        }
    }
    #pragma unroll
    for(int lm=0;lm<16;++lm)density[local_row(lm,n,N)*F+b*B+c]=acc[lm];
}

// Cache the receiver's density adjoints over its neighbor list. Initial
// species features are constant in Cartesian coordinates, so only edge
// derivatives are needed. Each edge belongs to exactly one receiver/species.
__global__ void __launch_bounds__(compact_basis,8) compact_first_density_reverse(
    int N,int T,const int* starts,const int* order,const int* senders,const int* types,
    const float4* geometry,const float* Y,const float* dY,const int* intervals,
    const float* coordinate,const float* basis,const float* grid,const float* cg,
    const float* density_adj,double* edge_adj)
{
    constexpr int B=compact_basis,Batch=32;
    struct Edge{float y[16],dy[48],t;float4 geometry;int e,k,b;};
    __shared__ Edge staged[Batch];
    __shared__ float sums[3][B/32];
    const int n=blockIdx.x,b=blockIdx.y,c=threadIdx.x,F=T*B;
    float adj[16];
    #pragma unroll
    for(int lm=0;lm<16;++lm)adj[lm]=cg[lm]*density_adj[local_row(lm,n,N)*F+b*B+c];
    for(int first=starts[n];first<starts[n+1];first+=Batch){
        const int count=min(Batch,starts[n+1]-first);
        __syncthreads();
        if(c<count){
            const int e=order[first+c];auto& q=staged[c];
            q.e=e;q.k=intervals[e];q.t=coordinate[e];q.b=types[senders[e]];q.geometry=geometry[e];
            for(int lm=0;lm<16;++lm)q.y[lm]=Y[size_t(e)*16+lm];
            for(int i=0;i<48;++i)q.dy[i]=dY[size_t(e)*48+i];
        }
        __syncthreads();
        for(int j=0;j<count;++j){
            const auto& q=staged[j];if(q.b!=b)continue;
            float value,derivative;cubic(basis+size_t(q.k)*4*B,B,c,q.t,value,derivative);
            float angular=0,dy[3]={};
            #pragma unroll
            for(int lm=0;lm<16;++lm){
                angular+=adj[lm]*q.y[lm];
                #pragma unroll
                for(int d=0;d<3;++d)dy[d]+=adj[lm]*q.dy[d*16+lm];
            }
            const float radial=derivative*grid[2]*angular;
            const float u[3]={q.geometry.x,q.geometry.y,q.geometry.z};
            #pragma unroll
            for(int d=0;d<3;++d){
                const float v=source_warp_sum(radial*u[d]+value*dy[d]);
                if((c&31)==0)sums[d][c/32]=v;
            }
            __syncthreads();
            if(c==0)for(int d=0;d<3;++d)edge_adj[size_t(q.e)*3+d]+=double(sums[d][0]+sums[d][1]);
            __syncthreads();
        }
    }
}
} // namespace lsc_cuda
