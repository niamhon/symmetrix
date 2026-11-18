#pragma once
#include "mace_lsc_cuda_kernels.cuh"
#include "mace_lsc_cuda_tables.hpp"

namespace lsc_cuda {
constexpr int local_channels=128;
// Polar's GPU feature layout: [l][node][m][channel].
__host__ __device__ inline size_t local_row(int lm,int node,int nodes)
{
    const int l=lm_l(lm);
    return size_t(nodes)*l*l+size_t(node)*(2*l+1)+lm-l*l;
}
__host__ __device__ inline size_t message_row(int lme,int node,int nodes,int layer)
{
    if(layer==0)return local_row(lme,node,nodes);
    const int off[5]={0,2,11,26,40},paths[4]={2,3,3,2};
    int l=0;while(lme>=off[l+1])++l;
    return size_t(nodes)*off[l]+size_t(node)*(2*l+1)*paths[l]+lme-off[l];
}

struct LocalEdge {float Y[16],u;int knot,sender;};
// Like Polar's k_Phi1, one block owns each receiver and one thread each channel.
// Read the already-tested geometry; evaluate radial values inside aggregation.
template<int Layer,bool Mixed,bool Ghosts=false>
__global__ void __launch_bounds__(local_channels) local_convolution(
    int N,const int* starts,const int* edge_order,const int* senders,
    const float* Y,const int* intervals,const float* coordinate,
    const float* radial,const __half2* radial_half,const float* cg,
    const float* up,float* message,int input_nodes=0)
{
    using Conv=LocalConvolution<Layer>;
    constexpr int C=local_channels,F=Conv::paths*C,Batch=64;
    __shared__ LocalEdge staged[Batch];
    const int node=blockIdx.x,channel=threadIdx.x;
    float accumulator[Conv::width]={};
    for(int first=starts[node];first<starts[node+1];first+=Batch){
        const int count=min(Batch,starts[node+1]-first);
        __syncthreads();
        if(channel<count){
            const int e=edge_order[first+channel];auto& q=staged[channel];
            q.sender=senders[e];q.knot=intervals[e];q.u=coordinate[e];
            for(int lm=0;lm<16;++lm)q.Y[lm]=Y[size_t(e)*16+lm];
        }
        __syncthreads();
        for(int j=0;j<count;++j){
            const auto& q=staged[j];float R[Conv::paths],h[Layer==0?1:4];
            for(int p=0;p<Conv::paths;++p){
                float derivative;
                if constexpr(Mixed) cubic(radial+size_t(q.knot)*2*F,radial_half+size_t(q.knot)*F,F,p*C+channel,q.u,R[p],derivative);
                else cubic(radial+size_t(q.knot)*4*F,F,p*C+channel,q.u,R[p],derivative);
            }
            for(int lm=0;lm<(Layer==0?1:4);++lm)h[lm]=up[local_row(lm,q.sender,Ghosts?input_nodes:N)*C+channel];
            static_for<Conv::count>([&](auto index){
                constexpr Term t=Conv::terms[index];
                // The instruction layout is specialized, but use the exported
                // coefficients rather than substituting different constants.
                accumulator[t.lme]+=(cg[index]*q.Y[t.lm1])*(R[t.path]*h[t.lm2]);
            });
        }
    }
    for(int q=0;q<Conv::width;++q)message[message_row(q,node,N,Layer)*C+channel]=accumulator[q];
}
__global__ void local_contraction(int N, int C, const int* types, const float* M_weights, const float* A, float* M) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int k = int(i%C);
    float v[Tables<1>::num_values];
    float w[Tables<1>::num_m];
    float out[4] = {};
    for (int lm = 0; lm < 16; ++lm) v[lm] = A[local_row(lm, n, N)*C + k];
    for (int q = 16; q < Tables<1>::num_values; ++q) v[q] = 0;
    for (int e = 0; e < Tables<1>::num_m; ++e) w[e] = M_weights[(size_t(types[n])*Tables<1>::num_m+e)*C+k];
    static_for<Tables<1>::num_b>([&](auto e) {
        constexpr BTerm t = Tables<1>::b[e];
        v[t.q] += float(t.c)*v[t.a]*v[t.b];
    });
    static_for<Tables<1>::num_m>([&](auto e) {
        constexpr MTerm t = Tables<1>::m[e];
        for (int m = 0; m < t.n; ++m) out[t.o + m] += w[e]*v[t.q + m];
    });
    for (int LM = 0; LM < 4; ++LM) M[local_row(LM, n, N)*C + k] = out[LM];
}


__global__ void initial_features(int N,const int* types,const float* embedding,float* features)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(N)*local_channels)features[i]=embedding[types[i/local_channels]*local_channels+i%local_channels];
}
__global__ void linear_bias(size_t count,int channels,float* features,const float* bias)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)features[i]+=bias[i%channels];
}
__global__ void add_species_residual(int N,const int* types,int species,const float* residual,float* hidden)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(N)*4*local_channels)return;
    const size_t scalar=size_t(N)*local_channels;
    const int node=i<scalar?i/local_channels:(i-scalar)/(3*local_channels);
    if(types[node]==species)hidden[i]+=residual[i];
}
// Preserve the atomic reference in FP64, as Polar does. Cast each learned
// readout before addition so the small local correction is not lost to E0.
__global__ void local_energies(int N,const int* types,const double* atomic,
                             const float* layer_energies,double* energies)
{
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node<N)energies[node]=atomic[types[node]]+double(layer_energies[node])+double(layer_energies[N+node]);
}
} // namespace lsc_cuda
