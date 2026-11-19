#pragma once
#include "mace_lsc_cuda_sources.cuh"
namespace lsc_cuda {
__global__ void local_contraction_reverse(int N, int C, const int* types, const float* M_weights, const float* A,
                                      const float* M_adj, float* A_adj) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)N*C) return;
    const int n = int(i/C);
    const int k = int(i%C);
    float v[Tables<1>::num_values];
    float w[Tables<1>::num_m];
    float g_out[4];
    float g[Tables<1>::num_values] = {};
    for (int lm = 0; lm < 16; ++lm) v[lm] = A[local_row(lm, n, N)*C + k];
    for (int q = 16; q < Tables<1>::num_values; ++q) v[q] = 0;
    for (int e = 0; e < Tables<1>::num_m; ++e) w[e] = M_weights[(size_t(types[n])*Tables<1>::num_m+e)*C+k];
    for (int LM = 0; LM < 4; ++LM) g_out[LM] = M_adj[local_row(LM, n, N)*C + k];
    static_for<Tables<1>::num_b>([&](auto e) {
        constexpr BTerm t = Tables<1>::b[e];
        v[t.q] += float(t.c)*v[t.a]*v[t.b];
    });
    static_for<Tables<1>::num_m>([&](auto e) {
        constexpr MTerm t = Tables<1>::m[e];
        for (int m = 0; m < t.n; ++m) g[t.q + m] += w[e]*g_out[t.o + m];
    });
    static_for<Tables<1>::num_b>([&](auto e) {
        constexpr BTerm t = Tables<1>::b[Tables<1>::num_b - 1 - e];
        const float s = float(t.c)*g[t.q];
        g[t.a] += s*v[t.b];
        g[t.b] += s*v[t.a];
    });
    for (int lm = 0; lm < 16; ++lm) A_adj[local_row(lm, n, N)*C + k] += g[lm];
}


__global__ void reverse_fill(size_t n,float value,float* data){const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)data[i]=value;}
__global__ void reverse_mask(int N,int C,const int* types,int species,const float* input,float* output)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(N)*4*C)return;
    const size_t scalars=size_t(N)*C;const int n=i<scalars?i/C:(i-scalars)/(3*C);
    output[i]=types[n]==species?input[i]:0.f;
}
__global__ void source_multipole_seed(int N,const double* density_adj,float* multipole_adj)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    multipole_adj[i]=0;
    for(int lm=1;lm<4;++lm)multipole_adj[size_t(N)+size_t(i)*3+lm-1]=float(density_adj[size_t(i)*4+lm]);
}
__global__ void source_modulation_reverse(int N,const float* adj,const float* weights,float* hidden_adj)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;constexpr int C=local_channels;
    if(i>=size_t(N)*C)return;const int n=i/C,u=i%C;
    for(int lm=0;lm<4;++lm){const size_t r=local_row(lm,n,N)*C+u;hidden_adj[r]+=adj[r]*weights[(size_t(lm>0)*N+n)*C+u];}
}
__device__ inline void add_edge_gradient(int edge,float* gradient,double* edge_adj)
{
    __shared__ float sums[3][4];
    for(int d=0;d<3;++d){const float value=source_warp_sum(gradient[d]);if((threadIdx.x&31)==0)sums[d][threadIdx.x/32]=value;}
    __syncthreads();
    if(threadIdx.x==0)for(int d=0;d<3;++d){float value=0;for(int w=0;w<4;++w)value+=sums[d][w];edge_adj[size_t(edge)*3+d]+=double(value);}
}
__global__ void source_transfer_reverse(int N,const int* senders,const int* receivers,const float4* geometry,
    const float* Y,const float* Y_grad,const int* intervals,const float* coordinate,const float* radial,const float* grid,
    const int* spec,const float* coefficients,int terms,const float* readout,const float* scale,
    const float* sender_features,const float* receiver_features,const double* density_adj,
    double fx,double fy,double fz,const double* vectors,float* sender_adj,float* receiver_adj,double* edge_adj)
{
    constexpr int C=local_channels,F=2*C;
    const int e=blockIdx.x,u=threadIdx.x,s=senders[e],r=receivers[e],k=intervals[e];
    float R[2],dR[2];for(int p=0;p<2;++p){cubic(radial+size_t(k)*4*F,F,p*C+u,coordinate[e],R[p],dR[p]);dR[p]*=grid[2];}
    const double field=fx*vectors[size_t(e)*3]+fy*vectors[size_t(e)*3+1]+fz*vectors[size_t(e)*3+2];
    const float upstream=float(double(scale[0])*(density_adj[size_t(r)*4]-density_adj[size_t(s)*4]+field))*readout[u];
    float dr=0,gy[16]={};
    for(int t=0;t<terms;++t){
        const int lm=spec[4*t+1]/C,b=spec[4*t+2],p=spec[4*t+3]/C;
        const float g=upstream*coefficients[t],angular=Y[size_t(e)*16+b];
        const size_t si=local_row(lm,s,N)*C+u,ri=local_row(lm,r,N)*C+u;
        const float value=sender_features[si]+receiver_features[ri];
        const float feature_adj=g*angular*R[p];atomicAdd(sender_adj+si,feature_adj);atomicAdd(receiver_adj+ri,feature_adj);
        dr+=g*angular*value*dR[p];gy[b]+=g*value*R[p];
    }
    const float4 q=geometry[e];const float unit[3]={q.x,q.y,q.z};float gradient[3];
    for(int d=0;d<3;++d){gradient[d]=dr*unit[d];for(int m=0;m<16;++m)gradient[d]+=gy[m]*Y_grad[(size_t(e)*3+d)*16+m];}
    add_edge_gradient(e,gradient,edge_adj);
}

// Polar/foundation receiver batches, split for enough parallelism with 128
// channels. The loader verifies the four scalar coupling indices before using
// this specialization; exported CG coefficients and learned splines stay runtime data.
template<int Splits>
__global__ void source_transfer_reverse_batched(int N,const int* starts,const int* order,
    const int* senders,const float4* geometry,const float* Y,const float* Y_grad,
    const int* intervals,const float* coordinate,const float* radial,const float* grid,
    const float* coefficients,const float* readout,const float* scale,
    const float* sender_features,const float* receiver_features,const double* density_adj,
    double fx,double fy,double fz,const double* vectors,float* sender_adj,float* receiver_adj,double* edge_adj)
{
    constexpr int C=local_channels,F=2*C,Batch=32;
    struct Edge {float y[4],dy[12],t,seed;float4 geometry;int e,s,k;};
    __shared__ Edge staged[Batch];
    const int r=blockIdx.x,u=threadIdx.x;
    const int first_edge=starts[r],edges=starts[r+1]-first_edge;
    const int begin=first_edge+int((long long)edges*blockIdx.y/Splits);
    const int end=first_edge+int((long long)edges*(blockIdx.y+1)/Splits);
    float receiver[4],receiver_gradient[4]={},cg[4];
    #pragma unroll
    for(int lm=0;lm<4;++lm){receiver[lm]=receiver_features[local_row(lm,r,N)*C+u];cg[lm]=coefficients[lm];}
    const float weight=readout[u];
    for(int first=begin;first<end;first+=Batch){
        const int count=min(Batch,end-first);
        __syncthreads();
        if(u<count){
            const int e=order[first+u];auto& q=staged[u];
            q.e=e;q.s=senders[e];q.k=intervals[e];q.t=coordinate[e];q.geometry=geometry[e];
            const double field=fx*vectors[size_t(e)*3]+fy*vectors[size_t(e)*3+1]+fz*vectors[size_t(e)*3+2];
            q.seed=float(double(scale[0])*(density_adj[size_t(r)*4]-density_adj[size_t(q.s)*4]+field));
            #pragma unroll
            for(int m=0;m<4;++m){
                q.y[m]=Y[size_t(e)*16+m];
                for(int d=0;d<3;++d)q.dy[d*4+m]=Y_grad[(size_t(e)*3+d)*16+m];
            }
        }
        __syncthreads();
        for(int j=0;j<count;++j){
            const auto& q=staged[j];float R[2],dR[2];
            #pragma unroll
            for(int p=0;p<2;++p){cubic(radial+size_t(q.k)*4*F,F,p*C+u,q.t,R[p],dR[p]);dR[p]*=grid[2];}
            const float upstream=q.seed*weight;float dr=0,gy[4];
            static_for<4>([&](auto index){
                constexpr int lm=index,p=lm>0;
                const float g=upstream*cg[lm],angular=q.y[lm];
                const size_t si=local_row(lm,q.s,N)*C+u;
                const float value=sender_features[si]+receiver[lm],feature_adj=g*angular*R[p];
                atomicAdd(sender_adj+si,feature_adj);receiver_gradient[lm]+=feature_adj;
                dr+=g*angular*value*dR[p];gy[lm]=g*value*R[p];
            });
            const float unit[3]={q.geometry.x,q.geometry.y,q.geometry.z};float gradient[3];
            #pragma unroll
            for(int d=0;d<3;++d){
                gradient[d]=dr*unit[d];
                #pragma unroll
                for(int m=0;m<4;++m)gradient[d]+=gy[m]*q.dy[d*4+m];
            }
            add_edge_gradient(q.e,gradient,edge_adj);
            // All four warps finish the shared edge reduction before reuse.
            __syncthreads();
        }
    }
    // Four neighbor partitions share a receiver; reduce once per partition,
    // rather than once per edge, while keeping sender updates atomic.
    #pragma unroll
    for(int lm=0;lm<4;++lm)atomicAdd(receiver_adj+local_row(lm,r,N)*C+u,receiver_gradient[lm]);
}

template<int Layer,bool Mixed,bool Ghosts=false>
__global__ void local_convolution_reverse(int N,const int* senders,const int* receivers,const float4* geometry,
    const float* Y,const float* Y_grad,const int* intervals,const float* coordinate,const float* radial,const __half2* radial_half,
    const float* grid,const float* coefficients,const float* up,const float* message_adj,float* up_adj,double* edge_adj,int input_nodes=0)
{
    const int all=Ghosts?input_nodes:N;
    using Conv=LocalConvolution<Layer>;constexpr int C=local_channels,F=Conv::paths*C;
    const int e=blockIdx.x,c=threadIdx.x,s=senders[e],r=receivers[e],k=intervals[e];
    float R[Conv::paths],dR[Conv::paths],gR[Conv::paths]={},gY[16]={},gH[Layer==0?1:4]={};
    for(int p=0;p<Conv::paths;++p){
        if constexpr(Mixed)cubic(radial+size_t(k)*2*F,radial_half+size_t(k)*F,F,p*C+c,coordinate[e],R[p],dR[p]);
        else cubic(radial+size_t(k)*4*F,F,p*C+c,coordinate[e],R[p],dR[p]);
    }
    static_for<Conv::count>([&](auto index){
        constexpr Term t=Conv::terms[index];
        const float g=coefficients[index]*message_adj[message_row(t.lme,r,N,Layer)*C+c];
        const float h=up[local_row(t.lm2,s,all)*C+c],y=Y[size_t(e)*16+t.lm1];
        gR[t.path]+=g*y*h;gY[t.lm1]+=g*R[t.path]*h;
        if constexpr(Layer>0)gH[t.lm2]+=g*y*R[t.path];
    });
    float radial_adj=0;for(int p=0;p<Conv::paths;++p)radial_adj+=dR[p]*gR[p];radial_adj*=grid[2];
    const float4 q=geometry[e];const float unit[3]={q.x,q.y,q.z};float gradient[3];
    for(int d=0;d<3;++d){gradient[d]=radial_adj*unit[d];for(int m=0;m<16;++m)gradient[d]+=gY[m]*Y_grad[(size_t(e)*3+d)*16+m];}
    add_edge_gradient(e,gradient,edge_adj);
    if constexpr(Layer>0)for(int lm=0;lm<4;++lm)atomicAdd(up_adj+local_row(lm,s,all)*C+c,gH[lm]);
}

// Foundation's tp_reverse factors terms by (path, input component, harmonic).
// The index groups are compile-time data; LSC's exported coefficients remain
// runtime data and already contain the required spherical-harmonic normalization.
template<class Conv> constexpr bool local_reverse_first_term(int q)
{
    const Term t=Conv::terms[q];
    for(int i=0;i<q;++i){const Term u=Conv::terms[i];
        if(u.path==t.path and u.lm2==t.lm2 and u.lm1==t.lm1)return false;}
    return true;
}
template<class Conv> constexpr int local_reverse_next_term(int q)
{
    const Term t=Conv::terms[q];
    for(int i=q+1;i<Conv::count;++i){const Term u=Conv::terms[i];
        if(u.path==t.path and u.lm2==t.lm2 and u.lm1==t.lm1)return i;}
    return -1;
}
template<class Conv> constexpr bool local_reverse_has_input(int path,int lm2)
{
    for(const Term& t:Conv::terms)if(t.path==path and t.lm2==lm2)return true;
    return false;
}
template<class Conv,int Q>
__device__ __forceinline__ float local_reverse_weight(const float* coefficients,const float* adj)
{
    constexpr Term t=Conv::terms[Q];
    constexpr int next=local_reverse_next_term<Conv>(Q);
    const float w=coefficients[Q]*adj[t.lme];
    if constexpr(next<0)return w;
    else return w+local_reverse_weight<Conv,next>(coefficients,adj);
}

// Polar's receiver batching with foundation's factorized tensor-product reverse.
// Stream one radial path at a time to avoid keeping R, dR and gR arrays live.
// Preserve the four-warp edge reduction and atomic sender-feature updates.
template<bool Mixed,bool Ghosts=false>
__global__ void __launch_bounds__(local_channels,3) local_convolution_reverse_batched(
    int N,int M,const int* starts,const int* order,const int* senders,
    const float4* geometry,const float* Y,const float* Y_grad,const int* intervals,
    const float* coordinate,const float* radial,const __half2* radial_half,const float* grid,
    const float* coefficients,const float* up,const float* message_adj,float* up_adj,double* edge_adj)
{
    using Conv=LocalConvolution<1>;constexpr int C=local_channels,F=Conv::paths*C,Batch=32;
    struct Edge {float y[16],dy[48],t;float4 geometry;int e,s,k;};
    __shared__ Edge staged[Batch];
    const int r=blockIdx.x,c=threadIdx.x,all=Ghosts?M:N;
    float adj[Conv::width];
    #pragma unroll
    for(int q=0;q<Conv::width;++q)adj[q]=message_adj[message_row(q,r,N,1)*C+c];
    for(int first=starts[r];first<starts[r+1];first+=Batch){
        const int count=min(Batch,starts[r+1]-first);
        __syncthreads();
        if(c<count){
            const int e=order[first+c];auto& q=staged[c];
            q.e=e;q.s=senders[e];q.k=intervals[e];q.t=coordinate[e];q.geometry=geometry[e];
            for(int lm=0;lm<16;++lm)q.y[lm]=Y[size_t(e)*16+lm];
            for(int j=0;j<48;++j)q.dy[j]=Y_grad[size_t(e)*48+j];
        }
        __syncthreads();
        for(int j=0;j<count;++j){
            const auto& q=staged[j];float gY[16]={},gH[4]={},h[4],radial_adj=0;
            for(int lm=0;lm<4;++lm)h[lm]=up[local_row(lm,q.s,all)*C+c];
            static_for<Conv::paths>([&](auto path){
                constexpr int P=path;float R,dR,gR=0;
                if constexpr(Mixed)cubic(radial+size_t(q.k)*2*F,radial_half+size_t(q.k)*F,F,P*C+c,q.t,R,dR);
                else cubic(radial+size_t(q.k)*4*F,F,P*C+c,q.t,R,dR);
                static_for<4>([&](auto component){
                    constexpr int LM=component;
                    if constexpr(local_reverse_has_input<Conv>(P,LM)){
                        const float rh=R*h[LM];float s=0;
                        static_for<Conv::count>([&](auto index){
                            constexpr Term t=Conv::terms[index];
                            if constexpr(t.path==P and t.lm2==LM and local_reverse_first_term<Conv>(index)){
                                const float w=local_reverse_weight<Conv,index>(coefficients,adj);
                                s+=w*q.y[t.lm1];gY[t.lm1]+=w*rh;
                            }
                        });
                        gH[LM]+=R*s;gR+=h[LM]*s;
                    }
                });
                radial_adj+=dR*gR;
            });
            radial_adj*=grid[2];
            const float unit[3]={q.geometry.x,q.geometry.y,q.geometry.z};float gradient[3];
            for(int d=0;d<3;++d){gradient[d]=radial_adj*unit[d];for(int m=0;m<16;++m)gradient[d]+=gY[m]*q.dy[d*16+m];}
            add_edge_gradient(q.e,gradient,edge_adj);
            for(int lm=0;lm<4;++lm)atomicAdd(up_adj+local_row(lm,q.s,all)*C+c,gH[lm]);
            // add_edge_gradient's shared reduction must be consumed before reuse.
            __syncthreads();
        }
    }
}

__global__ void reverse_field_nodes(int N,const int* types,const float* formal,double fx,double fy,double fz,
                                     double* density_adj,double* position_adj)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;const double field[3]={fx,fy,fz};constexpr int xyz[3]={3,1,2};
    for(int d=0;d<3;++d){density_adj[size_t(i)*4+xyz[d]]+=field[d];position_adj[size_t(i)*3+d]+=double(formal[types[i]])*field[d];}
}
__global__ void reverse_field_edges(int E,const double* flux,double fx,double fy,double fz,double* edge_adj)
{
    const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E)return;
    edge_adj[size_t(e)*3]=flux[e]*fx;edge_adj[size_t(e)*3+1]=flux[e]*fy;edge_adj[size_t(e)*3+2]=flux[e]*fz;
}
__global__ void reverse_force_nodes(int N,const int* types,const float* formal,const double* positions,
    const int* incoming_starts,const int* incoming_edges,const int* outgoing_starts,const int* outgoing_edges,
    const double* position_adj,const double* edge_adj,bool real,double fx,double fy,double fz,
    size_t stride,double* terms,double* forces,double* position_forces,int* error)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;const double field[3]={fx,fy,fz};
    for(int d=0;d<3;++d){
        double f=-position_adj[size_t(i)*3+d];position_forces[size_t(i)*3+d]=f;
        for(int j=incoming_starts[i];j<incoming_starts[i+1];++j)f-=edge_adj[size_t(incoming_edges[j])*3+d];
        for(int j=outgoing_starts[i];j<outgoing_starts[i+1];++j)f+=edge_adj[size_t(outgoing_edges[j])*3+d];
        forces[size_t(i)*3+d]=f;if(not source_finite(f))atomicOr(error,1);
        const double gradient=real?position_adj[size_t(i)*3+d]:double(formal[types[i]])*field[d];
        for(int a=0;a<3;++a)terms[size_t(3*a+d)*stride+i]=positions[size_t(i)*3+a]*gradient;
    }
}
__global__ void reverse_force_edges(int N,int E,const double* vectors,const double* adj,
                                     size_t stride,double* terms,double* forces)
{
    const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E)return;
    for(int d=0;d<3;++d){forces[size_t(e)*3+d]=-adj[size_t(e)*3+d];
        for(int a=0;a<3;++a)terms[size_t(3*a+d)*stride+N+e]=vectors[size_t(e)*3+a]*adj[size_t(e)*3+d];}
}
// The same per-atom partition used by the host LSC pair. Keep its optional
// output on-device; ordinary global-energy MD does not launch these kernels.
__global__ void node_energy_partition(int N,const int* types,const float* formal,
    const double* positions,const double* local,const double* rho,const double* gradient,
    double fx,double fy,double fz,double* energies)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    const double field[3]={fx,fy,fz};constexpr int cartesian[3]={3,1,2};
    double value=local[i];
    for(int lm=0;lm<4;++lm){
        double g=gradient[size_t(i)*4+lm];
        if(lm>0)g-=field[lm==1?1:(lm==2?2:0)];
        value+=0.5*rho[size_t(i)*4+lm]*g;
    }
    for(int d=0;d<3;++d)value+=field[d]*(double(formal[types[i]])*positions[size_t(i)*3+d]
        +rho[size_t(i)*4+cartesian[d]]);
    energies[i]=value;
}
__global__ void edge_energy_partition(int E,const int* senders,const int* receivers,
    const double* flux,const double* vectors,double fx,double fy,double fz,double* energies)
{
    const int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E)return;
    const double value=0.5*flux[e]*(fx*vectors[size_t(e)*3]+fy*vectors[size_t(e)*3+1]+fz*vectors[size_t(e)*3+2]);
    atomicAdd(energies+senders[e],value);atomicAdd(energies+receivers[e],value);
}
} // namespace lsc_cuda
