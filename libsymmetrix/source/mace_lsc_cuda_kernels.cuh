#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace lsc_cuda {
// Same integral-normalized real harmonics and ordering as MACE Polar CUDA.
__host__ __device__ inline void spherical_harmonics(float x, float y, float z, float* Y, float* Y_grad) {
    const float r_inv = 1/sqrtf(x*x + y*y + z*z);
    const float xx = x*x;
    const float yy = y*y;
    const float zz = z*z;
    const float S[16] = {1, y, z, x, x*y, y*z, 2*zz - xx - yy, x*z, xx - yy, y*(3*xx - yy), x*y*z, y*(4*zz - xx - yy),
                         z*(2*zz - 3*xx - 3*yy), x*(4*zz - xx - yy), z*(xx - yy), x*(xx - 3*yy)};
    const float dS[3][16] = {
        {0, 0, 0, 1, y, 0, -2*x, z, 2*x, 6*x*y, y*z, -2*x*y, -6*x*z, 4*zz - 3*xx - yy, 2*x*z, 3*xx - 3*yy},
        {0, 1, 0, 0, x, z, -2*y, 0, -2*y, 3*xx - 3*yy, x*z, 4*zz - xx - 3*yy, -6*y*z, -2*x*y, -2*y*z, -6*x*y},
        {0, 0, 1, 0, 0, y, 4*z, x, 0, 0, x*y, 8*y*z, 6*zz - 3*xx - 3*yy, 8*x*z, xx - yy, 0}};
    const float norm[16] = {0.28209479177387814f, 0.4886025119029199f, 0.4886025119029199f, 0.4886025119029199f,
                            1.0925484305920792f, 1.0925484305920792f, 0.31539156525252005f, 1.0925484305920792f,
                            0.5462742152960396f, 0.5900435899266435f, 2.890611442640554f, 0.4570457994644658f,
                            0.3731763325901154f, 0.4570457994644658f, 1.445305721320277f, 0.5900435899266435f};
    const float u[3] = {x*r_inv, y*r_inv, z*r_inv};
    const float r_inv_l[4] = {1, r_inv, r_inv*r_inv, r_inv*r_inv*r_inv};
    for (int lm = 0; lm < 16; ++lm) {
        const int l = lm < 1 ? 0 : (lm < 4 ? 1 : (lm < 9 ? 2 : 3));
        const float c = norm[lm]*r_inv_l[l];
        Y[lm] = c*S[lm];
        if (Y_grad)
            for (int a = 0; a < 3; ++a) Y_grad[a*16 + lm] = c*(dS[a][lm] - l*S[lm]*u[a]*r_inv);
    }
}


__global__ void validate_types(const int* types, int n, int species, int* error)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n and (types[i]<0 or types[i]>=species)) atomicOr(error,1);
}

template<bool Ghosts=false>
__global__ void geometry_kernel(const double* xyz, const int* senders, const int* receivers,
    int n, int ecount, double rmin, double rcut, int knots, const float* grid,
    float4* geometry, float* Y, float* dY, int* intervals, float* coordinate,
    int* incoming_count, int* outgoing_count, int* ids, int* error, int sender_nodes=0)
{
    const int e=blockIdx.x*blockDim.x+threadIdx.x;
    if(e>=ecount)return;
    const int s=senders[e], t=receivers[e];
    if(s<0 or s>=(Ghosts?sender_nodes:n) or t<0 or t>=n){atomicOr(error,2);return;}
    const double x=xyz[size_t(e)*3],y=xyz[size_t(e)*3+1],z=xyz[size_t(e)*3+2];
    // Bit checks retain NaN/Inf rejection even in builds using fast-math.
    if((__double_as_longlong(x)&0x7ff0000000000000LL)==0x7ff0000000000000LL or
       (__double_as_longlong(y)&0x7ff0000000000000LL)==0x7ff0000000000000LL or
       (__double_as_longlong(z)&0x7ff0000000000000LL)==0x7ff0000000000000LL){atomicOr(error,4);return;}
    const double r=sqrt(x*x+y*y+z*z);
    if(not (r>=rmin and r<=rcut) or r==0){atomicOr(error,8);return;}
    // Polar uses double input coordinates and stores unit directions/r in FP32.
    const float4 g=make_float4(float(x/r),float(y/r),float(z/r),float(r));
    geometry[e]=g;
    float values[16],grads[48];
    spherical_harmonics(g.x,g.y,g.z,values,grads);
    for(int m=0;m<16;++m)Y[size_t(e)*16+m]=values[m];
    // Helper differentiates w.r.t. its unit-vector input: chain rule adds 1/r.
    for(int a=0;a<3;++a)for(int m=0;m<16;++m)dY[(size_t(e)*3+a)*16+m]=grads[a*16+m]/g.w;
    const float tgrid=(g.w-grid[0])*grid[2];
    const int interval=min(max(int(tgrid),0),knots-2);
    intervals[e]=interval;coordinate[e]=tgrid-interval-0.5f;
    atomicAdd(incoming_count+t,1);atomicAdd(outgoing_count+s,1);ids[e]=e;
}
// Centered cubic helpers follow mace_polar_cuda.cpp. Keep them callable from
// the future fused convolution kernels as well as this diagnostic evaluator.
__device__ __forceinline__ void cubic(const float* b,int n,int f,float x,float& v,float& d)
{
    const float c1=b[n+f],c2=b[2*n+f],c3=b[3*n+f];
    v=b[f]+x*(c1+x*(c2+x*c3));d=c1+x*(2*c2+3*x*c3);
}
__device__ __forceinline__ void cubic(const float* b,const __half2* h,int n,int f,float x,float& v,float& d)
{
    const float2 c23=__half22float2(h[f]);const float c1=b[n+f];
    v=b[f]+x*(c1+x*(c23.x+x*c23.y));d=c1+x*(2*c23.x+3*x*c23.y);
}

template<bool Mixed>
__global__ void spline_kernel(const int* intervals,const float* coordinate,size_t count,int outputs,
    const float* coefficients,const __half2* c23,const float* grid,float* values,float* derivatives)
{
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const size_t e=i/outputs;const int f=i%outputs,k=intervals[e];float v,d;
    if constexpr(Mixed) cubic(coefficients+size_t(k)*2*outputs,c23+size_t(k)*outputs,outputs,f,coordinate[e],v,d);
    else cubic(coefficients+size_t(k)*4*outputs,outputs,f,coordinate[e],v,d);
    values[i]=v;derivatives[i]=d*grid[2];
}
} // namespace lsc_cuda
