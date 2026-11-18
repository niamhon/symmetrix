/*
   LSC one-process GPU mesh. B-splines, spreading, cuFFT and gathering adapted
   from pair_symmetrix_mace_polar_cuda_kokkos.cpp, wcwitt/symmetrix
   commit b907f04026f2dda4c43d66511a7c37db8a82a7d4.
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. Distributed under the GNU General Public
   License; see pair_symmetrix/LICENSE in this repository.
*/
#include "mace_lsc_cuda_mesh.hpp"
#include "mace_lsc.hpp"
#include "mace_lsc_cuda.hpp"
#include "mace_lsc_mesh_common.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
namespace {
void check(cudaError_t status,const char* what) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("LSCCudaMesh: ")+what+": "+cudaGetErrorString(status));
}
void fft_check(cufftResult status,const char* what) {
    if(status!=CUFFT_SUCCESS)throw std::runtime_error(std::string("LSCCudaMesh: ")+what+": cuFFT error "+std::to_string(status));
}
struct Grid {
    int shape[3],bounds[3],order;
    double b[9],J[9],volume,cutoff2,sigma2,prefactor,a0,a1,self[4];
    bool include_self;
};
Grid grid_parameters(const lsc_mesh::Geometry& g,const LSCElectrostaticsParameters& p,int order) {
    Grid out{};out.order=order;out.volume=g.volume;out.cutoff2=p.kspace_cutoff*p.kspace_cutoff;
    out.sigma2=p.sigma2;out.prefactor=p.field_constant/g.volume;
    out.a0=p.a[0]*lsc_mesh::Y00;out.a1=p.a[1]*lsc_mesh::Y1;out.include_self=p.include_self;
    for(int d=0;d<3;++d) {
        out.shape[d]=g.shape[d];out.bounds[d]=g.bounds[d];
        for(int a=0;a<3;++a) {out.b[3*d+a]=g.reciprocal[d][a];out.J[3*d+a]=g.J[d][a];}
    }
    std::copy(p.reciprocal_self.begin(),p.reciprocal_self.end(),out.self);return out;
}
// Same reversed cardinal stencil as Polar and the CPU mesh.
__device__ void cardinal(int P,double f,double* t) {
    double M[11][11]={};M[1][0]=1;
    for(int n=2;n<=P;++n)for(int j=0;j<n;++j)
        M[n][j]=((f+j)*M[n-1][j]+(n-f-j)*(j>0?M[n-1][j-1]:0.0))/(n-1);
    for(int k=0;k<P;++k) {int j=P-1-k;t[k]=M[P][j];
        t[P+k]=M[P-1][j]-(j>0?M[P-1][j-1]:0.0);
        t[2*P+k]=M[P-2][j]-2*(j>0?M[P-2][j-1]:0.0)+(j>1?M[P-2][j-2]:0.0);}
}
__device__ int wrap(int x,int n){return (x%n+n)%n;}
__device__ size_t index(Grid g,int x,int y,int z){return (size_t(wrap(z,g.shape[2]))*g.shape[1]+wrap(y,g.shape[1]))*g.shape[0]+wrap(x,g.shape[0]);}
__global__ void particles(int N,Grid g,const double* xyz,int* base,double* theta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    for(int d=0;d<3;++d) {
        double f=0;for(int a=0;a<3;++a)f+=xyz[3*size_t(i)+a]*g.b[3*d+a]/(2*lsc_mesh::PI);
        f-=floor(f);double u=f*g.shape[d];int i0=int(floor(u));
        base[3*i+d]=i0-(g.order-1)+g.order/2;
        cardinal(g.order,u-i0,theta+(size_t(i)*9+3*d)*g.order);
    }
}
__global__ void spline_modulus(Grid g,double* mod) {
    int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot>=g.shape[0]+g.shape[1]+g.shape[2])return;
    int d=0,k=slot;while(k>=g.shape[d])k-=g.shape[d++];
    double t[30];cardinal(g.order,0,t);double re=0,im=0;
    for(int j=0;j<=g.order-2;++j){double phase=2*lsc_mesh::PI*k*j/g.shape[d];
        re+=t[g.order-2-j]*cos(phase);im+=t[g.order-2-j]*sin(phase);}
    mod[slot]=re*re+im*im;
}
__global__ void spread(size_t work,Grid g,const double* rho,const int* base,const double* theta,cufftDoubleComplex* data) {
    size_t slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=work)return;
    int P=g.order,i=slot/(P*P),z=slot/P%P,y=slot%P;
    double c=g.a0*rho[4*size_t(i)],m[3]={g.a1*rho[4*size_t(i)+3],g.a1*rho[4*size_t(i)+1],g.a1*rho[4*size_t(i)+2]},mu[3]={};
    for(int d=0;d<3;++d)for(int a=0;a<3;++a)mu[d]+=g.J[3*d+a]*m[a];
    const auto tx=theta+size_t(i)*9*P,ty=tx+3*P,tz=ty+3*P;
    for(int x=0;x<P;++x) {
        double value=c*tx[x]*ty[y]*tz[z]+mu[0]*tx[P+x]*ty[y]*tz[z]+mu[1]*tx[x]*ty[P+y]*tz[z]+mu[2]*tx[x]*ty[y]*tz[P+z];
        atomicAdd(&data[index(g,base[3*i]+x,base[3*i+1]+y,base[3*i+2]+z)].x,value);
    }
}
__device__ bool active_mode(int* mode,Grid g) {
    int first=0;while(first<3 && mode[first]==0)++first;if(first==3)return false;
    int n[3];for(int a=0;a<3;++a)n[a]=mode[first]<0?-mode[a]:mode[a];
    return n[0]>=0 && n[0]<g.bounds[0] && n[1]>=(n[0]==0?0:-g.bounds[1]) && n[1]<g.bounds[1]
        && n[2]>=(n[0]==0 && n[1]==0?1:-g.bounds[2]) && n[2]<g.bounds[2];
}
struct Strain9 {double v[9];};
struct AddStrain {
    __device__ Strain9 operator()(const Strain9& a,const Strain9& b)const {
        Strain9 out;for(int k=0;k<9;++k)out.v[k]=a.v[k]+b.v[k];return out;
    }
};
__global__ void convolve(int points,Grid g,const double* mod,cufftDoubleComplex* data,
    double* energy,double* strain_terms,int stride) {
    using Reduce=cub::BlockReduce<Strain9,128>;
    __shared__ typename Reduce::TempStorage storage;
    int slot=blockIdx.x*blockDim.x+threadIdx.x;Strain9 value{};
    if(slot<points) {
        int ix=slot%g.shape[0],iy=slot/g.shape[0]%g.shape[1],iz=slot/(g.shape[0]*g.shape[1]);
        int mode[3]={ix,iy,iz};for(int d=0;d<3;++d)if(2*mode[d]>g.shape[d])mode[d]-=g.shape[d];
        double k[3]={};for(int a=0;a<3;++a)for(int d=0;d<3;++d)k[a]+=mode[d]*g.b[3*d+a];
        double norm=k[0]*k[0]+k[1]*k[1]+k[2]*k[2];
        bool active=active_mode(mode,g) && norm>0 && norm<=g.cutoff2;
        double factor=active?g.prefactor*exp(-g.sigma2*norm)/norm/(mod[ix]*mod[g.shape[0]+iy]*mod[g.shape[0]+g.shape[1]+iz]):0;
        double re=data[slot].x,im=data[slot].y;energy[slot]=.5*factor*(re*re+im*im);
        if(strain_terms && active)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
            value.v[3*a+b]=energy[slot]*(2*(g.sigma2+1/norm)*k[a]*k[b]-(a==b?1.:0.));
        data[slot].x=re*factor;data[slot].y=im*factor;
    }
    // Every thread participates, including the last partial block.
    if(strain_terms){auto sum=Reduce(storage).Reduce(value,AddStrain{});
        if(threadIdx.x==0)for(int ab=0;ab<9;++ab)strain_terms[size_t(ab)*stride+blockIdx.x]=sum.v[ab];}
}
__global__ void gather(int N,Grid g,const cufftDoubleComplex* data,const int* base,const double* theta,
    const double* rho,double* dr,double* self_terms,double* dx,double* strain_terms,int stride,int blocks) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    int P=g.order;const auto tx=theta+size_t(i)*9*P,ty=tx+3*P,tz=ty+3*P;
    double q=0,grad[3]={},Hu[3][3]={};
    for(int z=0;z<P;++z)for(int y=0;y<P;++y)for(int x=0;x<P;++x) {
        double v=data[index(g,base[3*i]+x,base[3*i+1]+y,base[3*i+2]+z)].x;
        q+=tx[x]*ty[y]*tz[z]*v;grad[0]+=tx[P+x]*ty[y]*tz[z]*v;
        grad[1]+=tx[x]*ty[P+y]*tz[z]*v;grad[2]+=tx[x]*ty[y]*tz[P+z]*v;
        if(dx){Hu[0][0]+=tx[2*P+x]*ty[y]*tz[z]*v;Hu[1][1]+=tx[x]*ty[2*P+y]*tz[z]*v;
            Hu[2][2]+=tx[x]*ty[y]*tz[2*P+z]*v;Hu[0][1]+=tx[P+x]*ty[P+y]*tz[z]*v;
            Hu[0][2]+=tx[P+x]*ty[y]*tz[P+z]*v;Hu[1][2]+=tx[x]*ty[P+y]*tz[P+z]*v;}
    }
    dr[4*size_t(i)]=g.a0*q;constexpr int cart[3]={3,1,2};
    for(int a=0;a<3;++a){double value=0;for(int d=0;d<3;++d)value+=g.a1*g.J[3*d+a]*grad[d];dr[4*size_t(i)+cart[a]]=value;}
    if(dx || strain_terms) {
        Hu[1][0]=Hu[0][1];Hu[2][0]=Hu[0][2];Hu[2][1]=Hu[1][2];
        double grad_cart[3]={},H[3][3]={},m[3]={g.a1*rho[4*size_t(i)+3],g.a1*rho[4*size_t(i)+1],g.a1*rho[4*size_t(i)+2]};
        for(int a=0;a<3;++a)for(int d=0;d<3;++d){grad_cart[a]+=g.J[3*d+a]*grad[d];
            if(dx)for(int b=0;b<3;++b)for(int e=0;e<3;++e)H[a][b]+=g.J[3*d+a]*Hu[d][e]*g.J[3*e+b];}
        for(int a=0;a<3;++a){
            if(dx){double value=g.a0*rho[4*size_t(i)]*grad_cart[a];for(int b=0;b<3;++b)value+=H[a][b]*m[b];dx[3*size_t(i)+a]=value;}
            // Fractional sites stay fixed under affine cell/position strain;
            // fixed Cartesian dipoles change their grid-coordinate assignment.
            if(strain_terms)for(int b=0;b<3;++b)strain_terms[size_t(3*a+b)*stride+blocks+i]=-m[a]*grad_cart[b];
        }
    }
    double correction=0;
    if(!g.include_self)for(int a=0;a<4;++a){double density=rho[4*size_t(i)+a];
        correction-=.5*density*density*g.self[a];dr[4*size_t(i)+a]-=density*g.self[a];}
    self_terms[i]=correction;
}
__global__ void total_energy(const double* components,double* energy) {if(threadIdx.x==0)*energy=components[0]+components[1];}
} // namespace

struct LSCCudaMesh::Workspace {
    struct Allocation {void* pointer=nullptr;size_t capacity=0;};
    std::mutex mutex;
    int device_id;
    cudaStream_t last_stream=nullptr,diagnostic_stream=nullptr;
    bool used=false;
    std::array<int,3> shape{};
    cufftHandle plan=0;
    std::map<std::string,Allocation> arrays;
    explicit Workspace(int device):device_id(device){check(cudaStreamCreateWithFlags(&diagnostic_stream,cudaStreamNonBlocking),"create diagnostic stream");}
    void active(){check(cudaSetDevice(device_id),"select mesh device");}
    void wait(){if(used)check(cudaStreamSynchronize(last_stream),"finish previous mesh work");}
    void begin(cudaStream_t stream){active();if(used && stream!=last_stream)wait();last_stream=stream;used=true;}
    ~Workspace(){int old=-1;cudaGetDevice(&old);cudaSetDevice(device_id);
        if(used)cudaStreamSynchronize(last_stream);if(plan)cufftDestroy(plan);
        for(auto& [name,a]:arrays)if(a.pointer)cudaFree(a.pointer);
        if(diagnostic_stream)cudaStreamDestroy(diagnostic_stream);if(old>=0)cudaSetDevice(old);}
    template<class T>T* buffer(const char* name,size_t count){auto& a=arrays[name];
        if(count>std::numeric_limits<size_t>::max()/sizeof(T))throw std::invalid_argument("LSCCudaMesh: excessive buffer size");
        size_t bytes=count*sizeof(T);if(bytes>a.capacity){wait();if(a.pointer)check(cudaFree(a.pointer),"free mesh buffer");a.pointer=nullptr;a.capacity=0;
            check(cudaMalloc(&a.pointer,bytes),"allocate mesh buffer");a.capacity=bytes;}
        return static_cast<T*>(a.pointer);}
    void configure(const std::array<int,3>& dims){if(shape==dims && plan)return;wait();if(plan){fft_check(cufftDestroy(plan),"destroy FFT plan");plan=0;}
        fft_check(cufftPlan3d(&plan,dims[2],dims[1],dims[0],CUFFT_Z2Z),"create FFT plan");shape=dims;}
    void sum(const double* input,double* output,int count,cudaStream_t stream){if(!count){check(cudaMemsetAsync(output,0,sizeof(double),stream),"clear empty sum");return;}
        size_t bytes=0;check(cub::DeviceReduce::Sum(nullptr,bytes,input,output,count,stream),"size mesh reduction");
        auto temp=buffer<char>("reduce_temp",bytes);check(cub::DeviceReduce::Sum(temp,bytes,input,output,count,stream),"reduce mesh terms");}
};
bool lsc_cuda_mesh_compiled(){return true;}
LSCCudaMesh::LSCCudaMesh(double mesh_spacing,int spline_order):spacing(mesh_spacing),order(spline_order) {
    lsc_mesh::require(lsc_electrostatics_finite(spacing) && spacing>0,"spacing must be finite and positive");
    lsc_mesh::require(order>=4 && order<=10,"order must be between 4 and 10");
    check(cudaGetDevice(&device_id),"get mesh device");cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,device_id),"get mesh device properties");
    device_name=prop.name;workspace=std::make_unique<Workspace>(device_id);
}
LSCCudaMesh::~LSCCudaMesh()=default;
void LSCCudaMesh::evaluate_energy(const LSCCudaElectrostaticsInput& input,double* energy,double* dr){
    std::lock_guard<std::mutex> lock(workspace->mutex);evaluate_impl(input,energy,dr);
}
void LSCCudaMesh::evaluate(const LSCCudaElectrostaticsInput& input,const LSCCudaElectrostaticsOutput& output){
    lsc_mesh::require(output.strain_derivative && (!input.num_nodes || output.position_gradient),"missing derivative device buffers");
    std::lock_guard<std::mutex> lock(workspace->mutex);
    evaluate_impl(input,output.energy,output.density_gradient,output.position_gradient,output.strain_derivative);
}
void LSCCudaMesh::evaluate_impl(const LSCCudaElectrostaticsInput& input,double* energy,double* dr,double* dx,double* ds){
    lsc_mesh::require(input.num_nodes<=size_t(std::numeric_limits<int>::max()/100),"too many atoms");
    lsc_mesh::require(energy && (!input.num_nodes || (input.positions && input.density && dr)),"missing device buffers");
    auto g=lsc_mesh::geometry(input.context,spacing,order);auto p=grid_parameters(g,input.context.parameters,order);
    const int N=int(input.num_nodes),points=g.shape[0]*g.shape[1]*g.shape[2];auto stream=input.stream;
    workspace->begin(stream);bool changed=workspace->shape!=g.shape || !workspace->plan;
    workspace->configure(g.shape);if(changed)++plan_generation;grid_shape=g.shape;
    auto data=workspace->buffer<cufftDoubleComplex>("grid",points);
    auto theta=workspace->buffer<double>("theta",size_t(N)*9*order);auto base=workspace->buffer<int>("base",size_t(N)*3);
    auto mod=workspace->buffer<double>("bsmod",g.shape[0]+g.shape[1]+g.shape[2]);
    auto terms=workspace->buffer<double>("energy_terms",size_t(points)+N);auto components=workspace->buffer<double>("components",2);
    const int blocks=(points+127)/128,stride=blocks+N;
    auto strain_terms=ds?workspace->buffer<double>("strain_terms",size_t(stride)*9):nullptr;
    check(cudaMemsetAsync(data,0,size_t(points)*sizeof(cufftDoubleComplex),stream),"clear mesh");
    if(changed)spline_modulus<<<(g.shape[0]+g.shape[1]+g.shape[2]+127)/128,128,0,stream>>>(p,mod);
    if(N){particles<<<(N+127)/128,128,0,stream>>>(N,p,input.positions,base,theta);
        size_t work=size_t(N)*order*order;spread<<<(work+127)/128,128,0,stream>>>(work,p,input.density,base,theta,data);}
    fft_check(cufftSetStream(workspace->plan,stream),"set FFT stream");
    fft_check(cufftExecZ2Z(workspace->plan,data,data,CUFFT_FORWARD),"forward FFT");
    convolve<<<blocks,128,0,stream>>>(points,p,mod,data,terms,strain_terms,stride);
    fft_check(cufftExecZ2Z(workspace->plan,data,data,CUFFT_INVERSE),"inverse FFT");
    if(N)gather<<<(N+127)/128,128,0,stream>>>(N,p,data,base,theta,input.density,dr,terms+points,dx,strain_terms,stride,blocks);
    check(cudaGetLastError(),"launch mesh energy/potential kernels");
    workspace->sum(terms,components,points,stream);workspace->sum(terms+points,components+1,N,stream);
    if(ds)for(int ab=0;ab<9;++ab)workspace->sum(strain_terms+size_t(ab)*stride,ds+ab,stride,stream);
    total_energy<<<1,32,0,stream>>>(components,energy);check(cudaGetLastError(),"assemble mesh energy");
}
LSCMeshEnergyResult LSCCudaMesh::evaluate_energy(const MACELSC& model,std::span<const double> xyz,
    std::span<const double> rho,const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode){
    lsc_mesh::require(xyz.size()%3==0 && rho.size()==xyz.size()/3*4,"expected positions[N,3] and density[N,4]");
    for(double x:xyz)lsc_mesh::require(lsc_electrostatics_finite(x),"positions must be finite");
    for(double x:rho)lsc_mesh::require(lsc_electrostatics_finite(x),"density must be finite");
    auto context=model.electrostatics_context(cell,pbc,mode);
    lsc_mesh::geometry(context,spacing,order); // Validate before allocations.
    std::lock_guard<std::mutex> lock(workspace->mutex);workspace->begin(workspace->diagnostic_stream);auto stream=workspace->diagnostic_stream;
    auto dx=workspace->buffer<double>("diagnostic_positions",xyz.size()),density=workspace->buffer<double>("diagnostic_density",rho.size());
    auto energy=workspace->buffer<double>("diagnostic_energy",1),dr=workspace->buffer<double>("diagnostic_dr",rho.size());
    if(!xyz.empty()){check(cudaMemcpyAsync(dx,xyz.data(),xyz.size_bytes(),cudaMemcpyHostToDevice,stream),"upload diagnostic positions");
        check(cudaMemcpyAsync(density,rho.data(),rho.size_bytes(),cudaMemcpyHostToDevice,stream),"upload diagnostic density");}
    evaluate_impl({xyz.size()/3,dx,density,context,stream},energy,dr);
    LSCMeshEnergyResult result;result.grid_shape=grid_shape;result.density_gradient.resize(rho.size());double components[2];
    check(cudaMemcpyAsync(&result.energy,energy,sizeof(double),cudaMemcpyDeviceToHost,stream),"read diagnostic energy");
    check(cudaMemcpyAsync(components,workspace->buffer<double>("components",2),sizeof(components),cudaMemcpyDeviceToHost,stream),"read diagnostic components");
    if(!rho.empty())check(cudaMemcpyAsync(result.density_gradient.data(),dr,rho.size_bytes(),cudaMemcpyDeviceToHost,stream),"read diagnostic potentials");
    workspace->wait();result.reciprocal_energy=components[0];result.self_correction=components[1];
    lsc_mesh::require(lsc_electrostatics_finite(result.energy),"nonfinite mesh energy");
    for(double x:result.density_gradient)lsc_mesh::require(lsc_electrostatics_finite(x),"nonfinite mesh potential");
    return result;
}

LSCElectrostaticsResult LSCCudaMesh::evaluate(const MACELSC& model,std::span<const double> xyz,
    std::span<const double> rho,const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode){
    lsc_mesh::require(xyz.size()%3==0 && rho.size()==xyz.size()/3*4,"expected positions[N,3] and density[N,4]");
    for(double x:xyz)lsc_mesh::require(lsc_electrostatics_finite(x),"positions must be finite");
    for(double x:rho)lsc_mesh::require(lsc_electrostatics_finite(x),"density must be finite");
    auto context=model.electrostatics_context(cell,pbc,mode);lsc_mesh::geometry(context,spacing,order);
    std::lock_guard<std::mutex> lock(workspace->mutex);workspace->begin(workspace->diagnostic_stream);auto stream=workspace->diagnostic_stream;
    auto positions=workspace->buffer<double>("diagnostic_positions",xyz.size()),density=workspace->buffer<double>("diagnostic_density",rho.size());
    auto energy=workspace->buffer<double>("diagnostic_energy",1),dr=workspace->buffer<double>("diagnostic_dr",rho.size());
    auto dx=workspace->buffer<double>("diagnostic_dx",xyz.size()),ds=workspace->buffer<double>("diagnostic_ds",9);
    if(!xyz.empty()){check(cudaMemcpyAsync(positions,xyz.data(),xyz.size_bytes(),cudaMemcpyHostToDevice,stream),"upload diagnostic positions");
        check(cudaMemcpyAsync(density,rho.data(),rho.size_bytes(),cudaMemcpyHostToDevice,stream),"upload diagnostic density");}
    evaluate_impl({xyz.size()/3,positions,density,context,stream},energy,dr,dx,ds);
    LSCElectrostaticsResult result;result.density_gradient.resize(rho.size());result.position_gradient.resize(xyz.size());
    check(cudaMemcpyAsync(&result.energy,energy,sizeof(double),cudaMemcpyDeviceToHost,stream),"read diagnostic energy");
    if(!xyz.empty()){check(cudaMemcpyAsync(result.position_gradient.data(),dx,xyz.size_bytes(),cudaMemcpyDeviceToHost,stream),"read diagnostic position gradient");
        check(cudaMemcpyAsync(result.density_gradient.data(),dr,rho.size_bytes(),cudaMemcpyDeviceToHost,stream),"read diagnostic density gradient");}
    check(cudaMemcpyAsync(result.strain_derivative.data(),ds,9*sizeof(double),cudaMemcpyDeviceToHost,stream),"read diagnostic strain derivative");
    workspace->wait();validate_lsc_electrostatics_result(result,xyz.size()/3);return result;
}
void MACELSCCuda::set_mesh_electrostatics(double spacing,int order){
    check(cudaSetDevice(device_id),"select model device for mesh");
    auto mesh=std::make_shared<LSCCudaMesh>(spacing,order);
    set_electrostatics_hook([mesh](const LSCCudaElectrostaticsInput& input,const LSCCudaElectrostaticsOutput& output){mesh->evaluate(input,output);});
}
