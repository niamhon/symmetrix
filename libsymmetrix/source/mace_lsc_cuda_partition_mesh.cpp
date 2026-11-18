/* LSC distributed mesh device work, adapted from Symmetrix/Polar.
   LAMMPS, https://www.lammps.org/, Sandia National Laboratories.
   Copyright (2003) Sandia Corporation. GNU GPL; see pair_symmetrix/LICENSE. */
#include "mace_lsc_cuda_partition_mesh.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <map>
#include <string>
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("LSC partition mesh: ")+cudaGetErrorString(e));}
void check(cufftResult e){if(e!=CUFFT_SUCCESS)throw std::runtime_error("LSC partition mesh: local cuFFT failed");}
__global__ void weights(int N,lsc_partition::Grid g,const double* xyz,int* base,double* theta,int* error){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N && !lsc_partition::weights(g,xyz+3*size_t(i),base+3*i,theta+size_t(i)*9*g.order))atomicOr(error,1);}
__global__ void spread(int work,lsc_partition::Grid g,const double* rho,const int* base,const double* theta,double* brick){
    int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=work)return;int P=g.order,i=k/(P*P),y=k%P,z=k/P%P;double c,mu[3];lsc_partition::sources(g,rho+4*size_t(i),c,mu);
    for(int x=0;x<P;++x)atomicAdd(brick+lsc_partition::at(g,base+3*i,x,y,z),lsc_partition::assignment(P,theta+size_t(i)*9*P,x,y,z,c,mu));}
__global__ void convolve(lsc_partition::Grid g,double* v,const double* mod,double* energy,double* strain){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=g.n_in)return;double e,w[9];double f=lsc_partition::kernel(g,i,mod,v[2*i],v[2*i+1],e,w);v[2*i]*=f;v[2*i+1]*=f;energy[i]=e;for(int d=0;d<9;++d)strain[size_t(d)*g.n_in+i]=w[d];}
__global__ void gather(int N,lsc_partition::Grid g,const double* phi,const int* base,const double* theta,const double* rho,double* dr,double* dx,double* energy,double* strain){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;double local[9];energy[i]=lsc_partition::gather(g,phi,base+3*i,theta+size_t(i)*9*g.order,rho+4*size_t(i),dr+4*size_t(i),dx+3*size_t(i),local);
    for(int d=0;d<9;++d)strain[size_t(d)*N+i]=local[d];}

}
struct LSCCudaMeshPartition::Workspace {
    struct Buffer{void* p=nullptr;size_t capacity=0;};std::map<std::string,Buffer> buffers;
    std::map<std::pair<int,int>,cufftHandle> plans;cudaStream_t stream=nullptr;int N=0,id=-1;
    Workspace(){check(cudaGetDevice(&id));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Workspace(){int previous=-1;cudaGetDevice(&previous);cudaSetDevice(id);cudaStreamSynchronize(stream);for(auto [k,p]:plans)cufftDestroy(p);for(auto [k,b]:buffers)cudaFree(b.p);cudaStreamDestroy(stream);if(previous>=0)cudaSetDevice(previous);}
    void active(){int now;check(cudaGetDevice(&now));if(now!=id)throw std::runtime_error("LSC partition mesh: activate its CUDA device");}
    template<class T>T* get(std::string name,size_t n){auto& b=buffers[name];size_t bytes=n*sizeof(T);if(bytes>b.capacity){check(cudaStreamSynchronize(stream));void* p=nullptr;check(cudaMalloc(&p,bytes));if(b.p)check(cudaFree(b.p));b={p,bytes};}return static_cast<T*>(b.p);}
    template<class T>T* upload(std::string name,std::span<const T> x){auto p=get<T>(name,x.size());if(!x.empty())check(cudaMemcpyAsync(p,x.data(),x.size_bytes(),cudaMemcpyHostToDevice,stream));return p;}
    void reduce(int n,const double* energy,const double* strain){
        auto out=get<double>("reductions",10);
        if(!n){check(cudaMemsetAsync(out,0,10*sizeof(double),stream));return;}
        size_t bytes=0;check(cub::DeviceReduce::Sum(nullptr,bytes,energy,out,n,stream));auto temp=get<char>("reduce_temp",bytes);
        check(cub::DeviceReduce::Sum(temp,bytes,energy,out,n,stream));
        for(int d=0;d<9;++d)check(cub::DeviceReduce::Sum(temp,bytes,strain+size_t(d)*n,out+1+d,n,stream));
    }
    template<class T>std::vector<T> read(std::string name,size_t n){std::vector<T> out(n);if(n)check(cudaMemcpyAsync(out.data(),get<T>(name,n),n*sizeof(T),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));return out;}
};
LSCCudaMeshPartition::LSCCudaMeshPartition():workspace(std::make_unique<Workspace>()){}
LSCCudaMeshPartition::~LSCCudaMeshPartition()=default;
std::vector<double> LSCCudaMeshPartition::spread(lsc_partition::Grid g,std::span<const double> xyz,std::span<const double> rho){
    auto& w=*workspace;w.active();w.N=xyz.size()/3;int N=w.N;auto stream=w.stream;
    auto p=w.upload("positions",xyz),r=w.upload("rho",rho);auto base=w.get<int>("base",size_t(N)*3);auto theta=w.get<double>("theta",size_t(N)*9*g.order);auto err=w.get<int>("error",1);check(cudaMemsetAsync(err,0,sizeof(int),stream));
    if(N)weights<<<(N+127)/128,128,0,stream>>>(N,g,p,base,theta,err);if(w.read<int>("error",1)[0])throw std::invalid_argument("LSC partition mesh: atoms outside grid stencil");
    auto brick=w.get<double>("brick",g.n_out);check(cudaMemsetAsync(brick,0,size_t(g.n_out)*sizeof(double),stream));int work=N*g.order*g.order;
    if(work)::spread<<<(work+127)/128,128,0,stream>>>(work,g,r,base,theta,brick);check(cudaGetLastError());return w.read<double>("brick",g.n_out);
}
lsc_partition::KernelResult LSCCudaMeshPartition::convolve(lsc_partition::Grid g,std::span<const double> values){
    auto& w=*workspace;w.active();auto mod=lsc_partition::moduli(g);auto v=w.upload("spectral",values),m=w.upload("moduli",std::span<const double>(mod));auto e=w.get<double>("energy",g.n_in),s=w.get<double>("strain",size_t(g.n_in)*9);
    if(g.n_in)::convolve<<<(g.n_in+127)/128,128,0,w.stream>>>(g,v,m,e,s);check(cudaGetLastError());lsc_partition::KernelResult out;out.values=w.read<double>("spectral",values.size());w.reduce(g.n_in,e,s);auto sums=w.read<double>("reductions",10);out.energy=sums[0];std::copy_n(sums.data()+1,9,out.strain.data());return out;
}
LSCElectrostaticsResult LSCCudaMeshPartition::gather(lsc_partition::Grid g,std::span<const double> potential){
    auto& w=*workspace;w.active();int N=w.N;auto phi=w.upload("potential",potential);auto dr=w.get<double>("dr",size_t(N)*4),dx=w.get<double>("dx",size_t(N)*3),e=w.get<double>("energy",N),s=w.get<double>("strain",size_t(N)*9);
    if(N)::gather<<<(N+127)/128,128,0,w.stream>>>(N,g,phi,w.get<int>("base",size_t(N)*3),w.get<double>("theta",size_t(N)*9*g.order),w.get<double>("rho",size_t(N)*4),dr,dx,e,s);check(cudaGetLastError());LSCElectrostaticsResult out;out.density_gradient=w.read<double>("dr",size_t(N)*4);out.position_gradient=w.read<double>("dx",size_t(N)*3);w.reduce(N,e,s);auto sums=w.read<double>("reductions",10);out.energy=sums[0];std::copy_n(sums.data()+1,9,out.strain_derivative.data());return out;
}
void LSCCudaMeshPartition::fft_local(std::span<double> values,int length,int total,int direction){
    if(!total)return;auto& w=*workspace;w.active();if(length<=0 || total%length || values.size()<size_t(total)*2)throw std::invalid_argument("LSC partition mesh: invalid local FFT extent");auto key=std::make_pair(length,total);
    if(!w.plans.count(key)){cufftHandle p;int dims[1]={length};check(cufftPlanMany(&p,1,dims,nullptr,1,length,nullptr,1,length,CUFFT_Z2Z,total/length));w.plans[key]=p;}
    auto ptr=w.upload("fft",std::span<const double>(values.first(size_t(total)*2)));auto p=w.plans.at(key);check(cufftSetStream(p,w.stream));check(cufftExecZ2Z(p,reinterpret_cast<cufftDoubleComplex*>(ptr),reinterpret_cast<cufftDoubleComplex*>(ptr),direction==1?CUFFT_FORWARD:CUFFT_INVERSE));
    auto host=w.read<double>("fft",size_t(total)*2);std::copy(host.begin(),host.end(),values.begin());++fft_calls;
}
