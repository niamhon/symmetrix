/*
   LSC one-process CPU mesh energy and analytic derivatives. Grid assignment/deconvolution adapted from
   pair_symmetrix_mace_polar.cpp, wcwitt/symmetrix commit b907f04026f2dda4c43d66511a7c37db8a82a7d4.

   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. Distributed under the GNU General Public
   License; see pair_symmetrix/LICENSE in this repository.
*/
#include "mace_lsc_mesh.hpp"
#include "mace_lsc.hpp"
#include "mace_lsc_mesh_common.hpp"
using namespace lsc_mesh;
#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numbers>
#include <numeric>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
std::mutex& planner_mutex() { static std::mutex mutex; return mutex; }
} // namespace

struct LSCCPUMesh::Workspace {
    std::mutex evaluation_mutex;
    std::array<int,3> shape{};
    size_t points=0;
    fftw_complex* data=nullptr;
    fftw_plan forward=nullptr,backward=nullptr;
    std::vector<double> brick,theta;
    std::vector<std::array<int,3>> base;
    ~Workspace() { release(); }
    void release()
    {
        std::lock_guard<std::mutex> lock(planner_mutex());
        if(forward) fftw_destroy_plan(forward);
        if(backward) fftw_destroy_plan(backward);
        if(data) fftw_free(data);
        forward=backward=nullptr;data=nullptr;points=0;shape={};
    }
    void configure(const std::array<int,3>& dimensions)
    {
        if(shape==dimensions && data) return;
        release();
        std::lock_guard<std::mutex> lock(planner_mutex());
        points=size_t(dimensions[0])*dimensions[1]*dimensions[2];
        data=fftw_alloc_complex(points);
        if(!data) throw std::bad_alloc();
        // As in Polar, use one FFT thread; OpenMP handles spreading/gathering.
        forward=fftw_plan_dft_3d(dimensions[2],dimensions[1],dimensions[0],data,data,FFTW_FORWARD,FFTW_ESTIMATE);
        backward=fftw_plan_dft_3d(dimensions[2],dimensions[1],dimensions[0],data,data,FFTW_BACKWARD,FFTW_ESTIMATE);
        if(!forward || !backward) throw std::runtime_error("LSCCPUMesh: FFTW plan creation failed");
        shape=dimensions;
    }
};

bool lsc_cpu_mesh_compiled() { return true; }
LSCCPUMesh::LSCCPUMesh(double mesh_spacing,int spline_order)
    : spacing(mesh_spacing),order(spline_order),workspace(std::make_unique<Workspace>())
{
    require(lsc_electrostatics_finite(spacing) && spacing>0,"spacing must be finite and positive");
    require(order>=4 && order<=MAXORDER,"order must be between 4 and 10");
}
LSCCPUMesh::~LSCCPUMesh()=default;

LSCMeshEnergyResult LSCCPUMesh::evaluate_energy(const LSCElectrostaticsInput& input)
{
    return evaluate_impl(input,nullptr,nullptr);
}

LSCElectrostaticsResult LSCCPUMesh::evaluate(const LSCElectrostaticsInput& input)
{
    LSCElectrostaticsResult result;
    auto energy=evaluate_impl(input,&result.position_gradient,&result.strain_derivative);
    result.energy=energy.energy;result.density_gradient=std::move(energy.density_gradient);
    validate_lsc_electrostatics_result(result,input.positions.size()/3);
    return result;
}

LSCMeshEnergyResult LSCCPUMesh::evaluate_impl(const LSCElectrostaticsInput& input,
    std::vector<double>* position_gradient,std::array<double,9>* strain_derivative)
{
    std::lock_guard<std::mutex> evaluation_lock(workspace->evaluation_mutex);
    require(input.positions.size()%3==0 && input.density.size()==input.positions.size()/3*4,"expected positions[N,3] and density[N,4]");
    const size_t n=input.positions.size()/3;
    require(n<=size_t(std::numeric_limits<int>::max()),"too many atoms");
    for(double x:input.positions) require(lsc_electrostatics_finite(x),"positions must be finite");
    for(double x:input.density) require(lsc_electrostatics_finite(x),"density must be finite");
    const auto g=geometry(input.context,spacing,order);const auto& p=input.context.parameters;
    workspace->configure(g.shape);const size_t points=workspace->points;
    const int nx=g.shape[0],ny=g.shape[1],nz=g.shape[2],P=order;
    auto index=[&](int x,int y,int z) {return (size_t(wrap(z,nz))*ny+wrap(y,ny))*nx+wrap(x,nx);};
    int nt=1;
#ifdef _OPENMP
    nt=std::min(omp_get_max_threads(),int(std::max(n,size_t(1))));
#endif
    require(size_t(nt)<=std::numeric_limits<size_t>::max()/points,"spread workspace is too large");
    workspace->brick.assign(points*nt,0);workspace->theta.resize(n*9*P);workspace->base.resize(n);
    auto& brick=workspace->brick;auto& theta=workspace->theta;auto& base=workspace->base;
    #pragma omp parallel for
    for(int i=0;i<int(n);++i) {
        for(int d=0;d<3;++d) {
            double fraction=0;
            for(int a=0;a<3;++a) fraction+=input.positions[3*size_t(i)+a]*g.reciprocal[d][a]/(2*PI);
            fraction-=std::floor(fraction);
            const double u=fraction*g.shape[d];const int i0=int(std::floor(u));
            base[i][d]=i0-(P-1)+P/2;bspline(P,u-i0,theta.data()+(size_t(i)*9+3*d)*P);
        }
    }
    #pragma omp parallel for schedule(static,1)
    for(int t=0;t<nt;++t)
        for(size_t i=size_t(t)*n/nt;i<size_t(t+1)*n/nt;++i) {
            const auto* rho=input.density.data()+4*i;
            const double c=p.a[0]*Y00*rho[0],m[3]={p.a[1]*Y1*rho[3],p.a[1]*Y1*rho[1],p.a[1]*Y1*rho[2]};
            double mu[3]={};for(int d=0;d<3;++d) for(int a=0;a<3;++a) mu[d]+=g.J[d][a]*m[a];
            const auto* tx=theta.data()+i*9*P;const auto* ty=tx+3*P;const auto* tz=ty+3*P;
            for(int z=0;z<P;++z) for(int y=0;y<P;++y) for(int x=0;x<P;++x)
                brick[size_t(t)*points+index(base[i][0]+x,base[i][1]+y,base[i][2]+z)]
                    +=c*tx[x]*ty[y]*tz[z]+mu[0]*tx[P+x]*ty[y]*tz[z]
                      +mu[1]*tx[x]*ty[P+y]*tz[z]+mu[2]*tx[x]*ty[y]*tz[P+z];
        }
    auto* data=workspace->data;
    #pragma omp parallel for
    for(long k=0;k<long(points);++k) {
        double value=brick[k];for(int t=1;t<nt;++t) value+=brick[size_t(t)*points+k];
        data[k][0]=value;data[k][1]=0;
    }
    fftw_execute(workspace->forward);
    std::array<std::vector<double>,3> bsmod;
    double t0[3*MAXORDER];bspline(P,0,t0);
    for(int d=0;d<3;++d) {
        bsmod[d].resize(g.shape[d]);
        for(int k=0;k<g.shape[d];++k) {
            double re=0,im=0;
            for(int j=0;j<=P-2;++j) {
                re+=t0[P-2-j]*std::cos(2*PI*k*j/g.shape[d]);
                im+=t0[P-2-j]*std::sin(2*PI*k*j/g.shape[d]);
            }
            bsmod[d][k]=re*re+im*im;
        }
    }
    std::vector<double> energies(points,0);
    const double prefactor=p.field_constant/g.volume;
    // Fixed chunks give bounded storage and deterministic thread-local strain sums.
    std::vector<std::array<double,9>> kernel_strain(nt);
    #pragma omp parallel for schedule(static,1)
    for(int t=0;t<nt;++t) for(size_t slot=size_t(t)*points/nt;slot<size_t(t+1)*points/nt;++slot) {
        const int ix=slot%nx,iy=(slot/nx)%ny,iz=slot/(long(nx)*ny);
        std::array<int,3> mode{ix,iy,iz};
        for(int d=0;d<3;++d) if(2*mode[d]>g.shape[d]) mode[d]-=g.shape[d];
        double k[3]={};for(int a=0;a<3;++a) for(int d=0;d<3;++d) k[a]+=mode[d]*g.reciprocal[d][a];
        const double norm=k[0]*k[0]+k[1]*k[1]+k[2]*k[2];
        const bool active=direct_mode(mode,g.bounds) && norm>0 && norm<=p.kspace_cutoff*p.kspace_cutoff;
        const double factor=active ? prefactor*std::exp(-p.sigma2*norm)/norm/(bsmod[0][ix]*bsmod[1][iy]*bsmod[2][iz]) : 0;
        energies[slot]=0.5*factor*(data[slot][0]*data[slot][0]+data[slot][1]*data[slot][1]);
        if(strain_derivative && active)
            for(int a=0;a<3;++a) for(int b=0;b<3;++b)
                kernel_strain[t][3*a+b]+=energies[slot]*(2*(p.sigma2+1/norm)*k[a]*k[b]-(a==b ? 1.0 : 0.0));
        data[slot][0]*=factor;data[slot][1]*=factor;
    }
    LSCMeshEnergyResult result;result.grid_shape=g.shape;
    result.reciprocal_energy=std::accumulate(energies.begin(),energies.end(),0.0);
    // FFTW's backward transform is unnormalized: this is exactly the full
    // grid gradient of the spectral energy above. No 1/Ngrid belongs here.
    fftw_execute(workspace->backward);
    result.density_gradient.assign(4*n,0);
    if(position_gradient) position_gradient->assign(3*n,0);
    if(strain_derivative) {
        strain_derivative->fill(0);
        for(const auto& part:kernel_strain) for(int ab=0;ab<9;++ab) (*strain_derivative)[ab]+=part[ab];
    }
    #pragma omp parallel for
    for(int i=0;i<int(n);++i) {
        const auto* tx=theta.data()+size_t(i)*9*P;const auto* ty=tx+3*P;const auto* tz=ty+3*P;
        double q=0,grad[3]={},Hu[3][3]={};
        for(int z=0;z<P;++z) for(int y=0;y<P;++y) for(int x=0;x<P;++x) {
            const double value=data[index(base[i][0]+x,base[i][1]+y,base[i][2]+z)][0];
            q+=tx[x]*ty[y]*tz[z]*value;
            grad[0]+=tx[P+x]*ty[y]*tz[z]*value;
            grad[1]+=tx[x]*ty[P+y]*tz[z]*value;
            grad[2]+=tx[x]*ty[y]*tz[P+z]*value;
            if(position_gradient) {
                Hu[0][0]+=tx[2*P+x]*ty[y]*tz[z]*value;
                Hu[1][1]+=tx[x]*ty[2*P+y]*tz[z]*value;
                Hu[2][2]+=tx[x]*ty[y]*tz[2*P+z]*value;
                Hu[0][1]+=tx[P+x]*ty[P+y]*tz[z]*value;
                Hu[0][2]+=tx[P+x]*ty[y]*tz[P+z]*value;
                Hu[1][2]+=tx[x]*ty[P+y]*tz[P+z]*value;
            }
        }
        result.density_gradient[4*size_t(i)]=p.a[0]*Y00*q;
        constexpr int cartesian[3]={3,1,2};
        for(int a=0;a<3;++a) for(int d=0;d<3;++d)
            result.density_gradient[4*size_t(i)+cartesian[a]]+=p.a[1]*Y1*g.J[d][a]*grad[d];
        if(position_gradient) {
            Hu[1][0]=Hu[0][1];Hu[2][0]=Hu[0][2];Hu[2][1]=Hu[1][2];
            const auto* rho=input.density.data()+4*size_t(i);
            const double c=p.a[0]*Y00*rho[0],m[3]={p.a[1]*Y1*rho[3],p.a[1]*Y1*rho[1],p.a[1]*Y1*rho[2]};
            // Gather the Cartesian potential gradient and Hessian as in Polar.
            for(int a=0;a<3;++a) {
                double value=0;
                for(int d=0;d<3;++d) {
                    value+=c*g.J[d][a]*grad[d];
                    for(int b=0;b<3;++b) for(int e=0;e<3;++e)
                        value+=g.J[d][a]*Hu[d][e]*g.J[e][b]*m[b];
                }
                (*position_gradient)[3*size_t(i)+a]=value;
            }
        }
    }
    if(strain_derivative) {
        // Under r -> r A and cell -> cell A the fractional sites stay fixed,
        // but the Cartesian dipoles are held fixed: d(J mu)/d strain != 0.
        // g_rho already includes a1*Y1, so these factors cancel exactly here.
        constexpr int cartesian[3]={3,1,2};
        for(size_t i=0;i<n;++i) for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            (*strain_derivative)[3*a+b]-=input.density[4*i+cartesian[a]]*result.density_gradient[4*i+cartesian[b]];
    }
    // The fixed-density self correction has no position or cell derivative.
    if(!p.include_self) {
        for(size_t i=0;i<4*n;++i) {
            result.self_correction-=0.5*input.density[i]*input.density[i]*p.reciprocal_self[i%4];
            result.density_gradient[i]-=input.density[i]*p.reciprocal_self[i%4];
        }
    }
    result.energy=result.reciprocal_energy+result.self_correction;
    require(lsc_electrostatics_finite(result.energy),"nonfinite mesh energy");
    for(double x:result.density_gradient) require(lsc_electrostatics_finite(x),"nonfinite density gradient");
    return result;
}

LSCMeshEnergyResult MACELSC::evaluate_mesh_electrostatics(LSCCPUMesh& mesh,
    std::span<const double> positions,std::span<const double> density,
    const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode) const
{
    return mesh.evaluate_energy({positions,density,electrostatics_context(cell,pbc,mode)});
}

LSCElectrostaticsResult MACELSC::evaluate_mesh_electrostatics_full(LSCCPUMesh& mesh,
    std::span<const double> positions,std::span<const double> density,
    const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode) const
{
    return mesh.evaluate({positions,density,electrostatics_context(cell,pbc,mode)});
}

void MACELSC::set_mesh_electrostatics(double spacing,int order)
{
    auto mesh=std::make_shared<LSCCPUMesh>(spacing,order);
    set_electrostatics_hook([mesh](const LSCElectrostaticsInput& input) {return mesh->evaluate(input);});
}
