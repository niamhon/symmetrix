#pragma once
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace lsc_cuda {
struct ElectrostaticParameters {
    double field_constant,sigma2,sigma,offset,epsilon,a[2],self_recip[4],self_real[4],displacements[9];
    bool include_self;
};
struct ReciprocalGeometry {std::vector<double> vectors;double volume;};
inline bool energy_finite(double x){return (std::bit_cast<uint64_t>(x)&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);}
inline double cell_volume(const std::array<double,9>& c){
    return std::abs(c[0]*(c[4]*c[8]-c[5]*c[7])-c[1]*(c[3]*c[8]-c[5]*c[6])+c[2]*(c[3]*c[7]-c[4]*c[6]));
}
// Same half-space enumeration and endpoints as native LSC/graph_longrange;
// Polar likewise removes the explicit factor 1/2 by using half reciprocal space.
inline ReciprocalGeometry reciprocal_geometry(const std::array<double,9>& c,double cutoff)
{
    auto fail=[](){throw std::invalid_argument("MACELSCCuda: invalid or excessive reciprocal cell");};
    for(double x:c)if(not energy_finite(x))fail();
    auto cross=[&](int i,int j){return std::array<double,3>{c[3*i+1]*c[3*j+2]-c[3*i+2]*c[3*j+1],c[3*i+2]*c[3*j]-c[3*i]*c[3*j+2],c[3*i]*c[3*j+1]-c[3*i+1]*c[3*j]};};
    auto a=cross(1,2);const double det=c[0]*a[0]+c[1]*a[1]+c[2]*a[2];
    if(not energy_finite(det) or det==0)fail();
    std::array<std::array<double,3>,3> reciprocal;std::array<int,3> bounds;
    long long candidates=1;
    for(int i=0;i<3;++i){
        auto b=cross((i+1)%3,(i+2)%3);for(auto& x:b)x*=2*std::numbers::pi/det;reciprocal[i]=b;
        const double bound=std::ceil(cutoff*std::hypot(c[3*i],c[3*i+1],c[3*i+2])/(2*std::numbers::pi));
        if(not energy_finite(bound) or bound<1 or bound>1000000)fail();
        bounds[i]=int(bound);candidates*=2LL*bounds[i];if(candidates>std::numeric_limits<int>::max())fail();
    }
    ReciprocalGeometry result{{},std::abs(det)};
    for(int a=0;a<bounds[0];++a)for(int b=a==0?0:-bounds[1];b<bounds[1];++b)
        for(int z=a==0 and b==0?1:-bounds[2];z<bounds[2];++z){
            double k[3],norm=0;for(int d=0;d<3;++d){k[d]=a*reciprocal[0][d]+b*reciprocal[1][d]+z*reciprocal[2][d];norm+=k[d]*k[d];}
            if(norm<=cutoff*cutoff)for(auto x:k)result.vectors.push_back(x);
        }
    return result;
}
__device__ inline void displaced_charges(const double* rho,double offset,double* q)
{
    q[1]=rho[3]/offset;q[2]=rho[1]/offset;q[3]=rho[2]/offset;q[0]=rho[0]-q[1]-q[2]-q[3];
}
__global__ void electrostatic_real(int N,ElectrostaticParameters p,const double* positions,const double* density,double* terms)
{
    using Reduce=cub::BlockReduce<double,128>;
    __shared__ typename Reduce::TempStorage storage;
    const int i=blockIdx.x;double qi[4],sum=0;displaced_charges(density+size_t(i)*4,p.offset,qi);
    for(int j=i+1+threadIdx.x;j<N;j+=blockDim.x){
        double qj[4];displaced_charges(density+size_t(j)*4,p.offset,qj);
        for(int a=0;a<4;++a)for(int b=0;b<4;++b){
            double delta[3],r2=0;
            for(int d=0;d<3;++d){delta[d]=positions[size_t(i)*3+d]-positions[size_t(j)*3+d]
                +(a==0?0:p.displacements[3*(a-1)+d])-(b==0?0:p.displacements[3*(b-1)+d]);r2+=delta[d]*delta[d];}
            const double r=sqrt(r2);sum+=qi[a]*qj[b]*erf(r*0.5/p.sigma)/(r+p.epsilon);
        }
    }
    sum=Reduce(storage).Sum(sum);
    if(threadIdx.x==0){
        sum*=p.field_constant/(4*3.14159265358979323846);
        if(p.include_self)for(int m=0;m<4;++m)sum+=0.5*density[size_t(i)*4+m]*density[size_t(i)*4+m]*p.self_real[m];
        terms[i]=sum;
    }
}
struct ComplexSum {
    __device__ double2 operator()(const double2& a,const double2& b)const{return make_double2(a.x+b.x,a.y+b.y);}
};
__global__ void electrostatic_reciprocal(int N,ElectrostaticParameters p,double volume,const double* vectors,
    const double* positions,const double* density,double2* structure,double* terms)
{
    using Reduce=cub::BlockReduce<double2,128>;
    __shared__ typename Reduce::TempStorage storage;
    const int t=blockIdx.x;const double* k=vectors+size_t(t)*3;
    const double norm=k[0]*k[0]+k[1]*k[1]+k[2]*k[2],g=exp(-0.5*p.sigma2*norm);
    const double monopole=p.a[0]*g/sqrt(4*3.14159265358979323846);
    const double dipole=-p.a[1]*g*sqrt(3/(4*3.14159265358979323846));
    double2 s=make_double2(0,0);
    for(int i=threadIdx.x;i<N;i+=blockDim.x){
        const double phase=positions[size_t(i)*3]*k[0]+positions[size_t(i)*3+1]*k[1]+positions[size_t(i)*3+2]*k[2];
        double sine,cosine;sincos(phase,&sine,&cosine);
        const double A=monopole*density[size_t(i)*4],B=dipole*(density[size_t(i)*4+3]*k[0]+density[size_t(i)*4+1]*k[1]+density[size_t(i)*4+2]*k[2]);
        s.x+=A*cosine+B*sine;s.y+=B*cosine-A*sine;
    }
    s=Reduce(storage).Reduce(s,ComplexSum{});
    if(threadIdx.x==0){structure[t]=s;terms[t]=p.field_constant/volume*(s.x*s.x+s.y*s.y)/norm;}
}
__global__ void electrostatic_self(int N,int K,ElectrostaticParameters p,const double* density,double* terms)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    double value=0;if(not p.include_self)for(int m=0;m<4;++m)value-=0.5*density[size_t(i)*4+m]*density[size_t(i)*4+m]*p.self_recip[m];
    terms[size_t(K)+i]=value;
}
__global__ void electrostatic_total(const double* local,const double* electrostatic,const double* dipole,
                                     double x,double y,double z,double* components)
{
    if(threadIdx.x==0){components[0]=*local;components[1]=*electrostatic;components[2]=dipole[0]*x+dipole[1]*y+dipole[2]*z;components[3]=components[0]+components[1]+components[2];}
}

// Analytic adjoints of the same energy formulas. Density and positions are
// independent here; the neural reverse pass propagates density dependence.
__global__ void electrostatic_real_reverse(int N,ElectrostaticParameters p,const double* xyz,
    const double* rho,double* density_adj,double* position_adj)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    double qi[4],qa[4]={},g[3]={};displaced_charges(rho+size_t(i)*4,p.offset,qi);
    const double pref=p.field_constant/(4*3.14159265358979323846);
    for(int j=0;j<N;++j){if(i==j)continue;double qj[4];displaced_charges(rho+size_t(j)*4,p.offset,qj);
        for(int a=0;a<4;++a)for(int b=0;b<4;++b){
            double delta[3],r2=0;for(int d=0;d<3;++d){delta[d]=xyz[size_t(i)*3+d]-xyz[size_t(j)*3+d]
                +(a==0?0:p.displacements[3*(a-1)+d])-(b==0?0:p.displacements[3*(b-1)+d]);r2+=delta[d]*delta[d];}
            const double r=sqrt(r2),smooth=erf(r*0.5/p.sigma),den=r+p.epsilon;
            qa[a]+=pref*smooth/den*qj[b];
            if(r>0){const double ds=exp(-r2/(4*p.sigma*p.sigma))/(p.sigma*sqrt(3.14159265358979323846));
                const double radial=pref*qi[a]*qj[b]*(ds/den-smooth/(den*den))/r;
                for(int d=0;d<3;++d)g[d]+=radial*delta[d];}
        }
    }
    double dr[4]={qa[0],(qa[2]-qa[0])/p.offset,(qa[3]-qa[0])/p.offset,(qa[1]-qa[0])/p.offset};
    for(int m=0;m<4;++m)density_adj[size_t(i)*4+m]=dr[m]+(p.include_self?rho[size_t(i)*4+m]*p.self_real[m]:0.);
    for(int d=0;d<3;++d)position_adj[size_t(i)*3+d]=g[d];
}
__global__ void electrostatic_reciprocal_reverse(int N,int K,ElectrostaticParameters p,double volume,
    const double* vectors,const double2* structure,const double* xyz,const double* rho,double* density_adj,double* position_adj)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
    double dr[4]={},g[3]={};constexpr int cartesian[3]={3,1,2};
    for(int t=0;t<K;++t){const auto k=vectors+size_t(t)*3;const auto s=structure[t];
        const double norm=k[0]*k[0]+k[1]*k[1]+k[2]*k[2],gauss=exp(-0.5*p.sigma2*norm);
        const double monopole=p.a[0]*gauss/sqrt(4*3.14159265358979323846),dipole=-p.a[1]*gauss*sqrt(3/(4*3.14159265358979323846));
        const double phase=xyz[size_t(i)*3]*k[0]+xyz[size_t(i)*3+1]*k[1]+xyz[size_t(i)*3+2]*k[2];
        double sine,cosine;sincos(phase,&sine,&cosine);
        const double weight=2*p.field_constant/volume/norm,dA=weight*(s.x*cosine-s.y*sine),dB=weight*(s.x*sine+s.y*cosine);
        dr[0]+=dA*monopole;for(int d=0;d<3;++d)dr[cartesian[d]]+=dB*dipole*k[d];
        const double A=monopole*rho[size_t(i)*4],B=dipole*(rho[size_t(i)*4+3]*k[0]+rho[size_t(i)*4+1]*k[1]+rho[size_t(i)*4+2]*k[2]);
        const double re=A*cosine+B*sine,im=B*cosine-A*sine,dp=weight*(s.x*im-s.y*re);
        for(int d=0;d<3;++d)g[d]+=dp*k[d];
    }
    for(int m=0;m<4;++m)density_adj[size_t(i)*4+m]=dr[m]-(p.include_self?0.:rho[size_t(i)*4+m]*p.self_recip[m]);
    for(int d=0;d<3;++d)position_adj[size_t(i)*3+d]=g[d];
}
__global__ void electrostatic_reciprocal_strain(int N,ElectrostaticParameters p,double volume,
    const double* vectors,const double2* structure,const double* xyz,const double* rho,
    size_t stride,size_t offset,double* terms)
{
    using Reduce=cub::BlockReduce<double2,128>;
    __shared__ typename Reduce::TempStorage storage;__shared__ double2 dipole_sum[3];
    const int t=blockIdx.x;const auto k=vectors+size_t(t)*3;const auto S=structure[t];
    const double norm=k[0]*k[0]+k[1]*k[1]+k[2]*k[2],gauss=exp(-0.5*p.sigma2*norm);
    const double dipole=-p.a[1]*gauss*sqrt(3/(4*3.14159265358979323846));
    constexpr int cartesian[3]={3,1,2};double2 sum[3]={};
    for(int i=threadIdx.x;i<N;i+=blockDim.x){
        const double phase=xyz[size_t(i)*3]*k[0]+xyz[size_t(i)*3+1]*k[1]+xyz[size_t(i)*3+2]*k[2];
        double sine,cosine;sincos(phase,&sine,&cosine);
        for(int d=0;d<3;++d){const double value=dipole*rho[size_t(i)*4+cartesian[d]];sum[d].x+=value*sine;sum[d].y+=value*cosine;}
    }
    for(int d=0;d<3;++d){const double2 value=Reduce(storage).Reduce(sum[d],ComplexSum{});if(threadIdx.x==0)dipole_sum[d]=value;__syncthreads();}
    if(threadIdx.x==0){
        const double pref=p.field_constant/volume,e=pref*(S.x*S.x+S.y*S.y)/norm;
        for(int a=0;a<3;++a){const double dre=dipole_sum[a].x-p.sigma2*k[a]*S.x,dim=dipole_sum[a].y-p.sigma2*k[a]*S.y;
            const double dk=2*pref/norm*(S.x*dre+S.y*dim)-2*e*k[a]/norm;
            for(int b=0;b<3;++b)terms[size_t(3*a+b)*stride+offset+t]=-dk*k[b]-(a==b?e:0.);}
    }
}
} // namespace lsc_cuda
