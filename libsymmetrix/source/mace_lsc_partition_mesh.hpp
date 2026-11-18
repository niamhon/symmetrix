/* LSC partition mesh, adapted from Symmetrix/Polar mesh routines.
   LAMMPS, https://www.lammps.org/, Sandia National Laboratories.
   Copyright (2003) Sandia Corporation. Distributed under GNU GPL;
   see pair_symmetrix/LICENSE. */
#pragma once
#include "mace_lsc_mesh_common.hpp"
#include <span>
#ifdef __CUDACC__
#define LSC_MESH_HD __host__ __device__
#else
#define LSC_MESH_HD
#endif
namespace lsc_partition {
struct Grid {
    int shape[3],bounds[3],lo[3],dims[3],in_lo[3],in_dims[3],order,n_in,n_out;
    double b[9],J[9],cutoff2,sigma2,prefactor,a0,a1,self[4];bool include_self;
};
inline Grid grid(const LSCElectrostaticsContext& c,double spacing,int order,
    const int* in_lo,const int* in_hi,const int* lo,const int* hi){
    auto geo=lsc_mesh::geometry(c,spacing,order);const auto& p=c.parameters;Grid g{};
    g.order=order;g.n_in=g.n_out=1;g.cutoff2=p.kspace_cutoff*p.kspace_cutoff;
    g.sigma2=p.sigma2;g.prefactor=p.field_constant/geo.volume;
    g.a0=p.a[0]*lsc_mesh::Y00;g.a1=p.a[1]*lsc_mesh::Y1;g.include_self=p.include_self;
    for(int d=0;d<3;++d){g.shape[d]=geo.shape[d];g.bounds[d]=geo.bounds[d];g.lo[d]=lo[d];g.dims[d]=hi[d]-lo[d]+1;
        g.in_lo[d]=in_lo[d];g.in_dims[d]=in_hi[d]-in_lo[d]+1;g.n_in*=g.in_dims[d];g.n_out*=g.dims[d];
        for(int a=0;a<3;++a){g.b[3*d+a]=geo.reciprocal[d][a];g.J[3*d+a]=geo.J[d][a];}}
    std::copy(p.reciprocal_self.begin(),p.reciprocal_self.end(),g.self);return g;
}
LSC_MESH_HD inline void cardinal(int P,double f,double* t){
    double M[11][11]={};M[1][0]=1.;for(int n=2;n<=P;++n)for(int j=0;j<n;++j)
        M[n][j]=((f+j)*M[n-1][j]+(n-f-j)*(j>0?M[n-1][j-1]:0.))/(n-1);
    for(int k=0;k<P;++k){int j=P-1-k;t[k]=M[P][j];t[P+k]=M[P-1][j]-(j>0?M[P-1][j-1]:0.);
        t[2*P+k]=M[P-2][j]-2*(j>0?M[P-2][j-1]:0.)+(j>1?M[P-2][j-2]:0.);}
}
LSC_MESH_HD inline bool weights(Grid g,const double* xyz,int* base,double* theta){
    bool valid=true;for(int d=0;d<3;++d){double f=0;for(int a=0;a<3;++a)f+=xyz[a]*g.b[3*d+a]/(2*lsc_mesh::PI);
        double u=f*g.shape[d];int i0=int(floor(u));base[d]=i0-(g.order-1)+g.order/2;
        valid=valid && base[d]>=g.lo[d] && base[d]+g.order<=g.lo[d]+g.dims[d];cardinal(g.order,u-i0,theta+3*d*g.order);}
    return valid;
}
LSC_MESH_HD inline int at(Grid g,const int* base,int x,int y,int z){
    return ((base[2]+z-g.lo[2])*g.dims[1]+base[1]+y-g.lo[1])*g.dims[0]+base[0]+x-g.lo[0];
}
LSC_MESH_HD inline void sources(Grid g,const double* rho,double& c,double* mu){
    c=g.a0*rho[0];double m[3]={g.a1*rho[3],g.a1*rho[1],g.a1*rho[2]};
    for(int d=0;d<3;++d){mu[d]=0;for(int a=0;a<3;++a)mu[d]+=g.J[3*d+a]*m[a];}
}
LSC_MESH_HD inline double assignment(int P,const double* t,int x,int y,int z,double c,const double* mu){
    const auto tx=t,ty=t+3*P,tz=t+6*P;return c*tx[x]*ty[y]*tz[z]+mu[0]*tx[P+x]*ty[y]*tz[z]
        +mu[1]*tx[x]*ty[P+y]*tz[z]+mu[2]*tx[x]*ty[y]*tz[P+z];
}
inline std::vector<double> moduli(Grid g){
    std::vector<double> out;double t[30];cardinal(g.order,0,t);
    for(int d=0;d<3;++d)for(int k=0;k<g.shape[d];++k){double re=0,im=0;
        for(int j=0;j<=g.order-2;++j){double p=2*lsc_mesh::PI*k*j/g.shape[d];re+=t[g.order-2-j]*cos(p);im+=t[g.order-2-j]*sin(p);}out.push_back(re*re+im*im);}return out;
}
LSC_MESH_HD inline double kernel(Grid g,int slot,const double* mod,double re,double im,double& energy,double* strain){
    int ix=g.in_lo[0]+slot%g.in_dims[0],iy=g.in_lo[1]+slot/g.in_dims[0]%g.in_dims[1],iz=g.in_lo[2]+slot/(g.in_dims[0]*g.in_dims[1]);
    int mode[3]={ix,iy,iz};for(int d=0;d<3;++d)if(2*mode[d]>g.shape[d])mode[d]-=g.shape[d];
    int first=0;while(first<3 && mode[first]==0)++first;bool active=first<3;int n[3]={};
    if(active){for(int d=0;d<3;++d)n[d]=mode[first]<0?-mode[d]:mode[d];
        active=n[0]>=0 && n[0]<g.bounds[0] && n[1]>=(n[0]==0?0:-g.bounds[1]) && n[1]<g.bounds[1]
            && n[2]>=(n[0]==0 && n[1]==0?1:-g.bounds[2]) && n[2]<g.bounds[2];}
    double k[3]={};for(int a=0;a<3;++a)for(int d=0;d<3;++d)k[a]+=mode[d]*g.b[3*d+a];
    double norm=k[0]*k[0]+k[1]*k[1]+k[2]*k[2];active=active && norm>0 && norm<=g.cutoff2;
    double f=active?g.prefactor*exp(-g.sigma2*norm)/norm/(mod[ix]*mod[g.shape[0]+iy]*mod[g.shape[0]+g.shape[1]+iz]):0.;
    energy=.5*f*(re*re+im*im);for(int a=0;a<3;++a)for(int b=0;b<3;++b)
        strain[3*a+b]=active?energy*(2*(g.sigma2+1/norm)*k[a]*k[b]-(a==b?1.:0.)):0.;return f;
}
LSC_MESH_HD inline double gather(Grid g,const double* phi,const int* base,const double* theta,const double* rho,
    double* dr,double* dx,double* strain){
    int P=g.order;const auto tx=theta,ty=theta+3*P,tz=theta+6*P;double q=0,grad[3]={},Hu[3][3]={};
    for(int z=0;z<P;++z)for(int y=0;y<P;++y)for(int x=0;x<P;++x){double v=phi[at(g,base,x,y,z)];
        q+=tx[x]*ty[y]*tz[z]*v;grad[0]+=tx[P+x]*ty[y]*tz[z]*v;grad[1]+=tx[x]*ty[P+y]*tz[z]*v;grad[2]+=tx[x]*ty[y]*tz[P+z]*v;
        Hu[0][0]+=tx[2*P+x]*ty[y]*tz[z]*v;Hu[1][1]+=tx[x]*ty[2*P+y]*tz[z]*v;Hu[2][2]+=tx[x]*ty[y]*tz[2*P+z]*v;
        Hu[0][1]+=tx[P+x]*ty[P+y]*tz[z]*v;Hu[0][2]+=tx[P+x]*ty[y]*tz[P+z]*v;Hu[1][2]+=tx[x]*ty[P+y]*tz[P+z]*v;}
    Hu[1][0]=Hu[0][1];Hu[2][0]=Hu[0][2];Hu[2][1]=Hu[1][2];double gc[3]={},H[3][3]={},m[3]={g.a1*rho[3],g.a1*rho[1],g.a1*rho[2]};
    for(int a=0;a<3;++a)for(int d=0;d<3;++d){gc[a]+=g.J[3*d+a]*grad[d];for(int b=0;b<3;++b)for(int e=0;e<3;++e)H[a][b]+=g.J[3*d+a]*Hu[d][e]*g.J[3*e+b];}
    constexpr int cart[3]={3,1,2};dr[0]=g.a0*q;
    for(int a=0;a<3;++a){dr[cart[a]]=g.a1*gc[a];dx[a]=g.a0*rho[0]*gc[a];for(int b=0;b<3;++b){dx[a]+=H[a][b]*m[b];strain[3*a+b]=-m[a]*gc[b];}}
    double self=0;if(!g.include_self)for(int d=0;d<4;++d){self-=.5*rho[d]*rho[d]*g.self[d];dr[d]-=rho[d]*g.self[d];}return self;
}
struct KernelResult {std::vector<double> values;double energy=0;std::array<double,9> strain{};};
} // namespace lsc_partition
#undef LSC_MESH_HD
