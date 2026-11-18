/*
   Shared mesh rules adapted from pair_symmetrix_mace_polar.cpp,
   wcwitt/symmetrix commit b907f04026f2dda4c43d66511a7c37db8a82a7d4.
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. Distributed under the GNU General Public
   License; see pair_symmetrix/LICENSE in this repository.
*/
#pragma once
#include "mace_lsc_electrostatics.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
// Shared Polar mesh geometry, cutoff convention and cardinal B-splines.
namespace lsc_mesh {
constexpr int MAXORDER = 10;
constexpr size_t MAXPOINTS = 1U << 24;
constexpr double PI = std::numbers::pi;
const double Y00 = 0.5/std::sqrt(PI), Y1 = std::sqrt(0.75/PI);
inline void require(bool valid, const char* message)
{
    if (!valid) throw std::invalid_argument(std::string("LSCMesh: ") + message);
}
// Polar's cardinal B-spline recurrence, first and second derivatives, reversed stencil.
inline void bspline(int P, double f, double* t)
{
    double M[MAXORDER+1][MAXORDER+1] = {};
    M[1][0] = 1.0;
    for (int n=2; n<=P; ++n)
        for (int j=0; j<n; ++j)
            M[n][j] = ((f+j)*M[n-1][j] + (n-f-j)*(j>0 ? M[n-1][j-1] : 0.0))/(n-1);
    for (int k=0; k<P; ++k) {
        const int j=P-1-k;
        t[k]=M[P][j];
        t[P+k]=M[P-1][j]-(j>0 ? M[P-1][j-1] : 0.0);
        t[2*P+k]=M[P-2][j]-2*(j>0 ? M[P-2][j-1] : 0.0)+(j>1 ? M[P-2][j-2] : 0.0);
    }
}
inline bool factorable(int n)
{
    for (int p : {2,3,5}) while (n%p==0) n/=p;
    return n==1;
}
inline int wrap(int x,int n) { return (x%n+n)%n; }
struct Geometry {
    std::array<int,3> shape{}, bounds{};
    std::array<std::array<double,3>,3> reciprocal{}, J{};
    double volume = 0;
};
inline Geometry geometry(const LSCElectrostaticsContext& context,double spacing,int order)
{
    require(context.pbc==std::array<bool,3>{true,true,true}, "mesh requires three periodic directions");
    require(context.mode=="pbc" || context.mode=="mixed_periodic", "mesh requires reciprocal periodic electrostatics");
    const auto& c=context.cell;const auto& p=context.parameters;
    for(double value:c) require(lsc_electrostatics_finite(value),"cell must be finite");
    require(lsc_electrostatics_finite(p.kspace_cutoff) && p.kspace_cutoff>0,"invalid reciprocal cutoff");
    require(lsc_electrostatics_finite(p.sigma2) && p.sigma2>0,"invalid Gaussian width");
    require(lsc_electrostatics_finite(p.field_constant) && p.field_constant>0,"invalid field constant");
    for(double value:p.a) require(lsc_electrostatics_finite(value) && value>0,"invalid basis amplitudes");
    for(double value:p.reciprocal_self) require(lsc_electrostatics_finite(value),"invalid self overlap");
    auto cross=[&](int i,int j) { return std::array<double,3>{c[3*i+1]*c[3*j+2]-c[3*i+2]*c[3*j+1],c[3*i+2]*c[3*j]-c[3*i]*c[3*j+2],c[3*i]*c[3*j+1]-c[3*i+1]*c[3*j]}; };
    const auto cofactor=cross(1,2);const double det=c[0]*cofactor[0]+c[1]*cofactor[1]+c[2]*cofactor[2];
    require(lsc_electrostatics_finite(det) && det!=0,"cell must be nonsingular");
    Geometry result;result.volume=std::abs(det);size_t points=1;
    for(int d=0;d<3;++d) {
        const auto v=cross((d+1)%3,(d+2)%3);
        for(int a=0;a<3;++a) result.reciprocal[d][a]=2*PI*v[a]/det;
        const double length=std::hypot(c[3*d],c[3*d+1],c[3*d+2]);
        const double bound=std::ceil(p.kspace_cutoff*length/(2*PI));
        const double sites=std::max({std::ceil(length/spacing),2*std::floor(p.kspace_cutoff*length/(2*PI))+1,double(order)});
        require(lsc_electrostatics_finite(sites) && sites<=double(MAXPOINTS) && bound>=1 && bound<=MAXPOINTS,"requested mesh is too large");
        result.bounds[d]=int(bound);result.shape[d]=int(sites);
        while(!factorable(result.shape[d])) ++result.shape[d];
        require(size_t(result.shape[d])<=MAXPOINTS/points,"requested mesh is too large");
        points*=size_t(result.shape[d]);
        for(int a=0;a<3;++a) result.J[d][a]=result.shape[d]*result.reciprocal[d][a]/(2*PI);
    }
    return result;
}
// Reflect a mode into the same half-space used by native LSC's direct sum.
// Keeping its endpoints also covers exact-cutoff integer boundary cases.
inline bool direct_mode(std::array<int,3> n,const std::array<int,3>& bounds)
{
    int first=0;while(first<3 && n[first]==0) ++first;
    if(first==3) return false;
    if(n[first]<0) for(auto& value:n) value=-value;
    return n[0]>=0 && n[0]<bounds[0] && n[1]>=(n[0]==0?0:-bounds[1]) && n[1]<bounds[1]
        && n[2]>=(n[0]==0 && n[1]==0?1:-bounds[2]) && n[2]<bounds[2];
}
} // namespace lsc_mesh
