#pragma once
#include "mace_foundation_tables.hpp"
#include <array>

namespace lsc_cuda {
// e3nn's exported UVU instructions group terms by radial path, input magnetic
// component, spherical harmonic, then output component. The underlying angular
// tensor products are the same Tables<0/1> used by the foundation/Polar kernels.
template<size_t N> constexpr std::array<Term,N> export_order(const Term (&input)[N])
{
    std::array<Term,N> result{};
    for(size_t i=0;i<N;++i)result[i]=input[i];
    auto less=[](const Term& a,const Term& b){
        return std::tuple(a.path,a.lm2,a.lm1,a.lme)<std::tuple(b.path,b.lm2,b.lm1,b.lme);
    };
    for(size_t i=1;i<N;++i){
        const auto value=result[i];size_t j=i;
        while(j>0 and less(value,result[j-1])){result[j]=result[j-1];--j;}
        result[j]=value;
    }
    return result;
}
template<int Layer> struct LocalConvolution {
    using T=Tables<Layer>;
    static constexpr auto terms=export_order(T::terms);
    static constexpr int paths=T::num_paths, count=T::num_terms;
    static constexpr int width=Layer==0?16:40;
};
} // namespace lsc_cuda
