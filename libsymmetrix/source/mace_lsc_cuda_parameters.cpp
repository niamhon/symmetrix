#include "mace_lsc_cuda.hpp"
#include "mace_lsc_model.hpp"
#include "mace_lsc_cuda_tables.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace {
using json = nlohmann::json;
using PackedShape = std::vector<size_t>;

[[noreturn]] void error(const std::string& message)
{
    throw std::invalid_argument("MACELSCCuda: " + message);
}

float fp32(double value, const std::string& name)
{
    if ((std::bit_cast<std::uint64_t>(value)&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000)
        or std::abs(value) > std::numeric_limits<float>::max()) error(name + " is outside the FP32 range");
    return static_cast<float>(value);
}

// IEEE binary16, round-to-nearest-even. Keeping packing on the host lets CPU
// tests verify the exact upload representation without requiring a CUDA device.
std::uint16_t half_bits(float value)
{
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
    const int exponent = (bits >> 23) & 255;
    const std::uint32_t fraction = bits & 0x7fffff;
    std::uint32_t result, remainder, halfway;
    if (exponent <= 112) {
        const int shift = 126-exponent;
        if (shift > 24) return sign;
        const auto mantissa = fraction | 0x800000;
        result = mantissa >> shift;
        remainder = mantissa & ((1u << shift)-1);
        halfway = 1u << (shift-1);
    } else {
        result = ((exponent-112) << 10) + (fraction >> 13);
        remainder = fraction & 0x1fff;
        halfway = 0x1000;
    }
    if (remainder > halfway or (remainder == halfway and (result & 1))) ++result;
    if (result >= 0x7c00) error("mixed-precision coefficient exceeds the FP16 range");
    return sign | static_cast<std::uint16_t>(result);
}

float half_value(std::uint16_t value)
{
    const int exponent = (value >> 10) & 31, fraction = value & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(float(fraction), -24)
                                          : std::ldexp(float(1024+fraction), exponent-25);
    return (value & 0x8000) ? -magnitude : magnitude;
}

void flatten(const json& value, std::vector<double>& out)
{
    if (value.is_array()) for (const auto& item : value) flatten(item, out);
    else out.push_back(value.get<double>());
}
std::vector<double> numbers(const json& value)
{
    std::vector<double> out;
    flatten(value, out);
    return out;
}

template<class T> void put(LSCCudaParameters& model, const std::string& name, const std::string& dtype,
                           PackedShape shape, std::vector<T> values)
{
    const size_t count = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (count != values.size()) error("incorrect packed shape for " + name);
    if (not model.arrays.emplace(name, LSCCudaArray{dtype, std::move(shape), std::move(values)}).second)
        error("duplicate packed parameter " + name);
}

void real(LSCCudaParameters& model, const std::string& name, PackedShape shape, const std::vector<double>& values)
{
    std::vector<float> out;
    out.reserve(values.size());
    for (double value : values) out.push_back(fp32(value, name));
    put(model, name, "float32", std::move(shape), std::move(out));
}
void real(LSCCudaParameters& model, const std::string& name, PackedShape shape, const json& value)
{
    real(model, name, std::move(shape), numbers(value));
}
void integer(LSCCudaParameters& model, const std::string& name, PackedShape shape, const json& value)
{
    std::vector<std::int32_t> out;
    for (double x : numbers(value)) {
        if (x < std::numeric_limits<std::int32_t>::min() or x > std::numeric_limits<std::int32_t>::max())
            error("integer is outside the int32 range: " + name);
        out.push_back(static_cast<std::int32_t>(x));
    }
    put(model, name, "int32", std::move(shape), std::move(out));
}

void mixed_matrix(LSCCudaParameters& model, const std::string& name, size_t in, size_t out)
{
    if (model.precision != "float32-mixed" or in == 0 or out == 0 or in%16 != 0 or out%32 != 0) return;
    const auto& weights = std::get<std::vector<float>>(model.arrays.at(name).values);
    float top = 0;
    for (float x : weights) top = std::max(top, std::abs(x));
    int exponent = 0;
    if (top > 0) std::frexp(top, &exponent);
    const int shift = top > 0 ? 13-exponent : 0; // Polar's matrix-wide power-of-two scale.
    std::vector<std::uint16_t> hi(in*out), lo(in*out), thi(in*out), tlo(in*out);
    for (size_t i = 0; i < in; ++i)
        for (size_t j = 0; j < out; ++j) {
            const float scaled = std::ldexp(weights[i*out+j], shift);
            const auto h = half_bits(scaled), l = half_bits(scaled-half_value(h));
            hi[j*in+i] = thi[i*out+j] = h;
            lo[j*in+i] = tlo[i*out+j] = l;
        }
    put(model, name+".mixed.forward_hi", "float16", {out,in}, std::move(hi));
    put(model, name+".mixed.forward_lo", "float16", {out,in}, std::move(lo));
    put(model, name+".mixed.reverse_hi", "float16", {in,out}, std::move(thi));
    put(model, name+".mixed.reverse_lo", "float16", {in,out}, std::move(tlo));
    put(model, name+".mixed.shift", "int32", {1}, std::vector<std::int32_t>{shift});
}

void linear(LSCCudaParameters& model, const std::string& prefix, const json& j)
{
    integer(model, prefix+".in", {j.at("in").size()}, j.at("in"));
    integer(model, prefix+".out", {j.at("out").size()}, j.at("out"));
    for (size_t l = 0; l < j.at("weights").size(); ++l) {
        const size_t in = j.at("in")[l], out = j.at("out")[l];
        const auto name = prefix+".weights."+std::to_string(l);
        real(model, name, {in,out}, j.at("weights")[l]);
        mixed_matrix(model, name, in, out);
    }
    real(model, prefix+".bias", {j.at("bias").size()}, j.at("bias"));
}

std::vector<double> oxidation_embeddings(const json& embedding, const std::vector<double>& formal)
{
    const double start = embedding.at("range")[0], factor = embedding.at("factor");
    const auto weights = numbers(embedding.at("linear").at("weights")[0]);
    const auto bias = numbers(embedding.at("linear").at("bias"));
    std::vector<double> out(formal.size()*128);
    for (size_t e = 0; e < formal.size(); ++e) {
        const double x = (formal[e]-start)*factor-1;
        double previous = 1, current = x;
        std::array<double,8> basis;
        for (int k = 0; k < 8; ++k) {
            basis[k] = current;
            const double next = 2*x*current-previous;
            previous = current; current = next;
        }
        for (int u = 0; u < 128; ++u) {
            double value = bias[u];
            for (int k = 0; k < 8; ++k) value += basis[k]*weights[k*128+u];
            out[e*128+u] = value;
        }
    }
    return out;
}

void embedding(LSCCudaParameters& model, const std::string& prefix, const json& j)
{
    linear(model, prefix+".linear", j.at("linear"));
    real(model, prefix+".range", {2}, j.at("range"));
    real(model, prefix+".factor", {1}, j.at("factor"));
}

void convolution(LSCCudaParameters& model, const std::string& prefix, const json& j)
{
    integer(model, prefix+".spec", {j.at("spec").size(),4}, j.at("spec"));
    real(model, prefix+".coefficients", {j.at("coefficients").size()}, j.at("coefficients"));
    integer(model, prefix+".in", {j.at("in").size()}, j.at("in"));
    integer(model, prefix+".out", {j.at("out").size()}, j.at("out"));
}

void radial(LSCCudaParameters& model, const std::string& prefix, const json& values,
             const json& derivs, const json& projection, bool compress)
{
    const size_t functions = values.size(), knots = values[0].size(), outputs = projection[0].size();
    const auto v = values.get<std::vector<std::vector<double>>>();
    const auto d = derivs.get<std::vector<std::vector<double>>>();
    const auto w = projection.get<std::vector<std::vector<double>>>();
    const double h = (model.r_cut-model.r_min)/(knots-1);
    real(model, prefix+".hidden_values", {functions,knots}, values);
    real(model, prefix+".hidden_derivs", {functions,knots}, derivs);
    real(model, prefix+".projection", {functions,outputs}, projection);
    real(model, prefix+".grid", {3}, std::vector<double>{model.r_min,h,1/h});
    std::vector<float> table((knots-1)*4*outputs);
    // Fold the last radial projection once, as in Polar's CUDA loader.
    #pragma omp parallel for
    for (long o = 0; o < outputs; ++o) {
        std::vector<double> pv(knots,0), pd(knots,0);
        for (size_t f = 0; f < functions; ++f)
            for (size_t k = 0; k < knots; ++k) {
                pv[k] += w[f][o]*v[f][k];
                pd[k] += w[f][o]*d[f][k];
            }
        for (size_t k = 0; k < knots-1; ++k) {
            const double a0=pv[k], a1=h*pd[k];
            const double a2=3*(pv[k+1]-pv[k])-h*(2*pd[k]+pd[k+1]);
            const double a3=2*(pv[k]-pv[k+1])+h*(pd[k]+pd[k+1]);
            // Check derived coefficients in serial below, outside OpenMP workers.
            table[(k*4+0)*outputs+o]=static_cast<float>(a0+a1/2+a2/4+a3/8);
            table[(k*4+1)*outputs+o]=static_cast<float>(a1+a2+0.75*a3);
            table[(k*4+2)*outputs+o]=static_cast<float>(a2+1.5*a3);
            table[(k*4+3)*outputs+o]=static_cast<float>(a3);
        }
    }
    for (float x : table)
        if ((std::bit_cast<std::uint32_t>(x)&0x7f800000u)==0x7f800000u)
            error(prefix+" projected spline is outside the FP32 range");
    if (not compress) put(model, prefix+".coefficients", "float32", {knots-1,4,outputs}, std::move(table));
    else {
        std::vector<float> c01((knots-1)*2*outputs);
        std::vector<std::uint16_t> c23((knots-1)*2*outputs);
        for (size_t k=0; k<knots-1; ++k)
            for (size_t o=0; o<outputs; ++o) {
                const auto c=table.data()+k*4*outputs+o;
                if (std::max(std::abs(c[2*outputs]),std::abs(c[3*outputs]))>6e4f)
                    error("second-layer spline coefficients exceed the FP16 range");
                c01[(k*2)*outputs+o]=c[0]; c01[(k*2+1)*outputs+o]=c[outputs];
                c23[(k*outputs+o)*2]=half_bits(c[2*outputs]);
                c23[(k*outputs+o)*2+1]=half_bits(c[3*outputs]);
            }
        put(model,prefix+".c01","float32",{knots-1,2,outputs},std::move(c01));
        put(model,prefix+".c23","float16",{knots-1,outputs,2},std::move(c23));
    }
}

// Foundation's exact first-layer basis: keep the radial MLP's 64 hidden
// functions and fold the fixed initial species features and linear maps into Q.
// Other basis widths retain the existing expanded convolution path.
void compact_first_layer(LSCCudaParameters& model,const json& j,const std::vector<double>& initial)
{
    constexpr int B=64,C=128;
    const auto& values=j.at("radial_spline_values_0")[0];
    if(values.size()!=B)return;
    const auto v=values.get<std::vector<std::vector<double>>>();
    const auto d=j.at("radial_spline_derivs_0")[0].get<std::vector<std::vector<double>>>();
    const auto U=j.at("radial_weights_0").get<std::vector<std::vector<double>>>();
    const int T=model.num_elements,K=model.num_knots-1,F=T*B;
    const double h=(model.r_cut-model.r_min)/K;
    std::vector<double> table(size_t(K)*4*B);
    for(int k=0;k<K;++k)for(int n=0;n<B;++n){
        const double a0=v[n][k],a1=h*d[n][k];
        const double a2=3*(v[n][k+1]-v[n][k])-h*(2*d[n][k]+d[n][k+1]);
        const double a3=2*(v[n][k]-v[n][k+1])+h*(d[n][k]+d[n][k+1]);
        table[(size_t(k)*4+0)*B+n]=a0+a1/2+a2/4+a3/8;
        table[(size_t(k)*4+1)*B+n]=a1+a2+0.75*a3;
        table[(size_t(k)*4+2)*B+n]=a2+1.5*a3;
        table[(size_t(k)*4+3)*B+n]=a3;
    }
    real(model,"compact_first.basis",{size_t(K),4,B},table);
    std::vector<double> cg(16);
    const auto& conv=j.at("conv_tp_0");
    for(int q=0;q<16;++q)cg[conv.at("spec")[q][2].get<int>()]=conv.at("coefficients")[q];
    real(model,"compact_first.cg",{16},cg);
    const auto up=numbers(j.at("linear_up_0").at("weights")[0]);
    const auto bias=numbers(j.at("linear_up_0").at("bias"));
    std::vector<double> lifted(size_t(T)*C);
    for(int b=0;b<T;++b)for(int c=0;c<C;++c){
        double x=bias[c];for(int i=0;i<C;++i)x+=initial[b*C+i]*up[i*C+c];
        lifted[b*C+c]=x;
    }
    for(int l=0;l<4;++l){
        const auto W=numbers(j.at("linear_0").at("weights")[l]);
        std::vector<double> map(size_t(F)*C);
        for(int b=0;b<T;++b)for(int n=0;n<B;++n)for(int k=0;k<C;++k){
            double x=0;for(int c=0;c<C;++c)x+=U[n][l*C+c]*lifted[b*C+c]*W[c*C+k];
            map[(size_t(b)*B+n)*C+k]=x;
        }
        const auto name="compact_first.weights."+std::to_string(l);
        real(model,name,{size_t(F),C},map);mixed_matrix(model,name,F,C);
    }
}

template<int Layer> void validate_convolution(const json& file)
{
    const auto name="conv_tp_"+std::to_string(Layer);
    const auto& spec=file.at(name).at("spec");
    using Conv=lsc_cuda::LocalConvolution<Layer>;
    bool same=spec.size()==Conv::count;
    for(int i=0;same and i<Conv::count;++i){
        const auto t=Conv::terms[i];
        same=spec[i]==json({128*t.lme,128*t.lm2,t.lm1,128*t.path});
    }
    if(not same)error("unsupported GPU convolution instructions in "+name+"; use the CPU evaluator");
}

void validate_tables(const json& file)
{
    using T=Tables<1>;
    for (int layer=0;layer<2;++layer) {
        const auto suffix=std::to_string(layer);
        const auto& b=file.at("B"+suffix+"_spec");
        const auto& c=file.at("B"+suffix+"_coefficients");
        const auto& m=file.at("M"+suffix+"_spec");
        bool same=b.size()==T::num_b and m.size()==T::num_m;
        for (int t=0;same and t<T::num_b;++t)
            same=b[t]==json({T::b[t].q,T::b[t].a,T::b[t].b}) and std::abs(c[t].get<double>()-T::b[t].c)<1e-6;
        for (int t=0;same and t<T::num_m;++t)
            same=m[t]==json({T::m[t].o,T::m[t].q,T::m[t].n});
        if (not same) error("unsupported tensor-product tables in layer "+suffix+"; use the CPU evaluator");
    }
}
} // namespace

size_t LSCCudaArray::size() const
{
    return shape.empty()?0:std::accumulate(shape.begin(),shape.end(),size_t(1),std::multiplies<size_t>());
}
size_t LSCCudaArray::bytes() const
{
    return size()*(dtype=="float64"?8:dtype=="float16"?2:4);
}
const void* LSCCudaArray::data() const
{
    return std::visit([](const auto& v)->const void* {return v.data();},values);
}
void* LSCCudaArray::data()
{
    return std::visit([](auto& v)->void* {return v.data();},values);
}
size_t LSCCudaParameters::bytes() const
{
    size_t sum=0;for(const auto& [name,array]:arrays)sum+=array.bytes();return sum;
}

LSCCudaParameters prepare_lsc_cuda_parameters(const std::string& filename,const std::string& precision)
{
    if (precision!="float32" and precision!="float32-mixed") error("precision must be float32 or float32-mixed");
    const auto host=read_validated_lsc_model(filename);
    const auto& j=host.document;
    if (not j.at("electrostatics").contains("energy_parameters")) error("missing electrostatic energy parameters; re-export the checkpoint");
    validate_tables(j);
    validate_convolution<0>(j);validate_convolution<1>(j);
    LSCCudaParameters model;
    model.precision=precision;
    model.num_elements=j.at("num_elements");model.num_channels=j.at("num_channels");
    model.num_interactions=j.at("architecture").at("num_interactions");
    model.num_knots=j.at("export_options").at("num_spline_points");
    model.r_cut=j.at("r_cut");model.r_min=j.at("radial_spline_r_min");
    model.atomic_numbers=j.at("atomic_numbers").get<std::vector<int>>();
    const size_t species=model.num_elements;
    integer(model,"atomic_numbers",{species},j.at("atomic_numbers"));
    integer(model,"source_element_indices",{species},j.at("source_element_indices"));
    put(model,"atomic_energies","float64",{species},numbers(j.at("atomic_energies")));
    real(model,"formal_charges",{species},j.at("formal_charges"));
    real(model,"node_embedding",{species,128},j.at("node_embedding"));
    const auto formal=numbers(j.at("formal_charges"));
    const auto& ox=j.at("oxidation_state_mixer").at("embedding");
    embedding(model,"oxidation_state_mixer.embedding",ox);
    auto initial=oxidation_embeddings(ox,formal);
    const auto base=numbers(j.at("node_embedding"));
    for(size_t i=0;i<initial.size();++i)initial[i]+=base[i];
    real(model,"initial_embedding",{species,128},initial);
    real(model,"radial_domain",{2},std::vector<double>{model.r_min,model.r_cut});
    for(int t=0;t<2;++t) {
        const auto suffix="_"+std::to_string(t), number=std::to_string(t);
        for(const std::string name:{"linear_up","linear","product_linear","readout"})linear(model,name+suffix,j.at(name+suffix));
        for(size_t e=0;e<species;++e)linear(model,"skip"+suffix+"."+std::to_string(e),j.at("skip"+suffix)[e]);
        convolution(model,"conv_tp"+suffix,j.at("conv_tp"+suffix));
        const auto B="B"+number,M="M"+number;
        integer(model,B+"_spec",{j.at(B+"_spec").size(),3},j.at(B+"_spec"));
        real(model,B+"_coefficients",{j.at(B+"_coefficients").size()},j.at(B+"_coefficients"));
        integer(model,M+"_spec",{j.at(M+"_spec").size(),3},j.at(M+"_spec"));
        real(model,M+"_weights",{species,j.at(M+"_spec").size(),128},j.at(M+"_weights"));
        radial(model,"radial"+suffix,j.at("radial_spline_values"+suffix)[0],j.at("radial_spline_derivs"+suffix)[0],j.at("radial_weights"+suffix),t==1 and precision=="float32-mixed");
        const auto& source=j.at("sources")[t];const auto prefix="sources."+number;
        embedding(model,prefix+".oxidation_embedding",source.at("oxidation_embedding"));
        real(model,prefix+".conditioning_embedding",{species,128},oxidation_embeddings(source.at("oxidation_embedding"),formal));
        for(int l=0;l<2;++l) {
            const auto name=prefix+".oxidation."+std::to_string(l);
            real(model,name,{128,128},source.at("oxidation")[l]);mixed_matrix(model,name,128,128);
        }
        for(const std::string name:{"linear1","linear2","transfer_readout","multipoles"})linear(model,prefix+"."+name,source.at(name));
        convolution(model,prefix+".conv_tp",source.at("conv_tp"));
        real(model,prefix+".transfer_scale",{1},source.at("transfer_scale"));
        radial(model,prefix+".radial",source.at("radial_spline_values"),source.at("radial_spline_derivs"),source.at("radial_weights"),false);
    }
    compact_first_layer(model,j,initial);
    const auto& electro=j.at("electrostatics");const auto& energy=electro.at("energy_parameters");
    real(model,"electrostatics.kspace_cutoff",{1},j.at("kspace_cutoff"));
    real(model,"electrostatics.density_smearing_width",{1},electro.at("density_smearing_width"));
    put(model,"electrostatics.include_self_interaction","int32",{1},std::vector<std::int32_t>{electro.at("include_self_interaction").get<bool>()?1:0});
    real(model,"electrostatics.field_constant",{1},energy.at("field_constant"));
    for(const std::string name:{"sigma2","a","self_overlap"})real(model,"electrostatics.reciprocal."+name,{name=="a"?2u:name=="self_overlap"?4u:1u},energy.at("reciprocal").at(name));
    for(const std::string name:{"sigma","offset","distance_epsilon","self_overlap"})real(model,"electrostatics.realspace."+name,{name=="self_overlap"?4u:1u},energy.at("realspace").at(name));
    real(model,"electrostatics.realspace.offset_vectors",{3,3},energy.at("realspace").at("offset_vectors"));
    return model;
}
