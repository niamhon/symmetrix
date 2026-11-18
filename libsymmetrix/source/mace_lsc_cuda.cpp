#include "mace_lsc_cuda.hpp"
#include "mace_lsc_forward.hpp"
#include "mace_lsc_charge_exchange.hpp"
#include "mace_lsc_reverse_exchange.hpp"
#include "mace_lsc_cuda_reverse_exchange.cuh"
#include "mace_lsc_cuda_charge_exchange.cuh"
#include "mace_lsc_cuda_forward.cuh"
#include <optional>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "mace_lsc_cuda_sources.cuh"
#include "mace_lsc_cuda_electrostatics.cuh"
#include "mace_lsc_cuda_gemm.cuh"
#include "mace_lsc_cuda_reverse.cuh"
#include <algorithm>
#include <bit>
#include <type_traits>
#include <limits>
#include <cublas_v2.h>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {
void check(cudaError_t result, const char* what)
{
    if (result != cudaSuccess) throw std::runtime_error(std::string("MACELSCCuda: ")+what+": "+cudaGetErrorString(result));
}
void check(cublasStatus_t result, const char* what)
{
    if (result != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("MACELSCCuda: ")+what+": cuBLAS error "+std::to_string(result));
}
struct Allocation {
    void* pointer=nullptr;
    size_t capacity=0;
    Allocation()=default;
    Allocation(const Allocation&)=delete;
    Allocation& operator=(const Allocation&)=delete;
    Allocation(Allocation&& other) noexcept : pointer(std::exchange(other.pointer,nullptr)), capacity(std::exchange(other.capacity,0)) {}
    ~Allocation(){if(pointer)cudaFree(pointer);}
};
LSCCudaArray host_array(const LSCCudaArray& metadata)
{
    LSCCudaArray out;out.dtype=metadata.dtype;out.shape=metadata.shape;
    const auto n=out.size();
    if(out.dtype=="float32")out.values=std::vector<float>(n);
    else if(out.dtype=="float64")out.values=std::vector<double>(n);
    else if(out.dtype=="int32")out.values=std::vector<std::int32_t>(n);
    else if(out.dtype=="float16")out.values=std::vector<std::uint16_t>(n);
    else throw std::runtime_error("MACELSCCuda: unknown parameter dtype");
    return out;
}
} // namespace

struct MACELSCCuda::DeviceData {
    std::optional<LSCGraph> graph;
    int local_stage=0,reverse_stage=0,sender_nodes=0;
    LSCForwardState forward;
    LSCChargeExchange charge_exchange;
    LSCReverseExchange reverse_exchange;
    std::array<double,9> mpi_electrostatic_strain{};
    bool charge_legacy=false;
    LSCCudaElectrostaticsHook electrostatics_hook;
    bool energy_uses_hook=false;
    cudaStream_t stream=nullptr;
    cublasHandle_t blas=nullptr;
    int device_id=-1,num_sms=0;
    lsc_cuda::ElectrostaticParameters electrostatic{};
    double electrostatic_cutoff=0,energy_volume=0;
    int energy_k_count=0;
    std::string energy_mode;
    std::array<double,9> energy_cell{};
    std::array<double,3> energy_field{};
    size_t sgemm_count=0,tensor_core_gemm_count=0;
    std::map<std::string,int> matrix_shifts;
    std::map<std::string,std::vector<size_t>> local_layouts;
    std::map<std::string,Allocation> allocations;
    std::map<std::string,LSCCudaArray> metadata;
    std::map<std::string,Allocation> workspace;
    std::map<std::string,LSCCudaArray> geometry_metadata, spline_metadata, local_metadata, charge_metadata, force_metadata;
    template<class T> T* buffer(const std::string& name,size_t count) {
        auto& a=workspace.try_emplace(name).first->second;
        if(count>std::numeric_limits<size_t>::max()/sizeof(T))
            throw std::invalid_argument("MACELSCCuda: excessive workspace size");
        const size_t bytes=count*sizeof(T);
        if(bytes>a.capacity){
            void* fresh=nullptr;check(cudaMalloc(&fresh,bytes),"allocate graph workspace");
            if(a.pointer)cudaFree(a.pointer);
            a.pointer=fresh;a.capacity=bytes;
        }
        return static_cast<T*>(a.pointer);
    }
    void active() const {
        int id=-1;check(cudaGetDevice(&id),"get active device");
        if(id!=device_id)throw std::runtime_error("MACELSCCuda: activate the model's CUDA device");
    }
    std::map<std::string,LSCCudaArray> download(const std::map<std::string,LSCCudaArray>& info) const {
        active();
        std::map<std::string,LSCCudaArray> out;
        for(const auto& [name,array]:info){
            auto host=host_array(array);
            if(host.bytes())check(cudaMemcpyAsync(host.data(),workspace.at(name).pointer,host.bytes(),cudaMemcpyDeviceToHost,stream),"read back graph results");
            out.emplace(name,std::move(host));
        }
        check(cudaStreamSynchronize(stream),"complete graph read-back");return out;
    }
    void result(std::map<std::string,LSCCudaArray>& info,const std::string& name,
                const std::string& dtype,std::vector<size_t> shape) {
        LSCCudaArray a;a.dtype=dtype;a.shape=std::move(shape);info.emplace(name,std::move(a));
    }
    template<class T> const T* parameter(const std::string& name) const {
        return static_cast<const T*>(allocations.at(name).pointer);
    }
    template<class T> T* working(const std::string& name) const {
        return static_cast<T*>(workspace.at(name).pointer);
    }
    void gemm(const std::string& weight,int m,int n,int k,const float* a,float* c,bool mixed) {
        if(m==0 or n==0)return;
        if(mixed and k%16==0 and n%32==0 and k%4==0 and n%2==0
           and reinterpret_cast<uintptr_t>(a)%16==0 and reinterpret_cast<uintptr_t>(c)%8==0
           and (m+127LL)/128*(n/32)>=4*num_sms){
            std::vector<lsc_cuda::GemmDesc> pending{{a,
                parameter<__half>(weight+".mixed.forward_hi"),parameter<__half>(weight+".mixed.forward_lo"),
                c,m,n,k,k,n,0,matrix_shifts.at(weight),1.f,0.f}};
            lsc_cuda::gemm_batches(pending,num_sms,stream);
            check(cudaGetLastError(),"launch mixed tensor-core GEMM");++tensor_core_gemm_count;
        }else{
            const float one=1.f,zero=0.f;
            check(cublasSgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&one,
                              parameter<float>(weight),n,a,k,&zero,c,n),"FP32 local linear map");
            ++sgemm_count;
        }
    }
    void gemm_reverse(const std::string& weight,int m,int n,int k,const float* a,float* c,bool mixed) {
        if(m==0 or n==0)return;
        if(mixed and k%16==0 and n%32==0 and reinterpret_cast<uintptr_t>(a)%16==0
           and reinterpret_cast<uintptr_t>(c)%8==0 and (m+127LL)/128*(n/32)>=4*num_sms){
            std::vector<lsc_cuda::GemmDesc> pending{{a,parameter<__half>(weight+".mixed.reverse_hi"),
                parameter<__half>(weight+".mixed.reverse_lo"),c,m,n,k,k,n,0,matrix_shifts.at(weight),1.f,1.f}};
            lsc_cuda::gemm_batches(pending,num_sms,stream);check(cudaGetLastError(),"launch mixed reverse GEMM");++tensor_core_gemm_count;
        }else{
            const float one=1.f;
            check(cublasSgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,parameter<float>(weight),k,a,k,&one,c,n),"FP32 reverse linear map");
            ++sgemm_count;
        }
    }
    void linear_reverse(const std::string& name,int nodes,const float* adj,float* input_adj,bool mixed) {
        size_t in_offset=0,out_offset=0;const int L=metadata.at(name+".out").shape[0];
        for(int l=0;l<L;++l){const auto weight=name+".weights."+std::to_string(l);const auto& shape=metadata.at(weight).shape;
            const int in=shape[0],out=shape[1];
            if(in>0 and out>0)gemm_reverse(weight,nodes*(2*l+1),in,out,adj+size_t(nodes)*out_offset,input_adj+size_t(nodes)*in_offset,mixed);
            in_offset+=(2*l+1)*in;out_offset+=(2*l+1)*out;
        }
    }
    // Same per-l GEMM layout as Polar::Linear::forward; zero input irreps
    // (notably the first residual's vector block) still get initialized.
    void linear(const std::string& name,int nodes,const float* input,float* output,bool mixed) {
        size_t in_offset=0,out_offset=0;
        const int L=metadata.at(name+".out").shape[0];
        for(int l=0;l<L;++l){
            const auto weight=name+".weights."+std::to_string(l);
            const auto& shape=metadata.at(weight).shape;
            const int in=shape[0],out=shape[1],rows=nodes*(2*l+1);
            auto y=output+size_t(nodes)*out_offset;
            if(out>0 and rows>0){
                if(in>0)gemm(weight,rows,out,in,input+size_t(nodes)*in_offset,y,mixed);
                else check(cudaMemsetAsync(y,0,size_t(rows)*out*sizeof(float),stream),"clear zero-input linear block");
            }
            in_offset+=(2*l+1)*in;out_offset+=(2*l+1)*out;
        }
        const int scalars=metadata.at(name+".weights.0").shape[1];
        if(nodes>0 and scalars>0){
            const size_t count=size_t(nodes)*scalars;
            lsc_cuda::linear_bias<<<(count+255)/256,256,0,stream>>>(count,scalars,output,parameter<float>(name+".bias"));
            check(cudaGetLastError(),"launch linear bias");
        }
    }
    ~DeviceData(){
        int previous=-1;cudaGetDevice(&previous);
        if(device_id>=0)cudaSetDevice(device_id);
        if(stream)cudaStreamSynchronize(stream);
        // Release solver plans/captured device resources while this device and
        // stream are still active. Member destruction happens after this body.
        electrostatics_hook = {};
        workspace.clear();
        allocations.clear();
        if(blas)cublasDestroy(blas);
        if(stream)cudaStreamDestroy(stream);
        if(previous>=0 and previous!=device_id)cudaSetDevice(previous);
    }
};

bool lsc_cuda_compiled(){return true;}
bool lsc_cuda_available()
{
    int count=0;
    const auto result=cudaGetDeviceCount(&count);
    if(result!=cudaSuccess){cudaGetLastError();return false;}
    return count>0;
}

MACELSCCuda::MACELSCCuda(const std::string& filename,const std::string& selected_precision)
{
    // Validate/pack before creating resources; failure leaves no device state.
    auto parameters=prepare_lsc_cuda_parameters(filename,selected_precision);
    if(not lsc_cuda_available())throw std::runtime_error("MACELSCCuda: no usable CUDA device; run inside a GPU allocation");
    device=std::make_unique<DeviceData>();
    check(cudaGetDevice(&device_id),"get device");device->device_id=device_id;
    cudaDeviceProp props;
    check(cudaGetDeviceProperties(&props,device_id),"device properties");
    device->num_sms=props.multiProcessorCount;
    device_name=props.name;compute_capability_major=props.major;compute_capability_minor=props.minor;
    precision=selected_precision;float32_mixed=precision=="float32-mixed";
    tensor_cores=float32_mixed and props.major>=8;
    if(tensor_cores){
        cudaFuncAttributes attributes;
        check(cudaFuncGetAttributes(&attributes,lsc_cuda::k_gemm<32,32,1,1>),"inspect mixed GEMM kernel");
        tensor_cores=attributes.ptxVersion>=80;
        if(tensor_cores){
            check(cudaFuncSetAttribute(lsc_cuda::k_gemm<128,128,4,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      lsc_cuda::gemm_smem<128,128>),"configure wide mixed GEMM shared memory");
            check(cudaFuncSetAttribute(lsc_cuda::k_gemm<128,64,4,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      lsc_cuda::gemm_smem<128,64>),"configure mixed GEMM shared memory");
        }
    }
    num_elements=parameters.num_elements;num_channels=parameters.num_channels;
    num_interactions=parameters.num_interactions;num_knots=parameters.num_knots;
    r_cut=parameters.r_cut;r_min=parameters.r_min;atomic_numbers=parameters.atomic_numbers;
    auto scalar=[&](const std::string& name,size_t i=0){return double(std::get<std::vector<float>>(parameters.arrays.at(name).values).at(i));};
    auto& ep=device->electrostatic;
    ep.field_constant=scalar("electrostatics.field_constant");ep.sigma2=scalar("electrostatics.reciprocal.sigma2");
    ep.sigma=scalar("electrostatics.realspace.sigma");ep.offset=scalar("electrostatics.realspace.offset");
    ep.epsilon=scalar("electrostatics.realspace.distance_epsilon");
    for(int i=0;i<2;++i)ep.a[i]=scalar("electrostatics.reciprocal.a",i);
    for(int i=0;i<4;++i){ep.self_recip[i]=scalar("electrostatics.reciprocal.self_overlap",i);ep.self_real[i]=scalar("electrostatics.realspace.self_overlap",i);}
    for(int i=0;i<9;++i)ep.displacements[i]=scalar("electrostatics.realspace.offset_vectors",i);
    ep.include_self=std::get<std::vector<std::int32_t>>(parameters.arrays.at("electrostatics.include_self_interaction").values).at(0)!=0;
    device->electrostatic_cutoff=scalar("electrostatics.kspace_cutoff");
    check(cudaStreamCreateWithFlags(&device->stream,cudaStreamNonBlocking),"create stream");
    check(cublasCreate(&device->blas),"create cuBLAS handle");
    check(cublasSetStream(device->blas,device->stream),"set cuBLAS stream");
    for(const auto& [name,array]:parameters.arrays){
        const std::string ending=".mixed.shift";
        if(name.ends_with(ending))
            device->matrix_shifts[name.substr(0,name.size()-ending.size())]=std::get<std::vector<std::int32_t>>(array.values).at(0);
        LSCCudaArray info;info.dtype=array.dtype;info.shape=array.shape;
        device->metadata.emplace(name,std::move(info));
        auto& allocation=device->allocations.try_emplace(name).first->second;
        if(array.bytes()>0){
            check(cudaMalloc(&allocation.pointer,array.bytes()),"allocate model parameter");
            check(cudaMemcpyAsync(allocation.pointer,array.data(),array.bytes(),cudaMemcpyHostToDevice,device->stream),"upload model parameter");
        }
        device_parameter_bytes+=array.bytes();
    }
    check(cudaStreamSynchronize(device->stream),"complete model upload");
    parameters_on_device=true;
}
MACELSCCuda::~MACELSCCuda()=default;

std::map<std::string,LSCCudaArray> MACELSCCuda::parameter_metadata() const{return device->metadata;}
LSCCudaArray MACELSCCuda::download_parameter(const std::string& name) const
{
    const auto it=device->metadata.find(name);
    if(it==device->metadata.end())throw std::out_of_range("MACELSCCuda: unknown parameter "+name);
    int active=-1;check(cudaGetDevice(&active),"get active device");
    if(active!=device_id)throw std::runtime_error("MACELSCCuda: activate the model's CUDA device before read-back");
    auto out=host_array(it->second);
    if(out.bytes()>0)check(cudaMemcpyAsync(out.data(),device->allocations.at(name).pointer,out.bytes(),cudaMemcpyDeviceToHost,device->stream),"read back model parameter");
    check(cudaStreamSynchronize(device->stream),"complete parameter read-back");
    return out;
}

size_t MACELSCCuda::workspace_bytes() const
{
    size_t total=0;for(const auto& [name,a]:device->workspace)total+=a.capacity;return total;
}

void MACELSCCuda::compute_geometry(std::span<const int> types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> vectors)
{
    prepare_geometry(types,senders,receivers,vectors,types.size());
    device->graph=LSCGraph(types,senders,receivers,vectors);device->forward.reset(*device->graph);
}
void MACELSCCuda::prepare_geometry(std::span<const int> types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> vectors,int sender_nodes)
{
    device->active();
    device->graph.reset();device->local_stage=0;device->forward=LSCForwardState{};device->sender_nodes=sender_nodes;
    clear_charge_results();
    has_geometry=false;has_spline_results=false;spline_name.clear();
    has_local_results=false;local_energy=0;
    device->local_metadata.clear();device->local_layouts.clear();
    device->sgemm_count=0;device->tensor_core_gemm_count=0;batched_reverse_calls=0;
    device->geometry_metadata.clear();device->spline_metadata.clear();num_nodes=0;num_edges=0;
    if(types.size()>size_t(std::numeric_limits<int>::max()-1) or senders.size()>size_t(std::numeric_limits<int>::max())
       or senders.size()!=receivers.size() or vectors.size()!=3*senders.size())
        throw std::invalid_argument("MACELSCCuda: inconsistent or excessive graph sizes");
    const int N=types.size(),E=senders.size(),M=sender_nodes;const size_t ne=E,nn=N;
    const auto stream=device->stream;
    auto upload=[&](const std::string& name,const auto& input){
        using T=typename std::decay_t<decltype(input)>::value_type;
        auto ptr=device->buffer<T>(name,input.size());
        if(input.size())check(cudaMemcpyAsync(ptr,input.data(),input.size_bytes(),cudaMemcpyHostToDevice,stream),"upload graph input");
        return ptr;
    };
    auto dt=upload("node_types",types);auto ds=upload("senders",senders);auto dr=upload("receivers",receivers);
    auto xyz=upload("edge_vectors",vectors);
    auto g=device->buffer<float4>("geometry",ne);
    auto Y=device->buffer<float>("spherical_harmonics",16*ne);
    auto dY=device->buffer<float>("spherical_harmonics_gradient",48*ne);
    auto intervals=device->buffer<int>("intervals",ne);
    auto coordinate=device->buffer<float>("spline_coordinate",ne);
    auto ni=device->buffer<int>("incoming_count",nn);
    auto no=device->buffer<int>("outgoing_count",M);
    auto starts_i=device->buffer<int>("incoming_starts",nn+1);
    auto starts_o=device->buffer<int>("outgoing_starts",size_t(M)+1);
    auto ids=device->buffer<int>("edge_ids",ne);
    auto incoming=device->buffer<int>("incoming_edges",ne);
    auto outgoing=device->buffer<int>("outgoing_edges",ne);
    auto sorted_keys=device->buffer<int>("sorted_keys",ne);
    auto error=device->buffer<int>("error",1);
    check(cudaMemsetAsync(error,0,sizeof(int),stream),"clear input status");
    if(N){
        check(cudaMemsetAsync(ni,0,nn*sizeof(int),stream),"clear incoming counts");

        lsc_cuda::validate_types<<<(N+255LL)/256,256,0,stream>>>(dt,N,num_elements,error);
        check(cudaGetLastError(),"launch species validation");
    }
    if(M)check(cudaMemsetAsync(no,0,size_t(M)*sizeof(int),stream),"clear outgoing counts");
    if(E){
        const auto grid=static_cast<const float*>(device->allocations.at("radial_0.grid").pointer);
        if(M==N)lsc_cuda::geometry_kernel<false><<<(E+127LL)/128,128,0,stream>>>(xyz,ds,dr,N,E,r_min,r_cut,num_knots,grid,g,Y,dY,intervals,coordinate,ni,no,ids,error);
        else lsc_cuda::geometry_kernel<true><<<(E+127LL)/128,128,0,stream>>>(xyz,ds,dr,N,E,r_min,r_cut,num_knots,grid,g,Y,dY,intervals,coordinate,ni,no,ids,error,M);
        check(cudaGetLastError(),"launch geometry kernel");
    }
    int status=0;check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read input status");
    check(cudaStreamSynchronize(stream),"validate graph geometry");
    if(status&1)throw std::invalid_argument("MACELSCCuda: node_types species index is out of range");
    if(status&2)throw std::invalid_argument("MACELSCCuda: edge_index node index is out of range");
    if(status&4)throw std::invalid_argument("MACELSCCuda: edge_vectors must be finite");
    if(status&8)throw std::invalid_argument("MACELSCCuda: edge distance must be nonzero and inside [r_min, r_cut]");
    check(cudaMemsetAsync(starts_i,0,(nn+1)*sizeof(int),stream),"clear incoming starts");
    check(cudaMemsetAsync(starts_o,0,(size_t(M)+1)*sizeof(int),stream),"clear outgoing starts");
    if(N){
        size_t bytes=0;
        check(cub::DeviceScan::InclusiveSum(nullptr,bytes,ni,starts_i+1,N,stream),"size edge scan");
        auto temp=device->buffer<char>("scan_temp",bytes);
        check(cub::DeviceScan::InclusiveSum(temp,bytes,ni,starts_i+1,N,stream),"scan incoming counts");
    }
    if(M){
        size_t bytes=0;
        check(cub::DeviceScan::InclusiveSum(nullptr,bytes,no,starts_o+1,M,stream),"size outgoing edge scan");
        auto temp=device->buffer<char>("scan_temp",bytes);
        check(cub::DeviceScan::InclusiveSum(temp,bytes,no,starts_o+1,M,stream),"scan outgoing counts");
    }
    if(E){
        // CUB radix sorting is stable: preserve original edge order within a node.
        size_t bytes=0;
        check(cub::DeviceRadixSort::SortPairs(nullptr,bytes,dr,sorted_keys,ids,incoming,E,0,32,stream),"size edge sort");
        auto temp=device->buffer<char>("sort_temp",bytes);
        check(cub::DeviceRadixSort::SortPairs(temp,bytes,dr,sorted_keys,ids,incoming,E,0,32,stream),"sort incoming edges");
        check(cub::DeviceRadixSort::SortPairs(temp,bytes,ds,sorted_keys,ids,outgoing,E,0,32,stream),"sort outgoing edges");
    }
    check(cudaStreamSynchronize(stream),"complete GPU geometry");
    auto& info=device->geometry_metadata;
    device->result(info,"geometry","float32",{ne,4});
    device->result(info,"spherical_harmonics","float32",{ne,16});
    device->result(info,"spherical_harmonics_gradient","float32",{ne,3,16});
    device->result(info,"intervals","int32",{ne});
    device->result(info,"spline_coordinate","float32",{ne});
    for(const auto& prefix:{"incoming","outgoing"}){
        device->result(info,std::string(prefix)+"_starts","int32",{std::string(prefix)=="incoming"?nn+1:size_t(M)+1});
        device->result(info,std::string(prefix)+"_edges","int32",{ne});
    }
    num_nodes=N;num_edges=E;has_geometry=true;
}

std::map<std::string,LSCCudaArray> MACELSCCuda::download_geometry() const
{
    if(not has_geometry)throw std::runtime_error("MACELSCCuda: compute geometry first");
    return device->download(device->geometry_metadata);
}

void MACELSCCuda::evaluate_splines(const std::string& name)
{
    device->active();has_spline_results=false;spline_name.clear();device->spline_metadata.clear();
    if(not has_geometry)throw std::runtime_error("MACELSCCuda: compute geometry first");
    if(name!="radial_0" and name!="radial_1" and name!="sources.0.radial" and name!="sources.1.radial")
        throw std::invalid_argument("MACELSCCuda: unknown radial spline "+name);
    const bool compressed=float32_mixed and name=="radial_1";
    const auto table=name+(compressed?".c01":".coefficients");
    const size_t outputs=device->metadata.at(table).shape[2],ne=num_edges,count=ne*outputs;
    // Diagnostic buffers are reused for one table at a time. Production
    // convolutions can evaluate the same helpers without materializing E x O.
    auto values=device->buffer<float>("values",count);
    auto derivatives=device->buffer<float>("radial_derivatives",count);
    if(count){
        const auto coeff=static_cast<const float*>(device->allocations.at(table).pointer);
        const auto half=compressed?static_cast<const __half2*>(device->allocations.at(name+".c23").pointer):nullptr;
        const auto grid=static_cast<const float*>(device->allocations.at(name+".grid").pointer);
        const auto intervals=static_cast<const int*>(device->workspace.at("intervals").pointer);
        const auto coordinate=static_cast<const float*>(device->workspace.at("spline_coordinate").pointer);
        const size_t blocks=(count+255)/256;
        if(blocks>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("MACELSCCuda: excessive spline output size");
        if(compressed)lsc_cuda::spline_kernel<true><<<blocks,256,0,device->stream>>>(intervals,coordinate,count,outputs,coeff,half,grid,values,derivatives);
        else lsc_cuda::spline_kernel<false><<<blocks,256,0,device->stream>>>(intervals,coordinate,count,outputs,coeff,nullptr,grid,values,derivatives);
        check(cudaGetLastError(),"launch radial spline kernel");
        check(cudaStreamSynchronize(device->stream),"complete radial spline evaluation");
    }
    device->result(device->spline_metadata,"values","float32",{ne,outputs});
    device->result(device->spline_metadata,"radial_derivatives","float32",{ne,outputs});
    has_spline_results=true;spline_name=name;
}

std::map<std::string,LSCCudaArray> MACELSCCuda::download_splines() const
{
    if(not has_spline_results)throw std::runtime_error("MACELSCCuda: evaluate splines first");
    return device->download(device->spline_metadata);
}

size_t MACELSCCuda::sgemm_calls() const{return device->sgemm_count;}
size_t MACELSCCuda::tensor_core_gemm_calls() const{return device->tensor_core_gemm_count;}

double MACELSCCuda::compute_local(std::span<const int> types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> vectors)
{
    compute_geometry(types,senders,receivers,vectors);
    return evaluate_local();
}

double MACELSCCuda::evaluate_local()
{
    compute_first_layer();compute_second_layer();return compute_readouts();
}
void MACELSCCuda::compute_first_layer()
{

    device->active();clear_charge_results();has_local_results=false;local_energy=0;device->local_stage=0;
    device->local_metadata.clear();device->local_layouts.clear();
    device->sgemm_count=0;device->tensor_core_gemm_count=0;batched_reverse_calls=0;
    if(not has_geometry)throw std::runtime_error("MACELSCCuda: compute geometry first");
    const int N=num_nodes,C=num_channels,M=device->sender_nodes;
    device->forward.reset(graph_layout());
    if(M>std::numeric_limits<int>::max()/7)
        throw std::invalid_argument("MACELSCCuda: excessive local linear row count");
    const size_t nn=N;const auto stream=device->stream;
    const bool mixed=float32_mixed and tensor_cores;
    auto initial=device->buffer<float>("initial_features",nn*C);
    auto energies=device->buffer<float>("layer_node_energies",2*nn);
    auto node_energies=device->buffer<double>("local_node_energies",nn);
    auto total=device->buffer<double>("local_energy_device",1);
    // Reserve the shared scratch buffers once at their maximum width. No
    // per-edge radial outputs are materialized by either interaction layer.
    device->buffer<float>("local_up_0",size_t(M)*C);
    device->buffer<float>("local_up_1",size_t(M)*4*C);
    auto message=device->buffer<float>("local_message",nn*40*C);
    auto residual=device->buffer<float>("local_residual",nn*4*C);
    for(int t=0;t<2;++t){
        device->buffer<float>("A"+std::to_string(t),nn*16*C);
        device->buffer<float>("M"+std::to_string(t),nn*4*C);
        device->buffer<float>("H"+std::to_string(t+1),nn*4*C);
    }
    if(N){

        const auto types=device->working<int>("node_types");
        lsc_cuda::initial_features<<<(nn*C+255)/256,256,0,stream>>>(N,types,device->parameter<float>("initial_embedding"),initial);
        check(cudaGetLastError(),"launch initial local features");
        const auto starts=device->working<int>("incoming_starts");
        const auto order=device->working<int>("incoming_edges");
        const auto senders=device->working<int>("senders");
        const auto Y=device->working<float>("spherical_harmonics");
        const auto intervals=device->working<int>("intervals");
        const auto coordinate=device->working<float>("spline_coordinate");
        if(M!=N){
            auto all=device->buffer<float>("initial_all",size_t(M)*C);
            lsc_cuda::initial_features<<<(size_t(M)*C+255)/256,256,0,stream>>>(M,device->working<int>("node_types_all"),device->parameter<float>("initial_embedding"),all);
            check(cudaGetLastError(),"initialize ghost species embeddings");
        }
        evaluate_local_layer(0);
    }
    device->local_stage=1;publish_forward_features(1);
}
void MACELSCCuda::evaluate_local_layer(int t)
{
    const int N=num_nodes,C=num_channels;const size_t nn=N;const auto stream=device->stream;
    const bool mixed=float32_mixed && tensor_cores;
    const auto types=device->working<int>("node_types"),starts=device->working<int>("incoming_starts"),order=device->working<int>("incoming_edges"),senders=device->working<int>("senders"),intervals=device->working<int>("intervals");
    const auto Y=device->working<float>("spherical_harmonics"),coordinate=device->working<float>("spline_coordinate");
    auto message=device->working<float>("local_message"),residual=device->working<float>("local_residual"),energies=device->working<float>("layer_node_energies");
    const float* input=device->working<float>(t==0?"initial_features":"H1");
    const int input_nodes=device->sender_nodes;const bool remote=input_nodes!=N;
    const float* lifted_input=remote?device->working<float>(t==0?"initial_all":"H1_halo"):input;

            const auto suffix="_"+std::to_string(t);
            auto up=device->working<float>("local_up"+suffix);
            auto A=device->working<float>("A"+std::to_string(t));
            auto M=device->working<float>("M"+std::to_string(t));
            auto H=device->working<float>("H"+std::to_string(t+1));
            device->linear("linear_up"+suffix,input_nodes,lifted_input,up,mixed);
            const auto cg=device->parameter<float>("conv_tp"+suffix+".coefficients");
            if(remote){
            if(t==0){
                lsc_cuda::local_convolution<0,false,true><<<N,C,0,stream>>>(N,starts,order,senders,Y,intervals,coordinate,
                    device->parameter<float>("radial_0.coefficients"),nullptr,cg,up,message,input_nodes);
            }else if(float32_mixed){
                lsc_cuda::local_convolution<1,true,true><<<N,C,0,stream>>>(N,starts,order,senders,Y,intervals,coordinate,
                    device->parameter<float>("radial_1.c01"),device->parameter<__half2>("radial_1.c23"),cg,up,message,input_nodes);
            }else{
                lsc_cuda::local_convolution<1,false,true><<<N,C,0,stream>>>(N,starts,order,senders,Y,intervals,coordinate,
                    device->parameter<float>("radial_1.coefficients"),nullptr,cg,up,message,input_nodes);
            }
            }else{
            if(t==0){
                lsc_cuda::local_convolution<0,false><<<N,C,0,stream>>>(N,starts,order,senders,Y,intervals,coordinate,
                    device->parameter<float>("radial_0.coefficients"),nullptr,cg,up,message);
            }else if(float32_mixed){
                lsc_cuda::local_convolution<1,true><<<N,C,0,stream>>>(N,starts,order,senders,Y,intervals,coordinate,
                    device->parameter<float>("radial_1.c01"),device->parameter<__half2>("radial_1.c23"),cg,up,message);
            }else{
                lsc_cuda::local_convolution<1,false><<<N,C,0,stream>>>(N,starts,order,senders,Y,intervals,coordinate,
                    device->parameter<float>("radial_1.coefficients"),nullptr,cg,up,message);
            }
            }
            check(cudaGetLastError(),"launch fused local convolution");
            // linear_t includes the exported average-neighbor normalization.
            device->linear("linear"+suffix,N,message,A,mixed);
            lsc_cuda::local_contraction<<<(nn*C+127)/128,128,0,stream>>>(N,C,types,
                device->parameter<float>("M"+std::to_string(t)+"_weights"),A,M);
            check(cudaGetLastError(),"launch species-dependent local contraction");
            device->linear("product_linear"+suffix,N,M,H,mixed);
            // LSC residuals depend on both the current features and species.
            // Batch on the GPU and select the applicable rows without a host
            // gather or transferring learned features off the device.
            for(int e=0;e<num_elements;++e){
                device->linear("skip"+suffix+"."+std::to_string(e),N,input,residual,mixed);
                lsc_cuda::add_species_residual<<<(nn*4*C+255)/256,256,0,stream>>>(N,types,e,residual,H);
                check(cudaGetLastError(),"add species-dependent local residual");
            }
            device->linear("readout"+suffix,N,H,energies+nn*t,mixed);

}
void MACELSCCuda::compute_second_layer()
{
    device->active();if(device->local_stage!=1)throw std::runtime_error("MACELSCCuda: compute first layer before second layer");
    complete_forward_exchange(1);
    if(num_nodes)evaluate_local_layer(1);device->local_stage=2;publish_forward_features(2);
}
double MACELSCCuda::compute_readouts()
{
    device->active();if(device->local_stage!=2)throw std::runtime_error("MACELSCCuda: compute second layer before readouts");
    complete_forward_exchange(2);
    const int N=num_nodes,C=num_channels;const size_t nn=N;const auto stream=device->stream;
    const auto types=device->working<int>("node_types");const auto energies=device->working<float>("layer_node_energies");
    auto node_energies=device->working<double>("local_node_energies"),total=device->working<double>("local_energy_device");
    if(N){

        lsc_cuda::local_energies<<<(nn+255)/256,256,0,stream>>>(N,types,device->parameter<double>("atomic_energies"),energies,node_energies);
        check(cudaGetLastError(),"assemble local node energies");
        size_t bytes=0;
        check(cub::DeviceReduce::Sum(nullptr,bytes,node_energies,total,N,stream),"size local energy reduction");
        auto temp=device->buffer<char>("local_reduce_temp",bytes);
        check(cub::DeviceReduce::Sum(temp,bytes,node_energies,total,N,stream),"reduce local energy");
    }else{
        check(cudaMemsetAsync(total,0,sizeof(double),stream),"clear empty local energy");
    }
    double energy=0;
    check(cudaMemcpyAsync(&energy,total,sizeof(double),cudaMemcpyDeviceToHost,stream),"read local energy");
    check(cudaStreamSynchronize(stream),"complete local GPU evaluation");
    if((std::bit_cast<std::uint64_t>(energy)&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))
        throw std::invalid_argument("MACELSCCuda: nonfinite local prediction");
    auto feature=[&](const std::string& name,std::vector<size_t> multiplicities){
        size_t width=0;for(size_t l=0;l<multiplicities.size();++l)width+=(2*l+1)*multiplicities[l];
        device->result(device->local_metadata,name,"float32",{nn,width});
        device->local_layouts[name]=std::move(multiplicities);
    };
    feature("initial_features",{size_t(C)});
    for(int t=0;t<2;++t){
        feature("A"+std::to_string(t),{size_t(C),size_t(C),size_t(C),size_t(C)});
        feature("M"+std::to_string(t),{size_t(C),size_t(C)});
        feature("H"+std::to_string(t+1),{size_t(C),size_t(C)});
    }
    device->result(device->local_metadata,"layer_node_energies","float32",{2,nn});
    device->result(device->local_metadata,"local_node_energies","float64",{nn});
    local_energy=energy;has_local_results=true;device->local_stage=3;return energy;
}

std::map<std::string,LSCCudaArray> MACELSCCuda::download_local(bool diagnostics) const
{
    if(not has_local_results)throw std::runtime_error("MACELSCCuda: evaluate local interactions first");
    if(not diagnostics){
        std::map<std::string,LSCCudaArray> info;
        for(const auto& name:{"local_node_energies","layer_node_energies"})info.emplace(name,device->local_metadata.at(name));
        return device->download(info);
    }
    auto out=device->download(device->local_metadata);
    // This is a diagnostic adapter only. All interaction calculations above
    // retain Polar's GPU layout and run without intermediate host read-back.
    for(const auto& [name,multiplicities]:device->local_layouts){
        auto& array=out.at(name);
        auto raw=std::move(std::get<std::vector<float>>(array.values));
        std::vector<float> canonical(raw.size());
        const size_t width=array.shape[1];size_t offset=0;
        for(size_t l=0;l<multiplicities.size();++l){
            const size_t block=(2*l+1)*multiplicities[l];
            for(int n=0;n<num_nodes;++n)
                std::copy_n(raw.data()+size_t(num_nodes)*offset+size_t(n)*block,block,
                            canonical.data()+size_t(n)*width+offset);
            offset+=block;
        }
        array.values=std::move(canonical);
    }
    return out;
}

void MACELSCCuda::clear_charge_results()
{
    device->reverse_exchange.reset();
    device->charge_exchange.reset();device->charge_legacy=false;
    clear_force_results();has_energy_results=false;energy=electrostatic_energy=field_energy=0;
    has_charge_results=false;dipole={};formal_charge_dipole={};transfer_dipole={};atomic_dipole_sum={};
    source_sgemm_calls=0;source_tensor_core_gemm_calls=0;device->charge_metadata.clear();
}

std::array<double,3> MACELSCCuda::compute_charges(std::span<const int> types,std::span<const int> senders,
    std::span<const int> receivers,std::span<const double> vectors,std::span<const double> positions)
{
    compute_local(types,senders,receivers,vectors);
    return evaluate_charges(positions);
}

std::array<double,3> MACELSCCuda::evaluate_charges(std::span<const double> positions)
{
    device->active();clear_charge_results();
    if(not has_local_results)throw std::runtime_error("MACELSCCuda: evaluate local interactions first");
    if(graph_layout().uses_remote_ghosts())throw std::runtime_error("MACELSCCuda: remote charge accumulation requires the staged MPI source API; reverse communication is not implemented yet");
    const int N=num_nodes,E=num_edges,C=num_channels;
    const size_t nn=N,ne=E,stride=std::max(nn,ne);
    if(positions.size()!=3*nn)throw std::invalid_argument("MACELSCCuda: expected positions[N,3]");
    const auto stream=device->stream;const bool mixed=float32_mixed and tensor_cores;
    auto coords=device->buffer<double>("positions",3*nn);
    if(nn)check(cudaMemcpyAsync(coords,positions.data(),positions.size_bytes(),cudaMemcpyHostToDevice,stream),"upload source positions");
    auto error=device->buffer<int>("source_error",1);
    check(cudaMemsetAsync(error,0,sizeof(int),stream),"clear source status");
    if(nn)lsc_cuda::validate_source_positions<<<(3*nn+255)/256,256,0,stream>>>(3*nn,coords,error);
    check(cudaGetLastError(),"validate source positions");
    int status=0;check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read positions status");
    check(cudaStreamSynchronize(stream),"complete positions validation");
    if(status&1)throw std::invalid_argument("MACELSCCuda: positions must be finite");
    auto features=device->buffer<float>("source_features",2*nn*4*C);
    auto layer_flux=device->buffer<float>("layer_edge_fluxes",2*ne);
    auto layer_density=device->buffer<double>("layer_density_coefficients",2*nn*4);
    auto density=device->buffer<double>("density_coefficients",nn*4);
    auto charges=device->buffer<double>("partial_charges",nn);
    auto dipoles=device->buffer<double>("partial_dipoles",nn*3);
    auto flux=device->buffer<double>("edge_fluxes",ne);
    auto conditioning=device->buffer<float>("source_conditioning",nn*C);
    auto modulation_base=device->buffer<float>("source_modulation",4*nn*C);
    auto sender_base=device->buffer<float>("source_sender",2*nn*4*C);
    auto receiver_base=device->buffer<float>("source_receiver",2*nn*4*C);
    auto multipoles=device->buffer<float>("source_multipoles",nn*4);
    auto terms=device->buffer<double>("source_dipole_terms",9*stride);
    auto formal=device->buffer<double>("formal_charge_dipole",3);
    auto atomic=device->buffer<double>("atomic_dipole_sum",3);
    auto transfer=device->buffer<double>("transfer_dipole",3);
    auto total=device->buffer<double>("dipole",3);
    if(stride)check(cudaMemsetAsync(terms,0,9*stride*sizeof(double),stream),"clear source reduction terms");
    const size_t initial_sgemm=device->sgemm_count,initial_tensor=device->tensor_core_gemm_count;
    if(N){
        const auto types=device->working<int>("node_types");
        for(int t=0;t<2;++t){
            const auto prefix="sources."+std::to_string(t);
            auto sender=sender_base+size_t(t)*nn*4*C,receiver=receiver_base+size_t(t)*nn*4*C;
            auto modulation=modulation_base+size_t(t)*2*nn*C;
            auto output=features+size_t(t)*nn*4*C;
            lsc_cuda::initial_features<<<(nn*C+255)/256,256,0,stream>>>(N,types,
                device->parameter<float>(prefix+".conditioning_embedding"),conditioning);
            check(cudaGetLastError(),"gather source conditioning");
            for(int l=0;l<2;++l)
                device->gemm(prefix+".oxidation."+std::to_string(l),N,C,C,conditioning,modulation+size_t(l)*nn*C,mixed);
            lsc_cuda::source_modulate<<<(nn*C+255)/256,256,0,stream>>>(N,
                device->working<float>("H"+std::to_string(t+1)),modulation,output);
            check(cudaGetLastError(),"modulate source features");
            device->linear(prefix+".linear1",N,output,sender,mixed);
            device->linear(prefix+".linear2",N,output,receiver,mixed);
            if(E){
                lsc_cuda::source_flux<<<E,C,0,stream>>>(N,device->working<int>("senders"),device->working<int>("receivers"),
                    device->working<float>("spherical_harmonics"),device->working<int>("intervals"),device->working<float>("spline_coordinate"),
                    device->parameter<float>(prefix+".radial.coefficients"),device->parameter<int>(prefix+".conv_tp.spec"),
                    device->parameter<float>(prefix+".conv_tp.coefficients"),device->metadata.at(prefix+".conv_tp.spec").shape[0],
                    sender,receiver,device->parameter<float>(prefix+".transfer_readout.weights.0"),
                    device->parameter<float>(prefix+".transfer_readout.bias"),device->parameter<float>(prefix+".transfer_scale"),
                    (ne?layer_flux+size_t(t)*ne:layer_flux),error);
                check(cudaGetLastError(),"evaluate fused directed source transfers");
            }
            device->linear(prefix+".multipoles",N,output,multipoles,mixed);
            lsc_cuda::source_density<<<(nn+255)/256,256,0,stream>>>(N,
                device->working<int>("incoming_starts"),device->working<int>("incoming_edges"),
                device->working<int>("outgoing_starts"),device->working<int>("outgoing_edges"),
                (ne?layer_flux+size_t(t)*ne:layer_flux),multipoles,layer_density+size_t(t)*nn*4,error);
            check(cudaGetLastError(),"accumulate conservative source density");
        }
        lsc_cuda::source_nodes<<<(nn+255)/256,256,0,stream>>>(N,types,device->parameter<float>("formal_charges"),
            coords,layer_density,density,charges,dipoles,stride,terms,error);
        check(cudaGetLastError(),"assemble source node outputs");
    }
    if(E){
        lsc_cuda::source_edges<<<(ne+255)/256,256,0,stream>>>(E,layer_flux,device->working<double>("edge_vectors"),flux,stride,terms);
        check(cudaGetLastError(),"assemble transfer polarization");
    }
    if(stride){
        size_t bytes=0;
        check(cub::DeviceReduce::Sum(nullptr,bytes,terms,formal,int(stride),stream),"size dipole reduction");
        auto temp=device->buffer<char>("source_reduce_temp",bytes);
        for(int d=0;d<9;++d){
            double* output=d<3?formal+d:(d<6?atomic+d-3:transfer+d-6);
            check(cub::DeviceReduce::Sum(temp,bytes,terms+size_t(d)*stride,output,int(stride),stream),"reduce dipole component");
        }
    }else{
        for(auto ptr:{formal,atomic,transfer})check(cudaMemsetAsync(ptr,0,3*sizeof(double),stream),"clear empty dipole component");
    }
    lsc_cuda::source_dipole<<<1,32,0,stream>>>(formal,atomic,transfer,total,error);
    check(cudaGetLastError(),"assemble total LSC polarization");
    std::array<double,3> p{},f{},a{},e{};
    for(auto [host,gpu]:{std::pair(p.data(),total),std::pair(f.data(),formal),std::pair(a.data(),atomic),std::pair(e.data(),transfer)})
        check(cudaMemcpyAsync(host,gpu,3*sizeof(double),cudaMemcpyDeviceToHost,stream),"read dipole component");
    check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read source status");
    check(cudaStreamSynchronize(stream),"complete GPU charge sources");
    if(status&2)throw std::invalid_argument("MACELSCCuda: nonfinite charge/dipole prediction");
    auto& info=device->charge_metadata;
    device->result(info,"source_features","float32",{2,nn,size_t(4*C)});
    device->result(info,"layer_edge_fluxes","float32",{2,ne});
    device->result(info,"layer_density_coefficients","float64",{2,nn,4});
    device->result(info,"density_coefficients","float64",{nn,4});
    device->result(info,"partial_charges","float64",{nn});
    device->result(info,"partial_dipoles","float64",{nn,3});
    device->result(info,"edge_fluxes","float64",{ne});
    for(const auto& name:{"dipole","formal_charge_dipole","atomic_dipole_sum","transfer_dipole"})
        device->result(info,name,"float64",{3});
    dipole=p;formal_charge_dipole=f;atomic_dipole_sum=a;transfer_dipole=e;
    source_sgemm_calls=device->sgemm_count-initial_sgemm;
    source_tensor_core_gemm_calls=device->tensor_core_gemm_count-initial_tensor;
    has_charge_results=true;return dipole;
}

std::map<std::string,LSCCudaArray> MACELSCCuda::download_charges(bool diagnostics) const
{
    if(not has_charge_results)throw std::runtime_error("MACELSCCuda: evaluate charge sources first");
    if(not diagnostics){
        auto info=device->charge_metadata;
        for(const auto& name:{"source_features","layer_edge_fluxes","layer_density_coefficients"})info.erase(name);
        return device->download(info);
    }
    auto out=device->download(device->charge_metadata);
    auto& array=out.at("source_features");auto raw=std::move(std::get<std::vector<float>>(array.values));
    std::vector<float> canonical(raw.size());const size_t N=num_nodes,C=num_channels;
    for(int t=0;t<2;++t)for(int l=0;l<2;++l){
        const size_t offset=l==0?0:C,block=(2*l+1)*C;
        for(size_t n=0;n<N;++n)
            std::copy_n(raw.data()+size_t(t)*N*4*C+N*offset+n*block,block,
                        canonical.data()+(size_t(t)*N+n)*4*C+offset);
    }
    array.values=std::move(canonical);return out;
}

void MACELSCCuda::set_electrostatics_hook(LSCCudaElectrostaticsHook hook)
{
    device->active();
    check(cudaStreamSynchronize(device->stream), "finish previous electrostatics backend");
    device->electrostatics_hook = std::move(hook);
    device->energy_uses_hook = false;
    has_energy_results = false; energy = electrostatic_energy = field_energy = 0;
    clear_force_results();
}

bool MACELSCCuda::has_electrostatics_hook() const
{
    return bool(device->electrostatics_hook);
}

void MACELSCCuda::set_electrostatics_host_hook(LSCElectrostaticsHook hook)
{
    if (!hook) { set_electrostatics_hook({}); return; }
    set_electrostatics_hook([hook = std::move(hook)](const LSCCudaElectrostaticsInput& input,
                                                   const LSCCudaElectrostaticsOutput& output) {
        std::vector<double> positions(input.num_nodes*3), density(input.num_nodes*4);
        if (input.num_nodes) {
            check(cudaMemcpyAsync(positions.data(),input.positions,positions.size()*sizeof(double),cudaMemcpyDeviceToHost,input.stream),"read custom-solver positions");
            check(cudaMemcpyAsync(density.data(),input.density,density.size()*sizeof(double),cudaMemcpyDeviceToHost,input.stream),"read custom-solver density");
        }
        check(cudaStreamSynchronize(input.stream),"complete host electrostatics inputs");
        const auto result = hook({positions,density,input.context});
        validate_lsc_electrostatics_result(result,input.num_nodes);
        check(cudaMemcpyAsync(output.energy,&result.energy,sizeof(double),cudaMemcpyHostToDevice,input.stream),"upload custom electrostatic energy");
        if (input.num_nodes) {
            check(cudaMemcpyAsync(output.density_gradient,result.density_gradient.data(),result.density_gradient.size()*sizeof(double),cudaMemcpyHostToDevice,input.stream),"upload custom density gradient");
            check(cudaMemcpyAsync(output.position_gradient,result.position_gradient.data(),result.position_gradient.size()*sizeof(double),cudaMemcpyHostToDevice,input.stream),"upload custom position gradient");
        }
        check(cudaMemcpyAsync(output.strain_derivative,result.strain_derivative.data(),9*sizeof(double),cudaMemcpyHostToDevice,input.stream),"upload custom strain derivative");
        // Host result storage must remain alive until its uploads finish.
        check(cudaStreamSynchronize(input.stream),"complete host electrostatics outputs");
    });
}

double MACELSCCuda::compute_energy(std::span<const int> types,std::span<const int> senders,std::span<const int> receivers,
    std::span<const double> vectors,std::span<const double> positions,const std::array<double,9>& cell,
    const std::array<bool,3>& pbc,const std::string& mode,const std::array<double,3>& field)
{
    compute_charges(types,senders,receivers,vectors,positions);
    return evaluate_energy(cell,pbc,mode,field);
}

double MACELSCCuda::evaluate_energy(const std::array<double,9>& cell,const std::array<bool,3>& pbc,
    const std::string& requested_mode,const std::array<double,3>& field)
{
    if(device->charge_exchange.active())throw std::runtime_error("MACELSCCuda: distributed electrostatic energy and reverse communication are not implemented yet");
    device->active();clear_force_results();has_energy_results=false;energy=electrostatic_energy=field_energy=0;
    if(not has_charge_results)throw std::runtime_error("MACELSCCuda: evaluate charge sources first");
    for(double x:cell)if(not lsc_cuda::energy_finite(x))throw std::invalid_argument("MACELSCCuda: cell must be finite");
    for(double x:field)if(not lsc_cuda::energy_finite(x))throw std::invalid_argument("MACELSCCuda: external_field must be finite");
    const int periodic=std::count(pbc.begin(),pbc.end(),true);
    std::string mode=requested_mode.empty()?"auto":requested_mode;
    if(mode=="auto")mode=periodic==0?"realspace":"mixed_periodic";
    if(mode!="realspace" and mode!="pbc" and mode!="mixed_periodic")
        throw std::invalid_argument("MACELSCCuda: supported modes: realspace, pbc, mixed_periodic, auto");
    if(mode!="realspace" and periodic!=3)throw std::invalid_argument("MACELSCCuda: reciprocal energy requires three periodic directions");
    const int N=num_nodes;const size_t nn=N;const auto stream=device->stream;
    auto total=device->buffer<double>("electrostatic_total",1);
    const auto positions=device->working<double>("positions"),density=device->working<double>("density_coefficients");
    int K=0;size_t count=nn;
    const bool hooked=has_electrostatics_hook();
    if(hooked){
        device->energy_volume=lsc_cuda::cell_volume(cell);
        if(mode!="realspace" and (not lsc_cuda::energy_finite(device->energy_volume) or device->energy_volume<=0))
            throw std::invalid_argument("MACELSCCuda: reciprocal energy requires a nonsingular cell");
        LSCElectrostaticsContext context{cell,pbc,mode,{}};
        auto& p=context.parameters;const auto& ep=device->electrostatic;
        p.field_constant=ep.field_constant;p.kspace_cutoff=device->electrostatic_cutoff;p.sigma2=ep.sigma2;
        std::copy_n(ep.a,2,p.a.begin());std::copy_n(ep.self_recip,4,p.reciprocal_self.begin());
        std::copy_n(ep.self_real,4,p.realspace_self.begin());std::copy_n(ep.displacements,9,p.displacements.begin());
        p.sigma=ep.sigma;p.offset=ep.offset;p.epsilon=ep.epsilon;p.include_self=ep.include_self;
        auto dr=device->buffer<double>("hook_density_gradient",nn*4);
        auto dx=device->buffer<double>("hook_position_gradient",nn*3);
        auto ds=device->buffer<double>("hook_strain_derivative",9);
        check(cudaMemsetAsync(total,0,sizeof(double),stream),"clear hook energy");
        if(N){check(cudaMemsetAsync(dr,0,nn*4*sizeof(double),stream),"clear hook density gradient");
              check(cudaMemsetAsync(dx,0,nn*3*sizeof(double),stream),"clear hook position gradient");}
        check(cudaMemsetAsync(ds,0,9*sizeof(double),stream),"clear hook strain derivative");
        device->electrostatics_hook({nn,positions,density,std::move(context),stream},{total,dr,dx,ds});
    }else if(mode!="realspace"){
        auto reciprocal=lsc_cuda::reciprocal_geometry(cell,device->electrostatic_cutoff);
        K=reciprocal.vectors.size()/3;device->energy_volume=reciprocal.volume;
        auto vectors=device->buffer<double>("energy_k_vectors",reciprocal.vectors.size());
        if(K)check(cudaMemcpyAsync(vectors,reciprocal.vectors.data(),reciprocal.vectors.size()*sizeof(double),cudaMemcpyHostToDevice,stream),"upload reciprocal vectors");
        auto structure=device->buffer<double2>("energy_structure",K);
        count=size_t(K)+nn;auto terms=device->buffer<double>("electrostatic_terms",count);
        if(K)lsc_cuda::electrostatic_reciprocal<<<K,128,0,stream>>>(N,device->electrostatic,reciprocal.volume,vectors,positions,density,structure,terms);
        if(N)lsc_cuda::electrostatic_self<<<(nn+255)/256,256,0,stream>>>(N,K,device->electrostatic,density,terms);
    }else{
        device->energy_volume=lsc_cuda::cell_volume(cell);
        auto terms=device->buffer<double>("electrostatic_terms",nn);
        if(N)lsc_cuda::electrostatic_real<<<N,128,0,stream>>>(N,device->electrostatic,positions,density,terms);
    }
    check(cudaGetLastError(),"launch electrostatic energy");
    if(not hooked and count){
        size_t bytes=0;auto terms=device->working<double>("electrostatic_terms");
        if(count>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("MACELSCCuda: excessive electrostatic reduction size");
        check(cub::DeviceReduce::Sum(nullptr,bytes,terms,total,int(count),stream),"size electrostatic reduction");
        auto temp=device->buffer<char>("electrostatic_reduce_temp",bytes);
        check(cub::DeviceReduce::Sum(temp,bytes,terms,total,int(count),stream),"reduce electrostatic energy");
    }else if(not hooked) check(cudaMemsetAsync(total,0,sizeof(double),stream),"clear empty electrostatic energy");
    auto components=device->buffer<double>("energy_components",4);
    lsc_cuda::electrostatic_total<<<1,32,0,stream>>>(device->working<double>("local_energy_device"),total,device->working<double>("dipole"),field[0],field[1],field[2],components);
    check(cudaGetLastError(),"assemble full LSC energy");
    std::array<double,4> values{};
    check(cudaMemcpyAsync(values.data(),components,4*sizeof(double),cudaMemcpyDeviceToHost,stream),"read energy components");
    check(cudaStreamSynchronize(stream),"complete electrostatic energy");
    for(double x:values)if(not lsc_cuda::energy_finite(x))throw std::invalid_argument("MACELSCCuda: nonfinite energy prediction");
    electrostatic_energy=values[1];field_energy=values[2];energy=values[3];
    device->energy_uses_hook=hooked;
    device->energy_mode=mode;device->energy_k_count=K;device->energy_cell=cell;device->energy_field=field;
    has_energy_results=true;return energy;
}

void MACELSCCuda::clear_force_results()
{
    device->reverse_exchange.reset();
    device->reverse_stage=0;has_force_results=has_stress_results=false;stress={};virial={};strain_derivative={};device->force_metadata.clear();
}
double MACELSCCuda::compute_energy_forces(std::span<const int> types,std::span<const int> senders,std::span<const int> receivers,
    std::span<const double> vectors,std::span<const double> positions,const std::array<double,9>& cell,
    const std::array<bool,3>& pbc,const std::string& mode,const std::array<double,3>& field,bool compute_stress)
{
    compute_energy(types,senders,receivers,vectors,positions,cell,pbc,mode,field);evaluate_forces(compute_stress);return energy;
}
void MACELSCCuda::evaluate_forces(bool compute_stress)
{
    if(compute_stress && (!lsc_cuda::energy_finite(device->energy_volume) || device->energy_volume<=0))throw std::invalid_argument("MACELSCCuda: stress requires a nonsingular cell");
    reverse_charges();reverse_second_layer();reverse_first_layer();finalize_forces(compute_stress);
}
void MACELSCCuda::reverse_charges()
{

    device->active();LSCReverseExchange::check(!device->charge_exchange.active(),"use distributed reverse after staged MPI sources");clear_force_results();
    if(not has_energy_results)throw std::runtime_error("MACELSCCuda: evaluate energy first");
    const int N=num_nodes,E=num_edges,C=num_channels,K=device->energy_k_count;
    const bool hooked=device->energy_uses_hook;
    const size_t nn=N,ne=E,stride=nn+ne+K+size_t(hooked);const auto stream=device->stream;
    const bool real=device->energy_mode=="realspace",mixed=float32_mixed and tensor_cores;
    const double volume=device->energy_volume;
    const auto field=device->energy_field;
    auto density_adj=device->buffer<double>("density_energy_gradient",nn*4);
    auto position_adj=device->buffer<double>("position_gradient",nn*3);
    auto edge_adj=device->buffer<double>("edge_gradient",ne*3);
    auto forces=device->buffer<double>("forces",nn*3);
    auto position_forces=device->buffer<double>("position_forces",nn*3);
    auto edge_forces=device->buffer<double>("edge_forces",ne*3);
    auto terms=device->buffer<double>("force_strain_terms",9*stride);
    auto strain=device->buffer<double>("force_strain",9);
    auto error=device->buffer<int>("force_error",1);
    check(cudaMemsetAsync(error,0,sizeof(int),stream),"clear force status");
    if(stride)check(cudaMemsetAsync(terms,0,9*stride*sizeof(double),stream),"clear strain terms");
    const auto xyz=device->working<double>("positions"),rho=device->working<double>("density_coefficients");
    if(N){
        if(hooked){
            check(cudaMemcpyAsync(density_adj,device->working<double>("hook_density_gradient"),nn*4*sizeof(double),cudaMemcpyDeviceToDevice,stream),"seed custom density gradient");
            check(cudaMemcpyAsync(position_adj,device->working<double>("hook_position_gradient"),nn*3*sizeof(double),cudaMemcpyDeviceToDevice,stream),"seed custom position gradient");
        }else if(real)lsc_cuda::electrostatic_real_reverse<<<(nn+127)/128,128,0,stream>>>(N,device->electrostatic,xyz,rho,density_adj,position_adj);
        else lsc_cuda::electrostatic_reciprocal_reverse<<<(nn+127)/128,128,0,stream>>>(N,K,device->electrostatic,volume,
            device->working<double>("energy_k_vectors"),device->working<double2>("energy_structure"),xyz,rho,density_adj,position_adj);
        check(cudaGetLastError(),"differentiate electrostatic energy");
        lsc_cuda::reverse_field_nodes<<<(nn+255)/256,256,0,stream>>>(N,device->working<int>("node_types"),device->parameter<float>("formal_charges"),field[0],field[1],field[2],density_adj,position_adj);
    }
    if(E)lsc_cuda::reverse_field_edges<<<(ne+255)/256,256,0,stream>>>(E,device->working<double>("edge_fluxes"),field[0],field[1],field[2],edge_adj);
    check(cudaGetLastError(),"differentiate external field");
    auto h0=device->buffer<float>("H1_adjoint",nn*4*C),h1=device->buffer<float>("H2_adjoint",nn*4*C);
    if(N){
        for(auto ptr:{h0,h1})check(cudaMemsetAsync(ptr,0,nn*4*C*sizeof(float),stream),"clear hidden adjoints");
        auto ones=device->buffer<float>("force_ones",nn);
        lsc_cuda::reverse_fill<<<(nn+255)/256,256,0,stream>>>(nn,1.f,ones);
        device->linear_reverse("readout_0",N,ones,h0,mixed);device->linear_reverse("readout_1",N,ones,h1,mixed);
        auto sf_adj=device->buffer<float>("source_features_adjoint",nn*4*C);
        auto sender_adj=device->buffer<float>("source_sender_adjoint",nn*4*C),receiver_adj=device->buffer<float>("source_receiver_adjoint",nn*4*C);
        auto multipole_adj=device->buffer<float>("source_multipole_adjoint",nn*4);
        lsc_cuda::source_multipole_seed<<<(nn+255)/256,256,0,stream>>>(N,density_adj,multipole_adj);
        for(int t=0;t<2;++t){
            const auto prefix="sources."+std::to_string(t);
            for(auto ptr:{sf_adj,sender_adj,receiver_adj})check(cudaMemsetAsync(ptr,0,nn*4*C*sizeof(float),stream),"clear source adjoints");
            device->linear_reverse(prefix+".multipoles",N,multipole_adj,sf_adj,mixed);
            if(E)lsc_cuda::source_transfer_reverse<<<E,C,0,stream>>>(N,device->working<int>("senders"),device->working<int>("receivers"),device->working<float4>("geometry"),
                device->working<float>("spherical_harmonics"),device->working<float>("spherical_harmonics_gradient"),device->working<int>("intervals"),device->working<float>("spline_coordinate"),
                device->parameter<float>(prefix+".radial.coefficients"),device->parameter<float>(prefix+".radial.grid"),device->parameter<int>(prefix+".conv_tp.spec"),
                device->parameter<float>(prefix+".conv_tp.coefficients"),device->metadata.at(prefix+".conv_tp.spec").shape[0],device->parameter<float>(prefix+".transfer_readout.weights.0"),
                device->parameter<float>(prefix+".transfer_scale"),device->working<float>("source_sender")+size_t(t)*nn*4*C,device->working<float>("source_receiver")+size_t(t)*nn*4*C,
                density_adj,field[0],field[1],field[2],device->working<double>("edge_vectors"),sender_adj,receiver_adj,edge_adj);
            check(cudaGetLastError(),"reverse directed source transfers");
            device->linear_reverse(prefix+".linear1",N,sender_adj,sf_adj,mixed);
            device->linear_reverse(prefix+".linear2",N,receiver_adj,sf_adj,mixed);
            lsc_cuda::source_modulation_reverse<<<(nn*C+255)/256,256,0,stream>>>(N,sf_adj,device->working<float>("source_modulation")+size_t(t)*2*nn*C,t==0?h0:h1);
            check(cudaGetLastError(),"reverse fixed-charge source modulation");
        }
    }
    device->reverse_stage=1;
}
void MACELSCCuda::reverse_local_layer(int t)
{
    const int N=num_nodes,E=num_edges,C=num_channels;const size_t nn=N,ne=E;const auto stream=device->stream;
    const bool mixed=float32_mixed && tensor_cores;
    auto h0=device->working<float>("H1_adjoint"),h1=device->working<float>("H2_adjoint");
    auto density_adj=device->working<double>("density_energy_gradient"),edge_adj=device->working<double>("edge_gradient");
        auto m_adj=device->buffer<float>("local_M_adjoint",nn*4*C),a_adj=device->buffer<float>("local_A_adjoint",nn*16*C);
        auto message_adj=device->buffer<float>("local_message_adjoint",nn*40*C),up_adj=device->buffer<float>("local_up_adjoint",nn*4*C);
        auto mask=device->buffer<float>("local_masked_adjoint",nn*4*C);

            const auto suffix="_"+std::to_string(t);const float* hidden_adj=t==0?h0:h1;
            check(cudaMemsetAsync(m_adj,0,nn*4*C*sizeof(float),stream),"clear product adjoint");
            check(cudaMemsetAsync(a_adj,0,nn*16*C*sizeof(float),stream),"clear interaction adjoint");
            check(cudaMemsetAsync(message_adj,0,nn*40*C*sizeof(float),stream),"clear message adjoint");
            check(cudaMemsetAsync(up_adj,0,nn*4*C*sizeof(float),stream),"clear lifted feature adjoint");
            device->linear_reverse("product_linear"+suffix,N,hidden_adj,m_adj,mixed);
            if(t==1)for(int e=0;e<num_elements;++e){
                lsc_cuda::reverse_mask<<<(nn*4*C+255)/256,256,0,stream>>>(N,C,device->working<int>("node_types"),e,hidden_adj,mask);
                device->linear_reverse("skip_1."+std::to_string(e),N,mask,h0,mixed);
            }
            lsc_cuda::local_contraction_reverse<<<(nn*C+127)/128,128,0,stream>>>(N,C,device->working<int>("node_types"),device->parameter<float>("M"+std::to_string(t)+"_weights"),
                device->working<float>("A"+std::to_string(t)),m_adj,a_adj);
            check(cudaGetLastError(),"reverse species product contraction");
            device->linear_reverse("linear"+suffix,N,a_adj,message_adj,mixed);
            if(E){
                const auto senders=device->working<int>("senders"),receivers=device->working<int>("receivers");const auto geometry=device->working<float4>("geometry");
                const auto Y=device->working<float>("spherical_harmonics"),dY=device->working<float>("spherical_harmonics_gradient"),coordinate=device->working<float>("spline_coordinate");
                const auto intervals=device->working<int>("intervals");const auto cg=device->parameter<float>("conv_tp"+suffix+".coefficients"),grid=device->parameter<float>("radial"+suffix+".grid");
                const auto up=device->working<float>("local_up"+suffix);
                if(t==0)lsc_cuda::local_convolution_reverse<0,false><<<E,C,0,stream>>>(N,senders,receivers,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_0.coefficients"),nullptr,grid,cg,up,message_adj,up_adj,edge_adj);
                else if(use_batched_reverse){
                    if(float32_mixed)
                        lsc_cuda::local_convolution_reverse_batched<true,false><<<N,C,0,stream>>>(N,N,device->working<int>("incoming_starts"),device->working<int>("incoming_edges"),senders,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.c01"),device->parameter<__half2>("radial_1.c23"),grid,cg,up,message_adj,up_adj,edge_adj);
                    else
                        lsc_cuda::local_convolution_reverse_batched<false,false><<<N,C,0,stream>>>(N,N,device->working<int>("incoming_starts"),device->working<int>("incoming_edges"),senders,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.coefficients"),nullptr,grid,cg,up,message_adj,up_adj,edge_adj);
                    ++batched_reverse_calls;
                }
                else if(float32_mixed)lsc_cuda::local_convolution_reverse<1,true><<<E,C,0,stream>>>(N,senders,receivers,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.c01"),device->parameter<__half2>("radial_1.c23"),grid,cg,up,message_adj,up_adj,edge_adj);
                else lsc_cuda::local_convolution_reverse<1,false><<<E,C,0,stream>>>(N,senders,receivers,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.coefficients"),nullptr,grid,cg,up,message_adj,up_adj,edge_adj);
                check(cudaGetLastError(),"reverse fused local convolution");
            }
            // The initial species/formal-charge embedding is independent of
            // Cartesian geometry, so its final discarded adjoint is unnecessary.
            if(t==1)device->linear_reverse("linear_up_1",N,up_adj,h0,mixed);

}
void MACELSCCuda::reverse_second_layer()
{
    device->active();if(device->reverse_stage!=1)throw std::runtime_error("MACELSCCuda: reverse sources before second layer");
    if(num_nodes)reverse_local_layer(1);device->reverse_stage=2;
}
void MACELSCCuda::reverse_first_layer()
{
    device->active();if(device->reverse_stage!=2)throw std::runtime_error("MACELSCCuda: reverse second layer before first layer");
    if(num_nodes)reverse_local_layer(0);device->reverse_stage=3;
}
void MACELSCCuda::finalize_forces(bool compute_stress)
{
    device->active();if(device->reverse_stage!=3)throw std::runtime_error("MACELSCCuda: reverse first layer before force assembly");
    const int N=num_nodes,E=num_edges,C=num_channels,K=device->energy_k_count;const bool hooked=device->energy_uses_hook;
    const size_t nn=N,ne=E,stride=nn+ne+K+size_t(hooked);const auto stream=device->stream;
    const bool real=device->energy_mode=="realspace";const double volume=device->energy_volume;const auto field=device->energy_field;
    if(compute_stress && (!lsc_cuda::energy_finite(volume) || volume<=0))throw std::invalid_argument("MACELSCCuda: stress requires a nonsingular cell");
    const auto xyz=device->working<double>("positions"),rho=device->working<double>("density_coefficients");
    auto position_adj=device->working<double>("position_gradient"),edge_adj=device->working<double>("edge_gradient");
    auto forces=device->working<double>("forces"),position_forces=device->working<double>("position_forces"),edge_forces=device->working<double>("edge_forces");
    auto terms=device->working<double>("force_strain_terms"),strain=device->working<double>("force_strain");auto error=device->working<int>("force_error");
    if(N){
        lsc_cuda::reverse_force_nodes<<<(nn+255)/256,256,0,stream>>>(N,device->working<int>("node_types"),device->parameter<float>("formal_charges"),xyz,
            device->working<int>("incoming_starts"),device->working<int>("incoming_edges"),device->working<int>("outgoing_starts"),device->working<int>("outgoing_edges"),
            position_adj,edge_adj,real and not hooked,field[0],field[1],field[2],stride,terms,forces,position_forces,error);
    }
    if(E)lsc_cuda::reverse_force_edges<<<(ne+255)/256,256,0,stream>>>(N,E,device->working<double>("edge_vectors"),edge_adj,stride,terms,edge_forces);
    if(hooked)check(cudaMemcpy2DAsync(terms+(nn+ne),stride*sizeof(double),
        device->working<double>("hook_strain_derivative"),sizeof(double),sizeof(double),9,cudaMemcpyDeviceToDevice,stream),"seed full custom electrostatic strain");
    if(K)lsc_cuda::electrostatic_reciprocal_strain<<<K,128,0,stream>>>(N,device->electrostatic,volume,device->working<double>("energy_k_vectors"),device->working<double2>("energy_structure"),xyz,rho,stride,nn+ne,terms);
    check(cudaGetLastError(),"assemble forces and strain");
    if(stride){
        if(stride>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("MACELSCCuda: excessive strain reduction size");
        size_t bytes=0;check(cub::DeviceReduce::Sum(nullptr,bytes,terms,strain,int(stride),stream),"size strain reduction");
        auto temp=device->buffer<char>("force_reduce_temp",bytes);
        for(int d=0;d<9;++d)check(cub::DeviceReduce::Sum(temp,bytes,terms+size_t(d)*stride,strain+d,int(stride),stream),"reduce strain derivative");
    }else check(cudaMemsetAsync(strain,0,9*sizeof(double),stream),"clear empty strain");
    std::array<double,9> raw{};int status=0;
    check(cudaMemcpyAsync(raw.data(),strain,9*sizeof(double),cudaMemcpyDeviceToHost,stream),"read strain derivative");
    check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read force status");
    check(cudaStreamSynchronize(stream),"complete analytic GPU derivatives");
    if(status)throw std::invalid_argument("MACELSCCuda: nonfinite force prediction");
    for(double x:raw)if(not lsc_cuda::energy_finite(x))throw std::invalid_argument("MACELSCCuda: nonfinite strain derivative");
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)strain_derivative[3*a+b]=0.5*(raw[3*a+b]+raw[3*b+a]);
    constexpr int ase[6]={0,4,8,5,2,1},polar[6]={0,4,8,1,2,5};
    for(int k=0;k<6;++k){virial[k]=-strain_derivative[polar[k]];if(compute_stress)stress[k]=strain_derivative[ase[k]]/volume;}
    auto& info=device->force_metadata;
    device->result(info,"forces","float64",{nn,3});device->result(info,"position_forces","float64",{nn,3});
    device->result(info,"edge_forces","float64",{ne,3});device->result(info,"density_energy_gradient","float64",{nn,4});
    has_force_results=true;has_stress_results=compute_stress;device->reverse_stage=4;
}

std::map<std::string,LSCCudaArray> MACELSCCuda::download_forces(bool diagnostics) const
{
    if(not has_force_results)throw std::runtime_error("MACELSCCuda: evaluate forces first");
    if(diagnostics)return device->download(device->force_metadata);
    std::map<std::string,LSCCudaArray> info{{"forces",device->force_metadata.at("forces")}};return device->download(info);
}

void MACELSCCuda::prepare_graph(const LSCGraph& input)
{
    LSCGraph graph=input;
    has_geometry=false;has_local_results=false;device->graph.reset();device->local_stage=0;device->forward=LSCForwardState{};clear_charge_results();
    for(int type:graph.types())if(type>=num_elements)throw std::invalid_argument("MACELSCCuda: node_types species index is out of range");
    auto senders=graph.feature_senders();
    const int M=graph.uses_remote_ghosts()?graph.num_nodes():graph.num_owned();
    prepare_geometry(graph.owned_types(),senders,graph.receivers(),graph.edge_vectors(),M);
    if(M!=graph.num_owned()){
        auto ptr=device->buffer<int>("node_types_all",M);
        check(cudaMemcpyAsync(ptr,graph.types().data(),size_t(M)*sizeof(int),cudaMemcpyHostToDevice,device->stream),"upload ghost species");
        check(cudaStreamSynchronize(device->stream),"complete ghost species upload");
    }
    if(graph.num_ghosts()){
        auto owners=device->buffer<int>("forward_owners",graph.num_nodes());
        check(cudaMemcpyAsync(owners,graph.local_owners().data(),size_t(graph.num_nodes())*sizeof(int),cudaMemcpyHostToDevice,device->stream),"upload local ghost owners");
        check(cudaStreamSynchronize(device->stream),"complete local ghost owner upload");
    }
    device->graph=std::move(graph);device->forward.reset(*device->graph);
}
const LSCGraph& MACELSCCuda::graph_layout()const
{
    if(!device->graph)throw std::runtime_error("MACELSCCuda: prepare a graph first");return *device->graph;
}
std::string MACELSCCuda::evaluation_stage()const
{
    if(has_force_results)return "forces";
    if(device->reverse_stage)return device->reverse_stage==1?"reverse_sources":device->reverse_stage==2?"reverse_second_layer":"reverse_first_layer";
    if(has_energy_results)return "energy";if(has_charge_results)return "sources";if(has_local_results)return "readouts";
    if(device->local_stage)return device->local_stage==1?"first_layer":"second_layer";
    return has_geometry?"graph":"empty";
}
float* MACELSCCuda::layer_features_device(int layer)
{
    device->active();if((layer!=1 && layer!=2) || device->local_stage<layer)throw std::runtime_error("MACELSCCuda: requested layer features are not ready");
    return device->working<float>("H"+std::to_string(layer));
}
float* MACELSCCuda::layer_feature_gradients_device(int layer)
{
    device->active();if((layer!=1 && layer!=2) || device->reverse_stage<1)throw std::runtime_error("MACELSCCuda: reverse sources before accessing layer gradients");
    return device->working<float>("H"+std::to_string(layer)+"_adjoint");
}
CUstream_st* MACELSCCuda::cuda_stream()const{return device->stream;}
LSCCudaArray MACELSCCuda::download_layer_features(int layer,bool gradients)
{
    auto ptr=gradients?layer_feature_gradients_device(layer):layer_features_device(layer);
    const size_t N=num_nodes,C=num_channels;std::vector<float> raw(N*4*C),canonical(raw.size());
    if(!raw.empty())check(cudaMemcpyAsync(raw.data(),ptr,raw.size()*sizeof(float),cudaMemcpyDeviceToHost,device->stream),"read staged layer features");
    check(cudaStreamSynchronize(device->stream),"complete staged layer readback");
    for(int l=0;l<2;++l){const size_t offset=l==0?0:C,block=(2*l+1)*C;
        for(size_t i=0;i<N;++i)std::copy_n(raw.data()+N*offset+i*block,block,canonical.data()+i*4*C+offset);}
    LSCCudaArray out;out.dtype="float32";out.shape={N,4*C};out.values=std::move(canonical);return out;
}

void MACELSCCuda::publish_forward_features(int layer)
{
    const auto& graph=graph_layout();const int all=graph.num_nodes(),C=num_channels;
    if(graph.num_ghosts()){
        auto owners=device->working<int>("forward_owners");
        auto halo=device->buffer<float>("H"+std::to_string(layer)+"_halo",size_t(all)*4*C);
        if(all)lsc_cuda::publish_forward_halo<<<(size_t(all)*4*C+255)/256,256,0,device->stream>>>(num_nodes,all,C,owners,device->working<float>("H"+std::to_string(layer)),halo);
        check(cudaGetLastError(),"publish owned/ghost feature storage");
    }
    device->forward.publish(layer,graph);
}
float* MACELSCCuda::forward_features_device(int layer) const
{
    device->active();device->forward.check_available(layer);
    return device->working<float>("H"+std::to_string(layer)+(graph_layout().num_ghosts()?"_halo":""));
}
void MACELSCCuda::pack_forward_features(int layer,std::span<const int> slots,std::span<double> values) const
{
    device->active();device->forward.check_pack(layer,slots,graph_layout());
    if(values.size()!=slots.size()*4*num_channels)throw std::invalid_argument("MACELSCCuda: incorrect feature send buffer size");
    if(slots.empty())return;
    auto indices=device->buffer<int>("forward_slots",slots.size());auto payload=device->buffer<double>("forward_payload",values.size());
    check(cudaMemcpyAsync(indices,slots.data(),slots.size_bytes(),cudaMemcpyHostToDevice,device->stream),"upload forward send slots");
    lsc_cuda::pack_forward_halo<<<(values.size()+255)/256,256,0,device->stream>>>(slots.size(),graph_layout().num_nodes(),num_channels,indices,forward_features_device(layer),payload);
    check(cudaGetLastError(),"pack ghost feature payload");
    check(cudaMemcpyAsync(values.data(),payload,values.size_bytes(),cudaMemcpyDeviceToHost,device->stream),"read forward feature payload");
    check(cudaStreamSynchronize(device->stream),"complete feature send staging");
}
void MACELSCCuda::unpack_forward_features(int layer,int first,std::span<const double> values)
{
    device->active();if(device->local_stage!=layer)throw std::invalid_argument("MACELSCCuda: receive features before advancing to the next stage");
    device->forward.check_receive(layer,first,values,graph_layout(),4*num_channels,std::numeric_limits<float>::max());
    if(!values.empty()){
        auto payload=device->buffer<double>("forward_payload",values.size());
        check(cudaMemcpyAsync(payload,values.data(),values.size_bytes(),cudaMemcpyHostToDevice,device->stream),"upload received feature payload");
        lsc_cuda::unpack_forward_halo<<<(values.size()+255)/256,256,0,device->stream>>>(values.size()/(4*num_channels),first,graph_layout().num_nodes(),num_channels,payload,forward_features_device(layer));
        check(cudaGetLastError(),"unpack ghost features");
        check(cudaStreamSynchronize(device->stream),"complete received feature staging");
    }
    device->forward.received(layer,first,values.size()/(4*num_channels));
}
void MACELSCCuda::complete_forward_exchange(int layer)
{
    device->forward.complete(layer,graph_layout());
}
bool MACELSCCuda::forward_exchange_complete(int layer) const{return device->forward.complete(layer);}
LSCCudaArray MACELSCCuda::download_forward_features(int layer) const
{
    auto ptr=forward_features_device(layer);const size_t N=graph_layout().num_nodes(),C=num_channels;
    std::vector<float> raw(N*4*C),canonical(raw.size());
    if(!raw.empty())check(cudaMemcpyAsync(raw.data(),ptr,raw.size()*sizeof(float),cudaMemcpyDeviceToHost,device->stream),"read halo features");
    check(cudaStreamSynchronize(device->stream),"complete halo readback");
    for(int l=0;l<2;++l){const size_t offset=l==0?0:C,block=(2*l+1)*C;
        for(size_t i=0;i<N;++i)std::copy_n(raw.data()+N*offset+i*block,block,canonical.data()+i*4*C+offset);}
    LSCCudaArray out;out.dtype="float32";out.shape={N,4*C};out.values=std::move(canonical);return out;
}

void MACELSCCuda::compute_partial_charges(std::span<const double> positions)
{
    device->active();if(!has_local_results)throw std::runtime_error("MACELSCCuda: compute readouts before partial charges");
    const auto& graph=graph_layout();const int N=num_nodes,M=graph.num_nodes(),E=num_edges,C=num_channels;
    if(positions.size()!=size_t(N)*3)throw std::invalid_argument("MACELSCCuda: expected owned positions[N_owned,3]");
    LSCChargeExchange::finite(positions);clear_charge_results();
    const auto stream=device->stream;const size_t mm=M,nn=N,ne=E;
    if(!graph.uses_remote_ghosts()){
        evaluate_charges(positions);device->charge_legacy=true;
        auto partial=device->buffer<double>("mpi_layer_density",mm*8);
        if(M)check(cudaMemsetAsync(partial,0,mm*8*sizeof(double),stream),"clear partial source slots");
        if(N)lsc_cuda::seed_partial_layers<<<(nn*8+255)/256,256,0,stream>>>(N,M,device->working<double>("layer_density_coefficients"),partial);
        check(cudaGetLastError(),"seed serial partial source density");
    }else{
        const LSCSlotAdjacency adjacency(graph);
        auto upload=[&](const char* name,const std::vector<int>& values){auto ptr=device->buffer<int>(name,values.size());
            if(!values.empty())check(cudaMemcpyAsync(ptr,values.data(),values.size()*sizeof(int),cudaMemcpyHostToDevice,stream),"upload source slot adjacency");return ptr;};
        auto is=upload("mpi_incoming_starts",adjacency.incoming_starts),ie=upload("mpi_incoming_edges",adjacency.incoming_edges);
        auto os=upload("mpi_outgoing_starts",adjacency.outgoing_starts),oe=upload("mpi_outgoing_edges",adjacency.outgoing_edges);
        auto coords=device->buffer<double>("positions",nn*3);
        if(N)check(cudaMemcpyAsync(coords,positions.data(),positions.size_bytes(),cudaMemcpyHostToDevice,stream),"upload owned source positions");
        auto features=device->buffer<float>("mpi_source_features",mm*8*C);
        auto conditioning=device->buffer<float>("mpi_source_conditioning",mm*C);
        auto modulation=device->buffer<float>("mpi_source_modulation",mm*4*C);
        auto senders=device->buffer<float>("mpi_source_sender",mm*8*C),receivers=device->buffer<float>("mpi_source_receiver",mm*8*C);
        auto multipoles=device->buffer<float>("mpi_source_multipoles",mm*4);
        auto partial=device->buffer<double>("mpi_layer_density",mm*8);
        auto flux=device->buffer<float>("layer_edge_fluxes",ne*2);
        auto error=device->buffer<int>("mpi_charge_error",1);check(cudaMemsetAsync(error,0,sizeof(int),stream),"clear partial source status");
        const size_t initial_sgemm=device->sgemm_count,initial_tensor=device->tensor_core_gemm_count;
        const bool mixed=float32_mixed && tensor_cores;
        for(int t=0;t<2;++t){const auto prefix="sources."+std::to_string(t);
            auto output=features+size_t(t)*mm*4*C;auto mod=modulation+size_t(t)*2*mm*C;
            auto sender=senders+size_t(t)*mm*4*C,receiver=receivers+size_t(t)*mm*4*C;
            lsc_cuda::initial_features<<<(mm*C+255)/256,256,0,stream>>>(M,device->working<int>("node_types_all"),device->parameter<float>(prefix+".conditioning_embedding"),conditioning);
            for(int l=0;l<2;++l)device->gemm(prefix+".oxidation."+std::to_string(l),M,C,C,conditioning,mod+size_t(l)*mm*C,mixed);
            lsc_cuda::source_modulate<<<(mm*C+255)/256,256,0,stream>>>(M,forward_features_device(t+1),mod,output);
            device->linear(prefix+".linear1",M,output,sender,mixed);device->linear(prefix+".linear2",M,output,receiver,mixed);
            if(E)lsc_cuda::source_flux<<<E,C,0,stream>>>(M,device->working<int>("senders"),device->working<int>("receivers"),
                device->working<float>("spherical_harmonics"),device->working<int>("intervals"),device->working<float>("spline_coordinate"),
                device->parameter<float>(prefix+".radial.coefficients"),device->parameter<int>(prefix+".conv_tp.spec"),
                device->parameter<float>(prefix+".conv_tp.coefficients"),device->metadata.at(prefix+".conv_tp.spec").shape[0],sender,receiver,
                device->parameter<float>(prefix+".transfer_readout.weights.0"),device->parameter<float>(prefix+".transfer_readout.bias"),
                device->parameter<float>(prefix+".transfer_scale"),flux+size_t(t)*ne,error);
            device->linear(prefix+".multipoles",M,output,multipoles,mixed);
            lsc_cuda::source_density<<<(mm+255)/256,256,0,stream>>>(M,is,ie,os,oe,flux+size_t(t)*ne,multipoles,partial+size_t(t)*mm*4,error);
            check(cudaGetLastError(),"compute partial directed charge sources");
        }
        int status=0;check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read partial source status");
        check(cudaStreamSynchronize(stream),"complete partial source calculation");
        if(status)throw std::invalid_argument("MACELSCCuda: nonfinite partial charge/dipole prediction");
        source_sgemm_calls=device->sgemm_count-initial_sgemm;source_tensor_core_gemm_calls=device->tensor_core_gemm_count-initial_tensor;
    }
    device->charge_exchange.begin(graph);has_charge_results=false;
}
LSCCudaArray MACELSCCuda::download_partial_charges() const
{
    device->active();
    LSCChargeExchange::require(device->charge_exchange.partial(),"compute partial charges before downloading contributions");
    const int M=graph_layout().num_nodes();auto payload=device->buffer<double>("mpi_charge_payload",size_t(M)*2);
    if(M)lsc_cuda::pack_partial_charges<<<(size_t(M)*2+255)/256,256,0,device->stream>>>(0,M,M,device->working<double>("mpi_layer_density"),payload,false);
    LSCCudaArray out;out.dtype="float64";out.shape={size_t(M),2};out.values=std::vector<double>(size_t(M)*2);
    if(M)check(cudaMemcpyAsync(out.data(),payload,out.bytes(),cudaMemcpyDeviceToHost,device->stream),"read partial charge contributions");
    check(cudaStreamSynchronize(device->stream),"complete partial charge readback");return out;
}
void MACELSCCuda::pack_reverse_charges(int first,int count,std::span<double> values)
{
    device->active();device->charge_exchange.check_pack(first,count,graph_layout());
    LSCChargeExchange::require(values.size()==size_t(count)*2,"incorrect reverse charge send buffer size");
    if(count){auto payload=device->buffer<double>("mpi_charge_payload",values.size());
        lsc_cuda::pack_partial_charges<<<(values.size()+255)/256,256,0,device->stream>>>(first,count,graph_layout().num_nodes(),device->working<double>("mpi_layer_density"),payload,true);
        check(cudaGetLastError(),"pack returning partial charges");
        check(cudaMemcpyAsync(values.data(),payload,values.size_bytes(),cudaMemcpyDeviceToHost,device->stream),"read reverse charge payload");
        check(cudaStreamSynchronize(device->stream),"complete reverse charge send staging");}
    device->charge_exchange.sent(first,count);
}
void MACELSCCuda::unpack_reverse_charges(std::span<const int> slots,std::span<const double> values)
{
    device->active();device->charge_exchange.check_add(slots,values,graph_layout());
    if(slots.empty())return;
    auto indices=device->buffer<int>("mpi_charge_slots",slots.size());auto payload=device->buffer<double>("mpi_charge_payload",values.size());
    auto error=device->buffer<int>("mpi_charge_error",1);check(cudaMemsetAsync(error,0,sizeof(int),device->stream),"clear reverse charge status");
    check(cudaMemcpyAsync(indices,slots.data(),slots.size_bytes(),cudaMemcpyHostToDevice,device->stream),"upload reverse charge slots");
    check(cudaMemcpyAsync(payload,values.data(),values.size_bytes(),cudaMemcpyHostToDevice,device->stream),"upload reverse charge payload");
    lsc_cuda::add_partial_charges<<<(values.size()+255)/256,256,0,device->stream>>>(slots.size(),graph_layout().num_nodes(),indices,payload,device->working<double>("mpi_layer_density"),error);
    check(cudaGetLastError(),"add partial charges on owners");
    int status=0;check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,device->stream),"read reverse charge status");
    check(cudaStreamSynchronize(device->stream),"complete reverse charge accumulation");
    if(status)throw std::invalid_argument("MACELSCCuda: nonfinite accumulated charge");
}
void MACELSCCuda::complete_charge_exchange()
{
    device->active();device->charge_exchange.finish(graph_layout());
    const int N=num_nodes,E=num_edges,M=graph_layout().num_nodes(),C=num_channels;
    const size_t nn=N,ne=E,stride=std::max(nn,ne);const auto stream=device->stream;
    auto layers=device->buffer<double>("layer_density_coefficients",nn*8);
    if(N)lsc_cuda::extract_partial_layers<<<(nn*8+255)/256,256,0,stream>>>(N,M,device->working<double>("mpi_layer_density"),layers);
    auto features=device->buffer<float>("source_features",nn*8*C);
    if(N && !device->charge_legacy)lsc_cuda::extract_source_features<<<(nn*8*C+255)/256,256,0,stream>>>(N,M,C,device->working<float>("mpi_source_features"),features);
    auto layer_flux=device->working<float>("layer_edge_fluxes");
    auto density=device->buffer<double>("density_coefficients",nn*4),charges=device->buffer<double>("partial_charges",nn),dipoles=device->buffer<double>("partial_dipoles",nn*3);
    auto flux=device->buffer<double>("edge_fluxes",ne),terms=device->buffer<double>("source_dipole_terms",9*stride);
    auto formal=device->buffer<double>("formal_charge_dipole",3),atomic=device->buffer<double>("atomic_dipole_sum",3),transfer=device->buffer<double>("transfer_dipole",3),total=device->buffer<double>("dipole",3);
    auto error=device->buffer<int>("source_error",1);check(cudaMemsetAsync(error,0,sizeof(int),stream),"clear assembled density status");
    if(stride)check(cudaMemsetAsync(terms,0,9*stride*sizeof(double),stream),"clear assembled dipole terms");
    const auto coords=device->working<double>("positions");int status=0;
    if(N){const auto types=device->working<int>("node_types");
        lsc_cuda::source_nodes<<<(nn+255)/256,256,0,stream>>>(N,types,device->parameter<float>("formal_charges"),
            coords,layers,density,charges,dipoles,stride,terms,error);
        check(cudaGetLastError(),"assemble source node outputs");
    }
    if(E){
        lsc_cuda::source_edges<<<(ne+255)/256,256,0,stream>>>(E,layer_flux,device->working<double>("edge_vectors"),flux,stride,terms);
        check(cudaGetLastError(),"assemble transfer polarization");
    }
    if(stride){
        size_t bytes=0;
        check(cub::DeviceReduce::Sum(nullptr,bytes,terms,formal,int(stride),stream),"size dipole reduction");
        auto temp=device->buffer<char>("source_reduce_temp",bytes);
        for(int d=0;d<9;++d){
            double* output=d<3?formal+d:(d<6?atomic+d-3:transfer+d-6);
            check(cub::DeviceReduce::Sum(temp,bytes,terms+size_t(d)*stride,output,int(stride),stream),"reduce dipole component");
        }
    }else{
        for(auto ptr:{formal,atomic,transfer})check(cudaMemsetAsync(ptr,0,3*sizeof(double),stream),"clear empty dipole component");
    }
    lsc_cuda::source_dipole<<<1,32,0,stream>>>(formal,atomic,transfer,total,error);
    check(cudaGetLastError(),"assemble total LSC polarization");
    std::array<double,3> p{},f{},a{},e{};
    for(auto [host,gpu]:{std::pair(p.data(),total),std::pair(f.data(),formal),std::pair(a.data(),atomic),std::pair(e.data(),transfer)})
        check(cudaMemcpyAsync(host,gpu,3*sizeof(double),cudaMemcpyDeviceToHost,stream),"read dipole component");
    check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read source status");
    check(cudaStreamSynchronize(stream),"complete GPU charge sources");
    if(status&2)throw std::invalid_argument("MACELSCCuda: nonfinite charge/dipole prediction");
    auto& info=device->charge_metadata;
    device->result(info,"source_features","float32",{2,nn,size_t(4*C)});
    device->result(info,"layer_edge_fluxes","float32",{2,ne});
    device->result(info,"layer_density_coefficients","float64",{2,nn,4});
    device->result(info,"density_coefficients","float64",{nn,4});
    device->result(info,"partial_charges","float64",{nn});
    device->result(info,"partial_dipoles","float64",{nn,3});
    device->result(info,"edge_fluxes","float64",{ne});
    for(const auto& name:{"dipole","formal_charge_dipole","atomic_dipole_sum","transfer_dipole"})
        device->result(info,name,"float64",{3});
    dipole=p;formal_charge_dipole=f;atomic_dipole_sum=a;transfer_dipole=e;
    has_charge_results=true;
}

void MACELSCCuda::set_owned_potentials(std::span<const double> values)
{
    clear_force_results();has_energy_results=false;
    device->active();LSCChargeExchange::require(has_charge_results,"assemble owned charges before setting potentials");
    device->charge_exchange.set_potentials(graph_layout(),values);
    const int N=num_nodes,M=graph_layout().num_nodes();auto owned=device->buffer<double>("mpi_owned_potentials",values.size());
    auto halo=device->buffer<double>("mpi_potentials",size_t(M)*4);
    if(N)check(cudaMemcpyAsync(owned,values.data(),values.size_bytes(),cudaMemcpyHostToDevice,device->stream),"upload owned electrostatic potentials");
    if(graph_layout().num_ghosts()){
        if(M)lsc_cuda::publish_potentials<<<(size_t(M)*4+255)/256,256,0,device->stream>>>(N,M,device->working<int>("forward_owners"),owned,halo);
    }else if(N)check(cudaMemcpyAsync(halo,owned,values.size_bytes(),cudaMemcpyDeviceToDevice,device->stream),"publish owned electrostatic potentials");
    check(cudaGetLastError(),"publish potential slots");check(cudaStreamSynchronize(device->stream),"complete potential input staging");
}
void MACELSCCuda::pack_potentials(std::span<const int> slots,std::span<double> values) const
{
    device->active();device->charge_exchange.check_potential_pack(slots,graph_layout());
    LSCChargeExchange::require(values.size()==slots.size()*4,"incorrect potential send buffer size");if(slots.empty())return;
    auto indices=device->buffer<int>("mpi_charge_slots",slots.size());auto payload=device->buffer<double>("mpi_charge_payload",values.size());
    check(cudaMemcpyAsync(indices,slots.data(),slots.size_bytes(),cudaMemcpyHostToDevice,device->stream),"upload potential send slots");
    lsc_cuda::pack_density_potentials<<<(values.size()+255)/256,256,0,device->stream>>>(slots.size(),indices,device->working<double>("mpi_potentials"),payload);
    check(cudaGetLastError(),"pack electrostatic potentials");
    check(cudaMemcpyAsync(values.data(),payload,values.size_bytes(),cudaMemcpyDeviceToHost,device->stream),"read potential payload");
    check(cudaStreamSynchronize(device->stream),"complete potential send staging");
}
void MACELSCCuda::unpack_potentials(int first,std::span<const double> values)
{
    device->active();device->charge_exchange.check_potential_receive(first,values,graph_layout());
    if(!values.empty())check(cudaMemcpyAsync(device->working<double>("mpi_potentials")+size_t(first)*4,values.data(),values.size_bytes(),cudaMemcpyHostToDevice,device->stream),"copy received density potentials");
    check(cudaStreamSynchronize(device->stream),"complete potential receive staging");device->charge_exchange.received_potentials(first,values.size()/4);
}
void MACELSCCuda::complete_potential_exchange(){device->charge_exchange.complete_potentials(graph_layout());}
bool MACELSCCuda::potential_exchange_complete()const{return device->charge_exchange.potentials_ready();}
std::string MACELSCCuda::charge_exchange_stage()const{return device->charge_exchange.stage();}
LSCCudaArray MACELSCCuda::download_electrostatic_potentials()const
{
    device->active();
    LSCChargeExchange::require(device->charge_exchange.potentials_available(),"set owned electrostatic potentials first");
    LSCCudaArray out;out.dtype="float64";out.shape={size_t(graph_layout().num_nodes()),4};out.values=std::vector<double>(out.shape[0]*4);
    if(out.bytes())check(cudaMemcpyAsync(out.data(),device->working<double>("mpi_potentials"),out.bytes(),cudaMemcpyDeviceToHost,device->stream),"read potential slots");
    check(cudaStreamSynchronize(device->stream),"complete potential readback");return out;
}

void MACELSCCuda::reverse_distributed_local_layer(int t)
{
    const int N=num_nodes,E=num_edges,C=num_channels;const int K=device->charge_legacy?N:graph_layout().num_nodes(),M=graph_layout().num_nodes();const size_t nn=N,ne=E,kk=K;const auto stream=device->stream;
    const bool mixed=float32_mixed && tensor_cores;
    auto hidden=device->buffer<float>("mpi_owned_hidden",nn*4*C);
    lsc_cuda::reverse_feature_prefix<<<(nn*4*C+255)/256,256,0,stream>>>(N,M,N,C,device->working<float>(t==0?"mpi_H1_adj":"mpi_H2_adj"),hidden,false);
    auto h0=device->buffer<float>("mpi_local_input_adj",kk*4*C);check(cudaMemsetAsync(h0,0,kk*4*C*sizeof(float),stream),"clear MPI interaction input adjoint");
    auto residual=device->buffer<float>("mpi_residual_adj",nn*4*C);check(cudaMemsetAsync(residual,0,nn*4*C*sizeof(float),stream),"clear MPI residual adjoint");
    auto density_adj=device->working<double>("density_energy_gradient"),edge_adj=device->working<double>("edge_gradient");
        auto m_adj=device->buffer<float>("local_M_adjoint",nn*4*C),a_adj=device->buffer<float>("local_A_adjoint",nn*16*C);
        auto message_adj=device->buffer<float>("local_message_adjoint",nn*40*C),up_adj=device->buffer<float>("local_up_adjoint",kk*4*C);
        auto mask=device->buffer<float>("local_masked_adjoint",nn*4*C);

            const auto suffix="_"+std::to_string(t);const float* hidden_adj=hidden;
            check(cudaMemsetAsync(m_adj,0,nn*4*C*sizeof(float),stream),"clear product adjoint");
            check(cudaMemsetAsync(a_adj,0,nn*16*C*sizeof(float),stream),"clear interaction adjoint");
            check(cudaMemsetAsync(message_adj,0,nn*40*C*sizeof(float),stream),"clear message adjoint");
            check(cudaMemsetAsync(up_adj,0,kk*4*C*sizeof(float),stream),"clear lifted feature adjoint");
            device->linear_reverse("product_linear"+suffix,N,hidden_adj,m_adj,mixed);
            if(t==1)for(int e=0;e<num_elements;++e){
                lsc_cuda::reverse_mask<<<(nn*4*C+255)/256,256,0,stream>>>(N,C,device->working<int>("node_types"),e,hidden_adj,mask);
                device->linear_reverse("skip_1."+std::to_string(e),N,mask,residual,mixed);
            }
            if(t==1)lsc_cuda::reverse_feature_prefix<<<(nn*4*C+255)/256,256,0,stream>>>(N,N,K,C,residual,h0,true);
            lsc_cuda::local_contraction_reverse<<<(nn*C+127)/128,128,0,stream>>>(N,C,device->working<int>("node_types"),device->parameter<float>("M"+std::to_string(t)+"_weights"),
                device->working<float>("A"+std::to_string(t)),m_adj,a_adj);
            check(cudaGetLastError(),"reverse species product contraction");
            device->linear_reverse("linear"+suffix,N,a_adj,message_adj,mixed);
            if(E){
                const auto senders=device->working<int>("senders"),receivers=device->working<int>("receivers");const auto geometry=device->working<float4>("geometry");
                const auto Y=device->working<float>("spherical_harmonics"),dY=device->working<float>("spherical_harmonics_gradient"),coordinate=device->working<float>("spline_coordinate");
                const auto intervals=device->working<int>("intervals");const auto cg=device->parameter<float>("conv_tp"+suffix+".coefficients"),grid=device->parameter<float>("radial"+suffix+".grid");
                const auto up=device->working<float>("local_up"+suffix);
                if(t==0)lsc_cuda::local_convolution_reverse<0,false,true><<<E,C,0,stream>>>(N,senders,receivers,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_0.coefficients"),nullptr,grid,cg,up,message_adj,up_adj,edge_adj,K);
                else if(use_batched_reverse){
                    if(float32_mixed)
                        lsc_cuda::local_convolution_reverse_batched<true,true><<<N,C,0,stream>>>(N,K,device->working<int>("incoming_starts"),device->working<int>("incoming_edges"),senders,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.c01"),device->parameter<__half2>("radial_1.c23"),grid,cg,up,message_adj,up_adj,edge_adj);
                    else
                        lsc_cuda::local_convolution_reverse_batched<false,true><<<N,C,0,stream>>>(N,K,device->working<int>("incoming_starts"),device->working<int>("incoming_edges"),senders,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.coefficients"),nullptr,grid,cg,up,message_adj,up_adj,edge_adj);
                    ++batched_reverse_calls;
                }
                else if(float32_mixed)lsc_cuda::local_convolution_reverse<1,true,true><<<E,C,0,stream>>>(N,senders,receivers,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.c01"),device->parameter<__half2>("radial_1.c23"),grid,cg,up,message_adj,up_adj,edge_adj,K);
                else lsc_cuda::local_convolution_reverse<1,false,true><<<E,C,0,stream>>>(N,senders,receivers,geometry,Y,dY,intervals,coordinate,device->parameter<float>("radial_1.coefficients"),nullptr,grid,cg,up,message_adj,up_adj,edge_adj,K);
                check(cudaGetLastError(),"reverse fused local convolution");
            }
            // The initial species/formal-charge embedding is independent of
            // Cartesian geometry, so its final discarded adjoint is unnecessary.
            if(t==1)device->linear_reverse("linear_up_1",K,up_adj,h0,mixed);

    if(t==1)lsc_cuda::reverse_feature_prefix<<<(kk*4*C+255)/256,256,0,stream>>>(K,K,M,C,h0,device->working<float>("mpi_H1_adj"),true);
}

int MACELSCCuda::distributed_reverse_phase()const{return device->reverse_exchange.phase;}
void MACELSCCuda::begin_distributed_reverse(std::span<const double> pg,const std::array<double,9>& es,
    const std::array<double,9>& cell,const std::array<double,3>& field,double el)
{
    device->active();const auto& g=graph_layout();const int N=num_nodes,M=g.num_nodes(),E=num_edges,C=num_channels,S=device->charge_legacy?N:M;
    const size_t nn=N,mm=M,ne=E,ss=S;const auto stream=device->stream;const bool mixed=float32_mixed && tensor_cores;
    LSCReverseExchange::check(device->charge_exchange.potentials_ready(),"complete potential exchange first");
    LSCReverseExchange::check(pg.size()==nn*3,"expected owned position gradient[N_owned,3]");
    LSCChargeExchange::finite(pg);LSCChargeExchange::finite(es);LSCChargeExchange::finite(cell);LSCChargeExchange::finite(field);LSCChargeExchange::finite({&el,1});
    clear_force_results();device->mpi_electrostatic_strain=es;device->energy_cell=cell;device->energy_field=field;
    device->energy_volume=std::abs(cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])-cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])+cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]));
    electrostatic_energy=el;field_energy=0;for(int d=0;d<3;++d)field_energy+=field[d]*dipole[d];energy=local_energy+el+field_energy;has_energy_results=true;
    LSCSlotAdjacency adj(g);auto upload=[&](const char* name,const std::vector<int>& v){auto p=device->buffer<int>(name,v.size());if(!v.empty())check(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(int),cudaMemcpyHostToDevice,stream),"upload reverse slot adjacency");};
    upload("mpi_incoming_starts",adj.incoming_starts);upload("mpi_incoming_edges",adj.incoming_edges);upload("mpi_outgoing_starts",adj.outgoing_starts);upload("mpi_outgoing_edges",adj.outgoing_edges);
    auto da=device->buffer<double>("mpi_density_adj",mm*4),pa=device->buffer<double>("position_gradient",nn*3),ea=device->buffer<double>("edge_gradient",ne*3);
    if(M)lsc_cuda::reverse_density_seed<<<(mm*4+255)/256,256,0,stream>>>(N,M,device->working<double>("mpi_potentials"),da);
    if(N){check(cudaMemcpyAsync(pa,pg.data(),pg.size_bytes(),cudaMemcpyHostToDevice,stream),"seed owned fixed-density position derivative");
        lsc_cuda::reverse_field_nodes<<<(nn+255)/256,256,0,stream>>>(N,device->working<int>("node_types"),device->parameter<float>("formal_charges"),field[0],field[1],field[2],da,pa);}
    if(E)lsc_cuda::reverse_field_edges<<<(ne+255)/256,256,0,stream>>>(E,device->working<double>("edge_fluxes"),field[0],field[1],field[2],ea);
    auto owned_da=device->buffer<double>("density_energy_gradient",nn*4);if(N)check(cudaMemcpyAsync(owned_da,da,nn*4*sizeof(double),cudaMemcpyDeviceToDevice,stream),"save owned density adjoints");
    auto owned=device->buffer<float>("mpi_owned_readout",nn*4*C),ones=device->buffer<float>("force_ones",nn);
    if(N)lsc_cuda::reverse_fill<<<(nn+255)/256,256,0,stream>>>(nn,1.f,ones);
    auto sf=device->buffer<float>("source_features_adjoint",ss*4*C),sa=device->buffer<float>("source_sender_adjoint",ss*4*C),ra=device->buffer<float>("source_receiver_adjoint",ss*4*C);
    auto mp=device->buffer<float>("source_multipole_adjoint",ss*4),ha=device->buffer<float>("mpi_source_hidden_adj",ss*4*C);
    if(S)lsc_cuda::source_multipole_seed<<<(ss+255)/256,256,0,stream>>>(S,da,mp);
    for(int t=0;t<2;++t){auto halo=device->buffer<float>(t==0?"mpi_H1_adj":"mpi_H2_adj",mm*4*C);
        if(M)check(cudaMemsetAsync(halo,0,mm*4*C*sizeof(float),stream),"clear all-slot hidden derivatives");
        if(N){check(cudaMemsetAsync(owned,0,nn*4*C*sizeof(float),stream),"clear readout derivatives");device->linear_reverse("readout_"+std::to_string(t),N,ones,owned,mixed);
            lsc_cuda::reverse_feature_prefix<<<(nn*4*C+255)/256,256,0,stream>>>(N,N,M,C,owned,halo,true);}
        if(!S)continue;for(auto p:{sf,sa,ra,ha})check(cudaMemsetAsync(p,0,ss*4*C*sizeof(float),stream),"clear source reverse workspace");
        const auto prefix="sources."+std::to_string(t);device->linear_reverse(prefix+".multipoles",S,mp,sf,mixed);
        const auto sender=device->working<float>(device->charge_legacy?"source_sender":"mpi_source_sender")+size_t(t)*ss*4*C;
        const auto receiver=device->working<float>(device->charge_legacy?"source_receiver":"mpi_source_receiver")+size_t(t)*ss*4*C;
        if(E)lsc_cuda::source_transfer_reverse<<<E,C,0,stream>>>(S,device->working<int>("senders"),device->working<int>("receivers"),device->working<float4>("geometry"),
            device->working<float>("spherical_harmonics"),device->working<float>("spherical_harmonics_gradient"),device->working<int>("intervals"),device->working<float>("spline_coordinate"),
            device->parameter<float>(prefix+".radial.coefficients"),device->parameter<float>(prefix+".radial.grid"),device->parameter<int>(prefix+".conv_tp.spec"),device->parameter<float>(prefix+".conv_tp.coefficients"),
            device->metadata.at(prefix+".conv_tp.spec").shape[0],device->parameter<float>(prefix+".transfer_readout.weights.0"),device->parameter<float>(prefix+".transfer_scale"),sender,receiver,da,field[0],field[1],field[2],device->working<double>("edge_vectors"),sa,ra,ea);
        device->linear_reverse(prefix+".linear1",S,sa,sf,mixed);device->linear_reverse(prefix+".linear2",S,ra,sf,mixed);
        const auto weights=device->working<float>(device->charge_legacy?"source_modulation":"mpi_source_modulation")+size_t(t)*2*ss*C;
        lsc_cuda::source_modulation_reverse<<<(ss*C+255)/256,256,0,stream>>>(S,sf,weights,ha);
        lsc_cuda::reverse_feature_prefix<<<(ss*4*C+255)/256,256,0,stream>>>(S,S,M,C,ha,halo,true);
    }
    check(cudaGetLastError(),"differentiate distributed sources");check(cudaStreamSynchronize(stream),"complete distributed source reverse");device->reverse_exchange.begin(g);
}
void MACELSCCuda::pack_reverse_gradients(int layer,int first,int count,std::span<double> values)
{
    device->active();LSCReverseExchange::check(layer==1 || layer==2,"expected layer 1 or 2");auto& s=device->reverse_exchange;
    s.check_pack(layer==2?1:3,first,count,4*num_channels,values,graph_layout());
    if(count){auto packet=device->buffer<double>("mpi_reverse_packet",values.size());lsc_cuda::reverse_feature_packet<<<(values.size()+255)/256,256,0,device->stream>>>(count,first,graph_layout().num_nodes(),num_channels,device->working<float>(layer==1?"mpi_H1_adj":"mpi_H2_adj"),packet);
        check(cudaGetLastError(),"pack reverse feature derivatives");check(cudaMemcpyAsync(values.data(),packet,values.size_bytes(),cudaMemcpyDeviceToHost,device->stream),"stage reverse feature packet");check(cudaStreamSynchronize(device->stream),"complete reverse packet");}s.sent(first,count);
}
void MACELSCCuda::unpack_reverse_gradients(int layer,std::span<const int> slots,std::span<const double> values)
{
    device->active();LSCReverseExchange::check(layer==1 || layer==2,"expected layer 1 or 2");device->reverse_exchange.check_add(layer==2?1:3,slots,4*num_channels,values,graph_layout());
    for(double x:values)LSCReverseExchange::check(std::abs(x)<=std::numeric_limits<float>::max(),"gradient exceeds FP32 range");if(slots.empty())return;
    auto ids=device->buffer<int>("mpi_reverse_slots",slots.size());auto packet=device->buffer<double>("mpi_reverse_packet",values.size());
    check(cudaMemcpyAsync(ids,slots.data(),slots.size_bytes(),cudaMemcpyHostToDevice,device->stream),"stage reverse slots");check(cudaMemcpyAsync(packet,values.data(),values.size_bytes(),cudaMemcpyHostToDevice,device->stream),"stage reverse derivatives");
    lsc_cuda::reverse_feature_add<<<(values.size()+255)/256,256,0,device->stream>>>(slots.size(),graph_layout().num_nodes(),num_channels,ids,packet,device->working<float>(layer==1?"mpi_H1_adj":"mpi_H2_adj"));
    check(cudaGetLastError(),"sum reverse feature derivatives");check(cudaStreamSynchronize(device->stream),"complete reverse receive");
}
void MACELSCCuda::complete_reverse_gradient_exchange(int layer){device->active();LSCReverseExchange::check(layer==1 || layer==2,"expected layer 1 or 2");device->reverse_exchange.finish(layer==2?1:3,graph_layout());}
void MACELSCCuda::reverse_distributed_second_layer(){device->active();device->reverse_exchange.require_phase(2);if(num_nodes)reverse_distributed_local_layer(1);device->reverse_exchange.next(2,3,graph_layout());}
void MACELSCCuda::reverse_distributed_first_layer(){device->active();device->reverse_exchange.require_phase(4);if(num_nodes)reverse_distributed_local_layer(0);device->reverse_exchange.next(4,5,graph_layout());}
void MACELSCCuda::finalize_distributed_forces(bool compute_stress)
{
    device->active();device->reverse_exchange.require_phase(5);const int N=num_nodes,M=graph_layout().num_nodes(),E=num_edges;const size_t nn=N,mm=M,ne=E,stride=nn+ne;const auto stream=device->stream;
    const auto field=device->energy_field;const double volume=device->energy_volume;if(compute_stress)LSCReverseExchange::check(lsc_cuda::energy_finite(volume) && volume>0,"stress requires nonsingular cell");
    auto forces=device->buffer<double>("mpi_force_slots",mm*3),pf=device->buffer<double>("position_forces",nn*3),ef=device->buffer<double>("edge_forces",ne*3);
    auto terms=device->buffer<double>("force_strain_terms",9*stride),strain=device->buffer<double>("force_strain",9);auto error=device->buffer<int>("force_error",1);check(cudaMemsetAsync(error,0,sizeof(int),stream),"clear force status");
    if(stride)check(cudaMemsetAsync(terms,0,9*stride*sizeof(double),stream),"clear distributed strain terms");
    if(M)lsc_cuda::reverse_force_slots<<<(mm+255)/256,256,0,stream>>>(N,M,device->working<int>("node_types"),device->parameter<float>("formal_charges"),device->working<double>("positions"),
        device->working<int>("mpi_incoming_starts"),device->working<int>("mpi_incoming_edges"),device->working<int>("mpi_outgoing_starts"),device->working<int>("mpi_outgoing_edges"),device->working<double>("position_gradient"),device->working<double>("edge_gradient"),field[0],field[1],field[2],stride,terms,forces,pf,error);
    if(E)lsc_cuda::reverse_force_edges<<<(ne+255)/256,256,0,stream>>>(N,E,device->working<double>("edge_vectors"),device->working<double>("edge_gradient"),stride,terms,ef);
    if(stride){size_t bytes=0;check(cub::DeviceReduce::Sum(nullptr,bytes,terms,strain,int(stride),stream),"size reverse strain reduction");auto temp=device->buffer<char>("force_reduce_temp",bytes);for(int d=0;d<9;++d)check(cub::DeviceReduce::Sum(temp,bytes,terms+size_t(d)*stride,strain+d,int(stride),stream),"reduce distributed strain");}
    else check(cudaMemsetAsync(strain,0,9*sizeof(double),stream),"clear empty strain");
    std::array<double,9> raw{};int status=0;check(cudaMemcpyAsync(raw.data(),strain,9*sizeof(double),cudaMemcpyDeviceToHost,stream),"read distributed strain");check(cudaMemcpyAsync(&status,error,sizeof(int),cudaMemcpyDeviceToHost,stream),"read force status");check(cudaStreamSynchronize(stream),"complete distributed force assembly");
    LSCReverseExchange::check(!status,"nonfinite force prediction");for(int d=0;d<9;++d)raw[d]+=device->mpi_electrostatic_strain[d];LSCChargeExchange::finite(raw);
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)strain_derivative[3*a+b]=.5*(raw[3*a+b]+raw[3*b+a]);constexpr int po[6]={0,4,8,1,2,5},ao[6]={0,4,8,5,2,1};
    for(int d=0;d<6;++d){virial[d]=-strain_derivative[po[d]];if(compute_stress)stress[d]=strain_derivative[ao[d]]/volume;}has_stress_results=compute_stress;device->reverse_exchange.next(5,6,graph_layout());
}
void MACELSCCuda::pack_reverse_forces(int first,int count,std::span<double> values)
{
    device->active();auto& s=device->reverse_exchange;s.check_pack(6,first,count,3,values,graph_layout());
    if(count){auto packet=device->buffer<double>("mpi_reverse_packet",values.size());lsc_cuda::reverse_force_packet<<<(values.size()+255)/256,256,0,device->stream>>>(count,first,device->working<double>("mpi_force_slots"),packet);check(cudaGetLastError(),"pack ghost forces");
        check(cudaMemcpyAsync(values.data(),packet,values.size_bytes(),cudaMemcpyDeviceToHost,device->stream),"stage ghost forces");check(cudaStreamSynchronize(device->stream),"complete force packet");}s.sent(first,count);
}
void MACELSCCuda::unpack_reverse_forces(std::span<const int> slots,std::span<const double> values)
{
    device->active();device->reverse_exchange.check_add(6,slots,3,values,graph_layout());if(slots.empty())return;
    auto ids=device->buffer<int>("mpi_reverse_slots",slots.size());auto packet=device->buffer<double>("mpi_reverse_packet",values.size());check(cudaMemcpyAsync(ids,slots.data(),slots.size_bytes(),cudaMemcpyHostToDevice,device->stream),"stage force slots");check(cudaMemcpyAsync(packet,values.data(),values.size_bytes(),cudaMemcpyHostToDevice,device->stream),"stage force returns");
    lsc_cuda::reverse_force_add<<<(values.size()+255)/256,256,0,device->stream>>>(slots.size(),ids,packet,device->working<double>("mpi_force_slots"));check(cudaGetLastError(),"sum force returns");check(cudaStreamSynchronize(device->stream),"complete force receive");
}
void MACELSCCuda::complete_force_exchange()
{
    device->active();device->reverse_exchange.finish(6,graph_layout());const size_t nn=num_nodes,ne=num_edges;auto f=device->buffer<double>("forces",nn*3);
    if(nn)check(cudaMemcpyAsync(f,device->working<double>("mpi_force_slots"),nn*3*sizeof(double),cudaMemcpyDeviceToDevice,device->stream),"publish owned forces");
    check(cudaStreamSynchronize(device->stream),"complete owned forces");auto& info=device->force_metadata;device->result(info,"forces","float64",{nn,3});device->result(info,"position_forces","float64",{nn,3});device->result(info,"edge_forces","float64",{ne,3});device->result(info,"density_energy_gradient","float64",{nn,4});has_force_results=true;
}

LSCElectrostaticsContext MACELSCCuda::electrostatics_context(const std::array<double,9>& cell,const std::array<bool,3>& pbc,const std::string& mode) const
{
    device->active();LSCElectrostaticsContext context{cell,pbc,mode=="auto"?"mixed_periodic":mode,{}};
    auto& p=context.parameters;const auto& ep=device->electrostatic;
    p.field_constant=ep.field_constant;p.kspace_cutoff=device->electrostatic_cutoff;p.sigma2=ep.sigma2;
    std::copy_n(ep.a,2,p.a.begin());std::copy_n(ep.self_recip,4,p.reciprocal_self.begin());
    std::copy_n(ep.self_real,4,p.realspace_self.begin());std::copy_n(ep.displacements,9,p.displacements.begin());
    p.sigma=ep.sigma;p.offset=ep.offset;p.epsilon=ep.epsilon;p.include_self=ep.include_self;return context;
}
