#include "mace_lsc_cuda.hpp"
#include <stdexcept>

struct MACELSCCuda::DeviceData {};
bool lsc_cuda_compiled(){return false;}
bool lsc_cuda_available(){return false;}
MACELSCCuda::MACELSCCuda(const std::string&,const std::string& selected_precision)
{
    if(selected_precision!="float32" and selected_precision!="float32-mixed")
        throw std::invalid_argument("MACELSCCuda: precision must be float32 or float32-mixed");
    throw std::runtime_error("MACELSCCuda: CUDA support is not built; use the separate CUDA development environment");
}
MACELSCCuda::~MACELSCCuda()=default;
void MACELSCCuda::prepare_graph_device(int,int,const int*,const int*,const int*,const double*)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::array<double,3> MACELSCCuda::evaluate_charges_device(const double*)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
const double* MACELSCCuda::forces_device()const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
const double* MACELSCCuda::partial_charges_device()const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
const double* MACELSCCuda::node_energies_device()
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::map<std::string,LSCCudaArray> MACELSCCuda::parameter_metadata() const{return {};}
LSCCudaArray MACELSCCuda::download_parameter(const std::string&) const
{
    throw std::runtime_error("MACELSCCuda: CUDA support is not built");
}

size_t MACELSCCuda::workspace_bytes() const{return 0;}
void MACELSCCuda::compute_geometry(std::span<const int>,std::span<const int>,std::span<const int>,std::span<const double>)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::evaluate_splines(const std::string&)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::map<std::string,LSCCudaArray> MACELSCCuda::download_geometry() const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::map<std::string,LSCCudaArray> MACELSCCuda::download_splines() const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}

size_t MACELSCCuda::sgemm_calls() const{return 0;}
size_t MACELSCCuda::tensor_core_gemm_calls() const{return 0;}
double MACELSCCuda::compute_local(std::span<const int>,std::span<const int>,std::span<const int>,std::span<const double>)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
double MACELSCCuda::evaluate_local()
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::map<std::string,LSCCudaArray> MACELSCCuda::download_local(bool) const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}

void MACELSCCuda::clear_charge_results(){}
std::array<double,3> MACELSCCuda::compute_charges(std::span<const int>,std::span<const int>,std::span<const int>,std::span<const double>,std::span<const double>)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::array<double,3> MACELSCCuda::evaluate_charges(std::span<const double>)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::map<std::string,LSCCudaArray> MACELSCCuda::download_charges(bool) const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}

double MACELSCCuda::compute_energy(std::span<const int>,std::span<const int>,std::span<const int>,std::span<const double>,std::span<const double>,const std::array<double,9>&,const std::array<bool,3>&,const std::string&,const std::array<double,3>&)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
double MACELSCCuda::evaluate_energy(const std::array<double,9>&,const std::array<bool,3>&,const std::string&,const std::array<double,3>&)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}

void MACELSCCuda::clear_force_results(){}
double MACELSCCuda::compute_energy_forces(std::span<const int>,std::span<const int>,std::span<const int>,std::span<const double>,std::span<const double>,const std::array<double,9>&,const std::array<bool,3>&,const std::string&,const std::array<double,3>&,bool)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::evaluate_forces(bool){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::map<std::string,LSCCudaArray> MACELSCCuda::download_forces(bool)const
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}

void MACELSCCuda::set_electrostatics_hook(LSCCudaElectrostaticsHook)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::set_electrostatics_host_hook(LSCElectrostaticsHook)
{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
bool MACELSCCuda::has_electrostatics_hook()const{return false;}

void MACELSCCuda::prepare_graph(const LSCGraph&){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
const LSCGraph& MACELSCCuda::graph_layout()const{throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
std::string MACELSCCuda::evaluation_stage()const{return "empty";}
void MACELSCCuda::compute_first_layer(){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::compute_second_layer(){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
double MACELSCCuda::compute_readouts(){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::reverse_charges(){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::reverse_second_layer(){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::reverse_first_layer(){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
void MACELSCCuda::finalize_forces(bool){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
float* MACELSCCuda::layer_features_device(int){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
float* MACELSCCuda::layer_feature_gradients_device(int){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}
CUstream_st* MACELSCCuda::cuda_stream()const{return nullptr;}
LSCCudaArray MACELSCCuda::download_layer_features(int,bool){throw std::runtime_error("MACELSCCuda: CUDA support is not built");}

void MACELSCCuda::pack_forward_features(int,std::span<const int>,std::span<double>) const {throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::unpack_forward_features(int,int,std::span<const double>) {throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::complete_forward_exchange(int) {throw std::runtime_error("CUDA support is not built");}
bool MACELSCCuda::forward_exchange_complete(int) const {return false;}
float* MACELSCCuda::forward_features_device(int) const {throw std::runtime_error("CUDA support is not built");}
LSCCudaArray MACELSCCuda::download_forward_features(int) const {throw std::runtime_error("CUDA support is not built");}

void MACELSCCuda::compute_partial_charges(std::span<const double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::pack_reverse_charges(int,int,std::span<double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::unpack_reverse_charges(std::span<const int>,std::span<const double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::complete_charge_exchange(){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::set_owned_potentials(std::span<const double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::pack_potentials(std::span<const int>,std::span<double>)const{throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::unpack_potentials(int,std::span<const double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::complete_potential_exchange(){throw std::runtime_error("CUDA support is not built");}
bool MACELSCCuda::potential_exchange_complete()const{return false;}
std::string MACELSCCuda::charge_exchange_stage()const{return "empty";}
LSCCudaArray MACELSCCuda::download_partial_charges()const{throw std::runtime_error("CUDA support is not built");}
LSCCudaArray MACELSCCuda::download_electrostatic_potentials()const{throw std::runtime_error("CUDA support is not built");}

void MACELSCCuda::begin_distributed_reverse(std::span<const double>,const std::array<double,9>&,const std::array<double,9>&,const std::array<double,3>&,double){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::pack_reverse_gradients(int,int,int,std::span<double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::unpack_reverse_gradients(int,std::span<const int>,std::span<const double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::complete_reverse_gradient_exchange(int){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::reverse_distributed_second_layer(){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::reverse_distributed_first_layer(){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::finalize_distributed_forces(bool){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::pack_reverse_forces(int,int,std::span<double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::unpack_reverse_forces(std::span<const int>,std::span<const double>){throw std::runtime_error("CUDA support is not built");}
void MACELSCCuda::complete_force_exchange(){throw std::runtime_error("CUDA support is not built");}
int MACELSCCuda::distributed_reverse_phase()const{throw std::runtime_error("CUDA support is not built");}

LSCElectrostaticsContext MACELSCCuda::electrostatics_context(const std::array<double,9>&,const std::array<bool,3>&,const std::string&)const{throw std::runtime_error("CUDA support is not built");}
