#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <span>
#include <variant>
#include <vector>
#include "mace_lsc_electrostatics.hpp"
#include "mace_lsc_graph.hpp"

// Host packing stays available in CPU builds, so layout/precision validation
// does not depend on access to a GPU. uint16 stores IEEE binary16 bit patterns.
struct LSCCudaArray {
    using Storage = std::variant<std::vector<float>, std::vector<double>,
                                 std::vector<std::int32_t>, std::vector<std::uint16_t>>;
    std::string dtype;
    std::vector<size_t> shape;
    Storage values;
    size_t size() const;
    size_t bytes() const;
    const void* data() const;
    void* data();
};

struct LSCCudaParameters {
    std::string precision;
    int num_elements = 0, num_channels = 0, num_interactions = 0, num_knots = 0;
    double r_cut = 0, r_min = 0;
    std::vector<int> atomic_numbers;
    std::map<std::string, LSCCudaArray> arrays;
    size_t bytes() const;
};
LSCCudaParameters prepare_lsc_cuda_parameters(const std::string& filename, const std::string& precision);
bool lsc_cuda_compiled();
bool lsc_cuda_available();

// Local interactions, charge sources, geometry and splines run on the device.
// Electrostatics and the full analytic reverse pass also run on the device.
class MACELSCCuda {
public:
    explicit MACELSCCuda(const std::string& filename, const std::string& precision = "float32");
    ~MACELSCCuda();
    MACELSCCuda(const MACELSCCuda&) = delete;
    MACELSCCuda& operator=(const MACELSCCuda&) = delete;
    std::string precision, device_name;
    int device_id = -1, compute_capability_major = 0, compute_capability_minor = 0;
    int num_elements = 0, num_channels = 0, num_interactions = 0, num_knots = 0;
    double r_cut = 0, r_min = 0;
    std::vector<int> atomic_numbers;
    bool float32_mixed = false, tensor_cores = false, parameters_on_device = false;
    // Receiver batches reuse message derivatives and shared edge geometry.
    bool use_batched_reverse = false;
    size_t batched_reverse_calls = 0;
    // One-rank source reverse batches reuse receiver features and avoid their atomics.
    bool use_batched_source_reverse = false;
    size_t batched_source_reverse_calls = 0;
    // One-rank scalar-source forward; generic instructions retain their path.
    bool use_batched_source_forward = false;
    size_t batched_source_forward_calls = 0;
    // Exact 64-function first-layer basis; retain the expanded path for checks.
    bool use_compact_first_layer = false, can_compact_first_layer = false;
    size_t compact_first_forward_calls = 0, compact_first_reverse_calls = 0;
    size_t device_parameter_bytes = 0;
    std::map<std::string, LSCCudaArray> parameter_metadata() const;
    LSCCudaArray download_parameter(const std::string& name) const;
    int num_nodes = 0, num_edges = 0;
    bool has_geometry = false, has_spline_results = false;
    std::string spline_name;
    void prepare_graph(const LSCGraph& graph);
    // One-rank device input. Senders/receivers index owned nodes; periodic
    // image displacements stay on edges. The caller finishes input writes
    // before entering; inputs are copied on-device into evaluator storage.
    void prepare_graph_device(int nodes,int edges,const int* types,
        const int* senders,const int* receivers,const double* vectors);
    std::array<double,3> evaluate_charges_device(const double* positions);
    const double* forces_device() const;
    const double* partial_charges_device() const;
    const double* node_energies_device();
    const LSCGraph& graph_layout() const;
    std::string evaluation_stage() const;
    void pack_forward_features(int layer,std::span<const int> slots,std::span<double> values) const;
    void unpack_forward_features(int layer,int first,std::span<const double> values);
    void complete_forward_exchange(int layer);
    bool forward_exchange_complete(int layer) const;
    // All-slot halo storage; owned layer_features_device() remains unchanged.
    float* forward_features_device(int layer) const;
    LSCCudaArray download_forward_features(int layer) const;
    // MPI source stage: reverse adds two layer monopoles; potentials are full dE_el/drho.
    void compute_partial_charges(std::span<const double> positions);
    void pack_reverse_charges(int first,int count,std::span<double> values);
    void unpack_reverse_charges(std::span<const int> slots,std::span<const double> values);
    void complete_charge_exchange();
    void set_owned_potentials(std::span<const double> values);
    void pack_potentials(std::span<const int> slots,std::span<double> values) const;
    void unpack_potentials(int first,std::span<const double> values);
    void complete_potential_exchange();
    bool potential_exchange_complete() const;
    std::string charge_exchange_stage() const;
    LSCCudaArray download_partial_charges() const;
    LSCCudaArray download_electrostatic_potentials() const;
    // Distributed reverse uses full backend derivatives; all ghost messages SUM.
    // Backend handoff: positive fixed-density owned dE_el/dr and rank strain/energy
    // contributions. Exclude external fields. Stress/virial are rank contributions.
    void begin_distributed_reverse(std::span<const double> position_gradient,
        const std::array<double,9>& strain,const std::array<double,9>& cell,
        const std::array<double,3>& field,double electrostatic_energy);
    void pack_reverse_gradients(int layer,int first,int count,std::span<double> values);
    void unpack_reverse_gradients(int layer,std::span<const int> slots,std::span<const double> values);
    void complete_reverse_gradient_exchange(int layer);
    void reverse_distributed_second_layer();
    void reverse_distributed_first_layer();
    void finalize_distributed_forces(bool compute_stress=false);
    void pack_reverse_forces(int first,int count,std::span<double> values);
    void unpack_reverse_forces(std::span<const int> slots,std::span<const double> values);
    void complete_force_exchange();
    int distributed_reverse_phase() const;
    void compute_first_layer();
    void compute_second_layer();
    double compute_readouts();
    void reverse_charges();
    void reverse_second_layer();
    void reverse_first_layer();
    void finalize_forces(bool compute_stress=false);
    // Borrowed l-major buffers, for future device communication on cuda_stream().
    float* layer_features_device(int layer);
    float* layer_feature_gradients_device(int layer);
    CUstream_st* cuda_stream() const;
    LSCCudaArray download_layer_features(int layer,bool gradients=false);
    // Physical edge vectors are receiver minus sender, including periodic images.
    void compute_geometry(std::span<const int> types, std::span<const int> senders,
                          std::span<const int> receivers, std::span<const double> vectors);
    std::map<std::string, LSCCudaArray> download_geometry() const;
    void evaluate_splines(const std::string& name);
    std::map<std::string, LSCCudaArray> download_splines() const;
    bool has_local_results = false;
    double local_energy = 0;
    bool can_evaluate_local() const { return parameters_on_device; }
    double compute_local(std::span<const int> types, std::span<const int> senders,
                         std::span<const int> receivers, std::span<const double> vectors);
    double evaluate_local(); // Reuse geometry already resident on this GPU.
    // Diagnostic feature arrays are returned in CPU's [node][l][m][channel]
    // order; the working GPU arrays retain Polar's [l][node][m][channel] layout.
    std::map<std::string, LSCCudaArray> download_local(bool diagnostics=true) const;
    bool has_charge_results = false;
    std::array<double,3> dipole{}, formal_charge_dipole{}, transfer_dipole{}, atomic_dipole_sum{};
    size_t source_sgemm_calls = 0, source_tensor_core_gemm_calls = 0;
    bool can_evaluate_sources() const { return parameters_on_device; }
    std::array<double,3> compute_charges(std::span<const int> types,std::span<const int> senders,
        std::span<const int> receivers,std::span<const double> vectors,std::span<const double> positions);
    std::array<double,3> evaluate_charges(std::span<const double> positions);
    std::map<std::string,LSCCudaArray> download_charges(bool diagnostics=true) const;
    bool has_energy_results = false;
    // Native mesh adapters receive device buffers and this evaluator's stream.
    void set_electrostatics_hook(LSCCudaElectrostaticsHook hook);
    bool has_electrostatics_hook() const;
    void set_mesh_electrostatics(double spacing=.7,int order=8);
    // Explicit host bridge for Python/custom-solver checks. Only this optional
    // bridge copies electrostatics buffers; native device hooks do not.
    void set_electrostatics_host_hook(LSCElectrostaticsHook hook);
    double energy = 0, electrostatic_energy = 0, field_energy = 0;
    bool can_evaluate_energy() const { return parameters_on_device; }
    double compute_energy(std::span<const int> types,std::span<const int> senders,std::span<const int> receivers,
        std::span<const double> vectors,std::span<const double> positions,const std::array<double,9>& cell,
        const std::array<bool,3>& pbc,const std::string& mode="auto",const std::array<double,3>& field={});
    LSCElectrostaticsContext electrostatics_context(const std::array<double, 9>& cell,
        const std::array<bool, 3>& pbc, const std::string& mode) const;
    double evaluate_energy(const std::array<double,9>& cell,const std::array<bool,3>& pbc,
        const std::string& mode="auto",const std::array<double,3>& field={});
    bool has_force_results = false,has_stress_results = false;
    std::array<double,6> stress{},virial{};
    std::array<double,9> strain_derivative{};
    bool can_evaluate_forces() const { return parameters_on_device; }
    bool can_evaluate_stress() const { return parameters_on_device; }
    double compute_energy_forces(std::span<const int> types,std::span<const int> senders,std::span<const int> receivers,
        std::span<const double> vectors,std::span<const double> positions,const std::array<double,9>& cell,
        const std::array<bool,3>& pbc,const std::string& mode="auto",const std::array<double,3>& field={},bool compute_stress=false);
    void evaluate_forces(bool compute_stress=false);
    std::map<std::string,LSCCudaArray> download_forces(bool diagnostics=true) const;
    size_t sgemm_calls() const;
    size_t tensor_core_gemm_calls() const;
    size_t workspace_bytes() const;
    bool can_evaluate() const { return parameters_on_device; }
private:
    void compact_first_forward();
    void compact_first_reverse(const float* interaction_adjoint);
    void evaluate_local_layer(int layer);
    void publish_forward_features(int layer);
    void prepare_geometry(std::span<const int> types,std::span<const int> senders,
        std::span<const int> receivers,std::span<const double> vectors,int sender_nodes,bool device_inputs=false);
    std::array<double,3> evaluate_charge_inputs(std::span<const double> positions,bool device_inputs);
    void reverse_local_layer(int layer);
    void reverse_distributed_local_layer(int layer);
    void clear_force_results();
    void clear_charge_results();
    struct DeviceData;
    std::unique_ptr<DeviceData> device;
};
