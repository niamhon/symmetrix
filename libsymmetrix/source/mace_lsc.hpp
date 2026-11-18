#pragma once

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "mace_lsc_electrostatics.hpp"
#include "mace_lsc_mesh.hpp"
#include "mace_lsc_graph.hpp"

struct LSCModelData;

// Native LSC energy, forces, stress and charge sources for the supported family.
class MACELSC {
public:
    explicit MACELSC(std::string filename);
    ~MACELSC();

    int format_version, num_elements, num_channels, num_interactions, l_max, L_max;
    int source_num_elements;
    std::string model_type, export_status, head, feature_layout, default_pbc_handling;
    double r_cut, radial_spline_r_min, radial_spline_scale, kspace_cutoff;
    double density_smearing_width;
    bool include_self_interaction;
    std::vector<int> atomic_numbers, source_element_indices;
    std::vector<double> atomic_energies, formal_charges;
    std::vector<std::vector<double>> node_embedding;
    std::vector<std::array<double, 2>> oxidation_state_ranges;
    std::vector<double> oxidation_state_factors;
    // Each shape is [hidden functions, knots, projected radial outputs].
    std::vector<std::array<int, 3>> interaction_radial_shapes, source_radial_shapes;

    std::array<std::vector<double>, 2> energy_readout_weights() const;
    void prepare_graph(const LSCGraph& graph);
    const LSCGraph& graph_layout() const;
    std::string evaluation_stage() const;
    // Canonical node-major MPI payload, owned slots followed by ghosts.
    void pack_forward_features(int layer,std::span<const int> slots,std::span<double> values) const;
    void unpack_forward_features(int layer,int first,std::span<const double> values);
    void complete_forward_exchange(int layer);
    bool forward_exchange_complete(int layer) const;
    std::span<const double> forward_features(int layer) const;
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
    std::vector<double> download_partial_charges() const;
    std::span<const double> electrostatic_potentials() const;
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
    std::array<double,3> evaluate_charges(std::span<const double> positions);
    double evaluate_energy(const std::array<double,9>& cell,const std::array<bool,3>& pbc,
        const std::string& mode="",const std::array<double,3>& field={});
    void evaluate_forces(bool compute_stress=false);
    void reverse_charges();
    void reverse_second_layer();
    void reverse_first_layer();
    void finalize_forces(bool compute_stress=false);
    std::span<double> layer_features(int layer);
    std::span<double> layer_feature_gradients(int layer);
    bool has_local_results=false;

    // A graph uses sender/receiver indices and physical receiver-sender image
    // vectors. Feature buffers follow Polar's [atom][l][m][channel] layout.
    void compute_local(std::span<const int> node_types,
                       std::span<const int> senders,
                       std::span<const int> receivers,
                       std::span<const double> edge_vectors);
    std::vector<double> initial_features, A0, A1, M0, M1, H1, H2;
    std::vector<double> local_node_energies;
    std::array<std::vector<double>, 2> layer_node_energies;
    double local_energy = 0.0;
    // Recompute local features for this graph, then charges and polarization.
    // Positions are physical Cartesian coordinates; edge vectors include images.
    void compute_charges(std::span<const int> node_types,
                         std::span<const int> senders,
                         std::span<const int> receivers,
                         std::span<const double> edge_vectors,
                         std::span<const double> positions);
    std::array<std::vector<double>, 2> source_features, layer_density_coefficients, layer_edge_fluxes;
    std::vector<double> density_coefficients, partial_charges, partial_dipoles, edge_fluxes;
    std::array<double, 3> dipole = {}, formal_charge_dipole = {}, transfer_dipole = {}, atomic_dipole_sum = {};
    bool has_charge_results = false;
    // cell rows are Cartesian lattice vectors; only finite clusters and fully
    // periodic cells are supported in this first energy implementation.
    void compute_energy(std::span<const int> node_types,
                        std::span<const int> senders, std::span<const int> receivers,
                        std::span<const double> edge_vectors, std::span<const double> positions,
                        const std::array<double, 9>& cell, const std::array<bool, 3>& pbc,
                        const std::string& pbc_handling, const std::array<double, 3>& external_field);
    // Polar-style optional replacement of the Coulomb solver. The hook returns
    // full energy derivatives; the native LSC reverse pass still handles rho(r).
    void set_electrostatics_hook(LSCElectrostaticsHook hook);
    bool has_electrostatics_hook() const;
    // Install a native, owned one-process CPU mesh solver (no Python callback).
    void set_mesh_electrostatics(double spacing = 0.7, int order = 8);
    LSCElectrostaticsResult evaluate_direct_electrostatics(
        std::span<const double> positions, std::span<const double> density,
        const std::array<double, 9>& cell, const std::array<bool, 3>& pbc,
        const std::string& pbc_handling = "") const;
    // Fixed-density mesh energy/potentials only. No state changes or force hook.
    LSCMeshEnergyResult evaluate_mesh_electrostatics(LSCCPUMesh& mesh,
        std::span<const double> positions, std::span<const double> density,
        const std::array<double, 9>& cell, const std::array<bool, 3>& pbc,
        const std::string& pbc_handling = "auto") const;
    // Full fixed-density analytic mesh derivatives, in the same convention as direct.
    LSCElectrostaticsResult evaluate_mesh_electrostatics_full(LSCCPUMesh& mesh,
        std::span<const double> positions, std::span<const double> density,
        const std::array<double, 9>& cell, const std::array<bool, 3>& pbc,
        const std::string& pbc_handling = "auto") const;
    double energy = 0.0, electrostatic_energy = 0.0, field_energy = 0.0;
    bool has_energy_results = false;
    void compute_energy_forces(std::span<const int> node_types,
                               std::span<const int> senders, std::span<const int> receivers,
                               std::span<const double> edge_vectors, std::span<const double> positions,
                               const std::array<double, 9>& cell, const std::array<bool, 3>& pbc,
                               const std::string& pbc_handling, const std::array<double, 3>& external_field,
                               bool compute_stress);
    // Flattened Cartesian force buffers; density adjoints retain [q,dy,dz,dx].
    std::vector<double> forces, edge_forces, position_forces, density_energy_gradient;
    std::array<double, 9> strain_derivative = {}, electrostatic_strain_derivative = {};
    // Virial: [xx,yy,zz,xy,xz,yz] in eV. Stress: ASE [xx,yy,zz,yz,xz,xy] in eV/A^3.
    std::array<double, 6> virial = {}, stress = {};
    bool has_force_results = false, has_stress_results = false;
    LSCElectrostaticsContext electrostatics_context(const std::array<double, 9>& cell,
        const std::array<bool, 3>& pbc, const std::string& mode) const;
    bool can_evaluate_energy() const;
    bool can_evaluate_forces() const { return can_evaluate_energy(); }
    bool can_evaluate_stress() const { return can_evaluate_energy(); }
    bool can_evaluate_local() const { return true; }
    bool can_evaluate_sources() const { return true; }
    bool can_evaluate() const { return can_evaluate_energy(); }

private:
    friend class LSCCudaMesh; // Fixed-density diagnostics use the same model parameters.
    // Keep bulk numerical storage out of the public header and Python API.
    friend LSCModelData read_validated_lsc_model(const std::string& filename);
    explicit MACELSC(const LSCModelData& model);
    void load_parameters(const LSCModelData& model);
    struct Parameters;
    std::unique_ptr<Parameters> parameters;
    void clear_charge_results();
    void clear_force_results();
    void publish_forward_features(int layer);
    void load_from_json(const std::string& filename);
};
