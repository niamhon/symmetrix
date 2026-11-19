/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Distributed under the GNU General Public License.
------------------------------------------------------------------------- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(symmetrix/mace/lsc,PairSymmetrixMACELSC<false>);
#ifdef SYMMETRIX_CUDA
PairStyle(symmetrix/mace/lsc/cuda,PairSymmetrixMACELSC<true>);
#endif
// clang-format on
#else
#ifndef LMP_PAIR_SYMMETRIX_MACE_LSC_H
#define LMP_PAIR_SYMMETRIX_MACE_LSC_H
#include "pair.h"
#include <array>
#include <memory>
#include <span>
class MACELSC;
class MACELSCCuda;
#include "mace_lsc_electrostatics.hpp"
namespace LAMMPS_NS {
class LSCDistributedMesh;
template <bool CUDA> class PairSymmetrixMACELSC : public Pair {
 public:
  PairSymmetrixMACELSC(class LAMMPS *);
  ~PairSymmetrixMACELSC() override;
  void compute(int, int) override;
  void reset_grid() override;
  // Staged evaluator, also used by normal multi-rank mesh MD.
  void compute_forward();
  void compute_distributed_sources();
  void exchange_electrostatic_potentials(std::span<const double> owned_values);
  std::vector<double> owned_density() const;
  std::array<double, 3> source_dipole() const;
  std::vector<double> density_potentials() const;
  void reverse_distributed(std::span<const double> position_gradient,
                           const std::array<double, 9> &strain, const std::array<double, 9> &cell,
                           double electrostatic_energy, bool reduce_diagnostics = true);
  std::vector<double> owned_forces() const;
  void evaluate_distributed_mesh(bool reduce_diagnostics = true);
  LSCElectrostaticsResult mesh_result;
  std::array<int, 3> mesh_shape{};
  int mesh_local_modes = 0, mesh_local_brick = 0;
  size_t mesh_plan_generation = 0, mesh_device_fft_calls = 0;
  void pack_forward_grid(int, void *, int, int *) override;
  void unpack_forward_grid(int, void *, int, int *) override;
  void pack_reverse_grid(int, void *, int, int *) override;
  void unpack_reverse_grid(int, void *, int, int *) override;
  double distributed_energy = 0;
  std::array<double, 6> distributed_virial{}, distributed_stress{};
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  std::vector<double> forward_features(int layer) const;
  std::vector<double> forward_node_energies() const;
  double forward_local_energy() const;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

 protected:
  virtual void allocate();
  virtual void request_neighbor_list();
  void configure_electrostatics();
  std::unique_ptr<LSCDistributedMesh> distributed_mesh;
  std::unique_ptr<MACELSC> cpu;
  std::unique_ptr<MACELSCCuda> gpu;
  bool mixed = false;
  double cutoff = 0;
  double mesh_spacing = 0;
  int mesh_order = 8;
  int forward_layer = 0, reverse_mode = 0;
  std::array<double, 3> field{};
  std::vector<int> species, types, senders, receivers;
  std::vector<double> positions, vectors, force_buffer;
};
}    // namespace LAMMPS_NS
#endif
#endif
