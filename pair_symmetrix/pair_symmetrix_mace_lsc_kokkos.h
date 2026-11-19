/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Distributed under the GNU General Public License.
------------------------------------------------------------------------- */
#ifdef PAIR_CLASS
// clang-format off
#ifdef SYMMETRIX_CUDA
PairStyle(symmetrix/mace/lsc/cuda/kk,PairSymmetrixMACELSCCudaKokkos);
PairStyle(symmetrix/mace/lsc/cuda/kk/device,PairSymmetrixMACELSCCudaKokkos);
#endif
// clang-format on
#else
#ifndef LMP_PAIR_SYMMETRIX_MACE_LSC_KOKKOS_H
#define LMP_PAIR_SYMMETRIX_MACE_LSC_KOKKOS_H
#ifdef SYMMETRIX_CUDA
#include "pair_symmetrix_mace_lsc.h"
#include "kokkos_base.h"
#include "pair_kokkos.h"

namespace LAMMPS_NS {
class PairSymmetrixMACELSCCudaKokkos : public PairSymmetrixMACELSC<true>, public KokkosBase {
 public:
  PairSymmetrixMACELSCCudaKokkos(class LAMMPS *);
  ~PairSymmetrixMACELSCCudaKokkos() override;
  void coeff(int, char **) override;
  void init_style() override;
  void compute(int, int) override;

 protected:
  void request_neighbor_list() override;
  Kokkos::View<int *> device_species, node_indices, node_slots, device_types, counts, starts;
  Kokkos::View<int *> device_senders, device_receivers;
  Kokkos::View<double *> device_vectors, device_positions;
  DAT::tdual_double_1d node_eatom;
};
}    // namespace LAMMPS_NS
#endif
#endif
#endif
