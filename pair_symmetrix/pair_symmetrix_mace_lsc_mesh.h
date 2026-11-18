/* LSC distributed grid adapted from Symmetrix/Polar and LAMMPS FFT.
   LAMMPS, https://www.lammps.org/, Sandia National Laboratories.
   Copyright (2003) Sandia Corporation. GNU GPL; see pair_symmetrix/LICENSE. */
#ifndef PAIR_CLASS
#ifndef LMP_SYMMETRIX_LSC_MESH_H
#define LMP_SYMMETRIX_LSC_MESH_H
#include "mace_lsc_partition_mesh.hpp"
#include <memory>
namespace LAMMPS_NS {
class LAMMPS;
class Pair;
class LSCDistributedMesh {
 public:
  LSCDistributedMesh(LAMMPS *, Pair *, double, int, bool);
  ~LSCDistributedMesh();
  LSCElectrostaticsResult evaluate(std::span<const double>, std::span<const double>,
                                   const LSCElectrostaticsContext &);
  void pack(void *, int, int *);
  void unpack(void *, int, int *, bool);
  int local_modes = 0, local_brick = 0;
  size_t plan_generation = 0, device_fft_calls = 0;
  std::array<int, 3> shape{};

 private:
  struct Workspace;
  std::unique_ptr<Workspace> workspace;
};
}    // namespace LAMMPS_NS
#endif
#endif
