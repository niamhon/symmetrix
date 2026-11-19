/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Distributed under the GNU General Public License.
------------------------------------------------------------------------- */
#include "pair_symmetrix_mace_lsc_kokkos.h"

#ifdef SYMMETRIX_CUDA
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "mace_lsc_cuda.hpp"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

PairSymmetrixMACELSCCudaKokkos::PairSymmetrixMACELSCCudaKokkos(LAMMPS *lmp) :
    PairSymmetrixMACELSC<true>(lmp)
{
  kokkosable = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<LMPDeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK;
  datamask_modify = F_MASK | Q_MASK | ENERGY_MASK | VIRIAL_MASK;
}

PairSymmetrixMACELSCCudaKokkos::~PairSymmetrixMACELSCCudaKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(node_eatom, eatom);
}

void PairSymmetrixMACELSCCudaKokkos::coeff(int narg, char **arg)
{
  PairSymmetrixMACELSC<true>::coeff(narg, arg);
  Kokkos::realloc(device_species, species.size());
  auto host = Kokkos::create_mirror_view(device_species);
  for (size_t i = 0; i < species.size(); ++i) host(i) = species[i];
  Kokkos::deep_copy(device_species, host);
}

void PairSymmetrixMACELSCCudaKokkos::request_neighbor_list()
{
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_device(1);
}

void PairSymmetrixMACELSCCudaKokkos::init_style()
{
  if (comm->nprocs != 1)
    error->all(FLERR, "LSC CUDA Kokkos currently requires one MPI process; use the LSC CUDA pair for MPI");
  if (atom->map_user == atom->MAP_NONE)
    error->all(FLERR, "LSC CUDA Kokkos requires atom_modify map yes, array or hash");
  PairSymmetrixMACELSC<true>::init_style();
}

void PairSymmetrixMACELSCCudaKokkos::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag, 0);
  if (vflag_atom)
    error->all(FLERR, "LSC does not provide per-atom stress; use the global pressure tensor");
  auto k_list = static_cast<NeighListKokkos<LMPDeviceType> *>(list);
  const int nodes = k_list->inum;
  if (nodes != atom->nlocal)
    error->one(FLERR, "LSC CUDA Kokkos needs every local atom in its neighbor list");
  atomKK->sync(execution_space, datamask_read);
  auto x = atomKK->k_x.view<LMPDeviceType>();
  auto f = atomKK->k_f.view<LMPDeviceType>();
  auto q = atomKK->k_q.view<LMPDeviceType>();
  auto tags = atomKK->k_tag.view<LMPDeviceType>();
  auto type = atomKK->k_type.view<LMPDeviceType>();
  auto ilist = k_list->d_ilist;
  auto numneigh = k_list->d_numneigh;
  auto neighbors = k_list->d_neighbors;
  const int map_style = atom->map_style;
  auto map_array = atomKK->k_map_array;
  auto map_hash = atomKK->k_map_hash;
  map_array.sync<LMPDeviceType>();
  const double cutoff_squared = cutoff * cutoff;
  if (node_indices.size() < size_t(nodes)) Kokkos::realloc(node_indices, nodes);
  if (node_slots.size() < size_t(nodes)) Kokkos::realloc(node_slots, nodes);
  if (device_types.size() < size_t(nodes)) Kokkos::realloc(device_types, nodes);
  if (counts.size() < size_t(nodes)) Kokkos::realloc(counts, nodes);
  if (starts.size() < size_t(nodes) + 1) Kokkos::realloc(starts, nodes + 1);
  if (device_positions.size() < 3 * size_t(nodes)) Kokkos::realloc(device_positions, 3 * size_t(nodes));
  auto indices = node_indices;
  auto slots = node_slots;
  auto types = device_types;
  auto degree = counts;
  auto first = starts;
  auto positions = device_positions;
  auto species_map = device_species;
  Kokkos::parallel_for("LSC::nodes", nodes, KOKKOS_LAMBDA(const int ii) {
    const int i = ilist(ii);
    indices(ii) = i;
    slots(i) = ii;
    types(ii) = species_map(type(i));
    for (int d = 0; d < 3; ++d) positions(3 * ii + d) = x(i, d);
    int count = 0;
    for (int jj = 0; jj < numneigh(i); ++jj) {
      const int j = neighbors(i, jj) & NEIGHMASK;
      const double dx = x(i, 0) - x(j, 0);
      const double dy = x(i, 1) - x(j, 1);
      const double dz = x(i, 2) - x(j, 2);
      const double rsq = dx * dx + dy * dy + dz * dz;
      if ((rsq < cutoff_squared) && !((tags(i) == tags(j)) && (rsq == 0.0))) ++count;
    }
    degree(ii) = count;
  });
  int edges = 0;
  Kokkos::parallel_scan("LSC::edge_offsets", nodes, KOKKOS_LAMBDA(const int ii, int &offset, const bool final) {
    if (final) first(ii) = offset;
    offset += degree(ii);
    if (final && (ii == nodes - 1)) first(nodes) = offset;
  }, edges);
  if (edges < 0) error->one(FLERR, "LSC CUDA Kokkos neighbor count is too large");
  if (device_senders.size() < size_t(edges)) Kokkos::realloc(device_senders, edges);
  if (device_receivers.size() < size_t(edges)) Kokkos::realloc(device_receivers, edges);
  if (device_vectors.size() < 3 * size_t(edges)) Kokkos::realloc(device_vectors, 3 * size_t(edges));
  auto senders = device_senders;
  auto receivers = device_receivers;
  auto vectors = device_vectors;
  Kokkos::parallel_for("LSC::edges", nodes, KOKKOS_LAMBDA(const int ii) {
    const int i = indices(ii);
    int edge = first(ii);
    for (int jj = 0; jj < numneigh(i); ++jj) {
      const int j = neighbors(i, jj) & NEIGHMASK;
      const double dx = x(i, 0) - x(j, 0);
      const double dy = x(i, 1) - x(j, 1);
      const double dz = x(i, 2) - x(j, 2);
      const double rsq = dx * dx + dy * dy + dz * dz;
      if (!(rsq < cutoff_squared) || ((tags(i) == tags(j)) && (rsq == 0.0))) continue;
      const int owner = AtomKokkos::map_kokkos<LMPDeviceType>(tags(j), map_style, map_array, map_hash);
      // On one rank the atom map must resolve images to an owned slot.
      // Invalid slots are passed to the evaluator's existing bounds checks.
      senders(edge) = ((owner >= 0) && (owner < nodes)) ? slots(owner) : -1;
      receivers(edge) = ii;
      vectors(3 * edge) = dx;
      vectors(3 * edge + 1) = dy;
      vectors(3 * edge + 2) = dz;
      ++edge;
    }
  });
  Kokkos::fence();
  const std::array<double, 9> box = {
      domain->xprd, 0, 0, domain->xy, domain->yprd, 0, domain->xz, domain->yz, domain->zprd};
  const std::array<bool, 3> pbc = {bool(domain->xperiodic), bool(domain->yperiodic), bool(domain->zperiodic)};
  try {
    gpu->prepare_graph_device(nodes, edges, types.data(), senders.data(), receivers.data(), vectors.data());
    gpu->compute_first_layer();
    gpu->compute_second_layer();
    gpu->compute_readouts();
    gpu->evaluate_charges_device(positions.data());
    gpu->evaluate_energy(box, pbc, "auto", field);
    gpu->reverse_charges();
    gpu->reverse_second_layer();
    gpu->reverse_first_layer();
    gpu->finalize_forces(false);
    const double *forces = gpu->forces_device();
    const double *charges = gpu->partial_charges_device();
    const double *energies = nullptr;
    if (eflag_atom) {
      if (node_eatom.view<LMPDeviceType>().extent(0) < size_t(maxeatom)) {
        memoryKK->destroy_kokkos(node_eatom, eatom);
        memoryKK->create_kokkos(node_eatom, eatom, maxeatom, "pair:eatom");
      }
      Kokkos::deep_copy(node_eatom.view<LMPDeviceType>(), 0.0);
      energies = gpu->node_energies_device();
    }
    auto per_atom = node_eatom.view<LMPDeviceType>();
    const bool update_q = atom->q_flag;
    const bool output_eatom = eflag_atom;
    Kokkos::parallel_for("LSC::forces_and_outputs", nodes, KOKKOS_LAMBDA(const int ii) {
      const int i = indices(ii);
      for (int d = 0; d < 3; ++d) f(i, d) += forces[3 * ii + d];
      if (update_q) q(i) = charges[ii];
      if (output_eatom) per_atom(i) = energies[ii];
    });
    if (eflag_global) eng_vdwl += gpu->energy;
    if (vflag_global)
      for (int d = 0; d < 6; ++d) virial[d] += gpu->virial[d];
    for (int d = 0; d < 3; ++d) pvector[d] = gpu->dipole[d];
    atomKK->modified(execution_space, F_MASK | (update_q ? Q_MASK : 0));
    if (eflag_atom) {
      node_eatom.modify<LMPDeviceType>();
      node_eatom.sync_host();
    }
  } catch (const std::exception &e) {
    error->one(FLERR, "LSC CUDA Kokkos evaluation failed: {}", e.what());
  }
}
#endif
