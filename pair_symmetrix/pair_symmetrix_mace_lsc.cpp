/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Distributed under the GNU General Public License.
------------------------------------------------------------------------- */
#include "pair_symmetrix_mace_lsc.h"

#include "pair_symmetrix_mace_lsc_mesh.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "mace_lsc.hpp"
#include "mace_lsc_cuda.hpp"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace LAMMPS_NS;
namespace {
const char *const atomic_symbols[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
    "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
}

template <bool CUDA> PairSymmetrixMACELSC<CUDA>::PairSymmetrixMACELSC(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  nextra = 3;
  pvector = new double[nextra]();
}

template <bool CUDA> PairSymmetrixMACELSC<CUDA>::~PairSymmetrixMACELSC()
{
  delete[] pvector;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::allocate()
{
  allocated = 1;
  memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1, "pair:setflag");
  memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "pair:cutsq");
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 0;
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::settings(int narg, char **arg)
{
  distributed_mesh.reset();
  mesh_result = {};
  mixed = false;
  field = {};
  mesh_spacing = 0;
  mesh_order = 8;
  bool direct = false, mesh = false, order = false;
  for (int i = 0; i < narg; ++i) {
    if (strcmp(arg[i], "no_domain_decomposition") == 0) continue;
    if (strcmp(arg[i], "direct") == 0) {
      direct = true;
    } else if (strcmp(arg[i], "mesh") == 0) {
      if (i + 1 >= narg) error->all(FLERR, "LSC mesh requires a spacing in Angstrom");
      mesh_spacing = utils::numeric(FLERR, arg[++i], false, lmp);
      if (!std::isfinite(mesh_spacing) || mesh_spacing <= 0)
        error->all(FLERR, "LSC mesh spacing must be finite and positive");
      mesh = true;
    } else if (strcmp(arg[i], "order") == 0) {
      if (i + 1 >= narg) error->all(FLERR, "LSC order requires an integer from 4 through 10");
      mesh_order = utils::inumeric(FLERR, arg[++i], false, lmp);
      if (mesh_order < 4 || mesh_order > 10)
        error->all(FLERR, "LSC mesh order must be from 4 through 10");
      order = true;
    } else if (strcmp(arg[i], "float32-mixed") == 0) {
      if (!CUDA) error->all(FLERR, "float32-mixed requires pair_style symmetrix/mace/lsc/cuda");
      mixed = true;
    } else if (strcmp(arg[i], "field") == 0) {
      if (i + 3 >= narg) error->all(FLERR, "LSC field requires three components");
      for (int d = 0; d < 3; ++d) field[d] = utils::numeric(FLERR, arg[++i], false, lmp);
    } else
      error->all(FLERR, "Unknown symmetrix/mace/lsc keyword {}", arg[i]);
  }
  if (direct && mesh) error->all(FLERR, "Choose either LSC direct or mesh electrostatics");
  if (order && !mesh) error->all(FLERR, "LSC order requires mesh electrostatics");
  configure_electrostatics();
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::configure_electrostatics()
{
  try {
    if constexpr (CUDA) {
      if (!gpu) return;
      if (mesh_spacing > 0)
        gpu->set_mesh_electrostatics(mesh_spacing, mesh_order);
      else
        gpu->set_electrostatics_hook({});
    } else {
      if (!cpu) return;
      if (mesh_spacing > 0)
        cpu->set_mesh_electrostatics(mesh_spacing, mesh_order);
      else
        cpu->set_electrostatics_hook({});
    }
  } catch (const std::exception &e) {
    error->all(FLERR, "Cannot select LSC electrostatics: {}", e.what());
  }
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  if (narg != 3 + atom->ntypes || strcmp(arg[0], "*") || strcmp(arg[1], "*"))
    error->all(FLERR, "LSC pair_coeff requires * * model.json and one element per atom type");
  try {
    if constexpr (CUDA) {
      gpu = std::make_unique<MACELSCCuda>(arg[2], mixed ? "float32-mixed" : "float32");
      gpu->use_batched_reverse = true;
      cutoff = gpu->r_cut;
    } else {
      cpu = std::make_unique<MACELSC>(arg[2]);
      cutoff = cpu->r_cut;
      if (!cpu->can_evaluate_forces())
        error->all(FLERR, "LSC model needs electrostatic energy parameters; export it again");
    }
  } catch (const std::exception &e) {
    error->all(FLERR, "Cannot load LSC model: {}", e.what());
  }
  configure_electrostatics();
  comm_forward = 4 * (CUDA ? gpu->num_channels : cpu->num_channels);
  if constexpr (CUDA)
    comm_reverse = 4 * gpu->num_channels;
  else
    comm_reverse = 4 * cpu->num_channels;
  const auto &numbers = CUDA ? gpu->atomic_numbers : cpu->atomic_numbers;
  species.resize(atom->ntypes + 1);
  for (int t = 1; t <= atom->ntypes; ++t) {
    int z = 0;
    for (int k = 0; k < 118; ++k)
      if (strcmp(arg[t + 2], atomic_symbols[k]) == 0) {
        z = k + 1;
        break;
      }
    auto found = std::find(numbers.begin(), numbers.end(), z);
    if (z == 0 || found == numbers.end())
      error->all(FLERR, "LSC model does not contain element {}", arg[t + 2]);
    species[t] = int(found - numbers.begin());
  }
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::init_style()
{
  if (strcmp(update->unit_style, "metal"))
    error->all(FLERR, "pair_style symmetrix/mace/lsc requires units metal");
  if (domain->dimension != 3)
    error->all(FLERR, "pair_style symmetrix/mace/lsc requires a three dimensional box");
  if (force->newton_pair == 0)
    error->all(FLERR, "pair_style symmetrix/mace/lsc requires newton pair on");
  if (!atom->tag_enable) error->all(FLERR, "pair_style symmetrix/mace/lsc requires atom IDs");
  const int periodic = domain->xperiodic + domain->yperiodic + domain->zperiodic;
  if (periodic != 0 && periodic != 3)
    error->all(FLERR,
               "LSC supports nonperiodic clusters or boxes periodic in all three directions");
  if (mesh_spacing > 0 && periodic != 3)
    error->all(FLERR, "LSC mesh requires a box periodic in all three directions");
  if constexpr (CUDA) {
    if (!gpu) error->all(FLERR, "LSC pair_coeff must be set before running");
    if (gpu->precision != (mixed ? "float32-mixed" : "float32"))
      error->all(FLERR, "LSC precision changed; pair_coeff must follow pair_style");
  } else {
    if (!cpu) error->all(FLERR, "LSC pair_coeff must be set before running");
  }
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

template <bool CUDA> double PairSymmetrixMACELSC<CUDA>::init_one(int i, int j)
{
  if (!setflag[i][j]) error->all(FLERR, "All LSC pair coefficients must be set");
  return cutoff;
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::compute_forward()
{
  mesh_result = {};
  distributed_energy = 0;
  distributed_virial = {};
  distributed_stress = {};
  const int n = atom->nlocal;
  if (list->inum != n) error->all(FLERR, "LSC requires a full neighbor list for every atom");
  const int nall = n + atom->nghost;
  types.resize(nall);
  std::vector<std::int64_t> atom_ids(nall);
  for (int i = 0; i < nall; ++i) {
    types[i] = species[atom->type[i]];
    atom_ids[i] = atom->tag[i];
  }
  positions.resize(size_t(n) * 3);
  for (int i = 0; i < n; ++i) {
    double unwrapped[3];
    domain->unmap(atom->x[i], atom->image[i], unwrapped);
    for (int d = 0; d < 3; ++d) positions[size_t(i) * 3 + d] = unwrapped[d];
  }
  std::vector<int> ghost_senders;
  senders.clear();
  receivers.clear();
  vectors.clear();
  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = list->firstneigh[i][jj] & NEIGHMASK;
      double delta[3], rsq = 0;
      for (int d = 0; d < 3; ++d) {
        delta[d] = atom->x[i][d] - atom->x[j][d];
        rsq += delta[d] * delta[d];
      }
      if (rsq >= cutoff * cutoff || (atom->tag[j] == atom->tag[i] && rsq == 0)) continue;
      ghost_senders.push_back(j);
      receivers.push_back(i);
      for (double x : delta) vectors.push_back(x);
    }
  }
  try {
    const LSCGraph graph(types, ghost_senders, receivers, vectors, n, atom_ids);
    // Fold local periodic images; retain remote slots for standard per-atom energy return.
    senders = graph.feature_senders();
    auto evaluate = [&](auto &model) {
      model.prepare_graph(graph);
      model.compute_first_layer();
      forward_layer = 1;
      // Periodic images of locally owned atoms are published on the device.
      // Only multiple MPI processes need feature transport.
      if (comm->nprocs > 1) comm->forward_comm(this);
      model.complete_forward_exchange(1);
      model.compute_second_layer();
      forward_layer = 2;
      // Periodic images of locally owned atoms are published on the device.
      // Only multiple MPI processes need feature transport.
      if (comm->nprocs > 1) comm->forward_comm(this);
      model.complete_forward_exchange(2);
      model.compute_readouts();
    };
    if constexpr (CUDA)
      evaluate(*gpu);
    else
      evaluate(*cpu);
    forward_layer = 0;
  } catch (const std::exception &e) {
    error->one(FLERR, "LSC forward evaluation failed: {}", e.what());
  }
}

template <bool CUDA>
int PairSymmetrixMACELSC<CUDA>::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                                  int * /*pbc*/)
{
  try {
    const std::span<const int> slots(list, size_t(n));
    const std::span<double> values(buf, size_t(n) * (forward_layer == 3 ? 4 : comm_forward));
    if (forward_layer == 3) {
      if constexpr (CUDA)
        gpu->pack_potentials(slots, values);
      else
        cpu->pack_potentials(slots, values);
      return n * 4;
    }
    if constexpr (CUDA)
      gpu->pack_forward_features(forward_layer, slots, values);
    else
      cpu->pack_forward_features(forward_layer, slots, values);
  } catch (const std::exception &e) {
    error->one(FLERR, "Cannot pack LSC features: {}", e.what());
  }
  return n * comm_forward;
}

template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::unpack_forward_comm(int n, int first, double *buf)
{
  try {
    const std::span<const double> values(buf, size_t(n) * (forward_layer == 3 ? 4 : comm_forward));
    if (forward_layer == 3) {
      if constexpr (CUDA)
        gpu->unpack_potentials(first, values);
      else
        cpu->unpack_potentials(first, values);
      return;
    }
    if constexpr (CUDA)
      gpu->unpack_forward_features(forward_layer, first, values);
    else
      cpu->unpack_forward_features(forward_layer, first, values);
  } catch (const std::exception &e) {
    error->one(FLERR, "Cannot unpack LSC features: {}", e.what());
  }
}

template <bool CUDA>
std::vector<double> PairSymmetrixMACELSC<CUDA>::forward_features(int layer) const
{
  if constexpr (CUDA) {
    const auto values = gpu->download_forward_features(layer);
    const auto &array = std::get<std::vector<float>>(values.values);
    return {array.begin(), array.end()};
  } else {
    const auto array = cpu->forward_features(layer);
    return {array.begin(), array.end()};
  }
}

template <bool CUDA> std::vector<double> PairSymmetrixMACELSC<CUDA>::forward_node_energies() const
{
  if constexpr (CUDA) {
    auto values = gpu->download_local(false);
    return std::get<std::vector<double>>(values.at("local_node_energies").values);
  } else
    return cpu->local_node_energies;
}

template <bool CUDA> double PairSymmetrixMACELSC<CUDA>::forward_local_energy() const
{
  if constexpr (CUDA)
    return gpu->local_energy;
  else
    return cpu->local_energy;
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::compute_distributed_sources()
{
  compute_forward();
  try {
    auto evaluate = [&](auto &model) {
      model.compute_partial_charges(positions);
      reverse_mode = 1;
      comm->reverse_comm(this, 2);
      reverse_mode = 0;
      model.complete_charge_exchange();
    };
    if constexpr (CUDA)
      evaluate(*gpu);
    else
      evaluate(*cpu);
  } catch (const std::exception &e) {
    error->one(FLERR, "LSC partial charge accumulation failed: {}", e.what());
  }
}

template <bool CUDA>
int PairSymmetrixMACELSC<CUDA>::pack_reverse_comm(int n, int first, double *buf)
{
  int width = reverse_mode == 1 ? 2 : reverse_mode == 4 ? 3 : comm_reverse;
  try {
    auto pack = [&](auto &model) {
      std::span<double> values(buf, size_t(n) * width);
      if (reverse_mode == 1)
        model.pack_reverse_charges(first, n, values);
      else if (reverse_mode == 2 || reverse_mode == 3)
        model.pack_reverse_gradients(reverse_mode == 2 ? 2 : 1, first, n, values);
      else if (reverse_mode == 4)
        model.pack_reverse_forces(first, n, values);
      else
        throw std::invalid_argument("No active reverse communication phase");
    };
    if constexpr (CUDA)
      pack(*gpu);
    else
      pack(*cpu);
  } catch (const std::exception &e) {
    error->one(FLERR, "Cannot pack LSC reverse data: {}", e.what());
  }
  return n * width;
}

template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::unpack_reverse_comm(int n, int *list, double *buf)
{
  int width = reverse_mode == 1 ? 2 : reverse_mode == 4 ? 3 : comm_reverse;
  try {
    auto unpack = [&](auto &model) {
      std::span<const int> slots(list, size_t(n));
      std::span<const double> values(buf, size_t(n) * width);
      if (reverse_mode == 1)
        model.unpack_reverse_charges(slots, values);
      else if (reverse_mode == 2 || reverse_mode == 3)
        model.unpack_reverse_gradients(reverse_mode == 2 ? 2 : 1, slots, values);
      else if (reverse_mode == 4)
        model.unpack_reverse_forces(slots, values);
      else
        throw std::invalid_argument("No active reverse communication phase");
    };
    if constexpr (CUDA)
      unpack(*gpu);
    else
      unpack(*cpu);
  } catch (const std::exception &e) {
    error->one(FLERR, "Cannot add LSC reverse data: {}", e.what());
  }
}

template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::reverse_distributed(std::span<const double> pg,
                                                     const std::array<double, 9> &strain,
                                                     const std::array<double, 9> &cell, double el,
                                                     bool reduce_diagnostics)
{
  try {
    auto reverse = [&](auto &model) {
      model.begin_distributed_reverse(pg, strain, cell, field, el);
      reverse_mode = 2;
      comm->reverse_comm(this, comm_reverse);
      model.complete_reverse_gradient_exchange(2);
      model.reverse_distributed_second_layer();
      reverse_mode = 3;
      comm->reverse_comm(this, comm_reverse);
      model.complete_reverse_gradient_exchange(1);
      model.reverse_distributed_first_layer();
      model.finalize_distributed_forces(true);
      reverse_mode = 4;
      comm->reverse_comm(this, 3);
      model.complete_force_exchange();
      reverse_mode = 0;
      // Separate diagnostics: normal Pair tallies reduce rank contributions later.
      if (reduce_diagnostics) {
        MPI_Allreduce(&model.energy, &distributed_energy, 1, MPI_DOUBLE, MPI_SUM, world);
        MPI_Allreduce(model.virial.data(), distributed_virial.data(), 6, MPI_DOUBLE, MPI_SUM,
                      world);
        MPI_Allreduce(model.stress.data(), distributed_stress.data(), 6, MPI_DOUBLE, MPI_SUM,
                      world);
      }
    };
    if constexpr (CUDA)
      reverse(*gpu);
    else
      reverse(*cpu);
  } catch (const std::exception &e) {
    error->one(FLERR, "LSC distributed reverse failed: {}", e.what());
  }
}

template <bool CUDA> std::vector<double> PairSymmetrixMACELSC<CUDA>::owned_forces() const
{
  if constexpr (CUDA) {
    auto output = gpu->download_forces(false);
    return std::get<std::vector<double>>(output.at("forces").values);
  } else
    return cpu->forces;
}

template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::exchange_electrostatic_potentials(std::span<const double> values)
{
  try {
    auto exchange = [&](auto &model) {
      model.set_owned_potentials(values);
      forward_layer = 3;
      comm->forward_comm(this, 4);
      model.complete_potential_exchange();
      forward_layer = 0;
    };
    if constexpr (CUDA)
      exchange(*gpu);
    else
      exchange(*cpu);
  } catch (const std::exception &e) {
    error->one(FLERR, "LSC electrostatic potential exchange failed: {}", e.what());
  }
}

template <bool CUDA> std::vector<double> PairSymmetrixMACELSC<CUDA>::owned_density() const
{
  if constexpr (CUDA) {
    auto output = gpu->download_charges(false);
    return std::get<std::vector<double>>(output.at("density_coefficients").values);
  } else
    return cpu->density_coefficients;
}

template <bool CUDA> std::array<double, 3> PairSymmetrixMACELSC<CUDA>::source_dipole() const
{
  if constexpr (CUDA)
    return gpu->dipole;
  else
    return cpu->dipole;
}

template <bool CUDA> std::vector<double> PairSymmetrixMACELSC<CUDA>::density_potentials() const
{
  if constexpr (CUDA) {
    const auto values = gpu->download_electrostatic_potentials();
    return std::get<std::vector<double>>(values.values);
  } else {
    const auto values = cpu->electrostatic_potentials();
    return {values.begin(), values.end()};
  }
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom)
    error->all(FLERR, "LSC does not provide per-atom stress; use the global pressure tensor");
  const bool distributed = comm->nprocs != 1;
  if (distributed && mesh_spacing <= 0)
    error->all(FLERR, "Multi-rank LSC MD requires mesh electrostatics; select mesh spacing");
  if (distributed)
    compute_distributed_sources();
  else
    compute_forward();
  const int n = atom->nlocal;
  const std::array<double, 9> box = {
      domain->xprd, 0, 0, domain->xy, domain->yprd, 0, domain->xz, domain->yz, domain->zprd};
  const std::array<bool, 3> pbc = {bool(domain->xperiodic), bool(domain->yperiodic),
                                   bool(domain->zperiodic)};
  double total;
  const std::vector<double> *force_values;
  std::array<double, 6> model_virial;
  try {
    auto evaluate_stages = [&](auto &model) {
      if (distributed) {
        evaluate_distributed_mesh(false);
        return;
      }
      model.evaluate_charges(positions);
      model.evaluate_energy(box, pbc, "auto", field);
      model.reverse_charges();
      model.reverse_second_layer();
      model.reverse_first_layer();
      model.finalize_forces(false);
    };
    if constexpr (CUDA) {
      evaluate_stages(*gpu);
      total = gpu->energy;
      auto output = gpu->download_forces(false);
      force_buffer = std::move(std::get<std::vector<double>>(output.at("forces").values));
      force_values = &force_buffer;
      model_virial = gpu->virial;
    } else {
      evaluate_stages(*cpu);
      total = cpu->energy;
      force_values = &cpu->forces;
      model_virial = cpu->virial;
    }
  } catch (const std::exception &e) {
    error->all(FLERR, "LSC evaluation failed: {}", e.what());
  }
  // Distributed reverse already returns ghost forces to owners. Only add owned forces.
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) atom->f[i][d] += (*force_values)[size_t(i) * 3 + d];
  const auto &model_dipole = CUDA ? gpu->dipole : cpu->dipole;
  for (int d = 0; d < 3; ++d) pvector[d] = model_dipole[d];
  // These are rank contributions; LAMMPS reduces energy, virial and pvector.
  if (eflag_global) eng_vdwl += total;
  if (vflag_global)
    for (int k = 0; k < 6; ++k) virial[k] += model_virial[k];
  if (atom->q_flag && !eflag_atom) {
    // Updating q uses physical charge output and skips neural diagnostics.
    if constexpr (CUDA) {
      auto output = gpu->download_charges(false);
      const auto &charges = std::get<std::vector<double>>(output.at("partial_charges").values);
      std::copy_n(charges.data(), n, atom->q);
    } else {
      for (int i = 0; i < n; ++i) atom->q[i] = cpu->density_coefficients[size_t(i) * 4];
    }
  } else if (eflag_atom) {
    std::vector<double> local, rho, gradient, flux, formal;
    if constexpr (CUDA) {
      auto a = gpu->download_local(false), b = gpu->download_charges(false),
           c = gpu->download_forces();
      local = std::move(std::get<std::vector<double>>(a.at("local_node_energies").values));
      rho = std::move(std::get<std::vector<double>>(b.at("density_coefficients").values));
      flux = std::move(std::get<std::vector<double>>(b.at("edge_fluxes").values));
      gradient = std::move(std::get<std::vector<double>>(c.at("density_energy_gradient").values));
      const auto formal_values = gpu->download_parameter("formal_charges");
      for (float x : std::get<std::vector<float>>(formal_values.values)) formal.push_back(x);
    } else {
      local = cpu->local_node_energies;
      rho = cpu->density_coefficients;
      flux = cpu->edge_fluxes;
      gradient = cpu->density_energy_gradient;
      formal = cpu->formal_charges;
    }
    constexpr int cartesian[3] = {3, 1, 2};
    for (int i = 0; i < n; ++i) {
      if (atom->q_flag) atom->q[i] = rho[size_t(i) * 4];
      double value = local[i];
      for (int m = 0; m < 4; ++m) {
        double g = gradient[size_t(i) * 4 + m];
        if (m > 0) g -= field[m == 1 ? 1 : (m == 2 ? 2 : 0)];
        value += 0.5 * rho[size_t(i) * 4 + m] * g;
      }
      for (int d = 0; d < 3; ++d)
        value += field[d] *
            (formal[types[i]] * positions[size_t(i) * 3 + d] + rho[size_t(i) * 4 + cartesian[d]]);
      eatom[i] = value;
    }
    // compute pe/atom returns remote ghost contributions with standard reverse communication.
    if (eflag_atom)
      for (size_t e = 0; e < senders.size(); ++e) {
        double value = 0;
        for (int d = 0; d < 3; ++d) value += 0.5 * flux[e] * field[d] * vectors[3 * e + d];
        eatom[senders[e]] += value;
        eatom[receivers[e]] += value;
      }
  }
}

template <bool CUDA> double PairSymmetrixMACELSC<CUDA>::memory_usage()
{
  return double(
             (types.capacity() + senders.capacity() + receivers.capacity() + species.capacity()) *
             sizeof(int)) +
      double((positions.capacity() + vectors.capacity() + force_buffer.capacity()) *
             sizeof(double));
}

template <bool CUDA> void PairSymmetrixMACELSC<CUDA>::reset_grid()
{
  // LAMMPS calls every rank after rebalancing, including ranks with unchanged bounds.
  // Grid/FFT plan reconstruction is collective and must happen on all ranks together.
  distributed_mesh.reset();
}

template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::evaluate_distributed_mesh(bool reduce_diagnostics)
{
  if (mesh_spacing <= 0)
    error->all(FLERR, "Select mesh spacing before distributed LSC electrostatics");
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "Distributed LSC mesh needs three periodic directions");
  try {
    if (!distributed_mesh)
      distributed_mesh =
          std::make_unique<LSCDistributedMesh>(lmp, this, mesh_spacing, mesh_order, CUDA);
    std::vector<double> xyz(size_t(atom->nlocal) * 3);
    for (int i = 0; i < atom->nlocal; ++i)
      for (int d = 0; d < 3; ++d) xyz[3 * size_t(i) + d] = atom->x[i][d] - domain->boxlo[d];
    const std::array<double, 9> cell = {
        domain->xprd, 0, 0, domain->xy, domain->yprd, 0, domain->xz, domain->yz, domain->zprd};
    LSCElectrostaticsContext context;
    if constexpr (CUDA)
      context = gpu->electrostatics_context(cell, {true, true, true}, "auto");
    else
      context = cpu->electrostatics_context(cell, {true, true, true}, "auto");
    auto rho = owned_density();
    mesh_result = distributed_mesh->evaluate(xyz, rho, context);
    mesh_shape = distributed_mesh->shape;
    mesh_local_modes = distributed_mesh->local_modes;
    mesh_local_brick = distributed_mesh->local_brick;
    mesh_plan_generation = distributed_mesh->plan_generation;
    mesh_device_fft_calls = distributed_mesh->device_fft_calls;
    exchange_electrostatic_potentials(mesh_result.density_gradient);
    reverse_distributed(mesh_result.position_gradient, mesh_result.strain_derivative, cell,
                        mesh_result.energy, reduce_diagnostics);
  } catch (const std::exception &e) {
    error->one(FLERR, "LSC distributed mesh failed: {}", e.what());
  }
}
template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::pack_forward_grid(int, void *buf, int n, int *list)
{
  distributed_mesh->pack(buf, n, list);
}
template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::unpack_forward_grid(int, void *buf, int n, int *list)
{
  distributed_mesh->unpack(buf, n, list, false);
}
template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::pack_reverse_grid(int, void *buf, int n, int *list)
{
  distributed_mesh->pack(buf, n, list);
}
template <bool CUDA>
void PairSymmetrixMACELSC<CUDA>::unpack_reverse_grid(int, void *buf, int n, int *list)
{
  distributed_mesh->unpack(buf, n, list, true);
}

template class LAMMPS_NS::PairSymmetrixMACELSC<false>;
#ifdef SYMMETRIX_CUDA
template class LAMMPS_NS::PairSymmetrixMACELSC<true>;
#endif
