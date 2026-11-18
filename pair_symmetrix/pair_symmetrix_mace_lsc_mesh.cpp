/* LSC distributed mesh, adapted from pair_symmetrix_mace_polar.cpp and
   the FFT/remap sequence in LAMMPS src/KSPACE/fft3d.cpp.
   LAMMPS, https://www.lammps.org/, Sandia National Laboratories.
   Copyright (2003) Sandia Corporation. GNU GPL; see pair_symmetrix/LICENSE. */
#include "pair_symmetrix_mace_lsc_mesh.h"
#ifdef SYMMETRIX_KSPACE
#include "comm.h"
#include "domain.h"
#include "fft3d.h"
#include "grid3d.h"
#include "lammps.h"
#include "neighbor.h"
#include "pair.h"
#include "remap.h"
#ifdef SYMMETRIX_CUDA
#include "mace_lsc_cuda_partition_mesh.hpp"
#endif
#include <algorithm>
#include <limits>
#include <utility>
using namespace LAMMPS_NS;
static_assert(sizeof(FFT_SCALAR) == sizeof(double),
              "LSC distributed mesh requires double FFT precision");
struct LSCDistributedMesh::Workspace {
  LAMMPS *lmp;
  Pair *owner;
  double spacing;
  int order;
  bool cuda;
  std::unique_ptr<Grid3d> grid;
  fft_plan_3d *fft = nullptr;
  int lo_in[3], hi_in[3], lo_out[3], hi_out[3];
  std::vector<double> key, buf1, buf2, brick, phi, what, theta;
  std::vector<int> owned, base;
  double *comm_mesh = nullptr;
#ifdef SYMMETRIX_CUDA
  std::unique_ptr<LSCCudaMeshPartition> gpu;
#endif
  Workspace(LAMMPS *l, Pair *p, double s, int o, bool c) :
      lmp(l), owner(p), spacing(s), order(o), cuda(c)
  {
#ifdef SYMMETRIX_CUDA
    if (cuda) gpu = std::make_unique<LSCCudaMeshPartition>();
#else
    if (cuda) throw std::invalid_argument("LSC distributed CUDA mesh is not built");
#endif
  }
  ~Workspace()
  {
    if (fft) fft_3d_destroy_plan(fft);
  }
  void configure(const LSCElectrostaticsContext &context, LSCDistributedMesh &out)
  {
    auto geom = lsc_mesh::geometry(context, spacing, order);
    auto domain = lmp->domain;
    std::vector<double> next(context.cell.begin(), context.cell.end());
    for (int d = 0; d < 3; ++d) {
      next.push_back(domain->boxlo[d]);
      // LAMMPS initializes only the active coordinate system's local bounds.
      next.push_back(domain->triclinic ? domain->sublo_lamda[d] : domain->sublo[d]);
      next.push_back(domain->triclinic ? domain->subhi_lamda[d] : domain->subhi[d]);
      next.push_back(geom.shape[d]);
    }
    next.push_back(lmp->neighbor->skin);
    next.push_back(lmp->comm->layout);
    if (next == key) return;
#ifdef SYMMETRIX_CUDA
    if (cuda && !key.empty()) gpu = std::make_unique<LSCCudaMeshPartition>();
#endif
    if (fft) {
      fft_3d_destroy_plan(fft);
      fft = nullptr;
    }
    grid.reset();
    grid = std::make_unique<Grid3d>(lmp, lmp->world, geom.shape[0], geom.shape[1], geom.shape[2]);
    grid->set_distance(.5 * lmp->neighbor->skin);
    grid->set_stencil_atom(order - 1 - order / 2, order / 2);
    grid->setup_grid(lo_in[0], hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2], lo_out[0],
                     hi_out[0], lo_out[1], hi_out[1], lo_out[2], hi_out[2]);
    int n1, n2;
    grid->setup_comm(n1, n2);
    if (!grid->ghost_adjacent())
      throw std::invalid_argument("LSC mesh stencil reaches beyond neighboring processes; use "
                                  "fewer ranks, a finer mesh or lower order");
    buf1.resize(std::max(n1, 1));
    buf2.resize(std::max(n2, 1));
    int nbuf = 0;
    fft = fft_3d_create_plan(lmp->world, geom.shape[0], geom.shape[1], geom.shape[2], lo_in[0],
                             hi_in[0], lo_in[1], hi_in[1], lo_in[2], hi_in[2], lo_in[0], hi_in[0],
                             lo_in[1], hi_in[1], lo_in[2], hi_in[2], 0, 0, &nbuf, 0, 0);
    if (!fft) throw std::runtime_error("LSC distributed FFT plan creation failed");
    owned.clear();
    int nx = hi_out[0] - lo_out[0] + 1, ny = hi_out[1] - lo_out[1] + 1;
    for (int z = lo_in[2]; z <= hi_in[2]; ++z)
      for (int y = lo_in[1]; y <= hi_in[1]; ++y)
        for (int x = lo_in[0]; x <= hi_in[0]; ++x)
          owned.push_back(((z - lo_out[2]) * ny + y - lo_out[1]) * nx + x - lo_out[0]);
    out.shape = geom.shape;
    out.local_modes = owned.size();
    out.local_brick = nx * ny * (hi_out[2] - lo_out[2] + 1);
    ++out.plan_generation;
    key = std::move(next);
  }
  void transform(int flag, LSCDistributedMesh &out)
  {
    // Use LAMMPS's existing pencil layouts and MPI remap plans. CUDA uses
    // cuFFT for each contiguous local axis batch with host MPI staging.
#ifdef SYMMETRIX_CUDA
    if (cuda) {
      auto input = reinterpret_cast<FFT_DATA *>(what.data());
      FFT_DATA *data = input;
      FFT_DATA *copy;
      auto remap = [&](FFT_DATA *from, FFT_DATA *to, remap_plan_3d *p) {
        remap_3d(reinterpret_cast<FFT_SCALAR *>(from), reinterpret_cast<FFT_SCALAR *>(to),
                 reinterpret_cast<FFT_SCALAR *>(fft->scratch), p);
      };
      auto local = [&](int total, int length) {
        if (total)
          gpu->fft_local({reinterpret_cast<double *>(data), size_t(total) * 2}, length, total,
                         flag);
      };
      if (fft->pre_plan) {
        copy = fft->pre_target == 0 ? input : fft->copy;
        remap(input, copy, fft->pre_plan);
        data = copy;
      }
      local(fft->total1, fft->length1);
      copy = fft->mid1_target == 0 ? input : fft->copy;
      remap(data, copy, fft->mid1_plan);
      data = copy;
      local(fft->total2, fft->length2);
      copy = fft->mid2_target == 0 ? input : fft->copy;
      remap(data, copy, fft->mid2_plan);
      data = copy;
      local(fft->total3, fft->length3);
      if (fft->post_plan) remap(data, input, fft->post_plan);
      out.device_fft_calls = gpu->fft_calls;
      return;
    }
#endif
    fft_3d(reinterpret_cast<FFT_DATA *>(what.data()), reinterpret_cast<FFT_DATA *>(what.data()),
           flag, fft);
  }
};
LSCDistributedMesh::LSCDistributedMesh(LAMMPS *lmp, Pair *owner, double spacing, int order,
                                       bool cuda) :
    workspace(std::make_unique<Workspace>(lmp, owner, spacing, order, cuda))
{
  lsc_mesh::require(lsc_electrostatics_finite(spacing) && spacing > 0, "spacing must be positive");
  lsc_mesh::require(order >= 4 && order <= 10, "order must be between 4 and 10");
}
LSCDistributedMesh::~LSCDistributedMesh() = default;
void LSCDistributedMesh::pack(void *v, int count, int *list)
{
  auto p = static_cast<double *>(v);
  for (int i = 0; i < count; ++i) p[i] = workspace->comm_mesh[list[i]];
}
void LSCDistributedMesh::unpack(void *v, int count, int *list, bool add)
{
  auto p = static_cast<double *>(v);
  for (int i = 0; i < count; ++i) {
    if (add)
      workspace->comm_mesh[list[i]] += p[i];
    else
      workspace->comm_mesh[list[i]] = p[i];
  }
}
LSCElectrostaticsResult LSCDistributedMesh::evaluate(std::span<const double> xyz,
                                                     std::span<const double> rho,
                                                     const LSCElectrostaticsContext &context)
{
  lsc_mesh::require(xyz.size() % 3 == 0 && rho.size() == xyz.size() / 3 * 4,
                    "expected owned positions[N,3],density[N,4]");
  for (double x : xyz) lsc_mesh::require(lsc_electrostatics_finite(x), "nonfinite position");
  for (double x : rho) lsc_mesh::require(lsc_electrostatics_finite(x), "nonfinite density");
  auto &w = *workspace;
  w.configure(context, *this);
  const int N = xyz.size() / 3, P = w.order;
  const auto g = lsc_partition::grid(context, w.spacing, P, w.lo_in, w.hi_in, w.lo_out, w.hi_out);
  w.brick.assign(g.n_out, 0);
  w.base.resize(size_t(N) * 3);
  w.theta.resize(size_t(N) * 9 * P);
#ifdef SYMMETRIX_CUDA
  if (w.cuda)
    w.brick = w.gpu->spread(g, xyz, rho);
  else
#endif
  {
    for (int i = 0; i < N; ++i) {
      if (!lsc_partition::weights(g, xyz.data() + 3 * size_t(i), w.base.data() + 3 * i,
                                  w.theta.data() + size_t(i) * 9 * P))
        throw std::invalid_argument("LSC mesh atom is outside the local grid stencil");
      double c, mu[3];
      lsc_partition::sources(g, rho.data() + 4 * size_t(i), c, mu);
      for (int z = 0; z < P; ++z)
        for (int y = 0; y < P; ++y)
          for (int x = 0; x < P; ++x)
            w.brick[lsc_partition::at(g, w.base.data() + 3 * i, x, y, z)] +=
                lsc_partition::assignment(P, w.theta.data() + size_t(i) * 9 * P, x, y, z, c, mu);
    }
  }
  w.comm_mesh = w.brick.data();
  w.grid->reverse_comm(Grid3d::PAIR, w.owner, 0, 1, sizeof(double), w.buf1.data(), w.buf2.data(),
                       MPI_DOUBLE);
  w.what.assign(std::max(size_t(2) * g.n_in, size_t(2)), 0);
  for (int k = 0; k < g.n_in; ++k) w.what[2 * k] = w.brick[w.owned[k]];
  w.transform(1, *this);
  lsc_partition::KernelResult kernel;
#ifdef SYMMETRIX_CUDA
  if (w.cuda)
    kernel = w.gpu->convolve(g, {w.what.data(), size_t(g.n_in) * 2});
  else
#endif
  {
    kernel.values.assign(w.what.begin(), w.what.begin() + size_t(g.n_in) * 2);
    auto mod = lsc_partition::moduli(g);
    for (int k = 0; k < g.n_in; ++k) {
      double e, s[9];
      double f = lsc_partition::kernel(g, k, mod.data(), w.what[2 * k], w.what[2 * k + 1], e, s);
      kernel.values[2 * k] *= f;
      kernel.values[2 * k + 1] *= f;
      kernel.energy += e;
      for (int d = 0; d < 9; ++d) kernel.strain[d] += s[d];
    }
  }
  std::copy(kernel.values.begin(), kernel.values.end(), w.what.begin());
  w.transform(-1, *this);
  w.phi.assign(g.n_out, 0);
  for (int k = 0; k < g.n_in; ++k) w.phi[w.owned[k]] = w.what[2 * k];
  w.comm_mesh = w.phi.data();
  w.grid->forward_comm(Grid3d::PAIR, w.owner, 0, 1, sizeof(double), w.buf1.data(), w.buf2.data(),
                       MPI_DOUBLE);
  LSCElectrostaticsResult result;
#ifdef SYMMETRIX_CUDA
  if (w.cuda)
    result = w.gpu->gather(g, w.phi);
  else
#endif
  {
    result.density_gradient.resize(size_t(N) * 4);
    result.position_gradient.resize(size_t(N) * 3);
    for (int i = 0; i < N; ++i) {
      double s[9];
      result.energy += lsc_partition::gather(
          g, w.phi.data(), w.base.data() + 3 * i, w.theta.data() + size_t(i) * 9 * P,
          rho.data() + 4 * size_t(i), result.density_gradient.data() + 4 * size_t(i),
          result.position_gradient.data() + 3 * size_t(i), s);
      for (int d = 0; d < 9; ++d) result.strain_derivative[d] += s[d];
    }
  }
  result.energy += kernel.energy;
  for (int d = 0; d < 9; ++d) result.strain_derivative[d] += kernel.strain[d];
  validate_lsc_electrostatics_result(result, N);
  return result;
}
#else
using namespace LAMMPS_NS;
struct LSCDistributedMesh::Workspace {};
LSCDistributedMesh::LSCDistributedMesh(LAMMPS *, Pair *, double, int, bool)
{
  throw std::runtime_error("LSC distributed mesh requires LAMMPS KSPACE");
}
LSCDistributedMesh::~LSCDistributedMesh() = default;
LSCElectrostaticsResult LSCDistributedMesh::evaluate(std::span<const double>,
                                                     std::span<const double>,
                                                     const LSCElectrostaticsContext &)
{
  throw std::runtime_error("LSC distributed mesh requires LAMMPS KSPACE");
}
void LSCDistributedMesh::pack(void *, int, int *) {}
void LSCDistributedMesh::unpack(void *, int, int *, bool) {}
#endif
