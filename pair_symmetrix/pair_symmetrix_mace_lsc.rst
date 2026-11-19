pair_style symmetrix/mace/lsc command
===================================

pair_style symmetrix/mace/lsc/cuda command
========================================

pair_style symmetrix/mace/lsc/cuda/kk command
===========================================

.. versionadded:: TBD

Syntax
""""""

.. code-block:: LAMMPS

   pair_style symmetrix/mace/lsc direct [field Ex Ey Ez]
   pair_style symmetrix/mace/lsc mesh spacing [order P] [field Ex Ey Ez]
   pair_style symmetrix/mace/lsc/cuda direct [float32-mixed] [field Ex Ey Ez]
   pair_style symmetrix/mace/lsc/cuda mesh spacing [order P] [float32-mixed] [field Ex Ey Ez]
   pair_style symmetrix/mace/lsc/cuda/kk mesh spacing [order P] [float32-mixed] [field Ex Ey Ez]
   pair_coeff * * model.json element1 element2 ...

* ``direct`` selects the complete-system reference evaluator (one MPI process).
* ``mesh spacing`` selects CPU or GPU mesh electrostatics with a target
  grid spacing in Angstrom and analytic forces and global stress. Requires
  periodicity in all three directions. CPU mesh requires the FFTW-enabled
  mesh build; GPU mesh uses cuFFT in the CUDA build.
* ``order P`` selects the B-spline order (default 8, supported range 4-10).
* ``float32-mixed`` selects Polar-style mixed neural GEMMs on CUDA; the default
  GPU neural mode is FP32. The CPU style uses float64.
* ``field Ex Ey Ez`` supplies the model's external-field vector in eV/(e Angstrom).
  Its energy convention follows the original LSC calculator: plus field dot dipole.
* One element symbol must be supplied for every LAMMPS atom type.

Examples
""""""""

.. code-block:: LAMMPS

   units metal
   pair_style symmetrix/mace/lsc/cuda direct float32-mixed
   pair_coeff * * water-stage2-energy.json H O
   compute mu all pair symmetrix/mace/lsc/cuda
   thermo_style custom step pe temp press c_mu[1] c_mu[2] c_mu[3]

Description
"""""""""""

Evaluate an exported local split charge MACE model, including local energy,
learned charge transfers, atomic dipoles and electrostatic energy. CUDA uses the
native specialized LSC evaluator. Electrostatics and physical reductions use
float64 in both GPU neural modes. The export must include the learned weights,
radial splines and electrostatic parameters accepted by the LSC architecture
validator.

Energy and global virial are returned directly. Per-atom energies sum to the
total energy. With ``atom_style charge``, predicted partial charges are written
to the atom ``q`` array; they are model outputs, not independent dynamical
variables. Do not add another Coulomb or kspace contribution to this potential.
The three additional ``compute pair`` values are Cartesian total dipole in
e Angstrom. Its polarization branch uses unwrapped formal-charge coordinates
and physical neighbor image vectors for charge transfers.

Restrictions
""""""""""""

The style requires three dimensions, ``units metal``, atom IDs and
``newton pair on``. Direct mode requires one MPI process and supports nonperiodic
clusters or boxes periodic in all three directions. Mesh mode supports one or
more MPI processes and requires periodicity in all three directions. It does not
support slabs, per-atom virials or binary restart storage of the model. Supply
``pair_style`` and ``pair_coeff`` again after reading a restart.

The mesh solver follows Polar's B-spline assignment, dipole Hessian gathering
and Fourier kernel/cell derivatives. It replaces only electrostatics: the native
LSC reverse pass still includes the dependence of learned charges and dipoles
on positions. Refining spacing or increasing spline order reduces the mesh
approximation error. Choose either ``direct`` or ``mesh``; direct is the default.

.. versionchanged:: TBD

Mesh mode supports normal multi-rank ``run`` commands. The pair evaluates owned
atoms and exchanges features, charge contributions, potentials and force
derivatives with neighboring processes. LAMMPS sums rank energy, global virial
and dipole contributions, and returns ghost per-atom energies to their owners.
Atom migration and sorting are handled by rebuilding the model graph from the
current atom and neighbor lists at each force evaluation. Mesh communication
plans are rebuilt when the box or processor subdomains change. The pair
responds to LAMMPS grid-reset notifications after processor rebalancing, so
all processes recreate communication plans together.

The GPU mesh uses double-precision cuFFT and mesh operations like Polar, while
neural layers retain their FP32 or mixed setting. One-process mesh operations
stay on the GPU. With multiple processes, mesh assignment, local Fourier
transforms, potentials and analytic derivatives run on the GPU; grid halos and
Fourier remaps use host-staged MPI transfers. Each process uses the CUDA device
visible to it. Assign devices through the job launcher for multi-GPU runs;
validation on two processes sharing one GPU does not establish multi-GPU scaling.

For multiple MPI processes, retain the original ``/cuda`` pair and launcher
without ``-sf kk``, which would otherwise select the one-process Kokkos variant.

The CUDA styles require a CUDA-enabled libsymmetrix build. The plain ``/cuda``
style also supports the staged MPI interface described above. The optional
``/cuda/kk`` style requires CUDA Kokkos and currently runs on one MPI process.
It uses the same model, local kernels and electrostatics; graph construction,
periodic-image mapping, forces and predicted charges stay on the GPU. It
supports both ``direct`` and ``mesh`` settings, clusters and periodic boxes,
global virial, dipoles and per-atom energies. It requires
``atom_modify map yes``, ``array`` or ``hash``. Models outside the validated LSC
architecture family are rejected.

.. versionadded:: TBD

Use the Kokkos launcher like Polar, for example::

   mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps

With ``-sf kk``, the existing ``pair_style symmetrix/mace/lsc/cuda`` command
selects the new device path automatically when it is built. Use the full
``symmetrix/mace/lsc/cuda/kk`` name for ``compute pair`` when selecting it
explicitly. Candidate neighbors are filtered at the physical cutoff every
force evaluation, including steps that reuse the LAMMPS neighbor list. Atom
sorting is handled by mapping model rows to the current local atom slots.

For one MPI process, local periodic ghost features are published by the native
evaluator and no H1/H2 MPI transport or host feature readback is performed.
This does not change the multi-process exchange sequence. The iterative build,
regression and benchmark entry points are documented in ``tools/lsc/README.md``.

The CUDA pair selects receiver-batched second-layer reverse convolution,
following Polar's reuse of receiver adjoints and shared neighbor geometry.
It uses the model's exported CG coefficients and retains FP32/mixed neural
arithmetic with double physical reductions. The native Python class also
exposes the legacy reverse path for numerical and performance comparison.

Updating the optional atom ``q`` array uses physical-only charge readback.
Per-atom energy requests use physical-only local/charge outputs and currently
retain the full force-diagnostic readback to obtain the density gradient.
Ordinary MD with global energy and virial output leaves neural diagnostics on
the device.
