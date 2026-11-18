# LSC-MACE reference, exporter and native evaluator tests

These tests preserve the original MACE-SCF calculation and cover the LSC
architecture, learned-parameter/radial-spline export, native CPU/CUDA
evaluation and direct LAMMPS integration. The native class loads the export and evaluates the local interaction
layers, energy readouts, charge transfers, dipoles and electrostatic energy on a
supplied graph. Native CPU forces and stress now differentiate the complete energy,
including predicted charges and dipoles.
The CUDA backend supports the complete energy, forces, stress, charge transfers
and dipoles in FP32 and mixed neural precision. Electrostatics and physical
reductions use double precision. Direct CPU and CUDA LAMMPS pair styles and an
ASE adapter use these same evaluators. Tests compare against
the original CPU float64 PyTorch blocks and frozen ASE energy/charge/dipole outputs;
the original snapshots remain unchanged.

## Environments

Reference: `/mnt/home/noneill/Programs/venvs/mace-scf`, using the unchanged
`MACELocalSplitCharges` ASE calculator from
`/mnt/home/noneill/Programs/mace-scf-dev-niamh` at commit `4d1fe8a`.

Development: `/mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev`, cloned from the
reference environment, with test/build tools and this Symmetrix checkout
installed in editable mode. This environment and build are CPU-only. The
separate CUDA development environment and isolated LAMMPS worktree are described
below.

Activate the development environment with:

```bash
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
```

Run the initial CPU checks directly on a Rusty login node, as requested:

```bash
bash -l /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/run_tests.sh
```

The runner uses four CPU threads by default (`LSC_TEST_THREADS` overrides this),
checks installed requirements, runs the LSC reference and exporter suites plus
the existing native cubic-spline tests, and saves a log plus JUnit XML. It never
regenerates the saved predictions. A Slurm test wrapper is also available for
optional use.

This helper selects the matching MACE-SCF source and keeps the older
`/mnt/home/noneill/Programs/graph_electrostatics` checkout off `PYTHONPATH`.

## Frozen inputs and predictions

The chosen stage-2 checkpoint has SHA256
`47a453e928adaf66efe67f5b4e9436322cbdc77ca211aa610dbd55a650634a79`.
An independent copy is kept at
`/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-data/water_stage2.model`.
This model is ignored by Git. Set `LSC_REFERENCE_MODEL` to use another copy of
the **same** checkpoint; the checksum is verified.

Small geometries, prediction arrays, and their provenance manifest are kept in
`data/water_stage2` beside this README. The fixtures contain no DFT energy,
force, dipole, or stress labels. Expected results are predictions from the
original calculator, not comparisons against the training targets.

| Case | Source selection (zero-based) | Electrostatics | Outputs |
| --- | --- | --- | --- |
| dimer | seed1 `test.xyz`, frame 154, six atoms | `realspace` | energy, forces, total dipole, charges, atomic dipoles, density coefficients |
| bulk | `periodic_gen2_pbed3_filtered.extxyz`, frame 0, 192 atoms | `mixed_periodic` | the same outputs plus ASE stress |

The dimer's explicit 30 Angstrom nonperiodic cell matches the fallback that the
reference data loader supplies for the original cell-less input. Its positions
are unchanged. The bulk geometry and cell are unchanged. Coordinates are saved
with enough decimal digits to round-trip float64 values exactly.

Reference calculations use CPU, float64, no `torch.compile`, zero external
field, and the model's first head. The manifest records the actual architecture,
formal charges, self-interaction flag, software versions, source revision,
checksums, units, and electrostatic settings. Atomic density coefficients use
`[q, dipole_y, dipole_z, dipole_x]`; Cartesian dipoles use `[x, y, z]`. Stress
uses ASE's `[xx, yy, zz, yz, xz, xy]` order in eV/Angstrom^3. The periodic total
dipole is compared using the original LSC polarization convention.

## Run the tests

After activating the development environment, the focused suite on the login
node is:

```bash
python -m pytest /mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc -q
```

The login-node runner shown above also runs the existing cubic-spline tests to
check the native Symmetrix build. A Slurm wrapper is retained as an optional
alternative.

The tests check fixture/model integrity, reference dependency versions,
reproduction of the frozen outputs, charge conservation, and a directional
finite-difference force check on the dimer. The two full evaluations are cached
for the test module. A missing or changed checkpoint produces an actionable
failure rather than a silent skip.

Energy comparisons use an absolute tolerance, because these energies contain
large atomic reference offsets. The initial float64 tolerances are deliberately
separate from any future spline, float32, or mixed-precision accuracy targets.
The dimer force finite difference uses a looser tolerance than snapshot parity
to account for numerical differentiation.

## Architecture and local neural export: code walkthrough

The exporter follows the existing MACE-POLAR organization: the public
`extract_mace_data` function loads the checkpoint and dispatches to a separate
model-specific module. The same CLI writes the resulting dictionary as JSON.
The common JSON header keeps Polar's field names and `format_version: 3`.
LSC is distinguished by `model_type: "LocalSplitCharges"` and, at this stage,
`export_status: "parameters_and_splines"`. The generic native MACE reader rejects
this model type; the new `MACELSC` reader accepts and validates its local neural
parameters. The earlier architecture-only example remains a record of the first
export stage; generate a separate file for the numerical export.

1. **Dispatch before generic head conversion.**
   `symmetrix/source/symmetrix/extract_mace_data.py` sends `LocalSplitCharges`
   models to `extract_mace_data_lsc`. This happens before standard MACE head
   conversion, which can reconstruct a model without the LSC-specific blocks.
   The LSC exporter validates the checkpoint's single head itself. The actual
   saved head here is `Default`, even though a training configuration may name
   a different head.

2. **Check that we understand every supported block.**
   `symmetrix/source/symmetrix/extract_mace_lsc.py` provides
   `validate_lsc_architecture`. Its first supported family is the frozen
   checkpoint's two-layer, 128-channel model: residual MACE interactions,
   element-dependent correlation-order-3 products, scalar and vector hidden
   features, linear energy readouts, per-species formal charges, a
   `SumOxidationStateMixer`, and oxidation-dependent symmetric transfer sources.
   Edges use spherical harmonics through l=3; the density has charge and dipole
   components (l=0 and l=1). The radial basis is eight Bessel functions with a
   p=5 polynomial cutoff, applied before the radial networks, with no distance
   transform. Explicit exceptions reject unsupported architectures
   even when Python assertions are disabled. These constraints define this
   initial implementation target, not all models that MACE-SCF can train.

3. **Describe the LSC-specific data flow.**
   An oxidation-state embedding represents the fixed formal charges with
   Chebyshev basis functions and a learned scalar map; the initial mixer adds
   this embedding to the species embedding. Each layer's source uses its node
   features, the same fixed formal charges, and edge geometry to predict
   conservative charge transfers and local dipoles. A final scalar linear map
   produces each edge transfer, divided by 40 in the original implementation.
   Charge transferred to one atom is removed from its partner. The final charge
   is the formal charge plus all layer transfer contributions; atomic dipoles
   are also summed over layers. These form the density used by electrostatics.
   The export records the block types, basis sizes, normalization factors,
   transfer scale, formal charges, density width, and self-interaction setting,
   together with the learned local maps described below.

   The initial oxidation embedding stores range `[-6, 6]` and factor `1/6`,
   whereas each transfer source stores range `[-4, 4]` and factor `1/4`. The
   exporter reads these saved values independently; it does not substitute one
   global training default. Density coefficients retain the reference order
   `[q, dipole_y, dipole_z, dipole_x]`.

4. **Build the common header and species mapping.**
   `extract_mace_data_lsc` returns the same common fields as Polar: format and
   model identifiers, channel count, cutoff, angular limits, species, and atomic
   reference energies. Species are sorted as in the existing exporter, with
   `source_element_indices` retaining their positions in the checkpoint. This
   matters when selecting only some species from an element-dependent model.
   Formal charges and reference energies follow the selected species order.
   This LSC family has no scale/shift block, so its atomic reference energies
   are copied directly. Invalid species, unknown heads, and basis compression
   are rejected. Spline options control the radial tables described below.

5. **Write JSON using the existing CLI.**
   `symmetrix/source/symmetrix/cli/extract_mace.py` preserves the Polar invocation
   and version metadata, adding MACE-SCF, graph-longrange and e3nn versions for
   LSC exports. It prints that parameters and splines have been exported and
   native finite-cluster/fully-periodic energy is supported; derivatives follow. For example:

   ```bash
   module load gcc/13.3.0
   source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
   symmetrix_extract_mace \
     --model /mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-data/water_stage2.model \
     --chemical-symbols H O \
     --output /mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/water-stage2-parameters.json
   ```

6. **Test the interface before implementing evaluation.**
   `symmetrix/test/lsc/test_export.py` tests real-checkpoint dispatch, supported
   architecture metadata, rejection of unsupported variants, species mapping,
   a strict JSON round-trip, and the CLI. It also checks that the Polar dispatch
   arguments/version fields and standard MACE validation path are preserved.
   The generic MACE reader test verifies rejection of the LSC model type. These are exporter
   unit and integration tests; they are separate from the original ASE
   energy/force/dipole/stress regression tests. The numerical tests below verify
   exported local blocks; full-model native parity needs a native evaluator.

## Learned parameters: what changes and why

The implementation is in `symmetrix/source/symmetrix/extract_mace_lsc.py`.
`_learned_parameters_and_splines` adds numerical data after validating all
arguments and checks learned parameters for nonfinite values. The shared
checkpoint/calculator fixtures are in `symmetrix/test/lsc/conftest.py` so both
exporter test modules use the same reference-loading rules.

1. **Reuse Polar's feature layout and effective linear maps.**
   Polar's `_linear` and `_layout` helpers represent features in
   `[l][m][channel]` order, and store a channel matrix for each l plus a scalar
   bias. The helpers probe the original operation to include e3nn normalization.
   Copying raw e3nn parameter tensors without those factors would change the
   model. `node_embedding` stores the effective species-embedding rows; the
   oxidation embeddings also contain a `linear` map. The Chebyshev functions
   are T1 through T8. The saved range and factor transform the fixed formal
   charges before evaluating those functions.

2. **Export interactions, products, and energy readouts.**
   Each layer has `linear_up_t`, `linear_t`, species-indexed `skip_t`,
   `product_linear_t`, and `readout_t`. The neighbor-count division is already
   folded into `linear_t`; do not apply it twice in a future reader. Products
   use the existing `symmetric_contraction` helper, retaining the shared
   `Bt_spec`, `Bt_coefficients`, `Mt_spec`, and species-indexed `Mt_weights`
   representation. The original checkpoint's U tensors are used without
   replacing them or modifying the model.

3. **Export the LSC transfer maps.**
   Each `sources` entry contains the oxidation embedding, Polar-format
   oxidation `modulation` weights under the key `oxidation`, sender/receiver
   maps `linear1` and `linear2`, a `transfer_readout`, and a `multipoles` map.
   The stored `transfer_scale` remains a separate 1/40 multiplier; it is not
   folded into `transfer_readout`. `zero_charges` records that the local
   multipole map's scalar output is replaced by conservative transfers.
   Polar's `_modulation` helper supplies the oxidation tensor-product weights;
   the exporter preserves the caller's random state around its probing checks.

4. **Export convolution coefficients with their normalization.**
   `_uvu_spec` probes channelwise e3nn tensor-product paths. Each term stores
   `[q, a, b, w]` and a coefficient, meaning for every channel u:
   `out[q+u] += coefficient * x[a+u] * Y[b] * radial[w+u]`.
   Feature indices use the shared layout, harmonics use e3nn's order, and radial
   weights retain their original path/channel order. The source TP has two
   paths accumulating into one scalar output, which the existing `phi1_spec`
   helper does not represent. This LSC helper covers both interaction and
   transfer convolutions and rejects incompatible path layouts.

## Radial splines and knots

A **knot** is a sampled distance where a radial function's value and slope are
stored. With the default 1,024 knots between 0.5 and 6 Angstrom, spacing is about
0.0054 Angstrom. The native cubic spline uses values and slopes at neighboring
knots to interpolate between them. Slopes are needed for forces. More knots
increase accuracy and table memory; distances below `r_min` are outside this
table's domain.

`_radial_spline` follows Polar's factorization: spline the hidden functions
immediately before the radial network's final linear layer, and retain the
effective final projection as `radial_weights`. The actual normalized e3nn
network is evaluated during export. Thus the 512/1,280 interaction radial
outputs share 64 hidden spline functions per layer, and each 256-output transfer
network uses 32 hidden spline functions.

The Bessel features already include the polynomial cutoff in this LSC model.
There is no additional cutoff multiplier after its radial networks. Applying
another cutoff there would change the model. Values and derivatives are scaled
by `radial_spline_scale = 2*sqrt(pi)`, following the existing exporter convention.
To reconstruct an e3nn-normalized radial output in the tests, evaluate the
splines, multiply by `radial_weights`, and divide by `radial_spline_scale`.
Spline derivatives use `not-a-knot` at the lower end and zero slope at the
cutoff, as in Polar. The actual cutoff values also vanish.

Interaction tables use `radial_spline_values_t` and `radial_spline_derivs_t`.
The outer table dimension has length one, with `radial_pair_dependent: false`,
because these LSC radial functions are shared by every species pair. Each
source stores its own two-dimensional `[function][knot]` tables under
`radial_spline_values` and `radial_spline_derivs`. Source tables use the same
grid, scale, projection convention and endpoint conditions.

`symmetrix/test/lsc/test_export_parameters.py` reconstructs the learned maps,
convolution couplings, recursive products and complete charge-source blocks
from a JSON round-trip and compares them against the original PyTorch blocks.
It checks charge conservation, export without model/RNG mutation, and rejection
of nonfinite weights. Radial tests use the native `symmetrix.CubicSpline` at
held-out interval midpoints, including the first and final intervals, and
compare values and derivatives with the network and PyTorch autodiff. A
separate test checks that doubling the knot count reduces interpolation errors.
Maximum radial errors are recorded as JUnit properties (use
`-o junit_family=xunit1` to avoid pytest's property-compatibility warning).
These radial tolerances are separate from the eventual full-model force and
stress accuracy target.

## Differences from the existing MACE-POLAR exporter

| Part | Selected LSC architecture | Existing MACE-POLAR architecture |
| --- | --- | --- |
| Interactions | Standard residual | Nonlinear residual, with gates and density normalization |
| Products | Element-dependent weights | Element-agnostic weights |
| Charge sources | Fixed formal charges, conservative edge transfers and local dipoles | Multipole sources, Fukui and field-dependent charge-update blocks |
| Additional learned energy | Local MACE readouts | Also a local electron-energy block |
| Interaction radial dependence | Shared by all species pairs | Ordered-pair-dependent |

Both exporters use the same common model header, linear-map layout and spline
conventions, and reuse the recursive contraction representation. LSC needs its
own formal-charge embeddings, transfer maps and species-dependent product and
residual data. Polar-specific Fukui, field-update and electron-energy blocks
are absent from this LSC model.

## Native class, JSON reader and Python binding

The initial reader increment follows the existing `MACEPolar` organization.
It loads and validates the numerical export. The subsequent local-forward
increment below adds explicitly named local energy outputs.

1. **Declare the native class.**
   `/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.hpp`
   declares `MACELSC`, its constructor, model metadata and inspection fields.
   Bulk numerical storage stays private. The class owns its loaded data, so
   deleting the JSON file after construction does not invalidate the model.
   `can_evaluate` remains false for full-model evaluation; local forward
   capability is reported separately as `can_evaluate_local`.

2. **Load and validate the numerical export.**
   `/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.cpp`
   implements the reader. It accepts format version 3, model type
   `LocalSplitCharges`, and status `parameters_and_splines`. Architecture-only
   files and unrecognized future statuses are rejected. Unsupported or
   malformed data produce field-specific `ValueError` messages through the
   binding; an unreadable file produces `RuntimeError`.

   The reader stores maps using `MACEPolar::Linear` and oxidation modulation
   using `MACEPolar::Modulation`. It prepares transposed weights and feature
   offsets as Polar does. Spline values and derivatives are converted into the
   existing `CubicSplineSet<double>` representation, retaining the final radial
   projection separately. No learned parameters are renormalized or evaluated.

   Checks cover the supported two-layer, 128-channel architecture; species and
   source indices; linear-map dimensions and biases; convolution path bounds;
   product dependencies and species weights; shared spline tables, knot counts
   and projection shapes; oxidation embeddings and transfer settings; and
   electrostatic metadata/component ordering. Integer types and ranges are
   checked before conversion. Floating-point finiteness uses an exponent-bit
   check so the project's `-ffast-math` flags cannot disable validation.

3. **Expose the class to Python and build it.**
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/cpp/mace_lsc.cpp`
   binds the class as `symmetrix.MACELSC`. Model fields are read-only; Python
   lists returned by the binding are copies. It exposes embeddings and energy
   readout weights for loading checks, plus radial shapes in the order
   `[hidden functions, knots, projected outputs]`. Oxidation ranges/factors are
   listed as `[initial mixer, source 0, source 1]`.

   The new files are registered in
   `/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/CMakeLists.txt` and
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/CMakeLists.txt`.
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/cpp/symmetrix.cpp`
   registers the binding with the compiled Python module. The existing package
   initializer automatically exposes it; no calculator dispatch changes are
   needed for this reader-only increment. The existing CPU build helper
   rebuilds the development installation on the login node.

4. **Test the compiled reader.**
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_reader.py`
   generates a numerical export from the frozen checkpoint and loads it through
   C++. It checks metadata and representative learned arrays, the oxygen-only
   species selection, data ownership after file removal, and read-only Python
   properties. Malformed inputs cover missing fields, incompatible shapes,
   invalid dependency indices, inconsistent spline data, unsupported settings,
   JSON syntax errors and a missing file. These loading tests use eight knots;
   the separate numerical suite retains its 1,024-knot interpolation checks.

After rebuilding, inspect the real 1,024-knot example with:

```python
import symmetrix

model = symmetrix.MACELSC(
    "/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/water-stage2-parameters.json"
)
print(model.atomic_numbers)              # [1, 8]
print(model.interaction_radial_shapes)   # [[64, 1024, 512], [64, 1024, 1280]]
print(model.source_radial_shapes)        # [[32, 1024, 256], [32, 1024, 256]]
print(model.can_evaluate_local)          # True
print(model.can_evaluate_sources)        # True
print(model.can_evaluate)                # False
```

## Native local interaction layers and energy readouts

The implementation extends
`/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.hpp` and
`/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.cpp`.
The Python entry point is added in
`/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/cpp/mace_lsc.cpp`.
No Polar numerical source or reference prediction is changed.

1. **Accept a directed graph and validate its geometry.**
   `compute_local(node_types, edge_index, edge_vectors)` takes NumPy arrays
   shaped `[N]`, `[2,E]` and `[E,3]`. Node types index the exported species table.
   Edge row zero is the sender and row one is the receiver. The physical vector
   is `position[receiver] - position[sender] + periodic_image_shift`, matching
   MACE-SCF. The caller supplies the neighbor graph, including periodic images;
   this increment does not implement a new neighbor list or ASE calculator.
   Shape, integer/range, real-number and finite-coordinate checks reject bad
   inputs. Edge distances must lie in `[r_min, r_cut]`. Empty graphs and nodes
   without neighbors are supported. Edges are grouped by receiver, preserving
   their order within each row so parallel workers own separate output rows.

2. **Build initial features from species and formal charges.**
   The native code uses the saved initial oxidation range and factor, evaluates
   Chebyshev functions T1 through T8, and calls `MACEPolar::Linear::forward` for
   the learned embedding. Adding it to the stored species embedding reproduces
   `SumOxidationStateMixer`. These features depend on fixed formal charges.

3. **Prepare CPU radial tables and harmonics.**
   On the first call with edges, `prepare_radial` folds the last learned radial
   projection into the spline values and derivatives. This follows Polar's CPU
   strategy: cache the 512/1,280 projected interaction functions for inexpensive
   edge queries, retaining the compact export for future GPU work. A linear
   projection commutes with spline interpolation; this introduces no new
   approximation beyond roundoff. The cache is reused by subsequent calls.

   Native sphericart integral harmonics in xyz have the same ordering as this
   model's e3nn harmonics with the `[y,z,x]` permutation. The existing radial
   scale `2*sqrt(pi)` supplies e3nn's component normalization once. No extra
   radial cutoff or neighbor-count division is applied: those are already in
   the exported functions/maps. The exporter now explicitly rejects harmonic
   settings other than normalized directions and component normalization.

4. **Run both interaction/product/residual layers.**
   `compute_layer` applies Polar's linear operator to the node features,
   evaluates projected radial splines, combines them with angular features
   using the exported convolution coefficients, and sums messages at receivers.
   It then applies the output linear map. The recursive B/M contraction format
   follows the existing Polar implementation, selecting the LSC weights for
   each atom's species. The product linear map and species-dependent residual
   maps give H1 and H2. Residual operations are batched by species through the
   existing Polar linear operator.

5. **Evaluate local energy readouts.**
   Each hidden layer passes through its exported scalar linear readout. The
   local node energy is the atomic reference energy plus both layer readouts.
   `local_energy` sums these node energies. This excludes the separate
   electrostatic/field path. The C++ computation uses the JSON parameters and
   native operations without calling PyTorch.

6. **Expose explicit outputs for testing.**
   The binding returns the local total energy and exposes read-only buffers:
   `initial_features` `[N,128]`, `A0/A1` `[N,16,128]`, and `M0/M1/H1/H2`
   `[N,4,128]`, all flattened in Python as in Polar. `A` is the interaction
   output before products, `M` the contracted product before its linear map,
   and `H` the final layer features after the residual. `layer_node_energies`
   has shape `[2,N]`; `local_node_energies` has shape `[N]`. The full-model
   `can_evaluate` remains false while `can_evaluate_local` is true.

Tests are in
`/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_local.py`.
They run the original model's PyTorch embeddings, interactions, products and
readouts directly on the identical graph. Expected local quantities are
computed from the pinned checkpoint, not from DFT labels or the full-energy
snapshot (which includes electrostatics).

The knot test uses prescribed edge lengths on the 1,024-knot grid, checking
native arithmetic/layout separately from interpolation. Real dimer and bulk
tests check all intermediate features, both atomic readouts and local energies
at actual distances. Separate readout checks prevent large atomic reference
offsets from masking an incorrect local calculation. Other tests cover rigid
rotation, atom/edge permutations, oxygen-only export, repeated calls, empty
graphs, cutoff behavior and invalid graph inputs. A convergence test checks
that increasing from 64 to 1,024 knots improves local readout accuracy.

For the initial passing focused run, the measured differences were:

| Case | Atoms | Directed edges | Max atomic/layer readout error (eV) | Total local energy error (eV) |
| --- | --- | --- | --- | --- |
| dimer | 6 | 30 | 3.093712e-9 | 6.388291e-9 |
| bulk | 192 | 17440 | 3.921901e-9 | 1.614972e-7 |

These are local forward differences with 1,024-knot splines, not full LSC
energy/force/stress parity. JUnit records feature/readout/energy errors and
graph sizes. To reproduce the focused tests with visible error measurements:

```bash
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
python -m pytest /mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_local.py \
  -q -s -o junit_family=xunit1
```

## Native charge transfers and dipoles

This increment extends the existing native files:
`/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.hpp`,
`/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.cpp` and
`/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/cpp/mace_lsc.cpp`.
It uses the already exported charge-source parameters and follows Polar's
native linear/modulation/spline conventions. The original Polar source is
unchanged. LSC's transfer divergence replaces Polar's Fukui/field-update path.

1. **Share existing embedding and geometry work.**
   `oxidation_embeddings` factors out the initial native Chebyshev embedding
   code so the source blocks can use it as well. Each embedding uses its own
   saved range and factor: the initial block is `[-6,6]` with factor `1/6`,
   while the source blocks are `[-4,4]` with factor `1/4`. The initial local
   calculation retains its validated radii, harmonics and receiver edge groups
   for source evaluation. The fixed formal charges condition every source;
   predicted layer charges are accumulated into the density without feeding
   back into H1/H2. Thus computing both local layers before their source maps
   gives the same result as the original interleaved forward loop.

2. **Evaluate each source using existing native operators.**
   `compute_source` embeds formal charges and calls `MACEPolar::Modulation` on
   that layer's H features. `MACEPolar::Linear` supplies the sender and receiver
   feature maps. The radial source projection is cached as native spline
   tables through the same `prepare_radial` helper used by local layers.
   The source tensor-product coefficients combine the sum of sender/receiver
   mapped features with edge harmonics and radial weights. Its scalar readout
   is multiplied by the saved transfer scale `1/40` exactly once.

3. **Build conservative charges and local atomic dipoles.**
   For each source layer, the transfer contribution at atom i is the sum of
   incoming edge transfers minus the sum of outgoing transfers. Receiver and
   sender edge groups let each parallel worker own an atom's accumulation.
   The multipole linear map gives local dipoles; its unused scalar output is
   replaced by this transfer divergence, matching the checkpoint's
   `zero_charges` source block. The final charge is the fixed formal charge
   plus both layer transfer contributions. Atomic dipoles are summed over
   layers. Internal density order stays `[q,dy,dz,dx]`; public atomic dipoles
   are Cartesian `[x,y,z]`.

4. **Reproduce the LSC polarization convention.**
   The total Cartesian dipole is
   `sum_i formal_charge_i * position_i + sum_edges flux_e * image_vector_e + sum_i atomic_dipole_i`.
   `flux_e` is the sum of both layers' directed transfers. All directed edges
   contribute once; there is no extra factor of 1/2. In nonperiodic geometry,
   the transfer term also equals the transfer-charge position moment. Across
   periodic boundaries, the image-vector formula retains the original model's
   formal-charge polarization branch. Replacing it with predicted charges
   times wrapped positions would change the model. Charges use e; atomic and
   total dipoles use e*Angstrom, as in the saved reference outputs.

5. **Expose outputs with consistent graph validation.**
   The binding shares graph conversion/validation between `compute_local` and
   `compute_charges`. The latter additionally accepts finite Cartesian positions
   shaped `[N,3]`. It recomputes local features for the supplied graph, then
   returns the total dipole and stores read-only outputs:

   | Field | Shape after reshaping | Convention |
   | --- | --- | --- |
   | `source_features` | `[2,N,512]` | Conditioned layer features in `[l][m][channel]` order |
   | `layer_edge_fluxes` | `[2,E]` | Directed transfers after the 1/40 scale |
   | `layer_density_coefficients` | `[2,N,4]` | Transfer charges and local dipoles, without formal charges |
   | `density_coefficients` | `[N,4]` | Final `[q,dy,dz,dx]` |
   | `partial_charges` | `[N]` | Final charges |
   | `partial_dipoles` | `[N,3]` | Cartesian atomic dipoles |
   | `edge_fluxes` | `[E]` | Sum of layer transfers |
   | `dipole` | `[3]` | Total Cartesian polarization dipole |

   `formal_charge_dipole`, `transfer_dipole` and `atomic_dipole_sum` expose the
   three terms for inspection. The existing flat-vector binding style is
   retained. `can_evaluate_sources` is true; `has_charge_results` becomes true
   after a successful charge calculation, including a valid empty graph.
   A subsequent local-only calculation clears the old charge outputs and
   resets that result flag. Full-model `can_evaluate` remains false because
   force/stress derivatives are pending. Native energy is now available as described below.

6. **Compare independently with PyTorch blocks and frozen ASE outputs.**
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_sources.py`
   tests every source layer at spline knots against the original PyTorch
   source blocks, including conditioning, transfers and multipoles. The real
   dimer/bulk tests compare final charges, atomic dipoles, density coefficients
   and total dipoles directly against the saved original ASE predictions.
   Those charge/dipole outputs are determined by the LSC source forward path,
   so their validation does not require native electrostatic energy yet.

   Other tests check conservation, rotation/translation covariance, atom/edge
   permutations, oxygen-only export, isolated atoms, cutoff behavior, empty
   graphs, repeated calls, result invalidation and malformed positions/graphs.
   A boundary-crossing test explicitly distinguishes the flux polarization
   from the wrapped-charge position moment and checks its formal-charge branch
   shift. A knot-count convergence test checks improved charge accuracy.

The first focused run passed all 16 tests. Its differences from frozen ASE were:

| Case | Max charge error (e) | Max atomic dipole error (eA) | Max total dipole component error (eA) | Total-charge conservation error (e) |
| --- | --- | --- | --- | --- |
| dimer | 5.503358e-10 | 2.924944e-10 | 1.685352e-10 | 2.220446e-16 |
| bulk | 4.423231e-10 | 3.618260e-10 | 1.129431e-9 | 2.220446e-16 |

To run the focused tests with printed comparison errors:

```bash
module load gcc/13.3.0
source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
python -m pytest /mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_sources.py \
  -q -s -o junit_family=xunit1
```

The runtime entry point is:

```python
dipole = model.compute_charges(node_types, edge_index, edge_vectors, positions)
charges = model.partial_charges
atomic_dipoles = model.partial_dipoles  # flat [N,3], Cartesian
```

## Native electrostatic and total energy

1. **Export the electrostatic numerical contract.**
   `_electrostatic_energy_parameters` in
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/symmetrix/extract_mace_lsc.py`
   saves `electrostatics.energy_parameters` (nested format version 1). This
   contains the installed reference library's field constant, saved Gaussian
   Fourier amplitudes/width, self overlaps, and real-space displacement buffers.
   The outer format stays version 3 with `parameters_and_splines`, consistent
   with the existing reader. Exports from earlier stages still support local
   and source evaluation, but need re-exporting for `can_evaluate_energy`.
   The frozen checkpoint, snapshots and earlier parameter export are preserved.

2. **Evaluate finite-cluster electrostatics with the original convention.**
   `realspace_energy` in
   `/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.cpp`
   represents each charge/dipole density by four charges: one at the atom and
   three displaced along Cartesian axes. The displacement is the checkpoint's
   0.02 Angstrom setting. Pairs belonging to distinct original atoms interact
   through `erf(r/(2*sigma))/(r+1e-6)`, including the reference's denominator
   regularization. Unique atom pairs account for both half-weighted directed
   reference contributions. Gaussian self overlaps are added when enabled.
   This deliberately reproduces the reference finite-difference approximation;
   an ideal analytic charge/dipole formula would give a different baseline.

3. **Evaluate fully periodic electrostatics through reciprocal space.**
   `reciprocal_cell` forms the reciprocal lattice for any nonsingular 3D cell.
   `reciprocal_energy` enumerates the reference half-space with the saved k
   cutoff, evaluates Gaussian charge/dipole Fourier coefficients, and sums
   `field_constant / volume * abs(structure_factor)**2 / k**2`.
   The zero vector contributes zero and is omitted. The half-space accounts
   for the usual 1/2 as in Polar. Reciprocal Gaussian self interactions are
   already included; if disabled their overlap energy is subtracted. OpenMP
   workers own individual k terms, with a deterministic final sum. The code
   avoids allocating the PyTorch atom-by-k-by-multipole tensor intermediates.
   It uses Polar's Fourier conventions, with LSC's saved density normalization.

4. **Combine energies and expose explicit capabilities.**
   The native `compute_energy` recomputes local layers and charge sources, then
   returns `energy = local_energy + electrostatic_energy + field_energy` (eV),
   where `field_energy = dot(dipole, external_field)` matches the reference sign.
   The external field defaults to zero (components in eV/(e*Angstrom)).
   `has_energy_results` is true after success; subsequent local/source calls
   clear the energy outputs. `can_evaluate_energy` reports the data capability;
   `can_evaluate` remains false until force/stress support is complete.

   The binding in
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/cpp/mace_lsc.cpp`
   validates numeric inputs, cell shape, boolean PBC flags and finite values,
   then releases the GIL for the C++ calculation. A model instance has mutable
   result/work buffers, like the existing native interface; use separate
   instances for simultaneous evaluations.

   ```python
   total = model.compute_energy(
       node_types, edge_index, edge_vectors, positions,
       cell, pbc, pbc_handling="mixed_periodic",  # bulk
       external_field=[0.0, 0.0, 0.0],
   )
   coulomb = model.electrostatic_energy
   ```

   Supported modes are `realspace` (ignores periodic images, as the reference
   does), `pbc`/`mixed_periodic` for three periodic directions, and `auto`
   (realspace for nonperiodic configurations, reciprocal for fully periodic).
   An omitted mode uses the exported checkpoint default, which is `realspace`
   here: bulk callers must select the periodic mode or `auto`. Partial PBC,
   slab corrections and mixed-periodic molecule-in-box corrections are not
   implemented and are rejected, rather than substituted with realspace.

5. **Test total energy and the electrostatic formula separately.**
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_energy.py`
   compares complete native dimer/bulk energies with the unchanged original
   ASE snapshots (absolute tolerance 2e-6 eV, accounting for splines). Another
   comparison supplies the native density to the original graph-longrange
   PyTorch energy class, isolating energy arithmetic (tolerance 2e-9 eV).
   Tests also cover self terms, an analytic isolated charge pair, external
   fields, translation and lattice wrapping, a triclinic cell, empty/repeated
   calculations, legacy exports and malformed numerical data/inputs.

   ```bash
   python -m pytest /mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_energy.py \
     -q -s -o junit_family=xunit1
   ```

The first focused run passed **27 tests** in 43.92 seconds. The full LSC and
existing cubic-spline regression passed **181 tests** in 232.38 seconds, with
zero failures or skips (60 upstream dependency warnings).

| Case | Total energy error versus frozen ASE (eV) | Electrostatic error with identical density (eV) |
| --- | --- | --- |
| dimer, 6 atoms | 6.412847e-9 | 2.327027e-13 |
| bulk, 192 atoms | 1.619919e-7 | 3.191891e-16 |

Evidence is saved under `.lsc-reference-logs`: `native-energy-pytest.log/xml`,
`pytest-login-20261116T235304Z.log/xml`, `native-energy-validation.json`, and
`native-energy-build.log`. The separate energy-capable model export is
`water-stage2-energy.json`; the earlier parameter export remains unchanged.

The optional `tools/lsc/benchmark_energy.py` compares five warm CPU float64
energy-only calls with the original uncompiled PyTorch model, four threads for
both. It excludes model loading and neighbor construction. Native receives
edge vectors; the original forward constructs them from the prebuilt graph.
Timings and all samples are saved in `native-energy-benchmark.json`.
These shared-login-node timings are illustrative, not GPU/MD benchmarks.

This is a CPU float64 energy implementation. Native linear/modulation/product
operators, preprojected radial splines and compiled parallel loops remove much
of the repeated tensor setup and intermediate allocation in the reference.
It does not yet include LSC GPU kernels, force/stress backpropagation or a
LAMMPS pair style, and cannot establish Polar-like GPU/MD speedup factors.
Forces must differentiate the predicted sources as well as electrostatics.

## Native analytic forces and stress

This stage extends the existing C++ LSC evaluator. The energy-capable JSON from
`water-stage2-energy.json` already contains every required numerical parameter,
including radial spline derivatives; no new export format or baseline is needed.
The frozen checkpoint and ASE predictions remain unchanged. The code uses native
analytic derivatives, with no PyTorch calls or numerical differentiation during
inference. Finite differences are used only to test the implementation.

1. **Differentiate electrostatics with density held independent.**
   `reverse_realspace` and `reverse_reciprocal` in
   `/mnt/home/noneill/Programs/symmetrix-dev/libsymmetrix/source/mace_lsc.cpp`
   compute derivatives with respect to density coefficients, positions and
   symmetric strain. An adjoint here means the derivative of total energy with
   respect to an intermediate quantity. The density adjoint is the input to
   the backward pass through the learned sources; holding predicted density
   fixed for the complete force calculation would omit that contribution.

   Realspace reverses the same four displaced-charge representation used by
   the forward energy. It differentiates both the erf screening and the
   `r+1e-6` denominator, then maps charge adjoints back to `[q,dy,dz,dx]` through
   the saved displacement setting. Self-energy derivatives are included.
   Cartesian displacement buffers stay fixed under strain, matching MACE-SCF.

   Reciprocal space reuses the forward half-space enumeration. Its reverse
   differentiates Fourier charge/dipole factors and phases for forces. For
   homogeneous strain, positions and the cell deform together, so the phase
   `k dot r` stays invariant. The remaining strain terms come from the Gaussian
   basis, dipole projection, inverse squared k norm and inverse volume. This
   follows Polar's reciprocal-space virial reasoning with LSC normalization.
   Self-energy subtraction is reversed when self interactions are disabled.
   Each parallel worker owns one atom's accumulation, avoiding atomics and
   atom-by-k tensor storage.

2. **Reverse the LSC sources, including conservative charge transfers.**
   `reverse_source` sends local dipole adjoints through
   `MACEPolar::Linear::reverse`. The unused local monopole output receives no
   gradient, since the forward source replaces it with transfer divergence.
   A directed transfer has adjoint `density_adj[receiver,q] -
   density_adj[sender,q]`, plus its external-field polarization contribution.
   The saved transfer factor `1/40` is applied once before reversing its
   scalar readout.

   The source convolution then propagates to both endpoint feature maps and
   to its radial/angular functions. Sender and receiver edge groups provide
   deterministic, atom-owned gathers. `MACEPolar::Modulation::reverse`
   propagates back to that layer's hidden features. Formal oxidation states
   are fixed species data, so their embeddings have no position derivative.
   Forward sender/receiver maps are retained for the reverse computation.

3. **Reverse both local interaction/product layers.**
   `reverse_layer` starts from the hidden-feature derivative, which includes
   both that layer's energy readout and charge-source contributions. It uses
   Polar's linear reverse operation for product/residual/output maps, with
   the original LSC species-dependent weights. The recursive B products are
   traversed in reverse order, applying the product rule as in Polar. A single
   atom's basis is reconstructed from saved A to avoid an all-atom B cache.

   Convolution derivatives propagate to sender features, radial splines and
   spherical harmonics. `CubicSplineSet::evaluate_derivs` supplies derivatives
   of the actual interpolated radial functions; sphericart supplies angular
   gradients. Their Cartesian contributions are combined by the chain rule.
   Layer 2 is reversed first and adds to the accumulated H1 derivative; then
   layer 1 is reversed. Initial species/formal-charge embeddings are constant.
   Lifted forward features are retained, reusing buffer capacity between calls.

4. **Assemble forces, virial and ASE stress.**
   `MACELSC::compute_energy_forces` first runs the existing energy calculation,
   then performs the reverse pass. For `edge_vector = r_receiver - r_sender`
   (including periodic images), its energy derivative contributes a negative
   force to the receiver and a positive force to the sender. Direct Coulomb
   position forces are added separately. External-field derivatives include
   formal-charge positions, predicted dipoles, transfer values and image vectors,
   preserving the original LSC polarization branch.

   Strain accumulates the edge-vector outer products plus the explicit Coulomb
   and formal-charge field terms, then is symmetrized as in MACE's displacement
   calculation. Periodic Coulomb strain already includes the joint position/cell
   response, so its position term is not counted twice. ASE stress is
   `+dE/dstrain / volume`; the virial has the opposite sign.

   | Output | Shape after reshaping | Convention |
   | --- | --- | --- |
   | `forces` | `[N,3]` | Total Cartesian force, eV/Angstrom |
   | `edge_forces` | `[E,3]` | Negative derivative with respect to each image vector |
   | `position_forces` | `[N,3]` | Direct Coulomb and formal-charge field forces |
   | `density_energy_gradient` | `[N,4]` | Density adjoint in `[q,dy,dz,dx]` order |
   | `strain_derivative` | `[3,3]` | Symmetric total energy derivative, eV |
   | `electrostatic_strain_derivative` | `[3,3]` | Coulomb contribution at fixed density |
   | `virial` | `[6]` | Polar `[xx,yy,zz,xy,xz,yz]`, eV |
   | `stress` | `[6]` | ASE `[xx,yy,zz,yz,xz,xy]`, eV/Angstrom^3 |

5. **Expose a shared, validated Python interface.**
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/source/cpp/mace_lsc.cpp`
   shares graph/geometry validation between energy-only and derivative calls,
   and releases the GIL for native computation. Stress additionally requires a
   nonsingular cell. Supported boundary modes remain finite `realspace` and
   fully periodic `pbc`/`mixed_periodic`, plus `auto`. Partial-PBC corrections,
   slab and molecule-in-box modes remain unsupported.

   ```python
   energy = model.compute_energy_forces(
       node_types, edge_index, edge_vectors, positions, cell, pbc,
       pbc_handling="mixed_periodic",  # periodic bulk; realspace for the dimer
       external_field=[0.0, 0.0, 0.0],
       compute_stress=True,
   )
   forces = np.asarray(model.forces).reshape(-1, 3)
   stress = np.asarray(model.stress)  # ASE ordering
   ```

   `has_force_results` and `has_stress_results` distinguish completed outputs.
   A subsequent local, charge-only or energy-only call clears derivative buffers
   and flags. Stress defaults to off; its output is zeroed when not requested.
   The virial is computed with the force calculation. `can_evaluate_forces`,
   `can_evaluate_stress` and `can_evaluate` are true for energy-capable exports;
   legacy exports missing electrostatic parameters require re-exporting.
   Earlier capability tests were updated accordingly; their numerical checks
   and all frozen expected arrays are unchanged. The CLI describes the new
   native energy/force/stress capability.

6. **Validate independently at several levels.**
   `/mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_derivatives.py`
   checks complete dimer forces and bulk forces/stress against frozen original
   ASE predictions. Absolute tolerances are 1e-5 eV/Angstrom for forces and
   5e-8 eV/Angstrom^3 for stress, allowing interpolation-derivative error.

   Separate tests compare fixed-density electrostatic adjoints against original
   graph-longrange PyTorch autograd, with self interactions on and off. Native
   force directional differences and all six bulk strain components are checked
   against native energies. Fixed atomic reference energies are omitted from
   these differences to avoid cancellation; their derivative is exactly zero.
   Steps stay inside the radial spline domain and away from neighbor changes.
   External-field derivatives are checked against the full original model.

   Further checks cover net force, periodic force/stress rotation covariance,
   translation, atom/edge ordering, an analytic isolated charge pair, empty
   graphs, legacy exports, invalid inputs and stale-result clearing. A 64- versus
   1,024-knot comparison verifies improved force accuracy with spline refinement.

   ```bash
   module load gcc/13.3.0
   source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-dev/activate-lsc-dev.sh
   python -m pytest /mnt/home/noneill/Programs/symmetrix-dev/symmetrix/test/lsc/test_native_derivatives.py \
     -q -s -o junit_family=xunit1
   ```

The focused derivative suite passed **26 tests** with no failures or skips.
The full regression passed **207 tests** with no failures or skips. Exact run
summaries, warnings, timings and all numerical test properties are recorded in
`.lsc-reference-logs/native-derivatives-pytest.log/xml`,
`pytest-login-20261117T001958Z.xml`, and `native-derivatives-validation.json`.

| Case | Maximum force error versus frozen ASE (eV/Angstrom) | RMS force error (eV/Angstrom) | Maximum stress error (eV/Angstrom^3) |
| --- | --- | --- | --- |
| dimer, 6 atoms | 1.357096e-6 | 7.096122e-7 | not requested |
| bulk, 192 atoms | 2.345443e-6 | 7.108432e-7 | 3.174431e-9 |

Independent electrostatic adjoints agree with original PyTorch autograd within
3.82e-12 for density coefficients, 1.30e-13 eV/Angstrom for positions, and
2.76e-13 eV for strain. Native directional force differences are 1.50e-8 and
2.40e-9 eV/Angstrom for dimer/bulk; all six bulk stress finite differences are
within 4.0e-11 eV/Angstrom^3. These checks distinguish analytic reverse errors
from the finite spline interpolation error relative to the original model.

This completes the CPU native energy/force/stress path for the supported model
family and boundary conditions. The public interface operates on supplied graphs;
the generic ASE wrapper and a LAMMPS pair style are separate integration work.
LSC GPU kernels and end-to-end LAMMPS benchmarks remain subsequent stages. The
previous CPU energy timing excludes forces, stress, neighbor building and LAMMPS.


## LSC CUDA class and parameter loading

This stage adds GPU model loading, not numerical LSC inference. `MACELSCCuda`
reports `can_evaluate=False` and has no energy/force entry points. FP32 and
`float32-mixed` are the only accepted GPU modes. CPU `MACELSC` remains FP64.
The original energy-capable export is reused without retraining or re-exporting.

1. **Share the validated host snapshot.**
   `mace_lsc_model.hpp` and the loading portion of `mace_lsc.cpp` separate file
   parsing from parameter validation. `read_validated_lsc_model` parses once,
   validates that exact snapshot through the existing CPU reader, and returns
   the snapshot for backend packing. It does not reopen the file after validation.
   CPU numerical operations are unchanged. The CUDA loader additionally checks
   Polar's shared L=1 product tables, avoiding an incompatible specialization.

2. **Prepare backend-specific arrays on the host.**
   `mace_lsc_cuda_parameters.cpp` packs effective linear maps, source maps,
   fixed-formal-charge embeddings, species-dependent residual/product weights,
   convolution specifications and electrostatic constants. Numerical parameters
   are converted to FP32 with explicit range checks. Atomic reference energies
   retain double storage, as in Polar; this does not add an FP64 neural GPU mode.
   Integer specifications use int32. Host packing can be tested in CPU builds.

   Initial and source conditioning embeddings are precomputed per species using
   their independent saved oxidation ranges/factors. These quantities depend on
   fixed formal charges, not positions. Geometry-dependent inference remains for
   later GPU kernels. Product weights retain a species axis; they cannot use
   Polar's species-independent weight lookup unchanged.

3. **Prepare radial splines and optional mixed representations.**
   The loader folds the final radial projection into the spline tables once,
   following Polar's loader. Cubics use centered normalized interval coordinate
   `u = (r-r_min)/h - interval - 0.5`; derivatives later include `1/h`.

   | Parameter | Shape | Storage |
   | --- | --- | --- |
   | `*.weights.l` | `[input_channels,output_channels]` | FP32 |
   | `M0_weights`, `M1_weights` | `[species,29,128]` | FP32 |
   | `radial_0.coefficients` and source radial coefficients | `[interval,4,output]` | FP32 |
   | `radial_1.coefficients`, ordinary mode | `[interval,4,output]` | FP32 |
   | `radial_1.c01`, mixed mode | `[interval,2,output]` | FP32 |
   | `radial_1.c23`, mixed mode | `[interval,output,2]` | FP16 pairs |

   Mixed mode follows Polar's matrix-wide power-of-two scaling and FP16 high/low
   weight decomposition. Eligible matrices retain FP32 weights plus packed
   forward and reverse high/low arrays. This preserves an ordinary GEMM fallback.
   Second-layer c2/c3 coefficients are range-checked before FP16 conversion;
   first-layer and source splines retain FP32. CPU packing implements IEEE
   binary16 round-to-nearest-even and is checked independently against NumPy.
   Accuracy of complete mixed-precision inference is not established at this
   loading stage; it will require the numerical kernels and model comparisons.

4. **Own CUDA resources and upload once.**
   `mace_lsc_cuda.hpp/.cpp` define `MACELSCCuda`. It owns a CUDA stream, cuBLAS
   handle and device allocations, checks runtime errors, and cleans up on
   destruction or partial initialization failure. All nonempty parameter arrays
   are uploaded once; temporary host copies are released after synchronization.
   Empty linear blocks retain their shape metadata without allocating memory.
   Device identity, precision, shapes and resident byte counts are exposed.

5. **Bind loading and read-back without enabling evaluation.**
   `symmetrix/source/cpp/mace_lsc_cuda.cpp` exposes the class, availability checks,
   host packing and debug parameter read-back. It releases the GIL for loading
   and transfers. `mace_lsc_cuda_stub.cpp` supplies a clear unavailable-CUDA error
   in CPU-only builds, while retaining host layout tests. CMake registers the GPU
   source under the same CUDA/Kokkos condition as Polar, and the stub otherwise.

   ```python
   import symmetrix
   host = symmetrix.prepare_lsc_cuda_parameters("water-stage2-energy.json", "float32")
   gpu = symmetrix.MACELSCCuda("water-stage2-energy.json", "float32-mixed")
   assert gpu.parameters_on_device
   assert gpu.can_evaluate  # The full evaluator is now implemented.
   info = gpu.manifest
   weights = gpu.download_parameter("linear_up_0.weights.0")  # test/debug copy
   ```

6. **Isolate the CUDA environment and build.**
   `/mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev` is a separate copy of
   the working CPU dependency environment, with activation paths relocated.
   Its activation helper loads GCC 13.3 and CUDA 12.8. The CPU environment and
   its build remain separate. `tools/lsc/build_cuda.sh` uses CMake directory
   `symmetrix/build-lsc-cuda`, CUDA architecture 80, Kokkos CUDA, and cuBLAS.
   The CUDA toolkit and existing Polar numerical kernels are not modified.
   Repeat configuration clears deprecated uppercase KokkosKernels cache aliases;
   the CUDA build directory is ignored by Git.
   GPU host code uses the portable x86-64-v3 target, with native host tuning
   disabled for this CUDA build. The CPU build retains its existing native
   tuning by default. `build_and_test_cuda.slurm` performs the first build and
   loading tests in one bounded GPU allocation; subsequent tests can use the
   shorter `run_cuda_tests.slurm` wrapper.

   ```bash
   source /mnt/home/noneill/Programs/venvs/symmetrix-lsc-cuda-dev/activate-lsc-cuda-dev.sh
   bash -l /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/build_cuda.sh
   ```

7. **Test host packing and actual GPU residency separately.**
   `symmetrix/test/lsc/test_cuda_loader.py` checks real learned weights, species
   mapping, initial embeddings against native CPU, radial coefficients against
   an independent cubic calculation, mixed-format rounding, and invalid inputs.
   GPU tests create the class in both modes and read every array back, comparing
   dtype, shape and bytes exactly with host preparation. They also load the full
   1,024-knot stage-2 export, repeat construction/destruction and check that
   inference remains unavailable.

   The `lsc_cuda` marker identifies device tests. Ordinary CPU runs can deselect
   them. The GPU runner sets `LSC_REQUIRE_CUDA=1`, so missing CUDA/device support
   fails rather than silently skipping. `tools/lsc/run_cuda_tests.slurm` requests
   a separate short, one-GPU allocation; it does not reuse interactive jobs.

   ```bash
   # CPU reference environment:
   python -m pytest symmetrix/test/lsc/test_cuda_loader.py -m 'not lsc_cuda' -q
   # Submit actual GPU loading tests:
   sbatch /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/run_cuda_tests.slurm
   ```

Validation passed: **21 host packing tests**, **26 tests in the CUDA environment**
(including all **5 actual device tests**, with zero skips), and all **207 existing
CPU regression tests**. Actual device test properties, byte counts, dependency
versions, source checksums and job details are recorded in
`.lsc-reference-logs/cuda-loader-validation.json`. Host/GPU pytest output and
JUnit are `cuda-loader-host.log/xml` and `cuda-loader-gpu-7484882.log/xml`.
The CPU regression JUnit is `pytest-login-20261117T010056Z.xml`. This validates model loading and
parameter formats; no GPU energy/force/stress calculation is claimed.

GPU geometry, radial/angular spline kernels and local layers/readouts are now
implemented, as described below. Charge transfers/dipoles, electrostatics and
the reverse pass remain subsequent GPU increments.
LAMMPS integration and performance measurements remain later stages. Loading a
model and preparing mixed buffers establishes neither inference correctness nor
mixed-precision speedup by itself.


## GPU local interactions and energy readouts

`MACELSCCuda` now reports `can_evaluate_local=True` and can calculate the local
part of the energy on the GPU. Its full-model `can_evaluate` remains false:
GPU charge sources, electrostatics and force/stress reverse passes are not yet
implemented. The local energy is the atomic reference plus both learned linear
readouts; it must not be substituted for the complete LSC energy in dynamics.

The supported target remains the validated two-layer, 128-channel LSC family.
The same frozen 1,024-knot energy-capable export is used. No new export or
reference regeneration is required.

1. **Map and validate the convolution specialization.**
   `mace_lsc_cuda_tables.hpp` orders the foundation/Polar angular tables in the
   exporter's UVU order. `mace_lsc_cuda_parameters.cpp` validates each exported
   convolution instruction against that specialization. The GPU kernels read
   the export's actual convolution coefficients. Unsupported instruction layouts
   produce an explicit error and can still use the generic CPU evaluator.

2. **Fuse radial evaluation with incoming messages.**
   `mace_lsc_cuda_local.cuh` assigns one block to each receiver and one thread
   to each channel, following Polar's second-layer convolution structure. It
   gathers sender features through the stable incoming CSR, uses the tested
   harmonics/intervals/coordinates from GPU geometry, and evaluates the radial
   cubic inside the message calculation. It never materializes an edge-by-radial
   array. Edges retain the physical receiver-minus-sender direction, including
   periodic images.

3. **Keep LSC's local architecture and species dependence.**
   Each layer computes `linear_up -> convolution -> linear -> contraction ->
   product_linear + species residual`. The exported `linear` already includes
   average-neighbor normalization, which is applied once. Polar's additional
   gates and density normalization are not part of this LSC family. The
   contraction follows Polar's static correlation-order-3 tables but selects
   `M_weights[species,term,channel]`. Residual maps also depend on species and
   remain GPU calculations. Working features use Polar's
   `[l][node][m][channel]` layout.

4. **Use Polar's FP32/mixed dense operations.**
   `mace_lsc_cuda_gemm.cuh` copies Polar's high/low FP16 tensor-core GEMM
   arithmetic, scaling and tile selection without modifying Polar itself.
   Eligible mixed matrices use the previously uploaded high/low weights, with
   FP32 accumulation and outputs. Small matrices use cuBLAS FP32 under the
   same dispatch heuristic. Mixed radial layer 1 additionally reads c2/c3 as
   FP16; first-layer and source radial tables remain FP32.

5. **Read out and reduce local energy on the device.**
   Both LSC readouts are scalar linear maps. `mace_lsc_cuda.cpp` computes the
   two readouts, adds the FP64 atomic references and reduces node energies in
   FP64 using CUB. Shared scratch buffers are reused across layers and calls.
   `compute_local` uploads/checks a graph then runs the layers; `evaluate_local`
   reuses already-resident geometry. `download_local` is a diagnostic adapter
   that returns features in the native CPU `[node][l][m][channel]` order. Neural
   intermediates stay on the GPU during evaluation. Starting new geometry
   invalidates previous local outputs.

   ```python
   import symmetrix

   gpu = symmetrix.MACELSCCuda(
       "/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/water-stage2-energy.json",
       "float32-mixed",
   )
   # node_types[N]: species indices; edge_index[2,E]: sender/receiver indices;
   # edge_vectors[E,3]: receiver-minus-sender vectors in Angstrom.
   energy = gpu.compute_local(node_types, edge_index, edge_vectors)
   stages = gpu.download_local()  # Explicit test/debug download only.
   assert gpu.can_evaluate_local and gpu.has_local_results
   assert gpu.can_evaluate  # The full evaluator is now implemented.
   # Alternative: compute_geometry(...) once, then evaluate_local().
   ```

6. **Compare every stage and exercise the mixed kernel.**
   `test_cuda_local.py` checks all intermediate A/M/H features, both learned
   readouts, node energies and total local energy against native FP64 and the
   original MACE-SCF blocks for dimer and bulk in both precision modes. Absolute
   learned-readout checks prevent atomic reference offsets masking errors.
   Additional tests cover rotations, atom/edge permutations, reuse, isolated and
   empty graphs, invalid state, unsupported instruction layouts, and modified
   species weights/biases/convolution coefficients. Dimer and 192-atom bulk
   matrices use the FP32 fallback under Polar's heuristic; a 32,768-atom isolated
   batch verifies actual mixed tensor-core execution against the FP64 reference.
   The CUDA run passed all **107 tests**: 19 new local tests (2 host, 17 device)
   plus the 88 existing loader/geometry/spline tests. The large test executed
   **21 mixed tensor-core GEMMs** and 2 FP32 readout GEMMs. This establishes
   correctness for these test cases, not end-to-end speedup or full LSC accuracy.

Build and test in a separate GPU allocation:

```bash
sbatch /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/build_and_test_cuda_local.slurm
# Reuse an already-built CUDA extension:
sbatch /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/run_cuda_local_tests.slurm
```

The runner uses `LSC_REQUIRE_CUDA=1` so unavailable CUDA fails instead of skipping.
Pytest output and JUnit are saved as `.lsc-reference-logs/cuda-local-gpu-JOBID.log`
and `.xml`; the build log includes the job ID. Job **7485195** passed on an A100.
The implementation changes and reference/source checksums are recorded in
`.lsc-reference-logs/cuda-local-validation.json`.


## Reference generation and CPU build

Reference generation used the reference environment and completed before the
pytest suite was installed/run. The generation script and an optional Slurm
wrapper record how this was done.

The generator refuses to overwrite existing reference outputs. For a deliberate
new baseline, pass a new output directory to `generate_reference.py` using the
reference Python; review the differences and provenance before changing tests.
Never replace expected outputs automatically to make a failing test pass.

The pinned Git submodules and Python build/test tools were installed during
setup. For a subsequent CPU rebuild on the login node:

```bash
bash -l /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/build_cpu.sh
```

`LSC_BUILD_JOBS` controls build parallelism (four jobs by default). The Slurm
build wrapper is retained as an optional alternative; it is not required for
the normal login-node workflow.

Job/test logs and JUnit results are in `.lsc-reference-logs` at the repository root
and are ignored by Git. The build's `-march=native` settings follow upstream
Symmetrix; rebuild if moving the compiled extension to incompatible CPUs.

## Completed GPU and direct LAMMPS stages

The sections above record the implementation order. The current evaluator also
includes the following stages:

1. **Charge sources and dipoles.** Reuse Polar's species modulation and dense
   kernels. Evaluate the LSC sender/receiver source convolution with exported
   splines, then accumulate incoming minus outgoing transfer flux. Add atomic
   dipoles and the formal-charge/transfer polarization terms. The GPU source
   tests compare both layers, final charges and dipoles to CPU and PyTorch.
2. **Electrostatic energy.** Reproduce the original model's finite-cluster
   displaced-charge calculation or direct periodic reciprocal sum, including
   Gaussian smearing, self terms and the external-field convention. These sums
   use double precision because the displaced-charge terms strongly cancel.
3. **Analytic reverse pass.** Differentiate electrostatics to density, positions
   and cell strain, reverse both source and local interaction layers, and collect
   Cartesian forces and the virial. This follows Polar's contraction and dense
   reverse kernels, using the same mixed-GEMM selection and scaling. It does not
   use finite differences or PyTorch autograd at runtime.
4. **Physical readback and ASE.** `download_forces(False)` and the corresponding
   charge/local methods omit intermediate diagnostic arrays. The ASE adapter
   builds the graph, invokes the native evaluator and returns ordinary ASE
   energy, forces, stress, charges and dipole results.
5. **LAMMPS pair styles.** A full receiver neighbor list supplies physical image
   vectors. Atom IDs map periodic ghosts to owners. Unwrapped coordinates define
   the formal-charge polarization branch. Return energy, forces, global virial,
   optional per-atom energies/charges and total dipole to LAMMPS.

The neural GPU modes remain `float32` and `float32-mixed`; CPU remains float64.
The original Polar files and frozen references are unchanged. The specialized
architecture validator still restricts the exporter to the supported LSC family;
adding another architecture requires extending and testing its dispatch/kernels.

The LSC pair styles support **direct electrostatics on one MPI process**,
for nonperiodic clusters and fully periodic boxes. Both CPU and CUDA styles also support
one-process mesh energy, analytic forces and global stress for fully periodic
boxes (see the mesh sections below). Distributed MPI, slabs and per-atom
virials remain subsequent stages. Direct reciprocal sums can limit large-cell
performance.

### Run the current ASE adapter

```python
from symmetrix.lsc_calculator import LSCCalculator
atoms.calc = LSCCalculator(
    "/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/water-stage2-energy.json",
    device="cuda", precision="float32-mixed",
)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # Requires a nonsingular cell.
charges = atoms.get_charges()
dipole = atoms.get_dipole_moment()
```

Use `device="cpu", precision="float64"` for the CPU implementation. The original
reference is still `mace_scf.calculators.localsources.MACELocalSplitCharges` from
the unchanged `mace-scf-dev-niamh` checkout.

### Rusty LAMMPS build and input

The isolated checkout is `/mnt/home/noneill/Programs/lammps-lsc-dev`, based on
LAMMPS commit `582a54bcaaec568889b5960793d318c1b863ffbc` on
`codex/lsc-symmetrix`. It does not modify the existing Polar LAMMPS checkout.
`tools/lsc/lammps_prebuilt.cmake` links the already validated native library, so
LAMMPS itself uses its normal host compiler. Build with:

```bash
bash -l /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/build_lammps.sh cpu
bash -l /mnt/home/noneill/Programs/symmetrix-dev/tools/lsc/build_lammps.sh cuda
```

The CPU standalone library is tuned for its build CPU; rebuild it on another CPU
before using that CPU binary there. The CUDA-linked LAMMPS binary includes a
portable x86-64-v3 CPU style as well as the GPU styles.

```lammps
units metal
atom_style atomic
# Set the box and read coordinates before pair_coeff.
pair_style symmetrix/mace/lsc/cuda direct float32-mixed
pair_coeff * * /mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/water-stage2-energy.json H O
compute mu all pair symmetrix/mace/lsc/cuda
thermo_style custom step pe temp press c_mu[1] c_mu[2] c_mu[3]
```

Use `symmetrix/mace/lsc direct` for CPU float64, or omit `float32-mixed` for GPU
FP32. No `-sf kk` suffix is used by this new direct CUDA wrapper. Run one MPI
process; on Rusty inside Slurm use the matching `mpirun -np 1 --bind-to none`
launcher (the test and benchmark wrappers set it). Atom types in this example are H then O. `atom_style charge` additionally
receives the predicted partial charges in `q`; do not add a separate Coulomb pair
or kspace style, since LSC already includes its electrostatics.

### Basic integration checks and timing

`test_native_calculator.py` compares ASE results against the frozen predictions.
`test_lammps_lsc.py` compares LAMMPS point energy, forces, dipole, charges, global
stress and the sum of per-atom energies for dimer and bulk. It then runs 10 NVE
steps at 0.1 fs and checks total-energy variation per atom. These short tests
verify the interface; they do not establish long-trajectory stability.

```bash
# Full CPU suite on the login node:
LSC_REQUIRE_LAMMPS=1 bash -l tools/lsc/run_tests.sh -m 'not lsc_cuda'
# Full CPU and actual GPU suite in an allocated Slurm job:
sbatch tools/lsc/run_full_tests.slurm
```

The GPU runner requires both CUDA and LAMMPS, so those checks fail rather than
silently skipping. It uses the CUDA-linked binary for the portable CPU style.
All runs save logs and JUnit XML in `.lsc-reference-logs`.

`benchmark_full.py` separately measures full ASE calls with neighbor construction,
energy/forces/stress and result transfers. It compares the original PyTorch
float64 calculator on CPU/GPU to native CPU float64 and CUDA FP32/mixed. A
2x2x2 periodic replication (1536 atoms) checks the same energy/force/stress
relationship and exercises mixed reverse tensor-core GEMMs. Small matrices use
Polar's FP32 fallback, so mixed is not automatically faster.
`benchmark_lammps.py` measures actual NVE loop time, excluding model loading and
10 warm-up steps. These are separate measurements with different overheads.
The saved benchmark JSON records hardware, backend and measured values.

## Electrostatics backend interface (mesh preparation)

Both native evaluators now accept an optional electrostatics callback, following
Polar's optional Coulomb-hook design. With no callback installed, they keep the
existing direct energy/derivative routines. The CPU mesh energy, force and stress
evaluators are described below; distributed MPI remains a subsequent stage.

`libsymmetrix/source/mace_lsc_electrostatics.hpp` defines its contract. Inputs are
positions, predicted density coefficients, cell, PBC, the resolved boundary mode
and the actual electrostatic parameters used by the backend. Density order is
`[q, dipole_y, dipole_z, dipole_x]`. The callback returns the complete Coulomb
contribution:

| Result | Shape | Convention |
| --- | --- | --- |
| `energy` | scalar | Electrostatic energy, eV |
| `density_gradient` | N x 4 | Full positive dE/drho, not Polar's half-potential |
| `position_gradient` | N x 3 | Positive dE/dr at fixed density and cell |
| `strain_derivative` | 3 x 3 | Full positive dE/dstrain with positions and cell deformed together, density components held Cartesian |

Local neural energies and the external-field energy stay in the LSC evaluator.
The callback excludes both. Its complete strain derivative already contains
explicit position terms, which must be counted only once. The evaluator still
differentiates the learned sources and local layers using `density_gradient`.
It converts the final derivative to the existing ASE stress/LAMMPS virial order
and signs.

On CPU, `set_electrostatics_hook(LSCElectrostaticsHook)` installs a native C++
function returning `LSCElectrostaticsResult`. On CUDA, the same-named setter takes
`LSCCudaElectrostaticsHook`, which receives borrowed device buffers and the native
CUDA stream. Its zero-initialized output buffers are also on the device. The
native GPU mesh solver launches kernels/FFTs directly on that stream. Native
callbacks must not retain borrowed buffer pointers. CPU stubs still build without
CUDA headers, using an opaque stream declaration in the shared contract.

The Python `set_electrostatics_hook(callback)` binding accepts a function from a
request dictionary to the result dictionary above. Python arrays are copies;
retaining or modifying a request does not alias native working storage. The CUDA
Python binding deliberately uses an explicit host bridge, copying electrostatic
inputs/results for custom-solver checks. Production GPU mesh integration uses
the native device callback. Normal direct evaluation has no bridge readbacks.

For a CPU passthrough check:

```python
def direct(request):
    return model.evaluate_direct_electrostatics(
        request["positions"], request["density_coefficients"],
        request["cell"], request["pbc"], request["pbc_handling"],
    )

model.set_electrostatics_hook(direct)
try:
    model.compute_energy_forces(types, edges, vectors, positions, cell, pbc,
                                "auto", compute_stress=True)
finally:
    model.set_electrostatics_hook(None)
```

`evaluate_direct_electrostatics` evaluates a fixed density without changing model
state or invoking the callback, so it can serve as the reference for a future
mesh unit test. Installing or clearing a callback invalidates energy, force and
stress results. Re-evaluate before reading new physical results. Callback
energy/derivatives are computed together and cached; CUDA `evaluate_forces` can
consume that cache without invoking the callback again.

Before this refactor, `capture_electrostatics_baseline.py` saved three native
repeats for dimer/bulk, zero/nonzero fields, CPU FP64 and CUDA FP32/mixed. These
separate NPZ/JSON files live in `.lsc-reference-logs` and cannot be overwritten by
the capture script. The original ASE snapshots and model/export remain unchanged.
The new pytest module compares current direct results with that before-change
range to a 1e-10 CPU absolute bound. GPU per-output bounds are measured with 32
repeats of the unchanged pre-interface native code embedded in the preserved
LAMMPS library, using twice its observed repeat/cross-process spread and a 1e-9
floor. FP32 reverse-kernel atomic reductions are not bitwise deterministic.
The original ASE tolerances are unchanged. It also tests CPU passthrough,
callback clearing and invalid results, an independent quadratic electrostatic
energy with directional force/stress differences, and empty graphs with a finite
cell-dependent energy/strain contribution.

```bash
# Focused CPU checks:
python -m pytest symmetrix/test/lsc/test_electrostatics_interface.py -m 'not lsc_cuda' -q -s
# Rebuild CUDA, run focused checks, relink LAMMPS and run full regressions:
sbatch tools/lsc/build_and_test_electrostatics_interface.slurm
```

## One-process CPU mesh electrostatic energy

`LSCCPUMesh` now evaluates the periodic electrostatic energy and charge/dipole
potentials at fixed density. It uses the electrostatics input contract above.
Its `evaluate_energy()` diagnostic returns `LSCMeshEnergyResult`, which has no
position or strain outputs. It does not install a model callback or change model
state. The full `evaluate()` and native model hook for mesh forces/stress are
described in the next section.

The native C++ implementation follows Polar's cardinal B-spline recurrence,
first-derivative dipole spreading, FFT-friendly grid sizing, Fourier assignment
correction and Gaussian Coulomb kernel. The grid is local to one process and
uses double-precision FFTW. Distributed LAMMPS grid/FFT communication will be
added with MPI. The source retains attribution and the license notice of the
Polar pair code from which the grid operations were adapted.

The calculation is:

1. Convert Cartesian positions to wrapped fractional coordinates, including
   general nonsingular tilted cells. Choose enough grid points for the requested
   spacing, assignment order and the model's reciprocal cutoff, using dimensions
   factorable by 2, 3 and 5 as in Polar.
2. Spread normalized Gaussian monopoles and dipoles using the B-spline values and
   first derivatives. Per-thread grids are reduced before the transform.
3. Transform the grid with FFTW. Retain the same reciprocal modes and endpoints
   as the native direct operator, omit the zero mode, correct the assignment
   function and apply the model's Gaussian-damped Coulomb kernel.
4. Accumulate the spectral energy. The full reciprocal grid carries a factor
   1/2, matching the existing half-space direct sum without that factor.
5. Inverse-transform the potential and gather full positive dE/drho in
   `[q, dipole_y, dipole_z, dipole_x]` order. The FFT normalization follows the
   spectral energy, so `energy == 0.5 * sum(rho * density_gradient)`.
6. Apply the same analytic self-energy correction and its density derivative
   when the model has `include_self_interaction=False`.

The model supplies the field constant, Gaussian width/amplitudes, reciprocal
cutoff and self overlaps. No learned parameters, export format or frozen
reference outputs change. Local energies and external fields are outside this
operator.

### Using the energy evaluator

```python
import numpy as np
from symmetrix import MACELSC, LSCCPUMesh

model = MACELSC("/mnt/home/noneill/Programs/symmetrix-dev/.lsc-reference-logs/water-stage2-energy.json")
model.compute_charges(types, edges, vectors, positions)
rho = np.asarray(model.density_coefficients).reshape(-1, 4)
mesh = LSCCPUMesh(spacing=0.7, order=8)
result = mesh.evaluate_energy(model, positions, rho, cell, pbc,
                              "mixed_periodic")
electrostatic_energy = result["energy"]
density_gradient = result["density_gradient"]
# At zero external field:
total_energy = model.local_energy + electrostatic_energy
```

Reuse `mesh` to reuse FFT plans. Geometry and particle weights are refreshed on
each call, and plans are recreated when grid dimensions change. Calls on the
same solver are serialized; the planner's creation/destruction is also protected.
FFT transforms use one thread, following Polar's small-grid CPU approach;
spreading and gathering use OpenMP.

The returned dictionary also contains `grid_shape`, `reciprocal_energy` and
`self_correction`. The energy-only result reports `can_evaluate_forces=False`
and `can_evaluate_stress=False`; the class now also provides full derivatives
through `evaluate()`. A nonzero external field can be added separately
using the existing LSC dipole convention. This energy-only result is not a full
electrostatics callback for the native reverse pass.

The evaluator currently requires three periodic directions and supports orders
4 through 10. Excessive grids, invalid cells and nonfinite inputs fail before
unsafe allocations/calculation. Nonperiodic dimer evaluation continues through
the existing direct path.

### Build and validation

`SYMMETRIX_LSC_MESH` is optional and defaults to OFF. Enabling it requires FFTW3
headers and its double-precision library. With it disabled, the CPU stub keeps
the package buildable and reports that mesh support is unavailable. The Rusty
CPU and CUDA development build helpers enable it; the CUDA environment gains
this same CPU backend; the separate GPU mesh class is described below. Rusty already provides FFTW 3.3.8
in `/usr/include` and `/usr/lib64`, so no dependency installation or CUDA toolkit
change was needed. The prebuilt LAMMPS helper also links the FFTW dependency.

```bash
bash -l tools/lsc/build_cpu.sh
python -m pytest symmetrix/test/lsc/test_cpu_mesh_energy.py -q -s
# Run the same checks under mpirun -np 1:
bash -l tools/lsc/run_cpu_mesh_energy_tests.sh
# Refresh the CUDA environment and run full CPU/CUDA/LAMMPS regressions:
sbatch tools/lsc/build_and_test_cpu_mesh_energy.slurm
```

The 14 basic tests use the saved bulk density and compare electrostatic energy
alone against the fixed-density direct operator, explicitly selecting periodic
boundary handling independently of the checkpoint's realspace training default.
They cover spacing refinement, charge-only/dipole-only/mixed densities,
self-correction, density finite differences, the quadratic energy identity,
periodic wrapping, reuse after a cell/grid change, a tilted cell, empty/zero
sources, validation and a standalone build of the no-FFTW stub. The dimer remains
part of the existing direct regression suite.

For this model's mixed bulk density at order 8, mesh spacings 0.7, 0.5 and 0.35
Angstrom produce energy errors approximately 7.9e-8, 7.0e-9 and 4.7e-10 eV. These
numbers characterize this fixed-density energy test. CPU mesh position forces,
cell stress and mesh MD are now available as described below. GPU mesh is
described next; distributed MPI remains a subsequent stage.


## One-process CPU mesh forces and stress

`LSCCPUMesh.evaluate(model, positions, rho, cell, pbc, "mixed_periodic")`
returns the full electrostatics contract: `energy`, `density_gradient[N,4]`,
`position_gradient[N,3]` and `strain_derivative[3,3]`. Gradients are positive
energy derivatives. Physical position forces have the opposite sign. Strain
means simultaneous `positions @ (I + strain)` and `cell @ (I + strain)` with
density coefficients held fixed in Cartesian components.

Like Polar, gathering includes the potential Hessian for dipole forces. The
cell derivative includes the volume/Gaussian Fourier kernel and the change of
the dipole assignment under strain. The density self correction contributes to
the density gradient, with zero explicit position/cell derivative. These are
analytic derivatives of the selected mesh, not runtime finite differences.

For full model forces, install the native solver before evaluation:

```python
from symmetrix.lsc_calculator import LSCCalculator
calc = LSCCalculator("model.json", device="cpu", pbc_handling="auto")
calc.model.set_mesh_electrostatics(spacing=0.7, order=8)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # ASE order; eV/Angstrom^3
```

The owned C++ solver reuses its FFT plans and supplies the full density gradient
to the existing learned LSC reverse pass. There is no Python callback in this
calculation. Use `calc.model.set_electrostatics_hook(None)` to restore direct
mode; when switching an already-used ASE calculator also call `calc.reset()`
to discard ASE's cached result. Mesh mode requires three periodic directions;
the nonperiodic dimer uses direct mode.

LAMMPS CPU mode uses the same solver and reverse pass:

```text
pair_style symmetrix/mace/lsc mesh 0.7 order 8
pair_coeff * * model.json H O
```

This supports energy, forces, global virial/stress, charges, dipole and per-atom
energy on one MPI process. `mesh` is the target spacing in Angstrom; spline
`order` defaults to 8 (supported range 4-10). Direct remains the default. CUDA pair mesh mode is described in the next section. Do not add a separate kspace solver.

```bash
bash -l tools/lsc/build_cpu.sh
bash -l tools/lsc/build_lammps.sh cpu
bash -l tools/lsc/run_cpu_mesh_derivatives_tests.sh
sbatch tools/lsc/build_and_test_cpu_mesh_derivatives.slurm
```

The new tests compare fixed-density and full-model force/stress with direct,
including a tilted cell, field terms and self-correction; independently check
position and all six symmetric strain derivatives by finite differences;
confirm switching back to direct; and compare LAMMPS against native mesh
predictions plus a ten-step NVE smoke test. Finite differences are taken away
from discrete mesh-dimension and reciprocal-cutoff changes. Small mesh aliasing
errors should decrease when spacing is refined; the stage-2 water model is the
current regression model, not a guarantee for every architecture or system.


## One-process GPU mesh energy, forces and stress

`LSCCudaMesh` follows Polar's local GPU mesh: cardinal B-spline charge/dipole
assignment with double atomic additions, double-precision cuFFT, Gaussian
Fourier convolution with assignment correction, and potential/Hessian gathering.
Grid/cutoff geometry is now shared with CPU mesh in `mace_lsc_mesh_common.hpp`.
The reciprocal energy and cell derivatives are reduced on the GPU with CUB.
As in CPU mesh, strain includes both the kernel/volume term and the change in
dipole assignment at fixed Cartesian density. Self-correction supplies its
density gradient and has no explicit position/cell derivative.

`LSCCudaMesh.evaluate_energy(cpu_model, positions, rho, cell, pbc)` is an explicit
host diagnostic: it uploads fixed positions/density and reads energy/potentials
back, using the CPU model's exact parameters to isolate mesh implementation
errors. `.evaluate(...)` also returns positive position and strain gradients.
They are not MD force APIs. The production solver uses the CUDA device hook:

```python
from symmetrix.lsc_calculator import LSCCalculator
calc = LSCCalculator("model.json", device="cuda", precision="float32-mixed")
calc.model.set_mesh_electrostatics(spacing=0.7, order=8)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

The native mesh hook takes resident device positions/density and writes energy,
density gradients, position gradients and strain derivatives on the supplied
stream. It does not call Python, copy particle/grid arrays to the CPU or invoke
FFTW. The existing neural reverse pass includes learned charge/dipole response.
Ordinary final energy/force/stress readbacks are unchanged. Plans and buffers
are reused; cells and atom weights update per call. Buffer growth, plan changes
and stream changes finish preceding mesh work before reuse. One solver instance
serializes host entry with a mutex. Its owned GPU resources are freed on their
device. Pure-device C++ callers must keep the supplied stream alive through the
solver lifetime, as the native model hook does.

Both FP32 and mixed neural modes are supported. Mesh operations and physical
reductions stay double precision; this does not add FP64 neural kernels or
change the CUDA toolkit. The existing sm80/PTX development build is retained;
GPU tests for this stage run in the user's H100 allocation. CPU-only builds
provide a clear unavailable CUDA mesh stub and need no CUDA/cuFFT dependencies.

```text
pair_style symmetrix/mace/lsc/cuda mesh 0.7 order 8 float32-mixed
pair_coeff * * model.json H O
```

The CPU and CUDA pair styles both support direct or mesh selection. This GPU
mesh stage still requires one GPU on one MPI rank and full 3D periodicity;
nonperiodic dimers use direct. Distributed neural/charge communication and
mesh/FFT decomposition remain subsequent stages. Global stress, per-atom
energy, charges and dipole use the existing pair output paths. Slabs and
per-atom virials are not implemented.

For an already-used ASE calculator, call `calc.reset()` after changing the
native solver. `calc.model.set_electrostatics_hook(None)` restores direct mode.
The energy-only diagnostic retains `can_evaluate_forces=False` and
`can_evaluate_stress=False`; the full mesh solver/evaluation supports both.

From the development checkout, with an active H100 allocation:

```bash
srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact bash -l tools/lsc/run_gpu_mesh_tests.sh
```

`LSC_REUSE_BUILD=1` skips the native rebuild when appropriate. The runner rebuilds
LAMMPS, tests both GPU mesh precisions and then runs the complete CPU/CUDA/LAMMPS
regression suite. The focused tests check CPU mesh agreement, direct convergence,
density/position/strain finite differences, fields, self correction, plan reuse,
wrapping, a tilted cell, empty inputs, direct restoration and basic LAMMPS
point/NVE behavior. No frozen model/export/ASE data are regenerated. A tiny
standalone C++ test also checks that the no-CUDA stub links without CUDA.


The existing direct-output regression supports `LSC_CUDA_DIRECT_BASELINE` and
`LSC_CUDA_DIRECT_REPEATABILITY` overrides for hardware-matched old-code controls.
The GPU mesh runner selects the independent H100 control on H100. It was captured
from the preserved pre-interface `electrostatics-old-control/liblammps.so.0`
(verified against its previously recorded SHA256), with 32 repeats for each
precision/configuration/field. No current mesh/new-code output was used to
create that baseline or its bounds. Original A100 native outputs and frozen ASE
references remain intact. Comparing the old binary itself on H100 with its A100
outputs reproduces the cross-hardware direct-output test discrepancies; the
physical ASE parity tolerances are unchanged.

## Staged evaluation and owned/ghost indexing

`LSCGraph` now records owned atoms first and ghost slots afterwards. Edges retain
sender slots, owned receivers and physical receiver-minus-sender image vectors.
64-bit atom IDs identify copies without merging their image vectors. All local
energies, charge sources and returned forces are counted on owned atoms only.
Local periodic copies resolve to owned feature rows. Remote ghost features now
have their own storage and are filled by forward H1/H2 exchanges, described
below. An edge using an unreceived remote feature raises a clear error. The
layout can retain unused remote slots, including an empty owned partition.
Distributed charge accumulation, forces and mesh/FFT remain subsequent work.

Both CPU and CUDA now expose the following sequence; their original one-call
entry points compose the same native stages:

```python
import symmetrix
layout = symmetrix.LSCGraph(node_types, edge_index, edge_vectors,
                           num_owned=nlocal, atom_ids=atom_ids)
model.prepare_graph(layout)             # Validate, resolve local images, geometry.
model.compute_first_layer()             # H1 ready; future forward exchange here.
model.compute_second_layer()            # H2 ready; LSC also needs H2 at endpoints.
model.compute_readouts()                # Owned local energies.
model.evaluate_charges(owned_positions) # Transfers, density, periodic dipole.
model.evaluate_energy(cell, pbc, "auto") # Current direct or installed mesh solver.
model.reverse_charges()                 # Source/readout seeds in H1/H2 gradients.
model.reverse_second_layer()            # Add second-layer contributions to H1.
model.reverse_first_layer()             # Differentiate first layer and geometry.
model.finalize_forces(compute_stress=True)
```

1. `libsymmetrix/source/mace_lsc_graph.hpp` centralizes slot/ID validation and the
   local periodic-owner map, shared by CPU, CUDA and LAMMPS.
2. `mace_lsc.cpp` and `mace_lsc_cuda.cpp` split their actual computations at layer
   boundaries, following Polar's forward/reverse structure. They cache graph,
   positions and electrostatic state between calls. `evaluation_stage` and
   result flags reject invalid order and invalidate downstream results on a new
   graph or first-layer restart. CUDA kernel formulas and launch ordering are
   retained. CPU remains FP64; CUDA neural modes remain FP32/mixed.
3. C++ `layer_features` / `layer_feature_gradients` provide CPU spans, and CUDA
   counterparts provide borrowed device pointers plus `cuda_stream()`. CPU rows
   use `[node][l][m][channel]`; GPU working storage uses `[l][node][m][channel]`.
   These original accessors retain owned rows. The new `forward_features` CPU
   accessor and `forward_features_device` CUDA accessor expose owned plus ghost
   rows for forward communication, as described below. Borrowed
   buffers expire when the evaluator restarts/resizes them; use the CUDA model's
   stream or synchronize before another stream accesses its data. Python
   `download_layer_features(layer, gradients=False)` returns diagnostic copies
   in canonical node-major order. Graph properties also return independent
   read-only copies. There is no feature host readback in the native stages.
4. The LAMMPS pair now constructs this layout from `nlocal + nghost`, original
   neighbor slots and atom tags, and invokes the stages. It retains the one-rank
   guard for full force evaluation, physical image vectors, and existing
   force/energy/dipole output rules. A separate C++ `compute_forward()` entry
   point now supports forward-only multi-rank integration checks.

`test_staged_api.py` contains 54 basic cases across CPU/FP32/mixed: dimer direct,
bulk direct/mesh, zero/nonzero fields, ordinary layouts and periodic ghost
copies, stage-order/restart/empty handling, and invalid ownership/ID/geometry.
Saved outputs were captured before the refactor in
`.lsc-reference-logs/staged-api-before-{cpu,cuda}.npz`, with SHA256 metadata.
The tests compare both one-call and staged execution against those outputs,
including intermediate features, transfers/density, energy, dipole, forces and
stress. CPU absolute limits remain 1e-10. GPU deterministic outputs use 1e-9;
force/strain rounding bounds are those independently measured with the old H100
binary, not adjusted from this patch. Frozen original ASE snapshots are unchanged
and remain separate full-suite parity checks.

With the builds active on the allocated H100:

```bash
srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_staged_api_tests.sh cuda
```

Use `bash -l tools/lsc/run_staged_api_tests.sh cpu` on the login node for CPU-only
checks. No before-output file or original ASE reference is regenerated by these
runners. MPI extensions are described below; distributed mesh/FFT remains later work.


## Remote ghost features and forward H1/H2 MPI communication

This stage extends forward local evaluation across ranks while preserving the
one-rank complete evaluator. Each rank computes only its owned receivers. The
initial embedding depends on known species/formal charges, so it can be built
for ghosts without communication. Layer one then reads those fixed embeddings,
produces owned H1 rows, and publishes them for exchange. Layer two reads owned
and received H1 rows, produces owned H2 rows, and publishes H2. The latter is
needed because LSC transfer sources use both endpoints' features in both layers.
Local energies/readouts still sum owned atoms only. Polar's CPU
pair exchanges its lifted H1e representation. LSC exchanges the unprojected H1
and H2 because its separate transfer-source blocks need those hidden features
at both endpoints; the second layer lifts received H1 with the existing learned
linear map. The LAMMPS forward-callback pattern is the same.

1. `mace_lsc_forward.hpp` shares packet/readiness checks across CPU and CUDA.
   Remote rows start as zero/unready. Owned rows and local periodic copies are
   ready immediately after their layer. Received rows overwrite ghosts without
   translation or summation. Send lists may include already-received ghosts,
   supporting the multi-hop swaps used by LAMMPS. Attempting to pack an unready
   ghost or advance with missing H1/H2 data fails explicitly. Preparing or
   restarting a graph clears old readiness, including after invalid input.
2. CPU uses node-major halo arrays separate from the original owned H1/H2
   arrays. Input linear maps can read all slots; products, residuals and
   readouts still produce owned rows. CUDA retains its l-major working layout,
   with separate all-slot halo buffers. The convolution kernel's halo
   specialization indexes sender features using the all-slot stride while
   writing owned messages. The legacy specialization retains its original
   arithmetic. New kernels publish halos and pack/unpack selected rows.
3. Both classes provide `pack_forward_features(layer, slots)`,
   `unpack_forward_features(layer, first, values)`,
   `complete_forward_exchange(layer)` and `forward_exchange_complete(layer)` in
   Python. Payloads use canonical `[row][l][m][channel]` order and double values,
   matching ordinary LAMMPS pair communication. CPU features remain FP64; GPU
   features remain FP32 (including mixed neural mode). A float-double-float
   transport preserves these FP32 values exactly. Invalid size, slot, stage,
   nonfinite or unrepresentable input is rejected before writing ghost rows.
4. The pair follows Polar's `pack_forward_comm`/`unpack_forward_comm` callbacks,
   calling `comm->forward_comm(this)` after H1 and H2. It uses the original
   LAMMPS ghost slots/tags and physical edge vectors. Its C++ `compute_forward()`
   runs this sequence and owned readouts. Normal `compute()` calls it before
   the existing one-rank charge/energy/reverse stages. Normal multi-rank force
   evaluation still errors explicitly before returning any partial physics.

```python
model.prepare_graph(layout)  # Owned slots first; remote sender ghosts allowed.
model.compute_first_layer()
# The communication layer requests owned/previously received rows:
send = model.pack_forward_features(1, send_slots)
# ... transfer send buffers with MPI, according to the domain decomposition ...
model.unpack_forward_features(1, first_ghost_slot, received_rows)
model.complete_forward_exchange(1)
model.compute_second_layer()
# Repeat the exchange for layer 2.
model.unpack_forward_features(2, first_ghost_slot, received_h2_rows)
model.complete_forward_exchange(2)
local_energy = model.compute_readouts()
```

`download_forward_features(layer)` gives diagnostic copies for all slots in
canonical node-major order. Original `download_layer_features` still gives
owned rows. C++ CPU `forward_features` is a read-only span; CUDA
`forward_features_device` is a borrowed pointer with the all-slot l-major
layout. Lifetimes follow the prepared graph/buffer resize. Native layer
calculation remains on the GPU. The initial CUDA MPI integration stages only
selected feature messages through host double buffers, synchronizing before
LAMMPS reuses them. It uses ordinary MPI, without a CUDA-aware MPI requirement.
Device-direct/Kokkos communication and bandwidth optimization remain future work.

`test_forward_halo.py` adds nine basic CPU/FP32/mixed cases: dimer/bulk two-way
partitions, local/repeated remote image slots, H1/H2 parity with saved serial
features, owned local-energy sum, packet validation, stale-state rejection and
an empty owned partition. It also checks that the ordinary serial charge API
rejects remote graphs; the staged source API below handles communication. The old staged test now expects remote graph preparation to
succeed and a missing H1 exchange to fail; its numerical limits are unchanged.

`test_forward_mpi.py` adds ten cases using a small C++ driver linked against the
real development LAMMPS library. The driver initializes LAMMPS domains, atom
exchange, ghost borders and neighbor lists as Verlet does, then calls only the
forward API. No dummy MD energies or forces are returned. Tests run one and two
MPI ranks for dimer/bulk in CPU/FP32/mixed modes, compare owned features and
summed local neural energy, require every ghost H1/H2 row to equal its owner's
row exactly, handle an empty rank, and check that full multi-rank forces remain
blocked. GPU checks run two MPI processes sharing the allocated H100 for
correctness; they are not two-GPU scaling or timing tests.

```bash
# Inside the active H100 allocation; builds native/library/driver, then tests.
srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_forward_mpi_tests.sh cuda
# Use validated existing native/LAMMPS builds:
LSC_REUSE_BUILD=1 srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_forward_mpi_tests.sh cuda
# Separate CPU environment on the login node:
bash -l tools/lsc/run_forward_mpi_tests.sh cpu
```

The driver build helper uses the existing matching OpenMPI and LAMMPS build;
there is no mpi4py dependency. Existing frozen model/export/ASE/native snapshots
remain unchanged. Partial-charge accumulation and potential communication are
extended below. Reverse feature/force communication, distributed mesh/FFT and
global reductions remain subsequent steps; forward MPI alone does not enable
multi-rank LSC MD.


## MPI partial charges and electrostatic density potentials

Forward H1/H2 exchange is now followed by an explicit source communication API.
It still does not enable full multi-rank MD: distributed electrostatic energy,
reverse feature gradients/forces/stress and distributed mesh/FFT remain later
steps. Ordinary one-rank full-model evaluation keeps its existing source path.

Each rank evaluates directed transfers only on edges with an owned receiver.
A transfer adds charge to its receiver and subtracts it from its sender. When
the sender is remote, its negative contribution is stored in that ghost slot.
Reverse MPI communication **adds** those contributions back to owners, including
through intermediate ghost relays. Here "reverse" means returning MPI data to
its owner; the analytic reverse differentiation for forces remains a separate
later extension. There are two scalar values per row, one per
source layer. Returned ghost contributions are consumed once. Formal charges
and predicted atomic dipoles are included on owned atoms only, after exchange;
ghost copies cannot add an extra formal charge or atomic dipole.

`mace_lsc_charge_exchange.hpp` shares phase, finite-packet, index and duplicate
return checks across CPU/CUDA. It prevents completing with a used remote sender
still unreturned, returning a row twice, or sending a nonzero contribution to a
ghost that was already returned. Normal LAMMPS reverse communication guarantees
all incoming messages have completed before final assembly. Native clients must
provide that same communication guarantee. The two-layer packet layout retains
per-layer conservation diagnostics without a charge normalization correction.

CPU reuses the validated source maps with all-slot inputs. CUDA reuses the
existing modulation, linear, source-flux and source-density kernels with the
all-slot stride. The new kernels seed/extract owned density, return/add partial
charges and stage potentials. Partial sums, density assembly and potentials are
FP64; neural operations remain FP32/mixed. Source-feature diagnostics retain
owned shapes. Graphs without used remote senders reuse the existing serial
source calculation and then enter the explicit communication protocol.

After charge assembly, an electrostatic backend supplies **full positive**
`dE_electrostatic/d[q,dy,dz,dx]` for each owned atom. This includes the scalar and
all three dipole-density derivatives, with no half factor or external-field
term. Forward MPI communication **copies** these four doubles to ghosts. Local
periodic copies are initialized from their owner, remote rows remain unavailable
until received, and completion checks the edge endpoint rows needed for the
later reverse source pass. Potential copying uses the same LAMMPS forward-swap
pattern as features. CUDA stages messages through host buffers on its stream.

```python
# After H1/H2 exchanges and compute_readouts():
model.compute_partial_charges(owned_positions)
partial = model.download_partial_charges()  # [all slots, 2], learned monopoles only
send = model.pack_reverse_charges(first_ghost, count)
# MPI reverse messages; receive lists may contain repeated/ghost relay slots.
model.unpack_reverse_charges(receive_slots, received_layer_charges)  # adds
model.complete_charge_exchange()
# Owned density/charges/dipoles and this rank's polarization are now available.
model.set_owned_potentials(owned_density_gradients)  # [N_owned,4], full dE_el/drho
send_phi = model.pack_potentials(send_slots)
# MPI forward messages:
model.unpack_potentials(first_ghost, received_potentials)  # copies
model.complete_potential_exchange()
assert model.potential_exchange_complete()
```

`charge_exchange_stage` reports `empty`, `partial_charges`, `assembled_charges`,
`potentials` or `potentials_complete`. Partial-charge readback is available only
while the reverse phase is active. `download_electrostatic_potentials()` returns
all-slot `[N_all,4]` diagnostic copies. Restarting/preparing a graph invalidates
both phases. Setting new owned potentials permits a fresh forward exchange.
The MPI source protocol does not set full-model energy/force flags, and
`evaluate_energy` rejects this protocol until a distributed electrostatics and
reverse path is supplied; it cannot accidentally return local-only Coulomb
energy. The original one-call one-rank API is unchanged.

The pair now has C++ `compute_distributed_sources()` and
`exchange_electrostatic_potentials(owned_values)` entry points. It calls real
`reverse_comm(this,2)` for the two layer charges and `forward_comm(this,4)` for
the full potentials. The full force/MD guard remains in place. No partial-physics
pair-style keyword or dummy force calculation is introduced.

`test_charge_exchange.py` adds nine native CPU/FP32/mixed cases using the saved
dimer/bulk graphs. They cover charge conservation, repeated receive-slot sums,
local/repeated remote image slots, no duplicate formal charges/dipoles, density
and polarization parity, real direct-operator potentials, exact ghost potential
copying, duplicate/missing returns, stale/invalid inputs and an empty owned rank.
`test_charge_mpi.py` adds nine real LAMMPS MPI cases for one/two ranks and empty
ranks. They check owned density, global charge and polarization, every ghost
potential against its owner exactly, and `E_el == 0.5 * sum(rho * dE_el/drho)`.

For potentials, the existing C++ test driver accepts an optional
`charges model.json` stage selector. It gathers the small test system's assembled
owned density/positions, runs the existing serial CPU direct operator as a
reference, and passes owned derivatives through the real LAMMPS exchange.
**That global gather/serial solve exists only in the test driver.** Production
pair/native communication contains no replicated Coulomb solver or new allgather.
It validates the potential handoff needed by a future distributed backend;
it is not a scalable electrostatics implementation or a multi-rank speed test.
Two CUDA MPI processes share the allocated H100 for correctness checks.

```bash
srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_charge_mpi_tests.sh cuda
# After validating the existing native/LAMMPS build:
LSC_REUSE_BUILD=1 srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_charge_mpi_tests.sh cuda
# Separate CPU environment on the login node:
bash -l tools/lsc/run_charge_mpi_tests.sh cpu
```

The runner builds the matching native/LAMMPS/driver unless reuse is requested,
then runs the new checks plus one-rank preservation tests. Existing checkpoint,
exported weights/splines, original ASE outputs and old native snapshots are not
regenerated. Complete one-rank CPU/CUDA direct/mesh ASE/LAMMPS performance can
already be benchmarked. Forward/source MPI timings are partial-stage timings;
the following section adds reverse forces; a complete multi-rank MD speed
comparison still needs the production global electrostatics backend.

## MPI analytic reverse: H2, H1, forces and rank stress

After the charge/potential stage, `begin_distributed_reverse` seeds the analytic
source/readout reverse pass. CPU and CUDA now return ghost H2 gradients, reverse
owned layer-2 outputs, return the combined H1 gradients, reverse owned layer-1
outputs, assemble forces, and return ghost forces. Receives SUM; sending consumes
that ghost contribution once, including during multi-hop LAMMPS reverse swaps.
`mace_lsc_reverse_exchange.hpp` shares phase/index/finite/duplicate checks.
Blocking callers must finish all sends/receives before marking an exchange done.

This follows Polar's reverse-feature exchange pattern. LSC needs an additional
H2 exchange because its second source layer depends on both edge endpoints'
H2 features. Both source layers differentiate atomic dipoles on owned atoms only;
transfer flux derivatives use the copied scalar potentials at both endpoints.
Local readout seeds are also owned-only. The original analytic maps are reused;
CPU reverse lifting accepts an all-slot input extent, and CUDA convolution reverse
has an optional ghost stride. Existing serial CUDA instantiations keep the owned
stride. Geometry, forward interactions, splines and source formulas are unchanged.

The backend handoff takes:

* owned positive fixed-density `dE_el/dr`, shape `[N_owned,3]`;
* this rank's full electrostatic strain derivative, nine row-major values,
  including explicit cell and position terms at fixed Cartesian density;
* row-major cell (nine values), external field (three) and this rank's
  electrostatic energy contribution.

Potentials must already be complete. Backend values exclude field/local neural
terms, which the native pass adds once. Energy and strain contributions must sum
to the global backend values. Do not add `r*dE_el/dr` again: the supplied full
strain already includes it. Stress/virial exposed by each native evaluator are
rank contributions, using the existing ASE/Polar orders and signs. Returning
forces does not add another strain term.

```python
model.begin_distributed_reverse(owned_position_gradient, rank_strain9,
                                cell9, field3, rank_electrostatic_energy)
# Phase 1: reverse source and local readout gradients have been seeded.
send = model.pack_reverse_gradients(2, first_ghost, count)
model.unpack_reverse_gradients(2, receive_slots, received)  # SUM
model.complete_reverse_gradient_exchange(2)              # phase 2
model.reverse_distributed_second_layer()                 # phase 3
send = model.pack_reverse_gradients(1, first_ghost, count)
model.unpack_reverse_gradients(1, receive_slots, received) # SUM
model.complete_reverse_gradient_exchange(1)              # phase 4
model.reverse_distributed_first_layer()                  # phase 5
model.finalize_distributed_forces(compute_stress=True)     # phase 6
send = model.pack_reverse_forces(first_ghost, count)
model.unpack_reverse_forces(receive_slots, received)       # SUM
model.complete_force_exchange()                          # phase 7
assert model.has_force_results
```

Here `received` denotes the matching MPI packet for that phase. Feature payloads
have `4*C` values per row; force payloads have three. CUDA keeps FP32/mixed neural
math, double physical derivatives and double host MPI packets. Packing FP32
values through double is lossless; sums still follow the native FP32 arithmetic.
Only sparse packets, small reductions and requested diagnostics are read back.
The working feature arrays retain Polar's l-major GPU layout.

`distributed_reverse_phase` reports 0..7. No force result is published until phase
7. Re-preparing/restarting the graph, changing potentials or changing the backend
invalidates reverse results. The old serial reverse entry point rejects the MPI
source protocol rather than reading incompatible caches. Ordinary one-call
one-rank direct/mesh evaluation continues through its original route.

The LAMMPS C++ `reverse_distributed` entry point invokes real `reverse_comm`
callbacks for H2, H1 and forces, with maximum buffer width `4*C` and per-phase
widths `4*C`, `4*C`, 3. Charge packets remain width 2. It reduces energy, virial
and stress into separate global diagnostics, leaving normal pair tallies alone.
Full multi-rank `compute`/MD is still guarded until production global
electrostatics is connected.

`test_reverse_exchange.py` has 15 native CPU/FP32/mixed cases: two partitions of
dimer/bulk at zero and nonzero field, serial E/F/stress parity, total force,
periodic/repeated ghost images, packet/state/reset guards. `test_reverse_mpi.py`
has 15 real LAMMPS one/two-rank cases for those systems/fields plus empty ranks.
The `reverse model.json` diagnostic driver selector supplies full direct-operator
derivatives. Its replicated strain is divided between ranks for testing and its
energy uses the quadratic density contraction. These are test-only partitions,
not an algorithm for a production mesh backend. The global gather/serial solver
remains confined to the driver. Two CUDA MPI ranks share one allocated H100;
these are correctness checks, not multi-GPU scaling measurements.

```bash
LSC_REUSE_BUILD=1 srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_reverse_mpi_tests.sh cuda
bash -l tools/lsc/run_reverse_mpi_tests.sh cpu
```

Omit reuse to rebuild first. The focused runner includes the prior one-rank
preservation tests. The full suite also keeps original ASE, finite-difference,
mesh and LAMMPS point/NVE checks. No frozen outputs or existing tolerances are
changed. The next production step is distributed mesh/FFT electrostatics and
consistent energy/strain ownership, then complete multi-rank MD validation and
performance/scaling. Complete one-rank benchmarking is already possible.

## Distributed LSC mesh/FFT electrostatics

The staged LAMMPS API now has `evaluate_distributed_mesh()`, called after
`compute_distributed_sources()`. It solves the global periodic electrostatic
problem using owned particles and distributed grid/FFT data, exchanges full
four-component density derivatives, and runs the existing analytic MPI reverse
through forces and stress. No global particle gather or replicated solver is
used in this path. Ordinary multi-rank `compute`/MD is still guarded; connecting
normal pair tallies/integration is the following step.

The new `LSCDistributedMesh` follows Polar's `Grid3d` reverse density sums,
LAMMPS pencil FFT/remap plans, reciprocal kernel, inverse transform and forward
potential halo copies. Each rank owns a unique subset of Fourier modes. It
accumulates spectral energy/strain from those modes and self-correction/dipole
assignment terms from owned atoms. These contributions sum to the global
backend values; no artificial `global_strain/nranks` partition remains. The
original LSC exact-cutoff mode rule, Gaussian factors, spline modulus,
unnormalized inverse and density ordering `[q,dy,dz,dx]` are preserved.

CPU uses LAMMPS's FFTW pencil transform. CUDA reuses the same pencil/remap plans
and performs each contiguous local axis batch with cuFFT. The new device class
also evaluates assignment, spreading, convolution and gathering on the GPU,
with CUB reductions returning compact energy/strain totals. Communication uses
host staging for local grid bricks, FFT pencils and owned derivative buffers,
as the current neural exchanges do. This is not GPU-aware/direct transport and
its overhead needs profiling before speed claims. It avoids introducing a new
LAMMPS Kokkos configuration or changing CUDA. Both neural FP32 and mixed modes
keep double mesh/FFT/physical operations.

The mesh requires three periodic directions and the same neighboring-process
stencil condition as Polar. An oversized stencil raises a clear error rather
than silently using a replicated solver. The grid/FFT plans are cached and
rebuilt when cell, partition, shape or skin changes. Changing pair settings
resets the solver. The original single-process CPU/GPU mesh classes and normal
one-rank calculation remain on their existing paths.

`test_distributed_mesh_mpi.py` adds 21 parameterized CPU/FP32/mixed cases. The
periodic dimer and bulk run on one/two ranks at zero/nonzero fields; additional
cases cover an empty particle rank, a tilted cell and disabled self interaction.
Checks include the existing serial mesh at identical density/physical
parameters, the unchanged normal one-rank pair, energy identities, owned-mode
partitioning, exact potential copies, dipole/force/stress parity, plan reuse and
six device FFT axis batches on CUDA. The driver poisons the inactive local-bound
arrays with NaNs, checking that rectangular/tilted plan caches read only initialized
LAMMPS bounds. The dimer is periodic for these mesh tests;
its original isolated direct references remain separate and unchanged.

The existing driver adds `mesh model.json` and `serial model.json` selectors.
The mesh selector uses the actual distributed solver, not its old diagnostic
global direct reference. The serial selector calls the normal one-rank pair as
an independent full-model oracle. Test-only Python combines rank outputs for
comparison with serial mesh. Tests use two MPI processes sharing one H100;
multiple-GPU/node scaling is not established by these checks.

The prebuilt LAMMPS helper enables double FFTW KSPACE and compiles the new helper.
Standard installation already includes the new pair helper files through its
existing source glob. Pure native CPU/GPU calculators acquire no MPI dependency.
Electrostatic context accessors expose immutable actual model constants to the
external distributed backend; CUDA constants retain their existing packed
precision. The fixed-density GPU oracle rounds its CPU constants identically,
so double mesh/transport errors are tested separately from parameter rounding.

A new H100 allocation on workergpu213 was used after workergpu169 neared expiry.
Before rebuilding successfully, the unchanged installed extension was copied
and captured on the new hardware. The independently preserved old control also
ran its existing 32-repeat direct protocol there. Original snapshots were not
overwritten. `LSC_STAGED_CUDA_BASELINE` optionally selects a matching before
capture; `LSC_CUDA_DIRECT_REPEATABILITY` selects independently measured bounds.
The preservation fixture's default paths and numerical criteria are unchanged.

```bash
cd /mnt/home/noneill/Programs/symmetrix-dev
# Captured before-code/control data for the H100 used in this stage:
export LSC_STAGED_CUDA_BASELINE="$PWD/.lsc-reference-logs/distributed-mesh-before-cuda.npz"
export LSC_CUDA_DIRECT_BASELINE="$PWD/.lsc-reference-logs/distributed-mesh-old-h100-baseline.npz"
export LSC_CUDA_DIRECT_REPEATABILITY="$PWD/.lsc-reference-logs/distributed-mesh-old-h100-repeatability.json"
LSC_REUSE_BUILD=1 srun --jobid=YOUR_H100_JOB --overlap -N1 -n1 -c4 --exact \
    bash -l tools/lsc/run_distributed_mesh_tests.sh cuda
bash -l tools/lsc/run_distributed_mesh_tests.sh cpu
```

Use hardware-matched before/control captures if changing GPUs. Omit reuse to
rebuild first. The focused run includes all prior one-rank preservation checks;
the full suite retains original ASE, analytic/finite-difference, direct/mesh and
LAMMPS point/short-MD checks. Complete normal multi-rank MD, atom migration,
longer trajectories, GPU assignment and scaling/profiling remain next.


### Normal multi-rank LAMMPS MD

`test_lammps_lsc_mpi_md.py` uses the ordinary LAMMPS executable and `run`
commands, without the staged diagnostic driver. CPU double precision and both
CUDA neural precisions compare one and two ranks for periodic dimer/bulk
energies, forces, global stress, dipoles, partial charges and per-atom energies,
with zero/nonzero fields. Ten 0.1 fs NVE steps in two consecutive runs force
atom migration, atom sorting and neighbor rebuilding. Empty-atom ranks and
box/tilt changes are checked separately. Saved pre-change one-rank trajectories
check preservation (CPU bitwise; CUDA within existing roundoff tolerances).
The frozen ASE references and native kernels are unchanged.

Run `bash -l tools/lsc/run_mpi_md_tests.sh cpu` on the login node or the `cuda`
variant within the H100 allocation. The pre-build captures are
`.lsc-reference-logs/mpi-md-before-{cpu,cuda}.npz` with SHA-256 metadata.
These are additional pair-integration controls, not replacements for the
original model/ASE snapshots. Two ranks on one H100 test MPI correctness,
not multi-GPU speed or long-trajectory stability.

The normal MD suite also checks processor rebalancing on three ranks. It moves
only the first nonuniform processor split; the third rank keeps the same
bounds. LSC follows Polar's `reset_grid()` hook to invalidate plans on every
rank. Without this hook the first ranks rebuild collectively while the third
reuses its old plans; the negative control reproduces `MPI_ERR_TRUNCATE`.
The final configuration is compared with a fresh run using the same final
three-rank split, without changing the established comparison tolerances.
