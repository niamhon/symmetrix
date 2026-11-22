"""ASE front end for native LSC inference, independent of PyTorch and LAMMPS."""

import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes


class MACELSC(Calculator):
    """Evaluate an exported LocalSplitCharges JSON with Symmetrix.

    CPU arithmetic is float64; CUDA supports float32 and float32-mixed.
    ``electrostatics='auto'`` uses the native mesh for fully periodic cells and
    direct real-space electrostatics for clusters. Partial periodicity is not
    supported. Charges are model predictions, not adjustable input charges.

    Energy/forces and periodic stress are evaluated together. Atomic energies,
    charges and dipole are read on request from that same evaluation. The
    compiled neighbor list is rebuilt for each changed configuration, as in
    the public Polar ASE calculator. No neighbor skin/cache is used.
    """

    implemented_properties = [
        "energy", "free_energy", "energies", "forces", "stress", "charges", "dipole"
    ]

    def __init__(self, model_file, device="cuda", dtype=None, electrostatics="auto",
                 mesh_spacing=0.7, mesh_order=8, external_field=None, **kwargs):
        super().__init__(**kwargs)
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        dtype = dtype or ("float64" if device == "cpu" else "float32")
        allowed = ("float64",) if device == "cpu" else ("float32", "float32-mixed")
        if dtype not in allowed:
            raise ValueError(f"dtype for {device} must be one of {allowed}")
        if electrostatics not in ("auto", "direct", "mesh"):
            raise ValueError("electrostatics must be 'auto', 'direct' or 'mesh'")
        if not np.isfinite(mesh_spacing) or mesh_spacing <= 0:
            raise ValueError("mesh_spacing must be finite and positive")
        if not isinstance(mesh_order, (int, np.integer)) or not 4 <= mesh_order <= 10:
            raise ValueError("mesh_order must be an integer between 4 and 10")
        field = None if external_field is None else self._vector(external_field)
        from ._mace_foundation import neighbor_list
        if device == "cpu":
            from .symmetrix import MACELSC as Evaluator
            evaluator = Evaluator(str(model_file))
        else:
            from .symmetrix import MACELSCCuda
            evaluator = MACELSCCuda(str(model_file), dtype)
            evaluator.use_batched_reverse = True
            evaluator.use_batched_source_reverse = True
            evaluator.use_batched_source_forward = True
            evaluator.use_compact_first_layer = True
        self.evaluator = evaluator
        self._device, self._dtype = device, dtype
        self._electrostatics = electrostatics
        self._mesh_spacing, self._mesh_order = float(mesh_spacing), int(mesh_order)
        self.external_field = field
        self._neighbor_list = neighbor_list
        self._model_numbers = [int(z) for z in evaluator.atomic_numbers]
        self._types = np.full(119, -1, dtype=np.int32)
        self._types[self._model_numbers] = np.arange(len(self._model_numbers), dtype=np.int32)
        self._solver = None
        self._last_field = None
        self._evaluated = False

    # Changing solver/hardware requires a new calculator, preventing stale ASE
    # results or a mesh hook configured with different constructor settings.
    device = property(lambda self: self._device)
    dtype = property(lambda self: self._dtype)
    electrostatics = property(lambda self: self._electrostatics)
    mesh_spacing = property(lambda self: self._mesh_spacing)
    mesh_order = property(lambda self: self._mesh_order)

    @staticmethod
    def _vector(value):
        field = np.asarray(value, dtype=np.float64)
        if field.shape != (3,) or not np.isfinite(field).all():
            raise ValueError("external_field must contain three finite components")
        return tuple(float(v) for v in field)

    def _field(self, atoms):
        if self.external_field is not None:
            return self._vector(self.external_field)
        return self._vector(atoms.info.get("external_field", (0.0, 0.0, 0.0)))

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        # Store a tuple: ASE's copy of info may share a mutable list/array.
        if self._last_field is not None and self._field(atoms) != self._last_field:
            changes.append("external_field")
        return changes

    def reset(self):
        super().reset()
        self._evaluated = False

    def get_spin_polarized(self):
        return False

    def _boundary(self, atoms):
        count = int(np.count_nonzero(atoms.pbc))
        if count not in (0, 3):
            raise ValueError("MACELSC supports clusters or cells periodic in all three directions")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("positions and cell must be finite")
        if count == 3 and not abs(np.linalg.det(atoms.cell.array)) > 0:
            raise ValueError("A periodic MACELSC calculation requires a nonsingular cell")
        solver = self.electrostatics
        if solver == "auto":
            solver = "mesh" if count == 3 else "direct"
        if solver == "mesh" and count != 3:
            raise ValueError("Mesh electrostatics requires three periodic directions")
        return count == 3, solver

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        unknown = set(properties) - set(self.implemented_properties)
        if unknown:
            raise PropertyNotImplementedError(f"Unsupported properties: {sorted(unknown)}")
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        periodic, solver = self._boundary(a)
        if "stress" in properties and not periodic:
            raise PropertyNotImplementedError("MACELSC stress is defined for fully periodic cells")
        field = self._field(a)
        if not self._evaluated or system_changes or field != self._last_field:
            self.results = {}
            self._evaluated = False
            numbers = np.asarray(a.numbers)
            if (numbers.min(initial=1) < 1 or numbers.max(initial=1) > 118
                    or (self._types[numbers] < 0).any()):
                missing = sorted(set(numbers.tolist()) - set(self._model_numbers))
                raise ValueError(f"atomic numbers {missing} are not in the model")
            if solver != self._solver:
                if solver == "mesh":
                    self.evaluator.set_mesh_electrostatics(self.mesh_spacing, self.mesh_order)
                else:
                    self.evaluator.set_electrostatics_hook(None)
                self._solver = solver
            x = np.ascontiguousarray(a.positions, dtype=np.float64)
            cell = np.ascontiguousarray(a.cell.array, dtype=np.float64)
            pbc = np.ascontiguousarray(a.pbc, dtype=bool)
            i, j, shift = self._neighbor_list(x, cell, pbc, float(self.evaluator.r_cut))
            edges = np.ascontiguousarray([i, j], dtype=np.int32)
            vectors = np.ascontiguousarray(x[j] - x[i] + shift @ cell)
            types = np.ascontiguousarray(self._types[numbers])
            self.evaluator.compute_energy_forces(
                types, edges, vectors, x, cell, pbc,
                "mixed_periodic" if periodic else "realspace", field,
                compute_stress=periodic)
            if self.device == "cuda":
                forces = self.evaluator.download_forces(False)["forces"]
            else:
                forces = np.asarray(self.evaluator.forces).reshape(-1, 3).copy()
            energy = float(self.evaluator.energy)
            self.results = dict(energy=energy, free_energy=energy, forces=forces)
            if periodic:
                self.results["stress"] = np.asarray(self.evaluator.stress).copy()
            self._last_field = field
            self._evaluated = True
        for prop in properties:
            if prop in self.results:
                continue
            if prop == "charges":
                self.results[prop] = (self.evaluator.download_atomic_charges()
                                      if self.device == "cuda" else
                                      np.asarray(self.evaluator.partial_charges).copy())
            elif prop == "dipole":
                self.results[prop] = np.asarray(self.evaluator.dipole).copy()
            elif prop == "energies":
                self.results[prop] = np.asarray(self.evaluator.download_node_energies()).copy()
