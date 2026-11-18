"""ASE interface to the validated native LSC CPU/CUDA evaluators."""
import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.neighborlist import neighbor_list
from symmetrix import MACELSC, MACELSCCuda


class LSCCalculator(Calculator):
    """Evaluate an exported LSC model with CPU float64 or CUDA float32/mixed.

    The CUDA backend uses the same neural kernels as the direct LAMMPS style.
    Electrostatic boundary handling supports clusters and fully periodic cells.
    """
    implemented_properties = ["energy", "free_energy", "forces", "stress", "charges", "dipole"]

    def __init__(self, model_path, device="cpu", precision=None, pbc_handling="auto", external_field=None, **kwargs):
        super().__init__(**kwargs)
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cpu":
            if precision not in (None, "float64"):
                raise ValueError("The CPU LSC evaluator uses float64")
            self.model = MACELSC(str(model_path))
        else:
            self.model = MACELSCCuda(str(model_path), precision or "float32")
        self.device = device
        self.pbc_handling = pbc_handling
        self.external_field = external_field
        self.species = {int(z): i for i, z in enumerate(self.model.atomic_numbers)}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        try:
            types = np.array([self.species[int(z)] for z in atoms.numbers], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"LSC model does not contain atomic number {error.args[0]}") from error
        sender, receiver, vectors = neighbor_list("ijD", atoms, self.model.r_cut, self_interaction=False)
        edges = np.array([sender, receiver], dtype=np.int64)
        cell = atoms.cell.array.copy()
        stress = atoms.cell.rank == 3
        if "stress" in properties and not stress:
            raise PropertyNotImplementedError("Stress requires a nonsingular cell")
        field = self.external_field
        if field is None:
            field = atoms.info.get("external_field", (0, 0, 0))
        self.model.compute_energy_forces(types, edges, vectors, atoms.positions.copy(), cell,
                                         atoms.pbc.copy(), self.pbc_handling, field, compute_stress=stress)
        if self.device == "cuda":
            forces = self.model.download_forces(False)["forces"]
            charges = self.model.download_charges(False)["partial_charges"]
        else:
            forces = np.asarray(self.model.forces).reshape(-1, 3)
            charges = np.asarray(self.model.partial_charges)
        self.results = dict(energy=self.model.energy, free_energy=self.model.energy,
                            forces=forces, charges=charges, dipole=np.asarray(self.model.dipole))
        if stress:
            self.results["stress"] = np.asarray(self.model.stress)
