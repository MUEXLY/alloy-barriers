from typing import Callable
from pathlib import Path

from ase.build import bulk
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.eam import EAM
from ase.calculators.emt import EMT
from ase.mep import NEB
from ase.optimize import FIRE, BFGS
from ase.optimize.optimize import Optimizer
from ase.filters import UnitCellFilter
from ase.geometry import find_mic
from ase.io import write
import numpy as np
from dotenv import load_dotenv


Constructor = Callable[[], Calculator]


def cantor_lammps() -> LAMMPS:

    parameters = {
        "pair_style": "meam",
        "pair_coeff": ["* * library.meam Co Ni Cr Fe Mn CoNiCrFeMn.meam Co Ni Cr Fe Mn"],
        "mass": ["1 58.933", "2 58.693", "3 51.996", "4 55.845", "5 54.938"]
    }
    lammps_files = ["library.meam", "CoNiCrFeMn.meam"]

    return LAMMPS(files=lammps_files, **parameters)

def iron_chrome_eam() -> EAM:

    return EAM(potential="FeCr.eam.alloy")


def emt() -> EMT:

    return EMT()


def main():

    lattice_parameter: float = 4.073
    atoms: Atoms = bulk("Al", "fcc", a=lattice_parameter, cubic=True).repeat((5, 5, 4))
    tolerance: float = 1.0e-3
    first_neighbor_cutoff: float = (1.0 + tolerance) * 0.5 * np.sqrt(2.0) * lattice_parameter
    constructor: Constructor = emt
    relaxer: Optimizer = BFGS
    relaxer_kwargs: dict = {"fmax": 0.01}
    climber: Optimizer = FIRE
    climber_kwargs: dict = {"fmax": 0.05}
    num_paths: int = 250
    n_images: int = 7
    neb_spring: float = 0.1
    trajectory_dir: Path = Path("trajectories")
    rng: np.random.Generator = np.random.default_rng(seed=0)

    trajectory_dir.mkdir(exist_ok=True)

    num_atoms = len(atoms)
    atoms.symbols = rng.choice(["Al", "Cu", "Au"], size=num_atoms)

    atoms.calc = constructor()
    relaxer(UnitCellFilter(atoms), logfile="-").run(**relaxer_kwargs)

    forward_barriers = np.zeros(num_paths)
    backward_barriers = np.zeros(num_paths)

    for sample, vacant_idx in enumerate(rng.choice(np.arange(num_atoms), size=num_paths, replace=False)):

        print(f"{(sample + 1) / num_paths:.2%}")

        initial = atoms.copy()

        vacancy_position = initial.positions[vacant_idx].copy()
        del initial[vacant_idx]

        final = initial.copy()

        mic_deltas, _ = find_mic(final.positions - vacancy_position, atoms.cell, atoms.pbc)
        distances = np.linalg.norm(mic_deltas, axis=1)
        indices, = np.where((0.0 <= distances) & (distances <= first_neighbor_cutoff))
        
        final.positions[rng.choice(indices)] = vacancy_position

        disp, _ = find_mic(final.positions - initial.positions, atoms.cell, atoms.pbc)
        final.positions = initial.positions + disp

        initial.calc = constructor()
        relaxer(initial, logfile="-").run(**relaxer_kwargs)

        final.calc = constructor()
        relaxer(final, logfile="-").run(**relaxer_kwargs)

        images = [initial]
        for _ in range(n_images):
            images.append(initial.copy())
        images.append(final)

        neb = NEB(images, k=neb_spring)
        neb.interpolate(method="idpp")

        for image in images:
            image.calc = constructor()

        opt = climber(neb, logfile="-")
        opt.run(**climber_kwargs)

        energies = np.fromiter((image.get_potential_energy() for image in images), dtype=float)
        forward_barrier = energies.max() - energies[0]
        backward_barrier = energies.max() - energies[-1]
        assert forward_barrier > 0 and backward_barrier > 0
        
        forward_barriers[sample] = forward_barrier
        backward_barriers[sample] = backward_barrier

        write(trajectory_dir / f"trajectory_{sample + 1:.0f}.xyz", images)

    print()
    np.savetxt("barriers.txt", np.concatenate((forward_barriers, backward_barriers)))



if __name__ == "__main__":

    load_dotenv()
    main()
