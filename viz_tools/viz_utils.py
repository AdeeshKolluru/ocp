import os
from ase.constraints import FixAtoms
import ase
import matplotlib
import matplotlib.pyplot as plt
import torch
from ase import Atoms
from ase.visualize.plot import plot_atoms

matplotlib.use("Agg")


params = {
    "axes.labelsize": 14,
    "font.size": 14,
    "font.family": " DejaVu Sans",
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # "axes.labelsize": 25,
    "axes.titlesize": 25,
    "text.usetex": False,
    "figure.figsize": [12, 12],
}
matplotlib.rcParams.update(params)


def create_is2rs_plots(batch, output, results_dir=os.getcwd()):
    viz_path = os.path.join(results_dir, "visuals")
    os.makedirs(viz_path, exist_ok=True)
    atom_sets = batch_to_atoms(batch, output)

    for idx, system in enumerate(atom_sets):
        randomid = batch.sid[idx]
        fig, ax = plt.subplots(1, 3)
        labels = ["initial", "predicted", "relaxed"]
        for i in range(3):
            ax[i].axis("off")
            ax[i].set_title(labels[i])

        ase.visualize.plot.plot_atoms(
            system[0], ax[0], radii=0.8, rotation=("-75x, 45y, 10z")
        )
        ase.visualize.plot.plot_atoms(
            system[1], ax[1], radii=0.8, rotation=("-75x, 45y, 10z")
        )
        ase.visualize.plot.plot_atoms(
            system[2], ax[2], radii=0.8, rotation=("-75x, 45y, 10z")
        )
        fig.tight_layout()
        fig.savefig(f"{viz_path}/{randomid}_is2rs.png")

def batch_to_atoms(batch, output):
    n_systems = batch.sid.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    target_positions = torch.split(batch.pos_relaxed, natoms)
    output_positions = torch.split(output, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell

    atoms_objects = []
    for idx in range(n_systems):
        initial_atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
        pred_atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=output_positions[idx].cpu().detach().numpy()+positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
        relaxed_atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=target_positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
 
    atoms_objects.append([initial_atoms, pred_atoms, relaxed_atoms])

    return atoms_objects

def create_atoms(predictions):
    system = []

    for idx in range(len(predictions["id"])):
        pred_atoms = Atoms(
            numbers=predictions["numbers"][idx].tolist(),
            positions=(
                predictions["positions"][idx] + predictions["pos_initial"][idx]
            ),
            cell=predictions["cell"][idx].tolist(),
            pbc=[True, True, True],
        )
        relaxed_atoms = Atoms(
            numbers=predictions["numbers"][idx].tolist(),
            positions=predictions["pos_relaxed"][idx],
            cell=predictions["cell"][idx].tolist(),
            pbc=[True, True, True],
        )
        initial_atoms = Atoms(
            numbers=predictions["numbers"][idx].tolist(),
            positions=predictions["pos_initial"][idx],
            cell=predictions["cell"][idx].tolist(),
            pbc=[True, True, True],
        )
        system.append([initial_atoms, pred_atoms, relaxed_atoms])

    return system
