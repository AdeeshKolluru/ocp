import os

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


def create_is2rs_plots(predictions, results_dir):
    viz_path = os.path.join(results_dir, "visuals")
    os.makedirs(viz_path, exist_ok=True)
    atom_sets = create_atoms(predictions)

    for idx, system in enumerate(atom_sets):
        randomid = predictions["id"][idx]
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
