import os
from ase.constraints import FixAtoms
import ase
import matplotlib
import matplotlib.pyplot as plt
import torch
from ase import Atoms
from ase.visualize.plot import plot_atoms
import datetime
import os
import sys
import json
import glob
import yaml
import math
import numpy as np
import re
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import pandas as pd

import ase
from ase.io.trajectory import Trajectory
import pickle
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchviz import make_dot
from torch.utils.data import DataLoader

from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
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

class model_viz:
    def __init__(self, checkpoint_path, viz_path=os.getcwd()):
        self.viz_path = os.path.join(viz_path, "visuals")
        self.checkpoint_path = checkpoint_path

    def computation_graph(self, model, data):
        out = model(data)
        dot = make_dot(out, params=dict(list(model.named_parameters()) + [("pos", data.pos)]))
        dot.format = "png"
        path = os.path.join(self.viz_path, "computation_graph")
        os.makedirs(path, exist_ok=True)
        dot.render(path)

    def tsne_viz_emb(self, emb):
        tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300)
        res = tsne.fit_transform(emb)
        df = pd.DataFrame()
        df["y"] = np.arange(1, 96)
        df["tsne-dim-1"] = res[:, 0]
        df["tsne-dim-2"] = res[:, 1]
        plt.figure(figsize=(16,10))
        sns_plot = sns.scatterplot(
            x="tsne-dim-1", y="tsne-dim-2",
            hue="y",
            palette=sns.color_palette("hls", 95),
            data=df,
            legend="full",
            alpha=0.3
        )
        plt.savefig(os.path.join(self.viz_path, "tsne_emb.png"))

    def pca(self, emb_weight, n):
        pca = PCA(n_components=n)
        res = pca.fit_transform(emb_weight)
        return res

    def raw_weights_viz(self, emb_weight):
        f, (ax1, ax2) = plt.subplots(1,2,sharey=True)
        g1 = sns.heatmap(emb_weight.T, cmap="Reds", yticklabels=False, xticklabels=5, cbar=False, ax=ax1)
        g2 = sns.heatmap(-1.5 + 3.0 * np.random.rand(512, 95), cmap="Reds", yticklabels=False, xticklabels=5, ax=ax2)
        f.savefig(os.path.join(self.viz_path, "raw_emb_weights.png"))

    def create_is2rs_plots(self, batch, output):
        atom_sets = batch_to_atoms(batch[0], output)

        for idx, system in enumerate(atom_sets):
            randomid = batch[0].sid[idx]
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
            fig.savefig(f"{self.viz_path}/{randomid}_is2rs.png")

def modify_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    new_dict = {}
    for k, v in checkpoint["state_dict"].items():
        name = k[14:]
        new_dict[name] = v
    return new_dict

def lmdb_to_batch(path):
    dataset = SinglePointLmdbDataset({'src': path})
    collater = ParallelCollater(1)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collater,
    )
    data = dataset[0]
    batch = collater([data])
    return batch

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
