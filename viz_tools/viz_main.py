import os
import random
import sys
import numpy as np
import torch
import argparse
from ase.io import read
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.transforms import RandomRotate
from ocpmodels.datasets import (
    TrajectoryLmdbDataset,
    data_list_collater,
    SinglePointLmdbDataset,
)
from ocpmodels.models import *
from dimenet_plus_plus import DimeNetPlusPlusWrap
from ocpmodels.preprocessing import AtomsToGraphs
import yaml
from viz_utils import *

def main(args):
    with open(args.config_yml) as f:
        yml_file = yaml.safe_load(f)

    model_name = yml_file["model"].pop('name', None)
    print(f"#### Loading model: {model_name}")

    checkpoint_path = yml_file['checkpoint']['src']
    checkpoint = modify_checkpoint(checkpoint_path)

    model = DimeNetPlusPlusWrap(**yml_file["model"])
    model.load_state_dict(checkpoint)
    model = OCPDataParallel(model, output_device=0, num_gpus=1)

    if yml_file['dataset']['src']:
        batch = lmdb_to_batch(yml_file['dataset']['src'])
    else:
        atoms = ase.io.read(
                "../tests/models/atoms.json",
                index=0,
                format="json",
        )
        a2g = AtomsToGraphs(
                max_neigh=12,
                radius=6,
                dummy_distance=7,
                dummy_index=-1,
                r_energy=True,
                r_forces=True,
                r_distances=True,
        )
        batch = Batch.from_data_list(a2g.convert_all([atoms]))

    output = model(batch)
    viz = model_viz(checkpoint_path)

    if yml_file["task"]["computation_graph"]:
        print("#### Plotting computation graph")
        viz.computation_graph(model, batch)
    if yml_file["task"]["t-sne_viz"]:
        print("#### Plotting t-sne")
        emb_weight = checkpoint["emb.emb.weight"].cpu().numpy()
        viz.tsne_viz_emb(emb_weight)
    if yml_file["task"]["pca_t-sne_viz"]:
        print("#### Plotting PCA reduced t-sne")
        emb_weight = checkpoint["emb.emb.weight"].cpu().numpy()
        res = viz.pca(emb_weight, n=50)
        viz.tsne_viz_emb(res)
    if yml_file["task"]["raw_weights"]:
        print("#### Plotting raw emb weights")
        emb_weight = checkpoint["emb.emb.weight"].cpu().numpy()
        viz.raw_weights_viz(emb_weight)
    if yml_file["task"]["is2rs_plot"]:
        print("#### Plotting is2rs comparison")
        viz.create_is2rs_plots(batch, output)

if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yml",
        help="Path to config file",
    )
    args = parser.parse_args()
    main(args)
