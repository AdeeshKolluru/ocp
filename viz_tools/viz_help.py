import os
import random
import sys
import numpy as np
import torch
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
from ocpmodels.preprocessing import AtomsToGraphs
import yaml

yaml_link = "sample_config.yml"

with open(yaml_link) as f:
    yml_file = yaml.load(f)

model_name = yml_file["model"].pop('name', None)

def load_model():
    try:
        model = model_name(**yml_file["model"])
    except:
        print("model doesn't exist")
    return model

if __name__ == "__main__":
    torch.manual_seed(0)
    checkpoint_path = yml_file['checkpoint']['src']
    checkpoint = torch.load(checkpoint_path)

    new_dict = {}
    for k, v in checkpoint["state_dict"].items():
        name = k[14:]
        new_dict[name] = v

    model = load_model()
    model.load_state_dict(new_dict)
    model = OCPDataParallel(model, output_device=0, num_gpus=1)
    dataset = SinglePointLmdbDataset(
    )

    collater = ParallelCollater(1)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        collate_fn=collater,
    )
    data = dataset[5]
    batch = collater([data])
    output1 = model(batch)

