import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader

from dGbyG.config import *
from dGbyG.Chemistry.Compound import Compound
from dGbyG.Chemistry.Reaction import Reaction
from dGbyG.utils.NNFunc import compound_to_graph_data
from dGbyG.network.GNNetwork import MP_network

network = MP_network(emb_dim=200, num_layer=2)
network.load_state_dict(torch.load(inference_model_path))
network.eval()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def predict(reaction:Reaction):

    network
    return


def predict_dGf(compound:Compound):
    graph_data = compound_to_graph_data(compound)
    for data in DataLoader([graph_data], batch_size=1):
        data.to(device)

    with torch.no_grad():
        cids_energy = network(data).view(1)

    return cids_energy.cpu().numpy()



