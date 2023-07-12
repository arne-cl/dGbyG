import rdkit
import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader

from dGbyG.config import *
from dGbyG.utils.constants import *
from dGbyG.utils.NNFunc import mol_to_graph_data
from dGbyG.Chemistry.Compound import Compound
from dGbyG.Chemistry.Reaction import Reaction
from dGbyG.network.GNNetwork import MP_network

if inference_model_path:
    network = torch.load(inference_model_path)
elif inference_model_state_dict_path:
    network = MP_network(emb_dim=200, num_layer=2)
    network.load_state_dict(torch.load(inference_model_state_dict_path))
network.eval()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'




def predict_standard_dGf_prime(mol:rdkit.Chem.rdchem.Mol) -> np.ndarray:
    graph_data = mol_to_graph_data(mol)
    for data in DataLoader([graph_data], batch_size=1):
        data.to(device)

    with torch.no_grad():
        cids_energy = network(data).view(1)

    return cids_energy.cpu().numpy()

    
    


class Compound(Compound):
    def __init__(self, mol:rdkit.Chem.rdchem.Mol) -> None:
        super().__init__(mol)

    @property
    def standard_dGf_prime(self) -> np.ndarray:
        standard_dg = predict_standard_dGf_prime(self.mol)
        return standard_dg
    
    @property
    def transformed_standard_dGf_prime(self) -> np.ndarray:
        transformed_standard_dg = self.standard_dGf_prime + self.transform(default_condition, self.condition)
        return transformed_standard_dg
    



class Reaction(Reaction):
    def __init__(self, reaction, rxn_type='str', cid_type='smiles') -> None:
        
        if cid_type == 'compound':
            self.rxn = compound_dict if not False in compound_dict else False
        else:
            super().__init__(reaction, rxn_type, cid_type)
            compound_dict = dict(map(lambda item: (Compound(item[0]), item[1]),
                                     self.mol_dict.items()))
            self.rxn = compound_dict if not False in compound_dict else False

    @property
    def standard_dGr_prime(self) -> np.ndarray:
        standard_dGr = 0
        for comp, coeff in self.rxn.items():
            standard_dGr += coeff * comp.standard_dGf_prime
        return standard_dGr
    
    @property
    def transformed_standard_dGr_prime(self):
        transformed_standard_dGr_prime = 0
        for comp, coeff in self.rxn.items():
            transformed_standard_dGr_prime += coeff * comp.transformed_standard_dGf_prime
        return transformed_standard_dGr_prime