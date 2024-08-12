import numpy as np
import pandas as pd
from typing import List

import torch
from torch_geometric.data import Dataset, Data

from dGbyG.Chemistry import Compound, Reaction
from dGbyG.utils.ChemFunc import *
from dGbyG.utils.NNFunc import mol_to_graph_data


class Train_Dataset(Dataset):
    #
    def __init__(self, 
                 equations:list,
                 dGs:list,
                 weights:list=None,
                 ):
        super(Train_Dataset, self).__init__()
        self.equations = equations

        self.S = torch.tensor(equations_to_S(self.equations).to_numpy()).float()
        self.cids = equations_to_S(self.equations).index.to_list()
        self.dGs = torch.tensor(dGs, dtype=torch.float)
        self.weight = torch.tensor(weights).float() if weights is not None else None

        self.mols = np.asarray([to_mol(cid, cid_type='smiles') for cid in self.cids])
        self.compounds = np.asarray([Compound(mol) for mol in self.mols])
        

    def len(self) -> int:
        return self.__len__()
    
    def get(self, idx) -> Data:
        return self.__getitem__(idx)

    def __len__(self) -> int:
        return self.S.shape[0]
    
    def __getitem__(self, idx) -> Data:
        mol = self.compounds[idx].mol
        graph_data = mol_to_graph_data(mol)
        return graph_data

    
