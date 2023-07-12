import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Dataset, Data

from dGbyG.Chemistry.Compound import Compound
from dGbyG.utils.ChemFunc import *
from dGbyG.utils.NNFunc import mol_to_graph_data


class Train_Dataset(Dataset):
    #
    def __init__(self, dataframe:pd.DataFrame,
                 equation_column:str,
                 dG_column:str,
                 weight_column:str=None,
                 ):
        super(Train_Dataset, self).__init__()
        self.dataframe = dataframe
        self.equation = dataframe.loc[:, equation_column].to_list()
        self.dGs = torch.tensor(dataframe.loc[:, dG_column].to_numpy(), dtype=torch.float)
        self.weight = torch.tensor(dataframe.loc[:, weight_column].to_numpy()).float() if weight_column!=None else None

    def len(self) -> int:
        return self.__len__()
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self) -> int:
        assert self.S.shape[0]==len(self.compounds)
        return self.S.shape[0]
    
    def __getitem__(self, idx) -> Data:
        mol = to_mol(self.compounds[idx], cid_type='smiles')
        mol = Compound(mol).mol
        graph_data = mol_to_graph_data(mol)
        return graph_data
    
    @property
    def S(self) -> torch.Tensor:
        S = equations_to_S(self.equation)
        return torch.tensor(S.to_numpy()).float()
    
    @property
    def compounds(self) -> list:
        compounds = equations_to_S(self.equation).index.to_list()
        return compounds
    
