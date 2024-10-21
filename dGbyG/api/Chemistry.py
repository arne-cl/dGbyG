import time
from typing import Any, Dict, Tuple, Union
import rdkit
import numpy as np
import pandas as pd
from functools import lru_cache

import torch
from torch_geometric.loader import DataLoader

from dGbyG.config import *
from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *
from dGbyG.Chemistry import _Compound, _Reaction
from dGbyG.utils.CustomError import *

from .utils import predict_standard_dGf_prime, predict_standard_dGr_prime


class Compound(_Compound):
    def __init__(self, mol: Union[rdkit.Chem.rdchem.Mol, str], mol_type:str=None, input_type:str = 'mol') -> None:
        if mol_type is None:
            mol_type = input_type
        if isinstance(mol_type, str):
            if mol_type=='mol':
                mol = mol
            else:
                mol = to_mol(mol, mol_type)
            super().__init__(mol)
        else:
            raise InputValueError(f'mol_type must be a string, but got {type(mol_type)}')
        self.name = None
        self.compartment = None
        
    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime_list(self) -> Union[np.ndarray, None]:
        if self.mol:
            return predict_standard_dGf_prime(self.mol).squeeze()
        else:
            return None
    
    @property
    @lru_cache(maxsize=None)
    def standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        if self.standard_dGf_prime_list is not None:
            return np.mean(self.standard_dGf_prime_list), np.std(self.standard_dGf_prime_list)
        else:
            return np.nan, np.nan
    
    @property
    def transformed_ddGf(self) -> Union[np.float32, None]:
        if self.can_be_transformed:
            return self.transform(default_condition, self.condition)
        else:
            return None
    
    @property
    def transformed_standard_dGf_prime(self) -> Tuple[np.float32, np.float32]:
        if self.can_be_transformed:
            transformed_standard_dg = (self.standard_dGf_prime[0] + self.transformed_ddGf)
            return transformed_standard_dg, self.standard_dGf_prime[1]
        else:
            return self.standard_dGf_prime
    




class Reaction(_Reaction):
    '''
    '''
    def __init__(self, reaction, mol_type:str) -> None:
        if isinstance(reaction, str):
            reaction_dict = parse_equation(reaction)
        elif isinstance(reaction, dict):
            reaction_dict = reaction
        else:
            raise InputValueError(f'Cannot accept type{type(reaction)} as the input of reaction.')
        
        if not isinstance(mol_type, str):
            raise InputValueError(f'Type of mol_type should be str, but got type{mol_type}')
        
        self.reaction = {}
        for comp, coeff in reaction_dict.items():
            if isinstance(comp, Compound) or mol_type.lower()=='compound':
                if not (isinstance(comp, Compound) and mol_type.lower()=='compound'):
                    raise InputValueError(f"Key's of reaction is {type(comp)}, but mol_type is {mol_type}")
            elif isinstance(comp, rdkit.Chem.rdchem.Mol) or mol_type.lower()=='mol':
                if not (isinstance(comp, rdkit.Chem.rdchem.Mol) and mol_type.lower()=='mol'):
                    raise InputValueError(f"Key's of reaction is {type(comp)}, but mol_type is {mol_type}")
                else:
                    comp = Compound(comp)
            elif isinstance(comp, str):
                mol = to_mol(comp, mol_type)
                comp = Compound(mol)
            else:
                raise InputValueError('Cannot accept type{0}'.format(type(comp)))
            self.reaction.update({comp:coeff})
            
        super().__init__(self.reaction)
        
    @property
    @lru_cache(maxsize=None)
    def standard_dGr_prime_list(self) -> Union[np.ndarray, None]:
        if self.is_balanced() == True:
            standard_dGr_list = np.sum([comp.standard_dGf_prime_list * coeff for comp, coeff in self.rxn.items()], axis=0)
            return standard_dGr_list
        else:
            return None

    @property
    @lru_cache(maxsize=None)
    def standard_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        # 
        if self.standard_dGr_prime_list is not None:
            return np.mean(self.standard_dGr_prime_list), np.std(self.standard_dGr_prime_list)
        else:
            return np.nan, np.nan
        
    @property
    def transformed_standard_dGr_prime(self) -> Tuple[np.float32, np.float32]:
        # 
        if self.can_be_transformed:
            transformed_ddGr = np.sum([comp.transformed_ddGf * coeff for comp, coeff in self.rxn.items()], axis=0)
            transformed_standard_dGr_prime = self.standard_dGr_prime[0] + transformed_ddGr
            return transformed_standard_dGr_prime, self.standard_dGr_prime[1]
        else:
            return self.standard_dGr_prime
            
    

