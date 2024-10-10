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
from dGbyG.Chemistry import _Compound, Reaction

from .utils import predict_standard_dGf_prime, predict_standard_dGr_prime


class Compound(_Compound):
    def __init__(self, mol, input_type = 'mol') -> None:
        if input_type=='mol':
            mol = mol
        else:
            mol = to_mol(mol, input_type)
        super().__init__(mol)
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
    




class Reaction(Reaction):
    def __init__(self, reaction, cid_type='compound', balance_it=True) -> None:
        self.input_reaction = reaction
        if isinstance(reaction, str):
            self.reaction_dict = parse_equation(reaction)
        elif isinstance(reaction, dict):
            self.reaction_dict = reaction
        else:
            raise ValueError('Cannot accept type{0}'.format(type(reaction)))
        
        self.reaction = {}
        for comp, coeff in self.reaction_dict.items():
            if isinstance(comp, Compound):
                pass
            elif isinstance(comp, rdkit.Chem.rdchem.Mol):
                comp = Compound(comp)
            elif isinstance(comp, str):
                mol = to_mol(comp, cid_type)
                comp = Compound(mol)
            else:
                raise ValueError('Cannot accept type{0}'.format(type(comp)))
            self.reaction.update({comp:coeff})

        if balance_it:
            self.rxn = self.balance(self.reaction)
        elif not balance_it:
            self.rxn = self.reaction

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
            
            
    
    def balance(self, reaction: Dict[Compound, float], with_H2O=True, with_H_ion=True):
        reaction = reaction.copy()
        diff_atom = {}
        for comp, coeff in reaction.items():
            if comp.atom_bag is None:
                return reaction
            for atom, num in comp.atom_bag.items():
                diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num
        num_H_ion = diff_atom.get('charge')
        num_H2O = diff_atom.get('O')

        compounds_smiles = [comp.Smiles for comp in reaction.keys()]
        if with_H_ion and num_H_ion:
            if '[H+]' not in compounds_smiles:
                reaction[Compound(to_mol('[H+]', cid_type='smiles'))] = -num_H_ion
            else:
                for comp in reaction.keys():
                    if comp.Smiles == '[H+]':
                        reaction[comp] = reaction[comp] - num_H_ion
                        break

        if with_H2O and num_H2O:
            if '[H]O[H]' not in compounds_smiles:
                reaction[Compound(to_mol('[H]O[H]', cid_type='smiles'))] = -num_H2O
            else:
                for comp in reaction.keys():
                    if comp.Smiles == '[H]O[H]':
                        reaction[comp] = reaction[comp] - num_H2O
                        break

        return reaction
    

