import time
from typing import Any, Dict
import rdkit
import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader

from dGbyG.config import *
from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *
from dGbyG.Chemistry import Compound, Reaction

from .utils import predict_standard_dGf_prime, predict_standard_dGr_prime


class Compound(Compound):
    def __init__(self, mol:rdkit.Chem.rdchem.Mol) -> None:
        super().__init__(mol)
        self.name = None
        self.compartment = None
        #self.standard_dGf_prime = self.compute_standard_dGf_prime()
    
    @property
    def transformed_ddGf(self):
        ddG = super().transform(default_condition, self.condition)
        return ddG if ddG else False
    
    @property
    def transformed_standard_dGf_prime(self) -> np.float32:
        ddGf = self.transform(default_condition, self.condition)
        ddGf = ddGf if ddGf else False
        transformed_standard_dg = (self.standard_dGf_prime + ddGf) if self.mol else np.nan
        return np.round(transformed_standard_dg, 3)
    
    
    def compute_standard_dGf_prime(self) -> np.float32:
        standard_dg = predict_standard_dGf_prime(self.mol).squeeze() if self.mol else np.nan
        return standard_dg






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

        
        #self.standard_dGr_prime = self.compute_standard_dGr_prime()

    @property
    def standard_dGr_prime(self) -> np.float32:
        rxn_dict = {comp.mol:coeff for comp, coeff in self.rxn.items()}
        if None in rxn_dict.keys():
            return np.nan, np.nan
        else:
            return predict_standard_dGr_prime(rxn_dict)
        

    @property
    def transformed_standard_dGr_prime(self) -> np.float32:
        transformed_standard_dGr_prime = self.standard_dGr_prime[0]
        for comp, coeff in self.rxn.items():
            transformed_standard_dGr_prime += coeff * comp.transformed_ddGf
        return transformed_standard_dGr_prime, self.standard_dGr_prime[1]

    
    
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
    

