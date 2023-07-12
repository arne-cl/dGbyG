import numpy as np
import rdkit
from rdkit import Chem

from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *



class Compound(object):
    def __init__(self, mol:rdkit.Chem.rdchem.Mol) -> None:
        self.mol = normalize_mol(mol)
        self._condition = default_condition.copy()
        self._l_concentration = default_l_concentration
        self._u_concentration = None


    @property
    def Smiles(self):
        return Chem.MolToSmiles(self.mol, canonical=True)
    
    @property
    def InChI(self):
        return Chem.MolToInchi(self.mol)
    
    @property
    def atom_bag(self):
        return atom_bag(self.mol)
    

    def pKa(self, temperature=default_T):
        return get_pKa(self, temperature)
    
    
    @property
    def condition(self):
        return self._condition
    
    @condition.setter
    def condition(self, condition:dict):
        for k, v in condition.items():
            self._condition[k] = v

    @property
    def u_concentration(self):
        return self._u_concentration
    
    @u_concentration.setter
    def u_concentration(self, concentration:float):
        self._u_concentration = concentration

    @property
    def l_concentration(self):
        return self._l_concentration
    
    @l_concentration.setter
    def l_concentration(self, concentration:float):
        self._l_concentration = concentration
    

    @property
    def can_be_transformed(self):
        if self.Smiles == '[H+]':
            return True
        return True if self.pKa(default_T) else False
    
    
    def transform(self, condition1, condition2):
        if self.Smiles == '[H+]':
            return False
        return ddGf(self, condition1, condition2) if self.can_be_transformed else None
    
    
    def ddGf(self):
        if self.Smiles == '[H+]':
            return False
        return ddGf_to_single(self, self.condition) if self.can_be_transformed else None
    
    


