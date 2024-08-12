from typing import Dict, List
import numpy as np
import rdkit
from rdkit import Chem
from functools import lru_cache

from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *



class Compound(object):
    def __init__(self, mol:rdkit.Chem.rdchem.Mol) -> None:
        self.input_mol = mol
        self.mol = normalize_mol(mol) if mol else None
        self.atom_bag = atom_bag(self.mol) if mol else None
        self._condition = default_condition.copy()
        self._l_concentration = None
        self._u_concentration = None
        self._lz = None
        self._uz = None


    @property
    def Smiles(self) -> str:
        return Chem.MolToSmiles(self.mol, canonical=True) if self.mol else None
    
    @property
    def InChI(self) -> str:
        return Chem.MolToInchi(self.mol) if self.mol else None
    
    @lru_cache(16)
    def pKa(self, temperature=default_T):
        return get_pKa(self, temperature) if self.mol else None
    
    
    @property
    def condition(self) -> Dict[str, float]:
        return self._condition
    
    @condition.setter
    def condition(self, condition: Dict[str, float or int]):
        for k, v in condition.items():
            self._condition[k] = float(v)

    @property
    def uz(self):
        return self._uz
    
    @uz.setter
    def uz(self, uz:float):
        self._uz = uz
        self._u_concentration = 10 ** uz
        
    @property
    def lz(self):
        return self._lz
    
    @lz.setter
    def lz(self, lz:float):
        self._lz = lz
        self._l_concentration = 10 ** lz

    @property
    def u_concentration(self) -> float:
        return self._u_concentration
    
    @u_concentration.setter
    def u_concentration(self, concentration:float):
        self._u_concentration = concentration
        self._uz = np.log10(concentration)

    @property
    def l_concentration(self) -> float:
        return self._l_concentration
    
    @l_concentration.setter
    def l_concentration(self, concentration:float):
        self._l_concentration = concentration
        self._lz = np.log10(concentration)
    

    @property
    def can_be_transformed(self) -> bool:
        if self.Smiles == '[H+]':
            return True
        return True if self.pKa(default_T) else False
    
    
    def transform(self, condition1, condition2):
        if self.Smiles == '[H+]':
            return 0
        return ddGf(self, condition1, condition2) if self.can_be_transformed else None
    
    
    def ddGf(self):
        if self.Smiles == '[H+]':
            return False
        return ddGf_to_single(self, self.condition) if self.can_be_transformed else None
    
    


