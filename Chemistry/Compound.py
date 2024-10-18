from typing import Dict, List, Union, Callable
import numpy as np
import rdkit
from rdkit import Chem
from functools import lru_cache

from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *
from dGbyG.utils.CustomError import *


class _Compound(object):
    def __init__(self, mol:rdkit.Chem.rdchem.Mol | None) -> None:
        '''
        mol: rdkit.Chem.rdchem.Mol or None
        '''
        self.input_mol = mol
        if isinstance(mol, rdkit.Chem.rdchem.Mol):
            self.mol = normalize_mol(mol)
            self.atom_bag = atom_bag(self.mol)
        elif mol is None:
            self.mol = None
            self.atom_bag = None
        else:
            raise InputValueError('The input of _Compound() must be rdkit.Chem.rdchem.Mol or None.')
        self._condition = default_condition.copy()
        self._l_concentration = None
        self._u_concentration = None
        self._lz = None
        self._uz = None
    
    @property
    def condition(self) -> Dict[str, float]:
        return self._condition
    
    @condition.setter
    def condition(self, condition: Dict[str, float | int]):
        if not isinstance(condition, dict):
            raise InputValueError('The input of condition must be a dict.')
        elif x:=set(condition.keys()) - set(self.condition.keys()):
            raise InputValueError(f'Condition includes {', '.join(self.condition.keys())}, but got {', '.join(x)}.')
        else:
            for k,v in condition.items():
                if isinstance(v, (int, float)):
                    self._condition[k] = float(v)
                else:
                    raise InputValueError(f'The value of {k} must be a float or int, but got {type(condition[k])}.')

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

    @lru_cache(16)
    def pKa(self, temperature=default_T):
        return get_pKa(self, temperature) if self.mol else None
    
    @property
    def can_be_transformed(self) -> bool:
        if self.Smiles == '[H+]':
            return True
        return True if self.pKa(default_T) else False
    
    def transform(self, condition1, condition2):
        if self.can_be_transformed == True:
            return ddGf(self, condition1, condition2)
        elif self.can_be_transformed == False:
            raise NoPkaError('This compound has no available Pka value, so it cannot be transformed.')
        else:
            raise ValueError('Unknown value of self.can_be_transformed')

    @property
    def Smiles(self) -> str:
        return Chem.MolToSmiles(self.mol, canonical=True) if self.mol else None
    
    @property
    def InChI(self) -> str:
        return Chem.MolToInchi(self.mol) if self.mol else None
    
    @property
    def image(self, remove_Hs=True):
        if self.mol is None:
            return None
        if remove_Hs is True:
            mol = Chem.RemoveHs(self.mol)
        else:
            mol = Chem.AddHs(self.mol)
        return Chem.Draw.MolToImage(mol)
    
    
    


