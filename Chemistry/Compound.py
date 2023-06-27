import numpy as np
from rdkit import Chem

from dGbyG.utils.constants import *
from dGbyG.utils.func import *
from dGbyG.api.inference import predict


class Compound(object):
    def __init__(self, mol) -> None:
        self.mol = mol
        self._condition = default_condition
        

    def pKa(self, temperature=default_T):
        return get_pKa(self, temperature, source='file')



    @property
    def Smiles(self):
        return Chem.MolToSmiles(self.mol)
    
    @property
    def InChI(self):
        return Chem.MolToInchi(self.mol)
    
    @property
    def atom_bag(self):
        return atom_bag(self.mol)
    
    @property
    def can_be_transformed(self):
        return True if self.pKa(default_T) else False
    
    
    def transform(self, condition1, condition2):
        return ddGf(self, condition1, condition2) if self.can_be_transform else False
    

    @property
    def condition(self):
        return self._condition
    
    @condition.setter
    def condition(self, condition:dict):
        for k, v in condition.items():
            self._condition[k] = v
    

    @property
    def standard_dGf_prime(self):
        return predict(self.mol, self.condition)
    


