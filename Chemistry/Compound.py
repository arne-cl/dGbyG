import numpy as np
from rdkit import Chem

from utils.constants import *
from utils.func import *



class Compound(object):
    def __init__(self, mol) -> None:
        self.mol = mol
        

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
    

    def standard_dGf(self):
        pass
