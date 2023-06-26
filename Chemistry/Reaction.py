import numpy as np
from rdkit import Chem

from .Compound import Compound
from utils.constants import *
from utils.func import *



class Reaction(object):
    def __init__(self, reaction) -> None:
        self.reaction = reaction
        self.equation_dict = parse_equation(reaction)
        

    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.rxn:
            pKa.append(compound.pKa(temperature))
        return pKa
    
    
    @property
    def rxn(self) -> dict:
        if type(self.cid_type)==str:
            cid_Types = [self.cid_type] * len(self.equation_dict)
        elif type(self.cid_type)==list:
            assert len(self.cid_type)==len(self.equation_dict)
            cid_Types = self.cid_type
        rxn_dict = dict(map(lambda item: (Compound(to_mol(item[0][0], item[1])), item[0][1]), 
                            zip(self.equation_dict.items(),cid_Types)))
        rxn_dict = rxn_dict if not False in rxn_dict else False
        return rxn_dict
    
    @property
    def rxnSmiles(self):
        rxn_dict_smiles = map(lambda item: (item[0].Smiles, item[1]), self.rxn.items())
        return dict(rxn_dict_smiles)
    
    @property
    def rxnInChI(self):
        rxn_dict_inchi = map(lambda item: (item[0].InChI, item[1]), self.rxn.items())
        return dict(rxn_dict_inchi)
    
    @property
    def equationSmiles(self):
        return build_equation(self.rxnSmiles)
    
    @property
    def equiationInChI(self):
        return build_equation(self.rxnInChI)
    
    @property
    def can_be_transformed(self):
        for x in self.rxn:
            if not x.can_be_transformed:
                return False
        return True

    def transform(self, condition1, condition2):
        return ddGr(self.rxn, condition1, condition2) if self.can_be_transform else None
    

    def standard_dGf(self):
        pass
