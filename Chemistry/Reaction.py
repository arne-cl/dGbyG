import numpy as np
from rdkit import Chem

from dGbyG.utils.constants import *
from dGbyG.utils.func import *
from dGbyG.Chemistry.Compound import Compound
from dGbyG.api.inference import predict


class Reaction(object):
    def __init__(self, reaction, rxn_type='str', cid_type='smiles') -> None:
        if cid_type == 'compound':
            self.rxn = compound_dict if not False in compound_dict else False
        else:
            if rxn_type == 'str':
                self.reaction = reaction
                self.equation_dict = parse_equation(reaction)
                self.mol_dict = equation_to_mol_dict(reaction, cid_type)
            elif rxn_type == 'dict' and cid_type == 'mol':
                self.mol_dict = reaction
            else:
                self.mol_dict = equation_dict_to_mol_dict(reaction, cid_type)

            compound_dict = dict(map(lambda item: (Compound(item[0]), item[1]),
                                     self.mol_dict.items()))
            self.rxn = compound_dict if not False in compound_dict else False
            

    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.rxn:
            pKa.append(compound.pKa(temperature))
        return pKa
    
    
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
    

    @property
    def standard_dGr_prime(self):
        return self._standard_dGr_prime
    
    @standard_dGr_prime.setter
    def standard_dGr_prime(self, conditions):
        self._standard_dGr_prime = 0
        for comp, coeff in self.rxn:
            self._standard_dGr_prime = 3
