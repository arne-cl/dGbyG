from typing import Dict, List
import numpy as np
from rdkit import Chem

from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *
from dGbyG.utils.CustomError import *
from dGbyG.Chemistry.Compound import _Compound

class _Reaction(object):
    def __init__(self, reaction:Dict[_Compound, float|int]) -> None:
        self.input_reaction = reaction
        if not isinstance(reaction, dict):
            raise InputValueError(f'Input of _Reaction should be a dict, but got {type(reaction)}')
        
        for comp, coeff in reaction.items():
            if not isinstance(comp, _Compound):
                raise InputValueError(f"The key's type of input dict should be Compound, but got {type(comp)}")
            elif not isinstance(coeff, (float, int)):
                raise InputValueError(f"The value's type of input dict should be float or int, but got {type(coeff)}")

        self.reaction = self.balance(reaction, with_H2O=True, with_H_ion=True)

    
    @property
    def condition(self) -> Dict[str, float]:
        conditions = {}
        for comp, coeff in self.reaction.items():
            conditions[comp] = comp.condition
        return conditions
    
    @condition.setter
    def condition(self, condition: Dict[str, float | int]):
        for comp, coeff in self.reaction.items():
            comp.condition = condition
        
    @property
    def rxnSmiles(self) -> str:
        rxn_dict_smiles = map(lambda item: (item[0].Smiles, item[1]), self.reaction.items())
        return dict(rxn_dict_smiles)
    
    @property
    def rxnInChI(self) -> str:
        rxn_dict_inchi = map(lambda item: (item[0].InChI, item[1]), self.reaction.items())
        return dict(rxn_dict_inchi)
    
    @property
    def equationSmiles(self) -> str:
        return build_equation(self.rxnSmiles)
    
    @property
    def equiationInChI(self) -> str:
        return build_equation(self.rxnInChI)
    
    @property
    def substrates(self) -> Dict[_Compound, float]:
        return dict([(c,v) for c,v in self.reaction.items() if v<0])
    
    @property
    def products(self) -> Dict[_Compound, float]:
        return dict([(c,v) for c,v in self.reaction.items() if v>0])

    
    
    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.reaction:
            pKa.append(compound.pKa(temperature))
        return pKa
    
    
    
    def is_balanced(self, ignore_H_ion=False, ignore_H2O=False) -> bool:
        mol_dict = dict([(comp.mol, coeff) for comp, coeff in self.reaction.items()])
        return is_balanced(mol_dict, ignore_H_ion=ignore_H_ion, ignore_H2O=ignore_H2O)
    
        
    def balance(self, reaction: Dict[_Compound, float], with_H2O=True, with_H_ion=True) -> Dict[_Compound, float]:
        original_reaction = reaction
        reaction = reaction.copy()
        diff_atom = {}
        for comp, coeff in reaction.items():
            for atom, num in comp.atom_bag.items():
                diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num
        num_H_ion = diff_atom.get('charge')
        num_H2O = diff_atom.get('O')

        compounds_smiles = [comp.Smiles for comp in reaction.keys()]
        if with_H_ion and num_H_ion:
            if '[H+]' not in compounds_smiles:
                reaction[_Compound(to_mol('[H+]', cid_type='smiles'))] = -num_H_ion
            else:
                for comp in reaction.keys():
                    if comp.Smiles == '[H+]':
                        reaction[comp] = reaction[comp] - num_H_ion
                        break

        if with_H2O and num_H2O:
            if '[H]O[H]' not in compounds_smiles:
                reaction[_Compound(to_mol('[H]O[H]', cid_type='smiles'))] = -num_H2O
            else:
                for comp in reaction.keys():
                    if comp.Smiles == '[H]O[H]':
                        reaction[comp] = reaction[comp] - num_H2O
                        break

        if is_balanced(dict([(comp.mol, coeff) for comp, coeff in reaction.items()]), ):
            return reaction
        else:
            return original_reaction


    
    @property
    def can_be_transformed(self) -> bool:
        for x in self.reaction.keys():
            if not x.can_be_transformed:
                return False
        return True

    def transform(self, condition1, condition2):
        if self.can_be_transformed:
            ddGf_list = [coeff * comp.transform(condition1, condition2) for comp, coeff in self.reaction.items()]
            return np.sum(ddGf_list)
        else:
            return None
        

    
