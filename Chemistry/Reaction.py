from typing import Dict, List
import numpy as np
from rdkit import Chem

from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *
from dGbyG.Chemistry.Compound import Compound



class Reaction(object):
    def __init__(self, reaction, cid_type='smiles') -> None:
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
            elif isinstance(comp, Chem.rdchem.Mol):
                comp = Compound(comp)
            elif isinstance(comp, str):
                mol = to_mol(comp, cid_type)
                comp = Compound(mol)
            else:
                raise ValueError('Cannot accept type{0}'.format(type(comp)))
            self.reaction.update({comp:float(coeff)})

        self.rxn = self.balance(self.reaction, with_H2O=True, with_H_ion=True)

    
    @property
    def rxnSmiles(self) -> str:
        rxn_dict_smiles = map(lambda item: (item[0].Smiles, item[1]), self.rxn.items())
        return dict(rxn_dict_smiles)
    
    @property
    def rxnInChI(self) -> str:
        rxn_dict_inchi = map(lambda item: (item[0].InChI, item[1]), self.rxn.items())
        return dict(rxn_dict_inchi)
    
    @property
    def equationSmiles(self) -> str:
        return build_equation(self.rxnSmiles)
    
    @property
    def equiationInChI(self) -> str:
        return build_equation(self.rxnInChI)
    
    @property
    def substrates(self) -> Dict[Compound, float]:
        return dict([(c,v) for c,v in self.rxn.items() if v<0])
    
    @property
    def products(self) -> Dict[Compound, float]:
        return dict([(c,v) for c,v in self.rxn.items() if v>0])

    
    
    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.rxn:
            pKa.append(compound.pKa(temperature))
        return pKa
    
    
    
    def is_balanced(self, ignore_H_ion=False, ignore_H2O=False) -> bool:
        diff_atom = {}
        for comp, coeff in self.rxn.items():
            if comp.atom_bag==None:
                return None
            for atom, num in comp.atom_bag.items():
                diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num

        if (diff_atom.get('R', 0) + diff_atom.get('*', 0)) == 0:
            diff_atom.pop('R', None)
            diff_atom.pop('*', None)

        if ignore_H_ion:
            diff_atom['H'] = diff_atom.get('H', 0) - diff_atom.get('charge', 0)
            diff_atom['charge'] = 0
        if ignore_H2O:
            diff_atom['H'] = diff_atom.get('H', 0) - 2 * diff_atom.get('O', 0)
            diff_atom['O'] = 0
            
        unbalanced_atom = {}
        for atom, num in diff_atom.items():
            if num!=0:
                unbalanced_atom[atom] = num

        return False if unbalanced_atom else True
    
    
        
    def balance(self, reaction: Dict[Compound, float], with_H2O=True, with_H_ion=True) -> Dict[Compound, float]:
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


    
    @property
    def can_be_transformed(self) -> bool:
        for x in self.rxn.keys():
            if not x.can_be_transformed:
                return False
        return True

    def transform(self, condition1, condition2):
        if self.can_be_transformed:
            ddGf_list = [coeff * comp.transform(condition1, condition2) for comp, coeff in self.rxn.items()]
            return np.sum(ddGf_list)
        else:
            return None
        

    
