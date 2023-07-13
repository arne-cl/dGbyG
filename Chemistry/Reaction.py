import numpy as np
from rdkit import Chem

from dGbyG.utils.constants import *
from dGbyG.utils.ChemFunc import *
from dGbyG.Chemistry.Compound import Compound



class Reaction(object):
    def __init__(self, reaction, rxn_type='str', cid_type='smiles') -> None:
        if cid_type == 'compound':
            self.rxn = reaction if not None in reaction else None
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
    
    
    def pKa(self, temperature=default_T):
        pKa = []
        for compound in self.rxn:
            pKa.append(compound.pKa(temperature))
        return pKa
    
    
    
    def is_balanced(self, ignore_H_ion=True, ignore_H2O=True):
        diff_atom = {}
        for comp, coeff in self.rxn.items():
            for atom, num in comp.atom_bag.items():
                diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num
        if ignore_H_ion:
            diff_atom['H'] = diff_atom.get('H', 0) - diff_atom.get('charge', 0)
            diff_atom['charge'] = 0
        if ignore_H2O:
            diff_atom['H'] = diff_atom.get('H', 0) - 2 * diff_atom.get('O', 0)
            diff_atom['O'] = 0
            
        if len(set(diff_atom.values()))==1 and 0 in diff_atom.values():
            return True
        else:
            #print(diff_atom)
            return False
        
    def balance(self, with_H_ion=True, with_H2O=True):
        diff_atom = {}
        for comp, coeff in self.rxn.items():
            for atom, num in comp.atom_bag.items():
                diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num
        num_H_ion = diff_atom.get('charge')
        num_H2O = diff_atom.get('O')

        if with_H_ion and num_H_ion:
            if '[H+]' not in self.rxnSmiles.keys():
                self.rxn[Compound(to_mol('[H+]', cid_type='smiles'))] = -num_H_ion
            else:
                for compound in self.rxn.keys():
                    if compound.Smiles == '[H+]':
                        self.rxn[compound] = self.rxn[compound] - num_H_ion
                        break

        if with_H2O and num_H2O:
            if '[H]O[H]' not in self.rxnSmiles.keys():
                self.rxn[Compound(to_mol('[H]O[H]', cid_type='smiles'))] = -num_H2O
            else:
                for compound in self.rxn.keys():
                    if compound.Smiles == '[H]O[H]':
                        self.rxn[compound] = self.rxn[compound] - num_H2O
                        break


    
    @property
    def can_be_transformed(self):
        for x in self.rxn.keys():
            if not x.can_be_transformed:
                return False
        return True

    def transform(self, condition1, condition2):
        if self.can_be_transformed:
            ddGf_list = [coeff * comp.transform(condition1, condition2) for comp, coeff in self.rxn.items()]
            return sum(ddGf_list)
        

    def ddGr(self):
        if self.can_be_transformed:
            ddGf_list = [coeff * comp.ddGf() for comp, coeff in self.rxn.items()]
            return sum(ddGf_list)

    
