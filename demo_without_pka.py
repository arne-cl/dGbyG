#!/usr/bin/env python
# coding: utf-8

# import package
from dGbyG.api import Compound, Reaction

# Print default condition
from dGbyG.utils.constants import default_condition
print('default_condition is', default_condition)

# Example 1: Simple reaction prediction
# Using 4-hydroxy-2-oxoglutarate = pyruvate + glyoxylate
equation = '[H]OC(=O)C(=O)C([H])([H])C([H])(O[H])C(=O)O[H] = [H]OC(=O)C(=O)C([H])([H])[H] + [H]OC(=O)C([H])=O'

# Create Reaction object without transformed conditions
reaction = Reaction(equation, mol_type='smiles')

# Print the standard Gibbs energy change (untransformed)
print("Standard Gibbs energy change:", reaction.standard_dGr_prime)

# Example 2: Transport reaction (without pH transformation)
# Create compounds
substrate = Compound('C00033', mol_type='kegg')
product = Compound('C00033', mol_type='kegg')

# Create reaction
equation_dict = {substrate:-1, product:1}
reaction = Reaction(equation_dict, mol_type='compound')

# Print the standard Gibbs energy change (untransformed)
print("\nTransport reaction standard Gibbs energy change:", reaction.standard_dGr_prime)
