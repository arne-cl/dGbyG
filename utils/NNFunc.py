import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, ChiralType, BondType, BondStereo

import torch
from torch_geometric.data import Data

from dGbyG.Chemistry.Compound import Compound
from dGbyG.Chemistry.Reaction import Reaction


def mol_to_graph_data(mol:rdkit.Chem.rdchem.Mol) -> Data:
    # return (x, bond_index, bond_attr)
        # atoms features: including such below(ordered).
        atomic_number, hybridization, aromaticity, chirality, charge, degree = [], [], [], [], [], []
        features = [atomic_number, hybridization, aromaticity, chirality, charge, degree]
        atom_features = torch.empty(size=(0,len(features)), dtype=torch.int64)

        # bonds index
        bond_index = torch.empty(size=(2,0), dtype=torch.int64)
        
        # bonds attributes: including such below(ordered)
        # bond_type + bond_begin_atom_features + bond_end_atom_features
        bond_attr = torch.empty(size=(0, 2*len(features)+1), dtype=torch.int64)

        for atom in mol.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            hybridization = atom.GetHybridization().real
            aromaticity = atom.GetIsAromatic().real
            chirality = atom.GetChiralTag().real
            charge = atom.GetFormalCharge() + 4
            degree = atom.GetDegree()

            features = torch.tensor([atomic_number, hybridization, aromaticity, chirality, charge, degree])
            atom_features = torch.cat((atom_features, features.view(1,len(features))), dim=0)
            
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_index = torch.cat((bond_index, torch.tensor([start,end]).view(2,1)), dim=1)
            bond_index = torch.cat((bond_index, torch.tensor([end,start]).view(2,1)), dim=1)

            bond_type = torch.tensor(bond.GetBondType().real, dtype=torch.long).view(1)
            x_i = atom_features[start,:]
            x_j = atom_features[end,:]
            bond_i_j = torch.cat((bond_type, x_i, x_j), dim=0)
            bond_j_i = torch.cat((bond_type, x_j, x_i), dim=0)

            bond_attr = torch.cat((bond_attr, bond_i_j.view(1, 2*len(features)+1)), dim=0)
            bond_attr = torch.cat((bond_attr, bond_j_i.view(1, 2*len(features)+1)), dim=0)
        
        return Data(x=atom_features, edge_index=bond_index, edge_attr=bond_attr)


