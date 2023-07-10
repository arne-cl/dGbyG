import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops

from rdkit.Chem.rdchem import HybridizationType, ChiralType, BondType, BondStereo


# Atom's features
Num_atomic_number = 119 # including the extra mask tokens
Num_atom_hybridization = len(HybridizationType.values)
Num_atom_aromaticity = 2 # Atom's aromaticity (not aromatic or aromactic)
Num_atom_chirality = len(ChiralType.values)
Num_atom_charge = 9

# Bond's features
Num_bond_type = len(BondType.values) # #including aromatic and self-loop edge(value = 22)
# probably necessary
Num_bond_atom_i = Num_atomic_number # soure atom
Num_bond_atom_j = Num_atomic_number # taget atom



class MP_layer(MessagePassing):
    def __init__(self, node_emb_dim:int, edge_emb_dim:int):
        super().__init__()
        self.edge_emb_dim = edge_emb_dim
        self.node_emb_dim = node_emb_dim
        

    def forward(self, x_emb, edge_index, edge_emb):
        # 
        x_emb = self.propagate(edge_index, x = x_emb, edge_attr = edge_emb)
        return x_emb

    def message(self, x_j, edge_attr):
        # Hadamard product is better than plus
        return x_j * edge_attr


class MP_network(nn.Module):
    def __init__(self, emb_dim:int=600, num_layer:int=2):
        super().__init__()
        self.node_emb_dim = emb_dim
        self.edge_emb_dim = emb_dim
        self.num_layer = num_layer

        '''# node embedding block'''
        self.atom_number_embedding = nn.Embedding(Num_atomic_number, self.node_emb_dim)
        self.atom_hybridization_embedding = nn.Embedding(Num_atom_hybridization, self.node_emb_dim)
        self.atom_aromaticity_embedding = nn.Embedding(Num_atom_aromaticity, self.node_emb_dim)
        self.atom_chirality_embedding = nn.Embedding(Num_atom_chirality, self.node_emb_dim)
        self.atom_charge_embedding = nn.Embedding(Num_atom_charge, self.node_emb_dim)

        
        '''# edge embedding block'''
        self.bond_type_embedding = nn.Embedding(Num_bond_type, self.edge_emb_dim)
        self.bond_atom_i_embedding = nn.Embedding(Num_bond_atom_i, self.edge_emb_dim)

        # embedding the self_loops attributes into edge_attr space
        self.self_loops_AtomType_embedding = nn.Embedding(Num_atomic_number, self.edge_emb_dim)
        self.self_loops_embedding = nn.Embedding(1, self.edge_emb_dim)
        

        '''# List of MLPs'''
        self.MP_layers = nn.ModuleList()
        for _ in range(self.num_layer):
            self.MP_layers.append(MP_layer(node_emb_dim = self.node_emb_dim, edge_emb_dim = self.edge_emb_dim))


        '''# energy linear'''
        self.energy_lin = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.node_emb_dim, self.node_emb_dim),
            nn.ReLU(),
            nn.Linear(self.node_emb_dim, self.node_emb_dim//2),
            nn.ReLU(),
            nn.Linear(self.node_emb_dim//2, 1, bias=False)
        )

        #
        self.pool = global_add_pool

        # init
        self.weight_init()


    def weight_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')
            elif isinstance(layer, nn.Embedding):
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity='relu')


    def forward(self, data):
        #
        # 
        atom = data.x 
        bond_index = data.edge_index
        bond_attr = data.edge_attr

        '''# Step 1: embedding atoms to nodes'''
        # Step 1.1: embedding atoms to nodes
        node_emb = self.atom_number_embedding(atom[:,0])
        node_emb += self.atom_hybridization_embedding(atom[:,1])
        node_emb += self.atom_aromaticity_embedding(atom[:,2])
        node_emb += self.atom_chirality_embedding(atom[:,3])
        node_emb += self.atom_charge_embedding(atom[:,4])


        '''# Step 2. embedding bonds to edges'''
        # Step 2.1: add self_loops to the adjacency matrix.
        edge_index, _ = add_self_loops(bond_index, num_nodes=node_emb.size(0))

        # Step 2.2: add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(atom.size(0), 2).to(bond_attr.device).to(bond_attr.dtype)
        self_loop_attr[:, 0] = 0
        self_loop_attr[:, 1] = atom[:, 0]
        self_loop_attr = self_loop_attr
        
        #Step 2.3: embed the bond_attr and self_loop_attr, and then torch.cat them and transform them to edge_emb
        bond_emb = self.bond_type_embedding(bond_attr[:,0])
        bond_emb += self.bond_atom_i_embedding(bond_attr[:,1])
        
        self_loop_emb = self.self_loops_AtomType_embedding(self_loop_attr[:, 1])
        self_loop_emb += self.self_loops_embedding(self_loop_attr[:, 0])
        
        edge_emb = torch.cat((bond_emb, self_loop_emb), dim = 0)
        


        '''# Step 3: graph convolution'''
        for layer in range(self.num_layer):
            node_emb = self.MP_layers[layer](node_emb, edge_index, edge_emb)

        '''# Step 4: transform x to a single value(energy value)'''
        node_energy = self.energy_lin(node_emb)

        '''# Step 5: add all the energies of nodes '''
        dg = self.pool(node_energy, data.batch)

        return dg