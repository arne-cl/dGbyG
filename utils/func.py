import os, re, json, requests
from functools import reduce
from Bio.KEGG import REST
import rdkit
from rdkit import Chem
import numpy as np
import pandas as pd

from config import *
from .constants import *


def get_pKa(compound, temperature:float=default_T, source='auto') -> list:
    # compound: 
    # source: 

    def get_pka_from_chemaxon(compound, temperature):
        # 
        chemaxon_pka_api = 'https://jchem-microservices.chemaxon.com/jws-calculations/rest-v1/calculator/calculate/pka'
        headers = {'accept': '*/*', 'Content-Type': 'application/json'}
        pka_req_body = json.dumps({
            "inputFormat": "smiles",
            "micro": False,
            "outputFormat": "smiles",
            "outputStructureIncluded": False,
            "pKaLowerLimit": -20,
            "pKaUpperLimit": 20,
            "prefix": "STATIC",
            "structure": compound.Smiles,
            "temperature": temperature,
            "types": "acidic, basic",
            })
        try:
            pka = requests.post(chemaxon_pka_api, data=pka_req_body, headers=headers).json()
            return pka if not pka.get('error') else None
        except:
            return None
        
    def get_pka_from_file(compound, temperature):
        # 
        with open(pKa_json_file_path, 'r') as f:
            f = json.loads(f.read())

        return f.get(compound.Smiles).get(str(temperature))

        
    # the main body of this function
    methods = {'file': get_pka_from_file,
               'chemaxon': get_pka_from_chemaxon,}
    if source=='auto':
        for source in methods:
            if pKa := methods[source](compound, temperature=temperature):
                break
    else:
        _get_pka = methods.get(source)
        pKa = _get_pka(compound, temperature=temperature)
   
    return pKa


def atom_bag(mol:rdkit.Chem.rdchem.Mol):
    atom_bag = {}
    charge = 0
    for atom in mol.GetAtoms():
        atom_bag[atom.GetSymbol()] = 1 + atom_bag.get(atom.GetSymbol(), 0)
        charge += atom.GetFormalCharge()
    atom_bag['charge'] = charge
        
    return atom_bag


def to_mol(cid, cid_type='auto') -> rdkit.Chem.rdchem.Mol:
    #
    
    def inchi_to_mol(inchi:str):
        mol = Chem.MolFromInchi(inchi, removeHs=False)
        return Chem.AddHs(mol)
    
    def smiles_to_mol(smiles:str):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.AddHs(mol)
    
    def file_to_mol(path:str):
        mol = Chem.MolFromMolFile(path, removeHs=False)
        return Chem.AddHs(mol)
    
    def kegg_entry_to_mol(entry:str):
        path = os.path.join(kegg_compound_data_path, entry+'.mol')
        if os.path.exists(path):
            mol = file_to_mol(path)
        elif download_kegg_compound(entry):
            mol = file_to_mol(path)
        else:
            kegg_additions_df = pd.read_csv(kegg_additions_csv_path, index_col=0)
            if entry in kegg_additions_df.index:
                inchi = kegg_additions_df.loc[entry, 'inchi']
                mol = inchi_to_mol(inchi) if pd.notna(inchi) else None
        return Chem.AddHs(mol)
    
    def metanetx_id_to_mol(id:str):
        matenetx_df = pd.read_csv('./data/MetaNetX/chem_prop.tsv', sep='\t', header=351, index_col=0)
        smiles = (matenetx_df.loc[id, 'SMILES'])
        mol = smiles_to_mol(smiles)
        return Chem.AddHs(mol)

    
    # the main body
    methods = {'inchi': inchi_to_mol,
               'InChI': inchi_to_mol,
               'smiles': smiles_to_mol,
               'Smiles': smiles_to_mol,
               'file': file_to_mol,
               'kegg': kegg_entry_to_mol,
               'KEGG': kegg_entry_to_mol,
               'metanetx': metanetx_id_to_mol,
               'MetaNetX': metanetx_id_to_mol,
               }
    if cid_type=='auto':
        pass
    _to_mol = methods.get(cid_type)
    try:
        mol = _to_mol(cid)
        mol = Chem.AddHs(mol)
    except:
        mol = False
    
    return mol



def debye_hueckel(sqrt_ionic_strength: float, T_in_K: float) -> float:
    """Compute the ionic-strength-dependent transformation coefficient.

    For the Legendre transform to convert between chemical and biochemical
    Gibbs energies, we use the extended Debye-Hueckel theory to calculate the
    dependence on ionic strength and temperature.

    Parameters
    ----------
    sqrt_ionic_strength : float
        The square root of the ionic-strength in M
    temperature : float
        The temperature in K


    Returns
    -------
    Quantity
        The DH factor associated with the ionic strength at this
        temperature in kJ/mol

    """
    _a1 = 9.20483e-3  # kJ / mol / M^0.5 / K
    _a2 = 1.284668e-5  # kJ / mol / M^0.5 / K^2
    _a3 = 4.95199e-8  # kJ / mol / M^0.5 / K^3
    B = 1.6  # 1 / M^0.5
    alpha = _a1 * T_in_K - _a2 * T_in_K ** 2 + _a3 * T_in_K ** 3  # kJ / mol
    return alpha / (1.0 / sqrt_ionic_strength + B)  # kJ / mol


def ddGf_to_aqueous(pH: float, pMg: float, I: float, T: float, net_charge: float, num_H: float, num_Mg: float) -> float:
    RT = R * T
    H_term = num_H * RT * np.log(10) * pH

    _dg_mg = (T / default_T) * standard_formation_dg_Mg + (1.0 - T / default_T) * standard_formation_dh_Mg
    Mg_term = num_Mg * (RT * np.log(10) * pMg - _dg_mg)

    if I > 0:
        dh_factor = debye_hueckel(I ** 0.5, T)
        is_term = dh_factor * (net_charge ** 2 - num_H - 4 * num_Mg)
    else:
        is_term = 0.0

    return (H_term + Mg_term - is_term)


def ddGf_to_dissociation(pH: float, T: float, pKa):
    # 
    RT = R * T
    def pseudoisomers_dG(chemaxon_pKa, pH):
        acidicV = dict([(x['atomIndex'],x['value']) for x in chemaxon_pKa['acidicValuesByAtom']])
        basicV = dict([(x['atomIndex'],x['value']) for x in chemaxon_pKa['basicValuesByAtom']])
        dG_dis = np.array([0])
        t = list(set(acidicV.keys())|set(basicV.keys()))
        def iter_pseudo(dG_dis, n):
            if n==len(t):
                return dG_dis
            else:
                i = t[n]
                pka = acidicV.get(i)
                pkb = basicV.get(i)
                if pka != None:
                    dG_dis_a = dG_dis - RT * np.log(10) * (pH - pka)
                else:
                    dG_dis_a = np.array([])
                if pkb != None:
                    dG_dis_b = dG_dis - RT * np.log(10) * (pkb - pH)
                else:
                    dG_dis_b = np.array([])
                dG_dis = np.concatenate([dG_dis, dG_dis_a, dG_dis_b])
                return iter_pseudo(dG_dis, n+1)

        return iter_pseudo(dG_dis, 0)
    
    ddGf_standard_prime_j_array = pseudoisomers_dG(pKa, pH)

    ddGf_standard_prime = -RT * np.log(np.sum(np.exp(-ddGf_standard_prime_j_array/RT)))
    #ratio = np.exp(-ddGf_standard_prime_j_array/RT)/np.sum(np.exp(-ddGf_standard_prime_j_array/RT))
    #print(ratio.round(decimals=2))

    return ddGf_standard_prime


def ddGf(compound, condition1, condition2):
    num_H = compound.atom_bag['H']
    num_Mg = compound.atom_bag.get('Mg', 0)
    net_charge = compound.atom_bag.get('charge', 0)

    ddGf_1_2 = []
    for condition in [dict(condition1), dict(condition2)]:
        pH = condition.get('pH', default_pH)
        T = condition.get('T', default_T)
        I = condition.get('I', default_I)
        pMg = condition.get('pMg', default_pMg)
        pKa = compound.pKa(T)
        if pKa == None:
            return None

        ddGf_1_2.append(ddGf_to_dissociation(pH, T, pKa) + ddGf_to_aqueous(pH, pMg, I, T, net_charge, num_H, num_Mg))
    
    return ddGf_1_2[1] - ddGf_1_2[0]


def ddGr(reaction, condition1, condition2):
    ddGf_list = [coeff * ddGf(comp, condition1, condition2) for comp, coeff in reaction.items()]
    if None in ddGf_list:
        print(ddGf_list)
        return None
    return sum(ddGf_list)




def parse_equation(equation:str, eq_sign=None) -> dict:
    # 
    eq_Signs = [' = ', ' <=> ', ' -> ']
    if eq_sign:
        equation = equation.split(eq_sign)
    else:
        for eq_sign in eq_Signs:
            if eq_sign in equation:
                equation = equation.split(eq_sign)
                break
    
    equation_dict = {}
    equation = [x.split(' + ') for x in equation]

    for side, coefficient in zip(equation, (-1,1)):
        for t in side:
            if (reactant := re.match(r'^(\d+) (.+)$', t)) or (reactant := re.match(r'^(\d+\.\d+) (.+)$', t)):
                value = float(reactant.group(1))
                entry = reactant.group(2)
                equation_dict[entry] = equation_dict.get(entry,0) + value * coefficient
            else:
                equation_dict[t] = equation_dict.get(t,0) + coefficient

    return equation_dict


def build_equation(equation_dict, eq_sign='=') -> str:
    # 
    left, right = [], []
    for comp, coeff in equation_dict.items():
        if coeff < 0:
            x = comp if coeff==-1 else str(-coeff)+' '+comp
            left.append(x)
        elif coeff > 0:
            x = comp if coeff==1 else str(coeff)+' '+comp
            right.append(x)
        elif coeff == 0:
            left.append(comp)
            right.append(comp)

    equation = ' + '.join(left)+' '+eq_sign.strip()+' '+' + '.join(right)
    return equation


def equations_to_S(equations) -> pd.DataFrame:
    comps = set()
    for cs in [parse_equation(x).keys() for x in equations]:
        for c in cs:
            comps.add(c) 
    S = pd.DataFrame(index=range(len(equations)), columns=list(comps))
    for i in range(len(equations)):
        for comp, coeff in parse_equation(equations[i]).items():
            S.loc[i,comp] = coeff

    return S



def download_kegg_compound(entry:str) -> bool:
    # download MolFile from kegg compound database
    # return True if download successful, return False if failed, return None if file existed. 

    file_name = entry + '.mol'
    file_path = os.path.join(kegg_compound_data_path, file_name)
    
    # judge if the direaction kegg_compound exists
    if not os.path.isdir(kegg_compound_data_path):
        os.mkdir(kegg_compound_data_path)
        
    # judge if the mol file exists, if not then download the mol file
    try:
        mol = REST.kegg_get(entry+'/mol')
        with open(file_path, 'w') as f:
            f.write( REST.kegg_get(entry+'/mol').read() )
            print(entry, 'download successful!')
        return True
    except:
        try:
            mol = REST.kegg_get(entry)
            mol_name = mol.readlines()[1].split()[-1]
            print(f'{entry} is {mol_name}, which has no Molfile')
        except:
            print(f'{entry} is not found in kegg')
        return False
