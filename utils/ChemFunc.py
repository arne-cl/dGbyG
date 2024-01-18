import os, re, json, requests
import numpy as np
import pandas as pd
from functools import reduce
from Bio.KEGG import REST
from typing import Tuple, List
from copy import deepcopy

import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dGbyG.config import kegg_additions_csv_path, kegg_compound_data_path, metanetx_database_path, hmdb_database_path, chemaxon_pka_csv_path
from dGbyG.utils.constants import *

cache = {}


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
        
    if not type(equation)==list:
        return {equation:1}
    
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


def build_equation(equation_dict:dict, eq_sign='=') -> str:
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



def to_mol(cid:str, cid_type:str) -> rdkit.Chem.rdchem.Mol:
    # 
    if not isinstance(cid, str):
        raise TypeError('cid must be String type, but got {0}'.format(type(cid)))
    
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
        #elif download_kegg_compound(entry):
        #    mol = file_to_mol(path)
        else:
            kegg_additions_df = pd.read_csv(kegg_additions_csv_path, index_col=0)
            if entry in kegg_additions_df.index:
                inchi = kegg_additions_df.loc[entry, 'inchi']
                mol = inchi_to_mol(inchi) if pd.notna(inchi) else None
        return Chem.AddHs(mol)
    
    def metanetx_id_to_mol(id:str):
        if 'metanetx' not in cache:
            cache['metanetx'] = pd.read_csv(metanetx_database_path, sep='\t', header=351, index_col=0)
        metanetx_df = cache['metanetx']
        smiles = (metanetx_df.loc[id, 'SMILES'])
        mol = smiles_to_mol(smiles)
        return Chem.AddHs(mol)
    
    def hmdb_id_to_mol(id:str):
        if 'hmdb' not in cache:
            cache['hmdb'] = pd.read_csv(hmdb_database_path, index_col=0, dtype={33: object})
        hmdb_df = cache['hmdb']
        if len(id.replace('HMDB', '')) < 7:
            id = 'HMDB' + '0'*(7-len(id.replace('HMDB', ''))) + id.replace('HMDB', '')
        smiles = hmdb_df.loc[id, 'SMILES']
        mol = smiles_to_mol(smiles)
        return Chem.AddHs(mol)
    
    def inchi_key_to_mol(inchi_key:str):
        if 'hmdb' not in cache:
            cache['hmdb'] = pd.read_csv(hmdb_database_path, index_col=0, dtype={33: object})
        if 'metanetx' not in cache:
            cache['metanetx'] = pd.read_csv(metanetx_database_path, sep='\t', header=351, index_col=0)
        
        if smiles := cache['hmdb'].loc[cache['hmdb'].INCHI_KEY == inchi_key, 'SMILES'].to_list():
            return smiles_to_mol(smiles[0])
        elif smiles := cache['metanetx'].loc[cache['metanetx'].InChIKey == 'InChIKey='+inchi_key, 'SMILES'].to_list():
            return smiles_to_mol(smiles[0])
        elif smiles := cache['hmdb'].loc[cache['hmdb'].INCHI_KEY.apply(lambda x:x[:-2]) == inchi_key[:-2], 'SMILES'].to_list():
            return smiles_to_mol(smiles[0])
        elif smiles := cache['metanetx'].loc[cache['metanetx'].InChIKey.apply(lambda x:x[:-2] if pd.notna(x) else x) == 'InChIKey='+inchi_key[:-2], 'SMILES'].to_list():
            return smiles_to_mol(smiles[0])
        else:
            return None
    
    def name_to_mol(name:str):
        if 'name' not in cache:
            cache['name'] = pd.read_csv(metanetx_database_path, sep='\t', header=351, index_col=1)
        metanetx_df = cache['name']
        smiles = (metanetx_df.loc[name, 'SMILES'])
        mol = smiles_to_mol(smiles)
        return Chem.AddHs(mol)
    

    # the main body
    cid_type = cid_type.lower()
    methods = {'inchi': inchi_to_mol,
               'smiles': smiles_to_mol,
               'kegg': kegg_entry_to_mol,
               'metanetx': metanetx_id_to_mol,
               'hmdb': hmdb_id_to_mol,
               'file': file_to_mol,
               'inchi-key': inchi_key_to_mol,
               'name': name_to_mol,
               }

    if cid_type=='auto':
        _to_mols = methods
    else:
        assert cid_type in methods.keys(), f'{cid_type} id cannot be recognized'
        _to_mols = {cid_type: methods.get(cid_type)}

    output = {}
    for _cid_type, _to_mol in _to_mols.items():
        try:
            mol = Chem.AddHs(_to_mol(cid))
        except:
            mol = None
        if mol:
            output[_cid_type] = mol
    if len(output)>1:
        raise ValueError(f'Which {cid} is {tuple(output.keys())}?')
    return tuple(output.values())[0] if output else None


def equation_dict_to_mol_dict(equation_dict:dict, cid_type):
    if type(cid_type)==str:
        cid_Types = [cid_type.lower()] * len(equation_dict)
    elif type(cid_type)==list:
        assert len(cid_type)==len(equation_dict)
        cid_Types = cid_type

    mol_dict = dict(map(lambda item: (to_mol(item[0][0], item[1]), item[0][1]),
                        zip(equation_dict.items(), cid_Types)))
    
    return mol_dict if not None in mol_dict else None


def equation_to_mol_dict(equation, cid_type):
    equation_dict = parse_equation(equation)
    
    return equation_dict_to_mol_dict(equation_dict, cid_type)


def equations_to_S(equations) -> pd.DataFrame:
    comps = set()
    for cs in [parse_equation(x).keys() for x in equations]:
        for c in cs:
            comps.add(c) 
    S = pd.DataFrame(index=list(comps), columns=range(len(equations)), dtype=float, data=0)
    for i in range(len(equations)):
        for comp, coeff in parse_equation(equations[i]).items():
            S.loc[comp,i] = coeff

    return S


def S_to_equations(S, mets) -> list:
    if len(mets) not in S.shape:
        raise ValueError('S.shape not match mets length')
    elif S.shape[0]==S.shape[1]:
        print('S is square, make sure dim 0 match mets')
    elif S.shape[0] == len(mets):
        S = S.T
    elif S.shape[1] == len(mets):
        pass
    else:
        print('S.shape=', S.shape)

    equations = []
    for s in S:
        reactants = np.array(mets)[s!=0]
        coeff = s[s!=0]
        equations.append(build_equation(dict(zip(reactants, coeff))))

    return equations



def normalize_mol(mol:rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    #mol = rdMolStandardize.Uncharger().uncharge(mol)
    #te = rdMolStandardize.TautomerEnumerator() # idem
    #mol = te.Canonicalize(mol)
    mol = rdMolStandardize.ChargeParent(mol)
    return Chem.AddHs(mol)



def atom_bag(mol:rdkit.Chem.rdchem.Mol):
    atom_bag = {}
    charge = 0
    for atom in mol.GetAtoms():
        atom_bag[atom.GetSymbol()] = 1 + atom_bag.get(atom.GetSymbol(), 0)
        charge += atom.GetFormalCharge()
    atom_bag['charge'] = charge
        
    return atom_bag



def get_pKa(compound, temperature:float=default_T, source='chemaxon_file') -> list:
    # compound: 
    # source: 

    def get_pka_from_chemaxon(compound, temperature):
        # 
        types = 'acidic,basic'
        pKaLowerLimit = -20
        pKaUpperLimit = 20
        prefix = 'dynamic'
        T = temperature
        a = 8
        b = 8
        smiles = compound.Smiles

        pka = os.popen(f"cxcalc pKa -m macro -t {types} -T {T} -P {prefix} -a {a} -b {b}\
                       -i {pKaLowerLimit} -x {pKaUpperLimit} '{smiles}'").readlines()
        
        pka_copy = {'acidicValuesByAtom':[], 'basicValuesByAtom': []}
        pka = dict(zip(pka[0].strip().split('\t'), pka[1].strip().split('\t')))
        atoms_idx = 0
        for k, v in pka.items():
            if v!='' and k.startswith('apKa'):
                atomIndex = pka['atoms'].split(',')[atoms_idx]
                pka_copy['acidicValuesByAtom'].append({'atomIndex':int(atomIndex), 'value':float(v)})
                atoms_idx += 1
            elif v!='' and k.startswith('bpKa'):
                atomIndex = pka['atoms'].split(',')[atoms_idx]
                pka_copy['basicValuesByAtom'].append({'atomIndex':int(atomIndex), 'value':float(v)})
                atoms_idx += 1
        
        return pka_copy



    def get_pka_from_chemaxon_rest(compound, temperature):
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
        smiles = compound.Smiles
        if 'chemaxon_file' not in cache.keys():
            cache['chemaxon_file'] = pd.read_csv(chemaxon_pka_csv_path, index_col=0)
        pKa_df = cache['chemaxon_file']

        if smiles not in pKa_df.index:
            print(smiles, 'pka not in file')
            calculate_pKa_batch_to_file([smiles])
            pKa_df = pd.read_csv(chemaxon_pka_csv_path, index_col=0)
            cache['chemaxon_file'] = pKa_df

        pka_copy = {'acidicValuesByAtom':[], 'basicValuesByAtom': []}
        pka = pKa_df.loc[smiles, :]
        atoms_idx = 0
        for k, v in pka.items():
            if pd.notna(v) and k.startswith('apKa'):
                atomIndex = pka['atoms'].split(',')[atoms_idx]
                pka_copy['acidicValuesByAtom'].append({'atomIndex':int(atomIndex), 'value':float(v)})
                atoms_idx += 1
            elif pd.notna(v) and k.startswith('bpKa'):
                atomIndex = pka['atoms'].split(',')[atoms_idx]
                pka_copy['basicValuesByAtom'].append({'atomIndex':int(atomIndex), 'value':float(v)})
                atoms_idx += 1

        return pka_copy
        
    # the main body of this function
    methods = {'chemaxon':get_pka_from_chemaxon,
               'chemaxon_file': get_pka_from_file,
               'chemaxon_rest': get_pka_from_chemaxon_rest,}
    if source=='auto':
        for source in methods:
            if pKa := methods[source](compound, temperature=temperature):
                break
    else:
        _get_pka = methods.get(source)
        pKa = _get_pka(compound, temperature=temperature)
   
    return pKa



def calculate_pKa_batch_to_file(smiles_list:list, temperature=default_T) -> None:
    # 
    with open('comps.smi', 'w') as f:
        f.writelines([x+'\n' for x in smiles_list])

    types = 'acidic,basic'
    pKaLowerLimit = -20
    pKaUpperLimit = 20
    prefix = 'dynamic'
    T = temperature
    a = 8
    b = 8

    pka = os.popen(f"cxcalc pKa -m macro -t {types} -T {T} -P {prefix} -a {a} -b {b}\
                       -i {pKaLowerLimit} -x {pKaUpperLimit} comps.smi")
    
    #a = os.popen(f"cxcalc pKa -t acidic,basic -a 8 -b 8 comps.smi")
    with open('comps_pKa.tsv', 'w') as f:
        f.write(pka.read())
    
    p = pd.read_csv('comps_pKa.tsv', sep='\t').drop(columns='id')
    p.index = smiles_list
    p = p.reset_index()
    p = p.rename(columns={'index':'smiles'})
    op = pd.read_csv(chemaxon_pka_csv_path)
    p = pd.concat([op,p]).drop_duplicates(subset=['smiles'], keep='last')
    p.to_csv(chemaxon_pka_csv_path, index=False)
    print(f'Results saved at {chemaxon_pka_csv_path}.')

    return None





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


def ddGf_to_dissociation(pH: float, T: float, pKa:dict):
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


def ddGf_to_elec(charge, e_potential):
    # 
    dg = FARADAY * charge * e_potential
    return dg


def ddGf_to_single(compound, condition):
    # 
    num_H = compound.atom_bag.get('H', 0)
    num_Mg = compound.atom_bag.get('Mg', 0)
    net_charge = compound.atom_bag.get('charge', 0)
    pH = condition.get('pH', default_pH)
    T = condition.get('T', default_T)
    I = condition.get('I', default_I)
    pMg = condition.get('pMg', default_pMg)
    pKa = compound.pKa(T)
    if pKa == None:
        return None
    
    ddgf_to_single = (- ddGf_to_dissociation(pH, T, pKa) - ddGf_to_aqueous(pH, pMg, I, T, net_charge, num_H, num_Mg))

    return ddgf_to_single


def ddGf(compound, condition1, condition2):
    num_H = compound.atom_bag.get('H', 0)
    num_Mg = compound.atom_bag.get('Mg', 0)
    net_charge = compound.atom_bag.get('charge', 0)

    ddGf_1_2 = []
    for condition in [dict(condition1), dict(condition2)]:
        pH = condition.get('pH', default_pH)
        T = condition.get('T', default_T)
        I = condition.get('I', default_I)
        pMg = condition.get('pMg', default_pMg)
        e_potential = condition.get('e_potential', default_e_potential)
        pKa = compound.pKa(T)
        if pKa == None:
            return None

        ddGf_1_2.append(ddGf_to_dissociation(pH, T, pKa) + ddGf_to_aqueous(pH, pMg, I, T, net_charge, num_H, num_Mg) + ddGf_to_elec(net_charge, e_potential))
    
    return ddGf_1_2[1] - ddGf_1_2[0]


def ddGf_H_ion(compound, condition1, condition2):
    assert compound.Smiles == '[H+]'
    ddGf_1_2 = []
    for condition in [dict(condition1), dict(condition2)]:
        pH = condition.get('pH', default_pH)
        T = condition.get('T', default_T)
        e_potential = condition.get('e_potential', default_e_potential)
        ddGf_1_2.append(R * T * np.log(10) * pH + ddGf_to_elec(1, e_potential))
    return ddGf_1_2[1] - ddGf_1_2[0]


def ddGr(reaction, condition1, condition2):
    ddGf_list = [coeff * ddGf(comp, condition1, condition2) for comp, coeff in reaction.items()]
    if None in ddGf_list:
        print(ddGf_list)
        return None
    return sum(ddGf_list)



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
            f.write(REST.kegg_get(entry+'/mol').read())
            print(entry, 'download successful!')
        return True
    except:
        try:
            mol = REST.kegg_get(entry)
            mol_name = mol.readlines()[1].split()[-1]
            with open(file_path, 'w') as f:
                f.write('')
                #print(f'{entry} is {mol_name}, which has no Molfile')
            return True
        except:
            #print(f'{entry} is not found in kegg')
            return False


def remove_map_num(mol):
    mol = deepcopy(mol)
    [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()]
    return mol


def reacting_map_num_of_rxn(rxn):
    rxn.Initialize()
    map_num = []
    for mol, atoms_idx in zip(rxn.GetReactants(), rxn.GetReactingAtoms()):
        map_num = map_num + [mol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in atoms_idx]
    assert len(set(map_num))==len(map_num)
    return tuple(map_num)


def map_num_to_idx_with_radius(mol:rdkit.Chem.rdchem.Mol, map_num:tuple, radius:int) -> tuple:
    atoms_idx_radius = [set([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() in map_num]), ]
    for r in range(radius):
        atoms_idx_r = set()
        for atom_idx_i in atoms_idx_radius[r]:
            for atom_j in mol.GetAtomWithIdx(atom_idx_i).GetNeighbors():
                atoms_idx_r.add(atom_j.GetIdx())
        for exist_atoms in atoms_idx_radius:
            atoms_idx_r = atoms_idx_r - exist_atoms
        atoms_idx_radius.append(atoms_idx_r)

    idxs = set()
    for idxs_set in atoms_idx_radius:
        idxs |= idxs_set

    return tuple(idxs)
    
