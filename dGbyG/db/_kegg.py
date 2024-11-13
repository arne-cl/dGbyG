import os, time
from urllib.error import HTTPError
from typing import Dict, List
from Bio.KEGG import REST
import Bio.KEGG.Compound

from dGbyG.config import kegg_database_path, kegg_compound_data_path


def download_compound_mol(entry:str) -> bool:
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
        

def download_compound(entry:str, force:bool=False) -> None:
    file_path = os.path.join(kegg_database_path, 'compound', f'{entry}.txt')
    if force or not os.path.exists(file_path):
        t = REST.kegg_get(entry).read()
        with open(file_path, 'w') as f:
            f.write(t)
        print(entry, 'successfully downloaded')
        return True
    else:
        print(entry, 'already exists')
        return True



def batch_download_compound(entry_list:list, force:bool=False):
    # 
    for entry in entry_list:
        try:
            download_compound(entry, force)
        except HTTPError as e:
            if e.code == 404:
                print('no', entry)
            else:
                print(entry, 'error', e.code)
        except BaseException as e:
            print(entry, 'error', e)
        time.sleep(0.01)



def download_all_compound(force=False):
    entry_list = REST.kegg_list('compound').readlines()
    entry_list = [entry.split('\t')[0] for entry in entry_list]
    batch_download_compound(entry_list, force=force)
    return None



class KEGG_Compound(object):
    '''
    rewrite KEGG.Compound to add new features
    '''
    def __init__(self, entry:str):
        self.entry = entry
        self.file_path = os.path.join(kegg_database_path, 'compound', f'{entry}.txt')
        self._raw = self._read_file()
        self.__kegg_compound__ = self.__origon_kegg_compound__()
        self._data = self.__data__()
        self.entry = self.__kegg_compound__.entry
        self.name = self.__kegg_compound__.name
        self.enzyme = self.__kegg_compound__.enzyme
        #self.formula = self.__kegg_compound__.formula
        self.pathway = self._pathway()
        self.reaction = self._reaction()
        self.dblinks = self._dblinks()

    def _read_file(self):
        if not os.path.exists(self.file_path):
            raise FileExistsError(f'No such file: {self.file_path}')
        with open(self.file_path, 'r') as f:
            f = f.read()
        assert f.endswith('///\n'), f'Incomplete file {self.file_path}'
        return f[:-4]
    
    def __origon_kegg_compound__(self) -> Bio.KEGG.Compound.Record:
        with open(self.file_path, 'r') as f:
            x = next(Bio.KEGG.Compound.parse(f))
            return x
        
    def __data__(self):
        data = {}
        for i in self._raw.splitlines():
            if i[0] == i.strip()[0]:
                key = i.split()[0]
                i = i.split()[1:]
                data[key] = []
            else:
                i = i.split()
            data[key].extend(i)
        return data
        
    def _pathway(self) -> Dict[str, List[str]]:
        output = {'KEGG.pathway':[x[1:] for x in self.__kegg_compound__.pathway]}
        return output
    
    def _reaction(self) -> Dict[str, List[str]]:
        output = {'KEGG.reaction': self._data['REACTION']}
        return output
    
    def _dblinks(self) -> Dict[str, List[str]]:
        output = dict(self.__kegg_compound__.dblinks)
        return output