import os, time
from urllib.error import HTTPError
from Bio.KEGG import REST

from dGbyG.config import kegg_database_path


def download_compound(entry:str, force:bool=False) -> None:
    file_path = os.path.join(kegg_database_path, 'compound', f'{entry}.txt')
    if force or not os.path.exists(file_path):
        t = REST.kegg_get(entry).read()
        with open(file_path, 'w') as f:
            f.write(t)
        print(entry, 'successfully downloaded')
    else:
        print(entry, 'already exists')
    return None



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