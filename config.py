import os

package_path = os.path.split(os.path.abspath(__file__))[0]

# models
inference_model_state_dict_dir = os.path.join(package_path, 'network', 'best_model_params')

# training data
train_data_path = os.path.join(package_path, 'data', 'TrainingData.csv')

# compound databases
kegg_compound_data_path = os.path.join(package_path, 'data', 'kegg_compound')
kegg_additions_csv_path = os.path.join(package_path, 'data', 'kegg_compound', 'kegg_additions.csv')
metanetx_database_path = os.path.join(package_path, 'data', 'MetaNetX', 'chem_prop.tsv')
hmdb_database_path = os.path.join(package_path, 'data', 'HMDB', 'structures.csv')
chemaxon_pka_csv_path = os.path.join(package_path, 'data', 'chemaxon_pKa.csv')
pKa_json_file_path = os.path.join(package_path, 'data', 'pKa.json')