# dGbyG: a GNN-based tool for predicting standard Gibbs energy changes of metabolic reactions

## This is just a fork!

The original repo is here: https://gitee.com/f-wc/dGbyG . Please send your questions and pull requests there.
I only forked it because I had a hard time getting the software to install.

#### Description
This repository is the official implementation of dGbyG, which is model proposed in a paper: Unraveling principles of thermodynamics for genome-scale metabolic networks using graph neural networks

#### Software Architecture
├── api                                 # API of dGbyG  
├── Chemistry                           # custom classes for chemical objects  
├── data                                # databases, training data, and cross-validation results  
│   ├── experimental_dG_from_eQ         # raw data  
│   ├── HMDB                            # HMDB database  
│   ├── kegg_compound                   # kegg compound database  
│   ├── MetaNetX                        # metanetx database  
│   ├── results_data                    # cross-validation results  
│   ├── chemaxon_pKa.csv                # pKa predicted by chemaxon  
│   ├── formation_dg_smiles.csv         #   
│   └── TrainingData.csv                # training data of GNN model  
├── network                             # Networks and models  
│   ├── best_model_params               # the trained models (100 models in)  
│   ├── Dataset.py                      #   
│   ├── GNNetwork.py                    # construction of graph neural network  
│   └── trainer.py                      # custom class of trainer  
├── RunAnalysis                         # code for running analysis  
│   ├── CrossValidation_EC.ipynb        # code for leave-one-group-out cross validation classfied by EC classes  
│   ├── CrossValidation_K_fold.ipynb    # code for k-fold cross-validation  
│   ├── Figures.ipynb                   # code for plotting figures  
│   ├── OtherMethods.ipynb              # code for running other methods  
│   ├── PreprocessingRawData.ipynb      # code for pre-processing raw data to training data  
│   ├── SupplementaryFigures.ipynb      # code for plotting supplementary figures  
│   └── Training.ipynb                  # code for training the GNN model  
├── utils                               # basic functions  
├── config.py                           # enviroment variables ()  
├── demo.ipynb                          # tutorial  
└── requirements.txt                    # requirements  



#### Guidelines

1.  git clone this project to local
2.  install the requirements (in requirements)
3.  follow the tutorial (demo.ipynb)

#### References

If you use or extend our work, please cite the paper as follows:
doi: https://doi.org/10.1101/2024.01.15.575679
