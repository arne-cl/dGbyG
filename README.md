# dGbyG: a GNN-based tool for predicting standard Gibbs energy changes of metabolic reactions

## Description
This repository is the official implementation of dGbyG, which is model proposed in a paper: Unraveling principles of thermodynamics for genome-scale metabolic networks using graph neural networks

## Software Architecture
├── api                                 # API of dGbyG  
├── Chemistry                           # custom classes for chemical objects  
├── data                                # databases, training data, and cross-validation results  
│   ├── experimental_dG_from_eQ         # raw data  
│   ├── HMDB                            # HMDB database  
│   ├── Human1                          # Human1 model and relative data  
│   ├── kegg_compound                   # kegg compound database  
│   ├── libChEBI                        # libChEBI database  
│   ├── LIPID_MAPS                      # LIPID MAPS database  
│   ├── MetaNetX                        # metanetx database  
│   ├── Recon3D                         # Recon3D model and relative data  
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
│   ├── dGbyG_CrossValidation_EC.ipynb                  # code for leave-one-group-out cross validation classfied by EC classes  
│   ├── dGbyG_CrossValidation_K_fold.ipynb              # code for k-fold cross-validation  
│   ├── Figures.ipynb                                   # code for plotting figures  
│   ├── OtherMethods_CrossValidation.ipynb              # code for running other methods  
│   ├── Predicting_GibbsEnergy_by_dGbyG.ipynb           # code for predicting Gibbs energy by dGbyG for Recon3D and Human1  
│   ├── Predicting_GibbsEnergy_by_dGPredictor.ipynb     # code for predicting Gibbs energy by dGPredictor for Recon3D and Human1  
│   ├── Predicting_GibbsEnergy_by_eQuilibrator.ipynb    # code for predicting Gibbs energy by eQuilibrator for Recon3D and Human1  
│   ├── Preprocessing_RawData.ipynb                     # code for pre-processing raw data to training data  
│   ├── SupplementaryFigures.ipynb                      # code for plotting supplementary figures  
│   └── Training.ipynb                                  # code for training the GNN model  
├── utils                               # basic functions  
├── config.py                           # enviroment variables  
├── demo.ipynb                          # tutorial  
└── requirements.txt                    # requirements  



## Guidelines

### Prerequisites

- Git
- Conda (Anaconda or Miniconda)

### Step 1: Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone https://gitee.com/f-wc/dGbyG.git
```

### Step 2: Install Dependencies

Create a new conda environment (highly recommended) and activate it:

```bash
conda create -n dGbyG && conda activate dGbyG
```


### Step 3: Install Dependencies

Install the required dependencies:

```bash
cd /path/to/dGbyG
conda install --file requirements.txt
pip install libChEBIpy
pip install numpyarray_to_latex
```

### Step 4: Run the Code

Run the demo code using Jupyter Notebook:
demo.ipynb

## References

If you use or extend our work, please cite the paper as follows:
- doi: https://doi.org/10.1101/2024.01.15.575679
