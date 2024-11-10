# dGbyG: a GNN-based tool for predicting standard Gibbs energy changes of metabolic reactions

## Description
This repository is the official implementation of dGbyG, which is model proposed in a paper:  
Unraveling principles of thermodynamics for genome-scale metabolic networks using graph neural networks


## Software Architecture
dGbyG  
├── dGbyG  
│   ├── api &emsp; # API of dGbyG  
│   ├── Chemistry &emsp; # custom classes for chemical objects  
│   ├── data &emsp; # databases, training data, and cross-validation results  
│   │   ├── experimental_dG_from_eQ &emsp; # raw data  
│   │   ├── HMDB &emsp; # HMDB database  
│   │   ├── Human1 &emsp; # Human1 model and relative data  
│   │   ├── kegg_compound &emsp; # kegg compound database  
│   │   ├── libChEBI &emsp; # libChEBI database  
│   │   ├── LIPID_MAPS &emsp; # LIPID MAPS database  
│   │   ├── MetaNetX &emsp; # metanetx database  
│   │   ├── Recon3D &emsp; # Recon3D model and relative data  
│   │   ├── results_data &emsp; # cross-validation results  
│   │   ├── chemaxon_pKa.csv &emsp; # pKa predicted by chemaxon  
│   │   ├── formation_dg_smiles.csv &emsp; #   
│   │   └── TrainingData.csv &emsp; # training data of GNN model  
│   ├── network &emsp; # Networks and models  
│   │   ├── best_model_params &emsp; # the trained models (100 models in)  
│   │   ├── Dataset.py &emsp; #   
│   │   ├── GNNetwork.py &emsp; # construction of graph neural network  
│   │   └── trainer.py &emsp; # custom class of trainer  
│   ├── utils &emsp; # basic functions  
│   └── config.py &emsp; # enviroment variables  
├── RunAnalysis &emsp; # code for running analysis  
│   ├── dGbyG_CrossValidation_EC.ipynb &emsp; # code for leave-one-group-out cross validation classfied by EC classes  
│   ├── dGbyG_CrossValidation_K_fold.ipynb &emsp; # code for k-fold cross-validation  
│   ├── Figures.ipynb &emsp; # code for plotting figures  
│   ├── OtherMethods_CrossValidation.ipynb &emsp; # code for running other methods  
│   ├── Predicting_GibbsEnergy_by_dGbyG.ipynb &emsp; # code for predicting Gibbs energy by dGbyG for Recon3D and Human1  
│   ├── Predicting_GibbsEnergy_by_dGPredictor.ipynb &emsp; # code for predicting Gibbs energy by dGPredictor for Recon3D and Human1  
│   ├── Predicting_GibbsEnergy_by_eQuilibrator.ipynb &emsp; # code for predicting Gibbs energy by eQuilibrator for Recon3D and Human1  
│   ├── Preprocessing_RawData.ipynb &emsp; # code for pre-processing raw data to training data  
│   ├── SupplementaryFigures.ipynb &emsp; # code for plotting supplementary figures  
│   └── Training.ipynb &emsp; # code for training the GNN model  
├── demo.ipynb &emsp; # tutorial of dGbyG  
├── Document.ipynb &emsp; # document of dGbyG  
└── requirements.txt &emsp; # requirements  



## Guidelines
### Prerequisites
- Git
- Conda (Anaconda or Miniconda)

### Step 1: Clone the Repository
Clone the repository to your local machine using the following command:

```bash
git clone https://gitee.com/f-wc/dGbyG.git
```

### Step 2: Create a New Conda Environment (Highly Recommended)
Create a new conda environment (highly recommended) and activate it:

```bash
conda create -n dGbyG && conda activate dGbyG
```


### Step 3: Install Dependencies and dGbyG
#### Install the required dependencies:
```bash
cd /path/to/dGbyG
conda install --file requirements.txt
pip install libChEBIpy
```

#### Install `chemaxon.marvin.calculations.pKaPlugin`(optional)
The `chemaxon.marvin.calculations.pKaPlugin` is used for pKa calculation. It is not necessary for the basic usage of dGbyG, but it is recommended for users who want to calculate transformed standard Gibbs energy change between different pH. Note that this plugin is not free, and you can find more information from the ChemAxon website:
- https://docs.chemaxon.com/display/docs/calculators_index.md


#### Install dGbyG
```bash
cd /path/to/dGbyG
pip install .
```

### Step 4: Run the Code
Run the demo code using Jupyter Notebook:  
`demo.ipynb`

## References
If you use or extend our work, please cite the paper as follows:
- doi: https://doi.org/10.1101/2024.01.15.575679
