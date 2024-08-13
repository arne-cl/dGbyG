#!/usr/bin/env python
# coding: utf-8

# # Tutorial of dGbyG

# In[2]:


# import package
from dGbyG.api import Compound, Reaction


# ## Prediction of standard Gibbs energy change for reaction
# ### 1. Prediction in default condition 
# default condition is (pH=7.0, ionic strength=0.25 M, pMg=14, electrical potential)

# In[3]:


from dGbyG.utils.constants import default_condition
print('default_condition is', default_condition)


# ### 1.1. prepare the equation of reaction
# Using the reaction {4-hydroxy-2-oxoglutarate = pyruvate + glyoxylate} as example here.
# 
# We highly recommend using smiles or inchi as identifier of molecules, because they are not dependent on certain database. KEGG entry can be accepted as well if internet connection is available. We also support metanetx and hmdb id, but you need to make sure that necessary database files have been download and put into ./data/MetaNetX and ./data/HMDB  

# In[4]:


equation = '[H]OC(=O)C(=O)C([H])([H])C([H])(O[H])C(=O)O[H] = [H]OC(=O)C(=O)C([H])([H])[H] + [H]OC(=O)C([H])=O'


# #### 1.2. Creating Reaction object.
# The type of reactants' identifier need to be specified.

# In[5]:


reaction = Reaction(equation, cid_type='smiles')


# #### 1.3. Predicting standard Gibbs energy change of the reaction
# The first value is predicted standard Gibbs energy change, and the second is the standard deviation of the predicted value.

# In[6]:


print(reaction.transformed_standard_dGr_prime)


# ### 2. Prediction in other condition or for multicompartment reaction
# if you need to predict the reaction in a specific condition (adjusting pH, ion strength, electrical potential, pMg) or multicompartment reaction, it is better to follow the steps below. 
# 
# For example (here we use kegg entry.), Acetate Transport reaction (from cytosol to golgi Apparatus),  
# Acetic acid[c] <=> Acetic acid[g]. KEGG entry of cetic acid is C00033.  
# c: cytosol, pH=7.2  
# g: golgi apparatus, pH=6.35
# #### 2.1. Creating Compound object and setting condition for each reactant

# In[12]:


substrate = Compound('C00033', input_type='kegg')
substrate.condition['pH']=7.2

product = Compound('C00033', input_type='kegg')
product.condition['pH']=6.35


# #### 2.2. Creating Reaction object
# Stoichiometric number of substrates are negative, that of products are positive.

# In[13]:


equation_dict = {substrate:-1, product:1}
reaction = Reaction(equation_dict, cid_type='compound')


# #### 2.3. Predicting standard Gibbs energy change of the reaction
# The first value is predicted standard Gibbs energy change, and the second is the standard deviation of the predicted value.

# In[15]:


print(reaction.transformed_standard_dGr_prime)


# In[ ]:




