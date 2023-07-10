import tqdm, os
import numpy as np
import pandas as pd

from dGbyG.utils.Dataset import Train_Dataset
from dGbyG.config import tecrdb_path

tecrdb_df = pd.read_csv(tecrdb_path)
#formation_dg_df = pd.read_csv()

train_dataset = Train_Dataset(tecrdb_df, equation_column='reaction', dG_column='standard_dg_primes')

