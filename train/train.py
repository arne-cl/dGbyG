import tqdm, os
import numpy as np
import pandas as pd
import torch

from dGbyG.network.Dataset import Train_Dataset
from dGbyG.network.GNNetwork import MP_network
from dGbyG.train.trainer import Model
from dGbyG.config import train_data_path, inference_model_path



trainingData_df = pd.read_csv(train_data_path)

mean_std = trainingData_df.loc[:,'stderr'].mean()
SEM = np.nan_to_num(trainingData_df.loc[:,'SEM'], nan=mean_std)
weight = (1/(SEM+1))/np.median((1/(SEM+1)))
trainingData_df.loc[:,'weight'] = weight

TrainSet = Train_Dataset(trainingData_df, equation_column='reaction', dG_column='standard_dg_prime', weight_column='weight')

network = MP_network(emb_dim=300, num_layer=2)

model = Model()
model.network = network

Loss, Result_df = model.cross_validation(TrainSet, mode=10, epochs=9000, lr=1e-4, weight_decay=1e-6)

loss_history, Result_df, n = model.train(TrainSet, epochs=9000, lr=1e-4, weight_decay=1e-6)
torch.save(network, inference_model_path)