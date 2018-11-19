import numpy as np
import torch
from torch.utils import data
import pandas as pd
import utils.dataloader as my_dataloader


params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

a = pd.read_csv('train_object_id.csv')
b = pd.read_csv('training_set_metadata.csv')
obj_train_valid = a[a['group'] != 9]['object_id'].tolist()
df_pred = b[['object_id', 'target']].set_index('object_id').loc[obj_train_valid]

kfold_list = my_dataloader.generate_kfold(df_pred,5)

training_set = my_dataloader.Dataset(kfold_list[0][0]['train'], df_pred['target'])
training_generator = data.DataLoader(training_set, **params)

validation_set = my_dataloader.Dataset(kfold_list[0][0]['val'], df_pred['target'])
validation_generator = data.DataLoader(validation_set, **params)

dataiter = iter(training_generator)
images, labels = dataiter.next()
