import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('preprocessing/pytorch_obj/' + str(ID) + '.pt')
        y = self.labels[ID]

        return X, y


def generate_kfold(df_target, kfold):
    df_len = len(df_target)
    random_list = np.random.randint(0, kfold - 1, df_len * kfold).reshape(kfold, df_len)
    df_obj = df_target.index.tolist()

    train_val_split = []
    for i in range(len(random_list)):
        train_ind = np.where(random_list[i] != 0)[0]
        val_ind = np.where(random_list[i] == 0)[0]
        val_obj = [df_obj[i] for i in val_ind]
        train_obj = [df_obj[i] for i in train_ind]
        train_val_split_temp = [{'train': train_obj,
                                 'val': val_obj}]
        train_val_split.append(train_val_split_temp)
        # train_val_split_temp = []

    return train_val_split