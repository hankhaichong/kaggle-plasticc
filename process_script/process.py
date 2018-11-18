import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
import preprocessing.processing as process
from joblib import Parallel, delayed
import os
import gc

def main():

    train_df = pd.read_csv('training_set.csv')
    train_metadata_df = pd.read_csv('training_set_metadata.csv')
    train_df_process = pd.read_csv('processed_training_set.csv').set_index(['object_id', 'input', 'passband'])

    train_metadata_df_process = process.metadata_process_class(train_metadata_df,train_df)
    train_metadata_df_process.ts_raw_process()
    train_metadata_df_process.metadata_process()
    train_metadata_df_process.combine()
    train_metadata_df_process.scaler()
    train_metadata_df_v2 = train_metadata_df_process.final()

    file_splitter = process.metadata_file_splitter(train_metadata_df_v2,'/home/hchong/Documents/kaggle/plasticc/train_object_id.csv')
    file_splitter.splitter()


if __name__ == '__main__':
    main()
