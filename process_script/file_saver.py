import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
import preprocessing.processing as plasticc_process
from joblib import Parallel, delayed
import os
import gc

def main():

    files = os.listdir('/home/hchong/Documents/kaggle/plasticc/train_split/train_ts')

    for file in files:
        ts_df_filename = "/home/hchong/Documents/kaggle/plasticc/train_split/train_ts/{file}".format(file=file)
        meta_df_filename = "/home/hchong/Documents/kaggle/plasticc/train_split/train_meta/{file}".format(file=file)

        objs = pd.read_csv(meta_df_filename)['object_id'].tolist()
        obj_list = np.array_split(objs,8)

        Parallel(n_jobs=8)(delayed(process_into_file)(ts_df_filename,meta_df_filename,obj_list,batch_num) for batch_num in range(8))


def process_into_file(ts_df_filename, meta_df_filename, object_id_list_split, batch_num):

    obj_id_list = object_id_list_split[batch_num]

    full_df_class = plasticc_process.process_into_tensor(ts_df_filename, meta_df_filename, obj_id_list)

    for obj_id in obj_id_list:

        full_df_class.process_metadata_df(obj_id)
        full_df_class.process_ts_obj(obj_id)
        full_df_class.save_file(obj_id)

    del full_df_class
    gc.collect()

if __name__ == '__main__':
    main()
