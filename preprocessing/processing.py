import numpy as np
import pandas as pd
import torch
import math
from sklearn import preprocessing
import re
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("error")

class metadata_process_class:

    def __init__(self, metadata_df, ts_raw_df):
        self.df = metadata_df
        self.ts_df = ts_raw_df

    def quantile_check(self, np_array, threshold_list):
        return [(np.quantile(np_array, j), np.quantile(np_array, 1 - j)) for j in threshold_list]

    def metadata_process(self):
        df = self.df

        df2 = df[['object_id', 'ddf', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']]
        df2.loc[pd.isnull(df2.distmod), 'type'] = 1
        df2['type'].fillna(0, inplace=True)
        df2['distmod'].fillna(-1, inplace=True)
        df2.loc[df2.hostgal_photoz == 0, 'hostgal_photoz'] = -1
        df2.loc[df2.hostgal_photoz == 0, 'hostgal_photoz_err'] = -1

        self.metadata = df2

    def ts_raw_process(self):
        df = self.ts_df

        df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
        df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
        # train[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
        df['mjd_detected'] = np.NaN
        df.loc[df.detected == 1, 'mjd_detected'] = df.loc[df.detected == 1, 'mjd']
        gr_mjd = df.groupby('object_id').mjd_detected
        df['mjd_diff'] = gr_mjd.transform('max') - gr_mjd.transform('min')

        aggs = {
            'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
            'detected': ['mean'],
            'flux_ratio_sq': ['sum', 'skew'],
            'flux_by_flux_ratio_sq': ['sum', 'skew'],
        }

        agg_df = df.groupby(['object_id', 'mjd_diff']).agg(aggs)
        new_columns = [
            k + '_' + agg for k in aggs.keys() for agg in aggs[k]
        ]
        agg_df.columns = new_columns
        agg_df['flux_diff'] = agg_df['flux_max'] - agg_df['flux_min']
        agg_df['flux_dif2'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_mean']
        agg_df['flux_w_mean'] = agg_df['flux_by_flux_ratio_sq_sum'] / agg_df['flux_ratio_sq_sum']
        agg_df['flux_dif3'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_w_mean']

        self.agg_df = agg_df

    def combine(self):
        agg_df = self.agg_df
        metadata = self.metadata
        full_train = agg_df.reset_index(['mjd_diff']).join(metadata.set_index(['object_id']), how='left').drop(
            ['target'], axis=1)
        self.final_df_temp = full_train
        self.scaler_df = full_train.drop(['ddf', 'type'], axis=1)

    def scaler(self):
        df = self.scaler_df

        scaled_df = pd.DataFrame()
        corr_df = pd.DataFrame()
        scaler_dict = {}

        for i, column in enumerate(df):

            # print(i, column)
            pd_series = df[column]

            gaussian_scaler = preprocessing.QuantileTransformer(output_distribution='normal')
            x = pd_series.values.reshape(-1, 1)  # returns a numpy array
            x_scaled = gaussian_scaler.fit_transform(x)
            x_scaled = pd.Series(x_scaled[:, 0])

            # check for values correlation with original values at decile 10 and 90
            thres_list = [0.05, 0.1]
            x_thres = self.quantile_check(x, thres_list)

            df2 = pd.concat([x_scaled, pd_series.reset_index(drop=True)], axis=1, ignore_index=True)
            df2.columns = ['transform', 'ori']

            for tuples in x_thres:
                df3 = df2.loc[df2['ori'].between(tuples[0], tuples[1])]
                corr = df3.corr()
                corr_value = corr.iloc[0, 1]
                temp_df = pd.DataFrame([column, tuples[0], corr_value]).T
                corr_df = corr_df.append(temp_df, ignore_index=True)

            scaled_df = pd.concat((scaled_df, x_scaled), axis=1, ignore_index=True)
            scaler_filename = '/home/hchong/Documents/kaggle/plasticc/scaler/{column}.save'.format(column=column)
            joblib.dump(gaussian_scaler, scaler_filename)
            scaler_dict[column] = gaussian_scaler

        scaled_df.columns = [x for x in df.columns]
        corr_df.columns = ['columns', 'threshold', 'correlation']

        self.scaled_df = scaled_df
        self.scaler_dict = scaler_dict
        self.scaled_corr_df = corr_df

    def final(self):
        non_scaled_df = self.final_df_temp[['ddf', 'type']]
        scaled_df = self.scaled_df
        final_df = pd.concat((non_scaled_df.reset_index(), scaled_df), axis=1, ignore_index=True)
        col = non_scaled_df.reset_index().columns.tolist() + scaled_df.columns.tolist()
        final_df.columns = col
        final_df = final_df.set_index('object_id')

        return final_df

class metadata_file_splitter:

    def __init__(self, final_df, object_id_filename):
        self.df = final_df
        self.obj_df = pd.read_csv(object_id_filename).set_index('group')

    def splitter(self):

        for group in set(self.obj_df.index):
            obj_ids = self.obj_df.loc[group]['object_id']
            df_to_save = self.df.loc[obj_ids]

            min_id = min(obj_ids)
            max_id = max(obj_ids)

            filename = "/home/hchong/Documents/kaggle/plasticc/train_split/train_meta/{min_id}to{max_id}.csv".format(min_id=min_id,max_id=max_id)
            df_to_save.to_csv(filename)



class process_into_tensor:

    def __init__(self, ts_df_filename, metadata_df_filename, obj_ids):
        self.ts_df = pd.read_csv(ts_df_filename).set_index(['object_id', 'input', 'passband']).loc[obj_ids]
        self.df = pd.read_csv(metadata_df_filename).set_index('object_id').loc[obj_ids]

    def ts_scaler(self, object_id):
        test = self.ts_df.drop('mean_detected', axis=1).loc[object_id]

        # object_id_set = set(object_id_list)
        # scaled_df_v3 = pd.DataFrame()
        # for object_id in object_id_set:
        # test = scaler_df.loc[object_id]
        passband_set = set(test.index.get_level_values('passband'))

        scaled_df_v2 = pd.DataFrame()
        for passband in passband_set:
            tester = test.xs(passband, level='passband')
            scaled_df = pd.DataFrame()
            for i, column in enumerate(tester):

                # print(i, column)
                pd_series = tester[column]
                series_index = pd_series.index.tolist()

                if bool(re.search("range", column)):

                    # maxabs_scaler = preprocessing.MaxAbsScaler()
                    x = pd_series.values.reshape(-1, 1)  # returns a numpy array
                    x_scaled = preprocessing.maxabs_scale(x)
                    x_scaled = pd.Series(x_scaled[:, 0], index=series_index)

                else:

                    # power_scaler = preprocessing.PowerTransformer()
                    x = pd_series.values.reshape(-1, 1)  # returns a numpy array
                    try:
                        x_scaled = preprocessing.power_transform(x,method='yeo-johnson')
                        x_scaled = pd.Series(x_scaled[:, 0], index=series_index)
                    except RuntimeWarning:
                        print(object_id,passband,column)

                scaled_df = pd.concat((scaled_df, x_scaled), axis=1)

            scaled_df.index.names = ['input']
            scaled_df.columns = [x for x in tester.columns]
            scaled_df['passband'] = passband
            scaled_df.set_index('passband', append=True, inplace=True)
            scaled_df_v2 = pd.concat((scaled_df_v2, scaled_df), axis=0)
            scaled_df_v2 = scaled_df_v2.reorder_levels(['input', 'passband'])

        scaled_df_v2['object_id'] = object_id
        scaled_df_v2.set_index('object_id', append=True, inplace=True)
        scaled_df_v2 = scaled_df_v2.reorder_levels(['object_id', 'input', 'passband'])
        # scaled_df_v3 = pd.concat((scaled_df_v3, scaled_df_v2), axis=0)

        final_scaled_df = pd.concat((scaled_df_v2, self.ts_df[['mean_detected']]), axis=1, sort=False)
        return final_scaled_df

    def process_ts_obj(self, object_id):

        df = self.ts_scaler(object_id)

        df = df.loc[object_id]
        max_input = max(df.index.get_level_values('input'))

        if max_input != 139:
            if max_input % 2 == 0:  # odd number of inputs, pad one extra in front
                front_pad_size = int(math.ceil((139 - max_input) / 2))
                pad_front = torch.zeros(5, 6, front_pad_size)
                pad_back = torch.zeros(5, 6, front_pad_size - 1)
            else:
                pad_size = int((139 - max_input) / 2)
                pad_front = pad_back = torch.zeros(5, 6, pad_size)

        torch_init = torch.zeros(5, 6, int(max_input) + 1)

        # print(torch_init.shape)

        for index, rows in df.iterrows():
            torch_init[:, index[1], index[0]] = torch.from_numpy(rows.values)

        try:
            output_tensor = torch.cat((pad_front, torch_init, pad_back), 2)
        except:
            try:
                output_tensor = torch.cat((pad_front, torch_init), 2)
            except:
                try:
                    output_tensor = torch_init
                except:
                    print("Error")

        assert output_tensor.shape == torch.Size([5, 6, 140]), "tensor size not correct, {object_id}, {shape}".format(object_id=object_id,shape=output_tensor.shape)
        assert torch.isnan(output_tensor).sum().item() == 0, "nan detected in tensor, {object_id}".format(object_id=object_id)

        # print(output_tensor.shape)
        self.ts_tensor = output_tensor


    def process_metadata_df(self, object_id):
        df = self.df.loc[object_id]
        try:
            self.obj_tensor = torch.from_numpy(np.asarray(df))
        except RuntimeWarning:
            print(object_id)

        assert self.obj_tensor.shape == torch.Size([28]), "tensor size not correct, {object_id}, {shape}".format(object_id=object_id,shape=self.obj_tensor.shape)
        assert torch.isnan(self.obj_tensor).sum().item() == 0, "nan detected in tensor, {object_id}".format(object_id=object_id)

    def save_file(self, object_id):

        filename = 'preprocessing/pytorch_obj/' + str(object_id) + '.pt'

        torch.save([self.ts_tensor,self.obj_tensor], filename)