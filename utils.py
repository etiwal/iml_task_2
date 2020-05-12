from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
        
def dataset_imputer(df_original, method, pid_list, fillna=True):
    
    # get column names for imputed df
    df_imputed = pd.DataFrame(columns=df_original.columns)
    
    use_age = False
    if 'Age' in df_original.columns:
        use_age = True
    
    if method is not None:
        for i, pid in enumerate(pid_list):

            pid_df = df_original[df_original['pid']==pid]
            if pid_df.empty:
                print('pid is not in dataset')

            if method == 'mean':
                data = pid_df.mean()
            elif method == 'count':
                data = pid_df.isna().sum()
                data = -1*pid_df + 12
            else:
                print('the method', method, 'does not exist!')

            data['pid'] = pid

            if use_age:
                data['Age'] = pid_df['Age'].iloc[0]

            data = pd.DataFrame(data).transpose()
            df_imputed = pd.concat([df_imputed, data])

            if i % round(0.1*len(pid_list)) == 0:
                print(round(i/len(pid_list),2)*100, '%')

            del data
            
    else:
        df_imputed = df_original
        
    if fillna:
        df_imputed.fillna(0)
        
    df_imputed = df_imputed.sort_values(['pid'])
    
    print('100.0 % - completed')
    print('')
    
    return df_imputed


def get_score(df_true, df_submission, tasks=['task1', 'task2', 'task3']):
    
    TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
             'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
             'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    
    VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    
    task_score = np.array([np.nan, np.nan, np.nan])
    
    for task in tasks:
        if task == 'task1':
            task_score[0] = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
        elif task == 'task2':
            task_score[1]  = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
        elif task == 'task3':
            task_score[2]  = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    
    score = np.nanmean([task_score])
    
    return task_score, score
        


class healthDataset(Dataset):
    def __init__(self, np_features, np_labels, pid_list):
        self.np_features = np_features
        self.np_labels = np_labels
        self.pid_list = pid_list

        bool_array = np.sum([self.np_features[:,0] == pid for pid in self.pid_list], axis=0).astype('bool')
        self.np_features_filtered = self.np_features[bool_array, :]

    def __len__(self):
        return int(self.np_features_filtered.shape[0]/12)

    def __getitem__(self, index):
        """Generates one sample of data"""
        pid = self.np_features_filtered[index*12,0]

        features = self.np_features_filtered[index*12:index*12+12,3:]
        stacked_features = np.reshape(features, -1, order='F')

        bool_array = np.array([self.np_labels[:, 0] == pid])[0].astype('bool')
        labels = self.np_labels[bool_array, 1:12]
        stacked_labels = np.reshape(labels, -1, order='F')

        return (stacked_features, stacked_labels)
