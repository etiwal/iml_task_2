import numpy as np
import pandas as pd
import os
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.svm import LinearSVR, SVR, SVC,LinearSVC
from sklearn.datasets import make_regression
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import dataset_imputer, get_score


# In[2]:

def task2_train():
    # Read in data
    df_training_features = pd.read_csv('train_features.csv')
    df_training_labels = pd.read_csv('train_labels.csv')
    all_pids = [pid for pid in df_training_features['pid'].unique()]

    pids_train, pids_val = train_test_split(all_pids, test_size = 0.1)

    X_pid_train = dataset_imputer(df_training_features, method='mean_and_count', pid_list=pids_train, fillna=True)
    X_pid_val = dataset_imputer(df_training_features, method='mean_and_count', pid_list=pids_val, fillna=True)


    Y_pid_train = dataset_imputer(df_training_labels, method=None, pid_list=pids_train, fillna=True)
    Y_pid_val = dataset_imputer(df_training_labels, method=None, pid_list=pids_val, fillna=True)

    x_train, x_val = X_pid_train.fillna(0).copy(), X_pid_val.fillna(0).copy()
    y_train, y_val = Y_pid_train.fillna(0).copy(), Y_pid_val.fillna(0).copy()

    feature_columns = ['pid', 'PTT', 'HCO3', 'BaseExcess', 'PaCO2', 'FiO2', 'SaO2','Chloride', 'Hct', 'pH',
                       'EtCO2_count',
                       'PTT_count', 'BUN_count', 'Lactate_count', 'Temp_count', 'Hgb_count',
                       'HCO3_count', 'BaseExcess_count', 'RRate_count', 'Fibrinogen_count',
                       'Phosphate_count', 'WBC_count', 'Creatinine_count', 'PaCO2_count',
                       'AST_count', 'FiO2_count', 'Platelets_count', 'SaO2_count',
                       'Glucose_count', 'ABPm_count', 'Magnesium_count', 'Potassium_count',
                       'ABPd_count', 'Calcium_count', 'Alkalinephos_count', 'SpO2_count',
                       'Bilirubin_direct_count', 'Chloride_count', 'Hct_count',
                       'Heartrate_count', 'Bilirubin_total_count', 'TroponinI_count',
                       'ABPs_count', 'pH_count']

    x_train, x_val = x_train[feature_columns], x_val[feature_columns]

    label_columns = ['pid', 'LABEL_Sepsis']

    y_train, y_val = y_train[label_columns], y_val[label_columns]

    print(x_train.head())
    print('*'*100)
    print(y_train.head())


    # In[27]:


    print(x_train.shape)
    print(y_train.shape)


    # In[30]:


    # Train MLPClassifier
    regr = MLPClassifier(alpha=1e-4, hidden_layer_sizes=(200,200,200), random_state=1, solver='sgd', max_iter=300)
    regr.fit(x_train.iloc[:,1:], y_train.iloc[:,1].values)


    # In[39]:


    f = regr.predict_proba(x_val.iloc[:,1:])
    f = pd.DataFrame(f)
    f.columns = ['False', 'LABEL_Sepsis']
    f.drop(['False'], axis=1, inplace=True)
    f['pid'] = x_val.iloc[:,0].reset_index(drop=True)

    print(get_score(y_val, f, tasks=['task2'])[1])

    return regr


def task2_test(regr):
    df_test_features = pd.read_csv('test_features.csv')

    all_pids_test = [pid for pid in df_test_features['pid'].unique()]

    X_pid_test = dataset_imputer(df_test_features, method='mean_and_count', pid_list=all_pids_test, fillna=True)

    x_test = X_pid_test.fillna(0).copy()

    feature_columns = ['pid', 'PTT', 'HCO3', 'BaseExcess', 'PaCO2', 'FiO2', 'SaO2','Chloride', 'Hct', 'pH',
                       'EtCO2_count',
                       'PTT_count', 'BUN_count', 'Lactate_count', 'Temp_count', 'Hgb_count',
                       'HCO3_count', 'BaseExcess_count', 'RRate_count', 'Fibrinogen_count',
                       'Phosphate_count', 'WBC_count', 'Creatinine_count', 'PaCO2_count',
                       'AST_count', 'FiO2_count', 'Platelets_count', 'SaO2_count',
                       'Glucose_count', 'ABPm_count', 'Magnesium_count', 'Potassium_count',
                       'ABPd_count', 'Calcium_count', 'Alkalinephos_count', 'SpO2_count',
                       'Bilirubin_direct_count', 'Chloride_count', 'Hct_count',
                       'Heartrate_count', 'Bilirubin_total_count', 'TroponinI_count',
                       'ABPs_count', 'pH_count']

    x_test= x_test[feature_columns]

    print(x_test.head())
    print('*'*100)

    f = regr.predict_proba(x_test.iloc[:,1:])
    f = pd.DataFrame(f)
    f.columns = ['False', 'LABEL_Sepsis']
    f.drop(['False'], axis=1, inplace=True)
    f['pid'] = x_test.iloc[:,0].reset_index(drop=True)

    f.to_csv('task2_test.csv', index=None)