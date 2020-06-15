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

def task1_train():
    # Read in data
    df_training_features = pd.read_csv('train_features.csv')
    df_training_labels = pd.read_csv('train_labels.csv')
    all_pids = [pid for pid in df_training_features['pid'].unique()]

    pids_train, pids_val = train_test_split(all_pids, test_size = 0.1)


    # In[12]:


    X_pid_train = dataset_imputer(df_training_features, method='count', pid_list=pids_train, fillna=True)
    X_pid_val = dataset_imputer(df_training_features, method='count', pid_list=pids_val, fillna=True)


    # In[13]:


    Y_pid_train = dataset_imputer(df_training_labels, method=None, pid_list=pids_train, fillna=True)
    Y_pid_val = dataset_imputer(df_training_labels, method=None, pid_list=pids_val, fillna=True)



    #x_train = X_pid_train[['pid', 'PTT', 'HCO3', 'BaseExcess', 'PaCO2', 'FiO2', 'SaO2','Chloride', 'Hct', 'pH']]
    x_train, x_val = X_pid_train.copy(), X_pid_val.copy()
    y_train, y_val = Y_pid_train.copy(), Y_pid_val.copy()

    feature_columns = ['pid', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb',
                     'HCO3', 'BaseExcess', 'RRate', 'Fibrinogen', 'Phosphate', 'WBC',
                     'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
                     'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos',
                     'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate',
                     'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']

    x_train, x_val = x_train[feature_columns], x_val[feature_columns]

    label_columns = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                     'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                     'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                     'LABEL_EtCO2']

    y_train, y_val = y_train[label_columns], y_val[label_columns]

    print(x_train.head())
    print('*'*100)
    print(y_train.head())


    # In[ ]:





    # In[22]:


    print(x_train.shape)
    print(y_train.shape)


    regr = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(100,100), random_state=1, solver='sgd', max_iter=200)
    regr.fit(x_train.iloc[:,1:], y_train.iloc[:,1:])


    # In[24]:


    print(x_train.head())
    print(y_train.iloc[:5,1:])


    # In[25]:


    print(x_val.head())
    print(y_val.iloc[:5,1:])


    # In[27]:


    f = regr.predict_proba(x_val.iloc[:,1:])
    f = pd.DataFrame(f)
    f.columns = y_val.columns[1:]
    #f.columns = ['LABEL_BaseExcess1', 'LABEL_BaseExcess']
    f['pid'] = x_val.iloc[:,0].reset_index(drop=True)

    print(get_score(y_val, f, tasks=['task1'])[1])
    #print(f)

    return regr


def task1_test(regr):
    # TEST TIME!
    df_test_features = pd.read_csv('test_features.csv')

    all_pids_test = [pid for pid in df_test_features['pid'].unique()]

    X_pid_test = dataset_imputer(df_test_features, method='count', pid_list=all_pids_test, fillna=True)

    x_test = X_pid_test.copy()

    feature_columns = ['pid', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb',
                     'HCO3', 'BaseExcess', 'RRate', 'Fibrinogen', 'Phosphate', 'WBC',
                     'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
                     'ABPm', 'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos',
                     'SpO2', 'Bilirubin_direct', 'Chloride', 'Hct', 'Heartrate',
                     'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']

    x_test= x_test[feature_columns]

    print(x_test.head())
    print('*'*100)

    f = regr.predict_proba(x_test.iloc[:,1:])
    f = pd.DataFrame(f)
    f.columns = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                 'LABEL_EtCO2']
    f['pid'] = x_test.iloc[:,0].reset_index(drop=True)

    f.to_csv('task1_test.csv', index=None)