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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import dataset_imputer, get_score
import sklearn.metrics as metrics


def task3_train():
    # Read in data
    df_training_features = pd.read_csv('train_features.csv')
    df_training_labels = pd.read_csv('train_labels.csv')
    all_pids = [pid for pid in df_training_features['pid'].unique()]

    pids_train, pids_val = train_test_split(all_pids, test_size = 0.1)


    # In[3]:


    # def getPatientData(trainingData, pids, patients=0, mode='pid'):
    #     if mode == 'number':
    #         pids = all_pids[:patients]
    #     if len(pids) == 0:
    #         return trainingData
    #     #pids = np.array(pids).astype(np.float)
    #     patients = [trainingData.iloc[idx] for idx in range(0, len(trainingData)) if trainingData['pid'][idx] in pids]
    #     #patientTrainingDataIndex = [trainingData.iloc[idx] for idx, col in enumerate(trainingData) if trainingData['pid'][idx] in pids]
    #     return pd.DataFrame(patients)

    # def partitionData(trainingDataPids, trainingPartition=80):
    #     validationPartition = 100 - trainingPartition
    #     countTraining = int((trainingPartition/100)*len(trainingDataPids))
    #     training = trainingDataPids[:countTraining]
    #     validation = trainingDataPids[countTraining:]
    #     print('')
    #     print('Training size: ' + str(countTraining))
    #     print('Validation size: ' + str(len(validation)))
    #     return training, validation

    # def populateData(X,Y):
    #     Z = pd.merge(X, Y, on='pid')
    #     YData = Z[Y.columns].iloc[:,1:]
    #     XData = Z[X.columns].iloc[:,1:]
    #     return XData, YData
    # import sklearn.metrics as metrics

    # TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
    #          'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
    #          'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

    # VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    # #def get_score(df_true, df_submission):
    #     df_submission = df_submission.sort_values('pid')
    #     df_true = df_true.sort_values('pid')

    #     #task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    #     #task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    #     task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    #     #score = np.mean([task1, task2, task3])
    #     return task3


    # In[4]:


    X_pid_train = dataset_imputer(df_training_features, method='mean', pid_list=pids_train, fillna=True)
    X_pid_val = dataset_imputer(df_training_features, method='mean', pid_list=pids_val, fillna=True)


    # In[5]:


    Y_pid_train = dataset_imputer(df_training_labels, method=None, pid_list=pids_train, fillna=True)
    Y_pid_val = dataset_imputer(df_training_labels, method=None, pid_list=pids_val, fillna=True)


    # In[6]:


    #x_train = X_pid_train[['pid', 'PTT', 'HCO3', 'BaseExcess', 'PaCO2', 'FiO2', 'SaO2','Chloride', 'Hct', 'pH']]
    x_train, x_val = X_pid_train.copy(), X_pid_val.copy()
    y_train, y_val = Y_pid_train.copy(), Y_pid_val.copy()

    feature_columns = ['pid','Age','Temp','RRate','Heartrate','ABPm','SpO2']

    x_train, x_val = x_train[feature_columns], x_val[feature_columns]

    label_columns = ['pid', 'LABEL_RRate',  'LABEL_ABPm',  'LABEL_SpO2', 'LABEL_Heartrate']

    y_train, y_val = y_train[label_columns], y_val[label_columns]

    print(x_train.head())
    print('*'*100)
    print(y_train.head())


    # In[7]:


    print(x_train.shape)
    print(y_train.shape)


    # In[ ]:





    # In[8]:


    # Train MLPClassifier
    regr = MLPRegressor(alpha=1e-4, hidden_layer_sizes=(200,200,200), random_state=1, solver='adam', max_iter=200)
    regr.fit(x_train.iloc[:,1:], y_train.iloc[:,1:])
    print(regr.loss_)


    # In[9]:


    print(x_train.iloc[:20,:])
    print(y_train.iloc[:20,:])

    f = regr.predict(x_val.iloc[:,1:])
    f = pd.DataFrame(f)
    f.columns = list(y_val.columns[1:])
    f['pid'] = x_val.iloc[:,0].reset_index(drop=True)

    print(get_score(y_val, f, tasks=['task3'])[1])

    return regr


def task3_test(regr):
    df_test_features = pd.read_csv('test_features.csv')

    all_pids_test = [pid for pid in df_test_features['pid'].unique()]

    X_pid_test = dataset_imputer(df_test_features, method='mean', pid_list=all_pids_test, fillna=True)

    x_test = X_pid_test.copy()

    feature_columns = ['pid','Age','Temp','RRate','Heartrate','ABPm','SpO2']

    x_test= x_test[feature_columns]

    print(x_test.head())
    print('*'*100)

    f = regr.predict(x_test.iloc[:,1:])
    f = pd.DataFrame(f)
    f.columns = ['LABEL_RRate',  'LABEL_ABPm',  'LABEL_SpO2', 'LABEL_Heartrate']
    f['pid'] = x_test.iloc[:,0].reset_index(drop=True)

    f.to_csv('task3_test.csv', index=None)