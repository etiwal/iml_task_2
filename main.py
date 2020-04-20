import pandas as pd
import numpy as np
import scipy
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

def impute_data(input_df, impute_strategy = 'SimpleImputer'):
    if impute_strategy == 'SimpleImputer':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        df = imp.fit_transform(input_df, input_df)

    return pd.DataFrame(df, columns=list(input_df.columns.values))

df_train_features = pd.read_csv('train_features.csv')
#df_train_labels = pd.read_csv('train_labels.csv')
#df_test_features = pd.read_csv('test_features.csv')

pids_train, pids_validate = df_train_features['pid'].unique(), 0

# build statistics
# train_features_stats = pd.DataFrame()
# train_features_stats['nan_count'] = df_train_features.isna().sum()
# train_features_stats['mean'] = df_train_features.mean()
# train_features_stats['std'] = df_train_features.std()
# train_features_stats = train_features_stats.T
# train_features_stats.to_csv('train_features_stats.csv', sep=',', float_format='%.3f', encoding='utf-8', index=False)



train_features_imputed = impute_data(df_train_features)
print(train_features_imputed)
train_features_imputed.to_csv('train_features_imputed.csv', sep=',', float_format='%.3f', encoding='utf-8', index=False)

# dict structure
#train_features = {'11402': {'age': 22, '1':{'EtCO2': 123, 'etc':133}, '2':{'EtCO2': 234, 'etc':432}}}

# x_train = np.array(df_train_features)
# y_train = np.array(df_train_labels)
# x_test = np.array(df_test_features)
#
# svm_reg = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, fit_intercept=False, dual=True, verbose=0, random_state=None, max_iter=1000)
# svm_reg.fit(x_train,y_train)
# y_pred = svm_reg.predict(x_test)
