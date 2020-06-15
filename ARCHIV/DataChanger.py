import pandas as pd
import numpy as np
import scipy
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


def impute_data(input_df,  impute_strategy = 'SimpleImputer'):

    if impute_strategy == 'SimpleImputer':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        df = pd.DataFrame(imp.fit_transform(input_df, input_df), columns=list(input_df.columns.values))

    if impute_strategy == 'SimpleImputer_pid':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        df = pd.DataFrame(imp.fit_transform(input_df, input_df), columns=list(input_df.columns.values))

        imputed_df = pd.DataFrame(columns=list(input_df.columns.values))

        for counter, pid in enumerate(df_train_features['pid'].unique()):
            print(counter, '  pid:', pid)
            pid_df = input_df[input_df['pid'] == pid]

            imputed_pid_df = pid_df.loc[:,['pid', 'Time', 'Age']]
            for column in list(pid_df.columns.values)[3:]:
                col_df = pid_df[[column]]
                if 12 - col_df.isna().sum().values >= 3:
                    imp_pid = SimpleImputer(missing_values=np.nan, strategy='median')
                    col_df = imp_pid.fit_transform(col_df, col_df)
                else:
                    pid_helper_df = df[df['pid'] == pid]
                    col_df = pid_helper_df[[column]]

                imputed_pid_df[column] = col_df

            imputed_df = pd.concat([imputed_df, imputed_pid_df], ignore_index=True)
            del imputed_pid_df

        df = imputed_df

    return df

def build_stats(input_df):
    train_features_stats = pd.DataFrame()
    train_features_stats['nan_count'] = input_df.isna().sum()
    train_features_stats['mean'] = input_df.mean()
    train_features_stats['std'] = input_df.std()
    train_features_stats['median'] = input_df.median()
    train_features_stats = train_features_stats.T
    train_features_stats.to_csv('train_features_stats.csv', sep=',', float_format='%.3f', encoding='utf-8')


