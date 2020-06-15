import pandas as pd

def merge_results():
    df_task1 = pd.read_csv('task1_test.csv')
    df_task2 = pd.read_csv('task2_test.csv')
    df_task3 = pd.read_csv('task3_test.csv')

    df = pd.merge(df_task1, df_task2, on='pid')
    df = pd.merge(df, df_task3, on='pid')

    ordered_columns_list = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2',
                            'LABEL_Sepsis',
                            'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    df = df[ordered_columns_list]

    df.to_csv('submission.csv', index=False, float_format='%.3f')
    df.to_csv('submission.zip', index=False, float_format='%.3f', compression='zip')



