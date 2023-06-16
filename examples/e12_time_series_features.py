import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from e1_create_dataset import create_regression_dataset


def daily_statistics(input_path=None, output_path='df_daily.csv'):
    # Read the dataset into a DataFrame
    if input_path is None:
        df, _, __ = create_regression_dataset()
    else:
        df = pd.read_csv(input_path, index_col=0)

    # Convert strings to NaNs and impute values
    columns = df.columns
    index = df.index
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype(float)
    df = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df)
    df = pd.DataFrame(df, index=index, columns=columns)

    # Convert the index to datetime format (this should also work for any columns)
    df.index = pd.to_datetime(df.index)

    # Make sure data is numerical
    df = df.astype(float)

    # Resample data to daily and calculate the minimum, maximum, and average
    df_daily_mean = df.resample('D').mean()
    df_daily_min = df.resample('D').min()
    df_daily_max = df.resample('D').max()
    df_daily = pd.concat([df_daily_mean, df_daily_min, df_daily_max], axis=1)

    # Provide columns names. This works even if multiple input columns existed in df
    column_names = []
    for col in df.columns:
        column_names.append(f'Mean_{col}')
        column_names.append(f'Minimum_{col}')
        column_names.append(f'Maximum_{col}')
    df_daily.columns = column_names

    # print(df_daily.shape) # It is a good idea to check shapes
    df_daily.to_csv(output_path)

if __name__ == '__main__':
    # daily_statistics()
    daily_statistics('hourly_data.csv', 'hourly_data_statistics.csv')
