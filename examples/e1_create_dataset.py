import pandas as pd
from sklearn.datasets import fetch_openml, make_classification


def create_regression_dataset():
    """Create a regression dataset (we could also have used sklearn.datasets.make_regression)

    :return: DataFrame of regression data
    """
    bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True, parser='auto')
    df = bike_sharing.frame
    df = df.head(200) # Take first 200 rows
    df = df[['temp', 'feel_temp', 'humidity', 'windspeed']]  # Take 4 columns
    timestamps = pd.date_range(start='2012-1-1 00:00:00', periods=len(df), freq='30T')
    df.index = timestamps
    df.to_csv('regression_data.csv')
    X = df[['feel_temp', 'humidity', 'windspeed']]
    y = df['temp']
    return df, X, y


def create_classification_dataset():
    """Create a classification dataset

    :return: DataFrame of classification data
    """
    X, y = make_classification()
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    df.to_csv('classification_data.csv', index=False)
    return df, X, y


def create_datasets():
    """Create example datasets"""

    df_regression, _, __ = create_regression_dataset()
    df_classification, _, __ = create_classification_dataset()
    return df_regression, df_classification

if __name__ == '__main__':
    create_datasets()
