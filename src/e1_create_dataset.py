"""
This exercise creates some example datasets that can be used in other exercises
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml, make_classification

DEFAULT_CLASSIFICATION_PATH = './data/classification_data.csv'
# These three examples give the same result:
DEFAULT_REGRESSION_PATH = './data/regression_data.csv'
DEFAULT_REGRESSION_PATH = str(Path('data')/'regression_data.csv')
DEFAULT_REGRESSION_PATH = os.path.join('data', 'regression_data.csv')

# Create a directory to hold data
os.makedirs('data', exist_ok=True)


def create_regression_dataset(path=DEFAULT_REGRESSION_PATH, num_rows=200):
    """Create a regression dataset (could also use sklearn.datasets.make_regression)

    :return tuple: DataFrame of regression data
    """
    # Download an example dataset
    bike_sharing = fetch_openml('Bike_Sharing_Demand', version=2, as_frame=True, parser='auto')
    df = bike_sharing['frame']

    # Keep 200 rows and 4 columns
    df = df.head(num_rows)
    df = df[['temp', 'feel_temp', 'humidity', 'windspeed']]

    # Add timestamps for time series exercises
    timestamps = pd.date_range(start='2012-1-1 00:00:00', periods=len(df), freq='30min')
    df.index = timestamps

    df.to_csv(path)
    X = df[['feel_temp', 'humidity', 'windspeed']]
    y = df['temp']
    return df, X, y


def create_classification_dataset(path: str = DEFAULT_CLASSIFICATION_PATH, n_samples: int = 200):
    """Create a classification dataset

    :param str path: DataFrame of classification data
    :return tuple: DataFrame of classification data
    """
    # Make some placeholder data
    X, y = make_classification(n_samples=n_samples)

    # Rename the columns
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

    # Specify the target variable
    df['target'] = y

    df.to_csv(path, index=False)
    return df, X, y


def create_datasets() -> tuple:
    """Create example datasets"""
    df_regression, _, __ = create_regression_dataset()
    df_classification, _, __ = create_classification_dataset()
    return df_regression, df_classification


def run():
    """Run this exercise"""
    create_datasets()


if __name__ == '__main__':
    run()
