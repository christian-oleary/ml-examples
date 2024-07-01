"""Feature engineering for scikit-learn regression models"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor

from src.e1_create_dataset import create_regression_dataset


FEATURES_DIR = Path('data')/'features'
FEATURES_DIR.mkdir(exist_ok=True)


def daily_statistics(input_path=None, output_path=FEATURES_DIR/'e12_daily_statistics.csv'):
    """Get daily mean, minimum and maximum values for each column in a time series dataset

    :param input_path: Path to input CSV file (str), defaults to None
    :param output_path: Path to output CSV file (str)
    """

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

    # Impute missing values
    # Can also use: http://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
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
    mean_column_names = []
    min_column_names = []
    max_column_names = []
    for col in df.columns:
        mean_column_names.append(f'Mean_{col}')
        min_column_names.append(f'Minimum_{col}')
        max_column_names.append(f'Maximum_{col}')
    df_daily.columns = mean_column_names + min_column_names + max_column_names

    # print(df_daily.shape) # It is a good idea to check shapes
    df_daily.to_csv(output_path)


def time_series_to_tabular():
    """Convert time series data to a tabular format usable by a regression model"""
    df, _, __ = create_regression_dataset()

    target_col = 'temp'  # The column in df we want to forecast
    lag = 6  # This is how far back we want to look for features
    horizon = 3  # This is how far forward we want forecast

    # Fill in missing values
    cols = df.columns
    index = df.index
    df = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df)
    df = pd.DataFrame(df, columns=cols, index=index)  # convert back to dataframe

    def create_lag_features(df, target, lag):
        """Create features for our ML model (X matrix).

        :param pd.DataFrame df: DataFrame
        :param str target: Name of target column (int)
        :param int lag: lag window (int)
        """
        for col in df.columns:
            for i in range(1, lag+1):
                df[f'{col}-{i}'] = df[col].shift(i)

            # Drop non-target values (we only keep historical feature values)
            if col != target:
                df = df.drop(col, axis=1)

        # OPTIONAL: Drop first N rows where N = lag
        # Alternatively, we could impute the missing data
        df = df.iloc[lag:]
        return df

    def create_future_values(df, target, horizon):
        """Create target columns for horizons greater than 1"""
        targets = [target]
        for i in range(1, horizon):
            col_name = f'{target}+{i}'
            df[col_name] = df[target].shift(-i)
            targets.append(col_name)

        # Optional: Drop rows missing future target values
        df = df[df[targets[-1]].notna()]
        return df, targets

    print('\nInitial df shape:', df.shape)

    # Create feature data (X)
    df = create_lag_features(df, target_col, lag)
    print('\ndf shape with feature columns:', df.shape)

    # Create targets to forecast (y)
    df, targets = create_future_values(df, target_col, horizon)
    print('\ndf shape with target columns:', df.shape)

    # Separate features (X) and targets (y)
    y = df[targets]
    X = df.drop(targets, axis=1)
    print('\nShape of X (features):', X.shape)
    print('Shape of y (target(s)):', y.shape)

    # Add features to capture hour-of-day
    # Read: https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html
    X['hour'] = X.index.hour
    X['sin_hour'] = np.sin(2 * np.pi * X['hour'].astype(int) / 24.0)
    X['cos_hour'] = np.cos(2 * np.pi * X['hour'].astype(int) / 24.0)
    del X['hour']  # Optional

    # An alternative for day-of-year:
    # X['day_of_year'] = X.index.day_of_year
    # X['sin_day_of_year'] = np.sin(2 * np.pi * X['day_of_year'].apply(lambda ts: int(ts))/365.0)
    # X['cos_day_of_year'] = np.cos(2 * np.pi * X['day_of_year'].apply(lambda ts: int(ts))/365.0)
    # del X['day_of_year']

    # An alternative for day-of-week:
    # X['weekday'] = X.index.weekday
    # X['sin_weekday'] = np.sin(2 * np.pi * X['weekday'].apply(lambda ts: int(ts))/7.0)
    # X['cos_weekday'] = np.cos(2 * np.pi * X['weekday'].apply(lambda ts: int(ts))/7.0)
    # del X['weekday']

    # Examine the shapes of the created dataframes and arrays.
    # Look at the column names, e.g.: "print(df.columns)"

    X.to_csv(FEATURES_DIR/'e12_features.csv')
    y.to_csv(FEATURES_DIR/'e12_targets.csv')
    return X, y


def forecasting_example():
    """Examples of fitting forecasting models using the methodology of time_series_to_tabular()"""
    X, y = time_series_to_tabular()

    # 1. Simple example using a model
    print('\nTraining model')
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y)
    preds = model.predict(X)
    print('Model works')

    # 2. Another example using a pipeline and RandomizedSearchCV
    print('\nTraining pipeline')
    scaler_space = {'scaler__norm': ['l1', 'l2', 'max']}

    feature_selector_space = {'multioutput__estimator__feature_selector__k': [1, 2, 3]}

    model_space = {'multioutput__estimator__model__max_depth': [5, 10, 15, 20]}

    distributions = {
        **scaler_space,
        **feature_selector_space,
        **model_space
    }

    pipeline = Pipeline([
        ('scaler', Normalizer()),
        ('multioutput', MultiOutputRegressor(
            Pipeline([
                ('feature_selector', SelectKBest(f_regression)),
                ('model', DecisionTreeRegressor())
            ])
        ))
    ])
    clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    print('Pipeline works\n')

    return preds, y


def multioutput_metrics(y_test, y_pred):
    """Metrics for multioutput data"""
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print('\nMAE', mae)
    # How many values are in MAE? Check horizon


def run():
    """Run this exercise"""
    daily_statistics()

    print('\n--- TS TO TABULAR ---')
    time_series_to_tabular()

    print('\n--- FORECASTING EXAMPLE ---')
    actual, predictions = forecasting_example()

    print('\n--- MULTIOUTPUT METRICS ---')
    multioutput_metrics(actual, predictions)
    print()


if __name__ == '__main__':
    run()

    # Some useful links:
    # https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/
    # https://www.kaggle.com/code/ryanholbrook/time-series-as-features
