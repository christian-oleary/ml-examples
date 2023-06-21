import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor

from e1_create_dataset import create_regression_dataset


def daily_statistics(input_path=None, output_path='df_daily.csv'):
    """Get daily mean, minimum and maximum values for each column in a time series dataset

    :param input_path: Path to input CSV file (str), defaults to None
    :param output_path: Path to output CSV file (str), defaults to 'df_daily.csv'
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


def time_series_to_tabular():
    df, _, __ = create_regression_dataset()

    TARGET = 'temp' # The column in df we want to forecast
    LAG = 6 # This is how far back we want to look for features
    HORIZON = 3 # This is how far forward we want forecast


    def create_lag_features(df, target, lag):
        """Create features for our ML model (X matrix).

        :param df: DataFrame
        :param lag: Lookback window (int)
        """
        for col in df.columns:
            if col != target:
                for i in range(1, lag+1):
                    df[f'{col}-{i}'] = df[col].shift(i)

                df = df.drop(col, axis=1)

        # OPTIONAL: Drop first N rows where N = lag
        df = df.iloc[lag:]
        return df


    def create_future_values(df, target, horizon):
        targets = [ target ]
        for i in range(1, horizon):
            col_name = f'{target}+{i}'
            df[col_name] = df[target].shift(-i)
            targets.append(col_name)

        # Drop rows missing future target values
        df = df[df[targets[-1]].notna()]
        return df, targets


    print('\nInitial df shape:', df.shape)

    # Create feature data (X)
    df = create_lag_features(df, TARGET, LAG)
    print('\ndf shape after feature creation:', df.shape)

    # Create targets to forecast (y)
    df, targets = create_future_values(df, TARGET, HORIZON)
    print('\ndf shape after feature creation:', df.shape)

    # Separate features (X) and targets (y)
    y = df[targets]
    X = df.drop(targets, axis=1)
    print('\nShape of X (features):', X.shape)
    print('Shape of y (target(s)):', y.shape)

    # Add features to capture hour of day. Try also: .day_of_year, .day_of_week, etc.
    # Read: https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html
    X['hour'] = X.index.hour
    X['sin_hour'] = np.sin(2 * np.pi * X['hour'].apply(lambda ts: int(ts))/24.0)
    X['cos_hour'] = np.cos(2 * np.pi * X['hour'].apply(lambda ts: int(ts))/24.0)
    del X['hour'] # Optional

    # Examine the shapes of the created dataframes and arrays.
    # Look at the column names, e.g.: "print(df.columns)"

    X.to_csv('e12_X.csv')
    y.to_csv('e12_y.csv')

    return X, y


def forecasting_example():
    """Examples of fitting forecasting models using the methodology of time_series_to_tabular()
    """
    X, y = time_series_to_tabular()

    # Using a model
    print('\nTraining model')
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y)
    preds = model.predict(X)
    print('Model works')

    # Using a pipeline and RandomizedSearchCV
    print('\nTraining pipeline')
    scaler_space = { f'scaler__norm': ['l1', 'l2', 'max'] }
    feature_selector_space = { f'multioutput__estimator__feature_selector__k': [1, 2, 3] }
    model_space = { f'multioutput__estimator__model__max_depth': [5, 10, 15, 20] }
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


if __name__ == '__main__':
    daily_statistics(output_path='df_daily.csv')
    # daily_statistics(input_path='hourly_data.csv', output_path='hourly_data_statistics.csv')

    time_series_to_tabular()

    forecasting_example()

