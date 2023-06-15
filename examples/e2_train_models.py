from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from e1_create_dataset import create_datasets

# create a regression dataset
df_regression, df_classification = create_datasets()


def train_regression():
    """Train a regression model"""
    # Separate the target variable from features
    y = df_regression['temp']
    X = df_regression.drop('temp', axis=1)

    model = DecisionTreeRegressor() # Create the model
    model.fit(X, y) # Train the model
    predictions = model.predict(X) # Make some predictions
    return model, predictions, X, y

    # Note that X and y are not very informative variable names, but they are
    # extremely common in Python-based machine learning. Investigate the objects,
    # e.g.: "print('X', type(X), X.shape)"


def train_classification():
    # Train a classification model
    y = df_classification['target']
    X = df_classification.drop('target', axis=1)

    model = DecisionTreeClassifier() # Create the model
    model.fit(X, y) # Train the model
    predictions = model.predict(X) # Make some predictions
    return model, predictions, X, y

    # Take note of the shapes of the prediction arrays. Why are they that shape?
    # Hint: take a look at the input data


if __name__ == '__main__':
    df_regression, df_classification = create_datasets()

    model, predictions, X, y = train_regression()
    print('\nRegression predictions:', predictions, type(predictions), predictions.shape)

    model, predictions, X, y = train_classification()
    print('\n\nClassification predictions:', predictions, type(predictions), predictions.shape)
