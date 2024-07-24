"""Examples of training scikit-learn models."""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml.e1_create_dataset import create_classification_dataset, create_regression_dataset


def train_regression():
    """Train a scikit-learn regression model."""
    _, X, y = create_regression_dataset()

    # Create the model
    model = DecisionTreeRegressor()

    # Train the model
    model.fit(X, y)

    # Make some predictions
    predictions = model.predict(X)

    # Note that X and y are not very informative variable names, but they are
    # extremely common in Python-based machine learning. Investigate the
    # objects, e.g.: "print('X', type(X), X.shape)"
    return model, predictions, X, y


def train_classification():
    """Train a scikit-learn classification model."""
    _, X, y = create_classification_dataset()

    # Create the model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X, y)

    # Make some predictions
    predictions = model.predict(X)

    # Take note of the shapes of the prediction arrays. Why are they that shape?
    # Hint: take a look at shape of the input data
    return model, predictions, X, y


def run():
    """Run this exercise."""
    regression_model, preds, _, __ = train_regression()
    print('\nRegression predictions:', preds, type(preds), preds.shape)
    print('Model type:', type(regression_model))

    classification_model, preds, _, __ = train_classification()
    print('\n\nClassification predictions:', preds, type(preds), preds.shape)
    print('Model type:', type(classification_model))


if __name__ == '__main__':
    run()
