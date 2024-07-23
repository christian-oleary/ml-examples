"""Using pipelines in scikit-learn."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.e1_create_dataset import create_regression_dataset
from src.e3_metrics import regression_scores
from src.e8_handling_models import regression_models


def build_pipeline(model_name, debug=False):
    """Build and test a scikit-learn Pipeline object."""
    # Read this: https://scikit-learn.org/stable/modules/compose.html#pipeline
    print(f'Training: {model_name}')

    # Create dataset
    _, X, y = create_regression_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model, model_space = regression_models[model_name]
    model_space = {f'model__{k}': v for k, v in model_space.items()}

    distributions = {
        **model_space,
        'feature_selector__k': [1, 2, 3],  # Included here for demonstration
    }

    # The "distributions" dictionary will something look like this:
    # {
    #     'model__criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
    #     'model__splitter': ['best', 'random'],
    #     'model__max_depth': [8, 16, 32, 64, 128, None],
    #     'feature_selector__k': [1, 2, 3]
    # }

    class Debugger(BaseEstimator, TransformerMixin):
        """Useful for debugging pipelines."""

        def __init__(self, name=''):
            self.name = name

        def fit(self, X, _):
            """No 'fitting' to be done. Print, log or save data for debugging."""
            print(f'Pipeline debugger {self.name} - X.shape: {X.shape}')
            return self

        def transform(self, X, **_):
            """No action needed here."""
            return X

    pipeline_parts = [
        ('scaler', StandardScaler()),
        ('feature_selector', SelectKBest(f_regression)),  # Remove least helpful features
        ('model', model()),
    ]
    if debug:
        pipeline_parts.insert(1, ('debugger', Debugger()))

    pipeline = Pipeline(pipeline_parts)

    # Train models
    clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1)
    search = clf.fit(X_train, y_train)

    # Evaluate
    predictions = clf.predict(X_test)
    scores = regression_scores(y_test, predictions)
    print('Training scores:', search.cv_results_['mean_test_score'].mean())
    print('Best parameters:', search.best_params_)
    print('Test scores:', scores['R2'])


def run():
    """Run this exercise using a Decision Tree model."""
    build_pipeline('DecisionTreeRegressor')


if __name__ == '__main__':
    run()
