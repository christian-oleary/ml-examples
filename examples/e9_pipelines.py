from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from e1_create_dataset import create_regression_dataset
from e3_metrics import regression_scores
from e8_handling_models import regression_models


def build_pipeline(model_name):
    # Create dataset
    _, X, y = create_regression_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model_space = { f'model__{k}': v for k, v in regression_models[model_name][1].items() }

    distributions = {
        **model_space,
        'feature_selector__k': [1, 2, 3] # Probably not useful in this example. For demonstration only
        }

    class Debugger(BaseEstimator, TransformerMixin):
        def __init__(self, name=''):
            self.name = name

        def fit(self, X, _):
            print(f'Pipeline debugger {self.name} - X.shape: {X.shape}')
            return self

        def transform(self, X, y=None):
            return X

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # ('debugger', Debugger()), # Useful for debugging pipeline errors
        ('feature_selector', SelectKBest(f_regression)), # Use if you have too many (unhelpful) features
        ('model', DecisionTreeRegressor()),
    ])


    # Train models
    clf = RandomizedSearchCV(pipeline, distributions, n_iter=10, cv=5, verbose=1)
    search = clf.fit(X_train, y_train)

    # Evaluate
    predictions = clf.predict(X_test)
    scores = regression_scores(y_test, predictions)
    print('Training scores:', search.cv_results_['mean_test_score'].mean())
    print('Best parameters:', search.best_params_)
    print('Test scores:', scores['R2'])


if __name__ == '__main__':
    build_pipeline('DecisionTreeRegressor')
