"""Basic example of optimizing a model."""

import os
from pathlib import Path
import time

import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # type: ignore # noqa: F401 # pylint: disable=W0611
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

from ml.e1_create_dataset import create_regression_dataset
from ml.e3_metrics import regression_scores

# Recommended reading: https://scikit-learn.org/stable/modules/grid_search.html#grid-search


def optimize_models(num_rows: int):
    """Optimize models and print the results."""
    _, X, y = create_regression_dataset(num_rows=num_rows)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def train_and_test(model, method: str):
        """Train and test a model."""
        print('---------------------------------------------------')
        start_time = time.time()

        # Train and test
        search = model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores = regression_scores(y_test, predictions)

        print(f'\n{method} took {time.time() - start_time} seconds')
        print('Training scores:', search.cv_results_['mean_test_score'].mean())
        print('Test scores:', scores['R2'])
        print('---------------------------------------------------')

    distributions = {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(3, 15)),
    }

    # Using a grid search
    model = GridSearchCV(DecisionTreeRegressor(), distributions, cv=5, verbose=1)
    train_and_test(model, 'GridSearchCV')

    # Using a randomized search
    model = RandomizedSearchCV(DecisionTreeRegressor(), distributions, n_iter=10, cv=5, verbose=1)
    train_and_test(model, 'RandomizedSearchCV')

    # Using a halving randomized search
    model = HalvingRandomSearchCV(
        DecisionTreeRegressor(), distributions, n_candidates=10, cv=5, verbose=1
    )
    train_and_test(model, 'HalvingRandomSearchCV')

    # See also: https://github.com/bayesian-optimization/BayesianOptimization

    # You can save the results of the search if you want to examine how hyperparameters affect the model's performance:
    inner_cv_results = pd.DataFrame(model.cv_results_)
    del inner_cv_results['params']
    path = Path('results') / 'inner_cv_results.csv'
    inner_cv_results.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    # E.g. get correlation between validation score and Max. Depth of Decision Tree
    correlation = inner_cv_results['mean_test_score'].corr(
        inner_cv_results['param_max_depth'].astype(float)
    )
    print('\nCorrelation between validation score and Max. Depth parameter:', round(correlation, 2))


def run(num_rows: int = 17000):
    """Run this exercise."""
    optimize_models(num_rows)

    # How do the methods compare in execution time?

    # With HalvingRandomSearchCV and RandomizedSearchCV, we can use distributions instead of lists:
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html


if __name__ == '__main__':
    run()
