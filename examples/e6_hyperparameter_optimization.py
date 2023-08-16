import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

from e1_create_dataset import create_regression_dataset
from e3_metrics import regression_scores

# Recommended reading: https://scikit-learn.org/stable/modules/grid_search.html#grid-search

def optimize_model():
    _, X, y = create_regression_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    distributions =  {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(3, 15)),
    }

    # Using a grid search
    clf = GridSearchCV(DecisionTreeRegressor(), distributions, cv=5, verbose=1)
    search = clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    scores = regression_scores(y_test, predictions)

    num_iterations = search.cv_results_["mean_test_score"].shape[0]
    print(f'\nGrid Search ({num_iterations} iterations):')
    print('Training scores:', search.cv_results_['mean_test_score'].mean())
    print('Test scores:', scores['R2'], '\n')


    # Using a randomized search
    # With RandomizedSearchCV, we can use distributions instead of lists: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    clf = RandomizedSearchCV(DecisionTreeRegressor(), distributions, n_iter=10, cv=5, verbose=1)
    search = clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    scores = regression_scores(y_test, predictions)

    print('\nRandomized Search (10 iterations):')
    print('Training scores:', search.cv_results_['mean_test_score'].mean())
    print('Test scores:', scores['R2'])

    # See also: https://github.com/bayesian-optimization/BayesianOptimization

    # You can save the results of the search if you want to examine how hyperparameters affect the model's performance:
    inner_cv_results = pd.DataFrame(search.cv_results_)
    del inner_cv_results['params']
    filename = 'inner_cv_results.csv'
    inner_cv_results.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    # E.g. get correlation between validation score and Max. Depth of Decision Tree
    correlation = inner_cv_results['mean_test_score'].corr(inner_cv_results['param_max_depth'].astype(float))
    print('\nCorrelation between validation score and Max. Depth parameter:', round(correlation, 2))


if __name__ == '__main__':
    optimize_model()
    # Which method is more efficient?
