from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from e1_create_dataset import create_regression_dataset
from e3_metrics import regression_scores


# Recommended reading: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
# "Model selection without nested CV uses the same data to tune model parameters and evaluate
# model performance. Information may thus “leak” into the model and overfit the data. The
# magnitude of this effect is primarily dependent on the size of the dataset and the stability
# of the model. See Cawley and Talbot [1] for an analysis of these issues."

def nested_cv():
    """Example of Nested K-Fold Cross-Validation
    """

    _, X, y = create_regression_dataset()

    distributions =  {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': list(range(3, 15)),
    }

    # Outer CV results are what we report as the final model accuracy
    # The inner CV results are useful for analyzing hyperparameter performance

    # Run nested K-Fold CV
    outer_results = []
    kfold = KFold(n_splits=5, random_state=1, shuffle=True)

    for train_index, test_index in kfold.split(X, y): # Outer CV
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inner CV (Train and test model)
        search = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions=distributions, n_iter=10, cv=5, verbose=1)
        search_result = search.fit(X_train, y_train)
        best_pipeline = search_result.best_estimator_ # Can reference the best estimator directly if neeed
        preds = best_pipeline.predict(X_test)
        r2 = regression_scores(y_test, preds)['R2']

        outer_results.append(r2)

    print('Scores:', outer_results)
    print('Average:', sum(outer_results)/len(outer_results))


if __name__ == '__main__':
    nested_cv()
