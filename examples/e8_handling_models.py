from sklearn.calibration import LinearSVC
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, LogisticRegression, PassiveAggressiveClassifier, PassiveAggressiveRegressor, Perceptron, Ridge, SGDRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeRegressor

from e1_create_dataset import create_regression_dataset
from e3_metrics import regression_scores


models_example = {
    'DecisionTreeRegressor': (

        DecisionTreeRegressor,

        {
            'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
            'splitter': ['best', 'random'],
            'max_depth': [8, 16, 32, 64, 128, None],
        }

    ),
}


regression_models = {
    BayesianRidge.__name__: (BayesianRidge, {
        'n_iter': [150, 300, 450],
        'tol': [1e-2, 1e-3, 1e-4],
        'alpha_1': [1e-5, 1e-6, 1e-7],
        'alpha_2': [1e-5, 1e-6, 1e-7],
        'lambda_1': [1e-5, 1e-6, 1e-7],
        'lambda_2': [1e-5, 1e-6, 1e-7],
    }),
    DecisionTreeRegressor.__name__: (DecisionTreeRegressor, {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': [8, 16, 32, 64, 128, None],
    }),
    DummyRegressor.__name__: (DummyRegressor, {}),
    ExtraTreeRegressor.__name__: (ExtraTreeRegressor, {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': [8, 16, 32, 64, 128, None],
    }),
    ElasticNet.__name__: (ElasticNet, {
        'alpha': [0.2, 0.4, 0.6, 0.8, 1],
        'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1],
        'tol': [1e-5, 1e-4, 1e-3],
        'selection': ['cyclic', 'random'],
    }),
    GaussianProcessRegressor.__name__: (GaussianProcessRegressor, {
        'alpha': [1e-8, 1e-9, 1e-10, 1e-11, 1e-12],
        'n_restarts_optimizer': [0, 1, 2, 3],
        'normalize_y': [True, False],
    }),
    KernelRidge.__name__: (KernelRidge, { # KRR uses squared error loss while support vector regression uses epsilon-insensitive loss
        'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'kernel': ['linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'cosine'], # additive_chi2 only works with postive values. Sigmoid slow
        'degree': [2, 3, 4, 5, 6],
        'coef0': [0.0, 0.5, 1.0],
    }),
    KNeighborsRegressor.__name__: (KNeighborsRegressor, {
        'n_neighbors': list(range(1, 50)),
        'weights': ['uniform', 'distance'],
        'p': [2, 3, 4],
    }),
    # Lars.__name__: (Lars, {}), # Very high errors across multiple datasets
    Lasso.__name__: (Lasso, {
        'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'tol': [1e-2, 1e-3, 1e-4],
        'selection': ['cyclic', 'random'],
    }),
    LinearRegression.__name__: (LinearRegression, {}),
    LinearSVR.__name__: (LinearSVR, {
        'epsilon': [0.0, 0.5, 1.0],
        'tol': [1e-3, 1e-4, 1e-5],
        'C': [0.01, 0.1, 1.0, 10],
        'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'],
        'intercept_scaling': [0.001, 0.1, 1, 10],
        'max_iter': [500, 1000, 1500],
    }),
    MLPRegressor.__name__: (MLPRegressor, {
        'learning_rate': [ 'constant', 'invscaling', 'adaptive' ],
        'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (1,1,), (2,2,), (3,3,), (4,4,)],
        'activation': ['relu'],
        'solver': ['lbfgs', 'adam', 'sgd'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [ 1, 0.1, 0.01, 0.001 ],
    }),
    NuSVR.__name__: (NuSVR, {
        'nu': [0.2, 0.4, 0.6, 0.8],
        'C': [0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'sigmoid'], # 'linear' too slow
        # 'degree': [2, 3, 4, 5, 6], # not using poly kernel
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.5, 1.0],
        'shrinking': [True, False],
        'tol': [1e-3, 1e-4], # 1e-5 too slow
        'cache_size': [ 500 ],
    }),
    PassiveAggressiveRegressor.__name__: (PassiveAggressiveRegressor, {
        'C': [0.001, 0.01, 0.1],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'epsilon': [0.001, 0.01, 0.1, 1.0],
        # 'average': [True, 10],
        'early_stopping': [True, False],
    }),
    # RadiusNeighborsRegressor # Errors in fitting due to NaNs. Possibly due to radius parameter
    RandomForestRegressor.__name__: (RandomForestRegressor, {
        'n_estimators': [10, 50, 100],
        'criterion': ['absolute_error', 'poisson', 'squared_error'],
        'max_depth': [16, 32, 64, 128, None],
        'max_features':['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
    }),
    Ridge.__name__: (Ridge, {
        'alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'tol': [1e-2, 1e-3, 1e-4],
        'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr'], # sag & saga removed due to NumPy related bugs
    }),
    SGDRegressor.__name__: (SGDRegressor, {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [1e-3, 1e-4, 1e-5],
        'l1_ratio': [0.0, 0.15, 0.5, 1.0],
        'max_iter': [100, 1000, 10000],
        'tol': [1e-2, 1e-3, 1e-4],
        'epsilon': [0.001, 0.01, 0.1, 1.0],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'early_stopping': [True, False],
    }),
    SVR.__name__: (SVR, {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5, 6],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.5, 1.0],
        'tol': [1e-3, 1e-4, 1e-5],
        'C': [0.01, 0.1, 1.0, 10, 100],
        'epsilon': [0.0, 0.5, 1.0],
        'shrinking': [True, False],
        'max_iter': [50, 100, 150, 200, 250, 300],
    }),
}


classification_models = {
    BernoulliNB.__name__: (BernoulliNB, {
        'alpha': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    }),
    DecisionTreeClassifier.__name__: (DecisionTreeClassifier, {
        'criterion': ['gini', 'entropy'], # log_loss available in v1.1.1
        'splitter': ['best', 'random'],
        'max_depth': [8, 16, 32, 64, 128, None],
        'min_samples_split': [2, 4, 8, 16, 32, 64, 128],
        'min_samples_leaf': [1, 10, 100],
        'max_features': [None], # handled before modelling
        'max_leaf_nodes': [32, 64, 128, None], # use > 8
        'class_weight': ['balanced'],
    }),
    DummyClassifier.__name__: (DummyClassifier, {}),
    GaussianNB.__name__: (GaussianNB, {
        'var_smoothing': [1e-8, 1e-9, 1e-10]
    }),
    KNeighborsClassifier.__name__: (KNeighborsClassifier, {
        'n_neighbors': [2, 4, 8, 16],
        'weights': ['distance'], # not uniform (slow and worse)
        'algorithm': ['brute'], # brute recommended for spare inputs
        'metric': ['euclidean', 'cosine', 'manhattan', 'chebyshev'],
    }),
    LinearSVC.__name__: (LinearSVC, {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'penalty': ['l1', 'l2'],
        'multi_class': ['ovr', 'crammer_singer'],
        'class_weight': ['balanced'],
        'max_iter': [100, 1000, 10000],
    }),
    LogisticRegression.__name__: (LogisticRegression, {
        'C': [0.1, 1, 10], # 0.01 and 100 gave volatile results
        'tol': [1e-4, 1e-5, 1e-6], # <1e-4 can be slow
        'class_weight': ['balanced'],
        'multi_class': ['multinomial', 'ovr'],
        'penalty': ['l2', 'none'], # penalties supported by lbfgs
        'solver': ['lbfgs'], # 'newton-cg' too slow, 'lbfgs' fastest, liblinear limited to ovr
    }),
    NearestCentroid.__name__: (NearestCentroid, {
        'metric': ['euclidean', 'cosine', 'manhattan', 'chebyshev'],
    }),
    PassiveAggressiveClassifier.__name__: (PassiveAggressiveClassifier, {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'class_weight': ['balanced'],
        'early_stopping' : [False],
        'max_iter': [100, 1000, 10000],
    }),
    Perceptron.__name__: (Perceptron, {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'alpha': [1e-6, 1e-5, 1e-4, 1e-3],
        'tol': [1e-2, 1e-3, 1e-4],
        'early_stopping': [True, False],
        'class_weight': ['balanced'],
        'max_iter': [100, 1000, 10000],
    }),
    RandomForestClassifier.__name__: (RandomForestClassifier, {
        'criterion': ['gini', 'entropy'], # log_loss
        'max_features': [None], # handled before modelling
        'max_depth': [8, 16, 32, 64, 128, None],
        'min_samples_split': [2, 4, 8, 16, 32, 64, 128],
        'min_samples_leaf': [1, 10, 100],
        'max_leaf_nodes': [32, 64, 128, None], # use > 8
        'class_weight': ['balanced', 'balanced_subsample'],
    }),
    SVC.__name__: (SVC, {
        'C': [1, 10, 100], # use >= 1
        'tol': [1e-3, 1e-4], # 1e-5], (too slow)
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'shrinking': [True, False],
        'class_weight': ['balanced'],
        'probability': [True],
        'max_iter': [100, 1000, 10000],
    }),
}


if __name__ == '__main__':

    _, X, y = create_regression_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for model_name, elements in regression_models.items():
        print('\n', model_name)
        model = elements[0]() # Constructor
        distributions = elements[1] # hyperparameter search space

        # Train the model
        search = RandomizedSearchCV(model, param_distributions=distributions, n_iter=10, verbose=1)
        predictions = search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        scores = regression_scores(y_test, predictions)
        print(scores['R2'])

