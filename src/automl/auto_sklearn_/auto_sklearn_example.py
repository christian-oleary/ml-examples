"""Example of running auto-sklearn"""

from pathlib import Path
from pprint import pprint
import shutil

import sklearn.datasets
import sklearn.metrics

import autosklearn.regression  # type: ignore
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ############################
    # Data Loading
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1
    )

    ###########################
    # Build and fit a regressor
    tmp_path = Path('tmp/auto-sklearn/')
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,  # 2 minutes
        per_run_time_limit=30,  # 30 seconds
        memory_limit=1024 * 10,  # 10 Gig memory
        n_jobs=2,
        tmp_folder=tmp_path,
    )
    automl.fit(X_train, y_train, dataset_name="diabetes")

    ############################################################################
    # View the models found by auto-sklearn
    print(automl.leaderboard())
    pprint(automl.show_models(), indent=4)

    #####################################
    # Get the Score of the final ensemble
    train_predictions = automl.predict(X_train)
    print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
    test_predictions = automl.predict(X_test)
    print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

    ######################
    # Plot the predictions
    plt.scatter(test_predictions, y_test, label="Test samples", c="#7570b3")
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.legend()
    plt.plot([30, 400], [30, 400], c="k", zorder=0)
    plt.xlim([30, 400])
    plt.ylim([30, 400])
    plt.tight_layout()
    plt.show()
