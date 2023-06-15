import math
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             median_absolute_error, mean_squared_error, r2_score)
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef)

from e2_train_models import train_classification, train_regression

def regression_scores(actual, predicted):
    """Calculate regression metrics

    :param actual: Original time series values
    :param predicted: Predicted time series values
    :return results: Dictionary of results
    """

    results = {
        'MAE': mean_absolute_error(actual, predicted),
        'MAE2': median_absolute_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted),
        'ME': np.mean(actual - predicted),
        'MSE': mean_squared_error(actual, predicted),
        'R2': r2_score(actual, predicted),
        'RMSE': math.sqrt(mean_squared_error(actual, predicted)),
    }

    return results


def classification_scores(actual, predicted):
    """Calculate classification metrics

    :param actual: Original labels
    :param predicted: Predicted Labels
    :return: Dict of scores
    """

    scores = {}
    scores['accuracy'] = accuracy_score(actual, predicted)
    scores['matthews_corrcoef'] = matthews_corrcoef(actual, predicted)

    # Get F1 scores, Average Precision, ROC AUC, etc.
    averages = ['micro', 'macro', 'weighted']
    for average in averages:
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average=average, zero_division=0)
        scores[f'{average}_precision'] = precision
        scores[f'{average}_recall'] = recall
        scores[f'{average}_f1'] = f1
    return scores


if __name__ == '__main__':

    model, predictions, X, y = train_regression()
    scores = regression_scores(y, predictions)
    print('\nregression_scores')
    pprint(scores)

    model, predictions, X, y = train_classification()
    scores = classification_scores(y, predictions)
    print('\nclassification_scores')
    pprint(scores)
