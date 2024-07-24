"""Metrics to use when evaluating models."""

import math
from pprint import pprint

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from ml.e2_train_models import train_classification, train_regression


def regression_scores(actual: np.ndarray, predicted: np.ndarray):
    """Calculate regression metrics.

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


def classification_scores(actual: np.ndarray, predicted: np.ndarray):
    """Calculate classification metrics.

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
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average=average, zero_division=0
        )
        scores[f'{average}_precision'] = precision
        scores[f'{average}_recall'] = recall
        scores[f'{average}_f1'] = f1
    return scores


def run():
    """Run this exercise."""
    _, predictions, __, y = train_regression()
    metrics = regression_scores(y, predictions)
    print('\nregression_scores:')
    pprint(metrics)

    _, predictions, __, y = train_classification()
    metrics = classification_scores(y, predictions)
    print('\nclassification_scores:')
    pprint(metrics)


if __name__ == '__main__':
    run()
