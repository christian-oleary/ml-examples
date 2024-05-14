"""Saving model results to an existing CSV file"""

import csv
import os
from pathlib import Path

import numpy as np

from examples.e2_train_models import train_classification
from examples.e3_metrics import classification_scores


def write_to_csv(scores_dir: str, scores_file: str, results: dict):
    """Record modelling results in a CSV file and append if it already exists.

    :param str path: the result file path
    :param dict results: a dict containing results from running a model
    """

    np.set_printoptions(precision=4)

    if len(results) > 0:
        # Sort the headings to make records consistent
        headers = sorted(list(results.keys()), key=lambda v: str(v).upper())
        if 'model' in headers:
            headers.insert(0, headers.pop(headers.index('model')))

        for key, value in results.items():
            if value is None or value == '':
                results[key] = 'None'

        # Create the results directory
        Path(scores_dir).mkdir(exist_ok=True)
        path = Path(scores_dir)/scores_file

        with open(path, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            # Only write the headers on the first attempt
            if not os.path.exists(path):
                writer.writerow(headers)
            # Write data in the order of the headers
            writer.writerow([results[header] for header in headers])


def run():
    """Run this exercise"""
    model, predictions, __, y = train_classification()
    scores = classification_scores(y, predictions)
    scores = {
        'model': 'Decision Tree',  # Record the model name
        'depth': model.tree_.max_depth,  # You can record information about models/experiments/etc.
        **scores
    }
    write_to_csv('results', 'scores.csv', scores)

    # If your data is in a DataFrame, you will see a better example in the next exercise


if __name__ == '__main__':
    run()
