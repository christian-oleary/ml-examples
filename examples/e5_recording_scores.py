import csv
import os

import numpy as np

from e2_train_models import train_classification
from e3_metrics import classification_scores


def write_to_csv(path, results):
    """Record modelling results in a CSV file.

    :param str path: the result file path
    :param dict results: a dict containing results from running a model
    """

    np.set_printoptions(precision=4)

    if len(results) > 0:
        headers = sorted(list(results.keys()), key=lambda v: str(v).upper())
        if 'model' in headers:
            headers.insert(0, headers.pop(headers.index('model')))

        for key, value in results.items():
            if value is None or value == '':
                results[key] = 'None'

        is_new_file = not os.path.exists(path)
        with open(path, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            if is_new_file:
                writer.writerow(headers)
            writer.writerow([results[header] for header in headers])


if __name__ == '__main__':
    model, predictions, __, y = train_classification()
    scores = classification_scores(y, predictions)
    scores = {
        'model': 'Decision Tree', # You can record the model name
        'depth': model.tree_.max_depth, # You can record extra information about the model/data/experiment/etc.
        **scores
    }
    write_to_csv('scores.csv', scores)
