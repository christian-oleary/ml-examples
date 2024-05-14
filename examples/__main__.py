"""Run ML examples"""

import sys

from examples import (
    e1_create_dataset, e2_train_models, e3_metrics, e4_model_testing, e5_recording_scores,
    e6_hyperparameter_optimization, e7_nested_cross_validation, e8_handling_models, e9_pipelines,
    e10_serialization, e12_time_series_features, e13_feature_analysis
)


def run_e11():
    """Example of delaying an import to prevent TensorFlow slowing down other executions"""
    from examples import e11_custom_models  # noqa pylint: disable=import-outside-toplevel
    return e11_custom_models


ERROR = 'Provide integer to select options. See README.md for details'

if len(sys.argv) == 1:
    print(ERROR)
    sys.exit()

option = int(sys.argv[1])
options = {
    1: e1_create_dataset,
    2: e2_train_models,
    3: e3_metrics,
    4: e4_model_testing,
    5: e5_recording_scores,
    6: e6_hyperparameter_optimization,
    7: e7_nested_cross_validation,
    8: e8_handling_models,
    9: e9_pipelines,
    10: e10_serialization,
    11: run_e11(),
    12: e12_time_series_features,
    13: e13_feature_analysis,
}

try:
    options[option].run()
except KeyError:
    print(ERROR)
