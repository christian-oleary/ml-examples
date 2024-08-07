"""Run ML examples."""

# pylint: disable=import-outside-toplevel
# flake8: noqa

from __future__ import annotations
import typing
import sys

from ml import (
    e1_create_dataset,
    e2_train_models,
    e3_metrics,
    e4_model_testing,
    e5_recording_scores,
    e6_hyperparameter_optimization,
    e7_nested_cross_validation,
    e8_handling_models,
    e9_pipelines,
    e10_serialization,
    e12_time_series_features,
    e13_feature_analysis,
)


class Runner:
    """Example of delaying imports to prevent TF/Torch slowing down other executions."""

    def __init__(self, exercise: int, error: str) -> None:
        """Set exercise but do not import."""
        self.exercise = exercise
        self.error = error

    def run(self):
        """Run exercise."""
        if self.exercise == 11:
            from ml import e11_custom_models

            e11_custom_models.run()
        elif self.exercise == 14:
            from ml import e14_pytorch

            e14_pytorch.run()
        else:
            raise ValueError(self.error)


ERROR = 'Provide integer to select options. See README.md for details'

if len(sys.argv) == 1:
    print(ERROR)
    sys.exit()

option = int(sys.argv[1])
options: typing.Any = {
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
    11: Runner(11, ERROR),
    12: e12_time_series_features,
    13: e13_feature_analysis,
    14: Runner(14, ERROR),
}

try:
    options[option].run()
except KeyError:
    print(ERROR)
