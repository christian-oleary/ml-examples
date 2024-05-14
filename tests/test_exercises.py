"""Example basic usage of pytest"""
# pylint: disable=multiple-statements

from examples.e1_create_dataset import run as run_e1
from examples.e2_train_models import run as run_e2
from examples.e3_metrics import run as run_e3
from examples.e4_model_testing import run as run_e4
from examples.e5_recording_scores import run as run_e5
from examples.e6_hyperparameter_optimization import run as run_e6
from examples.e7_nested_cross_validation import run as run_e7
from examples.e8_handling_models import run as run_e8
from examples.e9_pipelines import run as run_e9
from examples.e10_serialization import run as run_e10
from examples.e11_custom_models import run as run_e11
from examples.e12_time_series_features import run as run_e12
from examples.e13_feature_analysis import run as run_e13


def test_e1(): run_e1()
def test_e2(): run_e2()
def test_e3(): run_e3()
def test_e4(): run_e4()
def test_e5(): run_e5()
def test_e6(): run_e6(200)
def test_e7(): run_e7()
def test_e8(): run_e8()
def test_e9(): run_e9()
def test_e10(): run_e10()
def test_e11(): run_e11()
def test_e12(): run_e12()
def test_e13(): run_e13()
