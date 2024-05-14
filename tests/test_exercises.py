"""Example basic usage of pytest"""
# pylint: disable=multiple-statements

from src.e1_create_dataset import run as run_e1
from src.e2_train_models import run as run_e2
from src.e3_metrics import run as run_e3
from src.e4_model_testing import run as run_e4
from src.e5_recording_scores import run as run_e5
from src.e6_hyperparameter_optimization import run as run_e6
from src.e7_nested_cross_validation import run as run_e7
from src.e8_handling_models import run as run_e8
from src.e9_pipelines import run as run_e9
from src.e10_serialization import run as run_e10
from src.e11_custom_models import run as run_e11
from src.e12_time_series_features import run as run_e12
from src.e13_feature_analysis import run as run_e13


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
