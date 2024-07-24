"""Example basic usage of pytest."""

from ml.e1_create_dataset import run as test_e1
from ml.e2_train_models import run as test_e2
from ml.e3_metrics import run as test_e3
from ml.e4_model_testing import run as test_e4
from ml.e5_recording_scores import run as test_e5
from ml.e6_hyperparameter_optimization import run as run_e6
from ml.e7_nested_cross_validation import run as test_e7
from ml.e8_handling_models import run as test_e8
from ml.e9_pipelines import run as test_e9
from ml.e10_serialization import run as test_e10
from ml.e11_custom_models import run as test_e11
from ml.e12_time_series_features import run as test_e12
from ml.e13_feature_analysis import run as test_e13
from ml.e14_pytorch import run as test_e14


def test_e6():
    """Run exercise 6 with a small amount of data."""
    run_e6(200)


if __name__ == '__main__':
    test_e1()
    test_e2()
    test_e3()
    test_e4()
    test_e5()
    test_e6()
    test_e7()
    test_e8()
    test_e9()
    test_e10()
    test_e11()
    test_e12()
    test_e13()
    test_e14()
