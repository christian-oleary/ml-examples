# ml-examples

[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://opensource.org/licenses/MIT)
[![testing: bandit](https://github.com/christian-oleary/ml-examples/actions/workflows/bandit.yml/badge.svg)](https://github.com/christian-oleary/ml-examples/actions/workflows/bandit.yml)
[![linting: pylint](https://github.com/christian-oleary/ml-examples/actions/workflows/pylint.yml/badge.svg)](https://github.com/christian-oleary/ml-examples/actions/workflows/pylint.yml)
[![testing: pytest](https://github.com/christian-oleary/ml-examples/actions/workflows/pytest.yml/badge.svg)](https://github.com/christian-oleary/ml-examples/actions/workflows/pytest.yml)

Machine Learning Examples

## Running ML Examples

### Installation

With Miniconda:

```bash
conda create -n ml python=3.10 -y
conda activate ml
conda install pre_commit
pip install -r requirements.txt
```

Read through examples before running (copy and paste into terminal).

```bash
python -m ml 1   # Creating sample datasets          (e1_create_dataset.py)
python -m ml 2   # Training models with scikit-learn (e2_train_models.py)
python -m ml 3   # Metrics for evaluating models     (e3_metrics.py)
python -m ml 4   # Testing models                    (e4_model_testing.py)
python -m ml 5   # Recording data in CSV files       (e5_recording_scores.py)
python -m ml 6   # Tuning models                     (e6_hyperparameter_optimization.py)
python -m ml 7   # Nested CV                         (e7_nested_cross_validation.py)
python -m ml 8   # Training many models              (e8_handling_models.py)
python -m ml 9   # Pipelines in scikit-learn         (e9_pipelines.py)
python -m ml 10  # Serializing/loading models        (e10_serialization.py)
python -m ml 11  # Creating custom models            (e11_custom_models.py)
python -m ml 12  # Time series feature engineering   (e12_time_series_features.py)
python -m ml 13  # Feature analysis                  (e13_feature_analysis.py)
python -m ml 14  # PyTorch models                    (e14_pytorch.py)
```

## Library-Specific Examples

- [auto-sklearn](src/automl/auto_sklearn_/README.md)

## Concepts to Learn

Machine Learning:

- Training, validation and test sets
- Hyperparameter optimization
- Holdout, Cross-Validation, Nested Cross-Validation
- Leakage
- Underfitting, overfitting
- Curse of dimensionality
- Regularization
- Saving and loading models
- Time series data

Programming:

- Git
- Object-Oriented Programming

## Tips

Machine Learning:

- Collect any metrics that you think may be useful. You don't have  to use them all, but it is easier than repeating experiments because you forgot to record something important.
- Nested Cross-Validation is better than Cross-Validation, Cross-Validation is better than Holdout
- With neural networks, try different architectures/optimizers before tuning subtler hyperparameters like regularization
- If you have good performance on the validation set, but bad performance on test set, then use a bigger validation set
- If you get poor real-world performance, you may need a different/bigger test set or cost function
- Apply early stopping after you have some models performing somewhat well to save time
- Scaling/normalization should come before PCA
- If the train/test curves are not converging:
  - Increase the complexity - if you think you are underfitting due to high bias
    - Move to a more complex model, (e.g. polynomial of higher degree, or a different kind of model, such as a nonparametric one that makes fewer assumptions about the target function)
    - Move to a more complex variant of your current model
  - Add more useful features
  - Remove useless features
- If the curves are converging too slowly:
  - Decrease the complexity - if you think you are overfitting due to high variance
    - Move to a less complex model (e.g. polynomial of a lower degree or a different kind of model that makes more assumptions about the target function)
    - Move to a less complex variant of your current model (e.g. regularization, smoothing)
  - Add more instances
  - Remove noisy samples

Programming:

- Try to limit code repetition as much as possible
- Back up code, preferably in a repository using a website such as GitHub
- Use informative function/class/variable names

## Useful Visual Studio Code Extensions

- autoDocstring - Python Docstring Generator
- autopep8
- Excel Viewer
- FreeMarker
- Git
- indent-rainbow
- Jupyter
- markdownlint
- Markdown Language Features
- Path Intellisense
- Python
- Python Indent
- Trailing Spaces

## Recommended Reading

<https://machinelearningmastery.com/multi-output-regression-models-with-python/>
<https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html>
<https://machinelearningmastery.com/multi-step-time-series-forecasting/>
<https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/>

## Running Tests

```bash
conda activate ml
pip install -r requirements.txt
python -W ignore -m pytest  # run tests in current environment
```

## Pre-commit

```bash
conda activate ml
pip install -r requirements.txt
pre-commit install
pre-commit
```
