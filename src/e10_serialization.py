"""Saving models to disk"""

from joblib import dump, load

from src.e2_train_models import train_regression

# Recommended reading: https://scikit-learn.org/stable/model_persistence.html


def run():
    """Run this exercise"""
    # Train an example model
    model, _, X, ___ = train_regression()

    # Save model as a file
    path = 'model.joblib'
    dump(model, path)

    # Load model model from file
    loaded_model = load(path)
    predictions = loaded_model.predict(X)
    assert len(predictions) > 0
    print(f'Model saved at {path}')


if __name__ == '__main__':
    run()
