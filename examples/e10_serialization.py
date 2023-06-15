from joblib import dump, load
from e2_train_models import train_regression

# Recommended reading: https://scikit-learn.org/stable/model_persistence.html


if __name__ == '__main__':
    # Train an example model
    model, _, X, ___ = train_regression()

    # Save model as a file
    dump(model, 'model.joblib')

    # Load model model from file
    loaded_model = load('model.joblib')
    predictions = loaded_model.predict(X)
    print('Finished')
