"""Creating custom models in a class to be compatible with scikit-learn"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from src.e1_create_dataset import create_classification_dataset
from src.e3_metrics import classification_scores


class ExampleModel:
    """Empty class with methods that are needed to work with scikit-learn"""

    search_space: dict = {}

    def __init__(self, **kwargs):
        """Instantiate"""

    def fit(self, X, y):
        """Train a model"""

    def predict(self, X):
        """Make predictions"""
        return np.ones(len(X))

    def get_params(self, *_, **__):
        """Return parameters as a dictionary"""
        return {}

    def set_params(self, **params):
        """Update parameters"""
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DNN():
    """Densely-connected Neural Network classifier"""

    search_space: dict = {
        'architecture': [
            [64], [32], [16], [8], [4], [2], [1],  # 1 layer
            (64, 32), (32, 16), (16, 8),  # 2 layers
            # etc.
        ],
        'batch_normalization': [True, False],
        'batch_size': [32],
        'dropout': [None, 0.1, 0.2, 0.3, 0.4],
        'early_stopping': [None, 3, 5, 7, 9],
        'epochs': [5, 10, 15, 20],
        'final_activation': ['sigmoid', 'tanh'],
        'hidden_activation': ['sigmoid', 'tanh', 'relu'],
        'optimizer': ['adam'],
        'reduce_lr': [True, False],
    }

    def __init__(self, **kwargs):
        # Get values or assign default values if missing
        self.architecture = kwargs.get('architecture', (64, 32))
        self.batch_normalization = kwargs.get('batch_normalization', True)
        self.batch_size = kwargs.get('batch_size', 32)
        self.dropout = kwargs.get('dropout', 0.2)
        self.early_stopping = kwargs.get('early_stopping', 5)
        self.epochs = kwargs.get('epochs', 5)
        self.final_activation = kwargs.get('final_activation', 'sigmoid')
        self.hidden_activation = kwargs.get('hidden_activation', 'sigmoid')
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.reduce_lr = kwargs.get('reduce_lr', True)

        self.verbose = kwargs.get('verbose', 0)
        self.input_shape = None

    def fit(self, X, y):
        """Train a densely-connected neural network

        :param pd.DataFrame X: Features
        :param np.ndarray y: Target
        """
        self.input_shape = (X.shape[1],)

        # Initialize class weights
        label_counts = dict(pd.Series(y).value_counts())
        num_labels = len(label_counts)
        if num_labels < 2:
            raise ValueError('Only one class found in y')

        self.class_weights = {
            int(k): (1 / v) * (y.shape[0] / num_labels) / 2
            for k, v in label_counts.items()
        }
        y = to_categorical(y)

        # Initialize callbacks
        callbacks = [TerminateOnNaN()]
        if self.early_stopping is not None:
            callbacks += [EarlyStopping(
                monitor='val_loss', mode='min',
                patience=self.early_stopping, verbose=self.verbose
            )]

        if self.reduce_lr:
            callbacks += [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)]

        # Create model
        self.model = Sequential([Input(self.input_shape)])
        for units in self.architecture:
            self.model.add(Dense(units))

            if self.batch_normalization:
                self.model.add(BatchNormalization())

            self.model.add(Activation(self.hidden_activation))

            if self.dropout is not None:
                self.model.add(Dropout(self.dropout))

        self.model.add(Dense(num_labels, activation=self.final_activation))

        # Compile model
        metric = CategoricalAccuracy('balanced_accuracy')
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[metric])
        if self.verbose > 1:
            print(self.model.summary())

        # Train model
        self.model.fit(
            X, y,
            validation_split=0.1,
            class_weight=self.class_weights,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks
        )

    def predict(self, X):
        """Make predictions"""
        predict_raw = self.model.predict(X, verbose=self.verbose)
        predictions = np.argmax(predict_raw, axis=1)
        return predictions

    def get_params(self, **_):
        """Get parameters (scikit-learn compatible)"""
        return {
            'architecture': self.architecture,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'final_activation': self.final_activation,
            'hidden_activation': self.hidden_activation,
            'optimizer': self.optimizer,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        """Set parameters (scikit-learn compatible)"""
        if not params:
            return self

        for key, value in params.items():
            setattr(self, key, value)
        return self


def run():
    """Run this exercise"""
    custom_models = {
        'DecisionTreeClassifier': (DecisionTreeClassifier, {'max_depth': [8, 16, 32, 64, 128, None]}),
        ExampleModel.__name__: (ExampleModel, ExampleModel.search_space),
        DNN.__name__: (DNN, DNN.search_space),
    }

    _, features, target = create_classification_dataset(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=0, stratify=target)

    for model_name, (model, distributions) in custom_models.items():
        print(f'\nTraining: {model_name}')
        # Train the model
        search = RandomizedSearchCV(
            model(),
            distributions,
            n_iter=5,
            cv=2,
            n_jobs=1,
            scoring='accuracy',
            verbose=1,
        )
        predictions = search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        scores = classification_scores(y_test, predictions)
        print(scores['accuracy'])


if __name__ == '__main__':
    run()
