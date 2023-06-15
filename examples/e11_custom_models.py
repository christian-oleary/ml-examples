import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from e1_create_dataset import create_classification_dataset
from e2_train_models import train_classification
from e3_metrics import classification_scores



class DNN:
    """Densely-connected Neural Network classifier"""

    search_space = {
        'architecture': [
            [64], [32], # 1 layer
            (128, 64), (64, 32), (32, 16), # 2 shrinking layers
            (128, 64, 32), (64, 32, 16), # 3 shrinking layers
            (256, 128, 64, 32), (128, 64, 32, 16),  # 4 shrinking layers
            (256, 256), (128, 128), (64, 64), (32, 32), # 2 static layers
            (256, 256, 128, 64, 32), (128, 128, 64, 32), (64, 64, 32), (32, 32), # Static + shrinking layers
        ],
        'batch_normalization': [True, False],
        'batch_size': [32],
        'dropout': [None, 0.1, 0.2, 0.3, 0.4, 0.5],
        'early_stopping': [None, 3, 5, 7, 9],
        'epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'final_activation': ['sigmoid', 'tanh'], # relu performs poorly
        'hidden_activation': ['sigmoid', 'tanh', 'relu'],
        'optimizer': ['adadelta', 'adam', 'rmsprop', 'sgd'],
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
        self.input_shape = (256,)

    def fit(self, X, y):
        verbosity = int(os.environ.get('keras_verbosity', 0))
        self.input_shape = (X.shape[1],)

        # Initialize class weights
        label_counts = dict(pd.Series(y).value_counts())
        num_labels = len(label_counts)
        self.class_weights = {
            int(k): (1 / v) * (y.shape[0] / num_labels) / 2
            for k, v in label_counts.items()
        }
        y = to_categorical(y)

        # Initialize callbacks
        callbacks = [TerminateOnNaN()]
        if self.early_stopping != None:
            callbacks.append(EarlyStopping(monitor='val_loss', mode='min', verbose=verbosity,
                                           patience=self.early_stopping))

        if self.reduce_lr:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001))

        # Create model
        self.model = Sequential()
        for i, shape in enumerate(self.architecture):
            if i == 0:
                self.model.add(Dense(shape, input_shape=self.input_shape))
            else:
                self.model.add(Dense(shape))

            if self.batch_normalization:
                self.model.add(BatchNormalization())

            self.model.add(Activation(self.hidden_activation))

            if self.dropout != None:
                self.model.add(Dropout(self.dropout))

        self.model.add(Dense(num_labels, activation=self.final_activation))

        # Compile model
        metric = CategoricalAccuracy('balanced_accuracy')
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[metric])
        # print(self.model.summary())

        # Train model
        self.model.fit(X, y, validation_split=0.1, class_weight=self.class_weights, epochs=self.epochs,
                       batch_size=self.batch_size, verbose=verbosity, callbacks=callbacks)

    def predict(self, X):
        predict_raw = self.model.predict(X)
        preds = np.argmax(predict_raw, axis=1)
        return preds

    def get_params(self, deep=True):
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
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


class ExampleModel:
    search_space = {}

    def __init__(self, **kwargs):
        # initialize the model here
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.ones(len(X))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


custom_models = {
    'DecisionTreeClassifier': (DecisionTreeClassifier, { 'max_depth': [8, 16, 32, 64, 128, None] }),
    ExampleModel.__name__: (ExampleModel, ExampleModel.search_space),
    DNN.__name__: (DNN, DNN.search_space),
}


if __name__ == '__main__':
    _, X, y = create_classification_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    for model_name, elements in custom_models.items():
        print('\n', model_name)
        model = elements[0]() # Constructor
        distributions = elements[1] # hyperparameter search space

        # Train the model
        search = RandomizedSearchCV(model, param_distributions=distributions, scoring='accuracy', n_iter=5, verbose=1)
        predictions = search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        scores = classification_scores(y_test, predictions)
        print(scores['accuracy'])
