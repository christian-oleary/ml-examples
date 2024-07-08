"""PyTorch RNN, GRU and LSTM examples"""

from __future__ import annotations
from typing import Any

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data

from src.e1_create_dataset import create_regression_dataset


class RnnModel(nn.Module):
    """RNN model"""

    RNN_TYPE: Any = nn.RNN

    def __init__(self):
        super().__init__()
        self.lstm = self.RNN_TYPE(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        """Expected by nn.Module"""
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class LstmModel(RnnModel):
    """LSTM model for forecasting"""

    RNN_TYPE = nn.LSTM


class GruModel(RnnModel):
    """GRU model for forecasting"""

    RNN_TYPE = nn.GRU


def train_torch_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model: RnnModel,
    epochs: int = 500,
    batch_size: int = 32,
):
    """Train a PyTorch RNN-based model"""
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=batch_size
    )

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_rmse = np.sqrt(loss_fn(model(X_train), y_train))
                test_rmse = np.sqrt(loss_fn(model(X_test), y_test))
            print(f"Epoch {epoch}: train RMSE {float(train_rmse)}, test RMSE {float(test_rmse)}")
    print(f'Finished training {model}')


def run():
    """Train RNN, GRU and LSTM models using PyTorch

    Example adapted from:
    https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
    """

    _, __, timeseries = create_regression_dataset()
    timeseries = timeseries.values.astype('float32').reshape(-1, 1)
    print(f'timeseries shape: {timeseries.shape}')

    # train-test split for time series
    train_size = int(len(timeseries) * 0.8)
    train, test = timeseries[:train_size], timeseries[train_size:]

    def create_dataset(dataset: np.ndarray, lookback: int) -> tuple:
        features, target = [], []
        for i in range(len(dataset) - lookback):
            features.append(dataset[i: i + lookback])
            target.append(dataset[i + 1: i + lookback + 1])
        return torch.tensor(features), torch.tensor(target)

    lookback = 6
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    for name, model in (
        ('RNN', RnnModel),
        ('LSTM', LstmModel),
        ('GRU', GruModel),
    ):
        print(f'\nTraining {name} model')
        train_torch_model(X_train, y_train, X_test, y_test, model())
