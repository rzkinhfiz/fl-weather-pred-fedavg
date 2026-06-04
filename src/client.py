"""
Flower NumPyClient for federated learning weather forecasting.

Each city client loads its local data, trains its model locally,
and communicates only weights (not raw data) with the aggregator.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl

from src.model import WeatherLSTM, get_model_parameters, set_model_parameters
from src.utils import prepare_client_data

# --- Training Hyperparameters (per client, per round) ---
LOCAL_EPOCHS = 5              # Local training epochs per FL round
BATCH_SIZE = 32               # Mini-batch size
LEARNING_RATE = 0.001         # Adam optimizer learning rate

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeatherClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for federated weather forecasting.

    Each client:
    - Owns one city's weather data (never shared)
    - Trains a local LSTM model
    - Communicates only model weights to the aggregator
    - Receives globally averaged weights from aggregator
    """

    def __init__(self, city_index: int):
        """
        Initialize WeatherClient for a specific city.

        Args:
            city_index: Integer 0-14 for city assignment.

        Raises:
            FileNotFoundError: If city data file not found.
            ValueError: If data loading or preprocessing fails.
        """
        self.city_index = city_index

        # Load and preprocess city data
        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = (
            prepare_client_data(city_index)
        )

        # Determine input feature size from data
        input_size = self.X_train.shape[2]

        # Initialize model
        self.model = WeatherLSTM(input_size=input_size).to(DEVICE)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE
        )
        self.criterion = nn.MSELoss()

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Return current model weights to aggregator.

        Args:
            config: Config dict from Flower server (unused here).

        Returns:
            List of model parameters as numpy arrays.
        """
        return get_model_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally for LOCAL_EPOCHS on city data.

        Receives globally averaged weights from aggregator,
        trains locally, and returns updated weights.

        Args:
            parameters: List of numpy arrays (global weights from aggregator).
            config: Config dict from Flower server (may contain learning rate, etc).

        Returns:
            - Updated model weights as list of numpy arrays
            - Number of training samples used
            - Metrics dict with training loss
        """
        # Load global weights into local model
        self.model = set_model_parameters(self.model, parameters)

        # Create data loader for training
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )

        # Local training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                # Forward pass
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Return updated weights, number of training samples, and metrics
        return (
            get_model_parameters(self.model),
            len(self.X_train),
            {"train_loss": float(avg_loss)},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on local test set.

        Receives global weights, evaluates on local test data,
        and returns MSE loss (lower is better).

        Args:
            parameters: List of numpy arrays (global weights).
            config: Config dict from Flower server (unused here).

        Returns:
            - MSE loss value (float)
            - Number of test samples
            - Metrics dict with detailed MSE
        """
        # Load global weights into model
        self.model = set_model_parameters(self.model, parameters)

        # Create data loader for testing
        test_dataset = TensorDataset(self.X_test, self.y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        # Evaluation loop
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()
                num_batches += 1

        mse_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        return mse_loss, len(self.X_test), {"mse": float(mse_loss)}
