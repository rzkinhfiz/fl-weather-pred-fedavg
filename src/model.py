"""
PyTorch LSTM model architecture for weather forecasting.

This module defines the neural network used by each federated learning client
to predict Temperature_C from historical weather sequences.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

# --- LSTM Model Hyperparameters ---
SEQUENCE_LENGTH = 5           # Sliding window: 5 days of history → predict day 6
HIDDEN_SIZE = 64              # LSTM hidden units
NUM_LAYERS = 2                # Stacked LSTM layers
DROPOUT = 0.2                 # Dropout between LSTM layers
OUTPUT_SIZE = 1               # Single regression output: Temperature_C

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeatherLSTM(nn.Module):
    """
    Multi-layer LSTM for time-series weather forecasting.

    Architecture:
        Input (batch, seq_len, input_size)
        → LSTM (64 hidden, 2 layers, 0.2 dropout, batch_first=True)
        → Take last hidden state
        → Linear (64 → 1)
        → Output (batch, 1)

    This is a regression model for continuous Temperature_C prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        output_size: int = OUTPUT_SIZE,
    ):
        """
        Initialize WeatherLSTM.

        Args:
            input_size: Number of input features (numeric weather columns).
            hidden_size: LSTM hidden state dimension (default 64).
            num_layers: Number of stacked LSTM layers (default 2).
            dropout: Dropout probability between LSTM layers (default 0.2).
            output_size: Output dimension (default 1 for regression).
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers with batch_first=True: (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM and linear layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output tensor of shape (batch_size, 1) — predicted Temperature_C.
        """
        # Flatten LSTM parameters for memory efficiency on GPU
        self.lstm.flatten_parameters()
        
        # LSTM forward
        # lstm_out shape: (batch, seq_len, hidden_size)
        # h_n shape: (num_layers, batch, hidden_size)
        # c_n shape: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last time step's output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Fully connected layer
        output = self.fc(last_out)  # (batch, 1)

        return output


def get_model_parameters(model: WeatherLSTM) -> List[np.ndarray]:
    """
    Extract model parameters as numpy arrays for Flower FL.

    Used by Flower client to serialize weights for aggregation.

    Args:
        model: WeatherLSTM instance.

    Returns:
        List of numpy arrays, one per model parameter (weights and biases).
    """
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy())
    return params


def set_model_parameters(
    model: WeatherLSTM, parameters: List[np.ndarray]
) -> WeatherLSTM:
    """
    Set model parameters from numpy arrays received from Flower aggregator.

    Used by Flower client to load globally averaged weights.

    Args:
        model: WeatherLSTM instance to update.
        parameters: List of numpy arrays matching model.parameters() order.

    Returns:
        Updated WeatherLSTM model.

    Raises:
        ValueError: If parameter list length does not match model parameters.
    """
    if len(parameters) != len(list(model.parameters())):
        raise ValueError(
            f"Parameter count mismatch: "
            f"expected {len(list(model.parameters()))}, "
            f"got {len(parameters)}"
        )

    for param, new_value in zip(model.parameters(), parameters):
        param.data = torch.from_numpy(new_value).to(DEVICE)

    return model
