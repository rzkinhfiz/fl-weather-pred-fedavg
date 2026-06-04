"""
Data loading, scaling, and sequence generation utilities for FL weather forecasting.

This module handles:
- Loading city-specific CSV data
- Feature normalization with StandardScaler
- Sliding window sequence generation for LSTM training
- Train/test chronological split
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Global hyperparameters (imported from config/simulation later)
SEQUENCE_LENGTH = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# City index to city name mapping (alphabetical order)
CITY_NAMES = [
    "berlin", "cairo", "dammam", "doha", "dubai",
    "jeddah", "london", "mumbai", "new_york", "paris",
    "riyadh", "singapore", "sydney", "tokyo", "toronto"
]


def load_city_data(city_index: int) -> pd.DataFrame:
    """
    Load a city's weather CSV file and drop non-numeric columns.

    Args:
        city_index: Integer 0-14 corresponding to alphabetically sorted city.

    Returns:
        DataFrame with only numeric columns (Temperature_C, Humidity_%, etc.).

    Raises:
        FileNotFoundError: If city CSV file does not exist.
        ValueError: If city_index is out of valid range [0, 14].
    """
    if not 0 <= city_index <= 14:
        raise ValueError(f"city_index must be in [0, 14], got {city_index}")

    city_name = CITY_NAMES[city_index]
    csv_path = Path(__file__).parent.parent / "data" / "processed" / f"{city_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"City data file not found: {csv_path}\n"
            f"Expected file: data/processed/{city_name}.csv"
        )

    df = pd.read_csv(csv_path)

    # Drop non-numeric columns: Date, City, Condition
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError(
            f"No numeric columns found in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    return numeric_df


def create_sequences(
    data: np.ndarray, seq_len: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM training.

    The last column is assumed to be the target (Temperature_C).

    Args:
        data: 2D numpy array of shape (n_samples, n_features).
        seq_len: Sliding window length (default 5 days).

    Returns:
        X: Shape (n_windows, seq_len, n_features) — input sequences.
        y: Shape (n_windows, 1) — target temperature values.

    Raises:
        ValueError: If data has fewer rows than seq_len.
    """
    if data.shape[0] < seq_len:
        raise ValueError(
            f"Data has {data.shape[0]} rows but seq_len={seq_len}. "
            f"Need at least {seq_len} rows."
        )

    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :])  # All features for seq_len time steps
        y.append(data[i + seq_len, -1])     # Last column (Temperature_C) at time t+seq_len

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y


def prepare_client_data(
    city_index: int, train_ratio: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Load city data, normalize, split chronologically, and create sequences.

    StandardScaler is fit ONLY on training data to prevent data leakage.

    Args:
        city_index: City index 0-14.
        train_ratio: Fraction of data for training (default 0.8).

    Returns:
        X_train, y_train, X_test, y_test (as torch.FloatTensor on DEVICE),
        and the fitted StandardScaler object (for inverse transforms).

    Raises:
        FileNotFoundError: If city data file missing.
        ValueError: If data is invalid or too small.
    """
    # Load raw city data
    df = load_city_data(city_index)
    data = df.values.astype(np.float32)

    # Chronological train/test split (no shuffle)
    split_idx = int(len(data) * train_ratio)
    X_train_raw = data[:split_idx]
    X_test_raw = data[split_idx:]

    # Fit StandardScaler ONLY on training data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    # Transform both train and test
    X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, seq_len=SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, seq_len=SEQUENCE_LENGTH)

    # Convert to PyTorch tensors on device
    X_train = torch.from_numpy(X_train_seq).to(DEVICE)
    y_train = torch.from_numpy(y_train_seq).to(DEVICE)
    X_test = torch.from_numpy(X_test_seq).to(DEVICE)
    y_test = torch.from_numpy(y_test_seq).to(DEVICE)

    return X_train, y_train, X_test, y_test, scaler
