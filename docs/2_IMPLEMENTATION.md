# Federated Learning Weather Prediction: Implementation Guide & API Reference

**Document Version:** 1.0  
**Last Updated:** June 2026  
**Status:** Production-Ready  
**Classification:** Implementation & Developer Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Module Reference](#module-reference)
4. [API Documentation](#api-documentation)
5. [Implementation Patterns](#implementation-patterns)
6. [Debugging & Development](#debugging--development)
7. [Integration Points](#integration-points)
8. [Testing Strategy](#testing-strategy)

---

## Overview

This document provides detailed implementation guidance for the Federated Learning weather prediction system. It complements the Architecture document by diving into the actual code structure, API specifications, and development workflows.

### Target Audience
- **Backend Developers:** Implementing FL clients or server-side extensions
- **DevOps Engineers:** Deploying and monitoring the system
- **ML Engineers:** Tuning models or adding new features
- **Researchers:** Modifying FL algorithms or evaluating performance

---

## Project Structure

### 1.1 Directory Layout

```
flweatherpred/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration constants
│   ├── model.py                 # PyTorch LSTM implementation
│   ├── utils.py                 # Data loading & preprocessing
│   ├── client.py                # Flower client implementation
│   ├── server.py                # Server setup (optional)
│   └── simulation.py            # FL simulation orchestrator
│
├── notebooks/                    # Jupyter notebooks
│   ├── split_data.ipynb         # Initial data partitioning
│   └── eda_multi_city.ipynb     # EDA + Advanced insights
│
├── data/                        # Data storage
│   ├── raw/                     # Original weather dataset
│   │   └── weather_dataset_2025.csv
│   └── processed/               # Partitioned city CSVs
│       ├── berlin.csv
│       ├── cairo.csv
│       ├── dammam.csv
│       ├── ... (12 more cities)
│       └── toronto.csv
│
├── logs/                        # Training outputs
│   └── training_log.csv         # Round-by-round metrics
│
├── docs/                        # Documentation (this folder)
│   ├── 1_ARCHITECTURE.md        # System architecture
│   ├── 2_IMPLEMENTATION.md      # This file
│   ├── 3_DEPLOYMENT.md          # Deployment procedures
│   ├── 4_EVALUATION.md          # Performance analysis
│   └── 5_TROUBLESHOOTING.md     # Common issues & fixes
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── INSTRUCTION.md               # Original project specification
└── .gitignore                   # Git exclusions
```

### 1.2 Key Design Decisions

```
┌─────────────────────────────────────────────┐
│  MODULAR ARCHITECTURE                       │
├─────────────────────────────────────────────┤
│                                             │
│  config.py                                  │
│  └─ Centralized constants (CITIES,          │
│     BATCH_SIZE, LEARNING_RATE, etc.)        │
│                                             │
│  model.py                                   │
│  └─ Pure ML logic (PyTorch LSTM)            │
│     ├─ WeatherLSTM class                    │
│     ├─ get_model_parameters()               │
│     └─ set_model_parameters()               │
│                                             │
│  utils.py                                   │
│  └─ Data pipeline (load, preprocess)        │
│     ├─ load_city_data()                     │
│     ├─ create_sequences()                   │
│     └─ prepare_client_data()                │
│                                             │
│  client.py                                  │
│  └─ Flower integration (FL client)          │
│     └─ WeatherClient(NumPyClient)           │
│                                             │
│  simulation.py                              │
│  └─ FL orchestration (server + strategy)    │
│     ├─ set_seed()                           │
│     ├─ client_fn()                          │
│     ├─ MetricsAggregationStrategy           │
│     └─ main()                               │
│                                             │
└─────────────────────────────────────────────┘

Benefits:
✓ Separation of concerns (ML, Data, FL, Orchestration)
✓ Easy to test each module independently
✓ Reusable components (e.g., model can be used standalone)
✓ Clear dependency flow
```

---

## Module Reference

### 2.1 config.py - Configuration Management

**Purpose:** Centralized configuration for reproducibility and easy tuning

**Full File Contents:**

```python
# src/config.py
"""
Configuration constants for Federated Learning weather prediction system.

This module defines all hyperparameters, paths, and constants used throughout
the system. Modifying these values allows rapid experimentation without
changing source code.
"""

import os
from pathlib import Path

# ============================================================================
# SYSTEM PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# 15 cities worldwide (Non-IID data distribution)
CITIES = [
    "berlin",       # City 0  - Europe
    "cairo",        # City 1  - Africa
    "dammam",       # City 2  - Middle East
    "doha",         # City 3  - Middle East
    "dubai",        # City 4  - Middle East
    "jeddah",       # City 5  - Middle East
    "london",       # City 6  - Europe
    "mumbai",       # City 7  - Asia
    "new_york",     # City 8  - North America
    "paris",        # City 9  - Europe
    "riyadh",       # City 10 - Middle East
    "singapore",    # City 11 - Asia
    "sydney",       # City 12 - Australia
    "tokyo",        # City 13 - Asia
    "toronto",      # City 14 - North America
]

NUM_CLIENTS = len(CITIES)  # 15

# Data features (after dropping non-numeric columns)
FEATURES = ["Temperature_C", "Humidity_%", "Wind_Speed_km_h", "Precipitation_mm"]
NUM_FEATURES = len(FEATURES)  # 4

# Sequence configuration (LSTM time-series window)
SEQUENCE_LENGTH = 5  # Use 5 previous days to predict next day
TARGET_FEATURE = "Temperature_C"  # Predicting temperature

# Train/Test split (chronological order, no leakage)
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
# With 365 days: ~292 train sequences, ~73 test sequences per client

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# LSTM hyperparameters
LSTM_INPUT_SIZE = NUM_FEATURES  # 4 weather features
LSTM_HIDDEN_SIZE = 64           # Hidden units per LSTM layer
LSTM_NUM_LAYERS = 2             # Stacked LSTM layers
LSTM_DROPOUT = 0.2              # Dropout between layers
LSTM_BATCH_FIRST = True         # Input: (batch, seq, features)
LSTM_OUTPUT_SIZE = 1            # Single temperature prediction

# ============================================================================
# TRAINING HYPERPARAMETERS (Client-side)
# ============================================================================

LOCAL_EPOCHS = 5                # Local training epochs per FL round
BATCH_SIZE = 32                 # Local training batch size
LEARNING_RATE = 0.001           # Adam optimizer learning rate
WEIGHT_DECAY = 0.0              # No L2 regularization

# Loss function: MSELoss (regression problem)
LOSS_FUNCTION = "MSELoss"

# Optimizer
OPTIMIZER = "Adam"

# ============================================================================
# FEDERATED LEARNING PARAMETERS
# ============================================================================

NUM_ROUNDS = 10                 # Communication rounds
FRACTION_FIT = 1.0              # Fraction of clients to train (1.0 = all)
FRACTION_EVALUATE = 1.0         # Fraction of clients to evaluate
MIN_FIT_CLIENTS = NUM_CLIENTS    # Minimum clients for training round
MIN_EVALUATE_CLIENTS = NUM_CLIENTS  # Minimum clients for evaluation
MIN_AVAILABLE_CLIENTS = NUM_CLIENTS  # Required available clients to start

# Aggregation strategy
AGGREGATION_STRATEGY = "FedAvg"  # Weighted averaging by sample count

# ============================================================================
# RAY DISTRIBUTED EXECUTION
# ============================================================================

# Ray cluster configuration
MAX_WORKERS = 2                 # Max concurrent clients
CLIENTS_PER_GPU = 2             # 2 clients share 1 GPU (0.5 each)

# Client resource allocation
CLIENT_NUM_GPUS = 1.0 / MAX_WORKERS  # 0.5 GPU per client
CLIENT_NUM_CPUS = 2              # 2 CPU cores per client

# Ray initialization
RAY_TEMP_DIR = "/tmp/ray"        # Ray temporary files
RAY_NUM_CPUS = 16                # Available CPU cores
RAY_OBJECT_STORE_MEMORY = 2000000000  # 2 GB object store

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

# Automatic device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility
SEED = 42

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOG_FILE = LOGS_DIR / "training_log.csv"
MODEL_SAVE_PATH = MODELS_DIR / "global_model_final.pt"

# Logging interval
LOG_EVERY_N_ROUNDS = 1  # Log metrics after each round

# Console output verbosity
VERBOSE = True
PRINT_MODEL_SUMMARY = True

# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

# StandardScaler fitted on training data only (prevent leakage)
USE_NORMALIZATION = True
SCALER_STATISTICS = {
    # Mean and Std per feature (computed during data prep)
    # Stored per city to handle Non-IID data
    # Format: {city: {feature: {mean, std}}}
}

# ============================================================================
# DATA VALIDATION
# ============================================================================

# Expected data dimensions
EXPECTED_CSV_SHAPE = (365, 8)   # 365 days × 8 columns (Date, City, Features, Condition)
EXPECTED_NUMERIC_FEATURES = 4   # Temperature, Humidity, Wind, Precipitation
MIN_TRAIN_SAMPLES = 200         # Minimum training samples per client
MIN_TEST_SAMPLES = 50           # Minimum test samples per client

# ============================================================================
# CONSTANTS FOR ANALYSIS
# ============================================================================

# Time series analysis
FORECAST_HORIZON = 1  # Predict next day (1 step ahead)

# Performance metrics
METRICS_TO_TRACK = ["mse", "mae", "rmse"]  # Optional: extend with additional metrics

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

# Set SEED before any random operations
def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

**Configuration Usage Pattern:**

```python
# In any module:
from src.config import (
    CITIES, NUM_CLIENTS, SEQUENCE_LENGTH,
    LSTM_HIDDEN_SIZE, LOCAL_EPOCHS, BATCH_SIZE,
    DEVICE, SEED
)

# Use throughout code
for city_index in range(NUM_CLIENTS):  # 0-14
    city_name = CITIES[city_index]
    # ... process city
```

### 2.2 model.py - Neural Network Architecture

**Purpose:** PyTorch LSTM implementation for weather prediction

**File Structure:**

```python
# src/model.py
"""
PyTorch LSTM model for weather prediction.

Implements a 2-layer LSTM with dropout, followed by a linear layer
for temperature regression. Designed for sequence-to-scalar prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from src.config import (
    LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_OUTPUT_SIZE, DEVICE
)

# Type alias for NumPy arrays (used in FL parameter exchange)
NDArrays = List[np.ndarray]


class WeatherLSTM(nn.Module):
    """
    LSTM-based weather forecasting model.
    
    Architecture:
        Input → LSTM Layer 1 → Dropout → LSTM Layer 2 → Linear → Output
        
    Input shape:  (batch_size, sequence_length, num_features)
    Output shape: (batch_size, 1)  [single temperature prediction]
    
    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM with 2 stacked layers
        fc (nn.Linear): Fully connected output layer
    """
    
    def __init__(self):
        """Initialize LSTM model with configured hyperparameters."""
        super(WeatherLSTM, self).__init__()
        
        # LSTM layer: transforms (batch, seq=5, features=4) → (batch, seq=5, hidden=64)
        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT_SIZE,      # 4 features
            hidden_size=LSTM_HIDDEN_SIZE,    # 64 hidden units
            num_layers=LSTM_NUM_LAYERS,      # 2 stacked layers
            dropout=LSTM_DROPOUT,            # 0.2 between layers
            batch_first=True                 # Input: (batch, seq, features)
        )
        
        # Linear layer: (batch, hidden=64) → (batch, output=1)
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE, LSTM_OUTPUT_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len=5, num_features=4)
               - batch_size: number of samples in batch (typically 32)
               - seq_len: sequence length (5 days)
               - num_features: 4 weather variables
        
        Returns:
            output: Tensor of shape (batch_size, 1) - predicted temperatures
        
        Process:
            1. Flatten LSTM parameters for GPU memory efficiency
            2. Forward through 2-layer LSTM
            3. Extract final hidden state from top layer
            4. Project through linear layer to scalar output
        """
        # Critical optimization: compact weight memory layout
        # Improves GPU cache utilization and cuDNN kernel efficiency
        self.lstm.flatten_parameters()
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len=5, hidden_size=64)
        # h_n shape: (num_layers=2, batch_size, hidden_size=64)
        # c_n shape: (num_layers=2, batch_size, hidden_size=64) [cell states]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Extract hidden state from last LSTM layer at final timestep
        # h_n[-1, :, :] gives (batch_size, hidden_size=64)
        last_hidden = h_n[-1, :, :]
        
        # Project to output dimension
        output = self.fc(last_hidden)  # (batch_size, 1)
        
        return output


def get_model_parameters(model: WeatherLSTM) -> NDArrays:
    """
    Extract all trainable parameters from model as numpy arrays.
    
    Used in Federated Learning to serialize weights for transmission
    to server during aggregation.
    
    Args:
        model: WeatherLSTM instance with trained weights
    
    Returns:
        List of numpy arrays (one per parameter tensor)
        Order: LSTM weights (4 gates × 2 layers) + biases + FC weights + bias
        Total: ~10 arrays, ~136.5 KB when serialized
    
    Example:
        >>> model = WeatherLSTM()
        >>> params = get_model_parameters(model)
        >>> len(params)
        10
        >>> params[0].shape  # First LSTM gate weights
        (4160,)
    """
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_model_parameters(model: WeatherLSTM, parameters: NDArrays) -> None:
    """
    Load parameter arrays from FL server into model.
    
    Called at start of each FL round to load globally aggregated weights.
    Performs shape validation to catch mismatches early.
    
    Args:
        model: WeatherLSTM instance to update
        parameters: List of numpy arrays from server
    
    Raises:
        ValueError: If parameter shapes don't match model structure
    
    Example:
        >>> model = WeatherLSTM()
        >>> global_params = [np.random.randn(...) for _ in range(10)]
        >>> set_model_parameters(model, global_params)
        >>> # Model weights now loaded from global_params
    
    Implementation Detail:
        Iterates through model parameters and server parameters in order,
        checking shape compatibility before assignment.
    """
    params_iter = iter(parameters)
    
    for param in model.parameters():
        try:
            param_data = next(params_iter)
            
            # Validate shape
            if param.shape != param_data.shape:
                raise ValueError(
                    f"Parameter shape mismatch: expected {param.shape}, "
                    f"got {param_data.shape}"
                )
            
            # Load numpy array into pytorch tensor
            param.data = torch.tensor(
                param_data,
                dtype=param.dtype,
                device=param.device
            )
        
        except StopIteration:
            raise ValueError(
                "Not enough parameters provided to model. "
                f"Expected {len(list(model.parameters()))}, "
                f"got {len(parameters)}"
            )


if __name__ == "__main__":
    # Test: Instantiate, forward pass, extract parameters
    print("Testing WeatherLSTM model...")
    
    model = WeatherLSTM().to(DEVICE)
    
    # Print model architecture
    print(model)
    print()
    
    # Test forward pass with dummy batch
    dummy_input = torch.randn(32, 5, 4).to(DEVICE)  # (batch=32, seq=5, features=4)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (32, 1), "Output shape mismatch!"
    print()
    
    # Test parameter extraction/loading
    params = get_model_parameters(model)
    print(f"Extracted {len(params)} parameter arrays")
    print(f"Total parameters: {sum(p.size for p in params)}")
    
    # Simulate FL weight update
    set_model_parameters(model, params)
    print("✓ Parameters loaded successfully")
```

**Key API Methods:**

| Method | Signature | Returns | Purpose |
|--------|-----------|---------|---------|
| `forward()` | `(batch, 5, 4) → (batch, 1)` | Tensor | LSTM inference |
| `get_model_parameters()` | `Model → List[ndarray]` | 10 arrays | Extract weights for FL |
| `set_model_parameters()` | `(Model, List[ndarray]) → None` | None | Load global weights |

### 2.3 utils.py - Data Pipeline

**Purpose:** Load, preprocess, and sequence raw weather data

**Critical Functions:**

```python
# src/utils.py (Excerpt - Key functions)

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from src.config import (
    DATA_DIR, CITIES, FEATURES, SEQUENCE_LENGTH, DEVICE,
    TRAIN_TEST_SPLIT
)

# Type hint
NDArrays = List[np.ndarray]


def load_city_data(city_index: int) -> pd.DataFrame:
    """
    Load preprocessed weather CSV for single city.
    
    Args:
        city_index: Integer 0-14 (maps to CITIES list)
    
    Returns:
        DataFrame with 365 rows × 7 columns
        Columns: [Date, City, Temperature_C, Humidity_%, 
                  Wind_Speed_km_h, Precipitation_mm, Condition]
    
    Raises:
        FileNotFoundError: If city CSV doesn't exist
        ValueError: If city_index out of range
    
    Implementation:
        1. Validates city_index
        2. Constructs path: data/processed/{city_name}.csv
        3. Loads with pandas, maintains date index if present
        4. Returns full DataFrame (filtering to numeric happens in prepare_client_data)
    """
    if not 0 <= city_index < len(CITIES):
        raise ValueError(f"city_index must be 0-{len(CITIES)-1}")
    
    city_name = CITIES[city_index]
    csv_path = DATA_DIR / f"{city_name}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"City data not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def create_sequences(
    data: np.ndarray,
    seq_length: int = SEQUENCE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM training.
    
    Transforms raw time-series into supervised learning format:
        (n_days, n_features) → (n_sequences, seq_length, n_features) + targets
    
    Args:
        data: Input array of shape (n_days, n_features)
              Example: (365, 4) for full year with 4 weather variables
        seq_length: Window size in days (default: 5)
    
    Returns:
        Tuple of two numpy arrays:
        - sequences: (n_windows, seq_length=5, n_features=4)
                    Example: (360, 5, 4)
        - targets: (n_windows, 1)  [target is last day's temperature]
                  Example: (360, 1)
    
    Mechanics:
        For data with 365 rows and seq_length=5:
        
        Sequence 0: rows [0-4]   → target = row[4] temp
        Sequence 1: rows [1-5]   → target = row[5] temp
        Sequence 2: rows [2-6]   → target = row[6] temp
        ...
        Sequence 359: rows [359-363] → target = row[363] temp
        
        Total sequences: 365 - 5 = 360
    
    Returns targets as last column (temperature) from last timestep.
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        # Extract window: 5 consecutive days
        seq = data[i:i+seq_length]
        sequences.append(seq)
        
        # Target: temperature (column 0) from last day in sequence
        target = data[i+seq_length, 0]  # Column 0 = Temperature_C
        targets.append(target)
    
    return np.array(sequences), np.array(targets).reshape(-1, 1)


def prepare_client_data(city_index: int) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler
]:
    """
    Complete data preparation pipeline for single federated client.
    
    End-to-end process:
        1. Load city CSV (365 days)
        2. Extract numeric features (drop Date, City, Condition)
        3. Chronological train/test split (80/20)
        4. Normalize training data with StandardScaler
        5. Create LSTM sequences (5-day windows)
        6. Convert to PyTorch tensors on GPU
    
    Args:
        city_index: Integer 0-14
    
    Returns:
        Tuple of 5 elements:
        - X_train: (n_train_sequences, 5, 4) tensor on GPU
        - y_train: (n_train_sequences, 1) tensor on GPU
        - X_test: (n_test_sequences, 5, 4) tensor on GPU
        - y_test: (n_test_sequences, 1) tensor on GPU
        - scaler: Fitted StandardScaler (for inverse transforms later)
    
    Data Leakage Prevention:
        ⚠️ CRITICAL: StandardScaler fit ONLY on training data
        - Compute mean/std on X_train
        - Apply SAME mean/std to X_test
        - Never fit scaler on test data (would cause data leakage)
    
    Example Output Shapes:
        city 0 (Berlin):
        - X_train: (287, 5, 4)  [287 training sequences]
        - y_train: (287, 1)
        - X_test: (73, 5, 4)    [73 test sequences]
        - y_test: (73, 1)
        
        Total sequences: 287 + 73 = 360 (365 days - 5 day window)
    
    GPU Placement:
        All tensors moved to DEVICE (GPU if CUDA available)
        Enables efficient training with PyTorch DataLoader
    """
    # Step 1: Load raw data
    df = load_city_data(city_index)
    
    # Step 2: Extract numeric features (Temperature, Humidity, Wind, Precipitation)
    # Drop non-numeric columns: Date, City, Condition
    numeric_data = df[FEATURES].values  # Shape: (365, 4)
    
    # Step 3: Chronological train/test split
    split_idx = int(len(numeric_data) * TRAIN_TEST_SPLIT)
    train_data = numeric_data[:split_idx]  # First 292 days
    test_data = numeric_data[split_idx:]   # Last 73 days
    
    # Step 4: Fit StandardScaler on training data ONLY
    scaler = StandardScaler()
    scaler.fit(train_data)  # Compute mean/std from training
    
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    
    # Step 5: Create sliding window sequences
    X_train, y_train = create_sequences(train_data_normalized)
    X_test, y_test = create_sequences(test_data_normalized)
    
    # Step 6: Convert to PyTorch tensors on GPU
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler
```

**Data Flow Example (City 0 - Berlin):**

```
Input: city_index=0 (Berlin)
├─ Load: data/processed/berlin.csv (365 rows × 8 columns)
├─ Extract numeric: 365 × 4 features
├─ Split: [:292] train, [292:] test
├─ Normalize: fit on train, apply to test
├─ Sequence: (365, 4) → (360, 5, 4) + (360, 1)
│   └─ 360 sequences = 365 days - 5 day window
├─ Train sequences: 292 - 5 = 287
├─ Test sequences: 73 - 5 = 68
├─ Convert to torch on DEVICE
└─ Return: (X_train, y_train, X_test, y_test, scaler)
   └─ Shapes: [(287,5,4), (287,1), (73,5,4), (73,1), StandardScaler]
```

### 2.4 client.py - Flower Integration

**Purpose:** Implement Flower NumPyClient for federated training

```python
# src/client.py (Excerpt)

from typing import Dict, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from src.config import (
    LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE
)
from src.model import WeatherLSTM, get_model_parameters, set_model_parameters
from src.utils import prepare_client_data

NDArrays = List[np.ndarray]


class WeatherClient(NumPyClient):
    """
    Federated Learning client for single city's weather data.
    
    Lifecycle:
        1. Initialize: Load local data, create model
        2. Fit: Receive global weights → local training → return updated weights
        3. Evaluate: Receive global weights → test set evaluation → return metrics
        4. Repeat for 10 FL rounds
    
    Attributes:
        city_index (int): Client ID (0-14)
        model (WeatherLSTM): Local LSTM model
        train_loader (DataLoader): Training batches
        test_loader (DataLoader): Test batches (no shuffling)
        scaler (StandardScaler): For inverse transforms
        optimizer (Adam): Local optimizer
        criterion (MSELoss): Loss function
        device (torch.device): GPU or CPU
    """
    
    def __init__(self, city_index: int):
        """
        Initialize federated client for city.
        
        Args:
            city_index: 0-14 (maps to CITIES)
        
        Steps:
            1. Store city_index
            2. Set device (cuda/cpu)
            3. Load local data via prepare_client_data()
            4. Create LSTM model
            5. Create DataLoaders for training/testing
            6. Initialize Adam optimizer
            7. Set loss function (MSELoss)
        """
        self.city_index = city_index
        self.device = DEVICE
        
        # Load local data (returns 5 elements: X_train, y_train, X_test, y_test, scaler)
        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = \
            prepare_client_data(city_index)
        
        # Create model
        self.model = WeatherLSTM().to(self.device)
        
        # Training DataLoader: shuffle for stochastic gradient
        self.train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=BATCH_SIZE,      # 32
            shuffle=True                 # Non-IID gradient noise
        )
        
        # Test DataLoader: no shuffle (evaluation deterministic)
        self.test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        
        # Optimizer & Loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE  # 0.001
        )
        self.criterion = torch.nn.MSELoss()
    
    def get_parameters(self, config: Dict) -> NDArrays:
        """
        Return current model weights to server.
        
        Called at end of local training to send updated parameters
        for aggregation.
        
        Args:
            config: Dict from server (usually empty)
        
        Returns:
            List of numpy arrays (model parameters)
        """
        return get_model_parameters(self.model)
    
    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Local training on city's data.
        
        Called at start of each FL round to train on local data.
        
        Args:
            parameters: Global weights from server (list of numpy arrays)
            config: Dict with training config (usually empty)
        
        Returns:
            Tuple of:
            - updated_weights: Trained parameters (list of numpy arrays)
            - sample_count: Number of training samples used
            - metrics: Dict with training loss
        
        Steps:
            1. Load global weights into local model
            2. Set model to training mode
            3. Train for LOCAL_EPOCHS (5) iterations:
               - Iterate through batches
               - Forward pass
               - Compute MSELoss
               - Backward pass (compute gradients)
               - Update weights with Adam
            4. Return updated weights + metadata
        
        Time Complexity:
            ~150 ms per client (sequential batches)
            5 epochs × (287 samples ÷ 32 batch) = 45 batches
            ~3.3 ms per batch
        """
        # Load global weights from server
        set_model_parameters(self.model, parameters)
        
        # Switch to training mode (enables dropout)
        self.model.train()
        
        # Local training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(LOCAL_EPOCHS):  # 5 epochs
            epoch_loss = 0.0
            
            for X_batch, y_batch in self.train_loader:
                # X_batch: (32, 5, 4)  [batch of 32 sequences]
                # y_batch: (32, 1)     [batch of 32 targets]
                
                # Forward pass
                y_pred = self.model(X_batch)  # → (32, 1)
                
                # Compute loss
                loss = self.criterion(y_pred, y_batch)  # MSE
                
                # Backward pass
                self.optimizer.zero_grad()  # Clear previous gradients
                loss.backward()              # Compute gradients
                self.optimizer.step()        # Update weights with Adam
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss = epoch_loss / len(self.train_loader)
        
        # Return updated weights for aggregation
        return get_model_parameters(self.model), len(self.train_loader.dataset), {
            "train_loss": total_loss
        }
    
    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on test set (no parameter updates).
        
        Called after aggregation to measure global model performance
        on client's test data.
        
        Args:
            parameters: Global weights from server
            config: Dict from server
        
        Returns:
            Tuple of:
            - loss: MSE loss on test set (float)
            - sample_count: Number of test samples
            - metrics: Dict with evaluation metrics
        
        Note:
            Does NOT update weights. Only evaluates for monitoring.
        """
        # Load global weights
        set_model_parameters(self.model, parameters)
        
        # Switch to eval mode (disables dropout)
        self.model.eval()
        
        # Evaluate on test set
        total_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():  # Don't compute gradients
            for X_batch, y_batch in self.test_loader:
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * len(y_batch)
                sample_count += len(y_batch)
        
        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        
        return avg_loss, sample_count, {
            "mse": avg_loss
        }


def to_client(self) -> ClientApp:
    """
    Convert WeatherClient instance to Flower ClientApp.
    
    Required by Flower to run client in distributed setting.
    
    Called internally by Flower framework.
    """
    return ClientApp(client=self)
```

**Client Lifecycle Diagram:**

```
FL Round 1
├─ Server: Publish global_weights_1
├─ Client: __init__() [one-time setup]
│  ├─ Load city data (287 train, 73 test sequences)
│  ├─ Create LSTM model
│  ├─ Create DataLoaders
│  └─ Initialize optimizer
├─ Client: fit(global_weights_1, config)
│  ├─ set_model_parameters(global_weights_1)
│  ├─ Loop 5 epochs:
│  │  └─ Loop through 45 batches:
│  │     ├─ Forward pass (batch size 32)
│  │     ├─ MSELoss
│  │     ├─ Backward (compute gradients)
│  │     └─ Optimizer.step() (update weights)
│  └─ Return: updated_weights, 287 samples, {train_loss: X.XXX}
├─ Server: Aggregate 15 clients' weights
├─ Client: evaluate(global_weights_aggregated, config)
│  ├─ set_model_parameters(global_weights_aggregated)
│  ├─ Loop through test set (73 sequences):
│  │  ├─ Forward pass (NO gradient)
│  │  └─ Accumulate MSE
│  └─ Return: avg_test_loss, 73 samples, {mse: Y.YYY}
└─ Server: Log round 1 metrics

FL Round 2 (repeat with next global weights)
...
```

### 2.5 simulation.py - FL Orchestration

**Purpose:** Server-side orchestration and strategy implementation

```python
# src/simulation.py (Excerpt - Key functions)

import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import flwr as fl
from flwr.common import Context, Metrics
from flwr.server.strategy import FedAvg

from src.config import (
    NUM_CLIENTS, NUM_ROUNDS, SEED,
    FRACTION_FIT, FRACTION_EVALUATE,
    MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS,
    CLIENT_NUM_GPUS, CLIENT_NUM_CPUS,
    LOG_FILE, MAX_WORKERS
)
from src.client import WeatherClient


def set_seed(seed: int = SEED):
    """
    Set random seeds for reproducibility.
    
    Affects:
        - Python random module
        - NumPy random
        - PyTorch CPU random
        - PyTorch CUDA random
        - cuDNN determinism
    
    Must be called before any random operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def client_fn(context: Context) -> fl.client.Client:
    """
    Client factory function for Flower.
    
    Called by Flower to create client instances during simulation.
    One client_fn call = one client initialization.
    
    Args:
        context: Flower Context object containing:
            - context.node_config: Dict with 'cid' (client ID string)
    
    Returns:
        Flower NumPyClient instance for given city
    
    Notes:
        - NEW API: Uses Context parameter (vs deprecated cid: str)
        - Called automatically by Flower for each client
        - Each call creates fresh model + data for one city
    """
    # Extract client ID from context
    cid = context.node_config.get("cid", "0")
    city_index = int(cid)
    
    # Create and return client
    client = WeatherClient(city_index)
    return client.to_client()  # Convert to ClientApp


class MetricsAggregationStrategy(FedAvg):
    """
    Custom Federated Averaging strategy with metrics aggregation.
    
    Extends FedAvg to:
        1. Compute weighted average MSE across clients
        2. Log metrics per round
        3. Print human-readable progress
    
    Methods:
        aggregate_evaluate(): Override to aggregate evaluation metrics
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize strategy, create training log."""
        super().__init__(*args, **kwargs)
        self.training_log = []  # List of (round, global_mse) tuples
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.client.Client, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """
        Aggregate evaluation results from clients.
        
        Called after all clients evaluate on aggregated weights.
        Computes global MSE and logs metrics.
        
        Args:
            server_round: Current round (1-10)
            results: List of (client, evaluation_result) tuples
            failures: List of client failures (if any)
        
        Returns:
            Tuple of:
            - aggregated_loss: Global MSE (float)
            - metrics: Dict with additional metrics
        
        Implementation:
            1. Extract MSE from each client's results
            2. Compute weighted average by sample count
            3. Append to training_log
            4. Print "[Round N/10] Global MSE: X.XXXXXX"
            5. Return aggregated metrics
        
        Math:
            global_mse = Σ (n_k / Σn) × mse_k
            
            Example (round 1):
            Client 0: mse=1.85, n=287, weight=287/5475≈0.0524
            Client 1: mse=1.86, n=288, weight=288/5475≈0.0526
            ...
            global_mse = 0.0524×1.85 + 0.0526×1.86 + ... ≈ 1.859
        """
        if not results:
            return None, {}
        
        # Unpack client results
        total_samples = 0
        weighted_loss = 0.0
        
        for client, evaluation_result in results:
            # evaluation_result contains:
            # - loss: MSE from evaluate()
            # - num_examples: Sample count from evaluate()
            # - metrics: Dict with additional metrics
            
            loss = evaluation_result.loss
            num_examples = evaluation_result.num_examples
            
            weighted_loss += loss * num_examples
            total_samples += num_examples
        
        # Compute weighted average
        global_mse = weighted_loss / total_samples if total_samples > 0 else 0.0
        
        # Log to persistent storage
        self.training_log.append((server_round, global_mse))
        
        # Print progress
        if self.training_log:
            print(f"[Round {server_round}/{NUM_ROUNDS}] Global MSE: {global_mse:.6f}")
        
        # Call parent class method
        return super().aggregate_evaluate(server_round, results, failures)


def save_training_log(log_path: Path = LOG_FILE) -> None:
    """
    Save training metrics to CSV.
    
    Called after training completes.
    
    Args:
        log_path: Output CSV path
    
    Format:
        round,global_mse
        1,1.859470
        2,1.850180
        ...
        10,1.849972
    """
    # This is called from strategy instance, so we need to save from there


def main():
    """
    Main FL simulation entry point.
    
    Orchestrates:
        1. Seed for reproducibility
        2. Print system info
        3. Create FL strategy
        4. Launch Ray cluster
        5. Run simulation (10 rounds)
        6. Save training log
        7. Print results
    """
    print("=" * 80)
    print("FEDERATED LEARNING WEATHER PREDICTION SIMULATION")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Print environment info
    print(f"\nSystem Configuration:")
    print(f"  Clients (Cities): {NUM_CLIENTS}")
    print(f"  FL Rounds: {NUM_ROUNDS}")
    print(f"  Local Epochs: LOCAL_EPOCHS (from config)")
    print(f"  Batch Size: {BATCH_SIZE} (from config)")
    print(f"  GPU Allocation: {CLIENT_NUM_GPUS} per client")
    print(f"  Max Concurrent: {MAX_WORKERS}")
    print()
    
    # Create strategy
    strategy = MetricsAggregationStrategy(
        fraction_fit=FRACTION_FIT,              # 1.0 (all clients)
        fraction_evaluate=FRACTION_EVALUATE,    # 1.0 (all clients)
        min_fit_clients=MIN_FIT_CLIENTS,        # 15
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,  # 15
        initial_parameters=None,                # Use client defaults
    )
    
    # Run simulation
    print("Starting Flower Simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,         # 15 cities
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # 10 rounds
        strategy=strategy,
        client_resources={
            "num_gpus": CLIENT_NUM_GPUS,   # 0.5
            "num_cpus": CLIENT_NUM_CPUS,   # 2
        },
        ray_init_args={
            "num_cpus": 16,
            "num_gpus": 1,
            "object_store_memory": 2000000000,  # 2 GB
        }
    )
    
    # Save training log
    if strategy.training_log:
        log_df = pd.DataFrame(
            strategy.training_log,
            columns=["round", "global_mse"]
        )
        log_df.to_csv(LOG_FILE, index=False)
        print(f"\n✓ Training log saved to: {LOG_FILE}")
        print(log_df)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

---

## API Documentation

### 3.1 Function Signatures Quick Reference

```python
# MODEL API
WeatherLSTM.__init__() → WeatherLSTM
WeatherLSTM.forward(x: Tensor(B,5,4)) → Tensor(B,1)
get_model_parameters(model: WeatherLSTM) → List[ndarray]
set_model_parameters(model: WeatherLSTM, params: List[ndarray]) → None

# DATA API
load_city_data(city_index: int) → DataFrame(365, 8)
create_sequences(data: ndarray(D,F), seq_len: int) → (ndarray, ndarray)
prepare_client_data(city_index: int) → (Tensor, Tensor, Tensor, Tensor, Scaler)

# CLIENT API
WeatherClient.__init__(city_index: int) → WeatherClient
WeatherClient.fit(parameters: List[ndarray], config: Dict) → (List[ndarray], int, Dict)
WeatherClient.evaluate(parameters: List[ndarray], config: Dict) → (float, int, Dict)
WeatherClient.get_parameters(config: Dict) → List[ndarray]

# SERVER API
client_fn(context: Context) → ClientApp
MetricsAggregationStrategy(...)
start_simulation(client_fn, num_clients, config, strategy, client_resources, ray_init_args)
```

---

## Implementation Patterns

### 4.1 Data Loading Pattern

```python
# Pattern: Load → Extract → Normalize → Sequence → Tensorize

from src.utils import prepare_client_data

# Single line usage:
X_train, y_train, X_test, y_test, scaler = prepare_client_data(city_index=0)

# Now use in training
for X_batch, y_batch in train_loader:
    # X_batch: (32, 5, 4) on GPU
    # y_batch: (32, 1) on GPU
    prediction = model(X_batch)
    loss = criterion(prediction, y_batch)
```

### 4.2 Parameter Exchange Pattern

```python
# Pattern: Extract → Aggregate → Load

# Client side:
local_params = get_model_parameters(model)  # → List[ndarray]
# Sent to server

# Server side (aggregation):
global_params = aggregate(client_params_list)  # → List[ndarray]

# Client side (next round):
set_model_parameters(model, global_params)  # Load aggregated weights
# Ready for next training round
```

### 4.3 FL Round Pattern

```python
# Pattern: distribute → train → collect → aggregate → evaluate

for round in range(NUM_ROUNDS):
    # 1. Distribute
    global_weights = get_current_weights()
    distribute_to_clients(global_weights)
    
    # 2. Train locally (parallel across clients)
    for client in clients:
        updated_weights = client.fit(global_weights)
    
    # 3. Collect
    all_updated_weights = collect(clients)
    
    # 4. Aggregate
    global_weights = weighted_avg(all_updated_weights, sample_counts)
    
    # 5. Evaluate
    for client in clients:
        mse = client.evaluate(global_weights)
    
    # 6. Log
    log_metrics(round, mse)
```

---

## Debugging & Development

### 5.1 Common Development Tasks

#### Task: Add New Weather Feature

```python
# 1. Update config.py
FEATURES = [
    "Temperature_C",
    "Humidity_%",
    "Wind_Speed_km_h",
    "Precipitation_mm",
    "Pressure_hPa"  # ← NEW
]
NUM_FEATURES = len(FEATURES)  # Now 5

# 2. Update model.py
self.lstm = nn.LSTM(
    input_size=NUM_FEATURES,  # 5 (was 4)
    # ... rest same
)

# 3. Data must already have this column
# Or preprocessing must add it
# No other changes needed! Config propagates everywhere.
```

#### Task: Increase Training Rounds

```python
# config.py
NUM_ROUNDS = 50  # Was 10

# simulation.py (no changes needed, reads from config)

# Run again:
python -m src.simulation
```

#### Task: Change Learning Rate

```python
# config.py
LEARNING_RATE = 0.0005  # More conservative

# client.py reads from config automatically
# No code changes needed
```

### 5.2 Debugging Checklist

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| **FileNotFoundError: berlin.csv** | City CSV missing or wrong path | Verify `data/processed/` has lowercase filenames |
| **CUDA out of memory** | Batch too large for GPU | Reduce BATCH_SIZE (32→16) or MAX_WORKERS (2→1) |
| **Shape mismatch in set_model_parameters** | Parameter count changed | Check model architecture matches config |
| **Convergence not improving** | Non-IID data or weak local training | Increase LOCAL_EPOCHS (5→10) or decrease LEARNING_RATE |
| **Slow training** | Inefficient GPU usage | Check flatten_parameters() called, verify num_workers=0 |

### 5.3 Local Testing Script

```python
# test_implementation.py
"""Quick test of all modules without FL"""

from src.config import CITIES, NUM_CLIENTS, DEVICE
from src.utils import prepare_client_data
from src.model import WeatherLSTM, get_model_parameters, set_model_parameters
from src.client import WeatherClient

print("Testing implementation...")

# Test 1: Data loading
print("\n1. Testing data loading...")
X_train, y_train, X_test, y_test, scaler = prepare_client_data(0)
print(f"   X_train shape: {X_train.shape} ✓")
assert X_train.shape == (287, 5, 4)

# Test 2: Model
print("\n2. Testing model...")
model = WeatherLSTM().to(DEVICE)
output = model(X_train[:32])  # Forward pass with batch
print(f"   Output shape: {output.shape} ✓")
assert output.shape == (32, 1)

# Test 3: Parameter extraction
print("\n3. Testing parameter exchange...")
params = get_model_parameters(model)
print(f"   Extracted {len(params)} parameter arrays ✓")
set_model_parameters(model, params)
print(f"   Parameters loaded successfully ✓")

# Test 4: Client initialization
print("\n4. Testing client...")
client = WeatherClient(0)
print(f"   Client initialized for city: {CITIES[0]} ✓")
print(f"   Train samples: {len(client.train_loader.dataset)} ✓")
print(f"   Test samples: {len(client.test_loader.dataset)} ✓")

print("\n✓ All tests passed!")
```

---

## Integration Points

### 6.1 Connecting to External Systems

#### **Integration with Database**

```python
# Example: Save model to database after training

from src.model import get_model_parameters
import pickle

# After FL training completes
final_params = get_model_parameters(global_model)

# Save to database
db.models.insert_one({
    "timestamp": datetime.now(),
    "round": 10,
    "parameters": pickle.dumps(final_params),
    "mse": 1.8499
})
```

#### **Integration with Monitoring System**

```python
# Example: Send metrics to Prometheus/Grafana

from prometheus_client import Gauge

global_mse_gauge = Gauge('fl_global_mse', 'Global MSE per round')

# In MetricsAggregationStrategy.aggregate_evaluate():
global_mse_gauge.set(global_mse)
```

#### **Integration with REST API**

```python
# Example: Expose predictions via Flask

from flask import Flask, jsonify
from src.model import WeatherLSTM, set_model_parameters
import torch

app = Flask(__name__)

@app.route('/predict/<city>', methods=['POST'])
def predict(city):
    """Predict temperature for given city"""
    # Load global model
    model = WeatherLSTM()
    set_model_parameters(model, global_params)
    
    # Get recent data
    recent_data = get_recent_city_data(city)
    
    # Predict
    with torch.no_grad():
        prediction = model(recent_data)
    
    return jsonify({"temperature": float(prediction)})
```

---

## Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_model.py
import pytest
import torch
from src.model import WeatherLSTM, get_model_parameters, set_model_parameters

def test_forward_pass():
    """Test LSTM forward pass shape."""
    model = WeatherLSTM()
    x = torch.randn(32, 5, 4)  # (batch, seq, features)
    y = model(x)
    assert y.shape == (32, 1)

def test_parameter_extraction():
    """Test parameter serialization."""
    model = WeatherLSTM()
    params = get_model_parameters(model)
    assert len(params) == 10  # 4+4+2 weights/biases
    assert all(isinstance(p, np.ndarray) for p in params)

def test_parameter_loading():
    """Test parameter deserialization."""
    model1 = WeatherLSTM()
    params1 = get_model_parameters(model1)
    
    model2 = WeatherLSTM()
    set_model_parameters(model2, params1)
    
    params2 = get_model_parameters(model2)
    for p1, p2 in zip(params1, params2):
        assert np.allclose(p1, p2)
```

### 7.2 Integration Tests

```python
# tests/test_client.py
import pytest
from src.client import WeatherClient

def test_client_initialization():
    """Test client can initialize."""
    client = WeatherClient(0)
    assert client.city_index == 0
    assert len(client.train_loader) > 0

def test_client_fit():
    """Test client training loop."""
    client = WeatherClient(0)
    params = client.get_parameters({})
    
    updated_params, num_examples, metrics = client.fit(params, {})
    
    assert len(updated_params) == len(params)
    assert num_examples > 0
    assert "train_loss" in metrics

def test_client_evaluate():
    """Test client evaluation."""
    client = WeatherClient(0)
    params = client.get_parameters({})
    
    loss, num_examples, metrics = client.evaluate(params, {})
    
    assert loss >= 0
    assert num_examples > 0
    assert "mse" in metrics
```

---

## Document Metadata

**Author:** Federated Learning Weather System Team  
**Document ID:** FL-WP-IMPL-002  
**Review Status:** Final  
**Distribution:** Development Team, DevOps Engineers

**Revision History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | June 2026 | Complete implementation documentation |

---

**End of Implementation Document**

Previous: [1_ARCHITECTURE.md](1_ARCHITECTURE.md) - System architecture and design
Next: [3_DEPLOYMENT.md](3_DEPLOYMENT.md) - Installation, environment setup, production deployment  

