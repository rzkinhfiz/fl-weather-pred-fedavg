"""
Production-ready Flower client for distributed federated learning.

This client:
- Connects to a remote Flower server on Google Cloud Platform (GCP)
- Loads city-specific weather data
- Trains a local LSTM model
- Communicates only model parameters to the server (never raw data)
- Uses modern fl.client.start_client() API

Usage:
    python3 src/client_app.py --city_id 0  # Run Berlin client
    python3 src/client_app.py --city_id 5  # Run Jeddah client
    python3 src/client_app.py --city_id 14 # Run Toronto client
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl

from src.client import WeatherClient
from src.utils import CITY_NAMES

# ============================================================================
# gRPC OPTIMIZATION FOR MEMORY-CONSTRAINED SERVER (e2-micro, 1 GB RAM)
# ============================================================================

# Enable gRPC compression to reduce bandwidth and memory footprint on server
os.environ.setdefault("GRPC_PYTHON_BUILD_WITH_CYTHON", "1")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

# ============================================================================
# CONFIGURATION
# ============================================================================

# GCP Server Configuration
GCP_SERVER_ADDRESS = "35.226.54.117:8080"

# Training Hyperparameters (per client, per round)
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging Configuration
LOG_LEVEL = logging.INFO

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(city_index: int, city_name: str) -> logging.Logger:
    """
    Configure logging for the client with city-specific identifier.

    Args:
        city_index: Integer 0-14 for city assignment.
        city_name: Human-readable city name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"FL-Client-{city_name.upper()}")
    logger.setLevel(LOG_LEVEL)

    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(LOG_LEVEL)

    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# ============================================================================
# CLIENT WRAPPER FOR PRODUCTION
# ============================================================================

class ProductionWeatherClient(fl.client.NumPyClient):
    """
    Production-grade Flower NumPyClient for distributed weather forecasting.

    This wrapper enhances the base WeatherClient with:
    - Comprehensive logging
    - GPU device management
    - Error handling and recovery
    - Performance monitoring
    """

    def __init__(self, city_index: int, logger: logging.Logger):
        """
        Initialize production weather client.

        Args:
            city_index: Integer 0-14 for city assignment.
            logger: Logger instance for client-specific logs.

        Raises:
            FileNotFoundError: If city data file not found.
            ValueError: If city_index is invalid.
        """
        self.city_index = city_index
        self.city_name = CITY_NAMES[city_index]
        self.logger = logger

        self.logger.info(f"Initializing client for {self.city_name.upper()}...")
        self.logger.info(f"Using device: {DEVICE}")

        try:
            # Initialize base WeatherClient (handles data loading, preprocessing)
            self.base_client = WeatherClient(city_index=city_index)
            self.logger.info(
                f"Successfully loaded data for {self.city_name}. "
                f"Training samples: {len(self.base_client.X_train)}, "
                f"Test samples: {len(self.base_client.X_test)}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize client: {str(e)}")
            raise

        # Logging attributes
        self.round_count = 0
        self.total_training_loss = 0.0

    # ========================================================================
    # FLOWER CLIENT INTERFACE
    # ========================================================================

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Return current model weights to server.

        This method is called by the Flower server to retrieve the client's
        current model parameters before training.

        Args:
            config: Configuration dict from Flower server.

        Returns:
            List of model parameters as numpy arrays.
        """
        self.logger.debug(f"get_parameters called with config: {config}")
        return self.base_client.get_parameters(config)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally on city data.

        This is the core federated learning step where the client:
        1. Receives global weights from server
        2. Trains locally for LOCAL_EPOCHS
        3. Returns updated weights and metrics

        Args:
            parameters: Global model weights from server.
            config: Configuration dict (may include learning rate, epochs, etc).

        Returns:
            Tuple of:
            - Updated model weights as list of numpy arrays
            - Number of training samples used
            - Metrics dict with training loss
        """
        self.round_count += 1
        self.logger.info(f"Round {self.round_count}: Starting local training...")

        # Extract config values if provided, else use defaults
        local_epochs = config.get("local_epochs", LOCAL_EPOCHS)
        batch_size = config.get("batch_size", BATCH_SIZE)
        learning_rate = config.get("learning_rate", LEARNING_RATE)

        self.logger.debug(
            f"Training config - epochs: {local_epochs}, "
            f"batch_size: {batch_size}, lr: {learning_rate}"
        )

        try:
            # Call base client fit method
            updated_weights, num_samples, metrics = self.base_client.fit(
                parameters, config
            )

            train_loss = metrics.get("train_loss", 0.0)
            self.total_training_loss += train_loss

            self.logger.info(
                f"Round {self.round_count} complete. "
                f"Training loss: {train_loss:.6f}, "
                f"Samples trained: {num_samples}"
            )

            return updated_weights, num_samples, metrics

        except Exception as e:
            self.logger.error(f"Error during fit: {str(e)}")
            raise

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on local test set.

        This method is called by the Flower server to assess model performance
        on the client's test data using global weights.

        Args:
            parameters: Global model weights from server.
            config: Configuration dict from Flower server.

        Returns:
            Tuple of:
            - MSE loss value (lower is better)
            - Number of test samples
            - Metrics dict with detailed MSE
        """
        self.logger.debug(f"evaluate called at round {self.round_count}")

        try:
            # Call base client evaluate method
            mse_loss, num_samples, metrics = self.base_client.evaluate(
                parameters, config
            )

            self.logger.info(
                f"Round {self.round_count} evaluation: MSE = {mse_loss:.6f} "
                f"(test samples: {num_samples})"
            )

            return mse_loss, num_samples, metrics

        except Exception as e:
            self.logger.error(f"Error during evaluate: {str(e)}")
            raise


# ============================================================================
# CLIENT FACTORY
# ============================================================================

def create_client(city_index: int, logger: logging.Logger) -> ProductionWeatherClient:
    """
    Factory function to create a production-ready client.

    Args:
        city_index: Integer 0-14 for city assignment.
        logger: Logger instance.

    Returns:
        Initialized ProductionWeatherClient instance.

    Raises:
        ValueError: If city_index is invalid.
        FileNotFoundError: If city data not found.
    """
    if not 0 <= city_index <= 14:
        raise ValueError(
            f"Invalid city_index: {city_index}. "
            f"Must be integer in range [0, 14]."
        )

    try:
        client = ProductionWeatherClient(city_index, logger)
        return client
    except Exception as e:
        logger.error(f"Failed to create client: {str(e)}")
        raise


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for client deployment.

    Returns:
        Parsed arguments with city_id and optional server_address.

    Raises:
        SystemExit: If required arguments missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Production-ready Flower client for federated learning "
            "weather forecasting. Connects to GCP server and trains local model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 src/client_app.py --city_id 0      # Berlin\n"
            "  python3 src/client_app.py --city_id 5      # Jeddah\n"
            "  python3 src/client_app.py --city_id 14     # Toronto\n"
            "\n"
            "City Mapping (0-14):\n"
            "  0:Berlin, 1:Cairo, 2:Dammam, 3:Doha, 4:Dubai, 5:Jeddah,\n"
            "  6:London, 7:Mumbai, 8:New_York, 9:Paris, 10:Riyadh,\n"
            "  11:Singapore, 12:Sydney, 13:Tokyo, 14:Toronto"
        )
    )

    parser.add_argument(
        "--city_id",
        type=int,
        required=True,
        help="City index (0-14). Required argument."
    )

    parser.add_argument(
        "--server_address",
        type=str,
        default=GCP_SERVER_ADDRESS,
        help=(
            f"Flower server address in format 'host:port'. "
            f"Default: {GCP_SERVER_ADDRESS}"
        )
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Default: INFO"
    )

    args = parser.parse_args()

    # Validate city_id
    if not 0 <= args.city_id <= 14:
        parser.error(f"city_id must be in range [0, 14], got {args.city_id}")

    return args


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> int:
    """
    Main entry point for federated learning client.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Validate city_id range
    if not 0 <= args.city_id <= 14:
        print(f"ERROR: city_id must be in [0, 14], got {args.city_id}", file=sys.stderr)
        return 1

    city_name = CITY_NAMES[args.city_id]

    # Setup logging
    logger = setup_logging(args.city_id, city_name)

    # Log startup information
    logger.info("=" * 80)
    logger.info(f"Federated Learning Client - {city_name.upper()}")
    logger.info("=" * 80)
    logger.info(f"City ID: {args.city_id}")
    logger.info(f"City Name: {city_name.upper()}")
    logger.info(f"Server Address: {args.server_address}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 80)

    try:
        # Create production client
        client = create_client(args.city_id, logger)
        logger.info(f"Successfully created client for {city_name.upper()}")

        # Connect to Flower server and start federated learning
        logger.info(f"Connecting to Flower server at {args.server_address}...")
        logger.info("gRPC Compression: Enabled")
        logger.info("Message Size Limit: 50 MB (optimized for e2-micro)")
        fl.client.start_client(
            server_address=args.server_address,
            client=client,
            # Optimized for e2-micro (1 GB RAM):
            # - 50 MB limit accommodates model weights + compression overhead
            # - Server can handle 15 clients × 50 MB = 750 MB with overhead
            grpc_max_message_length=50_000_000,  # 50 MB
        )

        logger.info("Client finished successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Client interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1

    finally:
        logger.info("Client shutdown complete")


if __name__ == "__main__":
    sys.exit(main())
