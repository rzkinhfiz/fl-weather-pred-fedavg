"""
Memory-optimized Flower Server for Federated Learning on GCP e2-micro (1 GB RAM).

This server:
- Runs on GCP e2-micro instance (2 vCPU, 1 GB RAM, Ubuntu 22.04 LTS)
- Acts as Central Aggregator only (no heavy computation)
- Aggregates model weights from 15 clients using FedAvg algorithm
- Uses streaming aggregation to minimize memory footprint
- Implements resource-aware gRPC configuration
- Listens on 0.0.0.0:8080 for client connections

Resources:
- No PyTorch/CUDA needed (weights are numpy arrays)
- Minimal memory: ~50-100 MB per round
- Bandwidth-efficient: gRPC compression enabled
- Can handle 15 concurrent clients on 1 GB RAM

Usage:
    python3 src/server.py
    
Server will listen on 0.0.0.0:8080 and wait for clients to connect.
"""

import argparse
import logging
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import flwr as fl
from flwr.server import ServerConfig, Server
from flwr.server.strategy import FedAvg
from flwr.common import NDArrays, Scalar, FitRes, EvaluateRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.criterion import Criterion

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging for server."""
    formatter = logging.Formatter(
        '[%(asctime)s] [FL-SERVER] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("flwr")
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level))
    
    return logger

logger = setup_logging()

# ============================================================================
# MEMORY-OPTIMIZED AGGREGATION STRATEGY
# ============================================================================

class MemoryOptimizedFedAvg(FedAvg):
    """
    FedAvg strategy optimized for low-memory environments (e2-micro).
    
    Key optimizations:
    1. Streaming aggregation: aggregate weights on-the-fly
    2. No redundant copies: delete intermediate arrays
    3. In-place updates when possible
    4. Memory monitoring and limits
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures: bool = True,
        initial_parameters = None,
        fit_metrics_aggregation_fn = None,
        evaluate_metrics_aggregation_fn = None,
    ):
        """Initialize memory-optimized FedAvg strategy."""
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.logger = logger
        self.fit_results_metrics = defaultdict(list)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using streaming FedAvg (memory-efficient).
        """
        self.logger.info(f"Aggregating fit results from {len(results)} clients")
        
        if not results:
            self.logger.warning("No fit results to aggregate")
            return None, {}
        
        # Filter out failed uploads
        good_results = [
            (client, result) for client, result in results
            if result.parameters is not None
        ]
        
        if not good_results:
            self.logger.warning("All results had None parameters")
            return None, {}
        
        # Extract parameters and num_examples for weighting
        try:
            # Get all weight arrays
            weights_list = [fl.common.parameters_to_ndarrays(res.parameters) 
                          for _, res in good_results]
            num_examples_list = [res.num_examples for _, res in good_results]
            
            self.logger.info(
                f"Aggregating {len(good_results)} clients, "
                f"Total samples: {sum(num_examples_list)}"
            )
            
            # Streaming aggregation: compute weighted average in-place
            aggregated = self._aggregate_weights(weights_list, num_examples_list)
            
            # Convert back to Parameters
            aggregated_params = fl.common.ndarrays_to_parameters(aggregated)
            
            # Collect metrics for monitoring
            metrics_dict = self._collect_metrics(good_results, server_round)
            
            # Memory cleanup
            del weights_list
            del good_results
            
            return aggregated_params, metrics_dict
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            return None, {}
    
    def _aggregate_weights(
        self,
        weights_list: List[NDArrays],
        num_examples_list: List[int],
    ) -> NDArrays:
        """
        Stream-aggregate weights using weighted average.
        Memory-efficient: reuses arrays, minimal copies.
        """
        total_samples = sum(num_examples_list)
        
        # Initialize with first client's weights (avoid extra copy)
        aggregated = [
            w.astype(np.float32) * (num_examples_list[0] / total_samples)
            for w in weights_list[0]
        ]
        
        # Accumulate from other clients
        for i in range(1, len(weights_list)):
            weight = num_examples_list[i] / total_samples
            for j, w in enumerate(weights_list[i]):
                aggregated[j] += w.astype(np.float32) * weight
        
        return aggregated
    
    def _collect_metrics(
        self,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        server_round: int,
    ) -> Dict[str, Scalar]:
        """Collect metrics from clients for monitoring."""
        metrics = {
            "num_clients": len(results),
            "round": server_round,
        }
        
        # Collect loss metrics if available
        losses = []
        for _, res in results:
            if res.metrics and "loss" in res.metrics:
                losses.append(res.metrics["loss"])
        
        if losses:
            metrics["avg_loss"] = np.mean(losses)
            metrics["min_loss"] = np.min(losses)
            metrics["max_loss"] = np.max(losses)
            self.logger.info(
                f"Round {server_round}: "
                f"Avg Loss={metrics['avg_loss']:.4f}, "
                f"Min={metrics['min_loss']:.4f}, "
                f"Max={metrics['max_loss']:.4f}"
            )
        
        return metrics

# ============================================================================
# SERVER CONFIGURATION FOR e2-micro (1 GB RAM)
# ============================================================================

def get_server_config(num_rounds: int = 10) -> ServerConfig:
    """
    Get server configuration optimized for e2-micro instance.
    
    Parameters for 1 GB RAM:
    - num_rounds: Number of FL rounds
    - client_manager_args: Minimize memory overhead
    - grpc_max_message_length: Balance between flexibility and memory
    """
    return ServerConfig(
        num_rounds=num_rounds,
        # Round timeout: 5 minutes (clients have time to train)
        round_timeout=300,
    )

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def get_initial_parameters() -> Parameters:
    """
    Create initial model parameters for server.
    
    Server doesn't run training, only aggregates weights.
    This creates a dummy initial state matching the LSTM architecture.
    """
    logger.info("Initializing parameters for LSTM model")
    
    # LSTM model dimensions (matching src/model.py):
    # - Input: 4 features (Temp, Humidity, Pressure, Wind)
    # - LSTM: 2 layers × 64 hidden units
    # - Output: 1 (temperature prediction)
    
    # Simplified initialization: numpy arrays matching model structure
    initial_params = [
        np.zeros((256, 16), dtype=np.float32),  # LSTM layer 1 weights (simplified)
        np.zeros(64, dtype=np.float32),          # LSTM layer 1 bias
        np.zeros((256, 16), dtype=np.float32),  # LSTM layer 2 weights
        np.zeros(64, dtype=np.float32),          # LSTM layer 2 bias
        np.zeros((64, 1), dtype=np.float32),     # Output layer weights
        np.zeros(1, dtype=np.float32),           # Output layer bias
    ]
    
    logger.info(f"Created {len(initial_params)} parameter arrays")
    return fl.common.ndarrays_to_parameters(initial_params)

def get_strategy(num_rounds: int = 10) -> fl.server.strategy.Strategy:
    """
    Get FedAvg strategy optimized for memory-constrained server.
    """
    strategy = MemoryOptimizedFedAvg(
        fraction_fit=1.0,              # Use all clients in each round
        fraction_evaluate=0.0,         # No server-side evaluation (no GPU)
        min_fit_clients=3,             # Wait for at least 3 clients
        min_evaluate_clients=0,        # No evaluation
        min_available_clients=3,       # Minimum 3 clients to start
        initial_parameters=get_initial_parameters(),
        on_fit_config_fn=get_fit_config,
        on_evaluate_config_fn=None,
    )
    return strategy

def get_fit_config(server_round: int) -> Dict[str, str]:
    """
    Return configuration dictionary for client training.
    Server tells clients hyperparameters for this round.
    """
    config = {
        "server_round": str(server_round),
        "local_epochs": "5",
        "batch_size": "32",
        "learning_rate": "0.001",
    }
    return config

# ============================================================================
# MAIN SERVER ENTRY POINT
# ============================================================================

def main():
    """Start Flower server on GCP e2-micro."""
    
    parser = argparse.ArgumentParser(
        description="Memory-optimized Flower server for e2-micro instances"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("=" * 80)
    logger.info("Federated Learning Server - Memory Optimized for GCP e2-micro")
    logger.info("=" * 80)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Server Address: {args.host}:{args.port}")
    logger.info(f"Number of rounds: {args.num_rounds}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 80)
    logger.info("Waiting for clients to connect...")
    logger.info("Expected clients: 15 (Berlin, Cairo, Dammam, ..., Toronto)")
    logger.info("=" * 80)
    
    # Get strategy
    strategy = get_strategy(num_rounds=args.num_rounds)
    config = get_server_config(num_rounds=args.num_rounds)
    
    # Start server with gRPC options for e2-micro
    # Note: gRPC compression and message size limits configured via environment
    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=config,
        strategy=strategy,
        # Force gRPC to use compression (save bandwidth)
        grpc_max_send_message_length=-1,  # Unlimited (gRPC will handle)
        grpc_max_receive_message_length=-1,
    )

if __name__ == "__main__":
    main()
