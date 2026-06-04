"""
Flower FL Simulation Orchestrator for multi-city weather forecasting.

This is the main entry point for the federated learning simulation.
Orchestrates the aggregation of local models across 15 city clients using FedAvg.
"""

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

import flwr as fl
from flwr.common import Metrics, Context
from flwr.server.strategy import FedAvg

# Handle both direct execution and package import
try:
    from src.client import WeatherClient
except ImportError:
    from client import WeatherClient

# --- Reproducibility Seed ---
SEED = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- FL Simulation Constants ---
NUM_CLIENTS = 15          # Total number of city clients
NUM_ROUNDS = 10           # Global federation rounds
FRACTION_FIT = 1.0        # Fraction of clients sampled per round (1.0 = all)
MIN_FIT_CLIENTS = 15      # Minimum clients required to start a round
MAX_WORKERS = 2           # Reduced from 4 to 2 to avoid OOM with VS Code running

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global state for logging ---
training_log: List[Dict[str, Any]] = []


def client_fn(context: Context) -> fl.client.Client:
    """
    Client factory function for Flower simulation (new Context API).

    Creates a WeatherClient instance for the given city index from context.

    Args:
        context: Flower Context containing node_config with cid.

    Returns:
        WeatherClient instance wrapped as Flower Client.

    Raises:
        ValueError: If cid is not a valid city index.
    """
    # Extract cid from context node_config (new Flower API)
    cid = str(context.node_config.get("cid", "0"))
    city_index = int(cid)
    if not 0 <= city_index <= NUM_CLIENTS - 1:
        raise ValueError(
            f"Invalid city_index {city_index}. Must be in [0, {NUM_CLIENTS - 1}]"
        )
    return WeatherClient(city_index=city_index).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from all clients using weighted average.

    Args:
        metrics: List of (num_samples, metrics_dict) tuples from each client.

    Returns:
        Aggregated metrics dict.
    """
    if not metrics:
        return {}

    # Extract MSE values and sample counts
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    weighted_mse = sum(
        (num_samples / total_samples) * m.get("mse", 0.0)
        for num_samples, m in metrics
    )

    return {"global_mse": float(weighted_mse)}


class LoggingCallback(fl.common.FitRes):
    """
    Custom callback to log training metrics after each round.

    Logs global MSE to training_log and prints to stdout.
    """

    pass


def fit_config(server_round: int) -> Dict[str, Any]:
    """
    Return training configuration for the current round.

    Args:
        server_round: Current round number (1-indexed).

    Returns:
        Config dict for clients.
    """
    return {"server_round": server_round}


def evaluate_config(server_round: int) -> Dict[str, Any]:
    """
    Return evaluation configuration for the current round.

    Args:
        server_round: Current round number (1-indexed).

    Returns:
        Config dict for clients.
    """
    return {"server_round": server_round}


class MetricsAggregationCallback(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with metrics aggregation and logging.
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate evaluation results from all clients.

        Args:
            server_round: Current round number.
            results: List of (client_proxy, evaluate_res) from each client.
            failures: List of evaluation failures.

        Returns:
            Aggregated loss and metrics dict.
        """
        if not results:
            return None, {}

        # Extract MSE from evaluation results
        metrics_list = []
        for _, eval_res in results:
            if eval_res.metrics:
                mse = eval_res.metrics.get("mse", float("inf"))
                num_samples = eval_res.num_examples
                metrics_list.append((num_samples, {"mse": mse}))

        # Aggregate using weighted average
        aggregated_metrics = weighted_average(metrics_list)
        global_mse = aggregated_metrics.get("global_mse", float("inf"))

        # Log to global training log
        log_entry = {
            "round": server_round,
            "global_mse": float(global_mse),
        }
        training_log.append(log_entry)

        # Print to stdout
        print(
            f"[Round {server_round}/{NUM_ROUNDS}] Global MSE: {global_mse:.6f}"
        )

        return global_mse, aggregated_metrics


def save_training_log(log_path: str = "logs/training_log.csv") -> None:
    """
    Save training log to CSV file.

    Args:
        log_path: Path to save training log CSV.
    """
    if not training_log:
        print("⚠️ No training log to save")
        return

    # Ensure logs directory exists
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and save
    df = pd.DataFrame(training_log)
    df.to_csv(log_path, index=False)
    print(f"\n✅ Training log saved to {log_path}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Rows: {len(df)}")


def main() -> None:
    """
    Main entry point for FL simulation.

    Runs federated learning simulation with 15 city clients for NUM_ROUNDS.
    """
    # Set reproducibility seed
    set_seed(SEED)

    # Print environment info
    print("=" * 80)
    print("FEDERATED LEARNING WEATHER FORECASTING SIMULATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Fraction fit: {FRACTION_FIT}")
    print(f"  - Max workers: {MAX_WORKERS}")
    print(f"  - GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 80)
    print()

    # Create FedAvg strategy with custom callback
    strategy = MetricsAggregationStrategy(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    # Start FL simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_gpus": 1.0 / MAX_WORKERS, "num_cpus": 2},
    )

    # Save training log
    save_training_log("logs/training_log.csv")

    # Print final summary
    print()
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    if training_log:
        print(f"Final Global MSE: {training_log[-1]['global_mse']:.6f}")
        print(f"Initial Global MSE: {training_log[0]['global_mse']:.6f}")
        improvement = training_log[0]['global_mse'] - training_log[-1]['global_mse']
        print(f"Improvement: {improvement:.6f}")
    print("=" * 80)


class MetricsAggregationStrategy(FedAvg):
    """
    Custom FedAvg strategy with metrics aggregation and logging.
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate evaluation results from all clients.

        Args:
            server_round: Current round number.
            results: List of (client_proxy, evaluate_res) from each client.
            failures: List of evaluation failures.

        Returns:
            Aggregated loss and metrics dict.
        """
        if not results:
            return None, {}

        # Extract MSE from evaluation results
        metrics_list = []
        for _, eval_res in results:
            if eval_res.metrics:
                mse = eval_res.metrics.get("mse", float("inf"))
                num_samples = eval_res.num_examples
                metrics_list.append((num_samples, {"mse": mse}))

        # Aggregate using weighted average
        aggregated_metrics = weighted_average(metrics_list)
        global_mse = aggregated_metrics.get("global_mse", float("inf"))

        # Log to global training log
        log_entry = {
            "round": server_round,
            "global_mse": float(global_mse),
        }
        training_log.append(log_entry)

        # Print to stdout
        print(
            f"[Round {server_round}/{NUM_ROUNDS}] Global MSE: {global_mse:.6f}"
        )

        return global_mse, aggregated_metrics


if __name__ == "__main__":
    main()
