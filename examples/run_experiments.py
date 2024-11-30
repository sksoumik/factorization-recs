"""
Experiment runner module for recommender system evaluation.

This module provides a comprehensive framework for running and analyzing
recommender system experiments. It supports:

1. Configuration Management:
   - Loading and validating experiment configurations
   - Creating variations for parameter tuning
   - Saving experiment settings for reproducibility

2. Experiment Execution:
   - Running multiple model configurations
   - Cross-validation evaluation
   - Parallel experiment execution

3. Results Analysis:
   - Performance metric collection
   - Statistical analysis of results
   - Automated result visualization

4. Output Management:
   - Structured result storage
   - Detailed experiment logging
   - Summary report generation
   - Synthetic data persistence
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from data.data_generator import DataGenerator
from main import RecommenderPipeline
from utils.logger import Logger

LOGGER = Logger.get_logger()


def load_base_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """
    Load base configuration from config.yaml file

    Args:
        config_path (Path): Path to the config file

    Returns:
        Dict[str, Any]: Base configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path} ")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_experiment_configs() -> List[Dict[str, Any]]:
    """
    Create different configurations for experiments.

    Returns:
        List[Dict[str, Any]]: List of configuration dictionaries
    """
    experiments = []
    base_config = load_base_config()

    # Experiment 1: Default configuration from config.yaml
    experiments.append({"name": "default", "config": base_config})

    # Experiment 2: Higher learning rate
    high_lr_config = base_config.copy()
    high_lr_config["model"]["learning_rate"] = 0.1
    experiments.append({"name": "high_lr", "config": high_lr_config})

    # Experiment 3: More components
    more_components_config = base_config.copy()
    more_components_config["model"]["no_components"] = 128
    experiments.append({"name": "more_components", "config": more_components_config})

    # Experiment 4: Different loss function
    bpr_loss_config = base_config.copy()
    bpr_loss_config["model"]["loss"] = "bpr"
    experiments.append({"name": "bpr_loss", "config": bpr_loss_config})

    return experiments


def save_experiment_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save experiment configuration to file.

    Args:
        config (Dict[str, Any]): Configuration dictionary
        output_path (Path): Path to save the config
    """
    with open(output_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_synthetic_data(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save synthetic datasets to CSV files.

    Persists generated data with:
    - User features and demographics
    - Item characteristics
    - User-item interactions

    Args:
        users_df (pd.DataFrame): User features dataset
        items_df (pd.DataFrame): Item features dataset
        interactions_df (pd.DataFrame): User-item interactions dataset
        output_dir (Path): Directory to save the CSV files
    """
    data_dir = output_dir / "synthetic_data"
    data_dir.mkdir(exist_ok=True)

    users_df.to_csv(data_dir / "users.csv", index=False)
    items_df.to_csv(data_dir / "items.csv", index=False)
    interactions_df.to_csv(data_dir / "interactions.csv", index=False)
    LOGGER.info(f"Synthetic data saved to {data_dir}")


def generate_synthetic_data(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data for experiments.

    Creates realistic synthetic data including:
    - User profiles and demographics
    - Item features and attributes
    - User-item interaction patterns

    Args:
        config (Dict[str, Any]): Configuration containing data generation parameters

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three dataframes containing:
            - users_df: User features and demographics
            - items_df: Item characteristics
            - interactions_df: User-item interactions
    """
    data_config = config.get("data", {})
    data_generator = DataGenerator(
        n_users=data_config.get("n_users", 1000),
        n_items=data_config.get("n_items", 5000),
        n_interactions=data_config.get("n_interactions", 10000),
        random_seed=data_config.get("random_seed", 42),
    )
    return data_generator.generate_dataset()


def run_experiments(
    experiment_dir: Path, experiment_configs: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Execute multiple experiments with different configurations.

    Performs systematic evaluation:
    1. Configuration Setup:
       - Creates experiment directories
       - Saves configurations
       - Initializes logging

    2. Data Generation:
       - Creates synthetic datasets
       - Saves data to CSV files
       - Ensures reproducibility

    3. Model Evaluation:
       - Trains models with configurations
       - Performs cross-validation
       - Collects performance metrics

    4. Results Collection:
       - Aggregates metrics across experiments
       - Computes statistical measures
       - Saves detailed results

    Args:
        experiment_dir (Path): Base directory for experiment outputs
            Creates subdirectories for each experiment
        experiment_configs (List[Dict[str, Any]]): List of configurations
            Each config contains experiment name and parameters

    Returns:
        pd.DataFrame: Results summary with columns:
            - experiment: Configuration name
            - metric: Performance metric name
            - mean: Average metric value
            - std: Metric standard deviation
    """
    experiment_results = []

    for exp_config in experiment_configs:
        LOGGER.info(f"\nRunning experiment: {exp_config['name']}")
        exp_dir = experiment_dir / exp_config["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        save_experiment_config(exp_config["config"], exp_dir)

        # Generate and save synthetic data
        LOGGER.info("Generating synthetic data...")
        users_df, items_df, interactions_df = generate_synthetic_data(
            exp_config["config"]
        )
        save_synthetic_data(users_df, items_df, interactions_df, exp_dir)

        # create and run pipeline with this configuration
        pipeline = RecommenderPipeline(
            config_path=exp_dir / "config.yaml", output_path=exp_dir
        )
        current_results = pipeline.run()

        # collect results
        for metric, mean_value in current_results["mean_metrics"].items():
            std_value = current_results["std_metrics"][metric]
            experiment_results.append(
                {
                    "experiment": exp_config["name"],
                    "metric": metric,
                    "mean": mean_value,
                    "std": std_value,
                }
            )

        # save results
        with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(current_results, f, indent=4)

    return pd.DataFrame(experiment_results)


def print_experiment_summary(results_df: pd.DataFrame) -> None:
    """
    Print a summary of experiment results.

    Args:
        results_df (pd.DataFrame): DataFrame containing results
    """
    print("\nExperiment Results Summary:")
    print("=" * 80)

    for experiment in results_df["experiment"].unique():
        print(f"\n{experiment.upper()}:")
        print("-" * 40)
        exp_results = results_df[results_df["experiment"] == experiment]
        for _, row in exp_results.iterrows():
            print(f"{row['metric']:<15}: " f"{row['mean']:.4f} ± {row['std']:.4f}")


if __name__ == "__main__":
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)

    try:
        experiments = create_experiment_configs()
        results_df = run_experiments(output_dir, experiments)

        results_df.to_csv(output_dir / "summary_results.csv", index=False)
        print_experiment_summary(results_df)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except (ValueError, yaml.YAMLError, pd.errors.EmptyDataError) as e:
        print(f"Configuration or data error: {e}")
    except IOError as e:
        print(f"I/O error occurred: {e}")
