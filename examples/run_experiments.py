"""
Experiment runner module for testing different configurations.

This module provides functionality to:
1. Run multiple experiments with different configurations
2. Compare model performance across configurations
3. Save and analyze results
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

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


def run_experiments(
    experiment_dir: Path, experiment_configs: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Run multiple experiments with different configurations.

    Args:
        experiment_dir (Path): Directory to save results
        experiment_configs (List[Dict[str, Any]]): List of experiment configs

    Returns:
        pd.DataFrame: DataFrame containing experiment results
    """
    experiment_results = []

    for exp_config in experiment_configs:
        LOGGER.info(f"\nRunning experiment: {exp_config['name']}")
        exp_dir = experiment_dir / exp_config["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        save_experiment_config(exp_config["config"], exp_dir)

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
