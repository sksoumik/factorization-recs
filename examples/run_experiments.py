"""Experiment runner module for testing different configurations."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from main import RecommenderPipeline
from utils.config import ConfigManager


def create_experiment_configs() -> List[Dict[str, Any]]:
    """
    Create different configurations for experiments.

    Returns:
        List[Dict[str, Any]]: List of configuration dictionaries
    """
    experiments = []  # pylint: disable=redefined-outer-name

    # Experiment 1: Default configuration
    config_manager = ConfigManager()
    experiments.append(
        {"name": "default", "config": config_manager.get_config().dict()}
    )

    # Experiment 2: Higher learning rate
    config = config_manager.get_config()
    config.model.learning_rate = 0.1
    experiments.append({"name": "high_lr", "config": config.dict()})

    # Experiment 3: More components
    config = config_manager.get_config()
    config.model.no_components = 128
    experiments.append({"name": "more_components", "config": config.dict()})

    # Experiment 4: Different loss function
    config = config_manager.get_config()
    config.model.loss = "bpr"
    experiments.append({"name": "bpr_loss", "config": config.dict()})

    return experiments


def run_experiments(
    experiment_dir: Path, experiment_configs: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Run multiple experiments with different configurations.

    Args:
        experiment_dir (Path): Directory to save results
        experiment_configs (List[Dict[str, Any]]): List of experiment
        configurations

    Returns:
        pd.DataFrame: DataFrame containing experiment results
    """
    experiment_results = []

    for exp_config in experiment_configs:
        print(f"\nRunning experiment: {exp_config['name']}")

        # Create output directory for experiment
        exp_dir = experiment_dir / exp_config["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Update config and run pipeline
        config_manager = ConfigManager(exp_dir / "config.yaml")
        config_manager.update_config(exp_config["config"])

        pipeline = RecommenderPipeline(
            config_path=exp_dir / "config.yaml", output_path=exp_dir
        )
        current_results = pipeline.run()

        # Add results to experiment_results list
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

        # Save detailed results
        with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(current_results, f, indent=4)

    return pd.DataFrame(experiment_results)


if __name__ == "__main__":
    # Set up output directory
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)

    # Run experiments
    experiments = create_experiment_configs()
    results_df = run_experiments(output_dir, experiments)

    # Save summary results
    results_df.to_csv(output_dir / "summary_results.csv", index=False)

    # Print summary
    print("\nExperiment Results Summary:")
    for experiment in results_df["experiment"].unique():
        print(f"\n{experiment.upper()}:")
        exp_results = results_df[results_df["experiment"] == experiment]
        for _, row in exp_results.iterrows():
            print(f"{row['metric']}: " f"{row['mean']:.4f} ± {row['std']:.4f}")
