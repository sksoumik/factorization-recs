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
    experiments = []

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
    output_dir: Path, experiments: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Run multiple experiments with different configurations.

    Args:
        output_dir (Path): Directory to save results
        experiments (List[Dict[str, Any]]): List of experiment configurations

    Returns:
        pd.DataFrame: DataFrame containing experiment results
    """
    results = []

    for experiment in experiments:
        print(f"\nRunning experiment: {experiment['name']}")

        # Create output directory for experiment
        exp_dir = output_dir / experiment["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Update config and run pipeline
        config_manager = ConfigManager(exp_dir / "config.yaml")
        config_manager.update_config(experiment["config"])

        pipeline = RecommenderPipeline(
            config_path=exp_dir / "config.yaml", output_path=exp_dir
        )
        exp_results = pipeline.run()

        # Save detailed results
        with open(exp_dir / "results.json", "w") as f:
            json.dump(exp_results, f, indent=4)

        # Collect summary results
        for metric, value in exp_results["mean_metrics"].items():
            std = exp_results["std_metrics"][metric]
            results.append(
                {
                    "experiment": experiment["name"],
                    "metric": metric,
                    "mean": value,
                    "std": std,
                }
            )

    return pd.DataFrame(results)


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
