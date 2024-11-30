"""
Main module for the recommender system pipeline.

The pipeline implements an end-to-end workflow for:
1. Generating synthetic data that mimics e-commerce interactions
2. Processing and preparing data for the recommender model
3. Training and evaluating the model using various metrics
4. Saving and reporting results
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from data.data_generator import DataGenerator
from data.data_processor import DataProcessor
from evaluation.evaluator import RecommenderEvaluator
from models.model_factory import ModelFactory
from utils.config import ConfigManager
from utils.logger import Logger

LOGGER = Logger.get_logger()


class RecommenderPipeline:
    """
    Main pipeline for recommender system training and evaluation.

    This class orchestrates the entire recommendation system workflow:
    1. Data Generation:
       - Creates synthetic user-item interaction data
       - Generates user and item features
       - Simulates realistic e-commerce behavior

    2. Data Processing:
       - Transforms raw data into format suitable for LightFM
       - Creates user and item feature matrices
       - Handles train-test splitting

    3. Model Training:
       - Initializes the recommender model with configured parameters
       - Trains the model on processed data
       - Handles cross-validation

    4. Evaluation:
       - Computes various recommendation metrics
       - Generates performance reports
       - Saves results for analysis

    The pipeline is configurable through a YAML configuration file,
    allowing for easy experimentation with different parameters
    and settings.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:

        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.output_path = output_path or Path("output")
        self.output_path.mkdir(exist_ok=True)

        # initialize data generation with configured parameters
        self.data_generator = DataGenerator(
            n_users=self.config.data.n_users,
            n_items=self.config.data.n_items,
            n_interactions=self.config.data.n_interactions,
            random_seed=self.config.data.random_seed,
        )

        # initialize data processing and evaluation components
        self.data_processor = DataProcessor()
        self.evaluator = RecommenderEvaluator(
            metrics=self.config.evaluation.metrics,
            k_values=self.config.evaluation.k_values,
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete recommendation pipeline.

        This method runs the entire recommendation workflow:
        1. Generates synthetic interaction data
        2. Processes the data into required format
        3. Creates and trains the recommender model
        4. Evaluates model performance
        5. Saves results

        Returns:
            Dict[str, Any]: Results dictionary containing:
                - mean_metrics: Average performance metrics
                - std_metrics: Standard deviation of metrics
                - config: Configuration used for the run
        """
        try:
            # generate synthetic dataset
            LOGGER.info("Generating synthetic data...")
            users_df, items_df, interactions_df = self.data_generator.generate_dataset()

            # process data for model training
            LOGGER.info("Processing data...")
            (interaction_matrix, user_features, item_features, _) = (
                self.data_processor.process_data(
                    users_df=users_df,
                    items_df=items_df,
                    interactions_df=interactions_df,
                )
            )

            # initialize and evaluate model
            LOGGER.info("Creating model...")
            model = ModelFactory.create_model(
                model_type="lightfm",
                model_params=self.config.model.model_dump(),
            )

            # perform cross-validation evaluation
            LOGGER.info("Performing cross-validation...")
            mean_results, std_results = self.evaluator.cross_validate(
                model=model,
                interaction_matrix=interaction_matrix,
                user_features=user_features,
                item_features=item_features,
                n_folds=self.config.evaluation.n_folds,
                random_state=self.config.data.random_seed,
                num_epochs=self.config.training.num_epochs,
                num_threads=self.config.training.num_threads,
            )

            # compile and save results
            results = {
                "mean_metrics": mean_results,
                "std_metrics": std_results,
                "config": self.config.dict(),
            }
            self._save_results(results)

            return results

        except Exception as e:
            LOGGER.error(f"Pipeline failed: {str(e)}")
            raise

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Writes the evaluation results and configuration to a JSON file
        in the specified output directory. The results include model
        performance metrics and the configuration used for the run.

        Args:
            results (Dict[str, Any]): Dictionary containing:
                - Model performance metrics
                - Configuration parameters
                - Additional metadata
        """
        output_file = self.output_path / "results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        LOGGER.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # create and run the pipeline
    pipeline = RecommenderPipeline()
    results = pipeline.run()

    # display evaluation results
    print("\nEvaluation Results:")
    for metric, value in results["mean_metrics"].items():
        std = results["std_metrics"][metric]
        print(f"{metric}: {value:.4f} ± {std:.4f}")
