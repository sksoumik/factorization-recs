import logging
from pathlib import Path
from typing import Any, Dict, Optional

from data.data_generator import DataGenerator
from data.data_processor import DataProcessor
from evaluation.evaluator import RecommenderEvaluator
from models.model_factory import ModelFactory
from utils.config import ConfigManager
from utils.logger import Logger

logger = Logger.get_logger()


class RecommenderPipeline:
    """
    Main pipeline for recommender system training and evaluation.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            config_path (Optional[Path]): Path to configuration file
            output_path (Optional[Path]): Path to output directory
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.output_path = output_path or Path("output")
        self.output_path.mkdir(exist_ok=True)

        self.data_generator = DataGenerator(
            n_users=self.config.data.n_users,
            n_items=self.config.data.n_items,
            n_interactions=self.config.data.n_interactions,
            random_seed=self.config.data.random_seed,
        )
        self.data_processor = DataProcessor()
        self.evaluator = RecommenderEvaluator(
            metrics=self.config.evaluation.metrics,
            k_values=self.config.evaluation.k_values,
        )

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation results
        """
        try:
            # Generate data
            logger.info("Generating synthetic data...")
            users_df, items_df, interactions_df = (
                self.data_generator.generate_dataset()
            )

            # Process data
            logger.info("Processing data...")
            (interaction_matrix, user_features, item_features, mappings) = (
                self.data_processor.process_data(
                    users_df=users_df,
                    items_df=items_df,
                    interactions_df=interactions_df,
                )
            )

            # Create and evaluate model
            logger.info("Creating model...")
            model = ModelFactory.create_model(
                model_type="lightfm", model_params=self.config.model.dict()
            )

            # Perform cross-validation
            logger.info("Performing cross-validation...")
            mean_results, std_results = self.evaluator.cross_validate(
                model=model,
                interaction_matrix=interaction_matrix,
                user_features=user_features,
                item_features=item_features,
                n_folds=self.config.evaluation.n_folds,
                random_state=self.config.data.random_seed,
            )

            # Save results
            results = {
                "mean_metrics": mean_results,
                "std_metrics": std_results,
                "config": self.config.dict(),
            }
            self._save_results(results)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save results to output directory.

        Args:
            results (Dict[str, Any]): Results to save
        """
        import json

        output_file = self.output_path / "results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run pipeline
    pipeline = RecommenderPipeline()
    results = pipeline.run()

    # Print summary
    print("\nEvaluation Results:")
    for metric, value in results["mean_metrics"].items():
        std = results["std_metrics"][metric]
        print(f"{metric}: {value:.4f} ± {std:.4f}")
