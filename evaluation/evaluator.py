from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

from evaluation.metrics import RecommenderMetrics
from models.base_model import BaseRecommender
from utils.logger import Logger

logger = Logger.get_logger()


class RecommenderEvaluator:
    """
    Evaluator class for recommender systems.
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            metrics (Optional[List[str]]): List of metric names to compute
            k_values (Optional[List[int]]): List of k values for metrics
        """
        self.metrics = metrics or ["precision", "recall", "ndcg", "map"]
        self.k_values = k_values or [5, 10, 20]
        self.metrics_computer = RecommenderMetrics()

    def evaluate_fold(
        self,
        model: BaseRecommender,
        train_matrix: csr_matrix,
        test_matrix: csr_matrix,
        user_features: csr_matrix,
        item_features: csr_matrix,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single fold.

        Args:
            model (BaseRecommender): Trained recommender model
            train_matrix (csr_matrix): Training interaction matrix
            test_matrix (csr_matrix): Testing interaction matrix
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        results: Dict[str, float] = {}

        for user_id in range(test_matrix.shape[0]):
            true_items = test_matrix[user_id].indices
            if len(true_items) == 0:
                continue

            # Get recommendations
            pred_items, _ = model.recommend(
                user_id=user_id,
                user_features=user_features,
                item_features=item_features,
                n_items=max(self.k_values),
                exclude_seen=True,
                seen_items=train_matrix[user_id].indices,
            )

            # Calculate metrics
            for k in self.k_values:
                if "precision" in self.metrics:
                    key = f"precision@{k}"
                    score = self.metrics_computer.precision_at_k(
                        true_items, pred_items, k
                    )
                    results[key] = results.get(key, 0.0) + score

                if "recall" in self.metrics:
                    key = f"recall@{k}"
                    score = self.metrics_computer.recall_at_k(
                        true_items, pred_items, k
                    )
                    results[key] = results.get(key, 0.0) + score

                if "ndcg" in self.metrics:
                    key = f"ndcg@{k}"
                    score = self.metrics_computer.ndcg_at_k(
                        true_items, pred_items, k
                    )
                    results[key] = results.get(key, 0.0) + score

                if "map" in self.metrics:
                    key = f"map@{k}"
                    score = self.metrics_computer.map_at_k(
                        true_items, pred_items, k
                    )
                    results[key] = results.get(key, 0.0) + score

        # Average metrics
        n_users = test_matrix.shape[0]
        for key in results:
            results[key] /= n_users

        return results

    def cross_validate(
        self,
        model: BaseRecommender,
        interaction_matrix: csr_matrix,
        user_features: csr_matrix,
        item_features: csr_matrix,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform cross-validation evaluation.

        Args:
            model (BaseRecommender): Recommender model to evaluate
            interaction_matrix (csr_matrix): Full interaction matrix
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix
            n_folds (int): Number of cross-validation folds
            random_state (int): Random seed for reproducibility

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]:
                - Mean metrics across folds
                - Standard deviation of metrics across folds
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        all_results: List[Dict[str, float]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            kf.split(range(interaction_matrix.shape[0]))
        ):
            logger.info(f"Evaluating fold {fold_idx + 1}/{n_folds}")

            # Split data
            train_matrix = interaction_matrix[train_idx]
            test_matrix = interaction_matrix[test_idx]

            # Train model
            model.fit(
                interaction_matrix=train_matrix,
                user_features=user_features,
                item_features=item_features,
            )

            # Evaluate
            fold_results = self.evaluate_fold(
                model=model,
                train_matrix=train_matrix,
                test_matrix=test_matrix,
                user_features=user_features,
                item_features=item_features,
            )
            all_results.append(fold_results)

        # Calculate mean and std of metrics
        mean_results = {}
        std_results = {}

        for metric in all_results[0].keys():
            values = [r[metric] for r in all_results]
            mean_results[metric] = float(np.mean(values))
            std_results[metric] = float(np.std(values))

        return mean_results, std_results
