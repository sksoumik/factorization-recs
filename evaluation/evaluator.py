"""
Evaluator module for recommender system models.

This module provides comprehensive evaluation capabilities for recommender
systems:
1. Cross-validation evaluation with configurable folds
2. Multiple evaluation metrics at different K values
3. Performance statistics across evaluation folds
4. Detailed logging of evaluation progress

The evaluator supports various recommendation metrics including:
- Precision@K: Accuracy of recommendations
- Recall@K: Coverage of relevant items
- NDCG@K: Quality of ranking
- MAP@K: Overall recommendation quality
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

from evaluation.metrics import RecommenderMetrics
from models.base_model import BaseRecommender
from utils.logger import Logger

LOGGER = Logger.get_logger()


class RecommenderEvaluator:
    """
    Comprehensive evaluation system for recommender models.

    This class provides functionality for:
    1. Model Evaluation:
       - Single-fold evaluation
       - Cross-validation
       - Multiple metric computation

    2. Performance Metrics:
       - Configurable set of metrics
       - Multiple K values for top-K metrics
       - Statistical aggregation

    3. Result Analysis:
       - Mean performance across folds
       - Standard deviation of metrics
       - Detailed performance logging
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> None:

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
        Evaluate model performance on a single data fold.

        Computes all specified metrics for each user in the test set:
        1. Generates recommendations for each user
        2. Compares with ground truth interactions
        3. Computes metrics at different K values
        4. Aggregates results across users

        Args:
            model (BaseRecommender): Trained recommender model
            train_matrix (csr_matrix): Training interactions
            test_matrix (csr_matrix): Test interactions for evaluation
            user_features (csr_matrix): User feature matrix
            item_features (csr_matrix): Item feature matrix

        Returns:
            Dict[str, float]: Dictionary of computed metrics:
                - Keys: metric_name@k (e.g., "precision@10")
                - Values: Metric scores averaged across users
        """
        results: Dict[str, float] = {}

        for user_id in range(test_matrix.shape[0]):
            true_items = test_matrix[user_id].indices
            if len(true_items) == 0:
                continue

            # get recommendations
            pred_items, _ = model.recommend(
                user_id=user_id,
                user_features=user_features,
                item_features=item_features,
                n_items=max(self.k_values),
                exclude_seen=True,
                seen_items=train_matrix[user_id].indices,
            )

            # calculate metrics
            for k in self.k_values:
                if "precision" in self.metrics:
                    key = f"precision@{k}"
                    score = self.metrics_computer.precision_at_k(
                        true_items, pred_items, k
                    )
                    results[key] = results.get(key, 0.0) + score

                if "recall" in self.metrics:
                    key = f"recall@{k}"
                    score = self.metrics_computer.recall_at_k(true_items, pred_items, k)
                    results[key] = results.get(key, 0.0) + score

                if "ndcg" in self.metrics:
                    key = f"ndcg@{k}"
                    score = self.metrics_computer.ndcg_at_k(true_items, pred_items, k)
                    results[key] = results.get(key, 0.0) + score

                if "map" in self.metrics:
                    key = f"map@{k}"
                    score = self.metrics_computer.map_at_k(true_items, pred_items, k)
                    results[key] = results.get(key, 0.0) + score

        # average metrics
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
        num_epochs: int = 10,
        num_threads: int = 4,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform cross-validation evaluation of the model.

        Implements a complete cross-validation workflow:
        1. Splits data into n_folds
        2. For each fold:
           - Separates training and test data
           - Trains the model on training data
           - Evaluates on test data
        3. Aggregates results across folds

        Args:
            model (BaseRecommender): Model to evaluate
            interaction_matrix (csr_matrix): Complete interaction matrix
            user_features (csr_matrix): User features
            item_features (csr_matrix): Item features
            n_folds (int): Number of cross-validation folds
            random_state (int): Random seed for reproducibility
            num_epochs (int): Number of training epochs
            num_threads (int): Number of threads for training

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Two dictionaries:
                1. Mean metrics across folds
                2. Standard deviation of metrics across folds
                Each dictionary has keys like "metric@k" (e.g., "precision@10")
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        all_results: List[Dict[str, float]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            kf.split(range(interaction_matrix.shape[0]))
        ):
            LOGGER.info(f"Evaluating fold {fold_idx + 1}/{n_folds}")

            # split data
            train_matrix = interaction_matrix[train_idx]
            test_matrix = interaction_matrix[test_idx]

            # train model
            model.fit(
                interaction_matrix=train_matrix,
                user_features=user_features,
                item_features=item_features,
                num_epochs=num_epochs,
                num_threads=num_threads,
            )

            # evaluate
            fold_results = self.evaluate_fold(
                model=model,
                train_matrix=train_matrix,
                test_matrix=test_matrix,
                user_features=user_features,
                item_features=item_features,
            )
            all_results.append(fold_results)

        # calculate mean and std of metrics
        mean_results = {}
        std_results = {}

        for metric in all_results[0].keys():
            values = [r[metric] for r in all_results]
            mean_results[metric] = float(np.mean(values))
            std_results[metric] = float(np.std(values))

        return mean_results, std_results
