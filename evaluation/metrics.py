"""Evaluation metrics module for recommender system."""

import numpy as np

from utils.logger import Logger

LOGGER = Logger.get_logger()


class RecommenderMetrics:
    """
    Implementation of standard evaluation metrics for recommender systems.

    This class provides static methods for computing various recommendation
    quality metrics. Each metric evaluates different aspects:

    1. Precision@K:
       - Measures recommendation accuracy
       - Focuses on proportion of relevant items
       - Higher values indicate better accuracy

    2. Recall@K:
       - Measures recommendation coverage
       - Focuses on finding relevant items
       - Higher values indicate better coverage

    3. NDCG@K:
       - Measures ranking quality
       - Considers position of relevant items
       - Higher values indicate better ranking

    4. MAP@K:
       - Measures overall recommendation quality
       - Combines precision and ranking
       - Higher values indicate better overall performance
    """

    @staticmethod
    def precision_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
        """
        Calculate Precision@K metric.

        Measures the proportion of recommended items that are relevant:
        precision@k = (# of recommended items @k that are relevant) / k

        Args:
            true_items (np.ndarray): Ground truth relevant items
            pred_items (np.ndarray): Predicted/recommended items
            k (int): Number of top items to consider

        Returns:
            float: Precision@K score in range [0.0, 1.0]
                - 1.0: All recommended items are relevant
                - 0.0: No recommended items are relevant
        """
        if len(pred_items) > k:
            pred_items = pred_items[:k]

        common_items = np.intersect1d(true_items, pred_items)
        return len(common_items) / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
        """
        Calculate Recall@K metric.

        Measures the proportion of relevant items that are recommended:
        recall@k = (# of recommended items @k that are relevant) / (total # of relevant
                    items)

        Args:
            true_items (np.ndarray): Ground truth relevant items
            pred_items (np.ndarray): Predicted/recommended items
            k (int): Number of top items to consider

        Returns:
            float: Recall@K score in range [0.0, 1.0]
                - 1.0: All relevant items are recommended
                - 0.0: No relevant items are recommended
        """
        if len(pred_items) > k:
            pred_items = pred_items[:k]

        common_items = np.intersect1d(true_items, pred_items)
        return len(common_items) / len(true_items) if len(true_items) > 0 else 0.0

    @staticmethod
    def ndcg_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Measures the quality of ranking, considering item positions:
        - Relevant items ranked higher contribute more
        - Normalized by the ideal DCG (perfect ranking)

        Args:
            true_items (np.ndarray): Ground truth relevant items
            pred_items (np.ndarray): Predicted/recommended items
            k (int): Number of top items to consider

        Returns:
            float: NDCG@K score in range [0.0, 1.0]
                - 1.0: Perfect ranking of relevant items
                - 0.0: No relevant items in recommendations
        """
        if k <= 0:
            return 0.0

        pred_items = pred_items[:k]
        dcg = 0.0
        idcg = 0.0

        for i, item in enumerate(pred_items):
            if item in true_items:
                dcg += 1.0 / np.log2(i + 2)

        for i in range(min(len(true_items), k)):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(true_items: np.ndarray, pred_items: np.ndarray, k: int) -> float:
        """
        Calculate Mean Average Precision (MAP@K).

        Measures the overall quality of recommendations:
        - Combines precision with ranking quality
        - Averages precision values at each relevant item position
        - Penalizes relevant items appearing later in the list

        Args:
            true_items (np.ndarray): Ground truth relevant items
            pred_items (np.ndarray): Predicted/recommended items
            k (int): Number of top items to consider

        Returns:
            float: MAP@K score in range [0.0, 1.0]
                - 1.0: Perfect recommendations and ranking
                - 0.0: No relevant items in recommendations
        """
        if k <= 0:
            return 0.0

        pred_items = pred_items[:k]
        score = 0.0
        num_hits = 0.0

        for i, item in enumerate(pred_items):
            if item in true_items:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(true_items), k) if len(true_items) > 0 else 0.0
