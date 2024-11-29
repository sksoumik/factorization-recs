import numpy as np

from utils.logger import Logger

logger = Logger.get_logger()


class RecommenderMetrics:
    """
    Implementation of common evaluation metrics for recommender systems.
    """

    @staticmethod
    def precision_at_k(
        true_items: np.ndarray, pred_items: np.ndarray, k: int
    ) -> float:
        """
        Calculate Precision@K metric.

        Args:
            true_items (np.ndarray): Array of true relevant items
            pred_items (np.ndarray): Array of predicted items
            k (int): Number of items to consider

        Returns:
            float: Precision@K score
        """
        if len(pred_items) > k:
            pred_items = pred_items[:k]

        common_items = np.intersect1d(true_items, pred_items)
        return len(common_items) / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(
        true_items: np.ndarray, pred_items: np.ndarray, k: int
    ) -> float:
        """
        Calculate Recall@K metric.

        Args:
            true_items (np.ndarray): Array of true relevant items
            pred_items (np.ndarray): Array of predicted items
            k (int): Number of items to consider

        Returns:
            float: Recall@K score
        """
        if len(pred_items) > k:
            pred_items = pred_items[:k]

        common_items = np.intersect1d(true_items, pred_items)
        return (
            len(common_items) / len(true_items) if len(true_items) > 0 else 0.0
        )

    @staticmethod
    def ndcg_at_k(
        true_items: np.ndarray, pred_items: np.ndarray, k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Args:
            true_items (np.ndarray): Array of true relevant items
            pred_items (np.ndarray): Array of predicted items
            k (int): Number of items to consider

        Returns:
            float: NDCG@K score
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
    def map_at_k(
        true_items: np.ndarray, pred_items: np.ndarray, k: int
    ) -> float:
        """
        Calculate Mean Average Precision (MAP@K).

        Args:
            true_items (np.ndarray): Array of true relevant items
            pred_items (np.ndarray): Array of predicted items
            k (int): Number of items to consider

        Returns:
            float: MAP@K score
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
