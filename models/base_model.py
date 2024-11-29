from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix


class BaseRecommender(ABC):
    """
    Abstract base class for recommender models.
    """

    @abstractmethod
    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_features: csr_matrix,
        item_features: csr_matrix,
        **kwargs: Any,
    ) -> None:
        """
        Train the recommender model.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix
            **kwargs: Additional arguments for the model
        """

    @abstractmethod
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_features: csr_matrix,
        item_features: csr_matrix,
    ) -> np.ndarray:
        """
        Generate predictions for user-item pairs.

        Args:
            user_ids (np.ndarray): Array of user indices
            item_ids (np.ndarray): Array of item indices
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix

        Returns:
            np.ndarray: Predicted scores for each user-item pair
        """

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        user_features: csr_matrix,
        item_features: csr_matrix,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate recommendations for a user.

        Args:
            user_id (int): User index
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix
            n_items (int): Number of recommendations to generate
            exclude_seen (bool): Whether to exclude already seen items

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of recommended item indices
                - Array of prediction scores
        """

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
