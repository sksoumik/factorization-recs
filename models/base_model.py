"""
Base model interface module for recommender systems.

This module defines the abstract base class that all recommender models must
implement.
It establishes a consistent interface for:
1. Model training and prediction
2. Recommendation generation
3. Parameter management

The interface ensures that all recommender implementations provide:
- Standard training methods with feature support
- Flexible prediction capabilities
- Configurable recommendation generation
- Parameter access and management
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix


class BaseRecommender(ABC):
    """
    Abstract base class defining the recommender model interface.

    This class establishes the contract that all recommender implementations
    must follow, ensuring consistency across different models. It defines:

    1. Training Interface:
       - Feature-based training support
       - Flexible parameter handling
       - Progress monitoring capabilities

    2. Prediction Capabilities:
       - Batch prediction for user-item pairs
       - Feature-based score generation
       - Efficient matrix operations

    3. Recommendation Generation:
       - Top-N item recommendations
       - Exclusion of seen items
       - Score-based ranking

    4. Parameter Management:
       - Model parameter access
       - Configuration persistence
       - State management
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
        Train the recommender model on provided data.

        Processes the interaction data and features to learn the model parameters:
        - Analyzes user-item interaction patterns
        - Incorporates user and item features
        - Optimizes model parameters

        Args:
            interaction_matrix (csr_matrix): Sparse matrix of user-item interactions
                Shape: (n_users, n_items)
                Values: Interaction indicators or weights
            user_features (csr_matrix): Sparse matrix of user features
                Shape: (n_users, n_user_features)
            item_features (csr_matrix): Sparse matrix of item features
                Shape: (n_items, n_item_features)
            **kwargs: Additional training parameters
                May include:
                - learning_rate: Training learning rate
                - n_epochs: Number of training epochs
                - batch_size: Training batch size
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
        Generate prediction scores for user-item pairs.

        Computes interaction likelihood scores for specified user-item pairs:
        - Uses learned model parameters
        - Incorporates user and item features
        - Generates normalized prediction scores

        Args:
            user_ids (np.ndarray): Array of user indices to generate
                predictions for
            item_ids (np.ndarray): Array of item indices to generate
                predictions for
            user_features (csr_matrix): User feature matrix
            item_features (csr_matrix): Item feature matrix

        Returns:
            np.ndarray: Array of prediction scores
                Shape: (len(user_ids),)
                Values: Higher values indicate stronger recommendations
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
        Generate personalized recommendations for a user.

        Creates a ranked list of recommended items for the specified user:
        - Computes scores for all candidate items
        - Ranks items by prediction score
        - Optionally excludes previously seen items
        - Returns top-N recommendations

        Args:
            user_id (int): Index of the user to generate recommendations for
            user_features (csr_matrix): User feature matrix
            item_features (csr_matrix): Item feature matrix
            n_items (int): Number of recommendations to generate
            exclude_seen (bool): Whether to exclude items the user has
                interacted with

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing:
                1. Indices of recommended items (shape: (n_items,))
                2. Prediction scores for recommended items (shape: (n_items,))
        """

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve the model's current parameters.

        Provides access to the model's configuration and learned parameters:
        - Model hyperparameters
        - Training configuration
        - Current model state

        Returns:
            Dict[str, Any]: Dictionary containing:
                - Model hyperparameters
                - Training configuration
                - Additional model-specific parameters
        """
