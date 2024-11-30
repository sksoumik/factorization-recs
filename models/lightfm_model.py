"""
LightFM model implementation module.

This module provides a concrete implementation of the BaseRecommender interface
using the LightFM recommendation algorithm. LightFM is a hybrid recommendation
model that combines the following approaches:
1. Collaborative filtering through matrix factorization
2. Content-based filtering using user and item features
3. Flexible loss functions for different recommendation scenarios

Key features of this implementation:
- Support for various training objectives (WARP, BPR, etc.)
- Integration of side features for both users and items
- Efficient training through negative sampling
- Configurable model complexity and learning parameters
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix

from models.base_model import BaseRecommender
from utils.logger import Logger

LOGGER = Logger.get_logger()


class LightFMRecommender(BaseRecommender):
    """
    LightFM-based recommender model implementation.

    This class implements the BaseRecommender interface using LightFM, providing:
    1. Model Configuration:
       - Learning rate and training dynamics
       - Loss function selection
       - Embedding dimensionality
       - Negative sampling strategy

    2. Training Features:
       - Support for user and item features
       - Multiple loss functions for different scenarios
       - Efficient negative sampling
       - Multi-threaded training

    3. Prediction Capabilities:
       - Score prediction for user-item pairs
       - Top-N recommendations
       - Optional exclusion of seen items
       - Batch prediction support

    The model supports various loss functions:
    - WARP: Weighted Approximate-Rank Pairwise
    - BPR: Bayesian Personalized Ranking
    - WARP-kOS: k-Order Statistic WARP
    - Logistic: Standard logistic loss
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        loss: str = "warp",
        no_components: int = 64,
        max_sampled: int = 10,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the LightFM recommender model.

        Configures the model with specified parameters and initializes
        the underlying LightFM instance with appropriate settings.

        Args:
            learning_rate (float): Learning rate for gradient updates.
                Controls step size during optimization.
                Default: 0.05
            loss (str): Loss function for training optimization.
                Options:
                - 'warp': Weighted Approximate-Rank Pairwise
                - 'bpr': Bayesian Personalized Ranking
                - 'warp-kos': k-Order Statistic WARP
                - 'logistic': Standard logistic loss
                Default: 'warp'
            no_components (int): Number of latent dimensions.
                Controls model complexity and capacity.
                Default: 64
            max_sampled (int): Maximum number of negative samples.
                Higher values may improve accuracy but slow training.
                Default: 10
            random_state (int): Random seed for reproducibility.
                Controls initialization and sampling.
                Default: 42
        """
        self.model = LightFM(
            learning_rate=learning_rate,
            loss=loss,
            no_components=no_components,
            max_sampled=max_sampled,
            random_state=random_state,
        )
        self.params = {
            "learning_rate": learning_rate,
            "loss": loss,
            "no_components": no_components,
            "max_sampled": max_sampled,
            "random_state": random_state,
        }

    def fit(
        self,
        interaction_matrix: csr_matrix,
        user_features: csr_matrix,
        item_features: csr_matrix,
        num_epochs: int = 10,
        num_threads: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Train the LightFM model on provided data.

        Performs model training using the specified interaction data
        and feature matrices. Supports multi-threaded training and
        customizable training duration.

        Args:
            interaction_matrix (csr_matrix): User-item interactions.
                Shape: (n_users, n_items)
                Values: 1 for positive interactions, 0 otherwise
            user_features (csr_matrix): User feature matrix.
                Shape: (n_users, n_user_features)
            item_features (csr_matrix): Item feature matrix.
                Shape: (n_items, n_item_features)
            num_epochs (int): Number of training iterations.
                Default: 10
            num_threads (int): Number of parallel training threads.
                Default: 4
            **kwargs: Additional arguments passed to LightFM.fit()
        """
        LOGGER.info(
            f"Training LightFM model with {num_epochs} epochs "
            f"and {num_threads} threads..."
        )

        try:
            self.model.fit(
                interactions=interaction_matrix,
                user_features=user_features,
                item_features=item_features,
                epochs=num_epochs,
                num_threads=num_threads,
                **kwargs,
            )
            LOGGER.info("Model training completed successfully")

        except Exception as e:
            LOGGER.error(f"Error during model training: {str(e)}")
            raise

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_features: csr_matrix,
        item_features: csr_matrix,
    ) -> np.ndarray:
        """
        Generate prediction scores for user-item pairs.

        Computes interaction likelihood scores for specified pairs
        using the trained model parameters and feature matrices.

        Args:
            user_ids (np.ndarray): User indices for prediction.
                Shape: (n_predictions,)
            item_ids (np.ndarray): Item indices for prediction.
                Shape: (n_predictions,)
            user_features (csr_matrix): User feature matrix.
                Shape: (n_users, n_user_features)
            item_features (csr_matrix): Item feature matrix.
                Shape: (n_items, n_item_features)

        Returns:
            np.ndarray: Predicted scores for each user-item pair.
                Shape: (n_predictions,)
                Values: Higher scores indicate stronger recommendations
        """
        try:
            predictions = self.model.predict(
                user_ids=user_ids,
                item_ids=item_ids,
                user_features=user_features,
                item_features=item_features,
            )
            return predictions

        except Exception as e:
            LOGGER.error(f"Error during prediction: {str(e)}")
            raise

    def recommend(
        self,
        user_id: int,
        user_features: csr_matrix,
        item_features: csr_matrix,
        n_items: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate personalized recommendations for a user.

        Creates a ranked list of items for the specified user based on:
        - Learned user preferences
        - Item characteristics
        - Optional exclusion of previously seen items

        Args:
            user_id (int): Index of the target user
            user_features (csr_matrix): User feature matrix
                Shape: (n_users, n_user_features)
            item_features (csr_matrix): Item feature matrix
                Shape: (n_items, n_item_features)
            n_items (int): Number of items to recommend
                Default: 10
            exclude_seen (bool): Whether to exclude seen items
                Default: True
            seen_items (Optional[np.ndarray]): Indices of items to exclude
                Shape: (n_seen_items,)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing:
                1. Indices of recommended items (shape: (n_items,))
                2. Prediction scores for recommendations (shape: (n_items,))
        """
        try:
            n_items_total = item_features.shape[0]
            scores = self.model.predict(
                user_ids=np.repeat(user_id, n_items_total),
                item_ids=np.arange(n_items_total),
                user_features=user_features,
                item_features=item_features,
            )

            if exclude_seen and seen_items is not None:
                scores[seen_items] = -np.inf

            top_items = np.argsort(-scores)[:n_items]
            top_scores = scores[top_items]

            return top_items, top_scores

        except Exception as e:
            LOGGER.error(f"Error during recommendation: {str(e)}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve the model's current parameters.

        Returns a copy of the model's configuration parameters,
        including learning rate, loss function, and model dimensions.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - learning_rate: Current learning rate
                - loss: Loss function name
                - no_components: Number of latent dimensions
                - max_sampled: Maximum negative samples
                - random_state: Random seed used
        """
        return self.params.copy()
