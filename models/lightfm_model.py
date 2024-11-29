"""LightFM model implementation module."""

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
    """

    def __init__(
        # pylint: disable=too-many-arguments
        self,
        learning_rate: float = 0.05,
        loss: str = "warp",
        no_components: int = 64,
        max_sampled: int = 10,
        random_state: int = 42,
    ) -> None:
        """
        Initialize LightFM recommender.

        Args:
            learning_rate (float): Learning rate for model training
            loss (str): Loss function ('warp', 'bpr', 'warp-kos', 'logistic')
            no_components (int): Number of latent dimensions
            max_sampled (int): Maximum number of negative samples
            random_state (int): Random seed for reproducibility
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
        Train the LightFM model.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix
            num_epochs (int): Number of training epochs
            num_threads (int): Number of threads to use
            **kwargs: Additional arguments for the model
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
        Generate predictions for user-item pairs.

        Args:
            user_ids (np.ndarray): Array of user indices
            item_ids (np.ndarray): Array of item indices
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix

        Returns:
            np.ndarray: Predicted scores for each user-item pair
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

    def recommend(  # pylint: disable=too-many-positional-arguments
        self,
        user_id: int,
        user_features: csr_matrix,
        item_features: csr_matrix,
        n_items: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate recommendations for a user.

        Args:
            user_id (int): User index
            user_features (csr_matrix): User features matrix
            item_features (csr_matrix): Item features matrix
            n_items (int): Number of recommendations to generate
            exclude_seen (bool): Whether to exclude already seen items
            seen_items (Optional[np.ndarray]): Array of seen item indices

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of recommended item indices
                - Array of prediction scores
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
        Get model parameters.

        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return self.params.copy()
