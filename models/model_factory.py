"""Model factory module for creating recommender instances."""

from typing import Any, Dict, Optional

from models.base_model import BaseRecommender
from models.lightfm_model import LightFMRecommender
from utils.logger import Logger

LOGGER = Logger.get_logger()


class ModelFactory:
    """
    Factory class for creating recommender models.
    """

    _models = {"lightfm": LightFMRecommender}

    @classmethod
    def create_model(
        cls, model_type: str, model_params: Optional[Dict[str, Any]] = None
    ) -> BaseRecommender:
        """
        Create a recommender model instance.

        Args:
            model_type (str): Type of model to create
            model_params (Optional[Dict[str, Any]]): Model parameters

        Returns:
            BaseRecommender: Instance of the specified recommender model

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._models:
            LOGGER.error(f"Unsupported model type: {model_type}")
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available models: {list(cls._models.keys())}"
            )

        model_class = cls._models[model_type]
        params = model_params or {}

        try:
            LOGGER.info(f"Creating {model_type} model with params: {params}")
            return model_class(**params)

        except Exception as e:
            LOGGER.error(f"Error creating model: {str(e)}")
            raise
