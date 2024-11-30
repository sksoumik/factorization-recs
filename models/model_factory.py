"""
Model factory module for creating recommender instances.

This module implements the Factory Method pattern for recommender model creation.
It provides a centralized point for:
1. Model instantiation with configuration
2. Model type validation
3. Error handling during creation
4. Extensible model registry

The factory supports:
- Multiple model implementations
- Configurable model parameters
- Consistent model interface
- Graceful error handling
"""

from typing import Any, Dict, Optional

from models.base_model import BaseRecommender
from models.lightfm_model import LightFMRecommender
from utils.logger import Logger

LOGGER = Logger.get_logger()


class ModelFactory:
    """
    Factory class for creating recommender model instances.

    This class implements the Factory Method pattern to:
    1. Create Model Instances:
       - Instantiate specific model implementations
       - Configure model parameters
       - Validate model types

    2. Manage Model Registry:
       - Track available model implementations
       - Validate model requests
       - Support easy extension

    3. Handle Creation Process:
       - Parameter validation
       - Error handling
       - Logging of creation process

    The factory currently supports:
    - LightFM: Hybrid matrix factorization model
    - (Additional models can be easily registered)
    """

    # Registry of available model implementations
    _models = {"lightfm": LightFMRecommender}

    @classmethod
    def create_model(
        cls, model_type: str, model_params: Optional[Dict[str, Any]] = None
    ) -> BaseRecommender:
        """
        Create and configure a recommender model instance.

        This method handles the complete model creation process:
        1. Validates the requested model type
        2. Applies the provided configuration
        3. Instantiates the model
        4. Logs the creation process

        Args:
            model_type (str): Type of model to create.
                Must be one of the registered model types.
                Currently supported: ['lightfm']
            model_params (Optional[Dict[str, Any]]): Model configuration parameters.
                If None, uses default parameters for the specified model.
                Parameters are model-specific and should match the model's
                constructor arguments.

        Returns:
            BaseRecommender: Configured instance of the specified model.
                The instance will implement the BaseRecommender interface.

        Raises:
            ValueError: If the requested model_type is not supported.
                Includes list of available models in error message.
            Exception: If model creation fails for any other reason.
                Includes detailed error information in logs.

        Example:
            >>> model = ModelFactory.create_model(
            ...     model_type="lightfm",
            ...     model_params={"learning_rate": 0.05, "loss": "warp"}
            ... )
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
            LOGGER.error(
                f"Error creating {model_type} model: {str(e)}\n" f"Parameters: {params}"
            )
            raise
