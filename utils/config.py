"""
Configuration management module for the recommender system.

This module handles all configuration-related functionality, including:
- Loading configurations from YAML files
- Providing default configurations
- Validating configuration parameters
- Managing configuration updates
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from utils.logger import Logger

LOGGER = Logger.get_logger()


class ModelConfig(BaseModel):
    """
    Configuration settings for the recommendation model.

    This class defines the parameters that control the model's behavior:
    - learning_rate: Controls how much the model adjusts during training
    - loss: The optimization objective (warp, bpr, etc.)
    - no_components: Number of latent factors in the model
    - max_sampled: Maximum number of negative samples per positive
    - random_state: Seed for reproducibility
    """

    learning_rate: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Learning rate for model optimization"
    )
    loss: str = Field(
        default="warp",
        pattern="^(warp|bpr|warp-kos|logistic)$",
        description="Loss function for optimization",
    )
    no_components: int = Field(default=64, ge=1, description="Number of latent factors")
    max_sampled: int = Field(
        default=10, ge=1, description="Maximum number of negative samples"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class TrainingConfig(BaseModel):
    """
    Configuration settings for the training process.

    Controls the training behavior:
    - num_epochs: Number of training iterations
    - num_threads: Parallel processing threads
    - batch_size: Number of samples per training batch
    - validation_size: Proportion of data used for validation
    """

    num_epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    num_threads: int = Field(
        default=4, ge=1, description="Number of threads for parallel processing"
    )
    batch_size: int = Field(default=256, ge=1, description="Training batch size")
    validation_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fraction of data used for validation"
    )


class EvaluationConfig(BaseModel):
    """
    Configuration settings for model evaluation.

    Defines how the model's performance is measured:
    - metrics: List of evaluation metrics to compute
    - k_values: Top-K values for ranking metrics
    - n_folds: Number of cross-validation folds
    """

    metrics: list[str] = Field(
        default=["precision", "recall", "ndcg", "map"],
        description="Evaluation metrics to compute",
    )
    k_values: list[int] = Field(
        default=[5, 10, 20], description="K values for ranking metrics"
    )
    n_folds: int = Field(
        default=5, ge=2, description="Number of cross-validation folds"
    )


class DataConfig(BaseModel):
    """
    Configuration settings for data generation and processing.

    Controls the synthetic data generation:
    - n_users: Number of users in the dataset
    - n_items: Number of items in the dataset
    - n_interactions: Number of user-item interactions
    - random_seed: Seed for reproducible data generation
    """

    n_users: int = Field(default=1000, ge=1, description="Number of users to generate")
    n_items: int = Field(default=5000, ge=1, description="Number of items to generate")
    n_interactions: int = Field(
        default=10000, ge=1, description="Number of interactions to generate"
    )
    random_seed: int = Field(default=42, description="Random seed for data generation")


class Config(BaseModel):
    """
    Main configuration class that combines all configuration components.

    This class serves as the central configuration hub, combining:
    - Model configuration
    - Training configuration
    - Evaluation configuration
    - Data configuration

    It provides a single point of access to all configuration parameters
    and ensures their consistency and validity.
    """

    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration settings"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration settings"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation configuration settings",
    )
    data: DataConfig = Field(
        default_factory=DataConfig, description="Data configuration settings"
    )


class ConfigManager:
    """
    Configuration management for the recommender pipeline.

    This class handles all aspects of configuration management:
    - Loading configurations from YAML files
    - Providing default configurations when needed
    - Validating configuration parameters
    - Handling configuration updates
    - Saving configurations to files

    It ensures that all configuration parameters are valid and consistent
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Config:
        """
        Load and validate configuration from file.

        First attempts to load from the specified config file.
        If the file doesn't exist or contains invalid configuration,
        falls back to default values with appropriate warning messages.

        Returns:
            Config: A validated configuration object
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
                if config_dict is None:
                    LOGGER.warning(
                        f"Empty config file found at {self.config_path}. "
                        "Using default configuration."
                    )
                    return Config()
                return Config(**config_dict)
            except (yaml.YAMLError, ValidationError) as e:
                LOGGER.error(
                    f"Error loading config from {self.config_path}: {str(e)}"
                    "\nUsing default configuration."
                )
                return Config()
        return Config()

    def save_config(self) -> None:
        """
        Save the current configuration to file.

        Writes the configuration to the specified file path in YAML format.
        Creates necessary directories if they don't exist.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = self.config.model_dump()
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
        LOGGER.info(f"Configuration saved to {self.config_path}")

    def get_config(self) -> Config:
        """
        Retrieve the current configuration.

        Returns:
            Config: The current active configuration object
        """
        return self.config

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update the current configuration with new values.

        Merges the provided configuration dictionary with the current
        configuration, validates the result, and updates the current
        configuration if valid.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing
            configuration updates.

        Raises:
            ValidationError: If the updated configuration is invalid
        """
        try:
            current_dict = self.config.model_dump()
            current_dict.update(config_dict)
            self.config = Config(**current_dict)
            LOGGER.info("Configuration updated successfully")
        except ValidationError as e:
            LOGGER.error(f"Invalid configuration update: {str(e)}")
            raise
