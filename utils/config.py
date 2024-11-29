from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model parameters."""

    learning_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    loss: str = Field(default="warp", pattern="^(warp|bpr|warp-kos|logistic)$")
    no_components: int = Field(default=64, ge=1)
    max_sampled: int = Field(default=10, ge=1)
    random_state: int = Field(default=42)


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    num_epochs: int = Field(default=10, ge=1)
    num_threads: int = Field(default=4, ge=1)
    batch_size: int = Field(default=256, ge=1)
    validation_size: float = Field(default=0.2, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation parameters."""

    metrics: list[str] = Field(default=["precision", "recall", "ndcg", "map"])
    k_values: list[int] = Field(default=[5, 10, 20])
    n_folds: int = Field(default=5, ge=2)


class DataConfig(BaseModel):
    """Configuration for data generation and processing."""

    n_users: int = Field(default=1000, ge=1)
    n_items: int = Field(default=5000, ge=1)
    n_interactions: int = Field(default=10000, ge=1)
    random_seed: int = Field(default=42)


class Config(BaseModel):
    """Main configuration class."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    data: DataConfig = Field(default_factory=DataConfig)


class ConfigManager:
    """Manages configuration loading and saving."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """
        Initialize ConfigManager.

        Args:
            config_path (Optional[Path]): Path to configuration file
        """
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Config:
        """
        Load configuration from file or create default.

        Returns:
            Config: Configuration object
        """
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return Config(**config_dict)
        return Config()

    def save_config(self) -> None:
        """Save configuration to file."""
        config_dict = self.config.dict()
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config_dict, f)

    def get_config(self) -> Config:
        """
        Get configuration object.

        Returns:
            Config: Configuration object
        """
        return self.config

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            config_dict (Dict[str, Any]): Dictionary of configuration updates
        """
        current_dict = self.config.dict()
        current_dict.update(config_dict)
        self.config = Config(**current_dict)
