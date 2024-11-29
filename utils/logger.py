import logging
from pathlib import Path
from typing import Optional


class Logger:
    """Singleton logger class for the recommender system."""

    _instance: Optional[logging.Logger] = None

    @staticmethod
    def get_logger(name: str = "recommender_system") -> logging.Logger:
        """
        Get or create a logger instance.

        Args:
            name (str): Logger name

        Returns:
            logging.Logger: Logger instance
        """
        if Logger._instance is None:
            # Create logger
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create file handler
            log_path = Path("logs")
            log_path.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_path / "recommender.log")
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers to logger
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

            Logger._instance = logger

        return Logger._instance
