"""Synthetic data generation module for recommender system."""

import random
from typing import Tuple

import numpy as np
import pandas as pd
from faker import Faker

from schemas.data_schemas import Interaction, ItemFeatures, UserFeatures
from utils.logger import Logger

LOGGER = Logger.get_logger()


class DataGenerator:
    """
    Generates synthetic data for the recommender system.
    """

    def __init__(
        self,
        n_users: int = 1000,
        n_items: int = 5000,
        n_interactions: int = 10000,
        random_seed: int = 42,
    ):
        """
        Initialize the data generator.

        Args:
            n_users (int): Number of users to generate
            n_items (int): Number of items to generate
            n_interactions (int): Number of interactions to generate
            random_seed (int): Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.fake = Faker()
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.categories = [
            "Electronics",
            "Books",
            "Clothing",
            "Home & Kitchen",
            "Sports",
            "Beauty",
            "Toys",
            "Automotive",
            "Health",
        ]
        self.membership_levels = ["Bronze", "Silver", "Gold", "Platinum"]
        self.brands = [
            "Apple",
            "Samsung",
            "Nike",
            "Adidas",
            "Sony",
            "LG",
            "Dell",
            "HP",
            "Asus",
            "Lenovo",
        ]

    def generate_users(self) -> pd.DataFrame:
        """
        Generate synthetic user data.

        Returns:
            pd.DataFrame: DataFrame containing user features
        """
        LOGGER.info(f"Generating {self.n_users} users...")

        users = []
        for _ in range(self.n_users):
            user = UserFeatures(
                user_id=str(self.fake.uuid4()),
                age=random.randint(18, 90),
                gender=random.choice(["M", "F", "Other"]),
                location=self.fake.city(),
                membership_level=random.choice(self.membership_levels),
                registration_date=self.fake.date_time_between(
                    start_date="-2y", end_date="now"
                ),
                device_type=random.choice(["Desktop", "Mobile", "Tablet"]),
                browser=random.choice(["Chrome", "Firefox", "Safari"]),
                is_mobile=random.choice([True, False]),
                avg_session_time=random.uniform(1, 60),
                total_purchases=random.randint(0, 100),
            )
            users.append(user.dict())

        return pd.DataFrame(users)

    def generate_items(self) -> pd.DataFrame:
        """
        Generate synthetic item data.

        Returns:
            pd.DataFrame: DataFrame containing item features
        """
        LOGGER.info(f"Generating {self.n_items} items...")

        items = []
        for _ in range(self.n_items):
            item = ItemFeatures(
                item_id=str(self.fake.uuid4()),
                category=random.choice(self.categories),
                subcategory=self.fake.word(),
                price=round(random.uniform(1, 1000), 2),
                brand=random.choice(self.brands),
                avg_rating=round(random.uniform(1, 5), 1),
                total_reviews=random.randint(0, 1000),
                in_stock=random.choice([True, False]),
                discount_percentage=round(random.uniform(0, 50), 2),
                weight=round(random.uniform(0.1, 20), 2),
                is_new=random.choice([True, False]),
            )
            items.append(item.dict())

        return pd.DataFrame(items)

    def generate_interactions(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate synthetic interaction data.

        Args:
            users_df (pd.DataFrame): DataFrame containing user features
            items_df (pd.DataFrame): DataFrame containing item features

        Returns:
            pd.DataFrame: DataFrame containing interaction data
        """
        LOGGER.info(f"Generating {self.n_interactions} interactions...")

        interactions = []
        user_ids = users_df["user_id"].tolist()
        item_ids = items_df["item_id"].tolist()

        for _ in range(self.n_interactions):
            user_id = random.choice(user_ids)
            item_id = random.choice(item_ids)

            interaction = Interaction(
                user_id=user_id,
                item_id=item_id,
                timestamp=self.fake.date_time_between(
                    start_date="-30d", end_date="now"
                ),
                interaction_type=random.choice(["view", "cart", "purchase"]),
                conversion=random.choice([True, False]),
                session_id=str(self.fake.uuid4()),
                interaction_duration=random.uniform(0, 300),
                device_type=random.choice(["Desktop", "Mobile", "Tablet"]),
                price_at_interaction=round(random.uniform(1, 1000), 2),
            )
            interactions.append(interaction.dict())

        return pd.DataFrame(interactions)

    def generate_dataset(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset including users, items, and interactions.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple
            containing users, items, and interactions DataFrames
        """
        users_df = self.generate_users()
        items_df = self.generate_items()
        interactions_df = self.generate_interactions(users_df, items_df)

        return users_df, items_df, interactions_df
