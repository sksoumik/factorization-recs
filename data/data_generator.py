"""
Synthetic data generation module for recommender system.

This module is responsible for generating realistic synthetic data that mimics
e-commerce user-item interactions. It generates three types of data:
1. User data with demographic and behavioral features
2. Item data with product characteristics
3. Interaction data representing user-item engagements

The generated data follows patterns commonly seen in e-commerce platforms:
- User features include age, location, device preferences
- Item features include categories, prices, ratings
- Interactions include views, cart additions, and purchases
"""

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
    Synthetic data generator for recommender system testing and development.

    This class provides functionality to generate three types of synthetic data:
    1. User Data:
       - Demographic information (age, gender, location)
       - Platform behavior (device type, session time)
       - Membership details (level, registration date)

    2. Item Data:
       - Product details (category, brand, price)
       - Performance metrics (ratings, reviews)
       - Inventory information (stock status, discounts)

    3. Interaction Data:
       - User-item engagement events (views, carts, purchases)
       - Temporal information (timestamps, durations)
       - Contextual data (device type, session)
    """

    def __init__(
        self,
        n_users: int = 1000,
        n_items: int = 5000,
        n_interactions: int = 10000,
        random_seed: int = 42,
    ):
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
        Generate synthetic user profiles.

        Creates a DataFrame of user profiles with realistic features:
        - Unique user identifiers
        - Demographic information (age, gender, location)
        - Platform preferences (device type, browser)
        - Behavioral metrics (session time, purchase history)
        - Membership information (level, registration date)

        Returns:
            pd.DataFrame: DataFrame containing user profiles with features
                matching the UserFeatures schema
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
        Generate synthetic item catalog.

        Creates a DataFrame of items with realistic product features:
        - Unique item identifiers
        - Product categorization (category, subcategory)
        - Commercial attributes (price, brand, discount)
        - Performance metrics (rating, review count)
        - Inventory status (in stock, new item flag)

        Returns:
            pd.DataFrame: DataFrame containing item data with features
                matching the ItemFeatures schema
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
        Generate synthetic user-item interactions.

        Creates realistic interaction data between users and items:
        - Links users and items through various interaction types
        - Adds temporal aspects (timestamps, durations)
        - Includes interaction context (device, session)
        - Tracks conversion events and pricing

        Args:
            users_df (pd.DataFrame): DataFrame containing user profiles
            items_df (pd.DataFrame): DataFrame containing item catalog

        Returns:
            pd.DataFrame: DataFrame containing interaction data with
                features matching the Interaction schema
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
        Generate complete synthetic dataset.

        Creates a full dataset containing users, items, and their
        interactions. Ensures consistency across all generated data
        and maintains referential integrity between tables.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple
            containing:
                - users_df: DataFrame of user profiles
                - items_df: DataFrame of item catalog
                - interactions_df: DataFrame of user-item interactions
        """
        users_df = self.generate_users()
        items_df = self.generate_items()
        interactions_df = self.generate_interactions(users_df, items_df)

        return users_df, items_df, interactions_df
