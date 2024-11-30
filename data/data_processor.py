"""
Data processing module for preparing recommender system data.

This module handles the transformation of raw data into formats suitable for
the LightFM recommender model. It provides functionality for:
1. Converting categorical features into sparse matrix format
2. Creating ID mappings for users and items
3. Building interaction matrices
4. Managing feature mappings for both users and items

Key responsibilities:
- Feature engineering and transformation
- Sparse matrix construction
- Index mapping management
- Data format validation
"""

from typing import Any, Dict, Tuple

import pandas as pd
from scipy.sparse import csr_matrix

from utils.logger import Logger

LOGGER = Logger.get_logger()


class DataProcessor:
    """
    Data processor for recommender system input preparation.

    This class handles the transformation of raw data into the format
    required by the LightFM model. It provides functionality for:

    1. Feature Processing:
       - Converting categorical features to sparse format
       - Handling numerical features
       - Creating feature mappings

    2. ID Management:
       - Creating user ID mappings
       - Creating item ID mappings
       - Maintaining consistent indices

    3. Matrix Construction:
       - Building interaction matrices
       - Creating feature matrices
       - Ensuring proper matrix formats

    4. Data Validation:
       - Checking data consistency
       - Validating feature formats
       - Ensuring proper dimensions
    """

    def __init__(self) -> None:
        """
        Initialize the DataProcessor.

        Sets up the necessary mapping dictionaries for:
        - User IDs to indices
        - Item IDs to indices
        - User feature names to indices
        - Item feature names to indices
        """
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.user_features_mapping: Dict[str, int] = {}
        self.item_features_mapping: Dict[str, int] = {}

    def _create_id_mappings(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> None:
        """
        Create mappings from string IDs to integer indices.

        Generates consistent integer indices for users and items:
        - Maps user IDs to consecutive integers
        - Maps item IDs to consecutive integers
        - Ensures consistent mapping across the dataset

        Args:
            users_df (pd.DataFrame): DataFrame containing user data
                Must have 'user_id' column
            items_df (pd.DataFrame): DataFrame containing item data
                Must have 'item_id' column
        """
        self.user_mapping = {
            uid: idx for idx, uid in enumerate(users_df["user_id"].unique())
        }
        self.item_mapping = {
            iid: idx for idx, iid in enumerate(items_df["item_id"].unique())
        }

    def _create_feature_mappings(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> None:
        """
        Create mappings for categorical features to indices.

        Processes categorical features and creates mappings:
        - Maps each unique feature value to an index
        - Handles both user and item features
        - Maintains separate feature spaces

        Args:
            users_df (pd.DataFrame): DataFrame containing user features
                Must contain specified categorical columns
            items_df (pd.DataFrame): DataFrame containing item features
                Must contain specified categorical columns
        """
        feature_idx = 0

        # User features mapping
        categorical_user_cols = [
            "gender",
            "location",
            "membership_level",
            "device_type",
            "browser",
        ]

        for col in categorical_user_cols:
            unique_values = users_df[col].unique()
            for value in unique_values:
                feature_key = f"{col}_{value}"
                self.user_features_mapping[feature_key] = feature_idx
                feature_idx += 1

        # Item features mapping
        categorical_item_cols = ["category", "subcategory", "brand"]

        for col in categorical_item_cols:
            unique_values = items_df[col].unique()
            for value in unique_values:
                feature_key = f"{col}_{value}"
                self.item_features_mapping[feature_key] = feature_idx
                feature_idx += 1

    def _create_feature_matrix(
        self,
        df: pd.DataFrame,
        id_col: str,
        id_mapping: Dict[str, int],
        feature_mapping: Dict[str, int],
        categorical_cols: list,
    ) -> csr_matrix:
        """
        Create sparse feature matrix from categorical columns.

        Transforms categorical features into a sparse matrix format:
        - One-hot encodes categorical variables
        - Creates sparse matrix representation
        - Maintains proper dimensionality

        Args:
            df (pd.DataFrame): Input DataFrame with features
            id_col (str): Name of the ID column
            id_mapping (Dict[str, int]): Mapping from IDs to indices
            feature_mapping (Dict[str, int]): Mapping from features to indices
            categorical_cols (list): List of categorical column names

        Returns:
            csr_matrix: Sparse matrix of features
                Shape: (n_entities, n_features)
                Values: Binary indicators (1.0 for present features)
        """
        rows = []
        cols = []
        data = []

        for _, row in df.iterrows():
            entity_idx = id_mapping[row[id_col]]

            for col in categorical_cols:
                feature_key = f"{col}_{row[col]}"
                if feature_key in feature_mapping:
                    rows.append(entity_idx)
                    cols.append(feature_mapping[feature_key])
                    data.append(1.0)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(id_mapping), max(feature_mapping.values()) + 1),
        )

    def process_data(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Dict[str, Any]]:
        """
        Process raw data into format suitable for LightFM.

        Performs complete data processing workflow:
        1. Creates ID mappings for users and items
        2. Processes categorical features
        3. Builds interaction and feature matrices
        4. Validates data consistency

        Args:
            users_df (pd.DataFrame): DataFrame containing user data
                Must include 'user_id' and feature columns
            items_df (pd.DataFrame): DataFrame containing item data
                Must include 'item_id' and feature columns
            interactions_df (pd.DataFrame): DataFrame containing interactions
                Must include 'user_id', 'item_id', and 'conversion'

        Returns:
            Tuple[csr_matrix, csr_matrix, csr_matrix, Dict[str, Any]]:
                A tuple containing:
                - interaction_matrix: User-item interactions (n_users, n_items)
                - user_features: User feature matrix (n_users, n_user_features)
                - item_features: Item feature matrix (n_items, n_item_features)
                - mappings: Dictionary of ID and feature mappings
        """
        LOGGER.info("Processing data for LightFM model...")

        self._create_id_mappings(users_df, items_df)
        self._create_feature_mappings(users_df, items_df)

        rows = [self.user_mapping[uid] for uid in interactions_df["user_id"]]
        cols = [self.item_mapping[iid] for iid in interactions_df["item_id"]]
        data = [1.0 if conv else 0.0 for conv in interactions_df["conversion"]]

        interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_mapping), len(self.item_mapping)),
        )

        user_features = self._create_feature_matrix(
            users_df,
            "user_id",
            self.user_mapping,
            self.user_features_mapping,
            ["gender", "location", "membership_level", "device_type", "browser"],
        )

        item_features = self._create_feature_matrix(
            items_df,
            "item_id",
            self.item_mapping,
            self.item_features_mapping,
            ["category", "subcategory", "brand"],
        )

        mappings = {
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "user_features_mapping": self.user_features_mapping,
            "item_features_mapping": self.item_features_mapping,
        }

        return interaction_matrix, user_features, item_features, mappings
