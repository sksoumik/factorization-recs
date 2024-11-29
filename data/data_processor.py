from typing import Any, Dict, Tuple

import pandas as pd
from scipy.sparse import csr_matrix

from utils.logger import Logger

logger = Logger.get_logger()


class DataProcessor:
    """
    Processes raw data into format suitable for LightFM model.
    """

    def __init__(self) -> None:
        """Initialize the DataProcessor."""
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.user_features_mapping: Dict[str, int] = {}
        self.item_features_mapping: Dict[str, int] = {}

    def _create_id_mappings(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> None:
        """
        Create mappings from IDs to indices.

        Args:
            users_df (pd.DataFrame): DataFrame containing user data
            items_df (pd.DataFrame): DataFrame containing item data
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
        Create mappings for categorical features.

        Args:
            users_df (pd.DataFrame): DataFrame containing user data
            items_df (pd.DataFrame): DataFrame containing item data
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
        Create sparse feature matrix for users or items.

        Args:
            df (pd.DataFrame): Input DataFrame
            id_col (str): Column name for ID
            id_mapping (Dict[str, int]): Mapping from IDs to indices
            feature_mapping (Dict[str, int]): Mapping from features to indices
            categorical_cols (list): List of categorical column names

        Returns:
            csr_matrix: Sparse feature matrix
        """
        rows = []
        cols = []
        data = []

        for idx, row in df.iterrows():
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

        Args:
            users_df (pd.DataFrame): DataFrame containing user data
            items_df (pd.DataFrame): DataFrame containing item data
            interactions_df (pd.DataFrame): DataFrame containing
            interaction data

        Returns:
            Tuple containing:
                - interaction matrix (csr_matrix)
                - user features matrix (csr_matrix)
                - item features matrix (csr_matrix)
                - mappings dictionary (Dict[str, Any])
        """
        logger.info("Processing data for LightFM model...")

        # Create ID mappings
        self._create_id_mappings(users_df, items_df)
        self._create_feature_mappings(users_df, items_df)

        # Create interaction matrix
        rows = [self.user_mapping[uid] for uid in interactions_df["user_id"]]
        cols = [self.item_mapping[iid] for iid in interactions_df["item_id"]]
        data = [1.0 if conv else 0.0 for conv in interactions_df["conversion"]]

        interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_mapping), len(self.item_mapping)),
        )

        # Create feature matrices
        user_features = self._create_feature_matrix(
            users_df,
            "user_id",
            self.user_mapping,
            self.user_features_mapping,
            [
                "gender",
                "location",
                "membership_level",
                "device_type",
                "browser",
            ],
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
