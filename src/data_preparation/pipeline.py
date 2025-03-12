"""Defines the complete data preparation process. Includes the DataSet class
for encapsulating features and labels, and the DataPipeline class that handles
data import, exploration, cleaning, splitting and scaling. Converts raw data
into training, validation and test sets ready to feed to models."""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import CONFIG
from data_preparation.data_cleaner import DataFrameCleaner
from data_preparation.initial_exploration import InitialExploration
from data_preparation.test_validation_train import ManualDataFrameSplitter
from facade.import_interface import IMPORT_FACADE


@dataclass
class DataSet:
    """Class to hold features and target data for a dataset."""

    x: pd.DataFrame
    y: pd.Series


class DataPipeline:
    """Handles the entire data processing pipeline."""

    def process(self) -> Tuple[DataSet, DataSet, DataSet]:
        """Process raw data and return train, validation, and test datasets."""
        # Import raw data
        raw_data = IMPORT_FACADE(option="data_raw")

        # Explore data
        exploration = InitialExploration(raw_data)
        explored_data = exploration.run_all()

        # Clean data
        cleaner = DataFrameCleaner(explored_data)
        clean_data = cleaner.clean(raw_data)

        # Split data
        target_col_index = self._get_target_column_index(clean_data)
        splitter = ManualDataFrameSplitter(
            clean_data, target_index=target_col_index, divition=CONFIG["split_ratio"]
        )
        x_train, x_val, x_test, y_train, y_val, y_test = splitter.get_splits()

        # Scale features
        scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(
            scaler.fit_transform(x_train), columns=x_train.columns
        )
        x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        # Create and return dataset objects
        train_data = DataSet(x_train_scaled, y_train)
        val_data = DataSet(x_val_scaled, y_val)
        test_data = DataSet(x_test_scaled, y_test)

        return train_data, val_data, test_data

    def _get_target_column_index(self, df: pd.DataFrame) -> int:
        """Get the index of the target column."""
        target_col_name = CONFIG["target_column"]
        try:
            return df.columns.get_loc(target_col_name)
        except KeyError:
            # Fallback to the original index if column name not found
            return 30
