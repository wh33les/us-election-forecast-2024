# src/data/collectors.py
"""Data collection functions for election forecasting."""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PollingDataCollector:
    """Handles loading and initial processing of polling data."""

    def __init__(self, config):
        """Initialize with data configuration."""
        self.config = config

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw polling data from CSV file specified in ../config.py.
        """
        try:
            logger.info(f"Loading raw data from {self.config.raw_data_path}")
            raw_data = pd.read_csv(self.config.raw_data_path)

            # Parse dates
            raw_data["end_date"] = pd.to_datetime(
                raw_data["end_date"], format="mixed"
            ).dt.date

            logger.info(f"Loaded {len(raw_data)} raw polling records")
            return raw_data

        except FileNotFoundError:
            logger.error(f"Raw data file not found: {self.config.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise

    def load_previous_cleaned_data(self) -> Optional[pd.DataFrame]:
        """
        Load previously cleaned data if it exists.
        From your original forecast.py updating logic.
        """
        cleaned_path = Path(self.config.cleaned_data_path)
        if not cleaned_path.exists():
            logger.info("No previous cleaned data found, starting fresh")
            return None

        try:
            logger.info(f"Loading previous cleaned data from {cleaned_path}")
            old_cleaned_df = pd.read_csv(cleaned_path)

            # Parse dates
            old_cleaned_df["end_date"] = pd.to_datetime(
                old_cleaned_df["end_date"], format="mixed"
            ).dt.date

            logger.info(f"Loaded {len(old_cleaned_df)} previous records")
            return old_cleaned_df

        except Exception as e:
            logger.error(f"Error loading previous cleaned data: {e}")
            return None

    def determine_new_data_cutoff(self, old_cleaned_df: pd.DataFrame) -> pd.Timestamp:
        """
        Determine cutoff date for new data based on previous forecasts.
        From your original updating logic.
        """
        try:
            next_index = old_cleaned_df["drift_pred"].first_valid_index()
            if next_index is None:
                # No previous predictions, start from beginning
                return self.config.biden_dropout_date_parsed

            new_date = old_cleaned_df["end_date"][next_index]
            logger.info(f"Determined new data cutoff: {new_date}")
            return new_date

        except Exception as e:
            logger.warning(f"Error determining cutoff, using Biden dropout date: {e}")
            return self.config.biden_dropout_date_parsed

    def trim_raw_data_for_update(
        self, raw_data: pd.DataFrame, cutoff_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Trim raw data to only include new records for incremental updates.
        """
        trimmed_data = raw_data[raw_data["end_date"] >= cutoff_date]
        logger.info(
            f"Trimmed raw data from {len(raw_data)} to {len(trimmed_data)} records"
        )
        return trimmed_data

    def clean_old_predictions(
        self, old_cleaned_df: pd.DataFrame, cutoff_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Remove old predictions and clear model columns for re-calculation.
        From your original forecast.py logic.
        """
        # Remove rows from cutoff date onwards
        cleaned = old_cleaned_df.drop(
            old_cleaned_df[old_cleaned_df["end_date"] >= cutoff_date].index
        )

        # Clear model prediction columns
        cleaned["model"] = pd.Series(pd.NA, index=range(len(cleaned)))
        cleaned["drift_pred"] = pd.Series(pd.NA, index=range(len(cleaned)))

        logger.info(f"Cleaned old predictions, kept {len(cleaned)} historical records")
        return cleaneds
