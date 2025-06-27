# src/data/collectors.py
"""Data collection functions for election forecasting."""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PollingDataCollector:
    """Handles loading and initial processing of polling data."""

    def __init__(self, config):
        """Initialize with data configuration."""
        self.config = config

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw polling data from CSV file specified in config."""
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
