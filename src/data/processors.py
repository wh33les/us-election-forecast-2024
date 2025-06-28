# src/data/processors.py
"""Data processing functions for election forecasting."""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PollingDataProcessor:
    """Handles cleaning and processing of polling data."""

    def __init__(self, data_config):
        """Initialize with data configuration."""
        self.config = data_config

    def filter_polling_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Filter polling data based on configuration criteria."""
        logger.info("Filtering polling data...")

        # Extract relevant columns and candidates - UPDATED: Use config
        df = raw_data.loc[
            (raw_data["candidate_name"].isin(self.config.candidates))
            & (raw_data["population"] == self.config.population_filter),
            ["pollscore", "state", "candidate_name", "end_date", "pct"],
        ].drop_duplicates()

        logger.info(f"After candidate and population filter: {len(df)} records")

        # Restrict to national and swing state polls - UPDATED: Use config
        df = df.loc[df["state"].isin(self.config.swing_states) | df["state"].isnull()]
        logger.info(f"After geographic filter: {len(df)} records")

        # Only use polls with negative pollscore - UPDATED: Use config
        df = df.loc[df["pollscore"] < self.config.pollscore_threshold]
        logger.info(f"After pollscore filter: {len(df)} records")

        return df

    def calculate_daily_averages(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily averages for each candidate."""
        logger.info("Calculating daily averages...")

        # Group by date and candidate to get daily averages
        daily_average = (
            filtered_data.groupby(["end_date", "candidate_name"])["pct"]
            .mean()
            .reset_index(name="daily_average")
        )

        # Merge back with original data
        df_with_averages = pd.merge(filtered_data, daily_average)

        # Create cleaned dataframe with just what we need
        df_cleaned = df_with_averages[
            ["candidate_name", "end_date", "daily_average"]
        ].drop_duplicates()

        # Sort by candidate and date
        df_cleaned.sort_values(
            ["candidate_name", "end_date"],
            ascending=[True, True],
            inplace=True,
            ignore_index=True,
        )

        logger.info(
            f"Calculated daily averages for {len(df_cleaned)} candidate-date pairs"
        )
        return df_cleaned

    def split_by_candidate(
        self, df_cleaned: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split cleaned data by candidate for modeling."""
        trump = df_cleaned[(df_cleaned["candidate_name"] == "Donald Trump")].copy()
        harris = df_cleaned[(df_cleaned["candidate_name"] == "Kamala Harris")].copy()

        logger.info(
            f"Split data: {len(trump)} Trump records, {len(harris)} Harris records"
        )
        return trump, harris
