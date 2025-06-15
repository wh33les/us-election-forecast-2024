# src/data/processors.py
"""Data processing functions for election forecasting."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PollingDataProcessor:
    """Handles cleaning and processing of polling data."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def filter_polling_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter polling data based on configuration criteria.
        From your original forecast.py filtering logic.
        """
        logger.info("Filtering polling data...")

        # Extract relevant columns and candidates
        df = raw_data.loc[
            (raw_data["candidate_name"].isin(["Donald Trump", "Kamala Harris"]))
            & (raw_data["population"] == "lv"),  # likely voters only
            ["pollscore", "state", "candidate_name", "end_date", "pct"],
        ].drop_duplicates()

        logger.info(f"After candidate and population filter: {len(df)} records")

        # Restrict to national and swing state polls
        swing_states = [
            "Arizona",
            "Georgia",
            "Michigan",
            "Nevada",
            "North Carolina",
            "Pennsylvania",
            "Wisconsin",
        ]

        df = df.loc[df["state"].isin(swing_states) | df["state"].isnull()]
        logger.info(f"After geographic filter: {len(df)} records")

        # Only use polls with negative pollscore
        df = df.loc[df["pollscore"] < 0]
        logger.info(f"After pollscore filter: {len(df)} records")

        return df

    def calculate_daily_averages(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily averages for each candidate.
        From your original forecast.py groupby logic.
        """
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

    def merge_with_historical_data(
        self, new_data: pd.DataFrame, old_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Merge new data with historical data if available.
        From your original forecast.py merge logic.
        """
        if old_data is None:
            logger.info("No historical data to merge")
            return new_data

        logger.info("Merging new data with historical data...")

        # Merge old and new data
        df_merged = pd.concat([old_data, new_data], join="outer", ignore_index=True)

        # Sort again after merge
        df_merged.sort_values(
            ["candidate_name", "end_date"],
            ascending=[True, True],
            inplace=True,
            ignore_index=True,
        )

        logger.info(f"Merged data contains {len(df_merged)} total records")
        return df_merged

    def split_by_candidate(
        self, df_cleaned: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split cleaned data by candidate for modeling.
        From your original forecast.py candidate extraction.
        """
        trump = df_cleaned[(df_cleaned["candidate_name"] == "Donald Trump")].copy()

        harris = df_cleaned[(df_cleaned["candidate_name"] == "Kamala Harris")].copy()

        logger.info(
            f"Split data: {len(trump)} Trump records, {len(harris)} Harris records"
        )
        return trump, harris

    def filter_data_by_date(self, df: pd.DataFrame, cutoff_date) -> pd.DataFrame:
        """
        Filter data to only include records up to a specific date.
        Used for preventing future data leakage in rolling forecasts.
        """
        filtered = df[df["end_date"] <= cutoff_date].copy()
        logger.info(f"Filtered data to {cutoff_date}: {len(filtered)} records")
        return filtered

    def validate_data_quality(
        self, trump_data: pd.DataFrame, harris_data: pd.DataFrame
    ) -> bool:
        """
        Validate that we have sufficient data for modeling.
        """
        min_records = 10  # Minimum records needed for reliable modeling

        if len(trump_data) < min_records:
            logger.warning(
                f"Insufficient Trump data: {len(trump_data)} < {min_records}"
            )
            return False

        if len(harris_data) < min_records:
            logger.warning(
                f"Insufficient Harris data: {len(harris_data)} < {min_records}"
            )
            return False

        # Check for missing values in daily_average
        trump_missing = trump_data["daily_average"].isna().sum()
        harris_missing = harris_data["daily_average"].isna().sum()

        if trump_missing > 0:
            logger.warning(f"Trump data has {trump_missing} missing daily averages")

        if harris_missing > 0:
            logger.warning(f"Harris data has {harris_missing} missing daily averages")

        logger.info("Data quality validation passed")
        return True
