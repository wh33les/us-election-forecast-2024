# src/data/collectors.py
"""Data collection functions for election forecasting."""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PollingDataCollector:
    """Handles loading and initial processing of polling data."""

    def __init__(self, config):
        self.config = config

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw polling data from CSV file."""
        try:
            logger.info(f"Loading raw data from {self.config.raw_data_path}")
            raw_data = pd.read_csv(self.config.raw_data_path)
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

    def load_incremental_data(self, target_date=None) -> pd.DataFrame:
        """Load data incrementally for target date - PERFORMANCE OPTIMIZED."""
        logger.info(f"Loading data incrementally for target date: {target_date}")

        # Check what polling dates we already have processed
        comprehensive_path = Path("data/election_forecast_2024_comprehensive.csv")
        existing_polling_dates = set()

        if comprehensive_path.exists():
            # FIXED: Only load the specific columns we need to avoid memory issues
            existing_data = pd.read_csv(
                comprehensive_path,
                usecols=["date", "candidate", "record_type", "polling_average"],
                dtype={
                    "candidate": "string",
                    "record_type": "string",
                },  # Fix mixed type warnings
            )
            existing_data["date"] = pd.to_datetime(existing_data["date"]).dt.date

            # FIXED: Only get unique polling dates, not all records
            polling_records = existing_data[
                existing_data["record_type"] == "historical_polling"
            ]
            if len(polling_records) > 0:
                existing_polling_dates = set(polling_records["date"].unique())
                logger.info(
                    f"Found existing polling data for {len(existing_polling_dates)} unique dates"
                )

        # Determine target date range
        biden_dropout = self.config.biden_dropout_date_parsed
        target_date_obj = (
            pd.to_datetime(target_date).date()
            if target_date
            else pd.Timestamp.today().date()
        )

        # FIXED: Only process data up to target date (no future data)
        date_range_needed = pd.date_range(
            start=biden_dropout, end=target_date_obj, freq="D"
        ).date
        available_dates = [
            d for d in date_range_needed if d < target_date_obj
        ]  # Exclude target date itself

        missing_dates = [
            date for date in available_dates if date not in existing_polling_dates
        ]

        logger.info(
            f"Need data for {len(available_dates)} dates, {len(missing_dates)} are missing"
        )

        # Load and process only missing data
        if len(missing_dates) == 0:
            # All data exists, extract it efficiently
            return self._extract_existing_daily_averages_optimized(
                comprehensive_path, available_dates
            )

        # Process missing dates
        raw_data = self.load_raw_data()
        new_raw_data = raw_data[raw_data["end_date"].isin(missing_dates)].copy()

        if len(new_raw_data) == 0:
            logger.info("No new raw data found for missing dates")
            return self._extract_existing_daily_averages_optimized(
                comprehensive_path, available_dates
            )

        # Process new data
        from .processors import PollingDataProcessor

        processor = PollingDataProcessor(self.config)

        filtered_data = processor.filter_polling_data(new_raw_data)
        new_daily_averages = processor.calculate_daily_averages(filtered_data)

        # Combine with existing data efficiently
        existing_averages = self._extract_existing_daily_averages_optimized(
            comprehensive_path, available_dates
        )

        if len(existing_averages) == 0:
            combined_averages = new_daily_averages
        else:
            combined_averages = pd.concat(
                [existing_averages, new_daily_averages], ignore_index=True
            )
            # FIXED: Only keep unique candidate-date pairs
            combined_averages = combined_averages.drop_duplicates(
                subset=["candidate_name", "end_date"], keep="last"
            )

        # FIXED: Filter to only dates we actually need
        combined_averages = combined_averages[
            combined_averages["end_date"].isin(available_dates)
        ].copy()

        combined_averages.sort_values(
            ["candidate_name", "end_date"], inplace=True, ignore_index=True
        )

        logger.info(
            f"Final daily averages: {len(combined_averages)} total records for {len(available_dates)} dates"
        )
        return combined_averages

    # def load_data_for_incremental_pipeline(self, target_date=None) -> pd.DataFrame:
    #     """Main method to load data for incremental pipeline - compatibility wrapper."""
    #     return self.load_incremental_data(target_date)

    def _extract_existing_daily_averages_optimized(
        self, comprehensive_path, needed_dates
    ) -> pd.DataFrame:
        """Extract daily averages efficiently - PERFORMANCE OPTIMIZED."""
        if not comprehensive_path.exists():
            return pd.DataFrame(columns=["candidate_name", "end_date", "daily_average"])

        # FIXED: Only load what we need
        existing_data = pd.read_csv(
            comprehensive_path,
            usecols=["date", "candidate", "record_type", "polling_average"],
            dtype={"candidate": "string", "record_type": "string"},
        )
        existing_data["date"] = pd.to_datetime(existing_data["date"]).dt.date

        # FIXED: Only get polling records for dates we actually need
        polling_records = existing_data[
            (existing_data["record_type"] == "historical_polling")
            & (existing_data["polling_average"].notna())
            & (existing_data["date"].isin(needed_dates))  # Only needed dates
        ].copy()

        if len(polling_records) == 0:
            return pd.DataFrame(columns=["candidate_name", "end_date", "daily_average"])

        # FIXED: Get unique polling averages per candidate-date
        daily_averages = (
            polling_records.groupby(["candidate", "date"])["polling_average"]
            .first()
            .reset_index()
        )
        daily_averages.columns = ["candidate_name", "end_date", "daily_average"]

        return daily_averages.sort_values(["candidate_name", "end_date"]).reset_index(
            drop=True
        )

    # def _extract_existing_daily_averages(self, comprehensive_path) -> pd.DataFrame:
    #     """Legacy method - kept for compatibility."""
    #     return self._extract_existing_daily_averages_optimized(comprehensive_path, [])
