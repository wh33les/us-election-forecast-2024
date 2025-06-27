# src/data/collectors.py
"""Data collection functions for election forecasting."""

import pandas as pd
import logging
from pathlib import Path

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

    def load_incremental_raw_data(self, target_date=None) -> tuple:
        """
        Load raw data for ALL missing dates up to target_date.

        Args:
            target_date: Date to load data up to. Finds ALL missing dates from Biden dropout to target_date.

        Returns:
            tuple: (existing_processed_data, new_raw_data, missing_dates_list)
        """
        logger.info("Loading data for incremental update...")

        # Set target date (default to today if not specified)
        if target_date:
            target_date_obj = pd.to_datetime(target_date).date()
        else:
            target_date_obj = pd.Timestamp.today().date()

        logger.info(f"Target date for incremental update: {target_date_obj}")

        # Check if comprehensive dataset exists
        comprehensive_path = Path("data/election_forecast_2024_comprehensive.csv")

        if comprehensive_path.exists():
            logger.info("Loading existing comprehensive dataset...")
            existing_data = pd.read_csv(comprehensive_path)
            existing_data["date"] = pd.to_datetime(existing_data["date"]).dt.date

            # Get all dates with actual polling data (not forecasts)
            polling_data = existing_data[
                existing_data["record_type"] == "historical_polling"
            ]

            if len(polling_data) > 0:
                existing_dates = set(polling_data["date"].unique())
                logger.info(
                    f"Found existing polling data for {len(existing_dates)} dates"
                )
            else:
                existing_dates = set()
                logger.info("No existing polling data found in comprehensive dataset")
        else:
            logger.info("No existing comprehensive dataset - starting fresh")
            existing_data = pd.DataFrame()
            existing_dates = set()

        # Generate full date range from Biden dropout to target date
        biden_dropout = pd.to_datetime(self.config.biden_dropout_date).date()
        full_date_range = pd.date_range(
            start=biden_dropout, end=target_date_obj, freq="D"
        ).date

        # Find ALL missing dates in the range
        missing_dates = [date for date in full_date_range if date not in existing_dates]

        logger.info(f"Date range analysis:")
        logger.info(
            f"  Full range: {biden_dropout} to {target_date_obj} ({len(full_date_range)} days)"
        )
        logger.info(f"  Existing dates: {len(existing_dates)}")
        logger.info(f"  Missing dates: {len(missing_dates)}")

        if len(missing_dates) > 0:
            logger.info(
                f"  Missing date range: {min(missing_dates)} to {max(missing_dates)}"
            )

        # Load raw data
        raw_data = self.load_raw_data()

        # Filter raw data to only include missing dates
        if len(missing_dates) > 0:
            new_raw_data = raw_data[raw_data["end_date"].isin(missing_dates)].copy()
        else:
            new_raw_data = pd.DataFrame()

        logger.info(f"Raw records for missing dates: {len(new_raw_data)}")
        if len(new_raw_data) > 0:
            logger.info(
                f"New data date range: {new_raw_data['end_date'].min()} to {new_raw_data['end_date'].max()}"
            )

        return existing_data, new_raw_data, missing_dates

    def extract_existing_daily_averages(self, existing_data) -> pd.DataFrame:
        """
        Extract daily averages from existing comprehensive dataset.

        Args:
            existing_data: DataFrame from comprehensive dataset

        Returns:
            DataFrame: Daily averages in the format expected by the pipeline
        """
        if len(existing_data) == 0:
            return pd.DataFrame(columns=["candidate_name", "end_date", "daily_average"])

        # Extract records with polling averages
        polling_records = existing_data[
            (existing_data["record_type"] == "historical_polling")
            & (existing_data["polling_average"].notna())
        ].copy()

        if len(polling_records) == 0:
            return pd.DataFrame(columns=["candidate_name", "end_date", "daily_average"])

        # Convert to the format expected by the rest of the pipeline
        daily_averages = polling_records[
            ["candidate", "date", "polling_average"]
        ].copy()
        daily_averages.columns = ["candidate_name", "end_date", "daily_average"]

        # Sort by candidate and date
        daily_averages.sort_values(
            ["candidate_name", "end_date"],
            ascending=[True, True],
            inplace=True,
            ignore_index=True,
        )

        logger.info(f"Extracted {len(daily_averages)} existing daily averages")
        return daily_averages

    def load_data_for_incremental_pipeline(self, target_date=None) -> pd.DataFrame:
        """
        Main method to load data for incremental pipeline.
        Finds ALL missing dates up to target_date and processes them.

        Args:
            target_date: Date to process data up to

        Returns:
            DataFrame: Combined daily averages (existing + new)
        """
        logger.info(f"Loading data incrementally for target date: {target_date}")

        # Load incremental data - now gets ALL missing dates
        existing_data, new_raw_data, missing_dates = self.load_incremental_raw_data(
            target_date
        )

        # Extract existing daily averages
        existing_daily_averages = self.extract_existing_daily_averages(existing_data)

        # If no new data, return existing data
        if len(new_raw_data) == 0:
            if len(missing_dates) == 0:
                logger.info("No missing dates - data is up to date")
            else:
                logger.warning(
                    f"Found {len(missing_dates)} missing dates but no raw data available for them"
                )
                logger.warning(
                    f"Missing dates: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
                )
            return existing_daily_averages

        # Process new data using existing pipeline components
        from .processors import PollingDataProcessor

        processor = PollingDataProcessor(self.config)

        logger.info(f"Processing raw data for {len(missing_dates)} missing dates...")

        # Filter new data to Biden dropout and later (shouldn't be needed but safety check)
        biden_out = pd.to_datetime(self.config.biden_dropout_date).date()
        new_raw_data = new_raw_data[new_raw_data["end_date"] >= biden_out]

        # Process new data
        filtered_new_data = processor.filter_polling_data(new_raw_data)
        new_daily_averages = processor.calculate_daily_averages(filtered_new_data)

        # Combine existing and new daily averages
        if len(existing_daily_averages) == 0:
            combined_daily_averages = new_daily_averages
        elif len(new_daily_averages) == 0:
            combined_daily_averages = existing_daily_averages
        else:
            combined_daily_averages = pd.concat(
                [existing_daily_averages, new_daily_averages], ignore_index=True
            )

        # Remove duplicates and sort
        combined_daily_averages = combined_daily_averages.drop_duplicates(
            subset=["candidate_name", "end_date"], keep="last"
        )
        combined_daily_averages.sort_values(
            ["candidate_name", "end_date"],
            ascending=[True, True],
            inplace=True,
            ignore_index=True,
        )

        logger.info(
            f"Combined daily averages: {len(combined_daily_averages)} total records"
        )
        logger.info(
            f"Date range: {combined_daily_averages['end_date'].min()} to {combined_daily_averages['end_date'].max()}"
        )

        # Show what was actually added
        if len(new_daily_averages) > 0:
            new_dates = sorted(new_daily_averages["end_date"].unique())
            logger.info(
                f"✅ Successfully processed {len(new_dates)} new dates: {new_dates}"
            )

        return combined_daily_averages
