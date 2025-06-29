# src/data/collectors.py
"""Data collection functions for election forecasting with separate polling cache."""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PollingDataCollector:
    """Handles loading and initial processing of polling data with separate caching."""

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
        """Load data incrementally using separate polling cache for optimal performance."""
        logger.info(f"Loading data incrementally for target date: {target_date}")

        # Path for the separate polling cache
        polling_cache_path = Path("data/polling_averages_cache.csv")

        # Check what polling dates we already have in cache
        existing_polling_dates = self._get_existing_polling_dates(polling_cache_path)

        # Determine target date range
        biden_dropout = self.config.biden_dropout_date_parsed
        target_date_obj = (
            pd.to_datetime(target_date).date()
            if target_date
            else pd.Timestamp.today().date()
        )

        # Only process data up to target date (no future data)
        date_range_needed = pd.date_range(
            start=biden_dropout, end=target_date_obj, freq="D"
        ).date
        available_dates = [
            d for d in date_range_needed if d < target_date_obj
        ]  # Exclude target date itself

        # Check which dates need polling data processing
        missing_polling_dates = [
            date for date in available_dates if date not in existing_polling_dates
        ]

        logger.info(
            f"Need polling data for {len(available_dates)} dates, {len(missing_polling_dates)} need processing"
        )

        # If we have all the data in cache, load from cache
        if len(missing_polling_dates) == 0:
            logger.info("All polling data available in cache, loading efficiently")
            return self._load_from_polling_cache(polling_cache_path, available_dates)

        # Process missing dates from raw data
        logger.info(
            f"Processing {len(missing_polling_dates)} missing dates from raw data"
        )
        raw_data = self.load_raw_data()
        new_raw_data = raw_data[raw_data["end_date"].isin(missing_polling_dates)].copy()

        if len(new_raw_data) == 0:
            logger.info("No new raw polling data found for missing dates")
            return self._load_from_polling_cache(polling_cache_path, available_dates)

        # Process new data
        from .processors import PollingDataProcessor

        processor = PollingDataProcessor(self.config)
        filtered_data = processor.filter_polling_data(new_raw_data)
        new_daily_averages = processor.calculate_daily_averages(filtered_data)

        # Update polling cache with new data
        if len(new_daily_averages) > 0:
            self._update_polling_cache(polling_cache_path, new_daily_averages)

        # Load and return all requested data from cache
        combined_averages = self._load_from_polling_cache(
            polling_cache_path, available_dates
        )

        logger.info(
            f"Final daily averages: {len(combined_averages)} total records for {len(available_dates)} dates"
        )
        return combined_averages

    def _get_existing_polling_dates(self, polling_cache_path: Path) -> set:
        """Get existing polling dates from separate cache file."""
        if not polling_cache_path.exists():
            logger.debug("Polling cache file does not exist")
            return set()

        try:
            cache_data = pd.read_csv(polling_cache_path, usecols=["end_date"])
            cache_data["end_date"] = pd.to_datetime(cache_data["end_date"]).dt.date
            existing_dates = set(cache_data["end_date"].unique())

            logger.info(
                f"Found existing polling data for {len(existing_dates)} unique dates in cache"
            )
            return existing_dates

        except Exception as e:
            logger.warning(f"Error reading polling cache: {e}, treating as empty")
            return set()

    def _load_from_polling_cache(
        self, polling_cache_path: Path, needed_dates: list
    ) -> pd.DataFrame:
        """Load daily averages from polling cache for specified dates."""
        if not polling_cache_path.exists():
            logger.debug("Polling cache file does not exist, returning empty DataFrame")
            return pd.DataFrame(columns=["candidate_name", "end_date", "daily_average"])

        try:
            cache_data = pd.read_csv(polling_cache_path)
            cache_data["end_date"] = pd.to_datetime(cache_data["end_date"]).dt.date

            # Filter to only requested dates
            filtered_data = cache_data[cache_data["end_date"].isin(needed_dates)].copy()

            # Sort consistently
            filtered_data = filtered_data.sort_values(
                ["candidate_name", "end_date"]
            ).reset_index(drop=True)

            logger.debug(
                f"Loaded {len(filtered_data)} records from polling cache for {len(needed_dates)} dates"
            )
            return filtered_data

        except Exception as e:
            logger.error(f"Error loading from polling cache: {e}")
            return pd.DataFrame(columns=["candidate_name", "end_date", "daily_average"])

    def _update_polling_cache(
        self, polling_cache_path: Path, new_daily_averages: pd.DataFrame
    ):
        """Update polling cache with new daily averages."""
        try:
            if polling_cache_path.exists():
                # Load existing cache
                existing_cache = pd.read_csv(polling_cache_path)
                existing_cache["end_date"] = pd.to_datetime(
                    existing_cache["end_date"]
                ).dt.date

                # Combine with new data
                combined_cache = pd.concat(
                    [existing_cache, new_daily_averages], ignore_index=True
                )

                # Remove duplicates (keep latest)
                combined_cache = combined_cache.drop_duplicates(
                    subset=["candidate_name", "end_date"], keep="last"
                )

                logger.debug(
                    f"Updated existing cache: {len(existing_cache)} -> {len(combined_cache)} records"
                )
            else:
                # Create new cache
                combined_cache = new_daily_averages.copy()
                logger.debug(
                    f"Created new polling cache with {len(combined_cache)} records"
                )

            # Sort for consistency
            combined_cache = combined_cache.sort_values(
                ["candidate_name", "end_date"]
            ).reset_index(drop=True)

            # Save updated cache
            combined_cache.to_csv(polling_cache_path, index=False)
            logger.info(
                f"Updated polling cache: added {len(new_daily_averages)} new records"
            )

        except Exception as e:
            logger.error(f"Error updating polling cache: {e}")
            raise

    def _extract_existing_daily_averages_optimized(
        self, comprehensive_path, needed_dates
    ) -> pd.DataFrame:
        """Legacy method - now redirects to polling cache."""
        # This method is kept for compatibility but now uses the polling cache
        polling_cache_path = Path(self.config.polling_cache_path)
        return self._load_from_polling_cache(polling_cache_path, needed_dates)
