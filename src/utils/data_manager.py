# src/utils/data_manager.py
"""Data management utilities for comprehensive datasets - PERFORMANCE OPTIMIZED."""

import logging
import os
import time
import pandas as pd
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """Handles loading, creating, and saving comprehensive datasets."""

    def __init__(self):
        self.election_day = date(2024, 11, 5)

    def load_or_create_comprehensive_dataset(self):
        """Load existing comprehensive dataset or create new one - OPTIMIZED."""
        dataset_path = Path("data/election_forecast_2024_comprehensive.csv")

        if dataset_path.exists():
            logger.info("Loading existing comprehensive dataset...")

            # FIXED: Specify dtypes to avoid mixed type warnings and improve performance
            dtype_dict = {
                "candidate": "string",
                "record_type": "string",
                "data_source": "string",
                "electoral_winner_model": "string",
                "electoral_winner_baseline": "string",
                "actual_election_winner": "string",
            }

            df = pd.read_csv(dataset_path, dtype=dtype_dict, low_memory=False)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["forecast_run_date"] = pd.to_datetime(df["forecast_run_date"]).dt.date

            logger.info(
                f"Loaded {len(df)} records from {df['forecast_run_date'].nunique()} forecast runs"
            )
            return df
        else:
            logger.info("Creating new comprehensive dataset...")
            return pd.DataFrame()

    def create_comprehensive_forecast_record(
        self,
        training_data,
        all_dates,
        days_till_then,
        fitted_values,
        forecasts,
        baselines,
        forecast_date,
        electoral_results,
        best_params,
        complete_polling_data=None,
    ):
        """Create comprehensive forecast record - OPTIMIZED to avoid data explosion."""
        logger.debug(f"Creating comprehensive record for {forecast_date}")

        comprehensive_records = []

        # FIXED: Only use training data for historical polling to avoid duplicates
        # Don't use complete_polling_data as it causes exponential growth
        polling_data_to_save = training_data.copy()

        # Add historical polling data (ONLY from training period)
        for _, row in polling_data_to_save.iterrows():
            record = self._create_base_record(
                row["end_date"],
                row["candidate_name"],
                forecast_date,
                "historical_polling",
            )
            record["polling_average"] = row["daily_average"]
            comprehensive_records.append(record)

        # Add model fitted values for training dates only
        training_dates = (
            training_data["end_date"].drop_duplicates().sort_values().tolist()
        )

        for i, hist_date in enumerate(training_dates):
            for candidate, candidate_name in [
                ("trump", "Donald Trump"),
                ("harris", "Kamala Harris"),
            ]:
                if i < len(fitted_values[candidate]):
                    record = self._create_base_record(
                        hist_date, candidate_name, forecast_date, "model_fitted"
                    )
                    record.update(
                        {
                            "model_prediction": fitted_values[candidate][i],
                            "alpha": best_params[candidate]["alpha"],
                            "beta": best_params[candidate]["beta"],
                            "mase_score": best_params[candidate]["mase"],
                        }
                    )
                    comprehensive_records.append(record)

        # Add Election Day forecasts only
        election_day_index = next(
            (i for i, d in enumerate(days_till_then) if d == self.election_day), None
        )

        if election_day_index is not None:
            for candidate, candidate_name in [
                ("trump", "Donald Trump"),
                ("harris", "Kamala Harris"),
            ]:
                record = self._create_base_record(
                    self.election_day, candidate_name, forecast_date, "forecast"
                )
                record.update(
                    {
                        "model_prediction": forecasts[candidate][election_day_index],
                        "baseline_prediction": baselines[candidate][election_day_index],
                        "is_forecast": True,
                        "forecast_horizon": len(days_till_then),
                        "alpha": best_params[candidate]["alpha"],
                        "beta": best_params[candidate]["beta"],
                        "mase_score": best_params[candidate]["mase"],
                    }
                )
                record.update(self._get_electoral_data(electoral_results))
                comprehensive_records.append(record)

        df_comprehensive = pd.DataFrame(comprehensive_records)
        df_comprehensive = df_comprehensive.sort_values(
            ["date", "candidate", "record_type"]
        ).reset_index(drop=True)

        logger.debug(f"Created comprehensive record with {len(df_comprehensive)} rows")
        return df_comprehensive

    def _create_base_record(self, date_val, candidate, forecast_date, record_type):
        """Create a base record with common fields."""
        return {
            "date": date_val,
            "candidate": candidate,
            "forecast_run_date": forecast_date,
            "record_type": record_type,
            "data_source": (
                "polling_average"
                if record_type == "historical_polling"
                else "holt_exponential_smoothing"
            ),
            "polling_average": None,
            "model_prediction": None,
            "baseline_prediction": None,
            "days_to_election": (self.election_day - date_val).days,
            "weeks_to_election": round((self.election_day - date_val).days / 7, 1),
            "is_forecast": False,
            "forecast_horizon": None,
            "alpha": None,
            "beta": None,
            "mase_score": None,
            "electoral_winner_model": None,
            "electoral_votes_trump_model": None,
            "electoral_votes_harris_model": None,
            "electoral_winner_baseline": None,
            "electoral_votes_trump_baseline": None,
            "electoral_votes_harris_baseline": None,
        }

    def _get_electoral_data(self, electoral_results):
        """Extract electoral data for election day records."""
        return {
            "electoral_winner_model": electoral_results["model"]["winner"],
            "electoral_votes_trump_model": electoral_results["model"][
                "trump_electoral_votes"
            ],
            "electoral_votes_harris_model": electoral_results["model"][
                "harris_electoral_votes"
            ],
            "electoral_winner_baseline": electoral_results["baseline"]["winner"],
            "electoral_votes_trump_baseline": electoral_results["baseline"][
                "trump_electoral_votes"
            ],
            "electoral_votes_harris_baseline": electoral_results["baseline"][
                "harris_electoral_votes"
            ],
        }

    def save_comprehensive_dataset(self, comprehensive_dataset):
        """Save comprehensive dataset with retry logic for file locks."""
        if len(comprehensive_dataset) == 0:
            logger.warning("No records to save")
            return comprehensive_dataset

        comprehensive_dataset = comprehensive_dataset.copy()
        comprehensive_dataset["actual_election_winner"] = "Donald Trump"

        # Calculate prediction accuracy for forecasts
        comprehensive_dataset["prediction_correct"] = comprehensive_dataset.apply(
            lambda row: (
                (row["model_prediction"] > 50) == (row["candidate"] == "Donald Trump")
                if pd.notna(row["model_prediction"]) and row["is_forecast"]
                else None
            ),
            axis=1,
        )

        comprehensive_dataset = comprehensive_dataset.sort_values(
            ["date", "candidate", "record_type"]
        ).reset_index(drop=True)

        output_path = Path("data/election_forecast_2024_comprehensive.csv")

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Retry logic for file locks
        max_retries = 5
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Try to save the file
                comprehensive_dataset.to_csv(output_path, index=False)
                logger.info(
                    f"💾 Updated comprehensive dataset: {len(comprehensive_dataset)} records"
                )
                return comprehensive_dataset

            except PermissionError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"File locked, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Final attempt failed, try backup filename
                    backup_path = output_path.with_suffix(
                        f".backup_{int(time.time())}.csv"
                    )
                    try:
                        comprehensive_dataset.to_csv(backup_path, index=False)
                        logger.warning(f"💾 Saved to backup file: {backup_path}")
                        return comprehensive_dataset
                    except Exception as backup_error:
                        logger.error(f"Failed to save even to backup: {backup_error}")
                        raise e

            except Exception as e:
                logger.error(f"Unexpected error saving dataset: {e}")
                raise e

        return comprehensive_dataset

    def create_historical_data_for_plotting(self, comprehensive_dataset, forecast_date):
        """Create historical forecast data for plotting functions."""
        historical_forecasts = comprehensive_dataset[
            (comprehensive_dataset["date"] == self.election_day)
            & (comprehensive_dataset["record_type"] == "forecast")
            & (comprehensive_dataset["forecast_run_date"] <= forecast_date)
        ].copy()

        if len(historical_forecasts) == 0:
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        historical_data = []
        for _, row in historical_forecasts.iterrows():
            historical_data.append(
                {
                    "date": row["forecast_run_date"],
                    "candidate": row["candidate"],
                    "model": row["model_prediction"],
                    "baseline": row["baseline_prediction"],
                }
            )

        return pd.DataFrame(historical_data)
