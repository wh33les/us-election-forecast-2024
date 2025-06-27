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
        """Create OPTIMIZED comprehensive forecast record (much smaller)."""
        logger.debug(f"Creating optimized record for {forecast_date}")

        comprehensive_records = []

        # OPTIMIZATION 1: Store only Election Day forecasts (not all forecast dates)
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

        # OPTIMIZATION 2: Skip historical polling and fitted values entirely
        # This reduces file size by ~80%

        # OPTIMIZATION 3: Store only essential summary data
        # Add one summary record per forecast run
        summary_record = {
            "date": forecast_date,
            "candidate": "SUMMARY",
            "forecast_run_date": forecast_date,
            "record_type": "run_summary",
            "data_source": "pipeline_summary",
            "polling_average": None,
            "model_prediction": None,
            "baseline_prediction": None,
            "days_to_election": (self.election_day - forecast_date).days,
            "weeks_to_election": round((self.election_day - forecast_date).days / 7, 1),
            "is_forecast": False,
            "forecast_horizon": len(days_till_then),
            "alpha": None,
            "beta": None,
            "mase_score": None,
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
            "actual_election_winner": None,
            "prediction_correct": None,
        }
        comprehensive_records.append(summary_record)

        df_comprehensive = pd.DataFrame(comprehensive_records)
        df_comprehensive = df_comprehensive.sort_values(
            ["date", "candidate", "record_type"]
        ).reset_index(drop=True)

        logger.debug(f"Created optimized record with {len(df_comprehensive)} rows")
        return df_comprehensive

    def _create_base_record(
        self, record_date, candidate_name, forecast_date, record_type
    ):
        """Create base record with common fields."""
        return {
            "date": record_date,
            "candidate": candidate_name,
            "forecast_run_date": forecast_date,
            "record_type": record_type,
            "data_source": "holt_exponential_smoothing",
            "polling_average": None,
            "days_to_election": (self.election_day - record_date).days,
            "weeks_to_election": round((self.election_day - record_date).days / 7, 1),
        }

    def _get_electoral_data(self, electoral_results):
        """Extract electoral data from results."""
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
            "actual_election_winner": None,
            "prediction_correct": None,
        }

    def create_historical_data_for_plotting(self, comprehensive_dataset, forecast_date):
        """Create historical forecast data for plotting functions."""
        # Add defensive programming for None or empty dataset
        if comprehensive_dataset is None:
            logger.warning("comprehensive_dataset is None, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        if comprehensive_dataset.empty:
            logger.warning("comprehensive_dataset is empty, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        # Check if required columns exist
        required_columns = [
            "date",
            "record_type",
            "forecast_run_date",
            "candidate",
            "model_prediction",
            "baseline_prediction",
        ]
        missing_columns = [
            col for col in required_columns if col not in comprehensive_dataset.columns
        ]
        if missing_columns:
            logger.warning(
                f"Missing columns {missing_columns}, returning empty DataFrame"
            )
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        try:
            historical_forecasts = comprehensive_dataset[
                (comprehensive_dataset["date"] == self.election_day)
                & (comprehensive_dataset["record_type"] == "forecast")
                & (comprehensive_dataset["forecast_run_date"] <= forecast_date)
            ].copy()

            if len(historical_forecasts) == 0:
                logger.debug("No historical forecasts found for plotting")
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

            logger.debug(
                f"Created historical data for plotting with {len(historical_data)} records"
            )
            return pd.DataFrame(historical_data)

        except Exception as e:
            logger.error(f"Error creating historical data for plotting: {e}")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

    def save_comprehensive_dataset(self, dataset, filepath=None):
        """Save comprehensive dataset to CSV."""
        if filepath is None:
            filepath = Path("data/election_forecast_2024_comprehensive.csv")

        dataset.to_csv(filepath, index=False)
        logger.info(
            f"Saved comprehensive dataset with {len(dataset)} records to {filepath}"
        )

        # Return the dataset so it can be chained
        return dataset

    def append_to_comprehensive_dataset(self, existing_dataset, new_records):
        """Append new records to existing comprehensive dataset."""
        if existing_dataset.empty:
            return new_records

        combined_dataset = pd.concat([existing_dataset, new_records], ignore_index=True)
        combined_dataset = combined_dataset.sort_values(
            ["forecast_run_date", "date", "candidate", "record_type"]
        ).reset_index(drop=True)

        logger.debug(f"Combined dataset now has {len(combined_dataset)} records")
        return combined_dataset
