# src/utils/data_manager.py
"""Data management utilities for comprehensive datasets - CLEAN STREAMLINED VERSION."""

import logging
import os
import time
import pandas as pd
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """Handles loading, creating, and saving streamlined comprehensive datasets."""

    def __init__(self, data_config=None):
        self.election_day = (
            data_config.election_day_parsed if data_config else date(2024, 11, 5)
        )
        self.data_config = data_config

    def load_or_create_comprehensive_dataset(self):
        """Load existing comprehensive dataset or create new one - STREAMLINED."""
        # Use config path if available, otherwise fallback to hardcoded path
        if self.data_config:
            dataset_path = Path(self.data_config.comprehensive_dataset_path)
        else:
            dataset_path = Path("data/election_forecast_2024_comprehensive.csv")

        if dataset_path.exists():
            logger.info("Loading existing streamlined comprehensive dataset...")

            # Load with low_memory=False for better type inference
            df = pd.read_csv(dataset_path, low_memory=False)
            df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date

            logger.info(
                f"Loaded {len(df)} records from {df['forecast_date'].nunique()} forecast runs"
            )
            return df
        else:
            logger.info("Creating new streamlined comprehensive dataset...")
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
        """Create clean streamlined forecast record - just Trump and Harris, no SUMMARY."""
        logger.debug(f"Creating streamlined record for {forecast_date}")

        records = []

        # Get Election Day forecast index
        election_day_index = next(
            (i for i, d in enumerate(days_till_then) if d == self.election_day), None
        )

        if election_day_index is not None:
            for candidate, candidate_name in [
                ("trump", "Donald Trump"),
                ("harris", "Kamala Harris"),
            ]:
                # Core forecast data (8 essential columns)
                record = {
                    "forecast_date": forecast_date,
                    "candidate": candidate_name,
                    "model_prediction": forecasts[candidate][election_day_index],
                    "baseline_prediction": baselines[candidate][election_day_index],
                    "alpha": best_params[candidate]["alpha"],
                    "beta": best_params[candidate]["beta"],
                    "mase_score": best_params[candidate]["mase"],
                    "baseline_mase_score": best_params[candidate].get(
                        "baseline_mase", None
                    ),
                }

                # Add electoral data only for Election Day (+2 columns = 10 total)
                if forecast_date == self.election_day:
                    record.update(
                        {
                            "electoral_winner_model": electoral_results["model"][
                                "winner"
                            ],
                            "electoral_winner_baseline": electoral_results["baseline"][
                                "winner"
                            ],
                        }
                    )
                else:
                    # For non-Election Day, fill with None to maintain consistent schema
                    record.update(
                        {
                            "electoral_winner_model": None,
                            "electoral_winner_baseline": None,
                        }
                    )

                records.append(record)

        # Create DataFrame - just Trump and Harris records
        df_comprehensive = pd.DataFrame(records)
        df_comprehensive = df_comprehensive.sort_values(
            ["forecast_date", "candidate"]
        ).reset_index(drop=True)

        logger.debug(
            f"Created clean record with {len(df_comprehensive)} rows, {len(df_comprehensive.columns)} columns"
        )
        return df_comprehensive

    def create_historical_data_for_plotting(self, comprehensive_dataset, forecast_date):
        """Create historical forecast data for plotting with streamlined schema."""
        # Add defensive programming for None or empty dataset
        if comprehensive_dataset is None:
            logger.warning("comprehensive_dataset is None, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        if comprehensive_dataset.empty:
            logger.warning("comprehensive_dataset is empty, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        # Check if required columns exist (streamlined schema)
        required_columns = [
            "forecast_date",
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
            # Filter for Election Day predictions from different forecast dates
            historical_forecasts = comprehensive_dataset[
                (
                    comprehensive_dataset["candidate"].isin(
                        ["Donald Trump", "Kamala Harris"]
                    )
                )
                & (comprehensive_dataset["forecast_date"] <= forecast_date)
                & (comprehensive_dataset["model_prediction"].notna())
            ].copy()

            if len(historical_forecasts) == 0:
                logger.debug("No historical forecasts found for plotting")
                return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

            historical_data = []
            for _, row in historical_forecasts.iterrows():
                historical_data.append(
                    {
                        "date": row["forecast_date"],
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
        """Save streamlined comprehensive dataset to CSV."""
        if filepath is None:
            # Use config path if available, otherwise fallback to hardcoded path
            if self.data_config:
                filepath = Path(self.data_config.comprehensive_dataset_path)
            else:
                filepath = Path("data/election_forecast_2024_comprehensive.csv")

        dataset.to_csv(filepath, index=False)
        logger.info(
            f"Saved clean comprehensive dataset with {len(dataset)} records "
            f"({len(dataset.columns)} columns) to {filepath}"
        )

        # Log the clean schema
        if not dataset.empty:
            logger.debug(f"Clean schema: {list(dataset.columns)}")

        # Return the dataset so it can be chained
        return dataset
