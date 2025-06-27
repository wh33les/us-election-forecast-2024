# src/utils/data_manager.py
"""Data management utilities for comprehensive datasets."""

import logging
import pandas as pd
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class DataManager:
    """Handles loading, creating, and saving comprehensive datasets."""

    def load_or_create_comprehensive_dataset(self):
        """Load existing comprehensive dataset or create new one."""
        dataset_path = Path("data/election_forecast_2024_comprehensive.csv")

        if dataset_path.exists():
            logger.info("Loading existing comprehensive dataset...")
            df = pd.read_csv(dataset_path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["forecast_run_date"] = pd.to_datetime(df["forecast_run_date"]).dt.date

            existing_runs = df["forecast_run_date"].unique()
            logger.info(
                f"Loaded {len(df)} existing records from {len(existing_runs)} forecast runs"
            )
            logger.debug(f"Existing forecast runs: {sorted(existing_runs)}")
            logger.debug(f"Date range: {df['date'].min()} to {df['date'].max()}")
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
    ):
        """Create comprehensive forecast record combining all data."""
        logger.debug(
            f"Creating comprehensive record for {forecast_date}: training_data shape={training_data.shape}, forecast periods={len(days_till_then)}"
        )

        comprehensive_records = []
        election_day = date(2024, 11, 5)

        # Add historical polling data (from training set only)
        for _, row in training_data.iterrows():
            record = self._create_base_record(
                row, forecast_date, election_day, "historical_polling"
            )
            record.update(
                {
                    "polling_average": row["daily_average"],
                    "model_prediction": None,
                    "baseline_prediction": None,
                }
            )
            comprehensive_records.append(record)

        # Add model fitted values for historical dates (training period only)
        historical_dates = (
            training_data["end_date"].drop_duplicates().sort_values().tolist()
        )

        for i, hist_date in enumerate(historical_dates):
            for candidate, candidate_name in [
                ("trump", "Donald Trump"),
                ("harris", "Kamala Harris"),
            ]:
                if i < len(fitted_values[candidate]):
                    record = self._create_base_record_with_date(
                        hist_date,
                        candidate_name,
                        forecast_date,
                        election_day,
                        "model_fitted",
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

        # Add forecasts
        for i, forecast_day in enumerate(days_till_then):
            days_ahead = i + 1
            is_election_day = forecast_day == election_day

            for candidate, candidate_name in [
                ("trump", "Donald Trump"),
                ("harris", "Kamala Harris"),
            ]:
                record = self._create_base_record_with_date(
                    forecast_day,
                    candidate_name,
                    forecast_date,
                    election_day,
                    "forecast",
                )
                record.update(
                    {
                        "model_prediction": forecasts[candidate][i],
                        "baseline_prediction": baselines[candidate][i],
                        "is_forecast": True,
                        "forecast_horizon": days_ahead,
                        "alpha": best_params[candidate]["alpha"],
                        "beta": best_params[candidate]["beta"],
                        "mase_score": best_params[candidate]["mase"],
                    }
                )

                if is_election_day:
                    record.update(self._get_electoral_data(electoral_results))

                comprehensive_records.append(record)

        df_comprehensive = pd.DataFrame(comprehensive_records)
        df_comprehensive = df_comprehensive.sort_values(
            ["date", "candidate", "record_type"]
        ).reset_index(drop=True)

        logger.debug(f"Created comprehensive record with {len(df_comprehensive)} rows")
        return df_comprehensive

    def _create_base_record(self, row, forecast_date, election_day, record_type):
        """Create a base record with common fields."""
        return {
            "date": row["end_date"],
            "candidate": row["candidate_name"],
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
            "days_to_election": (election_day - row["end_date"]).days,
            "weeks_to_election": round((election_day - row["end_date"]).days / 7, 1),
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

    def _create_base_record_with_date(
        self, date_val, candidate, forecast_date, election_day, record_type
    ):
        """Create a base record with specified date and candidate."""
        return {
            "date": date_val,
            "candidate": candidate,
            "forecast_run_date": forecast_date,
            "record_type": record_type,
            "data_source": "holt_exponential_smoothing",
            "polling_average": None,
            "model_prediction": None,
            "baseline_prediction": None,
            "days_to_election": (election_day - date_val).days,
            "weeks_to_election": round((election_day - date_val).days / 7, 1),
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
        """Save the comprehensive dataset to CSV with metadata."""
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

        output_path = "data/election_forecast_2024_comprehensive.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        comprehensive_dataset.to_csv(output_path, index=False)

        logger.info(
            f"💾 Updated comprehensive dataset: {len(comprehensive_dataset)} records"
        )
        logger.debug(
            f"   📅 Date range: {comprehensive_dataset['date'].min()} to {comprehensive_dataset['date'].max()}"
        )
        logger.debug(
            f"   🔮 Forecast runs: {comprehensive_dataset['forecast_run_date'].nunique()}"
        )

        return comprehensive_dataset

    def create_historical_data_for_plotting(self, comprehensive_dataset, forecast_date):
        """Create historical forecast data in the format expected by plotting functions."""
        election_day = date(2024, 11, 5)
        historical_forecasts = comprehensive_dataset[
            (comprehensive_dataset["date"] == election_day)
            & (comprehensive_dataset["record_type"] == "forecast")
            & (comprehensive_dataset["forecast_run_date"] <= forecast_date)
        ].copy()

        if len(historical_forecasts) == 0:
            logger.debug("No historical forecasts available for plotting")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        historical_data = []
        for _, row in historical_forecasts.iterrows():
            record = {
                "date": row["forecast_run_date"],  # The date the forecast was made
                "candidate": row["candidate"],
                "model": row["model_prediction"],
                "baseline": row["baseline_prediction"],
            }
            historical_data.append(record)

        historical_df = pd.DataFrame(historical_data)
        logger.debug(
            f"Created historical plotting data with {len(historical_df)} records"
        )

        return historical_df
