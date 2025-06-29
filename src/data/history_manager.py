# src/data/history_manager.py
"""Complete forecast history management - records, storage, and retrieval."""

import logging
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HistoryManager:
    """Handles all forecast history operations - creating, saving, loading, and processing historical records."""

    def __init__(self, data_config):
        self.config = data_config
        self.election_day = data_config.election_day_parsed

    def check_data_status(self):
        """Log status of all key data files."""
        history_path = Path(self.config.forecast_history_path)
        polling_cache_path = Path(self.config.polling_cache_path)

        history_exists = history_path.exists()
        polling_cache_exists = polling_cache_path.exists()

        logger.info("Data file status:")
        logger.info(
            f"   - Forecast history: {'EXISTS' if history_exists else '❌ NOT FOUND'} ({history_path})"
        )
        logger.info(
            f"   - Polling averages cache: {'EXISTS' if polling_cache_exists else '❌ NOT FOUND'} ({polling_cache_path})"
        )

        if history_exists and polling_cache_exists:
            logger.info("   Using existing data files for faster processing")
        elif history_exists or polling_cache_exists:
            logger.info("   Partial cache available, will rebuild missing components")
        else:
            logger.info("   No cache files found, will build from scratch")

    def load_forecast_history(self) -> pd.DataFrame:
        """Load existing forecast history dataset or create empty one."""
        history_path = Path(self.config.forecast_history_path)

        if history_path.exists():
            logger.info("Loading existing forecast history dataset...")
            df = pd.read_csv(history_path, low_memory=False)
            df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
            logger.info(
                f"Loaded {len(df)} records from {df['forecast_date'].nunique()} forecast runs"
            )
            return df
        else:
            logger.info("Creating new forecast history dataset...")
            return pd.DataFrame()

    def initialize_data_pipeline(self) -> pd.DataFrame:
        """Check data status and load forecast history - main initialization function."""
        self.check_data_status()
        return self.load_forecast_history()

    def create_forecast_record(
        self,
        training_data: pd.DataFrame,
        all_dates: list,
        days_till_then: list,
        fitted_values: Dict,
        forecasts: Dict,
        baselines: Dict,
        forecast_date: date,
        electoral_results: Optional[Dict],
        best_params: Dict,
        complete_polling_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create clean forecast record for history dataset."""
        logger.debug(f"Creating forecast record for {forecast_date}")

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
        history_record = pd.DataFrame(records)
        history_record = history_record.sort_values(
            ["forecast_date", "candidate"]
        ).reset_index(drop=True)

        logger.debug(
            f"Created clean record with {len(history_record)} rows, {len(history_record.columns)} columns"
        )
        return history_record

    def save_forecast_history(
        self, dataset: pd.DataFrame, filepath: Optional[Path] = None
    ) -> pd.DataFrame:
        """Save forecast history dataset to CSV."""
        if filepath is None:
            filepath = Path(self.config.forecast_history_path)

        dataset.to_csv(filepath, index=False)
        logger.info(
            f"Saved clean forecast history with {len(dataset)} records "
            f"({len(dataset.columns)} columns) to {filepath}"
        )

        # Log the clean schema
        if not dataset.empty:
            logger.debug(f"Clean schema: {list(dataset.columns)}")

        # Return the dataset so it can be chained
        return dataset

    def create_historical_data_for_plotting(
        self, forecast_history: pd.DataFrame, forecast_date: date
    ) -> pd.DataFrame:
        """Create historical forecast data for plotting with streamlined schema."""
        # Add defensive programming for None or empty dataset
        if forecast_history is None:
            logger.warning("forecast_history is None, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        if forecast_history.empty:
            logger.warning("forecast_history is empty, returning empty DataFrame")
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        # Check if required columns exist (streamlined schema)
        required_columns = [
            "forecast_date",
            "candidate",
            "model_prediction",
            "baseline_prediction",
        ]
        missing_columns = [
            col for col in required_columns if col not in forecast_history.columns
        ]
        if missing_columns:
            logger.warning(
                f"Missing columns {missing_columns}, returning empty DataFrame"
            )
            return pd.DataFrame(columns=["date", "candidate", "model", "baseline"])

        try:
            # Filter for Election Day predictions from different forecast dates
            historical_forecasts = forecast_history[
                (forecast_history["candidate"].isin(["Donald Trump", "Kamala Harris"]))
                & (forecast_history["forecast_date"] <= forecast_date)
                & (forecast_history["model_prediction"].notna())
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
