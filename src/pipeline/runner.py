# src/pipeline/runner.py
"""Main forecast runner that orchestrates the election forecasting pipeline."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional

from src.data.collectors import PollingDataCollector
from src.data.processors import PollingDataProcessor
from src.models.holt_forecaster import HoltElectionForecaster
from src.models.electoral_calculator import ElectoralCollegeCalculator
from src.visualization.plotting import ElectionPlotter
from src.utils.data_manager import DataManager
from src.utils.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


class ForecastRunner:
    """Orchestrates the complete election forecasting pipeline."""

    def __init__(self, model_config, data_config, verbose=False, debug=False):
        self.model_config = model_config
        self.data_config = data_config
        self.verbose = verbose
        self.debug = debug
        self.election_day = datetime(2024, 11, 5).date()

        # Initialize components
        self.collector = PollingDataCollector(data_config)
        self.processor = PollingDataProcessor(model_config)
        self.calculator = ElectoralCollegeCalculator(model_config)
        self.plotter = ElectionPlotter(data_config)
        self.data_manager = DataManager()
        self.formatter = ResultFormatter(verbose, debug)

    def run_forecasts(self, forecast_dates: List[date]) -> bool:
        """Run forecasts for all specified dates."""
        logger.info("Starting Rolling Election Forecast 2024 pipeline...")

        # Determine loading strategy
        comprehensive_path = Path("data/election_forecast_2024_comprehensive.csv")
        use_incremental = comprehensive_path.exists()
        logger.info(
            f"Using {'incremental' if use_incremental else 'full'} data loading"
        )

        # Initialize dataset
        comprehensive_dataset = self.data_manager.load_or_create_comprehensive_dataset()
        logger.info(f"Starting forecasts for {len(forecast_dates)} days")

        # Run forecasts
        success_count = 0
        for i, forecast_date in enumerate(forecast_dates):
            try:
                self.formatter.log_forecast_start(
                    i + 1, len(forecast_dates), forecast_date
                )

                # Load data
                daily_averages = self._load_data_for_date(
                    forecast_date, use_incremental
                )
                if daily_averages is None or daily_averages.empty:
                    logger.warning(f"No data available for {forecast_date}")
                    continue

                # Execute forecast
                result = self._run_single_forecast(
                    forecast_date, daily_averages, comprehensive_dataset
                )

                if result is not None:
                    comprehensive_dataset = result
                    success_count += 1
                    self.formatter.log_forecast_success(forecast_date)
                else:
                    self.formatter.log_forecast_skip(forecast_date)

            except Exception as e:
                self.formatter.log_forecast_error(forecast_date, e)
                continue

        logger.info(
            f"Completed {success_count}/{len(forecast_dates)} forecasts successfully"
        )
        return success_count > 0

    def _load_data_for_date(
        self, forecast_date: date, use_incremental: bool
    ) -> Optional[pd.DataFrame]:
        """Load data for a specific forecast date."""
        if use_incremental:
            return self.collector.load_incremental_data(forecast_date)
        else:
            return self._load_full_data()

    def _load_full_data(self) -> pd.DataFrame:
        """Load and process all raw data."""
        logger.info("Loading all raw data...")
        raw_data = self.collector.load_raw_data()

        # Filter to Biden drop-out date and later
        biden_out = datetime(2024, 7, 21).date()
        raw_data = raw_data[raw_data["end_date"] >= biden_out]
        logger.info(
            f"Filtered to data from {biden_out} onwards: {len(raw_data)} records"
        )

        # Process the data
        filtered_data = self.processor.filter_polling_data(raw_data)
        return self.processor.calculate_daily_averages(filtered_data)

    def _run_single_forecast(
        self,
        forecast_date: date,
        daily_averages: pd.DataFrame,
        comprehensive_dataset: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Run forecast for a single date."""
        # Use only data available before forecast date
        available_data = daily_averages[
            daily_averages["end_date"] < forecast_date
        ].copy()
        logger.info(f"Using {len(available_data)} records before {forecast_date}")

        # Split by candidate
        trump_data, harris_data = self.processor.split_by_candidate(available_data)

        if len(trump_data) < 10 or len(harris_data) < 10:
            logger.warning(f"Insufficient data for {forecast_date}, skipping")
            return None

        # Prepare train/holdout split
        test_size = self.model_config.test_size
        train_cutoff_date = forecast_date - timedelta(days=test_size)

        trump_train = trump_data[trump_data["end_date"] < train_cutoff_date].copy()
        harris_train = harris_data[harris_data["end_date"] < train_cutoff_date].copy()
        trump_holdout = trump_data[trump_data["end_date"] >= train_cutoff_date].copy()
        harris_holdout = harris_data[
            harris_data["end_date"] >= train_cutoff_date
        ].copy()

        if len(trump_train) < 10 or len(harris_train) < 10:
            logger.warning("Insufficient training data, skipping")
            return None

        if self.verbose:
            logger.info(
                f"   📊 Train: {len(trump_train)} days, Holdout: {len(trump_holdout)} days"
            )

        # Execute forecasting
        forecast_results = self._execute_forecast(
            trump_train,
            harris_train,
            trump_holdout,
            harris_holdout,
            forecast_date,
            train_cutoff_date,
        )

        if forecast_results is None:
            return None

        # Update dataset and create visualizations
        comprehensive_dataset = self._update_and_visualize(
            comprehensive_dataset, forecast_results, forecast_date, daily_averages
        )

        # Log results
        self.formatter.log_forecast_results(
            forecast_results, forecast_date, self.election_day
        )
        return comprehensive_dataset

    def _execute_forecast(
        self,
        trump_train: pd.DataFrame,
        harris_train: pd.DataFrame,
        trump_holdout: pd.DataFrame,
        harris_holdout: pd.DataFrame,
        forecast_date: date,
        train_cutoff_date: date,
    ) -> Optional[Dict]:
        """Execute the forecasting models."""
        days_to_election = (self.election_day - forecast_date).days
        logger.info(f"Forecasting {days_to_election} days until election")

        # Train models
        forecaster = HoltElectionForecaster(self.model_config)
        x_train = pd.Series(range(len(trump_train)))

        logger.info("Running hyperparameter optimization...")
        best_params = forecaster.grid_search_hyperparameters(
            trump_train, harris_train, x_train
        )

        logger.info("Fitting final models...")
        fitted_models = forecaster.fit_final_models(trump_train, harris_train)

        # Calculate horizons
        holdout_dates = pd.date_range(
            start=train_cutoff_date, end=forecast_date, inclusive="left"
        ).date
        forecast_dates_list = pd.date_range(
            start=forecast_date, end=self.election_day, inclusive="both"
        ).date.tolist()

        holdout_horizon = len(holdout_dates)
        total_horizon = holdout_horizon + len(forecast_dates_list)

        logger.info(f"Generating predictions: {total_horizon} periods total")

        # Generate predictions
        all_predictions = forecaster.forecast(total_horizon)
        all_baselines = forecaster.generate_baseline_forecasts(
            trump_train, harris_train, total_horizon
        )

        # Split predictions
        forecasts, baselines, holdout_predictions, holdout_baselines = (
            self._split_predictions(all_predictions, all_baselines, holdout_horizon)
        )

        fitted_values = forecaster.get_fitted_values()

        # Log holdout performance
        if holdout_predictions["trump"].size > 0 and not trump_holdout.empty:
            self._log_holdout_performance(
                holdout_predictions, trump_holdout, harris_holdout
            )

        # Calculate electoral outcomes
        electoral_results = self._calculate_electoral_outcomes(
            forecasts, baselines, forecast_date
        )

        # Prepare plotting data
        plotting_data = self._prepare_plotting_data(
            trump_train,
            harris_train,
            trump_holdout,
            harris_holdout,
            fitted_values,
            holdout_predictions,
            forecast_dates_list,
            holdout_baselines,
        )

        return {
            "trump_train": trump_train,
            "harris_train": harris_train,
            "fitted_values": fitted_values,
            "forecasts": forecasts,
            "baselines": baselines,
            "days_till_then": forecast_dates_list,
            "electoral_results": electoral_results,
            "best_params": best_params,
            "train_cutoff_date": train_cutoff_date,
            **plotting_data,
        }

    def _split_predictions(
        self, all_predictions: Dict, all_baselines: Dict, holdout_horizon: int
    ) -> tuple:
        """Split predictions into holdout and forecast periods."""
        if holdout_horizon > 0:
            forecasts = {
                "trump": all_predictions["trump"][holdout_horizon:],
                "harris": all_predictions["harris"][holdout_horizon:],
            }
            baselines = {
                "trump": all_baselines["trump"][holdout_horizon:],
                "harris": all_baselines["harris"][holdout_horizon:],
            }
            holdout_predictions = {
                "trump": all_predictions["trump"][:holdout_horizon],
                "harris": all_predictions["harris"][:holdout_horizon],
            }
            holdout_baselines = {
                "trump": all_baselines["trump"][:holdout_horizon],
                "harris": all_baselines["harris"][:holdout_horizon],
            }
        else:
            forecasts = all_predictions
            baselines = all_baselines
            holdout_predictions = {"trump": np.array([]), "harris": np.array([])}
            holdout_baselines = None

        return forecasts, baselines, holdout_predictions, holdout_baselines

    def _log_holdout_performance(
        self,
        holdout_predictions: Dict,
        trump_holdout: pd.DataFrame,
        harris_holdout: pd.DataFrame,
    ):
        """Log holdout validation performance."""
        trump_actual = trump_holdout["daily_average"].mean()
        harris_actual = harris_holdout["daily_average"].mean()
        trump_pred = np.mean(holdout_predictions["trump"])
        harris_pred = np.mean(holdout_predictions["harris"])

        trump_error = abs(trump_actual - trump_pred)
        harris_error = abs(harris_actual - harris_pred)

        if self.debug:
            logger.debug(
                f"Holdout validation: Trump error={trump_error:.2f}%, Harris error={harris_error:.2f}%"
            )

    def _calculate_electoral_outcomes(
        self, forecasts: Dict, baselines: Dict, forecast_date: date
    ) -> Dict:
        """Calculate electoral college outcomes."""
        if (
            forecast_date == self.election_day
            and forecasts["trump"].size > 0
            and forecasts["harris"].size > 0
        ):
            electoral_data = pd.DataFrame(
                [
                    {
                        "candidate_name": "Donald Trump",
                        "end_date": self.election_day,
                        "daily_average": None,
                        "model": forecasts["trump"][-1],
                        "drift_pred": baselines["trump"][-1],
                    },
                    {
                        "candidate_name": "Kamala Harris",
                        "end_date": self.election_day,
                        "daily_average": None,
                        "model": forecasts["harris"][-1],
                        "drift_pred": baselines["harris"][-1],
                    },
                ]
            )

            logger.info("Calculating electoral college outcomes...")
            return self.calculator.calculate_all_outcomes(electoral_data)

        # Fallback for non-Election Day
        return self._create_fallback_electoral_results(forecasts, baselines)

    def _create_fallback_electoral_results(
        self, forecasts: Dict, baselines: Dict
    ) -> Dict:
        """Create fallback electoral results."""
        trump_pred = forecasts["trump"][-1] if forecasts["trump"].size > 0 else 0
        harris_pred = forecasts["harris"][-1] if forecasts["harris"].size > 0 else 0
        trump_base = baselines["trump"][-1] if baselines["trump"].size > 0 else 0
        harris_base = baselines["harris"][-1] if baselines["harris"].size > 0 else 0

        return {
            "model": {
                "winner": "N/A (interim forecast)",
                "trump_electoral_votes": None,
                "harris_electoral_votes": None,
                "trump_states": [],
                "harris_states": [],
                "trump_vote_pct": trump_pred,
                "harris_vote_pct": harris_pred,
            },
            "baseline": {
                "winner": "N/A (interim forecast)",
                "trump_electoral_votes": None,
                "harris_electoral_votes": None,
                "trump_states": [],
                "harris_states": [],
                "trump_vote_pct": trump_base,
                "harris_vote_pct": harris_base,
            },
        }

    def _prepare_plotting_data(
        self,
        trump_train: pd.DataFrame,
        harris_train: pd.DataFrame,
        trump_holdout: pd.DataFrame,
        harris_holdout: pd.DataFrame,
        fitted_values: Dict,
        holdout_predictions: Dict,
        forecast_dates_list: List,
        holdout_baselines: Optional[Dict],
    ) -> Dict:
        """Prepare data for plotting functions."""
        # Create complete historical datasets
        trump_dfs = [df for df in [trump_train, trump_holdout] if not df.empty]
        harris_dfs = [df for df in [harris_train, harris_holdout] if not df.empty]

        trump_complete = (
            pd.concat(trump_dfs, ignore_index=True).sort_values("end_date")
            if trump_dfs
            else pd.DataFrame()
        )
        harris_complete = (
            pd.concat(harris_dfs, ignore_index=True).sort_values("end_date")
            if harris_dfs
            else pd.DataFrame()
        )

        # Remove duplicates
        if not trump_complete.empty:
            trump_complete = trump_complete.drop_duplicates(
                subset=["end_date"], keep="first"
            ).reset_index(drop=True)
        if not harris_complete.empty:
            harris_complete = harris_complete.drop_duplicates(
                subset=["end_date"], keep="first"
            ).reset_index(drop=True)

        # Create complete fitted values
        complete_fitted_values = {
            "trump": list(fitted_values["trump"]) + list(holdout_predictions["trump"]),
            "harris": list(fitted_values["harris"])
            + list(holdout_predictions["harris"]),
        }

        # Ensure fitted values match dataset lengths
        complete_fitted_values = {
            "trump": (
                complete_fitted_values["trump"][: len(trump_complete)]
                if not trump_complete.empty
                else []
            ),
            "harris": (
                complete_fitted_values["harris"][: len(harris_complete)]
                if not harris_complete.empty
                else []
            ),
        }

        # Create historical dates
        historical_dates = []
        if not trump_complete.empty:
            historical_dates.extend(trump_complete["end_date"].tolist())
        if not harris_complete.empty:
            historical_dates.extend(harris_complete["end_date"].tolist())
        historical_dates = sorted(set(historical_dates)) if historical_dates else []

        return {
            "trump_complete": trump_complete,
            "harris_complete": harris_complete,
            "complete_fitted_values": complete_fitted_values,
            "historical_dates": historical_dates,
            "plotting_days_till_then": forecast_dates_list,
            "plotting_holdout_baselines": holdout_baselines,
        }

    def _update_and_visualize(
        self,
        comprehensive_dataset: pd.DataFrame,
        forecast_results: Dict,
        forecast_date: date,
        complete_polling_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Update dataset and create visualizations."""
        logger.info("Updating comprehensive dataset...")

        # Combine training data
        train_dfs = [
            df
            for df in [
                forecast_results["trump_train"],
                forecast_results["harris_train"],
            ]
            if not df.empty
        ]
        training_data = (
            pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
        )

        # Create forecast record
        daily_forecast_record = self.data_manager.create_comprehensive_forecast_record(
            training_data,
            forecast_results["historical_dates"],
            forecast_results["days_till_then"],
            forecast_results["fitted_values"],
            forecast_results["forecasts"],
            forecast_results["baselines"],
            forecast_date,
            forecast_results["electoral_results"],
            forecast_results["best_params"],
            complete_polling_data=complete_polling_data,
        )

        # Update dataset
        if not comprehensive_dataset.empty:
            comprehensive_dataset = comprehensive_dataset[
                comprehensive_dataset["forecast_run_date"] != forecast_date
            ].copy()

        if comprehensive_dataset.empty:
            comprehensive_dataset = daily_forecast_record
        else:
            dfs_to_concat = [
                df
                for df in [comprehensive_dataset, daily_forecast_record]
                if not df.empty
            ]
            comprehensive_dataset = (
                pd.concat(dfs_to_concat, ignore_index=True)
                if dfs_to_concat
                else daily_forecast_record
            )

        # Save dataset
        comprehensive_dataset = self.data_manager.save_comprehensive_dataset(
            comprehensive_dataset
        )

        # Generate visualizations
        self._create_visualizations(
            forecast_results, forecast_date, comprehensive_dataset
        )

        return comprehensive_dataset

    def _create_visualizations(
        self,
        forecast_results: Dict,
        forecast_date: date,
        comprehensive_dataset: pd.DataFrame,
    ):
        """Generate forecast and historical visualizations."""
        logger.info("Creating visualizations...")

        # Main forecast plot (directory already created in main.py)
        forecast_plot_path = (
            Path(self.data_config.forecast_images_dir)
            / f"{forecast_date.strftime('%d%b')}.png"
        )

        if forecast_plot_path.exists():
            forecast_plot_path.unlink()

        try:
            self.plotter.plot_main_forecast(
                forecast_results["historical_dates"],
                forecast_results["plotting_days_till_then"],
                forecast_results["trump_complete"],
                forecast_results["harris_complete"],
                forecast_results["forecasts"],
                forecast_results["baselines"],
                forecast_results["complete_fitted_values"],
                forecast_results["best_params"],
                forecast_results["plotting_days_till_then"],
                forecast_date=forecast_date,
                training_end_date=forecast_results["train_cutoff_date"],
                holdout_baselines=forecast_results["plotting_holdout_baselines"],
                save_path=forecast_plot_path,
            )
        except Exception as e:
            logger.error(f"Failed to create forecast plot: {e}")

        # Historical forecasts plot (directory already created in main.py)
        historical_data = self.data_manager.create_historical_data_for_plotting(
            comprehensive_dataset, forecast_date
        )
        historical_plot_path = (
            Path("outputs/previous_forecasts")
            / f"historical_{forecast_date.strftime('%m%d')}.png"
        )

        if historical_plot_path.exists():
            historical_plot_path.unlink()

        try:
            self.plotter.plot_historical_forecasts(
                historical_data,
                forecast_date=forecast_date,
                save_path=historical_plot_path,
            )
        except Exception as e:
            logger.error(f"Failed to create historical plot: {e}")
