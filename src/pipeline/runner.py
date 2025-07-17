# src/pipeline/runner.py
"""Main forecast runner that orchestrates the election forecasting pipeline."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.data.polling_manager import PollingDataManager
from src.data.history_manager import HistoryManager
from src.models.holt_forecaster import HoltElectionForecaster
from src.models.electoral_calculator import ElectoralCollegeCalculator
from src.visualization.plotting import ElectionPlotter
from src.utils.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


class ForecastRunner:
    """Orchestrates the complete election forecasting pipeline."""

    def __init__(self, model_config, data_config, verbose=False, debug=False):
        self.model_config = model_config
        self.data_config = data_config
        self.verbose = verbose
        self.debug = debug
        self.election_day = data_config.election_day_parsed

        self.polling_manager = PollingDataManager(data_config)
        self.history_manager = HistoryManager(data_config)
        self.calculator = ElectoralCollegeCalculator(model_config, data_config)
        self.plotter = ElectionPlotter(data_config)
        self.formatter = ResultFormatter(verbose, debug)

    def run_forecasts(self, forecast_dates: Sequence[date]) -> bool:
        """Run forecasts for all specified dates."""
        logger.info("Starting pipeline...")

        forecast_history = (
            self.history_manager.load_forecast_history()
        )  # Load forecast history for appending

        # Run forecasts
        success_count = 0
        for i, forecast_date in enumerate(forecast_dates):
            try:
                self.formatter.log_forecast_start(
                    i + 1, len(forecast_dates), forecast_date
                )

                daily_averages = self.polling_manager.load_incremental_data(
                    forecast_date
                )

                if daily_averages is None or daily_averages.empty:
                    logger.warning(f"No data available for {forecast_date}")
                    continue

                # Execute forecast
                result = self._run_single_forecast(
                    forecast_date, daily_averages, forecast_history
                )

                if result is not None:
                    forecast_history = result
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

    def _run_single_forecast(
        self,
        forecast_date: date,
        daily_averages: pd.DataFrame,
        forecast_history: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Run forecast for a single date."""
        # Use only data available before forecast date
        available_data = daily_averages[
            daily_averages["end_date"] < forecast_date
        ].copy()
        logger.info(f"Using {len(available_data)} records before {forecast_date}")

        # UPDATED: Split by candidate using polling manager
        trump_data, harris_data = self.polling_manager.split_by_candidate(
            available_data
        )

        # Only check for completely empty data (not arbitrary minimums)
        if len(trump_data) == 0 or len(harris_data) == 0:
            logger.warning(f"No polling data found for {forecast_date}")
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

        # Only check for completely empty training data (not arbitrary minimums)
        if len(trump_train) == 0 or len(harris_train) == 0:
            logger.warning("No training data available, skipping")
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
        forecast_history = self._update_and_visualize(
            forecast_history, forecast_results, forecast_date, daily_averages
        )

        # Log results
        self.formatter.log_forecast_results(
            forecast_results, forecast_date, self.election_day
        )
        return forecast_history

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
        forecaster = HoltElectionForecaster(self.model_config, self.data_config)
        x_train = pd.Series(range(len(trump_train)))

        best_params = forecaster.grid_search_hyperparameters(
            trump_train, harris_train, x_train
        )

        logger.info("Fitting final models...")
        forecaster.fit_final_models(trump_train, harris_train)

        # Calculate horizons
        # pylint: disable=no-member  # False positive: DatetimeIndex.date is valid but pylint doesn't recognize it
        holdout_dates = pd.date_range(
            start=train_cutoff_date, end=forecast_date, inclusive="left"
        ).date

        forecast_dates_list = pd.date_range(
            start=forecast_date, end=self.election_day, inclusive="both"
        ).date.tolist()
        # pylint: enable=no-member

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

        electoral_results = (
            self.calculator.calculate_electoral_outcomes_if_election_day(
                forecasts, baselines, forecast_date
            )
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
        candidates = ["trump", "harris"]

        if holdout_horizon > 0:
            # Split using dictionary comprehensions
            forecasts = {
                candidate: all_predictions[candidate][holdout_horizon:]
                for candidate in candidates
            }
            baselines = {
                candidate: all_baselines[candidate][holdout_horizon:]
                for candidate in candidates
            }
            holdout_predictions = {
                candidate: all_predictions[candidate][:holdout_horizon]
                for candidate in candidates
            }
            holdout_baselines = {
                candidate: all_baselines[candidate][:holdout_horizon]
                for candidate in candidates
            }
        else:
            forecasts = all_predictions
            baselines = all_baselines
            holdout_predictions = {candidate: np.array([]) for candidate in candidates}
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

        # Create complete fitted values (combining training fits + holdout predictions)
        complete_fitted_values = {
            "trump": list(fitted_values["trump"]) + list(holdout_predictions["trump"]),
            "harris": list(fitted_values["harris"])
            + list(holdout_predictions["harris"]),
        }

        # Ensure fitted values match dataset lengths
        trump_total_length = len(trump_train) + len(trump_holdout)
        harris_total_length = len(harris_train) + len(harris_holdout)

        complete_fitted_values = {
            "trump": complete_fitted_values["trump"][:trump_total_length],
            "harris": complete_fitted_values["harris"][:harris_total_length],
        }

        return {
            # Pass split datasets directly instead of recombining
            "trump_train": trump_train,
            "trump_holdout": trump_holdout,
            "harris_train": harris_train,
            "harris_holdout": harris_holdout,
            "complete_fitted_values": complete_fitted_values,
            "plotting_days_till_then": forecast_dates_list,
            "plotting_holdout_baselines": holdout_baselines,
        }

    def _update_and_visualize(
        self,
        forecast_history: pd.DataFrame,
        forecast_results: Dict,
        forecast_date: date,
        complete_polling_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Update dataset and create visualizations."""
        logger.info("Updating forecast history...")

        # Combine training data for history record
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

        # Create dates list for history record
        historical_dates = []
        if not forecast_results["trump_train"].empty:
            historical_dates.extend(
                forecast_results["trump_train"]["end_date"].tolist()
            )
        if not forecast_results["harris_train"].empty:
            historical_dates.extend(
                forecast_results["harris_train"]["end_date"].tolist()
            )
        if not forecast_results["trump_holdout"].empty:
            historical_dates.extend(
                forecast_results["trump_holdout"]["end_date"].tolist()
            )
        if not forecast_results["harris_holdout"].empty:
            historical_dates.extend(
                forecast_results["harris_holdout"]["end_date"].tolist()
            )
        historical_dates = sorted(set(historical_dates)) if historical_dates else []

        # Create forecast record using history manager
        daily_forecast_record = self.history_manager.create_forecast_record(
            training_data,
            historical_dates,
            forecast_results["days_till_then"],
            forecast_results["fitted_values"],
            forecast_results["forecasts"],
            forecast_results["baselines"],
            forecast_date,
            forecast_results["electoral_results"],
            forecast_results["best_params"],
            complete_polling_data=complete_polling_data,
        )

        # Update dataset with correct column name
        if not forecast_history.empty:
            forecast_history = forecast_history[
                forecast_history["forecast_date"] != forecast_date
            ].copy()

        if forecast_history.empty:
            forecast_history = daily_forecast_record
        else:
            dfs_to_concat = [
                df for df in [forecast_history, daily_forecast_record] if not df.empty
            ]
            forecast_history = (
                pd.concat(dfs_to_concat, ignore_index=True)
                if dfs_to_concat
                else daily_forecast_record
            )

        # Save dataset using history manager
        forecast_history = self.history_manager.save_forecast_history(forecast_history)

        # Generate visualizations
        self._create_visualizations(forecast_results, forecast_date, forecast_history)

        return forecast_history

    def _create_visualizations(
        self,
        forecast_results: Dict,
        forecast_date: date,
        forecast_history: pd.DataFrame,
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
                trump_train_data=forecast_results["trump_train"],
                trump_holdout_data=forecast_results["trump_holdout"],
                harris_train_data=forecast_results["harris_train"],
                harris_holdout_data=forecast_results["harris_holdout"],
                forecasts=forecast_results["forecasts"],
                baselines=forecast_results["baselines"],
                fitted_values=forecast_results["complete_fitted_values"],
                best_params=forecast_results["best_params"],
                forecast_period_dates=forecast_results["plotting_days_till_then"],
                forecast_date=forecast_date,
                _training_end_date=forecast_results["train_cutoff_date"],
                holdout_baselines=forecast_results["plotting_holdout_baselines"],
                save_path=forecast_plot_path,
            )
        except Exception as e:
            logger.error(f"Failed to create forecast plot: {e}")

        # Historical forecasts plot using history manager
        historical_data = self.history_manager.create_historical_data_for_plotting(
            forecast_history, forecast_date
        )
        historical_plot_path = (
            Path(self.data_config.historical_plots_dir)
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
