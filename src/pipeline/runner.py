# src/pipeline/runner.py
"""Main forecast runner that orchestrates the election forecasting pipeline."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

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
        """Initialize the forecast runner with configurations."""
        self.model_config = model_config
        self.data_config = data_config
        self.verbose = verbose
        self.debug = debug

        # Initialize components
        self.collector = PollingDataCollector(data_config)
        self.processor = PollingDataProcessor(model_config)
        self.calculator = ElectoralCollegeCalculator(model_config)
        self.plotter = ElectionPlotter(data_config)
        self.data_manager = DataManager()
        self.formatter = ResultFormatter(verbose, debug)

    def run_forecasts(self, forecast_dates):
        """Run forecasts for all specified dates using incremental data loading."""
        logger.info("Starting Rolling Election Forecast 2024 pipeline...")

        # Determine if we need incremental loading
        comprehensive_path = Path("data/election_forecast_2024_comprehensive.csv")
        use_incremental = comprehensive_path.exists()

        if use_incremental:
            logger.info(
                "📊 Using incremental data loading (existing comprehensive dataset found)"
            )
        else:
            logger.info(
                "📊 Using full data loading (no existing comprehensive dataset)"
            )

        # Load and process data
        if use_incremental:
            # For incremental loading, we'll load data on each iteration
            daily_averages = None
        else:
            # For full loading, load all data once
            daily_averages = self._load_and_process_data()
            if daily_averages is None:
                return False

        # Initialize comprehensive dataset
        comprehensive_dataset = self.data_manager.load_or_create_comprehensive_dataset()
        election_day = datetime(2024, 11, 5).date()

        if self.verbose:
            logger.info(f"\n🔮 STARTING ROLLING FORECASTS")
            logger.info("-" * 40)
            logger.info(f"Each forecast uses only data available before that day")

        logger.info(
            f"=== STARTING ROLLING FORECASTS FOR {len(forecast_dates)} DAYS ==="
        )

        # Run forecasts for each date
        success_count = 0
        for i, forecast_date in enumerate(forecast_dates):
            try:
                self.formatter.log_forecast_start(
                    i + 1, len(forecast_dates), forecast_date
                )

                # Load data for this specific date (incremental or full)
                if use_incremental:
                    daily_averages_for_date = self._load_incremental_data_for_date(
                        forecast_date
                    )
                else:
                    daily_averages_for_date = daily_averages

                if daily_averages_for_date is None:
                    logger.warning(f"No data available for {forecast_date}")
                    continue

                result = self._run_single_forecast(
                    forecast_date,
                    daily_averages_for_date,
                    election_day,
                    comprehensive_dataset,
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

    def _load_incremental_data_for_date(self, forecast_date):
        """Load data incrementally for a specific forecast date."""
        if self.verbose:
            logger.info(f"📊 Loading data incrementally for {forecast_date}")

        try:
            # Use the new incremental loading method
            daily_averages = self.collector.load_data_for_incremental_pipeline(
                forecast_date
            )

            if len(daily_averages) == 0:
                logger.warning(f"No data available for {forecast_date}")
                return None

            if self.verbose:
                logger.info(f"✅ Loaded {len(daily_averages)} daily average records")
                logger.info(
                    f"📅 Date range: {daily_averages['end_date'].min()} to {daily_averages['end_date'].max()}"
                )

            return daily_averages

        except Exception as e:
            logger.error(f"Failed to load incremental data for {forecast_date}: {e}")
            if self.debug:
                logger.exception("Incremental data loading error:")
            return None

    def _load_and_process_data(self):
        """Load and process all raw data once (original method for full loading)."""
        if self.verbose:
            logger.info("📊 LOADING AND PROCESSING DATA")
            logger.info("-" * 40)

        logger.info("=== LOADING ALL RAW DATA ===")
        raw_data = self.collector.load_raw_data()

        # Filter to Biden drop-out date and later
        biden_out = datetime(2024, 7, 21).date()
        raw_data = raw_data[raw_data["end_date"] >= biden_out]

        if self.verbose:
            logger.info(f"📅 Filtered to data from {biden_out} onwards")
            logger.info(f"📊 Total records: {len(raw_data)}")
        else:
            logger.info(
                f"Filtered to data from {biden_out} onwards: {len(raw_data)} records"
            )

        # Process the raw data
        if self.verbose:
            logger.info(
                "🔄 Processing polling data (filtering and calculating daily averages)..."
            )

        filtered_data = self.processor.filter_polling_data(raw_data)
        daily_averages = self.processor.calculate_daily_averages(filtered_data)

        if self.debug:
            logger.debug(f"Filtered data shape: {filtered_data.shape}")
            logger.debug(f"Daily averages shape: {daily_averages.shape}")
            logger.debug(
                f"Date range in daily averages: {daily_averages['end_date'].min()} to {daily_averages['end_date'].max()}"
            )

        return daily_averages

    def _run_single_forecast(
        self, forecast_date, daily_averages, election_day, comprehensive_dataset
    ):
        """Run forecast for a single date."""
        # Use only data available BEFORE this forecast date (NO FUTURE LEAKAGE)
        available_data = daily_averages[
            daily_averages["end_date"] < forecast_date
        ].copy()

        if self.verbose:
            logger.info(
                f"   → Using {len(available_data)} polling records available before {forecast_date}"
            )
        else:
            logger.info(
                f"Using data before {forecast_date}: {len(available_data)} records"
            )

        if self.debug:
            logger.debug(
                f"Available data date range: {available_data['end_date'].min()} to {available_data['end_date'].max()}"
            )

        # Split by candidate with available data only
        trump_data, harris_data = self.processor.split_by_candidate(available_data)

        if len(trump_data) < 10 or len(harris_data) < 10:
            if self.verbose:
                logger.warning(
                    f"   ⚠️  Insufficient data for {forecast_date} (Trump: {len(trump_data)}, Harris: {len(harris_data)})"
                )
            else:
                logger.warning(f"Insufficient data for {forecast_date}, skipping")
            return None

        # Prepare train/holdout split
        test_size = self.model_config.test_size
        train_cutoff_date = forecast_date - timedelta(days=test_size)

        if self.debug:
            logger.debug(f"Forecast date: {forecast_date}")
            logger.debug(f"Train-train cutoff: {train_cutoff_date}")

        # Split data based on the cutoff date
        trump_train = trump_data[trump_data["end_date"] < train_cutoff_date].copy()
        harris_train = harris_data[harris_data["end_date"] < train_cutoff_date].copy()
        trump_holdout = trump_data[trump_data["end_date"] >= train_cutoff_date].copy()
        harris_holdout = harris_data[
            harris_data["end_date"] >= train_cutoff_date
        ].copy()

        if len(trump_train) < 10 or len(harris_train) < 10:
            if self.verbose:
                logger.warning(
                    f"   ⚠️  Insufficient train-train data (Trump: {len(trump_train)}, Harris: {len(harris_train)})"
                )
            else:
                logger.warning(f"Insufficient train-train data, skipping")
            return None

        if self.verbose:
            logger.info(f"   📊 Proper time series split:")
            logger.info(
                f"      → Train-train: {len(trump_train)} days ({trump_train['end_date'].min()} to {trump_train['end_date'].max()})"
            )
            logger.info(
                f"      → Holdout: {len(trump_holdout)} days ({trump_holdout['end_date'].min()} to {trump_holdout['end_date'].max()})"
            )
            logger.info(f"      → Forecast from: {forecast_date}")
            logger.info(f"      → ✅ No data leakage: 7-day gap before forecast")

        # Run the actual forecasting
        forecast_results = self._execute_forecast(
            trump_train,
            harris_train,
            trump_holdout,
            harris_holdout,
            forecast_date,
            election_day,
            train_cutoff_date,
        )

        if forecast_results is None:
            return None

        # Update comprehensive dataset
        if self.verbose:
            logger.info("   💾 Updating comprehensive dataset...")
            logger.info(
                f"   📊 Current dataset size: {len(comprehensive_dataset)} records"
            )

        comprehensive_dataset = self._update_dataset(
            comprehensive_dataset,
            forecast_results,
            forecast_date,
            daily_averages,
        )

        if self.verbose:
            logger.info(
                f"   📊 Updated dataset size: {len(comprehensive_dataset)} records"
            )
            logger.info("   ✅ Dataset update completed")

        # Generate visualizations
        self._generate_visualizations(
            forecast_results, forecast_date, comprehensive_dataset
        )

        # Log results
        self.formatter.log_forecast_results(
            forecast_results, forecast_date, election_day
        )

        return comprehensive_dataset

    def _execute_forecast(
        self,
        trump_train,
        harris_train,
        trump_holdout,
        harris_holdout,
        forecast_date,
        election_day,
        train_cutoff_date,
    ):
        """Execute the actual forecasting models."""
        # Calculate days until election
        days_to_election = (election_day - forecast_date).days

        if self.verbose:
            logger.info(f"   → Forecasting {days_to_election} days until Election Day")
            logger.info(
                f"   → Training data points: Trump={len(trump_train)}, Harris={len(harris_train)}"
            )
        else:
            logger.info(f"Forecasting {days_to_election} days until election")

        # Train models
        if self.verbose:
            logger.info("   🤖 Training Holt exponential smoothing models...")

        forecaster = HoltElectionForecaster(self.model_config)
        x_train = pd.Series(range(len(trump_train)))

        if self.verbose:
            logger.info("   🔍 Running hyperparameter optimization on training data...")
        else:
            logger.info("Running hyperparameter optimization...")

        best_params = forecaster.grid_search_hyperparameters(
            trump_train, harris_train, x_train
        )

        if self.debug:
            logger.debug(f"Best parameters found: {best_params}")

        if self.verbose:
            logger.info("   ⚙️  Fitting final models on training data...")
        else:
            logger.info("Fitting final models...")

        fitted_models = forecaster.fit_final_models(trump_train, harris_train)

        # Calculate forecast horizons
        holdout_dates = pd.date_range(
            start=train_cutoff_date, end=forecast_date, inclusive="left"
        ).date
        holdout_horizon = len(holdout_dates)

        # Both CSV and plotting should store complete forecasts from forecast_date to Election Day
        # The historical plotting will extract only the Election Day prediction later
        forecast_days_till_then = pd.date_range(
            start=forecast_date, end=election_day, inclusive="both"
        ).date.tolist()
        total_horizon = holdout_horizon + len(forecast_days_till_then)

        if self.verbose:
            logger.info(
                f"   📈 Generating predictions for CSV storage: {holdout_horizon} holdout + {len(forecast_days_till_then)} forecast periods to Election Day..."
            )
            logger.info(
                f"   🎨 Generating predictions for visualization: same as CSV storage ({total_horizon} total periods)..."
            )
            if holdout_horizon > 0:
                logger.info(
                    f"   🔍 Holdout validation: {holdout_dates[0]} to {holdout_dates[-1]}"
                )
        else:
            logger.info(
                f"Generating predictions: {total_horizon} periods (holdout + forecast to Election Day)..."
            )

        # Generate predictions (complete forecast to Election Day)
        all_predictions = forecaster.forecast(total_horizon)
        logger.info("Generating baseline forecasts...")
        all_baselines = forecaster.generate_baseline_forecasts(
            trump_train, harris_train, total_horizon
        )

        # Split predictions between holdout and forecast periods
        if holdout_horizon > 0:
            holdout_predictions = {
                "trump": all_predictions["trump"][:holdout_horizon],
                "harris": all_predictions["harris"][:holdout_horizon],
            }
            forecasts = {
                "trump": all_predictions["trump"][holdout_horizon:],
                "harris": all_predictions["harris"][holdout_horizon:],
            }
            holdout_baselines = {
                "trump": all_baselines["trump"][:holdout_horizon],
                "harris": all_baselines["harris"][:holdout_horizon],
            }
            baselines = {
                "trump": all_baselines["trump"][holdout_horizon:],
                "harris": all_baselines["harris"][holdout_horizon:],
            }
        else:
            holdout_predictions = {"trump": [], "harris": []}
            holdout_baselines = None
            forecasts = all_predictions
            baselines = all_baselines

        fitted_values = forecaster.get_fitted_values()

        # Log holdout performance
        if len(holdout_predictions["trump"]) > 0 and len(trump_holdout) > 0:
            self._log_holdout_performance(
                holdout_predictions, trump_holdout, harris_holdout
            )

        # Calculate electoral outcomes - use complete forecasts
        electoral_results = self._calculate_electoral_outcomes(
            forecasts, baselines, forecast_date, election_day
        )

        # Prepare datasets for plotting (use same data as CSV - complete forecast)
        plotting_data = self._prepare_plotting_data(
            trump_train,
            harris_train,
            trump_holdout,
            harris_holdout,
            fitted_values,
            holdout_predictions,
            forecasts,
            baselines,
            forecast_days_till_then,
            holdout_baselines,
        )

        return {
            # Data for CSV storage (complete forecast from forecast_date to Election Day)
            "trump_train": trump_train,
            "harris_train": harris_train,
            "fitted_values": fitted_values,
            "forecasts": forecasts,  # Complete forecast to Election Day
            "baselines": baselines,  # Complete baseline to Election Day
            "holdout_baselines": holdout_baselines,
            "days_till_then": forecast_days_till_then,  # All forecast dates
            "future_forecast_dates": forecast_days_till_then,  # All forecast dates
            # Data for visualization (same as CSV)
            "trump_complete": plotting_data["trump_complete"],
            "harris_complete": plotting_data["harris_complete"],
            "complete_fitted_values": plotting_data["complete_fitted_values"],
            "plotting_forecasts": forecasts,  # Same as CSV forecasts
            "plotting_baselines": baselines,  # Same as CSV baselines
            "plotting_days_till_then": forecast_days_till_then,  # Same as CSV dates
            "plotting_holdout_baselines": holdout_baselines,
            # Shared data
            "electoral_results": electoral_results,
            "best_params": best_params,
            "train_cutoff_date": train_cutoff_date,
            "historical_dates": plotting_data["historical_dates"],
        }

    def _log_holdout_performance(
        self, holdout_predictions, trump_holdout, harris_holdout
    ):
        """Log holdout validation performance."""
        trump_holdout_actual = trump_holdout["daily_average"].mean()
        harris_holdout_actual = harris_holdout["daily_average"].mean()
        trump_holdout_pred = np.mean(holdout_predictions["trump"])
        harris_holdout_pred = np.mean(holdout_predictions["harris"])

        trump_holdout_error = abs(trump_holdout_actual - trump_holdout_pred)
        harris_holdout_error = abs(harris_holdout_actual - harris_holdout_pred)

        if self.verbose:
            logger.info(f"   🎯 Holdout validation performance:")
            logger.info(
                f"      Trump: Actual={trump_holdout_actual:.2f}%, Predicted={trump_holdout_pred:.2f}%, Error={trump_holdout_error:.2f}%"
            )
            logger.info(
                f"      Harris: Actual={harris_holdout_actual:.2f}%, Predicted={harris_holdout_pred:.2f}%, Error={harris_holdout_error:.2f}%"
            )
        elif self.debug:
            logger.debug(
                f"Holdout validation: Trump error={trump_holdout_error:.2f}%, Harris error={harris_holdout_error:.2f}%"
            )

    def _calculate_electoral_outcomes(
        self, forecasts, baselines, forecast_date, election_day
    ):
        """Calculate electoral college outcomes."""
        election_day_date = date(2024, 11, 5)

        if forecast_date == election_day_date:
            # Create data for electoral college calculation
            electoral_calc_data = []
            if len(forecasts["trump"]) > 0 and len(forecasts["harris"]) > 0:
                trump_final = forecasts["trump"][-1]
                harris_final = forecasts["harris"][-1]
                trump_baseline = baselines["trump"][-1]
                harris_baseline = baselines["harris"][-1]

                electoral_calc_data.extend(
                    [
                        {
                            "candidate_name": "Donald Trump",
                            "end_date": election_day_date,
                            "daily_average": None,
                            "model": trump_final,
                            "drift_pred": trump_baseline,
                        },
                        {
                            "candidate_name": "Kamala Harris",
                            "end_date": election_day_date,
                            "daily_average": None,
                            "model": harris_final,
                            "drift_pred": harris_baseline,
                        },
                    ]
                )

            electoral_calculation_data = pd.DataFrame(electoral_calc_data)

            if len(electoral_calculation_data) > 0:
                if self.verbose:
                    logger.info(
                        "   🗳️  Calculating electoral college outcomes for Election Day..."
                    )
                else:
                    logger.info("Calculating electoral college outcomes...")

                return self.calculator.calculate_all_outcomes(
                    electoral_calculation_data
                )
            else:
                logger.warning(
                    "Could not calculate electoral outcomes - insufficient data"
                )
                return self._create_fallback_electoral_results(forecasts, baselines)
        else:
            # For non-Election Day forecasts
            logger.info(
                "Skipping electoral calculation (only calculated for Election Day)"
            )
            return self._create_interim_electoral_results(forecasts, baselines)

    def _create_fallback_electoral_results(self, forecasts, baselines):
        """Create fallback electoral results when calculation fails."""
        trump_pred_pct = forecasts["trump"][-1] if len(forecasts["trump"]) > 0 else 0
        harris_pred_pct = forecasts["harris"][-1] if len(forecasts["harris"]) > 0 else 0

        return {
            "model": {
                "winner": "Unknown",
                "trump_electoral_votes": None,
                "harris_electoral_votes": None,
                "trump_states": [],
                "harris_states": [],
                "trump_vote_pct": trump_pred_pct,
                "harris_vote_pct": harris_pred_pct,
            },
            "baseline": {
                "winner": "Unknown",
                "trump_electoral_votes": None,
                "harris_electoral_votes": None,
                "trump_states": [],
                "harris_states": [],
                "trump_vote_pct": (
                    baselines["trump"][-1] if len(baselines["trump"]) > 0 else 0
                ),
                "harris_vote_pct": (
                    baselines["harris"][-1] if len(baselines["harris"]) > 0 else 0
                ),
            },
        }

    def _create_interim_electoral_results(self, forecasts, baselines):
        """Create interim electoral results for non-Election Day forecasts."""
        trump_pred_pct = forecasts["trump"][-1] if len(forecasts["trump"]) > 0 else 0
        harris_pred_pct = forecasts["harris"][-1] if len(forecasts["harris"]) > 0 else 0

        return {
            "model": {
                "winner": "N/A (interim forecast)",
                "trump_electoral_votes": None,
                "harris_electoral_votes": None,
                "trump_states": [],
                "harris_states": [],
                "trump_vote_pct": trump_pred_pct,
                "harris_vote_pct": harris_pred_pct,
            },
            "baseline": {
                "winner": "N/A (interim forecast)",
                "trump_electoral_votes": None,
                "harris_electoral_votes": None,
                "trump_states": [],
                "harris_states": [],
                "trump_vote_pct": (
                    baselines["trump"][-1] if len(baselines["trump"]) > 0 else 0
                ),
                "harris_vote_pct": (
                    baselines["harris"][-1] if len(baselines["harris"]) > 0 else 0
                ),
            },
        }

    def _prepare_plotting_data(
        self,
        trump_train,
        harris_train,
        trump_holdout,
        harris_holdout,
        fitted_values,
        holdout_predictions,
        forecasts,
        baselines,
        days_till_then,
        holdout_baselines,
    ):
        """Prepare data for plotting functions."""
        # Create complete historical datasets (training + holdout)
        trump_complete = pd.concat(
            [trump_train, trump_holdout], ignore_index=True
        ).sort_values("end_date")
        harris_complete = pd.concat(
            [harris_train, harris_holdout], ignore_index=True
        ).sort_values("end_date")

        # Remove duplicates
        trump_complete = trump_complete.drop_duplicates(
            subset=["end_date"], keep="first"
        ).reset_index(drop=True)
        harris_complete = harris_complete.drop_duplicates(
            subset=["end_date"], keep="first"
        ).reset_index(drop=True)

        # Create fitted values for complete datasets
        complete_fitted_values = {
            "trump": list(fitted_values["trump"]) + list(holdout_predictions["trump"]),
            "harris": list(fitted_values["harris"])
            + list(holdout_predictions["harris"]),
        }

        # Ensure fitted values match dataset lengths
        complete_fitted_values = {
            "trump": complete_fitted_values["trump"][: len(trump_complete)],
            "harris": complete_fitted_values["harris"][: len(harris_complete)],
        }

        historical_dates = sorted(
            set(
                trump_complete["end_date"].tolist()
                + harris_complete["end_date"].tolist()
            )
        )

        return {
            "trump_complete": trump_complete,
            "harris_complete": harris_complete,
            "complete_fitted_values": complete_fitted_values,
            "historical_dates": historical_dates,
        }

    def _update_dataset(
        self,
        comprehensive_dataset,
        forecast_results,
        forecast_date,
        complete_polling_data=None,
    ):
        """Update the comprehensive dataset with new forecast results."""
        training_data = pd.concat(
            [forecast_results["trump_train"], forecast_results["harris_train"]],
            ignore_index=True,
        )

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
            complete_polling_data=complete_polling_data,  # Pass complete polling data
        )

        # Remove existing records for this forecast date
        if len(comprehensive_dataset) > 0:
            existing_records = len(
                comprehensive_dataset[
                    comprehensive_dataset["forecast_run_date"] == forecast_date
                ]
            )
            if existing_records > 0:
                if self.verbose:
                    logger.info(
                        f"   🔄 Replacing {existing_records} existing records for {forecast_date}"
                    )
                comprehensive_dataset = comprehensive_dataset[
                    comprehensive_dataset["forecast_run_date"] != forecast_date
                ].copy()

        # Add new records
        if len(comprehensive_dataset) == 0:
            comprehensive_dataset = daily_forecast_record
        else:
            comprehensive_dataset = pd.concat(
                [comprehensive_dataset, daily_forecast_record], ignore_index=True
            )

        # Save dataset
        if self.verbose:
            logger.info("   💾 Saving comprehensive dataset...")
        else:
            logger.info("Saving comprehensive dataset...")

        return self.data_manager.save_comprehensive_dataset(comprehensive_dataset)

    def _generate_visualizations(
        self, forecast_results, forecast_date, comprehensive_dataset
    ):
        """Generate forecast and historical visualizations."""
        if self.verbose:
            logger.info("   📈 Creating forecast visualization...")
        else:
            logger.info("Creating forecast visualization...")

        # Main forecast plot
        forecast_plot_path = (
            Path(self.data_config.forecast_images_dir)
            / f"{forecast_date.strftime('%d%b')}.png"
        )

        # Debug: Show the exact path being used
        if self.verbose:
            logger.info(
                f"   📁 Creating forecast plot at: {forecast_plot_path.absolute()}"
            )
            logger.info(
                f"   📁 Directory configured as: {self.data_config.forecast_images_dir}"
            )

        forecast_plot_path.parent.mkdir(parents=True, exist_ok=True)

        # Debug: Confirm directory was created
        if self.verbose:
            logger.info(f"   📁 Directory exists: {forecast_plot_path.parent.exists()}")

        if forecast_plot_path.exists():
            forecast_plot_path.unlink()

        try:
            self.plotter.plot_main_forecast(
                forecast_results["historical_dates"],
                forecast_results[
                    "plotting_days_till_then"
                ],  # Use plotting data for visualization
                forecast_results["trump_complete"],
                forecast_results["harris_complete"],
                forecast_results[
                    "plotting_forecasts"
                ],  # Use plotting forecasts for visualization
                forecast_results[
                    "plotting_baselines"
                ],  # Use plotting baselines for visualization
                forecast_results["complete_fitted_values"],
                forecast_results["best_params"],
                forecast_results[
                    "plotting_days_till_then"
                ],  # Use plotting data for future dates
                forecast_date=forecast_date,
                training_end_date=forecast_results["train_cutoff_date"],
                holdout_baselines=forecast_results[
                    "plotting_holdout_baselines"
                ],  # Use plotting holdout baselines
                save_path=forecast_plot_path,
            )

            if self.debug:
                logger.debug(f"✅ Saved forecast plot to: {forecast_plot_path}")
                logger.debug(
                    f"✅ File exists after save: {forecast_plot_path.exists()}"
                )
                logger.debug(
                    f"✅ File size: {forecast_plot_path.stat().st_size if forecast_plot_path.exists() else 'N/A'} bytes"
                )

        except Exception as e:
            logger.error(f"❌ Failed to create forecast plot: {e}")
            if self.debug:
                logger.exception("Forecast plotting error details:")

        # Historical forecasts plot
        if self.verbose:
            logger.info("   📊 Creating historical performance visualization...")
        else:
            logger.info("Creating historical performance visualization...")

        historical_data = self.data_manager.create_historical_data_for_plotting(
            comprehensive_dataset, forecast_date
        )
        historical_plot_path = (
            Path("outputs/previous_forecasts")
            / f"historical_{forecast_date.strftime('%m%d')}.png"
        )
        historical_plot_path.parent.mkdir(parents=True, exist_ok=True)

        if historical_plot_path.exists():
            historical_plot_path.unlink()

        try:
            self.plotter.plot_historical_forecasts(
                historical_data,
                forecast_date=forecast_date,
                save_path=historical_plot_path,
            )

            if self.debug:
                logger.debug(f"✅ Saved historical plot to: {historical_plot_path}")

        except Exception as e:
            logger.error(f"❌ Failed to create historical plot: {e}")
            if self.debug:
                logger.exception("Historical plotting error details:")
