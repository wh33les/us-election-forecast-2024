# src/main.py
#!/usr/bin/env python3
"""
Rolling election forecast pipeline - matches original methodology.
For each day Oct 23 - Nov 5, uses only data available up to that day.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

from config import ModelConfig, DataConfig
from data.collectors import PollingDataCollector
from data.processors import PollingDataProcessor
from models.holt_forecaster import HoltElectionForecaster
from models.electoral_calculator import ElectoralCollegeCalculator
from visualization.plotting import ElectionPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_or_create_previous_forecasts():
    """
    Load existing previous forecasts or create new dataframe.
    From your swing_states.py logic.
    """
    previous_path = Path("previous.csv")

    if previous_path.exists():
        logger.info("Loading existing previous forecasts...")
        previous = pd.read_csv(previous_path)
        previous["date"] = pd.to_datetime(previous["date"]).dt.date
        logger.info(f"Loaded {len(previous)} historical forecast records")
        return previous
    else:
        logger.info("Creating new previous forecasts dataframe...")
        dates = pd.Series(
            pd.date_range(start=datetime(2024, 10, 23), end=datetime(2024, 11, 5))
        ).dt.date
        previous = pd.DataFrame(
            {
                "date": pd.concat([dates, dates]),
                "candidate": pd.concat(
                    [
                        pd.Series("Donald Trump", index=range(len(dates))),
                        pd.Series("Kamala Harris", index=range(len(dates))),
                    ]
                ),
                "model": np.nan,
                "baseline": np.nan,
            }
        )
        return previous


def save_forecasts_to_dataframe(
    df_cleaned, x, days_till_then, fitted_values, forecasts, baselines, config
):
    """
    Save forecasts to dataframe - from your original forecast.py logic.
    """
    # Create forecast data dates (x contains both historical and forecast dates)
    forecast_data_dates = pd.DataFrame(columns=["end_date", "candidate_name"])
    forecast_data_dates["end_date"] = pd.concat([x, x], ignore_index=True)

    # Extract candidate names - handle case where data might be empty
    trump_candidates = df_cleaned[df_cleaned["candidate_name"] == "Donald Trump"][
        "candidate_name"
    ]
    harris_candidates = df_cleaned[df_cleaned["candidate_name"] == "Kamala Harris"][
        "candidate_name"
    ]

    forecast_data_dates["candidate_name"] = pd.concat(
        [
            trump_candidates,
            pd.Series("Donald Trump", index=range(len(days_till_then))),
            harris_candidates,
            pd.Series("Kamala Harris", index=range(len(days_till_then))),
        ],
        ignore_index=True,
    )

    # Merge with cleaned data
    df_merged = pd.merge(
        df_cleaned, forecast_data_dates, on=["candidate_name", "end_date"], how="outer"
    )

    # Calculate the number of historical periods (before forecast period)
    num_historical = len(x) - len(days_till_then)

    # Fill in model predictions
    # Model: fitted values for historical data + forecasts for future dates
    df_merged["model"] = pd.concat(
        [
            pd.Series(fitted_values["trump"]),
            pd.Series(forecasts["trump"]),
            pd.Series(fitted_values["harris"]),
            pd.Series(forecasts["harris"]),
        ],
        ignore_index=True,
    )

    # Fill in baseline predictions (matches original logic exactly)
    # Baseline: NaN for historical data + baseline forecasts for future dates
    df_merged["drift_pred"] = pd.concat(
        [
            pd.Series(np.nan, index=range(num_historical)),  # Trump historical
            pd.Series(baselines["trump"]),  # Trump forecast
            pd.Series(np.nan, index=range(num_historical)),  # Harris historical
            pd.Series(baselines["harris"]),  # Harris forecast
        ],
        ignore_index=True,
    )

    return df_merged


def update_previous_forecasts(
    previous_data,
    trump_pred_pct,
    harris_pred_pct,
    trump_b_pct,
    harris_b_pct,
    forecast_date,
):
    """
    Update previous forecasts with today's results.
    From your swing_states.py update logic.
    """
    # Update with this day's numbers
    previous_data.loc[
        (previous_data["date"] == forecast_date)
        & (previous_data["candidate"] == "Donald Trump"),
        ["model", "baseline"],
    ] = [trump_pred_pct, trump_b_pct]

    previous_data.loc[
        (previous_data["date"] == forecast_date)
        & (previous_data["candidate"] == "Kamala Harris"),
        ["model", "baseline"],
    ] = [harris_pred_pct, harris_b_pct]

    return previous_data


def main():
    """Run rolling daily election forecasts for the final 2 weeks."""
    logger.info("Starting Rolling Election Forecast 2024 pipeline...")

    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()

    # Initialize components
    collector = PollingDataCollector(data_config)
    processor = PollingDataProcessor(model_config)
    calculator = ElectoralCollegeCalculator(model_config)
    plotter = ElectionPlotter(data_config)

    # Create output directories
    Path(data_config.forecast_images_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)  # For final historical plot
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Load and process ALL raw data once
    logger.info("=== LOADING ALL RAW DATA ===")
    raw_data = collector.load_raw_data()

    # Filter to Biden drop-out date and later (from your original code)
    biden_out = datetime(2024, 7, 21).date()
    raw_data = raw_data[raw_data["end_date"] >= biden_out]
    logger.info(f"Filtered to data from {biden_out} onwards: {len(raw_data)} records")

    # Process the raw data
    filtered_data = processor.filter_polling_data(raw_data)
    daily_averages = processor.calculate_daily_averages(filtered_data)

    # Initialize or load previous forecasts dataframe
    previous_forecasts = load_or_create_previous_forecasts()

    # Rolling forecast dates (final 2 weeks)
    forecast_dates = pd.date_range("2024-10-23", "2024-11-05").date
    election_day = datetime(2024, 11, 5).date()

    logger.info(f"=== STARTING ROLLING FORECASTS FOR {len(forecast_dates)} DAYS ===")

    for i, forecast_date in enumerate(forecast_dates):
        logger.info(f"\n{'='*60}")
        logger.info(f"FORECAST FOR {forecast_date} ({i+1}/{len(forecast_dates)})")
        logger.info(f"{'='*60}")

        # Use only data available BEFORE this forecast date (NO FUTURE LEAKAGE)
        # This matches your original logic: end_date < today
        available_data = daily_averages[
            daily_averages["end_date"] < forecast_date
        ].copy()
        logger.info(f"Using data before {forecast_date}: {len(available_data)} records")

        # Split by candidate with available data only
        trump_data, harris_data = processor.split_by_candidate(available_data)

        if len(trump_data) < 10 or len(harris_data) < 10:
            logger.warning(f"Insufficient data for {forecast_date}, skipping")
            continue

        # Calculate days until election from this forecast date
        days_to_election = (election_day - forecast_date).days
        logger.info(f"Forecasting {days_to_election} days until election")

        try:
            # Train models using only available data
            forecaster = HoltElectionForecaster(model_config)

            # For cross-validation, we need to create a dummy x_train (not used anyway)
            x_train = pd.Series(range(len(trump_data)))

            # Grid search and model fitting
            logger.info("Running hyperparameter optimization...")
            best_params = forecaster.grid_search_hyperparameters(
                trump_data, harris_data, x_train
            )

            logger.info("Fitting final models...")
            fitted_models = forecaster.fit_final_models(trump_data, harris_data)

            days_till_then = pd.Series(
                pd.date_range(start=forecast_date, end=election_day)
            ).dt.date

            # Generate forecasts and baselines for the entire test period
            forecast_horizon = len(days_till_then)  # From forecast_date to election_day
            logger.info(f"Generating forecasts for {forecast_horizon} periods...")
            forecasts = forecaster.forecast(forecast_horizon)
            baselines = forecaster.generate_baseline_forecasts(
                trump_data, harris_data, forecast_horizon
            )
            fitted_values = forecaster.get_fitted_values()

            # All dates = historical data dates + forecast period dates
            all_dates = (
                pd.concat(
                    [
                        available_data["end_date"].drop_duplicates().sort_values(),
                        days_till_then,
                    ],
                    ignore_index=True,
                )
                .drop_duplicates()
                .sort_values(ignore_index=True)
            )

            # Test dates = forecast period (from forecast_date to election_day)
            test_dates = days_till_then

            # Save forecasts to dataframe (for electoral college calculation)
            df_for_calculation = save_forecasts_to_dataframe(
                available_data,
                all_dates,
                days_till_then,
                fitted_values,
                forecasts,
                baselines,
                model_config,
            )

            # Calculate electoral college outcomes
            logger.info("Calculating electoral college outcomes...")
            electoral_results = calculator.calculate_all_outcomes(df_for_calculation)

            # Extract final predictions for previous forecasts tracking
            trump_pred_pct = electoral_results["model"]["trump_vote_pct"]
            harris_pred_pct = electoral_results["model"]["harris_vote_pct"]
            trump_b_pct = electoral_results["baseline"]["trump_vote_pct"]
            harris_b_pct = electoral_results["baseline"]["harris_vote_pct"]

            # Update previous forecasts dataframe
            previous_forecasts = update_previous_forecasts(
                previous_forecasts,
                trump_pred_pct,
                harris_pred_pct,
                trump_b_pct,
                harris_b_pct,
                forecast_date,
            )

            # Save cleaned data for next day's update (matches original forecast.py)
            logger.info("Saving cleaned data...")
            df_for_calculation.to_csv("data/processed/df_cleaned.csv", index=False)

            # Save updated previous forecasts
            previous_forecasts.to_csv("data/processed/previous.csv", index=False)

            # Generate visualizations (both images like your original)
            logger.info("Creating forecast visualization...")

            # Add timestamp to avoid overwriting (optional)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Main forecast plot (forecast_images/)
            forecast_plot_path = (
                Path(data_config.forecast_images_dir)
                / f"{forecast_date.strftime('%d%b')}.png"
                # Uncomment next line to add timestamps:
                # / f"{forecast_date.strftime('%d%b')}_{timestamp}.png"
            )
            plotter.plot_main_forecast(
                all_dates,
                test_dates,
                trump_data,
                harris_data,
                forecasts,
                baselines,
                fitted_values,
                best_params,
                days_till_then,
                forecast_date=forecast_date,  # Pass the actual forecast date for title
                save_path=forecast_plot_path,
            )

            # 2. Historical forecasts plot (only generate on final day - election day)
            if forecast_date == election_day:
                logger.info("Creating final historical performance visualization...")
                historical_plot_path = Path("outputs") / "final_historical.png"
                plotter.plot_historical_forecasts(
                    previous_forecasts, save_path=historical_plot_path
                )
            else:
                logger.info("Skipping historical plot - only generated on election day")

            logger.info(f"✅ Completed forecast for {forecast_date}")
            logger.info(
                f"   Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
            )
            logger.info(
                f"   Electoral outcome: {electoral_results['model']['winner']} wins"
            )

        except Exception as e:
            logger.error(f"❌ Failed forecast for {forecast_date}: {e}")
            continue

    # Save final previous forecasts
    logger.info(f"Saved all data to data/processed/ directory")

    logger.info("\n🎉 Rolling Election Forecast 2024 pipeline completed successfully!")
    logger.info(f"Forecast images saved to: {data_config.forecast_images_dir}")
    logger.info("Final historical image saved to: outputs/final_historical.png")
    logger.info("Cleaned data saved to: data/processed/df_cleaned.csv")
    logger.info("Historical forecasts saved to: data/processed/previous.csv")


if __name__ == "__main__":
    main()
