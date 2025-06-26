#!/usr/bin/env python3
"""
Rolling election forecast pipeline - matches original methodology.
For each day Oct 23 - Nov 5, uses only data available up to that day.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path

from src.config import ModelConfig, DataConfig
from src.data.collectors import PollingDataCollector
from src.data.processors import PollingDataProcessor
from src.models.holt_forecaster import HoltElectionForecaster
from src.models.electoral_calculator import ElectoralCollegeCalculator
from src.visualization.plotting import ElectionPlotter


def setup_logging(verbose=False, debug=False):
    """Setup logging with appropriate level for main.py and all src/ modules."""
    if debug:
        # Set your modules to DEBUG, but others to INFO
        root_level = logging.INFO  # Keep third-party libraries quieter
        your_level = logging.DEBUG  # Your code gets debug info
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    elif verbose:
        root_level = logging.INFO
        your_level = logging.INFO
        format_str = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        root_level = logging.WARNING
        your_level = logging.WARNING
        format_str = "%(message)s"

    # Configure the root logger (affects third-party libraries)
    logging.basicConfig(level=root_level, format=format_str, force=True)

    # Set DEBUG level specifically for your modules when --debug is used
    if debug:
        # Your main script
        logging.getLogger(__name__).setLevel(your_level)

        # All your src/ modules
        logging.getLogger("src").setLevel(your_level)
        logging.getLogger("src.data").setLevel(your_level)
        logging.getLogger("src.models").setLevel(your_level)
        logging.getLogger("src.visualization").setLevel(your_level)

        # Silence noisy third-party libraries
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rolling Election Forecast Pipeline 2024",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Process all dates (Oct 23 - Nov 5)
  python main.py --date 2024-10-25        # Process single date
  python main.py --date 10-25             # Process single date (flexible format)
  python main.py --start 10-23 --end 10-27 # Process date range
  python main.py --verbose --debug        # Show detailed output
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and explanations",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show technical debugging information",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Process single date (YYYY-MM-DD, MM-DD, or flexible formats)",
    )
    parser.add_argument(
        "--start", type=str, help="Start date for date range (YYYY-MM-DD or MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, help="End date for date range (YYYY-MM-DD or MM-DD)"
    )

    return parser.parse_args()


def parse_flexible_date(date_string):
    """Parse flexible date formats, defaulting to 2024."""
    if not date_string:
        return None

    # Try multiple date formats
    formats_to_try = [
        "%Y-%m-%d",  # 2024-10-25
        "%m-%d-%Y",  # 10-25-2024
        "%m-%d",  # 10-25 (will add 2024)
        "%m/%d",  # 10/25 (will add 2024)
        "%b %d",  # Oct 25 (will add 2024)
        "%B %d",  # October 25 (will add 2024)
    ]

    for date_format in formats_to_try:
        try:
            parsed_date = datetime.strptime(date_string, date_format).date()

            # If no year in format, assume 2024
            if date_format in ["%m-%d", "%m/%d", "%b %d", "%B %d"]:
                parsed_date = parsed_date.replace(year=2024)

            # Validate it's in the reasonable range for this election
            if date(2024, 10, 1) <= parsed_date <= date(2024, 11, 30):
                return parsed_date

        except ValueError:
            continue

    raise ValueError(
        f"Could not parse date '{date_string}'. Try formats like: 2024-10-25, 10-25, Oct 25"
    )


def determine_forecast_dates(args):
    """Determine which dates to process based on command line arguments."""
    # Default date range (final 2 weeks)
    default_start = date(2024, 10, 23)
    default_end = date(2024, 11, 5)

    # Single date specified
    if args.date:
        single_date = parse_flexible_date(args.date)
        return [single_date]

    # Date range specified
    if args.start or args.end:
        start_date = parse_flexible_date(args.start) if args.start else default_start
        end_date = parse_flexible_date(args.end) if args.end else default_end

        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        return pd.date_range(start=start_date, end=end_date).date

    # No date arguments - use default range
    return pd.date_range(start=default_start, end=default_end).date


def load_or_create_previous_forecasts():
    """
    Load existing previous forecasts or create new dataframe.
    From your swing_states.py logic.
    """
    logger = logging.getLogger(__name__)
    previous_path = Path("previous.csv")

    if previous_path.exists():
        logger.info("Loading existing previous forecasts...")
        previous = pd.read_csv(previous_path)
        previous["date"] = pd.to_datetime(previous["date"]).dt.date
        logger.info(f"Loaded {len(previous)} historical forecast records")
        logger.debug(f"Previous forecasts columns: {previous.columns.tolist()}")
        logger.debug(
            f"Date range: {previous['date'].min()} to {previous['date'].max()}"
        )
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
        logger.debug(f"Created new dataframe with shape: {previous.shape}")
        return previous


def save_forecasts_to_dataframe(
    df_cleaned, x, days_till_then, fitted_values, forecasts, baselines, config
):
    """
    Save forecasts to dataframe - from your original forecast.py logic.
    """
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Input parameters: df_cleaned shape={df_cleaned.shape}, x length={len(x)}, days_till_then length={len(days_till_then)}"
    )

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

    logger.debug(
        f"Trump candidates found: {len(trump_candidates)}, Harris candidates found: {len(harris_candidates)}"
    )

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
    logger.debug(f"After merge: {df_merged.shape}")

    # Calculate the number of historical periods (before forecast period)
    num_historical = len(x) - len(days_till_then)
    logger.debug(
        f"Historical periods: {num_historical}, Forecast periods: {len(days_till_then)}"
    )

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

    logger.debug(f"Final dataframe shape: {df_merged.shape}")
    logger.debug(
        f"Model predictions - Trump range: {df_merged[df_merged['candidate_name']=='Donald Trump']['model'].min():.2f} to {df_merged[df_merged['candidate_name']=='Donald Trump']['model'].max():.2f}"
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
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Updating forecasts for {forecast_date}: Trump={trump_pred_pct:.2f}%, Harris={harris_pred_pct:.2f}%"
    )

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

    logger.debug(f"Updated entries for {forecast_date}")
    return previous_data


def main():
    """Run rolling daily election forecasts for the final 2 weeks."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging(verbose=args.verbose, debug=args.debug)

    # Log startup information based on mode
    if args.verbose:
        logger.info("🚀 ROLLING ELECTION FORECAST 2024 PIPELINE")
        logger.info("=" * 60)
        logger.info("This pipeline generates daily forecasts for the 2024 election")
        logger.info(
            "Each day uses only polling data available up to that point (no future leakage)"
        )
        logger.info(
            "Models: Holt exponential smoothing with hyperparameter optimization"
        )
        logger.info("Outputs: Daily forecast images + historical performance plots")
        logger.info("=" * 60)
    elif not args.debug:
        print("Running rolling election forecasts...")

    # Determine which dates to process
    try:
        forecast_dates = determine_forecast_dates(args)
    except ValueError as e:
        if args.debug:
            logger.exception(f"Date parsing error: {e}")
        else:
            print(f"❌ {e}")
        return

    # Show what will be processed
    if args.verbose or args.debug:
        if len(forecast_dates) == 1:
            logger.info(f"📅 Processing single date: {forecast_dates[0]}")
        else:
            logger.info(
                f"📅 Processing {len(forecast_dates)} dates: {forecast_dates[0]} to {forecast_dates[-1]}"
            )
    elif len(forecast_dates) == 1:
        print(f"Processing forecast for {forecast_dates[0]}")
    else:
        print(
            f"Processing {len(forecast_dates)} dates: {forecast_dates[0]} to {forecast_dates[-1]}"
        )

    logger.info("Starting Rolling Election Forecast 2024 pipeline...")

    # Load configurations
    if args.debug:
        logger.debug("Loading configuration objects...")

    model_config = ModelConfig()
    data_config = DataConfig()

    if args.debug:
        logger.debug(
            f"Model config: grid_min={model_config.grid_min}, grid_max={model_config.grid_max}"
        )
        logger.debug(
            f"Data config: candidates={data_config.candidates}, swing_states={data_config.swing_states}"
        )

    # Initialize components
    if args.verbose:
        logger.info("⚙️  Initializing pipeline components...")

    collector = PollingDataCollector(data_config)
    processor = PollingDataProcessor(model_config)
    calculator = ElectoralCollegeCalculator(model_config)
    plotter = ElectionPlotter(data_config)

    # Create output directories
    Path(data_config.forecast_images_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)  # For final historical plot
    Path("data").mkdir(parents=True, exist_ok=True)

    # Load and process ALL raw data once
    if args.verbose:
        logger.info("📊 LOADING AND PROCESSING DATA")
        logger.info("-" * 40)

    logger.info("=== LOADING ALL RAW DATA ===")
    raw_data = collector.load_raw_data()

    # Filter to Biden drop-out date and later (from your original code)
    biden_out = datetime(2024, 7, 21).date()
    raw_data = raw_data[raw_data["end_date"] >= biden_out]

    if args.verbose:
        logger.info(f"📅 Filtered to data from {biden_out} onwards")
        logger.info(f"📊 Total records: {len(raw_data)}")
    else:
        logger.info(
            f"Filtered to data from {biden_out} onwards: {len(raw_data)} records"
        )

    # Process the raw data
    if args.verbose:
        logger.info(
            "🔄 Processing polling data (filtering and calculating daily averages)..."
        )

    filtered_data = processor.filter_polling_data(raw_data)
    daily_averages = processor.calculate_daily_averages(filtered_data)

    if args.debug:
        logger.debug(f"Filtered data shape: {filtered_data.shape}")
        logger.debug(f"Daily averages shape: {daily_averages.shape}")
        logger.debug(
            f"Date range in daily averages: {daily_averages['end_date'].min()} to {daily_averages['end_date'].max()}"
        )

    # Initialize or load previous forecasts dataframe
    previous_forecasts = load_or_create_previous_forecasts()

    election_day = datetime(2024, 11, 5).date()

    if args.verbose:
        logger.info(f"\n🔮 STARTING ROLLING FORECASTS")
        logger.info("-" * 40)
        logger.info(f"Each forecast uses only data available before that day")

    logger.info(f"=== STARTING ROLLING FORECASTS FOR {len(forecast_dates)} DAYS ===")

    for i, forecast_date in enumerate(forecast_dates):
        if args.verbose:
            logger.info(
                f"\n📅 DAY {i+1}/{len(forecast_dates)}: {forecast_date.strftime('%a %b %d, %Y')}"
            )
            logger.info("─" * 50)
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"FORECAST FOR {forecast_date} ({i+1}/{len(forecast_dates)})")
            logger.info(f"{'='*60}")

        # Use only data available BEFORE this forecast date (NO FUTURE LEAKAGE)
        available_data = daily_averages[
            daily_averages["end_date"] < forecast_date
        ].copy()

        if args.verbose:
            logger.info(
                f"   → Using {len(available_data)} polling records available before {forecast_date}"
            )
        else:
            logger.info(
                f"Using data before {forecast_date}: {len(available_data)} records"
            )

        if args.debug:
            logger.debug(
                f"Available data date range: {available_data['end_date'].min()} to {available_data['end_date'].max()}"
            )

        # Split by candidate with available data only
        trump_data, harris_data = processor.split_by_candidate(available_data)

        if len(trump_data) < 10 or len(harris_data) < 10:
            if args.verbose:
                logger.warning(
                    f"   ⚠️  Insufficient data for {forecast_date} (Trump: {len(trump_data)}, Harris: {len(harris_data)})"
                )
            else:
                logger.warning(f"Insufficient data for {forecast_date}, skipping")
            continue

        # Calculate days until election from this forecast date
        days_to_election = (election_day - forecast_date).days

        if args.verbose:
            logger.info(f"   → Forecasting {days_to_election} days until Election Day")
            logger.info(
                f"   → Data points: Trump={len(trump_data)}, Harris={len(harris_data)}"
            )
        else:
            logger.info(f"Forecasting {days_to_election} days until election")

        try:
            # Train models using only available data
            if args.verbose:
                logger.info("   🤖 Training Holt exponential smoothing models...")

            forecaster = HoltElectionForecaster(model_config)

            # For cross-validation, we need to create a dummy x_train (not used anyway)
            x_train = pd.Series(range(len(trump_data)))

            # Grid search and model fitting
            if args.verbose:
                logger.info("   🔍 Running hyperparameter optimization...")
            else:
                logger.info("Running hyperparameter optimization...")

            best_params = forecaster.grid_search_hyperparameters(
                trump_data, harris_data, x_train
            )

            if args.debug:
                logger.debug(f"Best parameters found: {best_params}")

            if args.verbose:
                logger.info("   ⚙️  Fitting final models with optimal parameters...")
            else:
                logger.info("Fitting final models...")

            fitted_models = forecaster.fit_final_models(trump_data, harris_data)

            days_till_then = pd.Series(
                pd.date_range(start=forecast_date, end=election_day)
            ).dt.date

            # Generate forecasts and baselines for the entire test period
            forecast_horizon = len(days_till_then)  # From forecast_date to election_day

            if args.verbose:
                logger.info(
                    f"   📈 Generating forecasts for {forecast_horizon} periods..."
                )
            else:
                logger.info(f"Generating forecasts for {forecast_horizon} periods...")

            forecasts = forecaster.forecast(forecast_horizon)
            baselines = forecaster.generate_baseline_forecasts(
                trump_data, harris_data, forecast_horizon
            )
            fitted_values = forecaster.get_fitted_values()

            if args.debug:
                logger.debug(
                    f"Forecast shapes: Trump={len(forecasts['trump'])}, Harris={len(forecasts['harris'])}"
                )
                logger.debug(
                    f"Final forecasts: Trump={forecasts['trump'][-1]:.2f}%, Harris={forecasts['harris'][-1]:.2f}%"
                )

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
            if args.verbose:
                logger.info("   🗳️  Calculating electoral college outcomes...")
            else:
                logger.info("Calculating electoral college outcomes...")

            electoral_results = calculator.calculate_all_outcomes(df_for_calculation)

            # Extract final predictions for previous forecasts tracking
            trump_pred_pct = electoral_results["model"]["trump_vote_pct"]
            harris_pred_pct = electoral_results["model"]["harris_vote_pct"]
            trump_b_pct = electoral_results["baseline"]["trump_vote_pct"]
            harris_b_pct = electoral_results["baseline"]["harris_vote_pct"]

            if args.debug:
                logger.debug(
                    f"Electoral results: {electoral_results['model']['winner']} wins"
                )
                logger.debug(
                    f"Model vote shares: Trump={trump_pred_pct:.2f}%, Harris={harris_pred_pct:.2f}%"
                )

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
            if args.verbose:
                logger.info("   💾 Saving processed data...")
            else:
                logger.info("Saving cleaned data...")

            df_for_calculation.to_csv("data/df_cleaned.csv", index=False)

            # Save updated previous forecasts
            previous_forecasts.to_csv("data/previous.csv", index=False)

            # Generate visualizations (both images like your original)
            if args.verbose:
                logger.info("   📈 Creating forecast visualization...")
            else:
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
                if args.verbose:
                    logger.info(
                        "   📊 Creating final historical performance visualization..."
                    )
                else:
                    logger.info(
                        "Creating final historical performance visualization..."
                    )

                historical_plot_path = Path("outputs") / "final_historical.png"
                plotter.plot_historical_forecasts(
                    previous_forecasts, save_path=historical_plot_path
                )
            else:
                if args.debug:
                    logger.debug(
                        "Skipping historical plot - only generated on election day"
                    )
                elif not args.verbose:
                    logger.info(
                        "Skipping historical plot - only generated on election day"
                    )

            # Log results based on verbosity
            if args.verbose:
                logger.info(f"   ✅ COMPLETED {forecast_date.strftime('%a %b %d')}")
                logger.info(
                    f"   📊 Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
                )
                logger.info(
                    f"   🏆 Electoral outcome: {electoral_results['model']['winner']} wins"
                )
                logger.info(f"   💾 Chart saved: {forecast_plot_path.name}")
            else:
                logger.info(f"✅ Completed forecast for {forecast_date}")
                logger.info(
                    f"   Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
                )
                logger.info(
                    f"   Electoral outcome: {electoral_results['model']['winner']} wins"
                )

        except Exception as e:
            if args.debug:
                logger.exception(f"❌ Failed forecast for {forecast_date}")
            else:
                logger.error(f"❌ Failed forecast for {forecast_date}: {e}")
            continue

    # Final summary
    if args.verbose:
        logger.info(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info("📁 Generated outputs:")
        logger.info(f"   • Daily forecast images: {data_config.forecast_images_dir}/")
        logger.info(f"   • Historical summary: outputs/final_historical.png")
        logger.info(f"   • Processed data: data/")
        logger.info("=" * 50)
    else:
        logger.info(f"Saved all data to data/ directory")
        logger.info(
            "\n🎉 Rolling Election Forecast 2024 pipeline completed successfully!"
        )
        logger.info(f"Forecast images saved to: {data_config.forecast_images_dir}")
        logger.info("Final historical image saved to: outputs/final_historical.png")
        logger.info("Cleaned data saved to: data/df_cleaned.csv")
        logger.info("Historical forecasts saved to: data/previous.csv")


if __name__ == "__main__":
    main()
