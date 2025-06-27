#!/usr/bin/env python3
"""
Rolling election forecast pipeline - matches original methodology.
For each day Oct 23 - Nov 5, uses only data available up to that day.
Creates single comprehensive CSV instead of separate df_cleaned and previous files.
Includes holdout validation data in visualizations.
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
        root_level = logging.INFO
        your_level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    elif verbose:
        root_level = logging.INFO
        your_level = logging.INFO
        format_str = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        root_level = logging.WARNING
        your_level = logging.WARNING
        format_str = "%(message)s"

    logging.basicConfig(level=root_level, format=format_str, force=True)

    if debug:
        logging.getLogger(__name__).setLevel(your_level)
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

    formats_to_try = [
        "%Y-%m-%d",  # 2024-10-25
        "%m-%d-%Y",  # 10-25-2024
    ]

    year_agnostic_formats = [
        "%m-%d",  # 10-25 (will add 2024)
        "%m/%d",  # 10/25 (will add 2024)
        "%b %d",  # Oct 25 (will add 2024)
        "%B %d",  # October 25 (will add 2024)
    ]

    # Try full date formats first
    for date_format in formats_to_try:
        try:
            parsed_date = datetime.strptime(date_string, date_format).date()
            if date(2024, 10, 1) <= parsed_date <= date(2024, 11, 30):
                return parsed_date
        except ValueError:
            continue

    # Try year-agnostic formats and manually add 2024
    for date_format in year_agnostic_formats:
        try:
            if date_format == "%m-%d":
                temp_date_str = f"2023-{date_string}"
                parsed_date = datetime.strptime(temp_date_str, "2023-%m-%d").date()
                parsed_date = parsed_date.replace(year=2024)
            elif date_format == "%m/%d":
                temp_date_str = f"2023-{date_string.replace('/', '-')}"
                parsed_date = datetime.strptime(temp_date_str, "2023-%m-%d").date()
                parsed_date = parsed_date.replace(year=2024)
            else:
                parsed_date = datetime.strptime(
                    f"{date_string} 2024", f"{date_format} %Y"
                ).date()

            if date(2024, 10, 1) <= parsed_date <= date(2024, 11, 30):
                return parsed_date

        except ValueError:
            continue

    raise ValueError(
        f"Could not parse date '{date_string}'. Try formats like: 2024-10-25, 10-25, Oct 25"
    )


def determine_forecast_dates(args):
    """Determine which dates to process based on command line arguments."""
    default_start = date(2024, 10, 23)
    default_end = date(2024, 11, 5)

    if args.date:
        return [parse_flexible_date(args.date)]

    if args.start or args.end:
        start_date = parse_flexible_date(args.start) if args.start else default_start
        end_date = parse_flexible_date(args.end) if args.end else default_end

        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        return pd.date_range(start=start_date, end=end_date).date

    return pd.date_range(start=default_start, end=default_end).date


def load_or_create_comprehensive_dataset():
    """Load existing comprehensive dataset or create new one."""
    logger = logging.getLogger(__name__)
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
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Creating comprehensive record for {forecast_date}: training_data shape={training_data.shape}, forecast periods={len(days_till_then)}"
    )

    comprehensive_records = []
    election_day = date(2024, 11, 5)

    # Add historical polling data (from training set only)
    for _, row in training_data.iterrows():
        record = {
            "date": row["end_date"],
            "candidate": row["candidate_name"],
            "forecast_run_date": forecast_date,
            "record_type": "historical_polling",
            "data_source": "polling_average",
            "polling_average": row["daily_average"],
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
        comprehensive_records.append(record)

    # Add model fitted values for historical dates (training period only)
    historical_dates = (
        training_data["end_date"].drop_duplicates().sort_values().tolist()
    )

    for i, hist_date in enumerate(historical_dates):
        if i < len(fitted_values["trump"]):
            # Trump fitted value
            record = {
                "date": hist_date,
                "candidate": "Donald Trump",
                "forecast_run_date": forecast_date,
                "record_type": "model_fitted",
                "data_source": "holt_exponential_smoothing",
                "polling_average": None,
                "model_prediction": fitted_values["trump"][i],
                "baseline_prediction": None,
                "days_to_election": (election_day - hist_date).days,
                "weeks_to_election": round((election_day - hist_date).days / 7, 1),
                "is_forecast": False,
                "forecast_horizon": None,
                "alpha": best_params["trump"]["alpha"],
                "beta": best_params["trump"]["beta"],
                "mase_score": best_params["trump"]["mase"],
                "electoral_winner_model": None,
                "electoral_votes_trump_model": None,
                "electoral_votes_harris_model": None,
                "electoral_winner_baseline": None,
                "electoral_votes_trump_baseline": None,
                "electoral_votes_harris_baseline": None,
            }
            comprehensive_records.append(record)

        if i < len(fitted_values["harris"]):
            # Harris fitted value
            record = {
                "date": hist_date,
                "candidate": "Kamala Harris",
                "forecast_run_date": forecast_date,
                "record_type": "model_fitted",
                "data_source": "holt_exponential_smoothing",
                "polling_average": None,
                "model_prediction": fitted_values["harris"][i],
                "baseline_prediction": None,
                "days_to_election": (election_day - hist_date).days,
                "weeks_to_election": round((election_day - hist_date).days / 7, 1),
                "is_forecast": False,
                "forecast_horizon": None,
                "alpha": best_params["harris"]["alpha"],
                "beta": best_params["harris"]["beta"],
                "mase_score": best_params["harris"]["mase"],
                "electoral_winner_model": None,
                "electoral_votes_trump_model": None,
                "electoral_votes_harris_model": None,
                "electoral_winner_baseline": None,
                "electoral_votes_trump_baseline": None,
                "electoral_votes_harris_baseline": None,
            }
            comprehensive_records.append(record)

    # Add forecasts
    for i, forecast_day in enumerate(days_till_then):
        days_ahead = i + 1
        is_election_day = forecast_day == election_day

        # Trump forecast
        record = {
            "date": forecast_day,
            "candidate": "Donald Trump",
            "forecast_run_date": forecast_date,
            "record_type": "forecast",
            "data_source": "holt_exponential_smoothing",
            "polling_average": None,
            "model_prediction": forecasts["trump"][i],
            "baseline_prediction": baselines["trump"][i],
            "days_to_election": (election_day - forecast_day).days,
            "weeks_to_election": round((election_day - forecast_day).days / 7, 1),
            "is_forecast": True,
            "forecast_horizon": days_ahead,
            "alpha": best_params["trump"]["alpha"],
            "beta": best_params["trump"]["beta"],
            "mase_score": best_params["trump"]["mase"],
            "electoral_winner_model": (
                electoral_results["model"]["winner"] if is_election_day else None
            ),
            "electoral_votes_trump_model": (
                electoral_results["model"]["trump_electoral_votes"]
                if is_election_day
                else None
            ),
            "electoral_votes_harris_model": (
                electoral_results["model"]["harris_electoral_votes"]
                if is_election_day
                else None
            ),
            "electoral_winner_baseline": (
                electoral_results["baseline"]["winner"] if is_election_day else None
            ),
            "electoral_votes_trump_baseline": (
                electoral_results["baseline"]["trump_electoral_votes"]
                if is_election_day
                else None
            ),
            "electoral_votes_harris_baseline": (
                electoral_results["baseline"]["harris_electoral_votes"]
                if is_election_day
                else None
            ),
        }
        comprehensive_records.append(record)

        # Harris forecast
        record = {
            "date": forecast_day,
            "candidate": "Kamala Harris",
            "forecast_run_date": forecast_date,
            "record_type": "forecast",
            "data_source": "holt_exponential_smoothing",
            "polling_average": None,
            "model_prediction": forecasts["harris"][i],
            "baseline_prediction": baselines["harris"][i],
            "days_to_election": (election_day - forecast_day).days,
            "weeks_to_election": round((election_day - forecast_day).days / 7, 1),
            "is_forecast": True,
            "forecast_horizon": days_ahead,
            "alpha": best_params["harris"]["alpha"],
            "beta": best_params["harris"]["beta"],
            "mase_score": best_params["harris"]["mase"],
            "electoral_winner_model": (
                electoral_results["model"]["winner"] if is_election_day else None
            ),
            "electoral_votes_trump_model": (
                electoral_results["model"]["trump_electoral_votes"]
                if is_election_day
                else None
            ),
            "electoral_votes_harris_model": (
                electoral_results["model"]["harris_electoral_votes"]
                if is_election_day
                else None
            ),
            "electoral_winner_baseline": (
                electoral_results["baseline"]["winner"] if is_election_day else None
            ),
            "electoral_votes_trump_baseline": (
                electoral_results["baseline"]["trump_electoral_votes"]
                if is_election_day
                else None
            ),
            "electoral_votes_harris_baseline": (
                electoral_results["baseline"]["harris_electoral_votes"]
                if is_election_day
                else None
            ),
        }
        comprehensive_records.append(record)

    df_comprehensive = pd.DataFrame(comprehensive_records)
    df_comprehensive = df_comprehensive.sort_values(
        ["date", "candidate", "record_type"]
    ).reset_index(drop=True)

    logger.debug(f"Created comprehensive record with {len(df_comprehensive)} rows")
    return df_comprehensive


def save_comprehensive_dataset(comprehensive_dataset):
    """Save the comprehensive dataset to CSV with metadata."""
    logger = logging.getLogger(__name__)

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


def create_historical_data_for_plotting(comprehensive_dataset, forecast_date):
    """Create historical forecast data in the format expected by plotting functions."""
    logger = logging.getLogger(__name__)

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
    logger.debug(f"Created historical plotting data with {len(historical_df)} records")

    return historical_df


def main():
    """Run rolling daily election forecasts for the final 2 weeks."""
    args = parse_arguments()
    logger = setup_logging(verbose=args.verbose, debug=args.debug)

    # Log startup information
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
        logger.info(
            "Outputs: Single comprehensive CSV + daily forecast images + historical plots"
        )
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
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    # Load and process ALL raw data once
    if args.verbose:
        logger.info("📊 LOADING AND PROCESSING DATA")
        logger.info("-" * 40)

    logger.info("=== LOADING ALL RAW DATA ===")
    raw_data = collector.load_raw_data()

    # Filter to Biden drop-out date and later
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

    # Initialize comprehensive dataset
    comprehensive_dataset = load_or_create_comprehensive_dataset()
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

        # Proper time series train/holdout split
        test_size = model_config.test_size  # Should be 7 from config
        train_cutoff_date = forecast_date - timedelta(days=test_size)

        if args.debug:
            logger.debug(f"Forecast date: {forecast_date}")
            logger.debug(f"Train-train cutoff: {train_cutoff_date}")
            logger.debug(
                f"Available data range: {available_data['end_date'].min()} to {available_data['end_date'].max()}"
            )

        # Split data based on the cutoff date
        trump_train = trump_data[trump_data["end_date"] < train_cutoff_date].copy()
        harris_train = harris_data[harris_data["end_date"] < train_cutoff_date].copy()
        trump_holdout = trump_data[trump_data["end_date"] >= train_cutoff_date].copy()
        harris_holdout = harris_data[
            harris_data["end_date"] >= train_cutoff_date
        ].copy()

        if len(trump_train) < 10 or len(harris_train) < 10:
            if args.verbose:
                logger.warning(
                    f"   ⚠️  Insufficient train-train data (Trump: {len(trump_train)}, Harris: {len(harris_train)})"
                )
            else:
                logger.warning(f"Insufficient train-train data, skipping")
            continue

        if args.verbose:
            logger.info(f"   📊 Proper time series split:")
            logger.info(
                f"      → Train-train: {len(trump_train)} days ({trump_train['end_date'].min()} to {trump_train['end_date'].max()})"
            )
            logger.info(
                f"      → Holdout: {len(trump_holdout)} days ({trump_holdout['end_date'].min()} to {trump_holdout['end_date'].max()})"
            )
            logger.info(f"      → Forecast from: {forecast_date}")
            logger.info(f"      → ✅ No data leakage: 7-day gap before forecast")
        elif args.debug:
            logger.debug(
                f"Train-train: {len(trump_train)} days, Holdout: {len(trump_holdout)} days"
            )

        # Calculate days until election from this forecast date
        days_to_election = (election_day - forecast_date).days

        if args.verbose:
            logger.info(f"   → Forecasting {days_to_election} days until Election Day")
            logger.info(
                f"   → Training data points: Trump={len(trump_train)}, Harris={len(harris_train)}"
            )
        else:
            logger.info(f"Forecasting {days_to_election} days until election")

        try:
            # Train models using only TRAINING data
            if args.verbose:
                logger.info("   🤖 Training Holt exponential smoothing models...")

            forecaster = HoltElectionForecaster(model_config)
            x_train = pd.Series(range(len(trump_train)))

            # Grid search and model fitting using ONLY training data
            if args.verbose:
                logger.info(
                    "   🔍 Running hyperparameter optimization on training data..."
                )
            else:
                logger.info("Running hyperparameter optimization...")

            best_params = forecaster.grid_search_hyperparameters(
                trump_train, harris_train, x_train
            )

            if args.debug:
                logger.debug(f"Best parameters found: {best_params}")

            if args.verbose:
                logger.info("   ⚙️  Fitting final models on training data...")
            else:
                logger.info("Fitting final models...")

            fitted_models = forecaster.fit_final_models(trump_train, harris_train)

            # Generate predictions for holdout period first (validation)
            holdout_dates = pd.date_range(
                start=train_cutoff_date, end=forecast_date, inclusive="left"
            ).date
            holdout_horizon = len(holdout_dates)
            days_till_then = pd.date_range(start=forecast_date, end=election_day).date

            # Generate forecasts for the entire period: holdout + forecast
            total_horizon = holdout_horizon + len(days_till_then)

            if args.verbose:
                logger.info(
                    f"   📈 Generating predictions: {holdout_horizon} holdout + {len(days_till_then)} forecast periods..."
                )
                if holdout_horizon > 0:
                    logger.info(
                        f"   🔍 Holdout validation: {holdout_dates[0]} to {holdout_dates[-1]}"
                    )
            else:
                logger.info(
                    f"Generating predictions for {total_horizon} total periods..."
                )

            all_predictions = forecaster.forecast(total_horizon)

            # Generate ONE continuous baseline for the entire period (holdout + future)
            logger.info("Generating continuous baseline forecasts...")
            all_baselines = forecaster.generate_baseline_forecasts(
                trump_train, harris_train, total_horizon
            )

            # Split predictions into holdout and forecast periods
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

            if args.debug:
                logger.debug(
                    f"Holdout period: {holdout_dates[0] if len(holdout_dates) > 0 else 'None'} to {holdout_dates[-1] if len(holdout_dates) > 0 else 'None'} ({holdout_horizon} days)"
                )
                logger.debug(
                    f"Forecast shapes: Trump={len(forecasts['trump'])}, Harris={len(forecasts['harris'])}"
                )
                logger.debug(
                    f"Holdout shapes: Trump={len(holdout_predictions['trump'])}, Harris={len(holdout_predictions['harris'])}"
                )
                if len(holdout_predictions["trump"]) > 0:
                    logger.debug(
                        f"Holdout predictions: Trump={holdout_predictions['trump'][-1]:.2f}%, Harris={holdout_predictions['harris'][-1]:.2f}%"
                    )
                logger.debug(
                    f"Final forecasts: Trump={forecasts['trump'][-1]:.2f}%, Harris={forecasts['harris'][-1]:.2f}%"
                )

            # Create complete historical datasets (training + holdout)
            trump_complete = pd.concat(
                [trump_train, trump_holdout], ignore_index=True
            ).sort_values("end_date")
            harris_complete = pd.concat(
                [harris_train, harris_holdout], ignore_index=True
            ).sort_values("end_date")

            # Remove any potential duplicates by date
            trump_complete = trump_complete.drop_duplicates(
                subset=["end_date"], keep="first"
            ).reset_index(drop=True)
            harris_complete = harris_complete.drop_duplicates(
                subset=["end_date"], keep="first"
            ).reset_index(drop=True)

            # Create fitted values for the complete datasets
            complete_fitted_values = {
                "trump": list(fitted_values["trump"])
                + list(holdout_predictions["trump"]),
                "harris": list(fitted_values["harris"])
                + list(holdout_predictions["harris"]),
            }

            # Ensure fitted values match the dataset lengths exactly
            trump_fitted_length = len(trump_complete)
            harris_fitted_length = len(harris_complete)

            complete_fitted_values = {
                "trump": complete_fitted_values["trump"][:trump_fitted_length],
                "harris": complete_fitted_values["harris"][:harris_fitted_length],
            }

            future_forecasts = forecasts
            future_baselines = baselines
            historical_dates = sorted(
                set(
                    trump_complete["end_date"].tolist()
                    + harris_complete["end_date"].tolist()
                )
            )
            future_forecast_dates = list(days_till_then)

            if args.debug:
                logger.debug(
                    f"Complete datasets: Trump={len(trump_complete)}, Harris={len(harris_complete)}"
                )
                logger.debug(f"Historical dates: {len(historical_dates)} unique dates")
                logger.debug(
                    f"Fitted values: Trump={len(complete_fitted_values['trump'])}, Harris={len(complete_fitted_values['harris'])}"
                )
                logger.debug(
                    f"Future forecasts: Trump={len(future_forecasts['trump'])}, Harris={len(future_forecasts['harris'])}"
                )

            # Add holdout performance logging
            if len(holdout_predictions["trump"]) > 0 and len(trump_holdout) > 0:
                trump_holdout_actual = trump_holdout["daily_average"].mean()
                harris_holdout_actual = harris_holdout["daily_average"].mean()
                trump_holdout_pred = np.mean(holdout_predictions["trump"])
                harris_holdout_pred = np.mean(holdout_predictions["harris"])

                trump_holdout_error = abs(trump_holdout_actual - trump_holdout_pred)
                harris_holdout_error = abs(harris_holdout_actual - harris_holdout_pred)

                if args.verbose:
                    logger.info(f"   🎯 Holdout validation performance:")
                    logger.info(
                        f"      Trump: Actual={trump_holdout_actual:.2f}%, Predicted={trump_holdout_pred:.2f}%, Error={trump_holdout_error:.2f}%"
                    )
                    logger.info(
                        f"      Harris: Actual={harris_holdout_actual:.2f}%, Predicted={harris_holdout_pred:.2f}%, Error={harris_holdout_error:.2f}%"
                    )
                elif args.debug:
                    logger.debug(
                        f"Holdout validation: Trump error={trump_holdout_error:.2f}%, Harris error={harris_holdout_error:.2f}%"
                    )

            # Only calculate electoral outcomes for Election Day forecasts
            election_day_date = date(2024, 11, 5)
            if forecast_date == election_day_date:
                # Create simple data for electoral college calculation
                electoral_calc_data = []
                if len(forecasts["trump"]) > 0 and len(forecasts["harris"]) > 0:
                    trump_final = forecasts["trump"][-1]
                    harris_final = forecasts["harris"][-1]
                    trump_baseline = baselines["trump"][-1]
                    harris_baseline = baselines["harris"][-1]

                    electoral_calc_data.append(
                        {
                            "candidate_name": "Donald Trump",
                            "end_date": election_day_date,
                            "daily_average": None,
                            "model": trump_final,
                            "drift_pred": trump_baseline,
                        }
                    )
                    electoral_calc_data.append(
                        {
                            "candidate_name": "Kamala Harris",
                            "end_date": election_day_date,
                            "daily_average": None,
                            "model": harris_final,
                            "drift_pred": harris_baseline,
                        }
                    )

                electoral_calculation_data = pd.DataFrame(electoral_calc_data)

                if len(electoral_calculation_data) > 0:
                    if args.verbose:
                        logger.info(
                            "   🗳️  Calculating electoral college outcomes for Election Day..."
                        )
                    else:
                        logger.info("Calculating electoral college outcomes...")

                    electoral_results = calculator.calculate_all_outcomes(
                        electoral_calculation_data
                    )
                    trump_pred_pct = electoral_results["model"]["trump_vote_pct"]
                    harris_pred_pct = electoral_results["model"]["harris_vote_pct"]

                    if args.debug:
                        logger.debug(
                            f"Electoral results: {electoral_results['model']['winner']} wins"
                        )
                        logger.debug(
                            f"Model vote shares: Trump={trump_pred_pct:.2f}%, Harris={harris_pred_pct:.2f}%"
                        )
                        logger.debug(
                            f"Trump: {electoral_results['model']['trump_electoral_votes']} EVs = 219 safe + {max(0, electoral_results['model']['trump_electoral_votes'] - 219)} swing states {electoral_results['model']['trump_states']}"
                        )
                        logger.debug(
                            f"Harris: {electoral_results['model']['harris_electoral_votes']} EVs = 226 safe + {max(0, electoral_results['model']['harris_electoral_votes'] - 226)} swing states {electoral_results['model']['harris_states']}"
                        )
                else:
                    logger.warning(
                        "Could not calculate electoral outcomes - insufficient data"
                    )
                    trump_pred_pct = (
                        forecasts["trump"][-1] if len(forecasts["trump"]) > 0 else 0
                    )
                    harris_pred_pct = (
                        forecasts["harris"][-1] if len(forecasts["harris"]) > 0 else 0
                    )
                    electoral_results = {
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
                                baselines["trump"][-1]
                                if len(baselines["trump"]) > 0
                                else 0
                            ),
                            "harris_vote_pct": (
                                baselines["harris"][-1]
                                if len(baselines["harris"]) > 0
                                else 0
                            ),
                        },
                    }
            else:
                # For non-Election Day forecasts, just use vote percentages
                logger.info(
                    "Skipping electoral calculation (only calculated for Election Day)"
                )
                trump_pred_pct = (
                    forecasts["trump"][-1] if len(forecasts["trump"]) > 0 else 0
                )
                harris_pred_pct = (
                    forecasts["harris"][-1] if len(forecasts["harris"]) > 0 else 0
                )
                electoral_results = {
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
                            baselines["harris"][-1]
                            if len(baselines["harris"]) > 0
                            else 0
                        ),
                    },
                }

            # Create comprehensive forecast record
            training_data = pd.concat([trump_train, harris_train], ignore_index=True)
            daily_forecast_record = create_comprehensive_forecast_record(
                training_data,
                historical_dates,
                days_till_then,
                fitted_values,
                forecasts,
                baselines,
                forecast_date,
                electoral_results,
                best_params,
            )

            # Update comprehensive dataset
            if len(comprehensive_dataset) > 0:
                existing_records = len(
                    comprehensive_dataset[
                        comprehensive_dataset["forecast_run_date"] == forecast_date
                    ]
                )
                if existing_records > 0:
                    if args.verbose:
                        logger.info(
                            f"   🔄 Replacing {existing_records} existing records for {forecast_date}"
                        )
                    comprehensive_dataset = comprehensive_dataset[
                        comprehensive_dataset["forecast_run_date"] != forecast_date
                    ].copy()

            comprehensive_dataset = pd.concat(
                [comprehensive_dataset, daily_forecast_record], ignore_index=True
            )

            # Save comprehensive dataset
            if args.verbose:
                logger.info("   💾 Saving comprehensive dataset...")
            else:
                logger.info("Saving comprehensive dataset...")

            comprehensive_dataset = save_comprehensive_dataset(comprehensive_dataset)

            # Generate visualizations
            if args.verbose:
                logger.info("   📈 Creating forecast visualization...")
            else:
                logger.info("Creating forecast visualization...")

            # 1. Main forecast plot
            forecast_plot_path = (
                Path(data_config.forecast_images_dir)
                / f"{forecast_date.strftime('%d%b')}.png"
            )
            forecast_plot_path.parent.mkdir(parents=True, exist_ok=True)

            if forecast_plot_path.exists():
                if args.verbose:
                    logger.info(
                        f"   🔄 Replacing existing forecast plot: {forecast_plot_path.name}"
                    )
                elif args.debug:
                    logger.debug(
                        f"Replacing existing forecast plot: {forecast_plot_path}"
                    )
                try:
                    forecast_plot_path.unlink()
                    if args.debug:
                        logger.debug(f"Deleted existing file: {forecast_plot_path}")
                except Exception as e:
                    logger.warning(f"Could not delete existing forecast plot: {e}")

            try:
                plotter.plot_main_forecast(
                    historical_dates,
                    future_forecast_dates,
                    trump_complete,
                    harris_complete,
                    future_forecasts,
                    future_baselines,
                    complete_fitted_values,
                    best_params,
                    future_forecast_dates,
                    forecast_date=forecast_date,
                    training_end_date=train_cutoff_date,
                    holdout_baselines=holdout_baselines,
                    save_path=forecast_plot_path,
                )

                if args.debug:
                    logger.debug(f"✅ Saved forecast plot to: {forecast_plot_path}")

            except Exception as e:
                logger.error(f"❌ Failed to create forecast plot: {e}")
                if args.debug:
                    logger.exception("Forecast plotting error details:")

            # 2. Historical forecasts plot
            if args.verbose:
                logger.info("   📊 Creating historical performance visualization...")
            else:
                logger.info("Creating historical performance visualization...")

            historical_data = create_historical_data_for_plotting(
                comprehensive_dataset, forecast_date
            )
            historical_plot_path = (
                Path("outputs/previous_forecasts")
                / f"historical_{forecast_date.strftime('%m%d')}.png"
            )
            historical_plot_path.parent.mkdir(parents=True, exist_ok=True)

            if historical_plot_path.exists():
                if args.verbose:
                    logger.info(
                        f"   🔄 Replacing existing historical plot: {historical_plot_path.name}"
                    )
                elif args.debug:
                    logger.debug(
                        f"Replacing existing historical plot: {historical_plot_path}"
                    )
                try:
                    historical_plot_path.unlink()
                    if args.debug:
                        logger.debug(f"Deleted existing file: {historical_plot_path}")
                except Exception as e:
                    logger.warning(f"Could not delete existing historical plot: {e}")

            try:
                plotter.plot_historical_forecasts(
                    historical_data,
                    forecast_date=forecast_date,
                    save_path=historical_plot_path,
                )

                if args.debug:
                    logger.debug(f"✅ Saved historical plot to: {historical_plot_path}")

            except Exception as e:
                logger.error(f"❌ Failed to create historical plot: {e}")
                if args.debug:
                    logger.exception("Historical plotting error details:")

            # Log results based on verbosity
            if args.verbose:
                logger.info(f"   ✅ COMPLETED {forecast_date.strftime('%a %b %d')}")
                logger.info(
                    f"   📊 Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
                )
                if forecast_date == election_day_date:
                    trump_swing_evs = max(
                        0, electoral_results["model"]["trump_electoral_votes"] - 219
                    )
                    harris_swing_evs = max(
                        0, electoral_results["model"]["harris_electoral_votes"] - 226
                    )

                    state_names = {
                        "AZ": "Arizona (11)",
                        "GA": "Georgia (16)",
                        "NC": "North Carolina (16)",
                        "NV": "Nevada (6)",
                        "PA": "Pennsylvania (19)",
                        "WI": "Wisconsin (10)",
                        "MI": "Michigan (15)",
                    }

                    winner = electoral_results["model"]["winner"]
                    winner_evs = (
                        electoral_results["model"]["trump_electoral_votes"]
                        if winner == "Trump"
                        else electoral_results["model"]["harris_electoral_votes"]
                    )

                    logger.info(
                        f"   🏆 Electoral outcome: {winner} wins with {winner_evs} electoral votes"
                    )
                    logger.info("   📊 Electoral Vote Breakdown:")
                    logger.info(
                        f"      Trump: {electoral_results['model']['trump_electoral_votes']} total = 219 safe + {trump_swing_evs} swing"
                    )
                    if electoral_results["model"]["trump_states"]:
                        trump_state_details = [
                            state_names.get(state, state)
                            for state in electoral_results["model"]["trump_states"]
                        ]
                        logger.info(
                            f"      Trump swing states: {', '.join(trump_state_details)}"
                        )
                    else:
                        logger.info(f"      Trump swing states: None")
                    logger.info(
                        f"      Harris: {electoral_results['model']['harris_electoral_votes']} total = 226 safe + {harris_swing_evs} swing"
                    )
                    if electoral_results["model"]["harris_states"]:
                        harris_state_details = [
                            state_names.get(state, state)
                            for state in electoral_results["model"]["harris_states"]
                        ]
                        logger.info(
                            f"      Harris swing states: {', '.join(harris_state_details)}"
                        )
                    else:
                        logger.info(f"      Harris swing states: None")
                else:
                    logger.info(
                        "   📈 Interim forecast (electoral calculation only on Election Day)"
                    )
                logger.info(f"   💾 Forecast chart: {forecast_plot_path.name}")
                logger.info(f"   💾 Historical chart: {historical_plot_path.name}")
                logger.info(
                    f"   📊 Dataset now contains {len(comprehensive_dataset)} total records"
                )
            else:
                logger.info(f"✅ Completed forecast for {forecast_date}")
                logger.info(
                    f"   Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
                )
                if forecast_date == election_day_date:
                    trump_swing_evs = max(
                        0, electoral_results["model"]["trump_electoral_votes"] - 219
                    )
                    harris_swing_evs = max(
                        0, electoral_results["model"]["harris_electoral_votes"] - 226
                    )

                    state_names = {
                        "AZ": "Arizona (11)",
                        "GA": "Georgia (16)",
                        "NC": "North Carolina (16)",
                        "NV": "Nevada (6)",
                        "PA": "Pennsylvania (19)",
                        "WI": "Wisconsin (10)",
                        "MI": "Michigan (15)",
                    }

                    winner = electoral_results["model"]["winner"]
                    winner_evs = (
                        electoral_results["model"]["trump_electoral_votes"]
                        if winner == "Trump"
                        else electoral_results["model"]["harris_electoral_votes"]
                    )

                    logger.info(
                        f"   Electoral outcome: {winner} wins with {winner_evs} electoral votes"
                    )
                    logger.info("   Electoral Vote Breakdown:")
                    logger.info(
                        f"   Trump: {electoral_results['model']['trump_electoral_votes']} total = 219 safe + {trump_swing_evs} swing"
                    )
                    if electoral_results["model"]["trump_states"]:
                        trump_state_details = [
                            state_names.get(state, state)
                            for state in electoral_results["model"]["trump_states"]
                        ]
                        logger.info(
                            f"   Trump swing states: {', '.join(trump_state_details)}"
                        )
                    else:
                        logger.info(f"   Trump swing states: None")
                    logger.info(
                        f"   Harris: {electoral_results['model']['harris_electoral_votes']} total = 226 safe + {harris_swing_evs} swing"
                    )
                    if electoral_results["model"]["harris_states"]:
                        harris_state_details = [
                            state_names.get(state, state)
                            for state in electoral_results["model"]["harris_states"]
                        ]
                        logger.info(
                            f"   Harris swing states: {', '.join(harris_state_details)}"
                        )
                    else:
                        logger.info(f"   Harris swing states: None")
                else:
                    logger.info(
                        "   Interim forecast (electoral calculation only on Election Day)"
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
        logger.info(
            f"   • Comprehensive dataset: data/election_forecast_2024_comprehensive.csv"
        )
        logger.info(f"   • Daily forecast images: {data_config.forecast_images_dir}/")
        logger.info(
            f"   • Daily historical plots: outputs/previous_forecasts/historical_MMDD.png"
        )
        logger.info("=" * 50)
    else:
        logger.info(
            f"\n🎉 Rolling Election Forecast 2024 pipeline completed successfully!"
        )
        logger.info(
            f"📊 Comprehensive dataset: data/election_forecast_2024_comprehensive.csv"
        )
        logger.info(f"📈 Forecast images: {data_config.forecast_images_dir}")
        logger.info("📊 Historical plots: outputs/previous_forecasts/")


if __name__ == "__main__":
    main()
