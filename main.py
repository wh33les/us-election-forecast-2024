# main.py
"""
Rolling election forecast pipeline - matches original methodology.
For each day Oct 23 - Nov 5, uses only data available up to that day.
"""

# import argparse
# import logging
from pathlib import Path

from src.config import ModelConfig, DataConfig
from src.pipeline.runner import ForecastRunner
from src.utils.logging_setup import setup_logging
from src.utils.cli_parser import parse_arguments, determine_forecast_dates


def main():
    """Run rolling daily election forecasts for the final 2 weeks."""
    args = parse_arguments()
    logger = setup_logging(verbose=args.verbose, debug=args.debug)

    # Show startup information
    if args.verbose:
        logger.info("U.S. election forecast 2024")
        logger.info("=" * 60)
        logger.info("This pipeline generates daily forecasts for the 2024 election.")
        logger.info(
            "Each day uses only polling data available up to that point (no future leakage)."
        )
        logger.info(
            "Models (2 ct): Holt exponential smoothing with hyperparameter "
            "optimization and random walk with drift baseline."
        )
        logger.info(
            "Outputs (4 ct): History of forecasts and daily average cache file for "
            "quicker processing, images for current forecast and history of forecasts."
        )
        logger.info("=" * 60)
    elif args.debug:
        logger.debug("Starting election forecast pipeline in debug mode")
    else:
        print("Running election forecast(s)...")

    # Determine which dates to process
    try:
        forecast_dates = determine_forecast_dates(args)
    except ValueError as e:
        if args.debug:
            logger.exception(f"Date parsing error: {e}")
        else:
            print(f"❌ {e}")
        return

    if len(forecast_dates) == 1:
        print(f"Processing single date: {forecast_dates[0]}")
    else:
        print(
            f"Processing {len(forecast_dates)} dates: {forecast_dates[0]} to {forecast_dates[-1]}"
        )

    # Load configurations
    model_config = ModelConfig()
    data_config = DataConfig()

    if args.debug:
        logger.debug("Loading configuration objects...")
        logger.debug(
            f"Model config: alpha_grid_min={model_config.alpha_grid_min}, alpha_grid_max={model_config.alpha_grid_max}"
            f"beta_grid_min={model_config.beta_grid_min}, beta_grid_max={model_config.beta_grid_max}"
        )
        logger.debug(
            f"Data config: candidates={data_config.candidates}, swing_states={data_config.swing_states}"
        )

    # Create output directories
    Path(data_config.forecast_images_dir).mkdir(
        parents=True, exist_ok=True
    )  # outputs/forecast_images
    Path(data_config.historical_plots_dir).mkdir(
        parents=True, exist_ok=True
    )  # outputs/previous_forecasts

    # Run the forecasting pipeline
    runner = ForecastRunner(model_config, data_config, args.verbose, args.debug)
    success = runner.run_forecasts(forecast_dates)

    # Final summary
    if success:
        if args.verbose:
            logger.info(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info("📁 Generated outputs:")
            logger.info(
                f"   • Comprehensive dataset: data/election_forecast_2024_comprehensive.csv"
            )
            logger.info(
                f"   • Daily forecast images: {data_config.forecast_images_dir}/"
            )
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
    else:
        logger.error("❌ Pipeline completed with errors")


if __name__ == "__main__":
    main()
