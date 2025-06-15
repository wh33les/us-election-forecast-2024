# src/visualization/plotting.py
"""Plotting and visualization functions for election forecasting."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, date
from typing import Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ElectionPlotter:
    """Handle all plotting and visualization for election forecasting."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

        # Set plotting style
        sns.set_style("whitegrid")

        # Enable LaTeX rendering for mathematical expressions
        plt.rcParams["text.usetex"] = False  # Use mathtext instead of full LaTeX

    def plot_main_forecast(
        self,
        all_dates: pd.Series,
        test_dates: pd.Series,
        trump_data: pd.DataFrame,
        harris_data: pd.DataFrame,
        forecasts: Dict[str, np.ndarray],
        baselines: Dict[str, np.ndarray],
        fitted_values: Dict[str, np.ndarray],
        best_params: Dict[str, Dict[str, float]],
        forecast_period_dates: pd.Series,
        forecast_date: Union[datetime, date] = None,  # Add forecast_date parameter
        save_path: Optional[Path] = None,
    ):
        """
        Create main forecast plot showing polling data and predictions.
        From your original forecast.py plotting logic.
        """
        logger.info("Creating main forecast plot...")

        # Prepare data for plotting (matches your original logic)
        null_averages = pd.Series(np.nan, index=range(len(forecast_period_dates)))

        # Observed data with NaN for forecast period
        y_trump_obs = pd.concat(
            [trump_data["daily_average"], null_averages], ignore_index=True
        )
        y_harris_obs = pd.concat(
            [harris_data["daily_average"], null_averages], ignore_index=True
        )

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot observed polling averages
        plt.plot(all_dates, y_trump_obs, "r", label="Trump daily polling average")
        plt.plot(all_dates, y_harris_obs, "b", label="Harris daily polling average")

        # Plot forecasts over the test period (including forecast date)
        plt.plot(
            test_dates,
            forecasts["trump"],
            "r--.",
            label="Trump prediction",
        )
        plt.plot(
            test_dates,
            forecasts["harris"],
            "b--.",
            label="Harris prediction",
        )

        # Plot baseline forecasts
        plt.plot(
            test_dates,
            baselines["trump"],
            "r:",
            label="Trump random walk with drift baseline forecast",
        )
        plt.plot(
            test_dates,
            baselines["harris"],
            "b:",
            label="Harris random walk with drift baseline forecast",
        )

        # Formatting
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Percentage", fontsize=12)
        plt.xlim(datetime(2024, 7, 21), datetime(2024, 11, 5))
        plt.ylim(40, 52)

        # Title with correct forecast date
        if forecast_date:
            if isinstance(forecast_date, date) and not isinstance(
                forecast_date, datetime
            ):
                title_date = forecast_date.strftime("%a %b %d %Y")
            else:
                title_date = forecast_date.strftime("%a %b %d %Y")
        else:
            title_date = (
                test_dates[0].strftime("%a %b %d %Y")
                if len(test_dates) > 0
                else pd.Timestamp.today().strftime("%a %b %d %Y")
            )

        plt.title(
            f"Predictions for Election Day, as of {title_date}",
            fontsize=16,
        )

        plt.legend()

        # Add parameter annotations with proper LaTeX formatting
        election_day_dt = datetime(2024, 11, 5)
        trump_final = round(forecasts["trump"][-1], 1)

        param_text = (
            f"$\\alpha_{{Trump}} = {best_params['trump']['alpha']:.2f}$\n"
            f"$\\beta_{{Trump}} = {best_params['trump']['beta']:.2f}$\n"
            f"$\\alpha_{{Harris}} = {best_params['harris']['alpha']:.2f}$\n"
            f"$\\beta_{{Harris}} = {best_params['harris']['beta']:.2f}$"
        )

        plt.annotate(
            param_text,
            xy=(election_day_dt, trump_final),
            xytext=(datetime(2024, 7, 23), 40.5),
        )

        # Save plot
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved main forecast plot to {save_path}")

    def plot_historical_forecasts(
        self,
        previous_forecasts: pd.DataFrame,
        save_path: Optional[Path] = None,
    ):
        """
        Create historical forecasts plot showing how predictions changed over time.
        From your swing_states.py plotting logic (without 538 data).
        """
        logger.info("Creating historical forecasts plot...")

        # Create the full expected date range (Oct 23 - Nov 5)
        full_date_range = pd.Series(
            pd.date_range(start=datetime(2024, 10, 23), end=datetime(2024, 11, 5))
        ).dt.date

        # Separate by candidate and merge with full date range
        trump_data = previous_forecasts[
            previous_forecasts["candidate"] == "Donald Trump"
        ].copy()
        harris_data = previous_forecasts[
            previous_forecasts["candidate"] == "Kamala Harris"
        ].copy()

        # Create full dataframes with the complete date range
        trump_full = pd.DataFrame({"date": full_date_range})
        harris_full = pd.DataFrame({"date": full_date_range})

        # Merge with actual data (this will include NaN for missing dates)
        trump_full = trump_full.merge(trump_data, on="date", how="left")
        harris_full = harris_full.merge(harris_data, on="date", how="left")

        # Extract data for plotting (full range)
        dates = full_date_range
        y_trump_mod = trump_full["model"].values
        y_trump_baseline = trump_full["baseline"].values
        y_harris_mod = harris_full["model"].values
        y_harris_baseline = harris_full["baseline"].values

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot model predictions (will automatically handle NaN gaps)
        plt.plot(dates, y_trump_mod, "r", label="Trump (model prediction)")
        plt.plot(dates, y_harris_mod, "b", label="Harris (model prediction)")

        # Plot baseline predictions (only where not NaN)
        trump_baseline_mask = ~pd.isna(y_trump_baseline)
        harris_baseline_mask = ~pd.isna(y_harris_baseline)

        if np.any(trump_baseline_mask):
            plt.plot(
                dates[trump_baseline_mask],
                y_trump_baseline[trump_baseline_mask],
                "r--",
                label="Trump (baseline)",
            )
        if np.any(harris_baseline_mask):
            plt.plot(
                dates[harris_baseline_mask],
                y_harris_baseline[harris_baseline_mask],
                "b--",
                label="Harris (baseline)",
            )

        # Formatting
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Percentage of popular vote", fontsize=12)
        plt.xlim(datetime(2024, 10, 23), datetime(2024, 11, 5))
        plt.ylim(45, 52)

        # Set x-axis ticks
        plt.xticks(
            [
                datetime(2024, 10, 23),
                datetime(2024, 10, 24),
                datetime(2024, 10, 26),
                datetime(2024, 10, 28),
                datetime(2024, 10, 30),
                datetime(2024, 11, 1),
                datetime(2024, 11, 3),
                datetime(2024, 11, 5),
            ]
        )

        # Title - use the latest date that has actual data
        available_trump_data = trump_data[trump_data["model"].notna()]
        if len(available_trump_data) > 0:
            latest_date = available_trump_data["date"].max()
        else:
            latest_date = datetime(2024, 10, 22).date()

        plt.title(
            f"Predictions up to {latest_date.strftime('%a %b %d %Y')}",
            fontsize=16,
        )

        plt.legend()

        # Clean historical plot without annotations

        # Save plot
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved historical forecasts plot to {save_path}")

    def create_summary_plot(
        self,
        final_results: Dict[str, Dict],
        save_path: Optional[Path] = None,
    ):
        """
        Create a summary plot showing final electoral college predictions.
        """
        logger.info("Creating summary plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Model predictions
        model = final_results["model"]
        ax1.bar(
            ["Trump", "Harris"],
            [model["trump_electoral_votes"], model["harris_electoral_votes"]],
            color=["red", "blue"],
            alpha=0.7,
        )
        ax1.axhline(y=270, color="black", linestyle="--", alpha=0.5, label="270 to win")
        ax1.set_title("Model Prediction", fontsize=14)
        ax1.set_ylabel("Electoral Votes", fontsize=12)
        ax1.legend()

        # Baseline predictions
        baseline = final_results["baseline"]
        ax2.bar(
            ["Trump", "Harris"],
            [baseline["trump_electoral_votes"], baseline["harris_electoral_votes"]],
            color=["red", "blue"],
            alpha=0.7,
        )
        ax2.axhline(y=270, color="black", linestyle="--", alpha=0.5, label="270 to win")
        ax2.set_title("Baseline Prediction", fontsize=14)
        ax2.set_ylabel("Electoral Votes", fontsize=12)
        ax2.legend()

        plt.suptitle("Electoral College Predictions", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info(f"Saved summary plot to {save_path}")
