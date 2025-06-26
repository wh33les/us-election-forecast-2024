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

        # Save plot with proper overwrite handling
        if save_path:
            try:
                # Force remove existing file if it exists
                if save_path.exists():
                    save_path.unlink()
                    logger.debug(f"Removed existing file: {save_path}")

                # Save with explicit parameters for clean output
                plt.savefig(
                    save_path,
                    bbox_inches="tight",
                    dpi=150,
                    facecolor="white",
                    edgecolor="none",
                )
                logger.info(f"Saved main forecast plot to {save_path}")

            except Exception as e:
                logger.error(f"Error saving main forecast plot: {e}")
            finally:
                # Always close the figure to free memory
                plt.close()
        else:
            # Close figure even if not saving
            plt.close()

    def plot_historical_forecasts(
        self,
        previous_forecasts: pd.DataFrame,
        forecast_date=None,
        save_path: Optional[Path] = None,
    ):
        """
        Create historical forecasts plot showing how predictions changed over time.
        """
        logger.info("Creating historical forecasts plot...")

        # Separate by candidate - only get data that actually exists (not NaN)
        trump_data = previous_forecasts[
            (previous_forecasts["candidate"] == "Donald Trump")
            & (previous_forecasts["model"].notna())
        ].copy()
        harris_data = previous_forecasts[
            (previous_forecasts["candidate"] == "Kamala Harris")
            & (previous_forecasts["model"].notna())
        ].copy()

        # If no data available, create a minimal plot with a message
        if len(trump_data) == 0 and len(harris_data) == 0:
            plt.figure(figsize=(12, 6))
            plt.text(
                0.5,
                0.5,
                "No historical data available yet",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
                fontsize=16,
            )
            plt.xlim(datetime(2024, 10, 23), datetime(2024, 11, 5))
            plt.ylim(45, 52)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Percentage of popular vote", fontsize=12)
            if forecast_date:
                plt.title(
                    f"Predictions up to {forecast_date.strftime('%a %b %d %Y')}",
                    fontsize=16,
                )
            else:
                plt.title("Election Forecast Evolution", fontsize=16)

            # Save empty plot with proper overwrite handling
            if save_path:
                try:
                    # Force remove existing file if it exists
                    if save_path.exists():
                        save_path.unlink()
                        logger.debug(f"Removed existing file: {save_path}")

                    plt.savefig(
                        save_path,
                        bbox_inches="tight",
                        dpi=150,
                        facecolor="white",
                        edgecolor="none",
                    )
                    logger.info(f"Saved empty historical forecasts plot to {save_path}")

                except Exception as e:
                    logger.error(f"Error saving empty historical plot: {e}")
                finally:
                    plt.close()
            else:
                plt.close()
            return

        # Create figure
        plt.figure(figsize=(12, 6))

        # Convert dates to datetime objects to ensure proper plotting
        if len(trump_data) > 0:
            trump_data_sorted = trump_data.sort_values("date").copy()
            # Ensure dates are datetime objects for matplotlib
            trump_data_sorted["plot_date"] = pd.to_datetime(trump_data_sorted["date"])

            plt.plot(
                trump_data_sorted["plot_date"],
                trump_data_sorted["model"],
                "r-",
                label="Trump (model prediction)",
                marker="o",
                markersize=4,
                linewidth=2,
            )

        if len(harris_data) > 0:
            harris_data_sorted = harris_data.sort_values("date").copy()
            # Ensure dates are datetime objects for matplotlib
            harris_data_sorted["plot_date"] = pd.to_datetime(harris_data_sorted["date"])

            plt.plot(
                harris_data_sorted["plot_date"],
                harris_data_sorted["model"],
                "b-",
                label="Harris (model prediction)",
                marker="o",
                markersize=4,
                linewidth=2,
            )

        # Plot baseline predictions - only where not NaN
        trump_baseline_data = trump_data[trump_data["baseline"].notna()]
        harris_baseline_data = harris_data[harris_data["baseline"].notna()]

        if len(trump_baseline_data) > 0:
            trump_baseline_sorted = trump_baseline_data.sort_values("date").copy()
            trump_baseline_sorted["plot_date"] = pd.to_datetime(
                trump_baseline_sorted["date"]
            )
            plt.plot(
                trump_baseline_sorted["plot_date"],
                trump_baseline_sorted["baseline"],
                "r--",
                label="Trump (baseline)",
                marker="s",
                markersize=3,
                linewidth=2,
            )

        if len(harris_baseline_data) > 0:
            harris_baseline_sorted = harris_baseline_data.sort_values("date").copy()
            harris_baseline_sorted["plot_date"] = pd.to_datetime(
                harris_baseline_sorted["date"]
            )
            plt.plot(
                harris_baseline_sorted["plot_date"],
                harris_baseline_sorted["baseline"],
                "b--",
                label="Harris (baseline)",
                marker="s",
                markersize=3,
                linewidth=2,
            )

        # Formatting
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Percentage of popular vote", fontsize=12)

        # Set x-axis limits - no padding
        plt.xlim(datetime(2024, 10, 23), datetime(2024, 11, 5))
        plt.ylim(45, 52)

        # Set x-axis ticks
        plt.xticks(
            [
                datetime(2024, 10, 23),
                datetime(2024, 10, 25),
                datetime(2024, 10, 27),
                datetime(2024, 10, 29),
                datetime(2024, 10, 31),
                datetime(2024, 11, 2),
                datetime(2024, 11, 4),
                datetime(2024, 11, 5),
            ]
        )

        # Title
        if forecast_date:
            title_date = forecast_date
        else:
            if len(trump_data) > 0:
                title_date = trump_data["date"].max()
            elif len(harris_data) > 0:
                title_date = harris_data["date"].max()
            else:
                title_date = datetime(2024, 10, 22).date()

        plt.title(
            f"Predictions up to {title_date.strftime('%a %b %d %Y')}", fontsize=16
        )
        plt.legend()

        # Save plot with proper overwrite handling
        if save_path:
            try:
                # Force remove existing file if it exists
                if save_path.exists():
                    save_path.unlink()
                    logger.debug(f"Removed existing file: {save_path}")

                # Save with explicit parameters for clean output
                plt.savefig(
                    save_path,
                    bbox_inches="tight",
                    dpi=150,
                    facecolor="white",
                    edgecolor="none",
                )
                logger.info(f"Saved historical forecasts plot to {save_path}")

            except Exception as e:
                logger.error(f"Error saving historical forecasts plot: {e}")
            finally:
                # Always close the figure to free memory
                plt.close()
        else:
            # Close figure even if not saving
            plt.close()
