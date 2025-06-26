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

    def _plot_continuous_segments(
        self, data, color, label_prefix, line_style="-", marker="o", markersize=4
    ):
        """
        Plot data in continuous segments, breaking lines at gaps.
        """
        if len(data) == 0:
            return

        # Sort data by date
        data_sorted = data.sort_values("date").copy()
        data_sorted["plot_date"] = pd.to_datetime(data_sorted["date"])

        # Find gaps in the date sequence (more than 1 day apart)
        gaps = []
        for i in range(1, len(data_sorted)):
            current_date = data_sorted.iloc[i]["plot_date"]
            previous_date = data_sorted.iloc[i - 1]["plot_date"]
            days_diff = (current_date - previous_date).days
            if days_diff > 1:
                gaps.append(i)

        # Split into continuous segments
        segments = []
        start_idx = 0
        for gap_idx in gaps:
            segments.append(data_sorted.iloc[start_idx:gap_idx])
            start_idx = gap_idx
        segments.append(data_sorted.iloc[start_idx:])  # Last segment

        # Plot each segment separately
        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            # Only add label to the first segment to avoid duplicate legend entries
            segment_label = label_prefix if i == 0 else None

            if len(segment) == 1:
                # Single point - just plot marker
                plt.plot(
                    segment["plot_date"],
                    segment["model"],
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linestyle="None",
                    label=segment_label,
                )
            else:
                # Multiple points - plot line segment
                plt.plot(
                    segment["plot_date"],
                    segment["model"],
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    markersize=markersize,
                    linewidth=2,
                    label=segment_label,
                )

    def _plot_baseline_segments(
        self, data, color, label_prefix, line_style="--", marker="s", markersize=3
    ):
        """
        Plot baseline data in continuous segments, breaking lines at gaps.
        """
        baseline_data = data[data["baseline"].notna()]
        if len(baseline_data) == 0:
            return

        # Sort data by date
        data_sorted = baseline_data.sort_values("date").copy()
        data_sorted["plot_date"] = pd.to_datetime(data_sorted["date"])

        # Find gaps in the date sequence (more than 1 day apart)
        gaps = []
        for i in range(1, len(data_sorted)):
            current_date = data_sorted.iloc[i]["plot_date"]
            previous_date = data_sorted.iloc[i - 1]["plot_date"]
            days_diff = (current_date - previous_date).days
            if days_diff > 1:
                gaps.append(i)

        # Split into continuous segments
        segments = []
        start_idx = 0
        for gap_idx in gaps:
            segments.append(data_sorted.iloc[start_idx:gap_idx])
            start_idx = gap_idx
        segments.append(data_sorted.iloc[start_idx:])  # Last segment

        # Plot each segment separately
        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            # Only add label to the first segment to avoid duplicate legend entries
            segment_label = label_prefix if i == 0 else None

            if len(segment) == 1:
                # Single point - just plot marker
                plt.plot(
                    segment["plot_date"],
                    segment["baseline"],
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linestyle="None",
                    label=segment_label,
                )
            else:
                # Multiple points - plot line segment
                plt.plot(
                    segment["plot_date"],
                    segment["baseline"],
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    markersize=markersize,
                    linewidth=2,
                    label=segment_label,
                )

    def plot_historical_forecasts(
        self,
        previous_forecasts: pd.DataFrame,
        forecast_date=None,
        save_path: Optional[Path] = None,
    ):
        """
        Create historical forecasts plot showing how predictions changed over time.
        Now with gap-aware plotting that breaks lines at missing dates.
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

        # Plot model predictions with gap-aware line breaking
        if len(trump_data) > 0:
            self._plot_continuous_segments(
                trump_data,
                color="red",
                label_prefix="Trump (model prediction)",
                line_style="-",
                marker="o",
                markersize=4,
            )

        if len(harris_data) > 0:
            self._plot_continuous_segments(
                harris_data,
                color="blue",
                label_prefix="Harris (model prediction)",
                line_style="-",
                marker="o",
                markersize=4,
            )

        # Plot baseline predictions with gap-aware line breaking
        if len(trump_data) > 0:
            self._plot_baseline_segments(
                trump_data,
                color="red",
                label_prefix="Trump (baseline)",
                line_style="--",
                marker="s",
                markersize=3,
            )

        if len(harris_data) > 0:
            self._plot_baseline_segments(
                harris_data,
                color="blue",
                label_prefix="Harris (baseline)",
                line_style="--",
                marker="s",
                markersize=3,
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

        # Add subtle grid for better readability
        plt.grid(True, alpha=0.3)

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
