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
        historical_dates: pd.Series,  # Historical dates (training + holdout)
        test_dates: pd.Series,  # Future forecast dates
        trump_data: pd.DataFrame,  # Trump data (training + holdout)
        harris_data: pd.DataFrame,  # Harris data (training + holdout)
        forecasts: Dict[str, np.ndarray],  # Future forecasts
        baselines: Dict[str, np.ndarray],  # Future baselines
        fitted_values: Dict[
            str, np.ndarray
        ],  # Combined: training fitted + holdout predictions
        best_params: Dict[str, Dict[str, float]],
        forecast_period_dates: pd.Series,
        forecast_date: Union[datetime, date] = None,
        training_end_date: Union[datetime, date] = None,  # When training period ends
        holdout_baselines: Optional[
            Dict[str, np.ndarray]
        ] = None,  # NEW: Baseline predictions for holdout period
        save_path: Optional[Path] = None,
    ):
        """
        Create main forecast plot showing polling data and predictions.
        Training period: only raw polling data (solid lines)
        Holdout period: raw data + model predictions (dashed lines) + baseline predictions (dotted lines)
        Future period: model forecasts (dashed lines)
        """
        logger.info("Creating main forecast plot...")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Split data into training and holdout periods
        if training_end_date:
            # Find the split point in historical_dates
            training_mask = pd.Series(historical_dates) < training_end_date
            training_dates = pd.Series(historical_dates)[training_mask]
            holdout_dates = pd.Series(historical_dates)[~training_mask]

            # Split the data accordingly
            trump_training_data = (
                trump_data[trump_data.index < len(training_dates)]
                if len(training_dates) > 0
                else trump_data.iloc[:0]
            )
            harris_training_data = (
                harris_data[harris_data.index < len(training_dates)]
                if len(training_dates) > 0
                else harris_data.iloc[:0]
            )
            trump_holdout_data = (
                trump_data[trump_data.index >= len(training_dates)]
                if len(training_dates) < len(trump_data)
                else trump_data.iloc[:0]
            )
            harris_holdout_data = (
                harris_data[harris_data.index >= len(training_dates)]
                if len(training_dates) < len(harris_data)
                else harris_data.iloc[:0]
            )

            # FIXED: Plot continuous daily polling averages that connect across training/holdout boundary

            # Plot training period with thick lines
            if len(training_dates) > 0:
                plt.plot(
                    training_dates,
                    trump_training_data["daily_average"],
                    "r",
                    linewidth=2,
                    label="Trump daily polling average",
                )
                plt.plot(
                    training_dates,
                    harris_training_data["daily_average"],
                    "b",
                    linewidth=2,
                    label="Harris daily polling average",
                )

            # Plot holdout period continuing from training (thinner for visual distinction)
            if len(holdout_dates) > 0:
                # Create continuous connection by including last training point
                if len(training_dates) > 0:
                    # Connect from last training point to holdout period
                    trump_continuous_dates = [training_dates.iloc[-1]] + list(
                        holdout_dates
                    )
                    harris_continuous_dates = [training_dates.iloc[-1]] + list(
                        holdout_dates
                    )
                    trump_continuous_values = [
                        trump_training_data["daily_average"].iloc[-1]
                    ] + list(trump_holdout_data["daily_average"])
                    harris_continuous_values = [
                        harris_training_data["daily_average"].iloc[-1]
                    ] + list(harris_holdout_data["daily_average"])

                    plt.plot(
                        trump_continuous_dates,
                        trump_continuous_values,
                        "r",
                        alpha=0.6,
                        linewidth=1,
                        label="Trump holdout data",
                    )
                    plt.plot(
                        harris_continuous_dates,
                        harris_continuous_values,
                        "b",
                        alpha=0.6,
                        linewidth=1,
                        label="Harris holdout data",
                    )
                else:
                    # No training data, just plot holdout
                    plt.plot(
                        holdout_dates,
                        trump_holdout_data["daily_average"],
                        "r",
                        alpha=0.6,
                        linewidth=1,
                        label="Trump holdout data",
                    )
                    plt.plot(
                        holdout_dates,
                        harris_holdout_data["daily_average"],
                        "b",
                        alpha=0.6,
                        linewidth=1,
                        label="Harris holdout data",
                    )

                # Model predictions during holdout (dashed lines)
                n_training = len(training_dates)
                if len(fitted_values["trump"]) > n_training:
                    trump_holdout_preds = fitted_values["trump"][n_training:]
                    harris_holdout_preds = fitted_values["harris"][n_training:]

                    n_holdout = min(len(holdout_dates), len(trump_holdout_preds))
                    if n_holdout > 0:
                        plt.plot(
                            holdout_dates[:n_holdout],
                            trump_holdout_preds[:n_holdout],
                            "r--",
                            linewidth=2,
                            label="Trump holdout predictions",
                        )
                        plt.plot(
                            holdout_dates[:n_holdout],
                            harris_holdout_preds[:n_holdout],
                            "b--",
                            linewidth=2,
                            label="Harris holdout predictions",
                        )

                # Baseline predictions during holdout (dotted lines)
                if holdout_baselines is not None:
                    trump_holdout_baselines = holdout_baselines.get("trump", [])
                    harris_holdout_baselines = holdout_baselines.get("harris", [])

                    n_holdout_baseline = min(
                        len(holdout_dates), len(trump_holdout_baselines)
                    )
                    if n_holdout_baseline > 0:
                        plt.plot(
                            holdout_dates[:n_holdout_baseline],
                            trump_holdout_baselines[:n_holdout_baseline],
                            "r:",
                            linewidth=1.5,
                            alpha=0.8,
                            label="Trump holdout baseline",
                        )
                        plt.plot(
                            holdout_dates[:n_holdout_baseline],
                            harris_holdout_baselines[:n_holdout_baseline],
                            "b:",
                            linewidth=1.5,
                            alpha=0.8,
                            label="Harris holdout baseline",
                        )
        else:
            # Fallback: show all historical data as training data
            plt.plot(
                historical_dates,
                trump_data["daily_average"],
                "r",
                linewidth=2,
                label="Trump daily polling average",
            )
            plt.plot(
                historical_dates,
                harris_data["daily_average"],
                "b",
                linewidth=2,
                label="Harris daily polling average",
            )

        # Plot future forecasts (connected to holdout predictions for continuity)
        if len(test_dates) > 0:
            # Create continuous prediction lines by combining holdout and future forecasts
            if training_end_date and len(holdout_dates) > 0:
                # Get the last holdout prediction values to ensure continuity
                n_training = len(training_dates)
                if len(fitted_values["trump"]) > n_training:
                    trump_holdout_preds = fitted_values["trump"][n_training:]
                    harris_holdout_preds = fitted_values["harris"][n_training:]

                    # Create continuous dates and predictions (last holdout point + future forecasts)
                    if len(trump_holdout_preds) > 0 and len(forecasts["trump"]) > 0:
                        continuous_trump_dates = [holdout_dates.iloc[-1]] + list(
                            test_dates
                        )
                        continuous_harris_dates = [holdout_dates.iloc[-1]] + list(
                            test_dates
                        )
                        continuous_trump_preds = [trump_holdout_preds[-1]] + list(
                            forecasts["trump"]
                        )
                        continuous_harris_preds = [harris_holdout_preds[-1]] + list(
                            forecasts["harris"]
                        )

                        plt.plot(
                            continuous_trump_dates,
                            continuous_trump_preds,
                            "r--",
                            linewidth=2,
                            alpha=0.8,
                            label="Trump future forecast",
                        )
                        plt.plot(
                            continuous_harris_dates,
                            continuous_harris_preds,
                            "b--",
                            linewidth=2,
                            alpha=0.8,
                            label="Harris future forecast",
                        )
                    else:
                        # Fallback: plot future forecasts without connection
                        plt.plot(
                            test_dates,
                            forecasts["trump"],
                            "r--",
                            linewidth=2,
                            alpha=0.8,
                            label="Trump future forecast",
                        )
                        plt.plot(
                            test_dates,
                            forecasts["harris"],
                            "b--",
                            linewidth=2,
                            alpha=0.8,
                            label="Harris future forecast",
                        )
                else:
                    # Fallback: plot future forecasts without connection
                    plt.plot(
                        test_dates,
                        forecasts["trump"],
                        "r--",
                        linewidth=2,
                        alpha=0.8,
                        label="Trump future forecast",
                    )
                    plt.plot(
                        test_dates,
                        forecasts["harris"],
                        "b--",
                        linewidth=2,
                        alpha=0.8,
                        label="Harris future forecast",
                    )
            else:
                # No holdout period: plot future forecasts normally
                plt.plot(
                    test_dates,
                    forecasts["trump"],
                    "r--",
                    linewidth=2,
                    alpha=0.8,
                    label="Trump future forecast",
                )
                plt.plot(
                    test_dates,
                    forecasts["harris"],
                    "b--",
                    linewidth=2,
                    alpha=0.8,
                    label="Harris future forecast",
                )

        # Plot baseline forecasts (connected to holdout baselines for continuity)
        if len(test_dates) > 0:
            if (
                training_end_date
                and len(holdout_dates) > 0
                and holdout_baselines is not None
            ):
                trump_holdout_baselines = holdout_baselines.get("trump", [])
                harris_holdout_baselines = holdout_baselines.get("harris", [])

                # Create continuous baseline lines
                if len(trump_holdout_baselines) > 0 and len(baselines["trump"]) > 0:
                    continuous_trump_baseline_dates = [holdout_dates.iloc[-1]] + list(
                        test_dates
                    )
                    continuous_harris_baseline_dates = [holdout_dates.iloc[-1]] + list(
                        test_dates
                    )
                    continuous_trump_baselines = [trump_holdout_baselines[-1]] + list(
                        baselines["trump"]
                    )
                    continuous_harris_baselines = [harris_holdout_baselines[-1]] + list(
                        baselines["harris"]
                    )

                    plt.plot(
                        continuous_trump_baseline_dates,
                        continuous_trump_baselines,
                        "r:",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Trump baseline forecast",
                    )
                    plt.plot(
                        continuous_harris_baseline_dates,
                        continuous_harris_baselines,
                        "b:",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Harris baseline forecast",
                    )
                else:
                    # Fallback: plot future baselines without connection
                    plt.plot(
                        test_dates,
                        baselines["trump"],
                        "r:",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Trump baseline forecast",
                    )
                    plt.plot(
                        test_dates,
                        baselines["harris"],
                        "b:",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Harris baseline forecast",
                    )
            else:
                # No holdout period: plot future baselines normally
                plt.plot(
                    test_dates,
                    baselines["trump"],
                    "r:",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Trump baseline forecast",
                )
                plt.plot(
                    test_dates,
                    baselines["harris"],
                    "b:",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Harris baseline forecast",
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
        trump_final = (
            round(forecasts["trump"][-1], 1) if len(forecasts["trump"]) > 0 else 47.0
        )

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
