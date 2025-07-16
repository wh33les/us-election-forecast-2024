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
        trump_train_data: pd.DataFrame,
        trump_holdout_data: pd.DataFrame,
        harris_train_data: pd.DataFrame,
        harris_holdout_data: pd.DataFrame,
        forecasts: Dict[str, np.ndarray],
        baselines: Dict[str, np.ndarray],
        fitted_values: Dict[str, np.ndarray],
        best_params: Dict[str, Dict[str, float]],
        forecast_period_dates: pd.Series,
        forecast_date: Union[datetime, date],
        training_end_date: Union[datetime, date],
        holdout_baselines: Dict[str, np.ndarray],
        save_path: Path,
    ):
        """Create main forecast plot showing polling data and predictions."""
        logger.info("Creating main forecast plot...")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot ALL polling data as continuous lines (combine training + holdout)
        trump_all_data = pd.concat(
            [trump_train_data, trump_holdout_data], ignore_index=True
        ).sort_values("end_date")
        harris_all_data = pd.concat(
            [harris_train_data, harris_holdout_data], ignore_index=True
        ).sort_values("end_date")

        # Plot continuous observed polling data (thick solid lines)
        plt.plot(
            trump_all_data["end_date"],
            trump_all_data["daily_average"],
            "r",
            linewidth=2,
            label="Trump polling average",
        )
        plt.plot(
            harris_all_data["end_date"],
            harris_all_data["daily_average"],
            "b",
            linewidth=2,
            label="Harris polling average",
        )

        # Plot all predictions (holdout + future) for both model and baseline
        self._plot_all_predictions(
            trump_train_data,
            trump_holdout_data,
            harris_train_data,
            harris_holdout_data,
            fitted_values,
            forecast_period_dates,
            forecasts,
            baselines,
            holdout_baselines,
        )

        # Formatting
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Percentage", fontsize=12)
        plt.xlim(datetime(2024, 7, 21), datetime(2024, 11, 5))
        plt.ylim(40, 52)

        # Title with forecast date
        title_date = forecast_date.strftime("%a %b %d %Y")
        plt.title(f"Predictions for Election Day, as of {title_date}", fontsize=16)
        plt.legend()

        # Add parameter annotations
        election_day_dt = datetime(2024, 11, 5)
        trump_final = round(forecasts["trump"][-1], 1)

        param_text = (
            f"$\\alpha_{{Trump}} = {best_params['trump']['alpha']:.2f}$\n"
            f"$\\beta_{{Trump}} = {best_params['trump']['beta']:.2f}$\n"
            f"$\\alpha_{{Harris}} = {best_params['harris']['alpha']:.2f}$\n"
            f"$\\beta_{{Harris}} = {best_params['harris']['beta']:.2f}$"
        )

        # Use type: ignore for matplotlib annotation since it handles datetime objects in practice
        plt.annotate(
            param_text,
            xy=(election_day_dt, trump_final),  # type: ignore[arg-type]
            xytext=(datetime(2024, 7, 23), 40.5),  # type: ignore[arg-type]
        )

        # Save plot
        save_path.unlink(missing_ok=True)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
            edgecolor="none",
        )
        logger.info(f"Saved main forecast plot to {save_path}")
        plt.close()

    def _plot_all_predictions(
        self,
        trump_train_data,
        trump_holdout_data,
        harris_train_data,
        harris_holdout_data,
        fitted_values,
        forecast_period_dates,
        forecasts,
        baselines,
        holdout_baselines,
    ):
        """Plot all model and baseline predictions for both holdout and future periods."""

        trump_train_len = len(trump_train_data)
        harris_train_len = len(harris_train_data)

        # 1. Holdout model predictions
        trump_holdout_fitted = fitted_values["trump"][
            trump_train_len : trump_train_len + len(trump_holdout_data)
        ]
        harris_holdout_fitted = fitted_values["harris"][
            harris_train_len : harris_train_len + len(harris_holdout_data)
        ]

        plt.plot(
            trump_holdout_data["end_date"],
            trump_holdout_fitted,
            "r--",
            linewidth=2,
            alpha=0.7,
            label="Trump model predictions",
        )
        plt.plot(
            harris_holdout_data["end_date"],
            harris_holdout_fitted,
            "b--",
            linewidth=2,
            alpha=0.7,
            label="Harris model predictions",
        )

        # 2. Future model forecasts (with connection from last holdout prediction)
        continuous_trump_dates = [trump_holdout_data["end_date"].iloc[-1]] + list(
            forecast_period_dates
        )
        continuous_harris_dates = [harris_holdout_data["end_date"].iloc[-1]] + list(
            forecast_period_dates
        )
        continuous_trump_values = [fitted_values["trump"][-1]] + list(
            forecasts["trump"]
        )
        continuous_harris_values = [fitted_values["harris"][-1]] + list(
            forecasts["harris"]
        )

        plt.plot(
            continuous_trump_dates,
            continuous_trump_values,
            "r--",
            linewidth=2,
            alpha=0.8,
            label="Trump forecast",
        )
        plt.plot(
            continuous_harris_dates,
            continuous_harris_values,
            "b--",
            linewidth=2,
            alpha=0.8,
            label="Harris forecast",
        )

        # 3. Holdout baseline predictions
        trump_baseline = holdout_baselines["trump"]
        harris_baseline = holdout_baselines["harris"]

        plt.plot(
            trump_holdout_data["end_date"],
            trump_baseline,
            "r:",
            linewidth=1.5,
            alpha=0.8,
            label="Trump baseline predictions",
        )
        plt.plot(
            harris_holdout_data["end_date"],
            harris_baseline,
            "b:",
            linewidth=1.5,
            alpha=0.8,
            label="Harris baseline predictions",
        )

        # 4. Future baseline forecasts
        plt.plot(
            forecast_period_dates,
            baselines["trump"],
            "r:",
            linewidth=1.5,
            alpha=0.7,
            label="Trump baseline forecast",
        )
        plt.plot(
            forecast_period_dates,
            baselines["harris"],
            "b:",
            linewidth=1.5,
            alpha=0.7,
            label="Harris baseline forecast",
        )

    def plot_historical_forecasts(
        self,
        previous_forecasts: pd.DataFrame,
        forecast_date: Union[datetime, date],
        save_path: Path,
    ):
        """Create historical forecasts plot showing how predictions changed over time."""
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

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot model predictions with gap-aware line breaking
        self._plot_continuous_segments(
            trump_data,
            color="red",
            label_prefix="Trump (model prediction)",
            line_style="-",
            marker="o",
            markersize=4,
        )
        self._plot_continuous_segments(
            harris_data,
            color="blue",
            label_prefix="Harris (model prediction)",
            line_style="-",
            marker="o",
            markersize=4,
        )

        # Plot baseline predictions with gap-aware line breaking
        self._plot_baseline_segments(
            trump_data,
            color="red",
            label_prefix="Trump (baseline)",
            line_style="--",
            marker="s",
            markersize=3,
        )
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
        plt.xlim(datetime(2024, 10, 23), datetime(2024, 11, 5))
        plt.ylim(45, 52)

        # Set x-axis ticks
        tick_dates = [
            datetime(2024, 10, 23),
            datetime(2024, 10, 25),
            datetime(2024, 10, 27),
            datetime(2024, 10, 29),
            datetime(2024, 10, 31),
            datetime(2024, 11, 2),
            datetime(2024, 11, 4),
            datetime(2024, 11, 5),
        ]
        plt.xticks(tick_dates)  # type: ignore[arg-type]

        # Title
        title_date = forecast_date.strftime("%a %b %d %Y")
        plt.title(f"Predictions up to {title_date}", fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        save_path.unlink(missing_ok=True)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
            facecolor="white",
            edgecolor="none",
        )
        logger.info(f"Saved historical forecasts plot to {save_path}")
        plt.close()

    def _plot_continuous_segments(
        self, data, color, label_prefix, line_style="-", marker="o", markersize=4
    ):
        """Plot data in continuous segments, breaking lines at gaps."""
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
        """Plot baseline data in continuous segments, breaking lines at gaps."""
        baseline_data = data[data["baseline"].notna()]

        # Sort data by date
        data_sorted = baseline_data.sort_values("date").copy()
        data_sorted["plot_date"] = pd.to_datetime(data_sorted["date"])

        # Find gaps and split into segments (same logic as _plot_continuous_segments)
        gaps = []
        for i in range(1, len(data_sorted)):
            current_date = data_sorted.iloc[i]["plot_date"]
            previous_date = data_sorted.iloc[i - 1]["plot_date"]
            days_diff = (current_date - previous_date).days
            if days_diff > 1:
                gaps.append(i)

        segments = []
        start_idx = 0
        for gap_idx in gaps:
            segments.append(data_sorted.iloc[start_idx:gap_idx])
            start_idx = gap_idx
        segments.append(data_sorted.iloc[start_idx:])

        # Plot each segment separately
        for i, segment in enumerate(segments):
            segment_label = label_prefix if i == 0 else None

            if len(segment) == 1:
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
