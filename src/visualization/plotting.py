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

        # Get candidate configuration from config and add data
        candidates = {}
        data_map = {
            "trump": (trump_train_data, trump_holdout_data),
            "harris": (harris_train_data, harris_holdout_data),
        }

        for key, config in self.config.candidate_config.items():
            if key in data_map:
                candidates[key] = {
                    "name": config["display_name"],
                    "color": config["color"],
                    "train_data": data_map[key][0],
                    "holdout_data": data_map[key][1],
                }

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot polling data for both candidates
        for key, candidate in candidates.items():
            # Combine training + holdout data for continuous line
            all_data = pd.concat(
                [candidate["train_data"], candidate["holdout_data"]], ignore_index=True
            ).sort_values("end_date")

            # Plot continuous observed polling data (thick solid lines)
            plt.plot(
                all_data["end_date"],
                all_data["daily_average"],
                color=candidate["color"],
                linewidth=2,
                label=f"{candidate['name']} polling average",
            )

        # Plot all predictions for both candidates
        self._plot_all_predictions(
            candidates,
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
        plt.ylim(42, 52)

        # Title with forecast date
        title_date = forecast_date.strftime("%a %b %d %Y")
        plt.title(f"Predictions for Election Day, as of {title_date}", fontsize=16)
        plt.legend(loc="lower right")

        # Add parameter annotations
        param_text = (
            f"$\\mathbf{{Hyperparameters:}}$\n"
            f"  $\\alpha_{{\\mathrm{{Trump}}}} = {best_params['trump']['alpha']:.2f}$\n"
            f"  $\\beta_{{\\mathrm{{Trump}}}} = {best_params['trump']['beta']:.2f}$\n"
            f"  $\\alpha_{{\\mathrm{{Harris}}}} = {best_params['harris']['alpha']:.2f}$\n"
            f"  $\\beta_{{\\mathrm{{Harris}}}} = {best_params['harris']['beta']:.2f}$"
        )
        plt.text(
            datetime(2024, 7, 25),  # type: ignore[arg-type]
            42.3,
            param_text,
            fontsize=11,
        )

        # Add MASE performance annotations
        mase_text = (
            f"$\\mathbf{{MASE\\ Scores:}}$\n"
            f"  Model - Trump: {best_params['trump']['mase']:.3f}\n"
            f"  Model - Harris: {best_params['harris']['mase']:.3f}\n"
            f"  Baseline - Trump: {best_params['trump']['baseline_mase']:.3f}\n"
            f"  Baseline - Harris: {best_params['harris']['baseline_mase']:.3f}"
        )
        plt.text(
            datetime(2024, 8, 17),  # type: ignore[arg-type]
            42.3,
            mase_text,
            fontsize=11,
        )

        # Save plot
        save_path.unlink(missing_ok=True)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
        )
        logger.info(f"Saved main forecast plot to {save_path}")
        plt.close()

    def _plot_all_predictions(
        self,
        candidates,
        fitted_values,
        forecast_period_dates,
        forecasts,
        baselines,
        holdout_baselines,
    ):
        """Plot all model and baseline predictions for both candidates."""

        for key, candidate in candidates.items():
            color = candidate["color"]
            name = candidate["name"]
            train_len = len(candidate["train_data"])
            holdout_data = candidate["holdout_data"]

            # Create continuous prediction lines (holdout + future)
            holdout_fitted = fitted_values[key][
                train_len : train_len + len(holdout_data)
            ]

            # Combine dates and values for continuous plotting
            prediction_dates = list(holdout_data["end_date"]) + list(
                forecast_period_dates
            )
            prediction_values = list(holdout_fitted) + list(forecasts[key])
            baseline_values = list(holdout_baselines[key]) + list(baselines[key])

            # Plot continuous model predictions
            plt.plot(
                prediction_dates,
                prediction_values,
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"{name} model predictions",
            )

            # Plot continuous baseline predictions
            plt.plot(
                prediction_dates,
                baseline_values,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=f"{name} baseline predictions",
            )

    def plot_historical_forecasts(
        self,
        previous_forecasts: pd.DataFrame,
        forecast_date: Union[datetime, date],
        save_path: Path,
    ):
        """Create historical forecasts plot showing how predictions changed over time."""
        logger.info("Creating historical forecasts plot...")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot for each candidate using configuration
        for key, config in self.config.candidate_config.items():
            full_name = config["full_name"]
            display_name = config["display_name"]
            color = config["color"]

            # Filter data for this candidate
            candidate_data = previous_forecasts[
                (previous_forecasts["candidate"] == full_name)
                & (previous_forecasts["model"].notna())
            ].copy()

            # Plot model predictions
            self._plot_continuous_segments(
                candidate_data,
                color=color,
                label_prefix=f"{display_name} (model prediction)",
                value_column="model",
                line_style="-",
                marker="o",
                markersize=4,
            )

            # Plot baseline predictions
            self._plot_continuous_segments(
                candidate_data,
                color=color,
                label_prefix=f"{display_name} (baseline)",
                value_column="baseline",
                line_style="--",
                marker="s",
                markersize=3,
            )

        # Formatting
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Percentage of popular vote", fontsize=12)
        plt.xlim(datetime(2024, 10, 23), datetime(2024, 11, 5))
        plt.ylim(42, 52)

        # Set x-axis ticks
        tick_dates = [
            datetime(2024, 10, 23),
            datetime(2024, 10, 25),
            datetime(2024, 10, 27),
            datetime(2024, 10, 29),
            datetime(2024, 10, 31),
            datetime(2024, 11, 2),
            datetime(2024, 11, 4),
            # datetime(2024, 11, 5),
        ]
        plt.xticks(tick_dates)  # type: ignore[arg-type]

        # Title
        title_date = forecast_date.strftime("%a %b %d %Y")
        plt.title(f"Predictions up to {title_date}", fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Save plot
        save_path.unlink(missing_ok=True)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
        )
        logger.info(f"Saved historical forecasts plot to {save_path}")
        plt.close()

    def _plot_continuous_segments(
        self,
        data,
        color,
        label_prefix,
        value_column,
        line_style="-",
        marker="o",
        markersize=4,
    ):
        """Plot data in continuous segments, breaking lines at gaps."""
        # Filter out NaN values for the specified column
        plot_data = data[data[value_column].notna()]

        # Sort data by date
        data_sorted = plot_data.sort_values("date").copy()
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
                    segment[value_column],
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
                    segment[value_column],
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    markersize=markersize,
                    linewidth=2,
                    label=segment_label,
                )
