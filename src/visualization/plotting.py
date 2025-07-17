# src/visualization/plotting.py
"""Plotting and visualization functions for election forecasting."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import logging
from datetime import datetime, date
from typing import Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ElectionPlotter:
    """Handle all plotting and visualization for election forecasting."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.plot_config = config.plot_config

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
        _training_end_date: Union[datetime, date],
        holdout_baselines: Dict[str, np.ndarray],
        save_path: Path,
    ):
        """Create main forecast plot showing polling data and predictions."""
        logger.info("Creating main forecast plot...")

        # Format date once at the top
        title_date = forecast_date.strftime("%a %b %d %Y")

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

        # Create figure with configured size
        plt.figure(figsize=self.plot_config.figure_size)

        # Plot polling data for both candidates
        for key, candidate in candidates.items():
            # Combine training + holdout data for continuous line
            all_data = pd.concat(
                [candidate["train_data"], candidate["holdout_data"]], ignore_index=True
            ).sort_values("end_date")

            # Plot continuous observed polling data using configured settings
            plt.plot(
                all_data["end_date"],
                all_data["daily_average"],
                color=candidate["color"],
                linewidth=self.plot_config.polling_line_width,
                linestyle=self.plot_config.polling_line_style,
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

        # Formatting using configured values
        plt.xlabel("Date", fontsize=self.plot_config.axis_label_font_size)
        plt.ylabel("Percentage", fontsize=self.plot_config.axis_label_font_size)
        plt.xlim(
            self.plot_config.date_range_start_parsed,
            self.plot_config.date_range_end_parsed,
        )
        plt.ylim(self.plot_config.y_axis_min, self.plot_config.y_axis_max)
        plt.title(
            f"Predictions for Election Day, as of {title_date}",
            fontsize=self.plot_config.title_font_size,
        )
        plt.legend(loc=self.plot_config.legend_location)

        # Add parameter annotations using configured positions
        self._add_parameter_annotations(best_params)
        self._add_mase_annotations(best_params)

        # Format x-axis ticks for forecast plot
        self._format_date_ticks(
            major_interval=self.plot_config.forecast_major_tick_interval
        )

        # Save plot with configured DPI
        save_path.unlink(missing_ok=True)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=self.plot_config.dpi,
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

            # Plot continuous model predictions using configured settings
            plt.plot(
                prediction_dates,
                prediction_values,
                color=color,
                linestyle=self.plot_config.model_line_style,
                linewidth=self.plot_config.prediction_line_width,
                alpha=self.plot_config.prediction_alpha,
                label=f"{name} model predictions",
            )

            # Plot continuous baseline predictions using configured settings
            plt.plot(
                prediction_dates,
                baseline_values,
                color=color,
                linestyle=self.plot_config.baseline_line_style,
                linewidth=self.plot_config.baseline_line_width,
                alpha=self.plot_config.baseline_alpha,
                label=f"{name} baseline predictions",
            )

    def _add_parameter_annotations(self, best_params):
        """Add hyperparameter annotations using configured position."""
        param_text = (
            f"$\\mathbf{{Hyperparameters:}}$\n"
            f"  $\\alpha_{{\\mathrm{{Trump}}}} = {best_params['trump']['alpha']:.2f}$\n"
            f"  $\\beta_{{\\mathrm{{Trump}}}} = {best_params['trump']['beta']:.2f}$\n"
            f"  $\\alpha_{{\\mathrm{{Harris}}}} = {best_params['harris']['alpha']:.2f}$\n"
            f"  $\\beta_{{\\mathrm{{Harris}}}} = {best_params['harris']['beta']:.2f}$"
        )
        plt.text(
            self.plot_config.hyperparameter_annotation_pos_parsed[0],
            self.plot_config.hyperparameter_annotation_pos_parsed[1],
            param_text,
            fontsize=self.plot_config.annotation_font_size,
        )

    def _add_mase_annotations(self, best_params):
        """Add MASE performance annotations using configured position."""
        mase_text = (
            f"$\\mathbf{{MASE\\ Scores:}}$\n"
            f"  Model - Trump: {best_params['trump']['mase']:.3f}\n"
            f"  Model - Harris: {best_params['harris']['mase']:.3f}\n"
            f"  Baseline - Trump: {best_params['trump']['baseline_mase']:.3f}\n"
            f"  Baseline - Harris: {best_params['harris']['baseline_mase']:.3f}"
        )
        plt.text(
            self.plot_config.mase_annotation_pos_parsed[0],
            self.plot_config.mase_annotation_pos_parsed[1],
            mase_text,
            fontsize=self.plot_config.annotation_font_size,
        )

    def plot_historical_forecasts(
        self,
        previous_forecasts: pd.DataFrame,
        forecast_date: Union[datetime, date],
        save_path: Path,
    ):
        """Create historical forecasts plot showing how predictions changed over time."""
        logger.info("Creating historical forecasts plot...")

        # Format date once at the top
        title_date = forecast_date.strftime("%a %b %d %Y")

        # Create figure with configured size
        plt.figure(figsize=self.plot_config.figure_size)

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
                line_style=self.plot_config.historical_model_line_style,
                marker=self.plot_config.model_marker,
                markersize=self.plot_config.model_marker_size,
            )

            # Plot baseline predictions
            self._plot_continuous_segments(
                candidate_data,
                color=color,
                label_prefix=f"{display_name} (baseline)",
                value_column="baseline",
                line_style=self.plot_config.historical_baseline_line_style,
                marker=self.plot_config.baseline_marker,
                markersize=self.plot_config.baseline_marker_size,
            )

        # Formatting using configured values
        plt.xlabel("Date", fontsize=self.plot_config.axis_label_font_size)
        plt.ylabel(
            "Percentage of popular vote", fontsize=self.plot_config.axis_label_font_size
        )
        plt.xlim(
            self.plot_config.historical_date_range_start_parsed,
            self.plot_config.historical_date_range_end_parsed,
        )
        plt.ylim(self.plot_config.y_axis_min, self.plot_config.y_axis_max)
        plt.title(
            f"Predictions up to {title_date}", fontsize=self.plot_config.title_font_size
        )
        plt.legend(loc=self.plot_config.legend_location)
        plt.grid(True, alpha=self.plot_config.grid_alpha)

        # Format x-axis ticks for historical plot
        self._format_date_ticks(
            major_interval=self.plot_config.historical_major_tick_interval
        )

        # Save plot with configured DPI
        save_path.unlink(missing_ok=True)
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=self.plot_config.dpi,
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
                    linewidth=self.plot_config.prediction_line_width,
                    label=segment_label,
                )

    def _format_date_ticks(self, major_interval=None):
        """Format x-axis date ticks using configured format and specified intervals."""
        ax = plt.gca()

        # Set tick intervals (use parameters if provided, otherwise use defaults)
        major_int = major_interval if major_interval is not None else 7

        # Set tick locators
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=major_int))

        # Create custom formatter based on configured style
        def custom_date_formatter(x, _pos):
            """Custom formatter for different date styles without leading zeros."""
            date_obj = mdates.num2date(x)

            if self.plot_config.date_format_style == "numeric":
                # Format: 10-7
                return f"{date_obj.month}-{date_obj.day}"
            elif self.plot_config.date_format_style == "short_month":
                # Format: Oct 7
                return f"{date_obj.strftime('%b')} {date_obj.day}"
            elif self.plot_config.date_format_style == "full_month":
                # Format: October 7
                return f"{date_obj.strftime('%B')} {date_obj.day}"
            else:
                # Default to numeric
                return f"{date_obj.month}-{date_obj.day}"

        # Set major tick formatter
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_date_formatter))

        # Rotate labels if configured
        if self.plot_config.rotate_tick_labels:
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                rotation=self.plot_config.tick_label_rotation,
                ha=self.plot_config.tick_label_ha,
            )
