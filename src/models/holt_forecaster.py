# src/models/holt_forecaster.py
"""Holt exponential smoothing forecaster for election predictions."""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HoltElectionForecaster:
    """Holt exponential smoothing forecaster for election polling data."""

    def __init__(self, config):
        """Initialize with model configuration."""
        self.config = config
        self.fitted_models = {}
        self.best_params = {}

    def mase(
        self, y_train: np.ndarray, y_test: np.ndarray, y_preds: np.ndarray
    ) -> float:
        """Calculate Mean Absolute Scaled Error.

        If MASE > 1: forecast performs worse than naive forecast
        If MASE < 1: forecast performs better than naive forecast
        """
        n = len(y_train)
        m = len(y_test)

        # Calculate denominator (naive forecast error on training set)
        denom = 0
        for i in range(n - m):
            naive_forecast = y_train[i] * np.ones(m)
            actual_segment = y_train[i + 1 : i + m + 1]
            denom += np.abs(actual_segment - naive_forecast).mean()
        denom = denom / (n - m)

        # Calculate numerator (forecast error on test set)
        num = np.abs(y_test - y_preds).mean()

        return num / denom

    def grid_search_hyperparameters(
        self, trump_data: pd.DataFrame, harris_data: pd.DataFrame, x_train: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Perform grid search for optimal Holt smoothing parameters."""
        logger.info("Starting hyperparameter grid search...")

        grid_numbers = self.config.grid_numbers
        kfold = TimeSeriesSplit(
            n_splits=self.config.n_splits, test_size=self.config.test_size
        )

        # Initialize error arrays
        exp_mase_trump = np.zeros(
            (self.config.n_splits, len(grid_numbers), len(grid_numbers))
        )
        exp_mase_harris = np.zeros(
            (self.config.n_splits, len(grid_numbers), len(grid_numbers))
        )

        split_idx = 0
        for train_index, test_index in kfold.split(x_train):
            trump_train = trump_data.iloc[train_index]
            trump_test = trump_data.iloc[test_index]
            harris_train = harris_data.iloc[train_index]
            harris_test = harris_data.iloc[test_index]

            alpha_idx = 0
            for alpha in grid_numbers:
                beta_idx = 0
                for beta in grid_numbers:
                    try:
                        # Fit Trump model
                        trump_model = Holt(trump_data.daily_average.values).fit(
                            smoothing_level=alpha, smoothing_trend=beta, optimized=False
                        )
                        trump_forecast = trump_model.forecast(len(trump_test))
                        exp_mase_trump[split_idx, alpha_idx, beta_idx] = self.mase(
                            trump_train.daily_average.values,
                            trump_test.daily_average.values,
                            trump_forecast,
                        )

                        # Fit Harris model
                        harris_model = Holt(harris_data.daily_average.values).fit(
                            smoothing_level=alpha, smoothing_trend=beta, optimized=False
                        )
                        harris_forecast = harris_model.forecast(len(harris_test))
                        exp_mase_harris[split_idx, alpha_idx, beta_idx] = self.mase(
                            harris_train.daily_average.values,
                            harris_test.daily_average.values,
                            harris_forecast,
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error in grid search at alpha={alpha}, beta={beta}: {e}"
                        )
                        exp_mase_trump[split_idx, alpha_idx, beta_idx] = np.inf
                        exp_mase_harris[split_idx, alpha_idx, beta_idx] = np.inf

                    beta_idx += 1
                alpha_idx += 1
            split_idx += 1

        # Find best parameters
        trump_best_idx = np.unravel_index(
            np.argmin(np.mean(exp_mase_trump, axis=0), axis=None),
            np.mean(exp_mase_trump, axis=0).shape,
        )
        harris_best_idx = np.unravel_index(
            np.argmin(np.mean(exp_mase_harris, axis=0), axis=None),
            np.mean(exp_mase_harris, axis=0).shape,
        )

        best_params = {
            "trump": {
                "alpha": grid_numbers[trump_best_idx[0]],
                "beta": grid_numbers[trump_best_idx[1]],
                "mase": np.mean(exp_mase_trump, axis=0)[trump_best_idx],
            },
            "harris": {
                "alpha": grid_numbers[harris_best_idx[0]],
                "beta": grid_numbers[harris_best_idx[1]],
                "mase": np.mean(exp_mase_harris, axis=0)[harris_best_idx],
            },
        }

        self.best_params = best_params
        self._log_best_parameters()
        return best_params

    def _log_best_parameters(self):
        """Log the best parameters found during grid search."""
        trump_params = self.best_params["trump"]
        harris_params = self.best_params["harris"]

        logger.info(
            f"Trump best params: α={trump_params['alpha']:.3f}, "
            f"β={trump_params['beta']:.3f}, MASE={trump_params['mase']:.3f}"
        )

        if trump_params["mase"] < 1:
            logger.info("Trump model beats naive baseline")
        else:
            logger.info("Trump model does not beat naive baseline")

        logger.info(
            f"Harris best params: α={harris_params['alpha']:.3f}, "
            f"β={harris_params['beta']:.3f}, MASE={harris_params['mase']:.3f}"
        )

        if harris_params["mase"] < 1:
            logger.info("Harris model beats naive baseline")
        else:
            logger.info("Harris model does not beat naive baseline")

    def fit_final_models(
        self, trump_data: pd.DataFrame, harris_data: pd.DataFrame
    ) -> Dict[str, Holt]:
        """Fit final Holt models using best parameters."""
        logger.info("Fitting final Holt models...")

        if not self.best_params:
            raise ValueError("Must run grid search before fitting final models")

        # Fit Trump model
        trump_model = Holt(trump_data.daily_average.values).fit(
            smoothing_level=self.best_params["trump"]["alpha"],
            smoothing_trend=self.best_params["trump"]["beta"],
            optimized=False,
        )

        # Fit Harris model
        harris_model = Holt(harris_data.daily_average.values).fit(
            smoothing_level=self.best_params["harris"]["alpha"],
            smoothing_trend=self.best_params["harris"]["beta"],
            optimized=False,
        )

        self.fitted_models = {"trump": trump_model, "harris": harris_model}

        logger.info("Final models fitted successfully")
        return self.fitted_models

    def forecast(self, horizon: int) -> Dict[str, np.ndarray]:
        """Generate forecasts for specified horizon."""
        if not self.fitted_models:
            raise ValueError("Must fit models before forecasting")

        logger.info(f"Generating forecasts for {horizon} periods ahead...")

        forecasts = {
            "trump": self.fitted_models["trump"].forecast(horizon),
            "harris": self.fitted_models["harris"].forecast(horizon),
        }

        logger.info("Forecasts generated successfully")
        return forecasts

    def generate_baseline_forecasts(
        self, trump_data: pd.DataFrame, harris_data: pd.DataFrame, horizon: int
    ) -> Dict[str, np.ndarray]:
        """Generate linear drift baseline forecasts (straight lines).

        This calculates a simple linear trend from the training data and
        extrapolates it forward as a straight line.
        """
        logger.info(f"Generating baseline forecasts for {horizon} periods...")

        # Trump baseline - calculate linear drift and extrapolate
        trump_values = trump_data.daily_average.values
        trump_drift = (trump_values[-1] - trump_values[0]) / len(trump_values)
        trump_baseline = trump_values[-1] + trump_drift * np.arange(1, horizon + 1)

        # Harris baseline - calculate linear drift and extrapolate
        harris_values = harris_data.daily_average.values
        harris_drift = (harris_values[-1] - harris_values[0]) / len(harris_values)
        harris_baseline = harris_values[-1] + harris_drift * np.arange(1, horizon + 1)

        baselines = {"trump": trump_baseline, "harris": harris_baseline}

        # Log baseline info for debugging
        logger.debug(f"Trump drift: {trump_drift:.4f} per period")
        logger.debug(f"Harris drift: {harris_drift:.4f} per period")
        logger.debug(f"Trump baseline first 3 values: {trump_baseline[:3]}")
        logger.debug(f"Harris baseline first 3 values: {harris_baseline[:3]}")

        logger.info("Baseline forecasts generated successfully")
        return baselines

    def get_fitted_values(self) -> Dict[str, np.ndarray]:
        """Get fitted values from trained models."""
        if not self.fitted_models:
            raise ValueError("Must fit models before getting fitted values")

        return {
            "trump": self.fitted_models["trump"].fittedvalues,
            "harris": self.fitted_models["harris"].fittedvalues,
        }

    def diagnose_holt_linearity(self, model, forecast_periods=10):
        """Diagnose if Holt forecasts are properly linear."""

        # Extract final level and trend
        final_level = model.level
        final_trend = model.trend

        print(f"\n🔍 HOLT LINEARITY DIAGNOSTICS")
        print(f"Final Level (L_t): {final_level:.6f}")
        print(f"Final Trend (T_t): {final_trend:.6f}")

        # Generate manual forecast to verify linearity
        manual_forecast = []
        for h in range(1, forecast_periods + 1):
            expected_value = final_level + h * final_trend
            manual_forecast.append(expected_value)

        # Get actual forecast from model
        actual_forecast = model.forecast(forecast_periods)

        # Compare manual vs actual
        print(f"\n📊 FORECAST COMPARISON (first 5 periods):")
        print(f"Period | Manual     | Actual     | Difference")
        print(f"-------|------------|------------|------------")

        max_diff = 0
        for i in range(min(5, forecast_periods)):
            diff = abs(manual_forecast[i] - actual_forecast[i])
            max_diff = max(max_diff, diff)
            print(
                f"{i+1:6d} | {manual_forecast[i]:10.6f} | {actual_forecast[i]:10.6f} | {diff:10.8f}"
            )

        # Check linearity of actual forecast
        is_linear = True
        if len(actual_forecast) > 2:
            diffs = np.diff(actual_forecast)
            is_linear = np.allclose(diffs, diffs[0], rtol=1e-10)
            print(f"\n📏 LINEARITY CHECK:")
            print(f"First difference: {diffs[0]:.8f}")
            print(f"All differences equal: {is_linear}")
            print(f"Difference range: {np.min(diffs):.8f} to {np.max(diffs):.8f}")

            if not is_linear:
                print(f"⚠️  FORECAST IS NOT LINEAR!")
                print(f"Differences: {diffs[:5]}")  # Show first 5 differences

        return {
            "final_level": final_level,
            "final_trend": final_trend,
            "manual_forecast": manual_forecast,
            "actual_forecast": actual_forecast,
            "max_difference": max_diff,
            "is_linear": is_linear,
        }

    def debug_forecast_generation(self, candidate_data, hyperparams, forecast_periods):
        """Debug the entire forecast generation process."""

        print(f"\n🚀 DEBUGGING FORECAST GENERATION")
        print(
            f"Candidate: {candidate_data.name if hasattr(candidate_data, 'name') else 'Unknown'}"
        )
        print(f"Data points: {len(candidate_data)}")
        print(
            f"Hyperparams: α={hyperparams.get('alpha', 'N/A')}, β={hyperparams.get('beta', 'N/A')}"
        )
        print(f"Forecast periods: {forecast_periods}")

        # Fit model with debugging
        model = Holt(candidate_data).fit(
            smoothing_level=hyperparams["alpha"],
            smoothing_trend=hyperparams["beta"],
            optimized=False,
        )

        # Diagnose the fitted model
        diagnostics = self.diagnose_holt_linearity(model, forecast_periods)

        return model, diagnostics

    def plot_forecast_debug(self, data, model, forecast, forecast_date, candidate_name):
        """Create detailed debug plot showing forecast components."""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Full data with forecast
        ax1.plot(
            range(len(data)), data, "o-", label=f"{candidate_name} Data", alpha=0.7
        )

        # Generate forecast index
        forecast_start = len(data)
        forecast_x = range(forecast_start, forecast_start + len(forecast))

        ax1.plot(
            forecast_x, forecast, "r--", linewidth=2, label=f"{candidate_name} Forecast"
        )
        ax1.axvline(
            x=forecast_start - 0.5,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label="Forecast Start",
        )
        ax1.set_title(f"{candidate_name} Forecast Debug - {forecast_date}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Just the forecast to check linearity
        ax2.plot(range(len(forecast)), forecast, "ro-", linewidth=2, markersize=4)
        ax2.set_title(f"{candidate_name} Forecast Linearity Check")
        ax2.set_xlabel("Forecast Period")
        ax2.set_ylabel("Predicted Value")
        ax2.grid(True, alpha=0.3)

        # Add trend line to verify linearity
        if len(forecast) > 1:
            x = np.arange(len(forecast))
            coeffs = np.polyfit(x, forecast, 1)
            trend_line = np.polyval(coeffs, x)
            ax2.plot(
                x,
                trend_line,
                "b--",
                alpha=0.7,
                label=f"Linear Trend (slope={coeffs[0]:.4f})",
            )
            ax2.legend()

        plt.tight_layout()

        # Save debug plot
        debug_path = f'outputs/debug_plots/forecast_debug_{candidate_name}_{forecast_date.strftime("%m%d")}.png'
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        plt.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"💾 Saved debug plot: {debug_path}")

    def forecast_with_diagnostics(
        self, candidate_data_dict, forecast_periods, forecast_date
    ):
        """Generate forecasts with full diagnostics."""

        results = {}

        for candidate, data in candidate_data_dict.items():
            print(f"\n{'='*50}")
            print(f"FORECASTING: {candidate}")
            print(f"{'='*50}")

            # Get best hyperparameters
            best_params = self.best_params.get(candidate, {"alpha": 0.2, "beta": 0.1})

            # Debug forecast generation
            model, diagnostics = self.debug_forecast_generation(
                data, best_params, forecast_periods
            )

            # Generate forecast
            forecast = model.forecast(forecast_periods)

            # Create debug plot
            self.plot_forecast_debug(data, model, forecast, forecast_date, candidate)

            # Store results
            results[candidate] = {
                "model": model,
                "forecast": forecast,
                "diagnostics": diagnostics,
            }

            # Report issues
            if not diagnostics["is_linear"]:
                print(f"⚠️  WARNING: {candidate} forecast is not linear!")

            if diagnostics["max_difference"] > 1e-6:
                print(
                    f"⚠️  WARNING: {candidate} manual vs actual forecast differs by {diagnostics['max_difference']}"
                )

        return results
