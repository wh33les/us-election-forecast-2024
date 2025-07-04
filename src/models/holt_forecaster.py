# src/models/holt_forecaster.py
"""Holt exponential smoothing forecaster for election predictions."""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HoltElectionForecaster:
    """Holt exponential smoothing forecaster for election polling data."""

    def __init__(self, model_config, data_config=None):
        """Initialize with model configuration."""
        self.model_config = model_config
        self.data_config = data_config  # Add data_config
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
        """Perform grid search for optimal Holt smoothing parameters with baseline comparison."""
        logger.info("Starting hyperparameter grid search with baseline evaluation...")

        grid_numbers = self.model_config.grid_numbers
        kfold = TimeSeriesSplit(
            n_splits=self.model_config.n_splits, test_size=self.model_config.test_size
        )

        # Initialize error arrays for Holt models
        exp_mase_trump = np.zeros(
            (self.model_config.n_splits, len(grid_numbers), len(grid_numbers))
        )
        exp_mase_harris = np.zeros(
            (self.model_config.n_splits, len(grid_numbers), len(grid_numbers))
        )

        # Initialize arrays for baseline models
        baseline_mase_trump = np.zeros(self.model_config.n_splits)
        baseline_mase_harris = np.zeros(self.model_config.n_splits)

        split_idx = 0
        for train_index, test_index in kfold.split(np.array(x_train).reshape(-1, 1)):
            trump_train = trump_data.iloc[train_index]
            trump_test = trump_data.iloc[test_index]
            harris_train = harris_data.iloc[train_index]
            harris_test = harris_data.iloc[test_index]

            # Calculate baseline MASE for this fold
            trump_baseline_forecast = self._generate_baseline_forecast_for_cv(
                trump_train, len(trump_test)
            )
            harris_baseline_forecast = self._generate_baseline_forecast_for_cv(
                harris_train, len(harris_test)
            )

            baseline_mase_trump[split_idx] = self.mase(
                np.array(trump_train.daily_average.values),  # Convert to numpy
                np.array(trump_test.daily_average.values),  # Convert to numpy
                trump_baseline_forecast,
            )
            baseline_mase_harris[split_idx] = self.mase(
                np.array(harris_train.daily_average.values),  # Convert to numpy
                np.array(harris_test.daily_average.values),  # Convert to numpy
                harris_baseline_forecast,
            )

            # Holt model grid search
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
                            np.array(trump_train.daily_average.values),
                            np.array(trump_test.daily_average.values),
                            trump_forecast,
                        )

                        # Fit Harris model
                        harris_model = Holt(harris_data.daily_average.values).fit(
                            smoothing_level=alpha, smoothing_trend=beta, optimized=False
                        )
                        harris_forecast = harris_model.forecast(len(harris_test))
                        exp_mase_harris[split_idx, alpha_idx, beta_idx] = self.mase(
                            np.array(harris_train.daily_average.values),
                            np.array(harris_test.daily_average.values),
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

        # Include baseline MASE in results
        best_params = {
            "trump": {
                "alpha": grid_numbers[trump_best_idx[0]],
                "beta": grid_numbers[trump_best_idx[1]],
                "mase": np.mean(exp_mase_trump, axis=0)[trump_best_idx],
                "baseline_mase": np.mean(baseline_mase_trump),
            },
            "harris": {
                "alpha": grid_numbers[harris_best_idx[0]],
                "beta": grid_numbers[harris_best_idx[1]],
                "mase": np.mean(exp_mase_harris, axis=0)[harris_best_idx],
                "baseline_mase": np.mean(baseline_mase_harris),
            },
        }

        self.best_params = best_params
        self._log_best_parameters_enhanced()
        return best_params

    def _generate_baseline_forecast_for_cv(
        self, train_data: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        """Generate baseline forecast for cross-validation."""
        values = train_data.daily_average.values
        if len(values) < 2:
            return np.full(horizon, values[-1])

        # Linear drift calculation
        drift = (values[-1] - values[0]) / len(values)
        return values[-1] + drift * np.arange(1, horizon + 1)

    def _log_best_parameters_enhanced(self):
        """Log the best parameters with baseline comparison."""
        for candidate in ["trump", "harris"]:
            params = self.best_params[candidate]
            holt_mase = params["mase"]
            baseline_mase = params["baseline_mase"]
            improvement = baseline_mase - holt_mase

            logger.info(f"{candidate.title()} Model Results:")
            logger.info(
                f"  Best Holt: α={params['alpha']:.3f}, β={params['beta']:.3f}, MASE={holt_mase:.3f}"
            )
            logger.info(f"  Baseline: MASE={baseline_mase:.3f}")

            if improvement > 0:
                logger.info(
                    f"  ✅ Holt beats baseline by {improvement:.3f} ({improvement/baseline_mase*100:.1f}%)"
                )
            else:
                logger.info(f"  ❌ Baseline beats Holt by {abs(improvement):.3f}")

            if holt_mase < 1:
                logger.info(f"  ✅ Holt beats naive forecast")
            else:
                logger.info(f"  ❌ Holt does not beat naive forecast")

    def fit_final_models(
        self, trump_data: pd.DataFrame, harris_data: pd.DataFrame
    ) -> Dict[str, Any]:
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
