# src/models/holt_forecaster.py
"""Holt exponential smoothing forecaster for election predictions."""

import pandas as pd
import numpy as np
import logging
from typing import Dict
from statsmodels.tsa.holtwinters import Holt
from sklearn.model_selection import TimeSeriesSplit

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
        """
        Calculate Mean Absolute Scaled Error.
        From your original forecast.py MASE function.

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
        """
        Perform grid search for optimal Holt smoothing parameters.
        From your original forecast.py grid search logic.
        """
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
        """
        Fit final Holt models using best parameters.
        From your original forecast.py final model fitting.
        """
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
        """
        Generate forecasts for specified horizon.
        """
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
        """
        Generate random walk with drift baseline forecasts.
        From your original forecast.py baseline logic.
        """
        logger.info(f"Generating baseline forecasts for {horizon} periods...")

        # Trump baseline
        trump_values = trump_data.daily_average.values
        trump_drift = (trump_values[-1] - trump_values[0]) / len(trump_values)
        trump_baseline = trump_values[-1] + trump_drift * np.arange(1, horizon + 1)

        # Harris baseline
        harris_values = harris_data.daily_average.values
        harris_drift = (harris_values[-1] - harris_values[0]) / len(harris_values)
        harris_baseline = harris_values[-1] + harris_drift * np.arange(1, horizon + 1)

        baselines = {"trump": trump_baseline, "harris": harris_baseline}

        logger.info("Baseline forecasts generated successfully")
        return baselines

    def get_fitted_values(self) -> Dict[str, np.ndarray]:
        """
        Get fitted values from trained models.
        """
        if not self.fitted_models:
            raise ValueError("Must fit models before getting fitted values")

        return {
            "trump": self.fitted_models["trump"].fittedvalues,
            "harris": self.fitted_models["harris"].fittedvalues,
        }
