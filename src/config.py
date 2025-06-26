# src/config.py
"""Configuration settings for the election forecasting pipeline."""

from datetime import datetime
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""

    # Cross-validation settings
    n_splits: int = 5
    test_size: int = 7
    no_test_days: int = 0

    # Grid search parameters for Holt smoothing
    grid_min: float = 0.0
    grid_max: float = 0.5
    grid_step: float = 0.1

    # Electoral college settings
    swing_state_electoral_votes: int = 93
    trump_safe_electoral_votes: int = 219
    harris_safe_electoral_votes: int = 226

    @property
    def grid_numbers(self):
        """Generate grid search parameter range."""
        return np.arange(self.grid_min, self.grid_max, self.grid_step)


@dataclass
class DataConfig:
    """Configuration for data sources and filtering."""

    # Data file paths
    raw_data_path: str = "data/president_polls.csv"
    cleaned_data_path: str = "data/df_cleaned.csv"  # Will be saved in data/
    previous_forecasts_path: str = "data/previous.csv"  # Will be saved in data/

    # Output directories
    forecast_images_dir: str = "outputs/forecast_images"
    # Note: previous_images_dir not used - final historical plot goes directly in outputs/

    # Data filtering criteria
    candidates: List[str] = None
    population_filter: str = "lv"  # likely voters
    swing_states: List[str] = None
    pollscore_threshold: float = 0.0  # negative pollscore only

    # Date ranges
    biden_dropout_date: str = "2024-07-21"
    election_day: str = "2024-11-05"
    forecast_start_date: str = "2024-10-23"

    # Swing states electoral vote mapping
    swing_states_electoral_votes: dict = None

    def __post_init__(self):
        """Initialize default values that depend on other attributes."""
        if self.candidates is None:
            self.candidates = ["Donald Trump", "Kamala Harris"]

        if self.swing_states is None:
            self.swing_states = [
                "Arizona",
                "Georgia",
                "Michigan",
                "Nevada",
                "North Carolina",
                "Pennsylvania",
                "Wisconsin",
            ]

        if self.swing_states_electoral_votes is None:
            self.swing_states_electoral_votes = {
                "AZ": 11,
                "GA": 16,
                "NC": 16,
                "NV": 6,
                "PA": 19,
                "WI": 10,
                "MI": 15,
            }

    @property
    def biden_dropout_date_parsed(self):
        """Return Biden dropout date as datetime.date object."""
        return datetime.strptime(self.biden_dropout_date, "%Y-%m-%d").date()

    @property
    def election_day_parsed(self):
        """Return election day as datetime.date object."""
        return datetime.strptime(self.election_day, "%Y-%m-%d").date()

    @property
    def forecast_start_date_parsed(self):
        """Return forecast start date as datetime.date object."""
        return datetime.strptime(self.forecast_start_date, "%Y-%m-%d").date()
