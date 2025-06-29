# src/config.py
"""Configuration settings for the election forecasting pipeline."""

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""

    # Cross-validation settings
    n_splits: int = 5
    test_size: int = 7

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
    polling_cache_path: str = "data/polling_averages_cache.csv"

    # Output directories and files
    forecast_images_dir: str = "outputs/forecast_images"
    historical_plots_dir: str = "outputs/previous_forecasts"
    comprehensive_dataset_path: str = "data/election_forecast_2024_comprehensive.csv"

    # Data filtering criteria
    candidates: List[str] = field(
        default_factory=lambda: ["Donald Trump", "Kamala Harris"]
    )
    population_filter: str = "lv"  # likely voters
    pollscore_threshold: float = 0.0  # negative pollscore only

    # Core date ranges
    earliest_available_data: str = "2021-04-07"
    biden_dropout_date: str = "2024-07-21"
    forecast_start_date: str = "2024-10-23"
    election_day: str = "2024-11-05"

    # CLI validation ranges - will be set in __post_init__
    min_valid_date: str = ""  # Will be set to earliest_available_data
    max_valid_date: str = ""  # Will be set to election_day

    # Single comprehensive swing states mapping
    swing_states_info: Dict[str, Dict] = field(
        default_factory=lambda: {
            "Arizona": {"code": "AZ", "electoral_votes": 11},
            "Georgia": {"code": "GA", "electoral_votes": 16},
            "Michigan": {"code": "MI", "electoral_votes": 15},
            "Nevada": {"code": "NV", "electoral_votes": 6},
            "North Carolina": {"code": "NC", "electoral_votes": 16},
            "Pennsylvania": {"code": "PA", "electoral_votes": 19},
            "Wisconsin": {"code": "WI", "electoral_votes": 10},
        }
    )

    def __post_init__(self):
        """Set CLI validation dates, derive swing state formats, and pre-compute all parsed dates."""
        # Set CLI validation dates based on other date fields
        if not self.min_valid_date:
            self.min_valid_date = self.earliest_available_data
        if not self.max_valid_date:
            self.max_valid_date = self.election_day

        # Derive swing states list and electoral mapping from single source
        self.swing_states = list(self.swing_states_info.keys())
        self.swing_states_electoral_votes = {
            info["code"]: info["electoral_votes"]
            for info in self.swing_states_info.values()
        }

        # Pre-compute all parsed dates (replaces 6 @property methods)
        self.earliest_available_data_parsed = datetime.strptime(
            self.earliest_available_data, "%Y-%m-%d"
        ).date()
        self.biden_dropout_date_parsed = datetime.strptime(
            self.biden_dropout_date, "%Y-%m-%d"
        ).date()
        self.forecast_start_date_parsed = datetime.strptime(
            self.forecast_start_date, "%Y-%m-%d"
        ).date()
        self.election_day_parsed = datetime.strptime(
            self.election_day, "%Y-%m-%d"
        ).date()
        self.min_valid_date_parsed = datetime.strptime(
            self.min_valid_date, "%Y-%m-%d"
        ).date()
        self.max_valid_date_parsed = datetime.strptime(
            self.max_valid_date, "%Y-%m-%d"
        ).date()
